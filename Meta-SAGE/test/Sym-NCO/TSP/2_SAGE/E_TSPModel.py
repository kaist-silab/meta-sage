import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from TSPModel_ours import TSPModel, _get_encoding, TSP_Encoder, TSP_Decoder, reshape_by_heads, multi_head_attention

class E_TSPModel(TSPModel):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = E_TSP_Decoder(**model_params)
        # conditioned network embedding scale N
        self.cond_1 = nn.Sequential(
                                    nn.Linear(1, 128, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(128, 128, bias=False)
                                    )
        # projection encoding nodes and embedded scale N
        self.proj_N_h = nn.Sequential(
                                    nn.Linear(128, 1024, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(1024, 128, bias=False)
                                    )
        # self.proj_N_h = nn.Sequential(
        #                             nn.Linear(128, 2048, bias=False),
        #                             nn.ReLU(),
        #                             nn.Linear(2048, 128, bias=False)
        #                             )
        self.encoded_nodes = None
        # shape: (batch, problem, embedding)

    def pre_forward(self, reset_state, map_feat=None):  # now includes decoder.set_q1
        self.encoded_nodes = self.encoder(reset_state.problems)

        batch_size = reset_state.problems.size(0)
        problem_size = reset_state.problems.size(1)
        all_nodes = torch.arange(problem_size)[None, :].expand(batch_size, problem_size)

        residual = self.encoded_nodes.detach()

        x =(self.encoded_nodes.shape[1])/100
        scale_input = (torch.ones(self.encoded_nodes.shape[0], 1, 1).cuda() * x) / map_feat
        scale_emb = self.cond_1(scale_input)
        self.encoded_nodes = self.encoded_nodes + self.proj_N_h(self.encoded_nodes + scale_emb)

        self.decoder.set_kv(self.encoded_nodes)
        self.decoder.set_mean_q(self.encoded_nodes)

        encoded_first_node = _get_encoding(self.encoded_nodes, all_nodes)
        # shape: (batch, pomo, embedding)
        self.decoder.set_q1(encoded_first_node)
        # shape: (batch, problem, EMBEDDING_DIM)
        return self.encoded_nodes, residual

    def forward(self, state, distance=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask, first_node=state.first_node, distance=distance)
            # shape: (batch, pomo, problem)

            if self.training or self.model_params['eval_type'] == 'softmax':
                selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                    .reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob

    def forward_w_incumbent(self, state, best_action, distance):
        # best_action.shape = (batch,)
        # first_action.shape = (batch,)

        batch_size = state.BATCH_IDX.size(0)
        pomo_size_p1 = state.BATCH_IDX.size(1)
        pomo_size = pomo_size_p1-1
        probs = None
        if state.current_node is None:  # First Move, POMO
            pomo_select = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            # shape: (batch, pomo)
            selected = torch.cat((pomo_select, best_action[:, None]), dim=1)
            # shape: (batch, pomo+1)
            prob = torch.ones(size=(batch_size, pomo_size_p1))
            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo+1, embedding)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask, first_node=state.first_node, distance=distance)
            # shape: (batch, pomo+1, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size_p1, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size_p1)
                    # shape: (batch, pomo+1)
                    selected[:, -1] = best_action
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size_p1)
                    # shape: (batch, pomo+1)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo+1)
                selected[:, -1] = best_action
                prob = None  # value not needed. Can be anything.

        return selected, prob


class E_TSP_Decoder(TSP_Decoder):

    def __init__(self, **model_params):
        super().__init__(**model_params)

        self.enable_EAS = None  # bool
        self.use_bias = None # bool
        self.eas_W1 = None
        # shape: (batch, embedding, embedding)
        self.eas_b1 = None
        # shape: (batch, embedding)
        self.eas_W2 = None
        # shape: (batch, embedding, embedding)
        self.eas_b2 = None
        # shape: (batch, embedding)
        self.iter = 0

    def init_eas_layers_random(self, batch_size):
        emb_dim = self.model_params['embedding_dim']  # 128
        init_lim = (1/emb_dim)**(1/2)

        weight1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((batch_size, emb_dim, emb_dim))
        bias1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((batch_size, emb_dim))

        self.eas_W1 = torch.nn.Parameter(weight1)
        self.eas_b1 = torch.nn.Parameter(bias1)

        self.eas_W2 = torch.nn.Parameter(torch.zeros(size=(batch_size, emb_dim, emb_dim)))
        self.eas_b2 = torch.nn.Parameter(torch.zeros(size=(batch_size, emb_dim)))

    def init_eas_layers_manual(self, W1, b1, W2, b2):
        self.eas_W1 = torch.nn.Parameter(W1)
        self.eas_b1 = torch.nn.Parameter(b1)
        self.eas_W2 = torch.nn.Parameter(W2)
        self.eas_b2 = torch.nn.Parameter(b2)

    def eas_parameters(self):
        return [self.eas_W1, self.eas_b1, self.eas_W2, self.eas_b2]

    def forward(self, encoded_last_node, ninf_mask, first_node=None, distance=None):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)
        # first_node.shape: (batch, modified_pomo)  # use first_node=None when pomo = {1, 2, ..., problem}

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # if first_node is None:
        #     q_first = self.q_first
        #     # shape: (batch, head_num, pomo, qkv_dim)
        # else:
        #     qkv_dim = self.model_params['qkv_dim']
        #     gathering_index = first_node[:, None, :, None].expand(-1, head_num, -1, qkv_dim)
        #     q_first = self.q_first.gather(dim=2, index=gathering_index)
        #     # shape: (batch, head_num, mod_pomo, qkv_dim)

        q = self.q_first.repeat(1, 1, q_last.shape[2]//self.q_first.shape[2], 1) + q_last + self.q_mean

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        # # EAS Layer Insert
        # #######################################################
        if self.enable_EAS:
            ms1 = torch.matmul(mh_atten_out, self.eas_W1)
            # shape: (batch, pomo, embedding)

            ms1 = ms1 + self.eas_b1[:, None, :]
            # shape: (batch, pomo, embedding)

            ms1_activated = F.relu(ms1)
            # shape: (batch, pomo, embedding)

            ms2 = torch.matmul(ms1_activated, self.eas_W2)
            # shape: (batch, pomo, embedding)

            ms2 = ms2 + self.eas_b2[:, None, :]
            # shape: (batch, pomo, embedding)

            mh_atten_out = mh_atten_out + ms2
            # shape: (batch, pomo, embedding)

        # #  Single-Head Attention, for probability calculation
        # #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        if self.use_bias:
            # distance = None
            if distance == None:
                score_scaled = score / sqrt_embedding_dim
                # shape: (batch, pomo, problem)
            else:
                # Exponential scheduler
                alpha = (1.0*math.exp(math.log(0.3/1.0)/200 * (self.iter)))

                score_scaled = score / sqrt_embedding_dim - distance**alpha

            score_clipped = logit_clipping * torch.tanh(score_scaled)
            score_masked = score_clipped + ninf_mask

            # Temperature annealing
            temp = 1.0*math.exp(math.log(0.3/1)/200 * (self.iter))

            # Exponential scheduler
            probs = F.softmax(score_masked/temp, dim=2)
            # shape: (batch, pomo, problem)
        else:
            score_scaled = score / sqrt_embedding_dim
            score_clipped = logit_clipping * torch.tanh(score_scaled)
            score_masked = score_clipped + ninf_mask

            # temp = 1.0*math.exp(math.log(0.3/1)/200 * (self.iter))
            temp = 1
            probs = F.softmax(score_masked/temp, dim=2)
            # shape: (batch, pomo, problem)
        return probs






