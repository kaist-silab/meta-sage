import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from CVRPModel import CVRPModel, _get_encoding, CVRP_Encoder, CVRP_Decoder, reshape_by_heads, multi_head_attention, multi_head_attention1

class E_CVRPModel(CVRPModel):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params
        self.iter = 0
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = EAS_CVRP_Decoder(**model_params)

        # # conditioned network embedding scale N
        self.cond_1 = nn.Sequential(
                                    nn.Linear(1, 128, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(128, 128, bias=False)
                                    )
        # # projection encoding nodes and embedded scale N

        self.proj_N_h = nn.Sequential(
                                    nn.Linear(128, 1024, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(1024, 128, bias=False)
                                    )
        self.encoded_nodes = None
        # shape: (batch, problem, embedding)

    def pre_forward(self, reset_state, map_feat=None, bias = True, ):

        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
        # shape: (batch, problem, EMBEDDING_DIM)
        if bias:
            x =(self.encoded_nodes.shape[1])/100
            scale_input = (torch.ones(self.encoded_nodes.shape[0], 1, 1).cuda() * x) / map_feat

            scale_emb = self.cond_1(scale_input)
            self.encoded_nodes = self.encoded_nodes + self.proj_N_h(self.encoded_nodes + scale_emb)

        self.decoder.set_kv(self.encoded_nodes)


    def forward_w_incumbent(self, state, best_action=None, distance=None, map_feat=None):
        # best_action.shape = (batch,)

        batch_size = state.BATCH_IDX.size(0)
        pomo_size_p1 = state.BATCH_IDX.size(1)
        pomo_size = pomo_size_p1

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size_p1), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size_p1))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            # selected = torch.cat((selected, best_action[:, None]), dim=1)
            # shape: (batch, pomo+1)
            prob = torch.ones(size=(batch_size, pomo_size_p1))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo+1, embedding)
            probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask, distance=distance, map_feat=map_feat)
            # shape: (batch, pomo+1, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                # while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                with torch.no_grad():
                    selected = probs.reshape(batch_size * pomo_size_p1, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size_p1)
                # shape: (batch, pomo+1)
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size_p1)

            else:
                selected = probs.argmax(dim=2)
                prob = None  # value not needed. Can be anything.

        return selected, prob


    def forward_w_incumbent_probs(self, state, best_action, distance=None, map_feat=None):
        # best_action.shape = (batch,)

        batch_size = state.BATCH_IDX.size(0)
        pomo_size_p1 = state.BATCH_IDX.size(1)
        pomo_size = pomo_size_p1-1

        probs = None

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size_p1), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size_p1))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            selected = torch.cat((selected, best_action[:, None]), dim=1)
            # shape: (batch, pomo+1)
            prob = torch.ones(size=(batch_size, pomo_size_p1))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo+1, embedding)
            probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask, distance=distance, map_feat=map_feat)
            # shape: (batch, pomo+1, problem+1)

            if (self.training or self.model_params['eval_type'] == 'softmax'):
                # while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                with torch.no_grad():
                    selected = probs.reshape(batch_size * pomo_size_p1, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size_p1)
                # shape: (batch, pomo+1)
                selected[:, -1] = best_action
                # print(probs.shape)
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size_p1)
                # shape: (batch, pomo+1)
            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo+1)
                selected[:, -1] = best_action
                prob = None  # value not needed. Can be anything.

        return selected, prob


class EAS_CVRP_Decoder(CVRP_Decoder):

    def __init__(self, **model_params):
        super().__init__(**model_params)

        self.enable_EAS = None  # bool

        self.eas_W11 = None
        # shape: (batch, embedding, embedding)
        self.eas_b11 = None
        # shape: (batch, embedding)
        self.eas_W21 = None
        # shape: (batch, embedding, embedding)
        self.eas_b21 = None
        # shape: (batch, embedding)
        self.iter = 0
        self.use_bias = None

        # self.proj_N_h = nn.Sequential(
        #                             nn.Linear(128, 1024, bias=False),
        #                             nn.ReLU(),
        #                             nn.Linear(1024, 128, bias=False)
        #                             )
        # self.cond_1 = nn.Sequential(
        #                             nn.Linear(1, 256, bias=False),
        #                             nn.ReLU(),
        #                             nn.Linear(256, 128, bias=False)
        #                             )
    def init_eas_layers_random(self, batch_size):
        emb_dim = self.model_params['embedding_dim']  # 128
        init_lim = (1 / emb_dim) ** (1 / 2)

        weight1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((batch_size, emb_dim, emb_dim))
        bias1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((batch_size, emb_dim))
        self.eas_W11 = torch.nn.Parameter(weight1)
        self.eas_b11 = torch.nn.Parameter(bias1)
        self.eas_W21 = torch.nn.Parameter(torch.zeros(size=(batch_size, emb_dim, emb_dim)))
        self.eas_b21 = torch.nn.Parameter(torch.zeros(size=(batch_size, emb_dim)))

    def eas_parameters(self):
        return [self.eas_W11, self.eas_b11, self.eas_W21, self.eas_b21]

    def forward(self, encoded_last_node, load, ninf_mask, distance=None, map_feat=None):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # out_concat = multi_head_attention(q_last, self.k, self.v, rank3_ninf_mask=ninf_mask, distance=distance)
        out_concat = multi_head_attention(q_last, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)

        # mh_atten_out += self.proj_N_h(mh_atten_out)
        # residual = mh_atten_out.detach()
        # shape: (batch, pomo, embedding)
        # mh_atten_out = mh_atten_out
        # EAS Layer Insert
        #######################################################
        if self.enable_EAS:

            ms1 = torch.matmul(mh_atten_out, self.eas_W11)
            # shape: (batch, pomo, embedding)

            ms1 = ms1 + self.eas_b11[:, None, :]
            # shape: (batch, pomo, embedding)

            ms1_activated = F.relu(ms1)
            # shape: (batch, pomo, embedding)

            ms2 = torch.matmul(ms1_activated, self.eas_W21)
            # shape: (batch, pomo, embedding)

            ms2 = ms2 + self.eas_b21[:, None, :]
            # shape: (batch, pomo, embedding)

            mh_atten_out = mh_atten_out + ms2
            # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        # score = torch.matmul(mh_atten_out+ self.proj_N_h(mh_atten_out), self.single_head_key)

        # x =(encoded_last_node.shape[1])/100
        # # # x =1100/100
        # scale_input = (torch.ones(encoded_last_node.shape[0], 1, 1).cuda() * x) / map_feat
        # scale_emb = self.cond_1(scale_input)
        # mh_atten_out += scale_emb
        score = torch.matmul(mh_atten_out, self.single_head_key)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        if self.use_bias:
            # if self.iter <=200:
            if distance == None:
                score_scaled = score / sqrt_embedding_dim
                # shape: (batch, pomo, problem)
            else:
                # Exponential scheduler
                alpha = math.exp(math.log(0.3)/200 * (self.iter))
                score_scaled = score / sqrt_embedding_dim - distance**alpha

            score_clipped = logit_clipping * torch.tanh(score_scaled)
            score_masked = score_clipped + ninf_mask
            # Exponential scheduler
            probs = F.softmax(score_masked / (math.exp(math.log(0.3)/200 * (self.iter))), dim=2)
            # probs = F.softmax(score_masked, dim=2)
            # else:
            #     score_scaled = score / sqrt_embedding_dim - distance**0.3
            #     score_clipped = logit_clipping * torch.tanh(score_scaled)
            #     score_masked = score_clipped + ninf_mask
            #     probs = F.softmax(score_masked /0.3, dim=2)
        else:
            score_scaled = score / sqrt_embedding_dim
            score_clipped = logit_clipping * torch.tanh(score_scaled)
            score_masked = score_clipped + ninf_mask
            probs = F.softmax(score_masked, dim=2)
        return probs




