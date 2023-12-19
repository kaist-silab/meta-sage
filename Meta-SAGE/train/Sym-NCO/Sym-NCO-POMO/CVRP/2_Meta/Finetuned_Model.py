import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from CVRPModel_ours import CVRPModel, _get_encoding, CVRP_Encoder, CVRP_Decoder, reshape_by_heads, multi_head_attention

class Finetuned_Model(CVRPModel):

    def __init__(self, batch, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = EAS_CVRP_Decoder(**model_params)
        
        self.batch_size = batch
        emb_dim = self.model_params['embedding_dim']  # 128
        init_lim = (1/emb_dim)**(1/2)
        
        weight1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((self.batch_size, emb_dim, emb_dim))
        bias1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((self.batch_size, emb_dim))
        
        self.eas_W1 = torch.nn.Parameter(weight1)
        self.eas_b1 = torch.nn.Parameter(bias1)
        self.eas_W2 = torch.nn.Parameter(torch.zeros(size=(self.batch_size, emb_dim, emb_dim)))
        self.eas_b2 = torch.nn.Parameter(torch.zeros(size=(self.batch_size, emb_dim)))
        
        self.encoded_nodes = None
        # shape: (batch, problem, embedding)
    
    def initial_eas_parameters(self):
        
        batch_size = self.batch_size
        emb_dim = self.model_params['embedding_dim']  # 128
        init_lim = (1/emb_dim)**(1/2)
        
        weight1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((batch_size, emb_dim, emb_dim))
        bias1 = torch.torch.distributions.Uniform(low=-init_lim, high=init_lim).sample((batch_size, emb_dim))
        
        self.eas_W1 = torch.nn.Parameter(weight1)
        self.eas_b1 = torch.nn.Parameter(bias1)
        self.eas_W2 = torch.nn.Parameter(torch.zeros(size=(batch_size, emb_dim, emb_dim)))
        self.eas_b2 = torch.nn.Parameter(torch.zeros(size=(batch_size, emb_dim)))
        
    def eas_parameters(self):
        return [self.eas_W1, self.eas_b1, self.eas_W2, self.eas_b2]

    def pre_forward(self, reset_state, return_h_mean=False):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
        
        residual = self.encoded_nodes.detach()
        ms1 = torch.matmul(self.encoded_nodes, self.eas_W1)
            # shape: (batch, pomo, embedding)

        ms1 = ms1 + self.eas_b1[:, None, :]
        # shape: (batch, pomo, embedding)

        ms1_activated = F.gelu(ms1)
        # shape: (batch, pomo, embedding)

        ms2 = torch.matmul(ms1_activated, self.eas_W2)
        # shape: (batch, pomo, embedding)

        ms2 = ms2 + self.eas_b2[:, None, :]
        # shape: (batch, pomo, embedding)

        self.encoded_nodes = residual + ms2
        # shape: (batch, problem+1, embedding)
        
        self.decoder.set_kv(self.encoded_nodes)

        self.decoder.set_mean_q(self.encoded_nodes)

        if return_h_mean:
            return self.decoder.projection(self.encoded_nodes)
        return ms2
    def forward(self, state):
        # best_action.shape = (batch,)

        batch_size = state.BATCH_IDX.size(0)
        pomo_size_p1 = state.BATCH_IDX.size(1)
        pomo_size = pomo_size_p1

        probs = None
        
        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size_p1), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size_p1))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            # shape: (batch, pomo)
            prob = torch.ones(size=(batch_size, pomo_size_p1))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo+1, embedding)
            probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo+1, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                # while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                with torch.no_grad():
                    selected = probs.reshape(batch_size * pomo_size_p1, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size_p1)
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size_p1)

            else:
                selected = probs.argmax(dim=2)
                prob = None  # value not needed. Can be anything.

        return selected, prob, probs


class EAS_CVRP_Decoder(CVRP_Decoder):

    def __init__(self, **model_params):
        super().__init__(**model_params)

        self.enable_EAS = None  # bool

        self.eas_W1 = None
        # shape: (batch, embedding, embedding)
        self.eas_b1 = None
        # shape: (batch, embedding)
        self.eas_W2 = None
        # shape: (batch, embedding, embedding)
        self.eas_b2 = None
        # shape: (batch, embedding)
        self.cond_1 = nn.Sequential(nn.Linear(1, 128, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(128, 128, bias=False)
        )

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
        
    def forward(self, encoded_last_node, load, ninf_mask):
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

        q = q_last + self.q_mean

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        # EAS Layer Insert
        #######################################################
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

        #  Single-Head Attention, for probability calculation
        #######################################################
       
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        score_scaled = score / sqrt_embedding_dim

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs





