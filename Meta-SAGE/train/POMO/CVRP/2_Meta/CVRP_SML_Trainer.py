import torch
from E_CVRPEnv import E_CVRPEnv as Env
from SML_Model import SML_Model as Model
from torch.optim import Adam as Optimizer
import numpy as np

class CVRP_SML_Trainer:
    def __init__(self,
                 env_params,
                 model_params,
                 run_params):

        # save arguments
        self.env_params_lst = env_params
        # self.env_params = env_params
        self.model_params = model_params
        self.run_params = run_params

        # cuda
        if self.run_params['use_cuda']:
            cuda_device_num = self.run_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV
        self.envs = {
                    "2H": Env(**self.env_params_lst[0]),
                    "3H": Env(**self.env_params_lst[1]),
                    "4H": Env(**self.env_params_lst[2]),
                    # "5H": Env(**self.env_params_lst[3])
                    }
        self.train_scale = ["2H","3H","4H"]

        # Model
        self.model = Model(**self.model_params)
        model_load = self.run_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)


    def run(self):

        train_data_load = self.run_params['train_data_load']

        np_load_200 = np.load('./emb_label/pomo_cvrp_label_200_100_3000.npy')
        np_load_300 = np.load('./emb_label/pomo_cvrp_label_300_100_3000.npy')
        np_load_400 = np.load('./emb_label/pomo_cvrp_label_400_100_3000.npy')


        label2 = torch.from_numpy(np_load_200).to('cuda')
        label3 = torch.from_numpy(np_load_300).to('cuda')
        label4 = torch.from_numpy(np_load_400).to('cuda')

        label = [label2, label3, label4]

        optimizer = Optimizer([{'params': self.model.cond_1.parameters()},
                            {'params': self.model.proj_N_h.parameters()}]
                            ,lr=0.001,weight_decay=1e-6)
        num_episode = min(self.run_params['num_episodes'], label2.shape[0])

        epoch = 0
        while(epoch < 2):
            print("epoch : {}".format(epoch))
            episode = 0
            step = 0
            while episode < num_episode:
                loss_q = 0
                total_rl = 0
                total_sl = 0
                remaining = num_episode - episode
                batch_size = min(self.run_params['eas_batch_size'], remaining)

                for i, scale in enumerate(self.train_scale):
                    filename = train_data_load['filename'][i]
                    num_problems = self.run_params['num_episodes']
                    index_begin = train_data_load['index_begin']
                    self.env = self.envs[self.train_scale[i]]
                    self.env.use_pkl_saved_problems(filename, num_problems, index_begin)
                    self.env_params = self.env_params_lst[i]
                    target = label[i][episode:episode+batch_size,:,:]
                    aug_factor = 1
                    pomo_size = self.env_params['pomo_size']
                    self.env.load_problems_by_index(episode, batch_size, aug_factor)
                    self.env.modify_pomo_size_for_eas(pomo_size)
                    rl_loss, sl_loss = self.train_scale_bias(target)
                    total_rl += rl_loss
                    total_sl += sl_loss
                loss_q = (total_rl + total_sl) / len(label)
                optimizer.zero_grad()
                loss_q.backward()
                optimizer.step()

                print("epoch : {}, step : {}, sl_loss : {}".format(epoch, step, total_sl))
                print("epoch : {}, step : {}, rl_loss : {}".format(epoch, step, total_rl))
                step += 1
                episode += batch_size

            epoch += 1
            torch.save({
                'model_state_dict': self.model.state_dict()
                }, "../pretrained/pomo_cvrp_with_sml.pt")

    def train_scale_bias(self, target):

        loss_SL = 0
        # Ready
        ###############################################

        target_scale = target.shape[1] - 1
        reset_state, _, _ = self.env.reset()
        # scale_lst, index = self.model.pre_forward(reset_state,initial=False)
        target_idx = target_scale // 100
        distance_matrix = torch.cdist(self.env.depot_node_xy, self.env.depot_node_xy)
        distance_mean = distance_matrix.mean(-1, keepdim=True)#.mean(1, keepdim=True)
        map_feat1 = distance_matrix.topk(k=100, dim=-1, largest=False)[0].mean(-1, keepdim=True)#.mean(1, keepdim=True)
        map_feat = distance_mean / map_feat1

        scale_bias = self.model.pre_forward(reset_state, map_feat=map_feat)
        prob_list = torch.zeros(size=(target.shape[0], target.shape[1]-1, 0))
        state, reward, done = self.env.pre_step()
        self.model.train()  # Must in train mode for EAS to work
        self.model.decoder.enable_EAS = False

        loss_SL = torch.sqrt(torch.sum((scale_bias- target)**2)) / scale_bias.shape[0]

        while not done:
            selected, prob = self.model(state)
            # shape: (aug_batch, pomo+1)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # # # Loss: POMO RL
        # # ###############################################
        pomo_prob_list = prob_list[:, :target_scale, :]
        # # shape: (aug_batch, pomo, tour_len)
        pomo_reward = reward[:, :target_scale]

        # shape: (aug_batch, pomo)
        advantage = (pomo_reward - pomo_reward.mean(dim=1, keepdim=True))
        # shape: (aug_batch, pomo)
        log_prob = pomo_prob_list.log().sum(dim=2)
        # size = (aug_batch, pomo)
        loss_RL = -advantage * log_prob  # Minus Sign: To increase REWARD

        loss_RL = (loss_RL).mean(dim=1).sum()

        return loss_RL, loss_SL

