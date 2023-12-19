import torch
import os
import numpy as np
from E_CVRPEnv import E_CVRPEnv as Env
from Finetuned_Model import Finetuned_Model as Model

from torch.optim import Adam as Optimizer

class Label_Trainer:
    def __init__(self,
                 env_params,
                 model_params,
                 run_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.run_params = run_params

        #Node embedding ground truth label
        self.embedding_label = torch.empty(size=(run_params['num_episodes'],env_params['pomo_size']+1,model_params['embedding_dim']))


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
        self.env = Env(**self.env_params)
        train_data_load = self.run_params['train_data_load']
        if train_data_load['enable']:
            filename = train_data_load['filename']
            num_problems = self.run_params['num_episodes']
            index_begin = train_data_load['index_begin']
            self.env.use_pkl_saved_problems(filename, num_problems, index_begin)

        # Model
        self.model = Model(self.run_params['eas_batch_size'],**self.model_params)
        model_load = self.run_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)


    def run(self):

        self._run_eas(num_iter=self.run_params['eas_num_iter'])

        embedding_label =self.embedding_label
        embedding_label = embedding_label.detach().numpy()
        # Check if the folder "emb_label" exists, and if not, create it
        if not os.path.exists("emb_label"):
            os.makedirs("emb_label")
        np.save('./emb_label/symnco_cvrp_label_{}_{}_{}'.format(self.env_params['pomo_size'], self.run_params['eas_num_iter'], self.run_params['num_episodes']),embedding_label)
    ###########################################################################################################
    ###########################################################################################################
    # EAS
    ###########################################################################################################
    ###########################################################################################################

    def _run_eas(self, num_iter=1):

        # Loop
        num_episode = self.run_params['num_episodes']
        episode = 0
        while episode < num_episode:

            self.model.initial_eas_parameters()
            optimizer = Optimizer([{'params': self.model.eas_parameters()}]
                               ,lr=self.run_params['lr'])

            self.model.decoder.iter = 0
            remaining = num_episode - episode
            batch_size = min(self.run_params['eas_batch_size'], remaining)

            # EAS
            self._eas_one_batch(episode, batch_size, num_iter, optimizer)

            episode += batch_size

    def _eas_one_batch(self, episode, batch_size, num_iter, optimizer):

        aug_factor = 1
        aug_batch_size = batch_size * aug_factor
        pomo_size = self.env_params['pomo_size']

        # Ready
        ###############################################
        self.env.load_problems_by_index(episode, batch_size, aug_factor)
        self.env.modify_pomo_size_for_eas(pomo_size)
        reset_state, _, _ = self.env.reset()

        # self.model.requires_grad_(False)
        self.model.pre_forward(reset_state)

        # EAS
        ###############################################
        self.model.train()  # Must in train mode for EAS to work
        self.model.decoder.enable_EAS = False

        pomo_size_p1 = pomo_size
        self.env.modify_pomo_size_for_eas(pomo_size_p1)

        for iter_i in range(num_iter):
            reset_state, _, _ = self.env.reset()
            label = self.model.pre_forward(reset_state)

            if iter_i == num_iter-1:
                self.embedding_label[episode:episode+batch_size,:,:] = label

            state, reward, done = self.env.pre_step()
            prob_list = torch.zeros(size=(aug_batch_size, pomo_size_p1, 0))


            # POMO Rollout with Incumbent
            ###############################################

            while not done:
                # Best_Action from incumbent solution

                selected, prob, probs = self.model(state)
                # shape: (aug_batch, pomo+1)

                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            # Incumbent solution
            ###############################################
            max_reward, max_idx = reward.max(dim=1)  # get best results from pomo + Incumbent

            # shape: (aug_batch,)
            incumbent_score = -max_reward
            self.model.decoder.iter += 1

            # Loss: POMO RL
            ###############################################
            pomo_prob_list = prob_list[:, :pomo_size, :]
            # shape: (aug_batch, pomo, tour_len)
            pomo_reward = reward[:, :pomo_size]
            # shape: (aug_batch, pomo)

            advantage = pomo_reward - pomo_reward.mean(dim=1, keepdim=True)
            # shape: (aug_batch, pomo)
            log_prob = pomo_prob_list.log().sum(dim=2)
            # size = (aug_batch, pomo)
            loss_RL = -advantage * log_prob  # Minus Sign: To increase REWARD

            loss_RL = loss_RL.mean(dim=1)
            # shape: (aug_batch,)

            # Back Propagation
            ###############################################
            optimizer.zero_grad()

            loss = loss_RL #+ self.run_params['lambda'] * loss_IL
            # shape: (aug_batch,)
            loss.sum().backward()

            optimizer.step()

            # Score Curve
            ###############################################
            augbatch_reward = max_reward.reshape(aug_factor, batch_size)
            # shape: (augmentation, batch)
            max_aug_reward, _ = augbatch_reward.max(dim=0)  # get best results from augmentation
            # shape: (batch,)
            print("iter{} : {}".format(iter_i,max_aug_reward.mean()))
