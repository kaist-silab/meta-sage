import torch
import logging
import time
import os

from E_CVRPEnv import E_CVRPEnv as Env
from E_CVRPModel import E_CVRPModel as Model

from torch.optim import Adam as Optimizer

from utils.utils import get_result_folder, TimeEstimator, AverageMeter, LogData, util_print_log_array


class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 run_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.run_params = run_params

        # result folder, logger
        self.logger = logging.getLogger()
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # .vrp file max coordinate value
        self.max_val = 0

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
        test_data_load = self.run_params['test_data_load']
        if test_data_load['enable']:
            filename = test_data_load['filename']
            num_problems = self.run_params['num_episodes']
            index_begin = test_data_load['index_begin']
            if filename.endswith(".pkl"):
                self.env.use_pkl_saved_problems(filename, num_problems, index_begin)
            else:
                # Read .vrp file
                problem_size, pomo_size, max_val = self.env.read_instance_vrp(filename)
                self.env_params['problem_size'] = problem_size
                self.env_params['pomo_size'] = pomo_size
                self.max_val = max_val

        # Store SAGE score
        num_episode = self.run_params['num_episodes']

        self.sage_score = torch.empty(size=(1, num_episode,self.run_params['sage_num_iter']), device='cpu')

        # Model
        self.model = Model(**self.model_params)

        model_load = self.run_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        # SAGE Loop
        self.logger.info("[{}] Meta-SAGE Loop Started ".format(self.device))
        start_time = time.time()
        self.time_estimator.reset()

        # SAGE

        sage_start_hr = (time.time() - start_time) / 3600.0
        score_curve = self._run_sage(num_iter=self.run_params['sage_num_iter'])
        sage_stop_hr = (time.time() - start_time) / 3600.0

        self.result_log.append('Meta-SAGE_start_time', sage_start_hr)
        self.result_log.append('Meta-SAGE_end_time', sage_stop_hr)
        self.result_log.append('Meta-SAGE_start_score', score_curve[0].item())
        self.result_log.append('Meta-SAGE_end_score', score_curve[-1].item())

        interval = (sage_stop_hr - sage_start_hr) * 3600.0
        print("Time : {} seconds".format(interval))
        print("Score : {}".format(score_curve[-1].item()))

        # Done
        self.logger.info("[{}] *** Done *** ".format(self.device))
        util_print_log_array(self.logger, self.result_log)
    ###########################################################################################################
    ###########################################################################################################
    # EAS
    ###########################################################################################################
    ###########################################################################################################

    def _run_sage(self, num_iter=1):

        result_curve = torch.zeros(size=(num_iter,))

        checkpoint = torch.load("../1_pretrained_model/pomo_cvrp_with_sml.pt", map_location=self.device)
        # # Load SML
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Loop
        num_episode = self.run_params['num_episodes']
        episode = 0
        while episode < num_episode:

            self.model.decoder.iter = 0
            remaining = num_episode - episode
            batch_size = min(self.run_params['sage_batch_size'], remaining)

            # SAGE
            sum_score_curve = self._sage_one_batch(episode, batch_size, num_iter)
            # shape: (num_iter,)
            result_curve += sum_score_curve

            episode += batch_size

            self.logger.info("\tMeta-SAGE batch [{}:{}] score: {:f}".format(
                episode-batch_size, episode, sum_score_curve[-1].item()/batch_size))

        # Done
        score_curve = result_curve / num_episode
        return score_curve

    def _sage_one_batch(self, episode, batch_size, num_iter):

        aug_factor = self.run_params['aug']
        aug_batch_size = batch_size * aug_factor
        pomo_size = self.env_params['pomo_size']
        sum_score_curve = torch.empty(size=(num_iter,))

        # Ready
        ###############################################
        self.env.load_problems_by_index(episode, batch_size, aug_factor)
        self.env.modify_pomo_size_for_eas(pomo_size)
        reset_state, _, _ = self.env.reset()

        # self.model.eval()
        map_feat = None
        distance_matrix = None

        if self.run_params['use_bias']:
            distance_matrix = torch.cdist(self.env.depot_node_xy, self.env.depot_node_xy)
            distance_mean = distance_matrix.mean(-1, keepdim=True)#.mean(1, keepdim=True)
            map_feat1 = distance_matrix.topk(k=100, dim=-1, largest=False)[0].mean(-1, keepdim=True)#.mean(1, keepdim=True)
            map_feat = distance_mean / map_feat1

        self.model.requires_grad_(False)
        # self.model.pre_forward(reset_state, map_feat)

        # Calculate distance per instance

        self.env.reset()
        pomo_size_p1 = pomo_size + 1
        self.env.modify_pomo_size_for_eas(pomo_size)
        prob_list = torch.zeros(size=(aug_batch_size, pomo_size_p1, 0))

        reset_state, _, _ = self.env.reset()
        incumbent_solution, incumbent_score = self._initial_pomo_greedy_rollout(self.env, self.model, distance_matrix, map_feat)

        # SAGE
        ###############################################

        self.model.train()  # Must in train mode for EAS to work
        self.model.decoder.enable_EAS = True
        self.model.decoder.use_bias = self.run_params['use_bias']

        self.model.decoder.init_eas_layers_random(aug_batch_size)

        optimizer = Optimizer([{'params': self.model.decoder.eas_parameters()}],
                              lr=self.run_params['lr'])

        pomo_size_p1 = pomo_size + 1
        self.env.modify_pomo_size_for_eas(pomo_size_p1)
        distance = None
        for iter_i in range(num_iter):

            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state, map_feat)

            prob_list = torch.zeros(size=(aug_batch_size, pomo_size_p1, 0))

            # POMO Rollout with Incumbent
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                # Best_Action from incumbent solution
                if distance_matrix != None:
                    if state.current_node != None:
                        distance_index = state.current_node[:, :, None].repeat(1, 1,
                                                                            distance_matrix.shape[
                                                                                2])
                        distance = torch.gather(distance_matrix, 1, distance_index.type(torch.int64))
                    else:
                        distance = None
                step_cnt = self.env.selected_count
                best_action = incumbent_solution[:, step_cnt]

                selected, prob = self.model.forward_w_incumbent_probs(state, best_action, distance, map_feat)
                # shape: (aug_batch, pomo+1)

                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            # Incumbent solution
            ###############################################
            max_reward, max_idx = reward.max(dim=1)  # get best results from pomo + Incumbent
            # shape: (aug_batch,)
            incumbent_score = -max_reward
            self.model.decoder.iter += 1
            self.model.iter += 1
            gathering_index = max_idx[:, None, None].expand(-1, 1, self.env.selected_count)
            new_incumbent_solution = self.env.selected_node_list.gather(dim=1, index=gathering_index)
            new_incumbent_solution = new_incumbent_solution.squeeze(dim=1)
            # shape: (aug_batch, tour_len)

            solution_max_length = self.run_params['solution_max_length']
            incumbent_solution = torch.zeros(size=(aug_factor*batch_size, solution_max_length), dtype=torch.long)
            incumbent_solution[:, :self.env.selected_count] = new_incumbent_solution

            # Loss: POMO RL
            ###############################################
            pomo_prob_list = prob_list[:, :pomo_size, :]
            # shape: (aug_batch, pomo, tour_len)
            pomo_reward = reward[:, :pomo_size]
            # shape: (aug_batch, pomo)
            advantage = (pomo_reward - pomo_reward.mean(dim=1, keepdim=True))
            # shape: (aug_batch, pomo)
            log_prob = pomo_prob_list.log().sum(dim=2)
            # size = (aug_batch, pomo)
            loss_RL = -advantage * log_prob  # Minus Sign: To increase REWARD

            loss_RL = loss_RL.mean(dim=1)
            # shape: (aug_batch,)

            # Loss: IL
            ###############################################
            imitation_prob_list = prob_list[:, -1, :]
            # shape: (aug_batch, tour_len)
            log_prob = imitation_prob_list.log().sum(dim=1)
            # shape: (aug_batch,)
            loss_IL = -log_prob  # Minus Sign: to increase probability
            # shape: (aug_batch,)

            # Back Propagation
            ###############################################
            optimizer.zero_grad()

            loss = loss_RL + self.run_params['lambda'] * loss_IL
            # shape: (aug_batch,)
            loss.sum().backward()

            optimizer.step()

            # Score Curve
            ###############################################
            augbatch_reward = max_reward.reshape(aug_factor, batch_size)
            # shape: (augmentation, batch)
            max_aug_reward, _ = augbatch_reward.max(dim=0)  # get best results from augmentation
            # shape: (batch,)
            if self.max_val > 0:
                print("iter{}: {}".format(iter_i, max_aug_reward.mean() * self.max_val))
                sum_score = -max_aug_reward.sum() * self.max_val  # negative sign to make positive value, and real tour length .vrp file
                sum_score_curve[iter_i] = sum_score
                # Store sage_score
                sage_score = max_aug_reward.reshape(1, batch_size).to('cpu') * self.max_val
                self.sage_score[:, episode:episode+batch_size] = sage_score
            else:
                print("iter{}: {}".format(iter_i, max_aug_reward.mean()))
                sum_score = -max_aug_reward.sum()  # negative sign to make positive value
                sum_score_curve[iter_i] = sum_score
                # sage_score = max_aug_reward.reshape(1, batch_size).to('cpu')
                # self.sage_score[:, episode:episode+batch_size] = sage_score

        return sum_score_curve

    def _initial_pomo_greedy_rollout(self, env, model, distance_matrix = None, map_feat= None):

        model.eval()
        model.decoder.enable_EAS = False

        reset_state, _, _ = env.reset()
        # model.requires_grad_(False)
        self.model.pre_forward(reset_state, map_feat, bias = self.run_params['use_bias'])

        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()
        distance = None
        while not done:
            if distance_matrix != None:
                if state.current_node != None:
                    distance_index = state.current_node[:, :, None].repeat(1, 1,
                                                                        distance_matrix.shape[
                                                                            2])
                    distance = torch.gather(distance_matrix, 1, distance_index.type(torch.int64))

                else:
                    distance = None
            selected, _ = model.forward_w_incumbent(state, best_action=None, distance=distance)
            state, reward, done = env.step(selected)

        # Score
        ###############################################
        max_pomo_reward, max_pomo_idx = reward.max(dim=1)  # get best results from pomo
        # shape: (aug_batch,)
        incumbent_score = -max_pomo_reward

        # Solution
        ###############################################
        all_solutions = self.env.selected_node_list
        # shape: (aug_batch, pomo+1, tour_len)
        tour_len = all_solutions.size(2)
        gathering_index = max_pomo_idx[:, None, None].expand(-1, 1, tour_len)
        best_solution = all_solutions.gather(dim=1, index=gathering_index).squeeze(dim=1)
        # shape: (aug_batch, tour_len)

        aug_batch_size = best_solution.size(0)
        solution_max_length = self.run_params['solution_max_length']
        incumbent_solution = torch.zeros(size=(aug_batch_size, solution_max_length), dtype=torch.long)
        incumbent_solution[:, :tour_len] = best_solution

        return incumbent_solution, incumbent_score
