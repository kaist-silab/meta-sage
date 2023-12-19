import math
import torch
import logging
import time
import copy
import pickle
import os, json
import matplotlib.pyplot as plt
import numpy as np
from E_CVRPEnv import E_CVRPEnv as Env
from E_CVRPModel import E_CVRPModel as Model

from torch.optim import Adam as Optimizer
import torch.optim as optim

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
        # self.logger = logging.getLogger()
        # self.result_folder = get_result_folder()
        # self.result_log = LogData()

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
            self.env.use_pkl_saved_problems(filename, num_problems, index_begin)

        # Model
        self.model = Model(**self.model_params)
        np_load_500 = np.load('./emb_label/symnco_tsp_label_500_3000_200.npy')

        # np_load_500 = np.load('../emb_label/symnco_emb_label_500_1000_200.npy')
        self.label = torch.from_numpy(np_load_500).to('cuda') 
        print(self.label.shape)
        num_episode = self.run_params['num_episodes']
        self.sml_similarity_score = torch.empty(size=(num_episode,), device='cpu')
        self.original_similarity_score = torch.empty(size=(num_episode,), device='cpu')

        model_load = self.run_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # utility

    def run(self):        
        # EAS
        self._run_eas(num_iter=self.run_params['eas_num_iter'])
        print("sml_similarity : {}".format(self.sml_similarity_score.mean()))
        print("original_similarity : {}".format(self.original_similarity_score.mean()))
    ###########################################################################################################
    ###########################################################################################################
    # EAS
    ###########################################################################################################
    ###########################################################################################################

    def _run_eas(self, num_iter=1):

        # checkpoint = torch.load("../1_pretrained_model/ICML_tsp_symnco_weighted_earlystop_SL_RL_1000_epoch50_SCA.pt", map_location=self.device)
        # checkpoint = torch.load("../1_pretrained_model/ICML_tsp_symnco_no_prior_earlystop_SL_RL_1000_epoch50_SCA.pt", map_location=self.device)
        # checkpoint = torch.load("../1_pretrained_model/tsp_symnco_prior_earlystop_SL_RL_1000_epoch50_SCA_new.pt", map_location=self.device)
        # checkpoint = torch.load("../1_pretrained_model/ICML_tsp_symnco_no_prior_earlystop_RL_1000_epoch50_SCA_final.pt", map_location=self.device)
        # checkpoint = torch.load("../1_pretrained_model/ICML_tsp_symnco_noprior_noearlystop_RL_1000_epoch50_SCA_new_zebal.pt", map_location=self.device) #no prior SL임
        # checkpoint = torch.load("../1_pretrained_model/ICML_tsp_symnco_prior_noearlystop_RL_1000_epoch50_SCA_new_zebal.pt", map_location=self.device) #RL임
        # checkpoint = torch.load("../1_pretrained_model/ICML_tsp_symnco_noprior_noearlystop_RL_3000_epoch50_SCA_new_zebal_3.pt", map_location=self.device) #data 2000 no prior SL임
        # checkpoint = torch.load("../1_pretrained_model/ICML_tsp_symnco_noprior_noearlystop_RL_3000_epoch50_SCA_new_zebal_RL.pt", map_location=self.device) #data 2000 no prior RL임
        # checkpoint = torch.load("../1_pretrained_model/ICML_confim_no_prior_1000_RL_SL_epoch1.pt", map_location=self.device) #data 5000  SL+RL인데,prior임
        # checkpoint = torch.load("../1_pretrained_model/ICML_confim_no_prior_1000_RL_epoch1_2.pt", map_location=self.device) #data  RL인데,prior임
        # checkpoint = torch.load("../1_pretrained_model/ICML_confim_no_prior_1000_RL_SL_epoch1.pt", map_location=self.device) #data 5000  SL+RL인데,prior임
        checkpoint = torch.load("../1_pretrained_model/ICML_cvrp_confim_no_prior_3000_SL_RL_epoch1.pt", map_location=self.device) #data 5000  SL+RL인데,prior임
        # checkpoint = torch.load("../1_pretrained_model/ICML_final_cvrp_symnco_weighted_earlystop_SL_RL_epoch50_1000data_SCA.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Loop
        num_episode = self.run_params['num_episodes']
        episode = 0
        while episode < num_episode:
            self.model.decoder.iter = 0
            remaining = num_episode - episode
            batch_size = min(self.run_params['eas_batch_size'], remaining)
            
            # EAS
            self._eas_one_batch(episode, batch_size, num_iter)
            
            # shape: (num_iter,)
            episode += batch_size

    def _eas_one_batch(self, episode, batch_size, num_iter):

        aug_factor = self.run_params['aug']
        aug_batch_size = batch_size * aug_factor
        pomo_size = self.env_params['pomo_size']
        # Ready
        ###############################################
        self.env.load_problems_by_index(episode, batch_size, aug_factor)
        self.env.modify_pomo_size_for_eas(pomo_size)
        reset_state, _, _ = self.env.reset()
        
        map_feat = None
        distance_matrix = None
        
        # Calculate distance per instance
        # if self.run_params['use_bias']:
        distance_matrix = torch.cdist(self.env.depot_node_xy, self.env.depot_node_xy)
        distance_mean = distance_matrix.mean(-1, keepdim=True)#.mean(1, keepdim=True)
        map_feat1 = distance_matrix.topk(k=100, dim=-1, largest=False)[0].mean(-1, keepdim=True)#.mean(1, keepdim=True)
        map_feat = distance_mean / map_feat1
        
        # self.env.reset()
        pomo_size_p1 = pomo_size + 1
        
        self.env.modify_pomo_size_for_eas(pomo_size)
    
        reset_state, _, _ = self.env.reset()
        
        label = self.label[episode:episode+batch_size,:,:]

        self.sml_similarity_score[episode:episode+batch_size], self.original_similarity_score[episode:episode+batch_size] = self.cal_similarity(distance_matrix, map_feat, label)
        
    def cal_similarity(self, distance_matrix, map_feat, label):

        self.model.eval()
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        # POMO Rollout
        ###############################################
        reset_state, _, _ = self.env.reset()

        SML_embedding, original_embedding = self.model.pre_forward(reset_state, map_feat)
        
        sml_similarity = abs(SML_embedding - original_embedding - label).sum(dim=-1).sum(dim=-1)
        original_similarity = abs(original_embedding - original_embedding- label).sum(dim=-1).sum(dim=-1)
        print(sml_similarity)
        print(original_similarity)
        # sml_similarity = cos(SML_embedding, label).mean(1)
        # original_similarity = cos(original_embedding, label).mean(1)
        return sml_similarity, original_similarity