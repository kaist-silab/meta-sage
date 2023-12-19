##########################################################################################
# Machine Environment Config

USE_CUDA = True
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
import torch
torch.manual_seed(1234)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# parameters

#########################
# Parameters - Base
#########################

env_params = {
    'problem_size': 200,
    'pomo_size': 200,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'softmax',
}

#########################
# Parameters - EAS
#########################

run_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '../1_pre_trained_model/20220226_114432_train__tsp_n100__3000epoch',
        # directory path of pre-trained model and log files saved.
        'epoch': 1900,  # epoch version of pre-trained model to laod.
    },

    'train_data_load': {
        'enable': True,
        'filename': ['../../../../data_generation/data/train/tsp/tsp200_train_seed12345.pkl'],
        'index_begin': 0,
    },

    'num_episodes': 1000,
    'solution_max_length': 2000,  # for buffer length storing solution

    # EAS Params
    'lr': 0.0032,
    'lambda': 0.01,
    'beta': 0.00,
    'eas_num_iter': 201,
    'eas_batch_size': 5,
}

##########################################################################################
# main

import argparse
from Label_Trainer import LabelTrainer as Trainer


def main():
    trainer = Trainer(env_params=env_params,
                    model_params=model_params,
                    run_params=run_params)
    trainer.run()

def parse_args(arg_str = None):
    global CUDA_DEVICE_NUM
    global run_params

    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', type=int)
    parser.add_argument('--jump', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument("--problem_size", type=int)
    parser.add_argument("--eas_batch_size", type=int)
    parser.add_argument("--eas_num_iter", type=int, default=101)
    parser.add_argument("--seed", type=int, default=12345)

    if arg_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=arg_str.split())

    if args.ep is not None:
        run_params['num_episodes'] = args.ep

    if args.jump is not None:
        num_episodes = run_params['num_episodes']
        run_params['train_data_load']['index_begin'] += args.jump * num_episodes

    if args.gpu is not None:
        CUDA_DEVICE_NUM = args.gpu
        run_params['cuda_device_num'] = args.gpu


    if args.eas_batch_size is not None:
        run_params['eas_batch_size'] = args.eas_batch_size
    if args.eas_num_iter is not None:
        run_params['eas_num_iter'] = args.eas_num_iter

    if args.problem_size is not None:
        env_params['problem_size'] = args.problem_size
        env_params['pomo_size'] = args.problem_size
        run_params['solution_max_length'] = args.problem_size
        run_params['train_data_load']['filename'] = '../../../../data_generation/data/train/tsp/tsp{}_train_seed{}.pkl'.format(args.problem_size, args.seed)


if __name__ == "__main__":
    parse_args()
    main()

