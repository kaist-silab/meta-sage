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

env_params = [
{
    'problem_size': 200,
    'pomo_size': 200,
},
{
    'problem_size': 300,
    'pomo_size': 300,
},
{
    'problem_size': 400,
    'pomo_size': 400,
},
]

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
        'path': '../1_pre_trained_model/Sym-NCO',
        # directory path of pre-trained model and log files saved.
        'epoch': 2010,  # epoch version of pre-trained model to laod.
    },
    'train_data_load': {
        'enable': True,
        'filename': ['../../../../../data_generation/data/train/tsp/tsp200_train_seed12345.pkl', '../../../../../data_generation/data/train/tsp/tsp300_train_seed12345.pkl','../../../../../data_generation/data/train/tsp/tsp400_train_seed12345.pkl'],
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
from TSP_SML_Trainer import TSP_SML_Trainer as Trainer


def main():

    trainer = Trainer(env_params=env_params,
                    model_params=model_params,
                    run_params=run_params)
    trainer.run()


def parse_args(arg_str = None):
    global CUDA_DEVICE_NUM
    global run_params
    global logger_params

    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', type=int)
    parser.add_argument('--jump', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument("--problem_size", type=int)
    parser.add_argument("--eas_batch_size", type=int)


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

    if args.problem_size is not None:
        env_params['problem_size'] = args.problem_size
        env_params['pomo_size'] = args.problem_size



if __name__ == "__main__":
    parse_args()
    main()
