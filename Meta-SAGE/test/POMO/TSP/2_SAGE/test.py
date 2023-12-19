##########################################################################################
# Machine Environment Config
import torch
USE_CUDA = True
CUDA_DEVICE_NUM = 0
torch.manual_seed(1234)

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# parameters

#########################
# Parameters - Base
#########################

env_params = {
    'problem_size':500,
    'pomo_size': 500,}

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
# Parameters - Meta-SAGE
#########################

run_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '../1_pretrained_model',  # directory path of pre-trained model and log files saved.
        'epoch': 1900,  # epoch version of pre-trained model to laod.
    },
    'test_data_load': {
        'enable': True,
        'filename': ['../../../../data_generation/data/test/tsp/tsp200_test_small_seed1235.pkl'],
        'index_begin': 0
    },
    'num_episodes': 128,
    'solution_max_length': 1000,  # for buffer length storing solution

    # SAGE Params
    'lr': 0.0032,
    'lambda': 0.005,
    'beta' : 0.005,
    'sage_num_iter': 201,
    'sage_batch_size': 10,

    #Use Bias
    'use_bias' : True,

    #augmentation factor
    'aug' : 1
}

#########################
# Parameters - Log
#########################

logger_params = {
    'log_file': {
        'desc': 'tsp_meta_sage',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

import logging
import argparse
from utils.utils import create_logger
from TSPTester import TSPTester as Tester

def main():
    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                   model_params=model_params,
                   run_params=run_params)

    tester.run()

def _print_config():
    logger = logging.getLogger('root')
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

def parse_args(arg_str = None):
    global CUDA_DEVICE_NUM
    global run_params
    global logger_params

    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', type=int)
    parser.add_argument('--jump', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument("--problem_size", type=int)
    parser.add_argument("--sage_batch_size", type=int)
    parser.add_argument("--aug", type=int)
    parser.add_argument("--iter", type=int)
    parser.add_argument("--use_bias", action="store_true")

    if arg_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=arg_str.split())

    if args.ep is not None:
        run_params['num_episodes'] = args.ep

    if args.jump is not None:
        num_episodes = run_params['num_episodes']
        run_params['test_data_load']['index_begin'] += args.jump #* num_episodes

    if args.gpu is not None:
        CUDA_DEVICE_NUM = args.gpu
        run_params['cuda_device_num'] = args.gpu

    if args.sage_batch_size is not None:
        run_params['sage_batch_size'] = args.sage_batch_size

    if args.problem_size is not None:
        env_params['problem_size'] = args.problem_size
        env_params['pomo_size'] = args.problem_size
        run_params['solution_max_length'] = args.problem_size

    if args.aug is not None:
        run_params['aug'] = args.aug
    if args.iter is not None:
        run_params['sage_num_iter'] = args.iter

    run_params['use_bias'] = args.use_bias

    if env_params['problem_size'] == 200:
        run_params['test_data_load']['filename'] = '../../../../data_generation/data/test/tsp/tsp200_test_small_seed1235.pkl'
    elif env_params['problem_size'] == 500:
        run_params['test_data_load']['filename'] = '../../../../data_generation/data/test/tsp/tsp500_main_table_seed1234.pkl'
    elif env_params['problem_size'] == 1000:
        run_params['test_data_load']['filename'] = '../../../../data_generation/data/test/tsp//tsp1000_main_table_seed1234.pkl'
    else:
        raise ValueError("problem size must be 200 or 500 or 1000")
if __name__ == "__main__":
    parse_args()
    main()