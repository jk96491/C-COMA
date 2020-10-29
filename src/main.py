import numpy as np
import torch as th
from utils.logging import get_logger
import random
from run import run
import config_util as cu

'''
algorithm 설정 가이드(config/algs 경로의 파일이름 그대로)
만일 QMIX 를 실행하고 싶다면 -> 'QMIX'
만일 C-COMA 를 실행하고 싶다면 -> 'COMA'
'''

if __name__ == '__main__':
    logger = get_logger()
    algorithm = 'C-COMA'
    minigame = 'Dynamic_env_Training'

    config = cu.config_copy(cu.get_config(algorithm, minigame))

    random_Seed = random.randrange(0, 16546)

    np.random.seed(random_Seed)
    th.manual_seed(random_Seed)
    config['env_args']['seed'] = random_Seed

    run(config, logger)