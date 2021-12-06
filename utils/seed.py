import torch.backends.cudnn as cudnn
import random
import numpy as np
import os
import torch

def set_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    print(f'RANDOM SEED : {seed}')
