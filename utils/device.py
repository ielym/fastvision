import os
import torch

def set_device(device_str):
    device_str = device_str.replace(' ', '')

    if device_str == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_str

    cuda = device_str != 'cpu' and torch.cuda.is_available()

    return torch.device('cuda' if cuda else 'cpu')