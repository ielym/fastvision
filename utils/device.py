import os
import torch

def set_device(device_str):
    device_str = device_str.replace(' ', '')

    if device_str == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_str

    cuda = device_str != 'cpu' and torch.cuda.is_available()

    device = torch.device('cuda' if cuda else 'cpu')

    device_msg = f'Device : {device.type}'
    if device.type == 'cuda':
        device_msg += f"CUDA_VISIBLE_DEVICES : {os.environ['CUDA_VISIBLE_DEVICES']}"
    print(device_msg)

    return device