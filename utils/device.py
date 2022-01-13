import os
import torch

def set_device(devices):

    if len(devices) == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        device_str = ','.join([str(device) for device in devices])
        os.environ['CUDA_VISIBLE_DEVICES'] = device_str

        shards = {}
        for shard_idx, device in enumerate(devices):
            shards[shard_idx] = device
        os.environ['FASTVISON_SHARDS'] = str(shards)


    cuda = len(devices) and torch.cuda.is_available()

    device = torch.device('cuda' if cuda else 'cpu')

    device_msg = f'Device : {device.type} \t'
    if device.type == 'cuda':
        device_msg += f"CUDA_VISIBLE_DEVICES : {os.environ['CUDA_VISIBLE_DEVICES']}\t"
        device_msg += f"shards : { os.environ['FASTVISON_SHARDS']}"
    print(device_msg)

    return device