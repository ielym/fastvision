import torch
from collections import OrderedDict
import os
import torch.nn as nn
from copy import deepcopy
from datetime import datetime

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def SqueezeModel(model, params, squeeze:bool):
    '''
    :param model:
    :param params: 'all' or list : ['neck', 'head', 'classifier']
    :param squeeze:
    :return:
    '''
    if params == 'all':
        for name, value in model.named_parameters():
            value.requires_grad = squeeze
    else:
        for name, value in model.named_parameters():
            for param in params:
                if param in name:
                    value.requires_grad = squeeze
    return model

def LoadStatedict(model, weights, device, strict=False):
    try:
        weights_dict = torch.load(weights, map_location=device)
        # torch.save(weights_dict, "yolov3-nonzip.pth", _use_new_zipfile_serialization=False)
    except:
        head, tail = os.path.split(weights)

        tail_split = tail.split('.')
        tail = ''.join(tail_split[:-1]) + '-nonzip.' + tail_split[-1]
        weights = os.path.join(head, tail)
        weights_dict = torch.load(weights, map_location=device)

    if 'model' in weights_dict.keys():
        weights_dict = weights_dict['model']

    model_keys = model.state_dict().keys()

    useful_keys = OrderedDict()
    no_use_keys = []
    for k, v in weights_dict.items():
        if k in model_keys and v.size() == model.state_dict()[k].size():
            useful_keys[k] = v
        else:
            no_use_keys.append(k)

    model.load_state_dict(useful_keys, strict=strict)

    print('Load state_dict not load keys : ',  no_use_keys)

    return model

def LoadFromParrel(model, weights, device, strict=False):
    try:
        weights_dict = torch.load(weights, map_location=device)

    except:
        head, tail = os.path.split(weights)

        tail_split = tail.split('.')
        tail = ''.join(tail_split[:-1]) + '-nonzip.' + tail_split[-1]
        weights = os.path.join(head, tail)
        weights_dict = torch.load(weights, map_location=device)

    if 'model' in weights_dict.keys():
        weights_dict = weights_dict['model']

    model_keys = model.state_dict().keys()

    useful_keys = OrderedDict()
    no_use_keys = []
    for k, v in weights_dict.items():
        if k[7:] in model_keys and v.size() == model.state_dict()[k].size():
            useful_keys[k[7:]] = v
        else:
            no_use_keys.append(k)

    model.load_state_dict(useful_keys, strict=strict)

    print('Load state_dict not load keys : ',  no_use_keys)

    return model

def SaveModel(ckpt, filename, weights_only=True):
    date = datetime.now().isoformat()
    ckpt['date'] = date

    model = ckpt['model']
    ckpt['model'] = deepcopy(model.module if is_parallel(model) else model)

    if weights_only:
        ckpt['model'] = ckpt['model'].state_dict()

    torch.save(ckpt, filename)




