import torch
from collections import OrderedDict
import os

def LoadFromSingle(model, weights, strict=False):
    try:
        weights_dict = torch.load(weights)
        # torch.save(weights_dict, "yolov3-nonzip.pth", _use_new_zipfile_serialization=False)
    except:
        head, tail = os.path.split(weights)

        tail_split = tail.split('.')
        tail = ''.join(tail_split[:-1]) + '-nonzip.' + tail_split[-1]
        weights = os.path.join(head, tail)
        weights_dict = torch.load(weights)

    model_keys = model.state_dict().keys()

    useful_keys = OrderedDict()

    for k, v in weights_dict.items():
        if k in model_keys and v.size() == model.state_dict()[k].size():
            useful_keys[k] = v
    model.load_state_dict(useful_keys, strict=strict)

    return model


def LoadFromParrel(model, weights, strict=False):
    try:
        weights_dict = torch.load(weights)
    except:
        head, tail = os.path.split(weights)

        tail_split = tail.split('.')
        tail = ''.join(tail_split[:-1]) + '-nonzip.' + tail_split[-1]
        weights = os.path.join(head, tail)
        weights_dict = torch.load(weights)

    model_keys = model.state_dict().keys()

    useful_keys = OrderedDict()

    for k, v in weights_dict.items():
        if k[7:] in model_keys and v.size() == model.state_dict()[k].size():
            useful_keys[k[7:]] = v
    model.load_state_dict(useful_keys, strict=strict)

    return model