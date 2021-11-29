import numpy as np
import torch
from .BOX import xywh2xyxy

def cal_iou(box1, box2, mode='xyxy', eps=1e-7):
    if mode == 'xyxy':
        return xyxy_iou(box1, box2, eps)
    else:
        return xywh_iou(box1, box2, eps)

def xywh_iou(xywh1, xywh2, eps=1e-7):
    '''
    :param xywh1: Tensor[N, 4]
    :param xywh2: Tensor[N, 4]
    :param eps:
    :return: Tensor[N, 1]
    '''

    xyxy1 = xywh2xyxy(xywh1)
    xyxy2 = xywh2xyxy(xywh2)

    def _numpy(xyxy1, xyxy2):

        area1 = (xyxy1[:, 2] - xyxy1[:, 0]) * (xyxy1[:, 3] - xyxy1[:, 1])
        area2 = (xyxy2[:, 2] - xyxy2[:, 0]) * (xyxy2[:, 3] - xyxy2[:, 1])

        inter = (np.minimum(xyxy1[:, 2], xyxy2[:, 2]) - np.maximum(xyxy1[:, 0], xyxy2[:, 0])).clip(0) * (np.minimum(xyxy1[:, 3], xyxy2[:, 3]) - np.maximum(xyxy1[:, 1], xyxy2[:, 1])).clip(0)

        union = area1 + area2 - inter + eps

        iou = inter / union
        return iou.reshape([-1, 1])

    def _torch(xyxy1, xyxy2):
        area1 = (xyxy1[:, 2] - xyxy1[:, 0]) * (xyxy1[:, 3] - xyxy1[:, 1])
        area2 = (xyxy2[:, 2] - xyxy2[:, 0]) * (xyxy2[:, 3] - xyxy2[:, 1])

        try:
            inter = (torch.minimum(xyxy1[:, 2], xyxy2[:, 2]) - torch.maximum(xyxy1[:, 0], xyxy2[:, 0])).clamp(0) * (torch.minimum(xyxy1[:, 3], xyxy2[:, 3]) - torch.maximum(xyxy1[:, 1], xyxy2[:, 1])).clamp(0)
        except:
            inter = (torch.min(xyxy1[:, 2], xyxy2[:, 2]) - torch.max(xyxy1[:, 0], xyxy2[:, 0])).clamp(0) * (torch.min(xyxy1[:, 3], xyxy2[:, 3]) - torch.max(xyxy1[:, 1], xyxy2[:, 1])).clamp(0)

        union = area1 + area2 - inter + eps

        iou = inter / union
        return iou.reshape([-1, 1])

    return _torch(xyxy1, xyxy2) if isinstance(xyxy1, torch.Tensor) else _numpy(xyxy1, xyxy2)

def xyxy_iou(xyxy1, xyxy2, eps=1e-7):
    '''
    :param wh1: Tensor[N, 4]
    :param wh2: Tensor[N, 4]
    :param eps:
    :return: Tensor[N, 1]
    '''

    def _numpy(xyxy1, xyxy2):

        area1 = (xyxy1[:, 2] - xyxy1[:, 0]) * (xyxy1[:, 3] - xyxy1[:, 1])
        area2 = (xyxy2[:, 2] - xyxy2[:, 0]) * (xyxy2[:, 3] - xyxy2[:, 1])

        inter = (np.minimum(xyxy1[:, 2], xyxy2[:, 2]) - np.maximum(xyxy1[:, 0], xyxy2[:, 0])).clip(0) * (np.minimum(xyxy1[:, 3], xyxy2[:, 3]) - np.maximum(xyxy1[:, 1], xyxy2[:, 1])).clip(0)

        union = area1 + area2 - inter + eps

        iou = inter / union
        return iou.reshape([-1, 1])

    def _torch(xyxy1, xyxy2):
        area1 = (xyxy1[:, 2] - xyxy1[:, 0]) * (xyxy1[:, 3] - xyxy1[:, 1])
        area2 = (xyxy2[:, 2] - xyxy2[:, 0]) * (xyxy2[:, 3] - xyxy2[:, 1])

        try:
            inter = (torch.minimum(xyxy1[:, 2], xyxy2[:, 2]) - torch.maximum(xyxy1[:, 0], xyxy2[:, 0])).clamp(0) * (torch.minimum(xyxy1[:, 3], xyxy2[:, 3]) - torch.maximum(xyxy1[:, 1], xyxy2[:, 1])).clamp(0)
        except:
            inter = (torch.min(xyxy1[:, 2], xyxy2[:, 2]) - torch.max(xyxy1[:, 0], xyxy2[:, 0])).clamp(0) * (torch.min(xyxy1[:, 3], xyxy2[:, 3]) - torch.max(xyxy1[:, 1], xyxy2[:, 1])).clamp(0)

        union = area1 + area2 - inter + eps

        iou = inter / union
        return iou.reshape([-1, 1])

    return _torch(xyxy1, xyxy2) if isinstance(xyxy1, torch.Tensor) else _numpy(xyxy1, xyxy2)


def wh_iou(wh1, wh2, eps=1e-7):
    '''
    :param wh1: Tensor[N, 2]
    :param wh2: Tensor[N, 2]
    :param eps:
    :return: Tensor[N, 1]
    '''
    def _numpy(wh1, wh2):

        area1 = wh1[:, 0] * wh1[:, 1]
        area2 = wh2[:, 0] * wh2[:, 1]

        inter = np.minimum(wh1[:, 0], wh2[:, 0]) * np.minimum(wh1[:, 1], wh2[:, 1])

        union = area1 + area2 - inter + eps

        iou = inter / union
        return iou.reshape([-1, 1])

    def _torch(wh1, wh2):
        area1 = wh1[:, 0] * wh1[:, 1]
        area2 = wh2[:, 0] * wh2[:, 1]

        try:
            inter = torch.minimum(wh1[:, 0], wh2[:, 0]) * torch.minimum(wh1[:, 1], wh2[:, 1])
        except:
            inter = torch.min(wh1[:, 0], wh2[:, 0]) * torch.min(wh1[:, 1], wh2[:, 1])

        union = area1 + area2 - inter + eps

        iou = inter / union
        return iou.reshape([-1, 1])

    return _torch(wh1, wh2) if isinstance(wh1, torch.Tensor) else _numpy(wh1, wh2)


def wh_iou_batch(wh1, wh2, eps=1e-7):
    '''
    :param wh1: Tensor[N, 2]
    :param wh2: Tensor[M, 2]
    :param eps:
    :return: Tensor[N, M]
    '''
    def _numpy(wh1, wh2):

        area1 = wh1[:, 0] * wh1[:, 1]
        area2 = wh2[:, 0] * wh2[:, 1]

        inter = np.minimum(wh1[:, None, 0], wh2[:, 0]) * np.minimum(wh1[:, None, 1], wh2[:, 1])

        union = area1[:, None] + area2 - inter + eps

        iou = inter / union
        return iou

    def _torch(wh1, wh2):
        area1 = wh1[:, 0] * wh1[:, 1]
        area2 = wh2[:, 0] * wh2[:, 1]

        try:
            inter = torch.minimum(wh1[:, None, 0], wh2[:, 0]) * torch.minimum(wh1[:, None, 1], wh2[:, 1])
        except:
            inter = torch.min(wh1[:, None, 0], wh2[:, 0]) * torch.min(wh1[:, None, 1], wh2[:, 1])

        union = area1[:, None] + area2 - inter + eps

        iou = inter / union
        return iou

    return _torch(wh1, wh2) if isinstance(wh1, torch.Tensor) else _numpy(wh1, wh2)