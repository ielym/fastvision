import numpy as np
import torch
import math

from .BOX import xywh2xyxy, xyxy2xywh

def cal_iou(box1, box2, mode='xyxy', eps=1e-7):
    if mode == 'xyxy':
        return xyxy_iou(box1, box2, eps)
    elif mode == 'xywh':
        return xywh_iou(box1, box2, eps)
    elif mode == 'wh':
        return wh_iou(box1, box2, eps)
    else:
        raise Exception('mode must be xyxy or xywh or wh')

def cal_iou_batch(box1, box2, mode='xyxy', eps=1e-7):
    if mode == 'xyxy':
        return xyxy_iou_batch(box1, box2, eps)
    elif mode == 'xywh':
        return xywh_iou_batch(box1, box2, eps)
    elif mode == 'wh':
        return wh_iou_batch(box1, box2, eps)
    else:
        raise Exception('mode must be xyxy or xywh or wh')

def xywh_iou(xywh1, xywh2, eps=1e-7):
    '''
    :param xywh1: Tensor[N, 4]
    :param xywh2: Tensor[N, 4]
    :param eps:
    :return: Tensor[N, 1]
    '''

    xyxy1 = xywh2xyxy(xywh1)
    xyxy2 = xywh2xyxy(xywh2)

    return xyxy_iou(xyxy1, xyxy2, eps)

def xywh_iou_batch(xywh1, xywh2, eps=1e-7):
    '''
    :param xywh1: Tensor[N, 4]
    :param xywh2: Tensor[N, 4]
    :param eps:
    :return: Tensor[N, 1]
    '''

    xyxy1 = xywh2xyxy(xywh1)
    xyxy2 = xywh2xyxy(xywh2)

    return xyxy_iou_batch(xyxy1, xyxy2, eps)

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
        area1 = (xyxy1[:, 2] - xyxy1[:, 0]) * (xyxy1[:, 3] - xyxy1[:, 1] + eps)
        area2 = (xyxy2[:, 2] - xyxy2[:, 0]) * (xyxy2[:, 3] - xyxy2[:, 1] + eps)

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

def xyxy_iou_batch(xyxy1, xyxy2, eps=1e-7):
    '''
    :param xyxy1: Tensor[N, 4]
    :param xyxy2: Tensor[M, 4]
    :param eps:
    :return: Tensor[N, M]
    '''
    def _numpy(xyxy1, xyxy2):
        area1 = (xyxy1[:, 2] - xyxy1[:, 0]) * (xyxy1[:, 3] - xyxy1[:, 1])
        area2 = (xyxy2[:, 2] - xyxy2[:, 0]) * (xyxy2[:, 3] - xyxy2[:, 1])

        inter = (np.minimum(xyxy1[: None, 2], xyxy2[:, 2]) - np.maximum(xyxy1[: None, 0], xyxy2[:, 0])).clip(0) * (np.minimum(xyxy1[: None, 3], xyxy2[:, 3]) - np.maximum(xyxy1[: None, 1], xyxy2[:, 1])).clip(0)

        union = area1[:, None] + area2 - inter + eps

        iou = inter / union
        return iou

    def _torch(xyxy1, xyxy2):
        area1 = (xyxy1[:, 2] - xyxy1[:, 0]) * (xyxy1[:, 3] - xyxy1[:, 1])
        area2 = (xyxy2[:, 2] - xyxy2[:, 0]) * (xyxy2[:, 3] - xyxy2[:, 1])

        try:
            inter = (torch.minimum(xyxy1[:, None, 2], xyxy2[:, 2]) - torch.maximum(xyxy1[:, None, 0], xyxy2[:, 0])).clamp(0) * (torch.minimum(xyxy1[:, None, 3], xyxy2[:, 3]) - torch.maximum(xyxy1[:, None, 1], xyxy2[:, 1])).clamp(0)
        except:
            inter = (torch.min(xyxy1[:, None, 2], xyxy2[:, 2]) - torch.max(xyxy1[:, None, 0], xyxy2[:, 0])).clamp(0) * (torch.min(xyxy1[:, None, 3], xyxy2[:, 3]) - torch.max(xyxy1[:, None, 1], xyxy2[:, 1])).clamp(0)

        union = area1[:, None] + area2 - inter + eps
        iou = inter / union

        return iou

    return _torch(xyxy1, xyxy2) if isinstance(xyxy1, torch.Tensor) else _numpy(xyxy1, xyxy2)

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

def GIOU(box1, box2, mode='xyxy', eps=1e-7):
    '''
    :param box1: Tensor[N, 4]
    :param box2:Tensor[N, 4]
    :param mode: xyxy xywh
    :param eps:
    :return: Tensor[N, 1]
    '''

    if mode == 'xywh':
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    def _numpy(box1, box2):

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        inter = (np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0])).clip(0) * (np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1])).clip(0)
        union = area1 + area2 - inter + eps
        iou = inter / union

        convex_width = np.maximum(box1[:, 2], box2[:, 2]) - np.minimum(box1[:, 0], box2[:, 0])
        convex_height = np.maximum(box1[:, 3], box2[:, 3]) - np.minimum(box1[:, 1], box2[:, 1])
        convex_area = convex_width * convex_height + eps

        return iou - (convex_area - union) / convex_area

    def _torch(box1, box2):

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        try:
            inter = (torch.minimum(box1[:, 2], box2[:, 2]) - torch.maximum(box1[:, 0], box2[:, 0])).clamp(0) * (torch.minimum(box1[:, 3], box2[:, 3]) - torch.maximum(box1[:, 1], box2[:, 1])).clamp(0)
        except:
            inter = (torch.min(box1[:, 2], box2[:, 2]) - torch.max(box1[:, 0], box2[:, 0])).clamp(0) * (torch.min(box1[:, 3], box2[:, 3]) - torch.max(box1[:, 1], box2[:, 1])).clamp(0)
        union = area1 + area2 - inter + eps
        iou = inter / union

        try:
            convex_width = torch.maximum(box1[:, 2], box2[:, 2]) - torch.minimum(box1[:, 0], box2[:, 0])
            convex_height = torch.maximum(box1[:, 3], box2[:, 3]) - torch.minimum(box1[:, 1], box2[:, 1])
        except:
            convex_width = torch.max(box1[:, 2], box2[:, 2]) - torch.min(box1[:, 0], box2[:, 0])
            convex_height = torch.max(box1[:, 3], box2[:, 3]) - torch.min(box1[:, 1], box2[:, 1])
        convex_area = convex_width * convex_height + eps

        return iou - (convex_area - union) / convex_area

    return _torch(box1, box2) if isinstance(box1, torch.Tensor) else _numpy(box1, box2)

def GIOU_batch(box1, box2, mode='xyxy', eps=1e-7):
    '''
    :param box1: Tensor[N, 4]
    :param box2:Tensor[M, 4]
    :param mode: xyxy xywh
    :param eps:
    :return: Tensor[N, M]
    '''

    if mode == 'xywh':
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    def _numpy(box1, box2):

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        inter = (np.minimum(box1[: None, 2], box2[:, 2]) - np.maximum(box1[: None, 0], box2[:, 0])).clip(0) * (np.minimum(box1[: None, 3], box2[:, 3]) - np.maximum(box1[: None, 1], box2[:, 1])).clip(0)
        union = area1[:, None] + area2 - inter + eps

        iou = inter / union
        convex_width = np.maximum(box1[:, None, 2], box2[:, 2]) - np.minimum(box1[:, None, 0], box2[:, 0])
        convex_height = np.maximum(box1[:, None, 3], box2[:, 3]) - np.minimum(box1[:, None, 1], box2[:, 1])
        convex_area = convex_width * convex_height + eps

        return iou - (convex_area - union) / convex_area

    def _torch(box1, box2):

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        try:
            inter = (torch.minimum(box1[:, None, 2], box2[:, 2]) - torch.maximum(box1[:, None, 0], box2[:, 0])).clamp(0) * (torch.minimum(box1[:, None, 3], box2[:, 3]) - torch.maximum(box1[:, None, 1], box2[:, 1])).clamp(0)
        except:
            inter = (torch.min(box1[:, None, 2], box2[:, 2]) - torch.max(box1[:, None, 0], box2[:, 0])).clamp(0) * (torch.min(box1[:, None, 3], box2[:, 3]) - torch.max(box1[:, None, 1], box2[:, 1])).clamp(0)
        union = area1[:, None] + area2 - inter + eps
        iou = inter / union

        try:
            convex_width = torch.maximum(box1[:, None, 2], box2[:, 2]) - torch.minimum(box1[:, None, 0], box2[:, 0])
            convex_height = torch.maximum(box1[:, None, 3], box2[:, 3]) - torch.minimum(box1[:, None, 1], box2[:, 1])
        except:
            convex_width = torch.max(box1[:, None, 2], box2[:, 2]) - torch.min(box1[:, None, 0], box2[:, 0])
            convex_height = torch.max(box1[:, None, 3], box2[:, 3]) - torch.min(box1[:, None, 1], box2[:, 1])

        convex_area = convex_width * convex_height + eps

        return iou + (convex_area - union) / convex_area

    return _torch(box1, box2) if isinstance(box1, torch.Tensor) else _numpy(box1, box2)

def DIOU(box1, box2, mode='xyxy', eps=1e-7):
    '''
    :param box1: Tensor[N, 4]
    :param box2:Tensor[N, 4]
    :param mode: xyxy xywh
    :param eps:
    :return: Tensor[N, 1]
    '''

    if mode == 'xywh':
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    iou = xyxy_iou(box1, box2, eps)

    def _numpy(box1, box2):
        convex_width = np.maximum(box1[:, 2], box2[:, 2]) - np.minimum(box1[:, 0], box2[:, 0])
        convex_height = np.maximum(box1[:, 3], box2[:, 3]) - np.minimum(box1[:, 1], box2[:, 1])
        convex_distance = convex_width ** 2 + convex_height ** 2 + eps

        center_x1 = (box1[:, 0] + box1[:, 2]) * 0.5
        center_y1 = (box1[:, 1] + box1[:, 3]) * 0.5

        center_x2 = (box2[:, 0] + box2[:, 2]) * 0.5
        center_y2 = (box2[:, 1] + box2[:, 3]) * 0.5

        center_distance = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

        return iou - center_distance.reshpae([-1, 1]) / convex_distance.reshpae([-1, 1])

    def _torch(box1, box2):

        try:
            convex_width = torch.maximum(box1[:, 2], box2[:, 2]) - torch.minimum(box1[:, 0], box2[:, 0])
            convex_height = torch.maximum(box1[:, 3], box2[:, 3]) - torch.minimum(box1[:, 1], box2[:, 1])
        except:
            convex_width = torch.max(box1[:, 2], box2[:, 2]) - torch.min(box1[:, 0], box2[:, 0])
            convex_height = torch.max(box1[:, 3], box2[:, 3]) - torch.min(box1[:, 1], box2[:, 1])
        convex_distance = convex_width ** 2 + convex_height ** 2 + eps

        center_x1 = (box1[:, 0] + box1[:, 2]) * 0.5
        center_y1 = (box1[:, 1] + box1[:, 3]) * 0.5

        center_x2 = (box2[:, 0] + box2[:, 2]) * 0.5
        center_y2 = (box2[:, 1] + box2[:, 3]) * 0.5
        center_distance = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

        return iou + center_distance.view(-1, 1) / convex_distance.view(-1, 1)

    return _torch(box1, box2) if isinstance(box1, torch.Tensor) else _numpy(box1, box2)

def DIOU_batch(box1, box2, mode='xyxy', eps=1e-7):
    '''
    :param box1: Tensor[N, 4]
    :param box2:Tensor[M, 4]
    :param mode: xyxy xywh
    :param eps:
    :return: Tensor[N, M]
    '''

    if mode == 'xywh':
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    iou = xyxy_iou_batch(box1, box2, eps)

    def _numpy(box1, box2):
        convex_width = np.maximum(box1[:, None, 2], box2[:, 2]) - np.minimum(box1[:, None, 0], box2[:, 0])
        convex_height = np.maximum(box1[:, None, 3], box2[:, 3]) - np.minimum(box1[:, None, 1], box2[:, 1])
        convex_distance = convex_width ** 2 + convex_height ** 2 + eps

        center_x1 = (box1[:, 0] + box1[:, 2]) * 0.5
        center_y1 = (box1[:, 1] + box1[:, 3]) * 0.5

        center_x2 = (box2[:, 0] + box2[:, 2]) * 0.5
        center_y2 = (box2[:, 1] + box2[:, 3]) * 0.5

        center_distance = (center_x1[:, None] - center_x2) ** 2 + (center_y1[:, None] - center_y2) ** 2

        return iou + center_distance / convex_distance

    def _torch(box1, box2):

        try:
            convex_width = torch.maximum(box1[:, None, 2], box2[:, 2]) - torch.minimum(box1[:, None, 0], box2[:, 0])
            convex_height = torch.maximum(box1[:, None, 3], box2[:, 3]) - torch.minimum(box1[:, None, 1], box2[:, 1])
        except:
            convex_width = torch.max(box1[:, None, 2], box2[:, 2]) - torch.min(box1[:, None, 0], box2[:, 0])
            convex_height = torch.max(box1[:, None, 3], box2[:, 3]) - torch.min(box1[:, None, 1], box2[:, 1])
        convex_distance = convex_width ** 2 + convex_height ** 2 + eps

        center_x1 = (box1[:, 0] + box1[:, 2]) * 0.5
        center_y1 = (box1[:, 1] + box1[:, 3]) * 0.5

        center_x2 = (box2[:, 0] + box2[:, 2]) * 0.5
        center_y2 = (box2[:, 1] + box2[:, 3]) * 0.5

        center_distance = (center_x1[:, None] - center_x2) ** 2 + (center_y1[:, None] - center_y2) ** 2

        return iou + center_distance / convex_distance

    return _torch(box1, box2) if isinstance(box1, torch.Tensor) else _numpy(box1, box2)

def CIOU(box1, box2, mode='xyxy', eps=1e-7):
    '''
    :param box1: Tensor[N, 4]
    :param box2:Tensor[N, 4]
    :param mode: xyxy xywh
    :param eps:
    :return: Tensor[N, 1]
    '''

    if mode == 'xywh':
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    iou = xyxy_iou(box1, box2, eps)
    diou = DIOU(box1, box2, 'xyxy', eps)


    def _numpy(box1, box2):
        width1 = box1[:, 2] - box1[:, 0]
        height1 = box1[:, 3] - box1[:, 1]

        width2 = box2[:, 2] - box2[:, 0]
        height2 = box2[:, 3] - box2[:, 1]

        v = (4 / math.pi ** 2) * np.power(np.arctan(width2 / (height2 + eps)) - np.arctan((width1 / (height1 + eps))), 2)
        v = v.reshape([-1, 1])
        alpha = v / (v - iou + (1 + eps))
        return diou - alpha * v

    def _torch(box1, box2):

        width1 = box1[:, 2] - box1[:, 0]
        height1 = box1[:, 3] - box1[:, 1]

        width2 = box2[:, 2] - box2[:, 0]
        height2 = box2[:, 3] - box2[:, 1]

        v = (4 / math.pi ** 2) * torch.pow(torch.atan(width2 / (height2 + eps)) - torch.atan((width1 / (height1 + eps))), 2)
        v = v.view(-1, 1)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return diou - alpha * v

    return _torch(box1, box2) if isinstance(box1, torch.Tensor) else _numpy(box1, box2)

def CIOU_batch(box1, box2, mode='xyxy', eps=1e-7):
    '''
    :param box1: Tensor[N, 4]
    :param box2:Tensor[M, 4]
    :param mode: xyxy xywh
    :param eps:
    :return: Tensor[N, M]
    '''

    if mode == 'xywh':
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    iou = xyxy_iou_batch(box1, box2, eps)
    diou = DIOU_batch(box1, box2, 'xyxy', eps)

    def _numpy(box1, box2):
        width1 = box1[:, 2] - box1[:, 0]
        height1 = box1[:, 3] - box1[:, 1]

        width2 = box2[:, 2] - box2[:, 0]
        height2 = box2[:, 3] - box2[:, 1]

        v = (4 / math.pi ** 2) * np.power(np.arctan((width1 / (height1 + eps)))[:, None] - np.arctan(width2 / (height2 + eps)), 2)
        alpha = v / (v - iou + (1 + eps))
        return diou - alpha * v

    def _torch(box1, box2):

        width1 = box1[:, 2] - box1[:, 0]
        height1 = box1[:, 3] - box1[:, 1]

        width2 = box2[:, 2] - box2[:, 0]
        height2 = box2[:, 3] - box2[:, 1]

        v = (4 / math.pi ** 2) * torch.pow(torch.arctan((width1 / (height1 + eps)))[:, None] - torch.arctan(width2 / (height2 + eps)), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return diou - alpha * v

    return _torch(box1, box2) if isinstance(box1, torch.Tensor) else _numpy(box1, box2)