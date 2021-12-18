import torch
import torch.nn as nn
from fastvision.detection.tools import cal_iou, GIOU, DIOU, CIOU

class IOULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(IOULoss, self).__init__()
        self.reduction=reduction

    def forward(self, y_pre, y_true, weights=None, mode='xyxy'):
        '''
        :param y_pre: Tensor[N, 4]
        :param y_true: Tensor[N, 4]
        :param weights: Tensor[N, 1]
        :return:
        '''

        iou = cal_iou(y_pre, y_true, mode=mode)

        loss = 1 - iou

        if isinstance(weights, type(None)):
            weights = torch.ones_like(loss)

        loss = loss * weights

        if self.reduction == 'mean':
            return torch.mean(loss)
        return torch.sum(loss)

class GIOULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(GIOULoss, self).__init__()
        self.reduction=reduction

    def forward(self, y_pre, y_true, weights=None, mode='xyxy'):
        '''
        :param y_pre: Tensor[N, 4]
        :param y_true: Tensor[N, 4]
        :param weights: Tensor[N, 1]
        :return:
        '''

        iou = GIOU(y_pre, y_true, mode=mode)

        loss = 1 - iou

        if isinstance(weights, type(None)):
            weights = torch.ones_like(loss)

        loss = loss * weights

        if self.reduction == 'mean':
            return torch.mean(loss)
        return torch.sum(loss)

class DIOULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(DIOULoss, self).__init__()
        self.reduction=reduction

    def forward(self, y_pre, y_true, weights=None, mode='xyxy'):
        '''
        :param y_pre: Tensor[N, 4]
        :param y_true: Tensor[N, 4]
        :param weights: Tensor[N, 1]
        :return:
        '''

        iou = DIOU(y_pre, y_true, mode=mode)

        loss = 1 - iou

        if isinstance(weights, type(None)):
            weights = torch.ones_like(loss)

        loss = loss * weights

        if self.reduction == 'mean':
            return torch.mean(loss)
        return torch.sum(loss)

class CIOULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CIOULoss, self).__init__()
        self.reduction=reduction

    def forward(self, y_pre, y_true, weights=None, mode='xyxy'):
        '''
        :param y_pre: Tensor[N, 4]
        :param y_true: Tensor[N, 4]
        :param weights: Tensor[N, 1]
        :return:
        '''

        iou = CIOU(y_pre, y_true, mode=mode)

        loss = 1 - iou

        if isinstance(weights, type(None)):
            weights = torch.ones_like(loss)

        loss = loss * weights

        if self.reduction == 'mean':
            return torch.mean(loss)
        return torch.sum(loss)