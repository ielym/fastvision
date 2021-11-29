import torch
import torch.nn as nn
import torch.nn.functional as F
from ..datasets.common import one_hot

class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pre, y_true, weights=None):
        num_classes = y_pre.size(-1)
        target_one_hot = one_hot(y_true, num_classes).float()

        predict_log_softmax = F.log_softmax(y_pre, dim=-1)

        loss = - torch.sum(target_one_hot * predict_log_softmax, dim=1)

        if isinstance(weights, type(None)):
            weights = torch.ones_like(loss)

        loss = loss * weights

        if self.reduction == 'mean':
            return torch.mean(loss)
        return torch.sum(loss)

class BiCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(BiCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pre, y_true, already_sigmoid=False, weights=None):
        num_classes = y_pre.size(-1)
        if num_classes > 1:
            target = one_hot(y_true, num_classes).float()
            target = target.view(-1, 1)
        else:
            target = y_true.float().view(-1, 1)
        y_pre = y_pre.view(-1, 1)

        if not already_sigmoid:
            loss = - target * torch.log(y_pre.sigmoid() + 1e-8) - (1 - target) * torch.log(1 - y_pre.sigmoid() + 1e-8)
        else:
            loss = - target * torch.log(y_pre + 1e-8) - (1 - target) * torch.log(1 - y_pre + 1e-8)

        loss = torch.sum(loss, dim=1)

        if isinstance(weights, type(None)):
            weights = torch.ones_like(loss)

        loss = loss * weights

        if self.reduction == 'mean':
            return torch.sum(loss) / y_pre.numel()
        return torch.sum(loss)