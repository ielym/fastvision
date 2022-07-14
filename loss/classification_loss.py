import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..datasets.common import one_hot

class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pre, y_true, weights=None):
        '''
        :param y_pre: [N, 1000]
        :param y_true: [N, 1]
        :param weights:
        :return:
        '''
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

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5),
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss