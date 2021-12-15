import torch

class Accuracy():

    def __init__(self):
        ...

    @torch.no_grad()
    def __call__(self, y_pred, y_true):
        '''
        :param y_pred: [N, num_classes]
        :param y_true: [N, 1]
        :return:
        '''
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = y_true
        batch_size = y_pred.size(0)
        correct = y_pred.eq(y_true.expand_as(y_pred)).float().sum(0, keepdim=True)
        acc = correct / batch_size
        return acc
