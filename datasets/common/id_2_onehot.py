import numpy as np
import torch

def one_hot(y, num_classes):

    def _numpy(y, num_classes):
        onehot_label = np.eye(num_classes, num_classes)[y]
        return onehot_label

    def _torch(y, num_classes):
        flatten_y = y.view(-1, 1).long()
        num_samples = flatten_y.size(0)

        onehot_label = torch.zeros((num_samples, num_classes)).to(y).scatter_(1, flatten_y, 1)
        return onehot_label

    return _torch(y, num_classes) if isinstance(y, torch.Tensor) else _numpy(y, num_classes)
