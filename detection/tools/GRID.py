import numpy as np
import torch

def grid(height, width, mode='xy', dtype='torch'):

    def _numpy(height, width, mode):
        ys = np.arange(0, height)
        xs = np.arange(0, width)

        offset_x, offset_y = np.meshgrid(xs, ys)
        offset_yx = np.stack([offset_x, offset_y]).transpose([1, 2, 0])

        if mode == 'xy':
            offset_xy = offset_yx.transpose([1, 0, 2])
            return offset_xy
        return offset_yx

    def _torch(height, width, mode):
        ys = torch.arange(0, height)
        xs = torch.arange(0, width)

        offset_x, offset_y = torch.meshgrid(xs, ys)
        offset_yx = torch.stack([offset_x, offset_y]).permute(1, 2, 0)

        if mode == 'xy':
            offset_xy = offset_yx.permute(1, 0, 2)
            return offset_xy

        return offset_yx

    return _torch(height, width, mode) if (dtype == 'torch') else _numpy(height, width, mode)