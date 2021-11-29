import numpy as np

def offset(height, width, mode='xy'):
    ys = np.arange(0, height)
    xs = np.arange(0, width)

    offset_x, offset_y = np.meshgrid(xs, ys)
    offset_yx = np.stack([offset_x, offset_y]).transpose([1, 2, 0])

    if mode == 'xy':
        offset_xy = offset_yx.transpose([1, 0, 2])
        return offset_xy
    return offset_yx