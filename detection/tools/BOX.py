import numpy as np
import torch

def xywh2xyxy(xywh):
    xyxy = xywh.clone() if isinstance(xywh, torch.Tensor) else np.copy(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # top left x
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # top left y
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # bottom right x
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # bottom right y
    return xyxy

def xyxy2xywh(xyxy):
    xywh = xyxy.clone() if isinstance(xyxy, torch.Tensor) else np.copy(xyxy)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh

def xyxy2xywhn(xyxy, heigth, width):
    xywhn = xyxy.clone() if isinstance(xyxy, torch.Tensor) else np.copy(xyxy)
    xywhn[:, 0] = ((xyxy[:, 0] + xyxy[:, 2]) / 2) / width
    xywhn[:, 1] = ((xyxy[:, 1] + xyxy[:, 3]) / 2) / heigth
    xywhn[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) / width
    xywhn[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) / heigth
    return xywhn

