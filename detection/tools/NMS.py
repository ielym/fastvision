import torch
import torchvision
from ..tools import xywh2xyxy

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):

    num_classes = prediction.shape[2] - 5

    conf_mask = prediction[..., 4] > conf_thres  # candidates
    prediction = prediction[conf_mask]

    if not prediction.size(0):
        return [], [], []

    prediction[:, 5:] *= prediction[:, 4:5]
    boxes = xywh2xyxy(prediction[:, :4])

    scores, categories = torch.max(prediction[:, 5:], dim=1)

    nms_mask = torchvision.ops.nms(boxes, scores, iou_thres)
    scores = scores[nms_mask]
    categories = categories[nms_mask]
    boxes = boxes[nms_mask, :]

    return scores[:max_det], categories[:max_det], boxes[:max_det]