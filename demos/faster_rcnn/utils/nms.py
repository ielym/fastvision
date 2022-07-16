import torch
import torchvision
from .box import xywh2xyxy

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):

    '''
    :param prediction: [x, y, w, h, categories, scores], within ori image size
    :param conf_thres:
    :param num_classes:
    :param max_det: maximum number of detections per image
    :return: torch.Size([17, 6]) [xmin, ymin, xmax, ymax,score,category_idx]
    '''

    max_wh = 4096
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    prediction = prediction[prediction[:, 5] > conf_thres]

    if len(prediction) == 0:
        return torch.zeros((0, 6), device=prediction.device)

    if prediction.size(0) > max_nms:
        prediction = prediction[prediction[:, 0].argsort(descending=True)[:max_nms]]

    categories = prediction[..., 4:5]
    scores = prediction[..., 5:6]

    # To increase the coordinate gap for different categories
    coordinate_gap = categories * max_wh
    boxes = prediction[..., :4] + coordinate_gap

    nms_mask = torchvision.ops.nms(boxes, scores.view(-1), iou_thres)  # NMS
    if nms_mask.size(0) > max_det:  # limit detections
        nms_mask = nms_mask[:max_det]

    output = prediction[nms_mask]

    return output

def non_max_suppression_batch(prediction_batch, conf_thres=0.25, iou_thres=0.45, max_det=300):

    '''
    :param prediction: [torch.Size([25200, 85]), torch.Size([25200, 85]), ...] from different images : [x,y,w,h,conf, num_classes * scores]
    :param conf_thres:
    :param num_classes:
    :param max_det: maximum number of detections per image
    :return: [torch.Size([17, 6]), torch.Size([4, 6]), torch.Size([2, 6]), ...] [xmin, ymin, xmax, ymax,score,category_idx]

    non_max_suppression_batch([predict1, predict2, ...], conf_thres=0.01, iou_thres=0.4, max_det=300)
    '''

    max_wh = 4096
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    output = [torch.zeros((0, 6), device="cpu")] * len(prediction_batch)

    for image_idx, prediction in enumerate(prediction_batch):

        prediction = prediction[prediction[:, 4] > conf_thres]

        prediction[:, 5:] *= prediction[:, 4:5]  # conf = obj_conf * cls_conf

        boxes = xywh2xyxy(prediction[:, :4]) # torch.Size([110, 4])

        scores, categories = prediction[:, 5:].max(1, keepdim=True) # torch.Size([110, 1]) torch.Size([110, 1])

        prediction = torch.cat((boxes, scores, categories.float()), 1)[scores.view(-1) > conf_thres] # torch.Size([110, 6]) [x,y,x,y,score,category_idx]

        if prediction.size(0) > max_nms:
            # sort by confidence
            prediction = prediction[prediction[:, 4].argsort(descending=True)[:max_nms]]

        # To increase the coordinate gap for different categories
        coordinate_gap = prediction[:, 5:6] * max_wh
        boxes, scores = prediction[:, :4] + coordinate_gap, prediction[:, 4]

        nms_mask = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if nms_mask.size(0) > max_det:  # limit detections
            nms_mask = nms_mask[:max_det]

        output[image_idx] = prediction[nms_mask].detach().cpu()

    return output
