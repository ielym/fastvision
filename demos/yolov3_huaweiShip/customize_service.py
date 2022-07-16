# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torchvision
import logging
import time
import pandas as pd
import yaml


import math
import platform
import warnings
from copy import copy

import cv2


import requests
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp

# ======================================================================================================================
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Resize,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop, PadIfNeeded,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize, RandomCrop
)
from albumentations.pytorch import ToTensorV2
# ======================================================================================================================

#from utils.datasets import LoadImages
from model_service.pytorch_model_service import PTServingBaseService

# ======================================================================================================================
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def grid(height, width, mode='xy'):
    ys = torch.arange(0, height)
    xs = torch.arange(0, width)

    offset_x, offset_y = torch.meshgrid(xs, ys)
    offset_yx = torch.stack([offset_x, offset_y]).permute(1, 2, 0)

    if mode == 'xy':
        offset_xy = offset_yx.permute(1, 0, 2)
        return offset_xy

    return offset_yx

def anchor_fn(device):

    anchors_small = torch.tensor([[462, 202], [340, 431], [522, 364]]) / 32
    anchors_medium = torch.tensor([[113, 270], [230, 134], [197, 397]]) / 16
    anchors_large = torch.tensor([[15, 24], [46, 63], [79, 134]]) / 8

    return ( anchors_small.to(device=device), anchors_medium.to(device=device), anchors_large.to(device=device) )

def Padding(image, size, fill_value=128):
    '''
    :param image: np.ndarray [H, W, c] RGB
    :param label: np.ndarray [n, 4] xyxy, without normalization
    :param size: int or tuple or list. if int size=(size, size) : (h, w)
    :return:
    '''
    if isinstance(size, int):
        size = (size, size)

    ori_height, ori_width, channel = image.shape

    if size[0] < ori_height:
        raise Exception("Padding's target height can not less than image's height")

    if size[1] < ori_width:
        raise Exception("Padding's target width can not less than image's width")

    padding_top = int((size[0] - ori_height) // 2)
    padding_left = int((size[1] - ori_width) // 2)

    padding_image = np.zeros((size[0], size[1], channel), dtype=image.dtype)
    padding_image.fill(fill_value)
    padding_image[padding_top:padding_top+ori_height, padding_left:padding_left+ori_width, :] = image

    return padding_image, padding_top, padding_left

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ======================================================================================================================

def select_device(device='', batch_size=0, newline=True):

    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    cuda = not cpu and torch.cuda.is_available()
    return torch.device('cuda:0' if cuda else 'cpu')

def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
    
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
    
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = prediction[xi]
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x_new = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x_new = torch.cat((x_new, v), 0)

        # If none remain process next image
        if not x_new.shape[0]:
            continue

        x = x_new
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            #LOGGER.warning(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

    
def get_model(model_path, **kwargs):
    # ======================================================================================================================
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    from models.yolov3 import YoloV3

    model = YoloV3(anchors=kwargs['anchors'], num_classes=kwargs['num_classes'])
    # weights = torch.load(os.path.join(model_path, 'best.pth'))
    weights = torch.load(model_path, map_location=kwargs['device'])
    # single_dict = {}
    # for k, v in weights.items():
    #     single_dict[k[7:]] = v
        # single_dict[k] = v
    model.load_state_dict(weights, True)

    model = model.to(device=kwargs['device'])
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ======================================================================================================================

    model.eval()
    return model


class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(PTVisionService, self).__init__(model_name, model_path)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # ======================================================================================================================
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.stride = [32, 16, 8]
        self.img_size = 608
        # conf_thres = 0.001 iou_thres = 0.6 score = 0.4026
        # conf_thres = 0.25 iou_thres = 0.60 score = 0.3338
        # conf_thres = 0.25 iou_thres = 0.45 score = 0.3335
        # conf_thres = 0.55 iou_thres = 0.25 score = 0.2743
        self.conf_thres = 0.001
        self.iou_thres = 0.6
        # self.conf_thres = 0.25
        # self.iou_thres = 0.6
        self.label = [0,1,2,3,4,5,6,7,8,9]
        self.num_classes = len(self.label)
        self.min_size = 5
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ======================================================================================================================


        # ======================================================================================================================
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.anchors = anchor_fn(device=self.device)
        self.model = get_model(model_path, anchors=self.anchors, num_classes=self.num_classes, device=self.device)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ======================================================================================================================

        self.input_image_key = 'images'
        self.data = {}
        self.data['nc'] = self.num_classes
        self.data['names'] = ['lighthouse', 'sailboat', 'buoy', 'railbar', 'cargoship', 'navalvessels', 'passengership', 'dock', 'submarine', 'fishingboat']
        self.class_map = self.data['names']

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content).convert('RGB')

                # ======================================================================================================================
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                ori_image = np.array(image)
                ori_image = np.ascontiguousarray(ori_image)

                ori_height, ori_width, _ = ori_image.shape
                resize_ratio = self.img_size / max(ori_height, ori_width)
                resize_width = int(ori_width * resize_ratio)
                resize_height = int(ori_height * resize_ratio)
                image = cv2.resize(ori_image, (resize_width, resize_height))

                image, padding_top, padding_left = Padding(image, self.img_size, fill_value=128)

                # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                transform = Compose([
                    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                    ToTensorV2(p=1.0),
                ], p=1.)

                image = transform(image=image)['image'].to(dtype=torch.float32)
                image = image / 255.

                image = image[None]

                preprocessed_data[k] = [image.to(self.device), file_name, resize_ratio, padding_left, padding_top, ori_height, ori_width]

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # ======================================================================================================================

        return preprocessed_data


    def _postprocess(self, data):
        return data

    def postprocess(self, predict_layers):
        # ======================================================================================================================
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ori_predict = []
        for layer_idx in range(len(predict_layers)):
            predict = predict_layers[layer_idx]
            anchor = self.anchors[layer_idx]
            num_anchors = anchor.size(0)
            stride = self.stride[layer_idx]

            bs, c, h, w = predict.size()
            num_classes = c // num_anchors - 5

            predict = predict.view(bs, num_anchors, num_classes+5, h, w).permute(0, 1, 3, 4, 2).contiguous() # torch.Size([1, 3, 19, 19, 15])

            # grid_xy = grid(height=h, width=w, mode='xy').repeat(1, 1, 1, 1).unsqueeze(3).to(predict)  # torch.Size([1, 1, 19, 19, 2])
            grid_xy = grid(height=h, width=w, mode='xy').repeat(1, 1, 1, 1, 1).to(predict)  # torch.Size([1, 3, 19, 19, 15])
            anchor_wh = anchor.repeat(1, 1, 1, 1, 1).permute(0, 3, 1, 2, 4)  # torch.Size([1, 3, 1, 1, 2])

            predict[..., 0:2] = (torch.sigmoid(predict[..., 0:2]) + grid_xy) * stride
            predict[..., 2:4] = (torch.exp(predict[..., 2:4]) * anchor_wh) * stride
            predict[..., 4:5] = torch.sigmoid(predict[..., 4:5])
            predict[..., 5:] = torch.sigmoid(predict[..., 5:])

            ori_predict.append(predict.view(bs, -1, num_classes+5))

        ori_predict = torch.cat(ori_predict, dim=1) # torch.Size([1, 7581, 45])

        out = non_max_suppression(ori_predict, conf_thres=self.conf_thres, iou_thres=self.iou_thres, labels=[], multi_label=True, agnostic=False)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ======================================================================================================================
        return out


    def _inference(self, data):

        # preprocess
        image, file_name, resize_ratio, padding_left, padding_top, ori_height, ori_width = data['images']

        # predict
        predict_layers = self.model(image)

        # postprocess 这里是自己写的后处理方法，区别于 _postprocess
        out = self.postprocess(predict_layers)

        # write to results
        result = {}
        result['detection_classes'] = []
        result['detection_scores'] = []
        result['detection_boxes'] = []
        for si, pred in enumerate(out): # 只有一个
            scale_coords(image.shape[2:], pred[:, :4], (ori_height, ori_width))  # native-space pred

            path = Path(file_name)
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            box = pred[:, :4]  # xyxy

            for p, b in zip(pred.tolist(), box.tolist()):
                b = [b[1],b[0],b[3],b[2]]  # y1 x1 y2 x2
                result['detection_classes'].append( self.class_map[int(p[5])] )
                result['detection_scores'].append( round(p[4], 5) )
                result['detection_boxes'].append( [round(x, 3) for x in b ] )
        return result