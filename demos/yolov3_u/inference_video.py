import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision
import cv2
import numpy as np
import os
from glob import glob
import tqdm
from PIL import Image
import yaml
import time
import requests

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Resize,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop, PadIfNeeded,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize, RandomCrop
)
from albumentations.pytorch import ToTensorV2

from utils import non_max_suppression, non_max_suppression_batch, mean_average_precision, mean_average_precision_ultralytics, grid, draw_box_label, xywh2xyxy
from models.yolov3 import YoloV3



os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def preProcess(ori_img, input_size):

    # ori_img = Image.open(img_path)
    # ori_img = ori_img.convert("RGB")
    # ori_img = np.array(ori_img)
    ori_img = cv2.resize(ori_img, (int(ori_img.shape[1] * 2), int(ori_img.shape[0] * 2)))

    ori_height, ori_width, _ = ori_img.shape

    resize_ratio = input_size / max(ori_height, ori_width)
    resize_width = int(ori_width * resize_ratio)
    resize_height = int(ori_height * resize_ratio)
    resize_image = cv2.resize(ori_img, (resize_width, resize_height))

    padding_left = (input_size - resize_width) // 2
    padding_top = (input_size - resize_height) // 2


    transform = Compose([
        PadIfNeeded(input_size, input_size, value=0, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        Normalize(mean=[0., 0., 0.,], std=[1., 1., 1.], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

    image = transform(image=resize_image)['image']

    image = image.unsqueeze(0)

    return image, ori_img, resize_ratio, padding_left, padding_top, ori_height, ori_width


def postProcess(predict_layers, strides, anchors, conf_thres, iou_thres, resize_ratio, padding_left, padding_top, ori_width, ori_height):
    '''
    :param predict_layers: [head_small, head_medium, head_large]
    :param strides:
    :param anchors:
    :return:
    '''
    ori_predict = []
    for layer_idx in range(len(predict_layers)):
        predict = predict_layers[layer_idx]
        anchor = anchors[layer_idx]
        num_anchors = anchor.size(0)
        stride = strides[layer_idx]

        bs, c, h, w = predict.size()
        num_classes = c // num_anchors - 5

        predict = predict.permute(0, 2, 3, 1).view(bs, h, w, num_anchors, -1)  # torch.Size([1, 3, 19, 19, 85])

        grid_xy = grid(height=h, width=w, mode='xy').repeat(1, 1, 1, 1).unsqueeze(3).to(predict)  # torch.Size([1, 3, 19, 19, 85])
        anchor_wh = anchor.repeat(1, 1, 1, 1, 1)  # torch.Size([1, 1, 1, 3, 2])

        # predict[..., 0:2] = (torch.sigmoid(predict[..., 0:2]) + grid_xy) * stride
        # predict[..., 2:4] = (torch.exp(predict[..., 2:4]) * anchor_wh) * stride
        # predict[..., 4:5] = torch.sigmoid(predict[..., 4:5])
        # predict[..., 5:] = torch.sigmoid(predict[..., 5:])
        # predict = predict.reshape(-1, num_classes + 5)

        predict[..., 0:2] = (torch.sigmoid(predict[..., 0:2]) * 2 - 0.5 + grid_xy) * stride
        predict[..., 2:4] = (torch.sigmoid(predict[..., 2:4]) * 2) ** 2 * anchor_wh * stride
        predict[..., 4:5] = torch.sigmoid(predict[..., 4:5])
        predict[..., 5:] = torch.sigmoid(predict[..., 5:])
        predict = predict.reshape(-1, num_classes + 5)

        predict[:, 0] = (predict[:, 0] - padding_left) / resize_ratio
        predict[:, 1] = (predict[:, 1] - padding_top) / resize_ratio
        predict[:, 2] = predict[:, 2] / resize_ratio
        predict[:, 3] = predict[:, 3] / resize_ratio

        predict[:, 0] = predict[:, 0].clamp_(0, ori_width - 1)
        predict[:, 1] = predict[:, 1].clamp_(0, ori_height - 1)
        predict[:, 2] = predict[:, 2].clamp_(0, ori_width)
        predict[:, 3] = predict[:, 3].clamp_(0, ori_height)

        keep = (predict[..., 2] > 5) & (predict[..., 3] > 5)
        predict = predict[keep, :]

        predict[:, 0:4] = xywh2xyxy(predict[:, 0:4])
        predict[:, 0] = predict[:, 0].clamp_(0, ori_width - 1)
        predict[:, 1] = predict[:, 1].clamp_(0, ori_height - 1)
        predict[:, 2] = predict[:, 2].clamp_(0, ori_width - 1)
        predict[:, 3] = predict[:, 3].clamp_(0, ori_height - 1)

        ori_predict.append(predict)
    ori_predict = torch.cat(ori_predict, dim=0)

    results = non_max_suppression(ori_predict, conf_thres=conf_thres, iou_thres=iou_thres, max_det=300)

    boxes = results[:, :4]
    scores = results[:, 4:5]
    categories = results[:, 5:6]

    return scores, categories, boxes


def model_fn(pretrainted_weights, anchors, num_classes=80):
    model = YoloV3(anchors=anchors, num_classes=num_classes)
    weights = torch.load(pretrainted_weights)
    model.load_state_dict(weights, True)
    model.eval()

    return model

def anchor_fn():

    anchors_small = torch.tensor([[116, 90], [156, 198], [373, 326]]) / 32
    anchors_medium = torch.tensor([[30, 61], [62, 45], [59, 119]]) / 16
    anchors_large = torch.tensor([[10, 13], [16, 30], [33, 23]]) / 8

    return (anchors_small.cuda(), anchors_medium.cuda(), anchors_large.cuda())

@torch.no_grad()
def Inference():

    pretrained_weights = r'./yolov3_weights_from_github_ultralytics.pth'
    image_dir = r'S:/datasets/coco2017/val/images'

    num_classes = 80
    input_size = 640
    conf_thres = 0.45
    iou_thres = 0.45
    anchors = anchor_fn()
    strides = [32, 16, 8]

    model = model_fn(pretrained_weights, anchors, num_classes=num_classes)
    model.cuda()

    data_dict = yaml.safe_load(open(r'./data/coco.yaml', 'r'))
    categories = data_dict['categories']

    idx_category_map = {}
    for idx, category in enumerate(categories):
        idx_category_map[idx] = category


    url_idx = 100
    while True:
        url_idx += 1

        # cap = cv2.VideoCapture(f'https://a24cbefee3fe63960a5ceeb4f400c6bf.livehwc3.cn/play.hngscloud.com/live/fb80317b-8947-449b-97ca-02fbee75f7f7_{url_idx}.ts?vhost=play.hngscloud.com&edge_slice=true')
        # cap = cv2.VideoCapture(f'https://a24cbefee3fe63960a5ceeb4f400c6bf.livehwc3.cn/play.hngscloud.com/live/fb80317b-8947-449b-97ca-02fbee75f7f7_{url_idx}.ts?vhost=play.hngscloud.com&edge_slice=true')
        cap = cv2.VideoCapture(f'https://a24cbefee3fe63960a5ceeb4f400c6bf.livehwc3.cn/play.hngscloud.com/live/80bd5d17-e6e9-446b-85d9-9004e5abf715_{url_idx}.ts?vhost=play.hngscloud.com&edge_slice=true')
        ret, frame = cap.read()
        print(url_idx, ret)
        if ret == False:
            continue

        cnt = -1
        stime = time.time()
        while ret:
            cnt += 1
            if cnt % 2 != 0:
                ret, frame = cap.read()
                continue
            image, ori_img, resize_ratio, padding_left, padding_top, ori_height, ori_width = preProcess(frame, input_size)
            out = model(image.cuda())
            scores, categories, boxes = postProcess(out, strides, anchors, conf_thres, iou_thres, resize_ratio, padding_left, padding_top, ori_width, ori_height)

            for box, category, score in zip(boxes, categories, scores):
                box = box.to(dtype=torch.int32)
                xmin, ymin, xmax, ymax = box
                ori_img = draw_box_label(ori_img, (xmin, ymin, xmax, ymax), text=idx_category_map[int(category)], line_color=int(category))
            cv2.imshow('img', ori_img)
            cv2.waitKey(1)

            ret, frame = cap.read()
        print(time.time() - stime)
        del cap

Inference()