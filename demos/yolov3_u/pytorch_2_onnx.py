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

def preProcess(img_path, input_size):

    ori_img = Image.open(img_path)
    ori_img = ori_img.convert("RGB")
    ori_img = np.array(ori_img)

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

    '''

    input_size = 640 conf_thres = 0.25 iou_thres = 0.65
    map@50	map@55	map@60	map@65	map@70	map@75	map@80	map@85	map@90	map@95	map@.5:.95
    0.3943	0.3558	0.3112	0.2591	0.2017	0.1421	0.0870	0.0462	0.0188	0.0032	0.1819440726740018

    input_size = 640 conf_thres = 0.25 iou_thres = 0.45
    map@50	map@55	map@60	map@65	map@70	map@75	map@80	map@85	map@90	map@95	map@.5:.95
    0.5136	0.4654	0.4095	0.3414	0.2675	0.1922	0.1202	0.0660	0.0280	0.0050	0.24088026155546705

    input_size = 640 conf_thres = 0.25 iou_thres = 0.35
    map@50	map@55	map@60	map@65	map@70	map@75	map@80	map@85	map@90	map@95	map@.5:.95
    0.5293	0.4790	0.4218	0.3523	0.2764	0.1985	0.1245	0.0681	0.0284	0.0051	0.24834044208465628

    input_size = 640 conf_thres = 0.25 iou_thres = 0.25
    map@50	map@55	map@60	map@65	map@70	map@75	map@80	map@85	map@90	map@95	map@.5:.95
    0.5325	0.4854	0.4292	0.3603	0.2838	0.2044	0.1287	0.0703	0.0291	0.0052	0.2529076876796873

    input_size = 640 conf_thres = 0.25 iou_thres = 0.15
    map@50	map@55	map@60	map@65	map@70	map@75	map@80	map@85	map@90	map@95	map@.5:.95
    0.5308	0.4842	0.4284	0.3606	0.2851	0.2063	0.1304	0.0722	0.0300	0.0054	0.25334121693509015

    input_size = 640 conf_thres = 0.35 iou_thres = 0.25
    map@50	map@55	map@60	map@65	map@70	map@75	map@80	map@85	map@90	map@95	map@.5:.95
    0.5421	0.4955	0.4395	0.3706	0.2931	0.2126	0.1351	0.0743	0.0311	0.0055	0.25993934870104235

    input_size = 640 conf_thres = 0.45 iou_thres = 0.25
    map@50	map@55	map@60	map@65	map@70	map@75	map@80	map@85	map@90	map@95	map@.5:.95
    0.5499	0.5047	0.4489	0.3800	0.3026	0.2207	0.1409	0.0782	0.0327	0.0059	0.2664507014444463

    input_size = 640 conf_thres = 0.55 iou_thres = 0.25
    map@50	map@55	map@60	map@65	map@70	map@75	map@80	map@85	map@90	map@95	map@.5:.95
    0.5544	0.5111	0.4571	0.3892	0.3118	0.2293	0.1472	0.0821	0.0347	0.0064	0.27233059916151336

    input_size = 640 conf_thres = 0.65 iou_thres = 0.25
    map@50	map@55	map@60	map@65	map@70	map@75	map@80	map@85	map@90	map@95	map@.5:.95
    0.5537	0.5145	0.4622	0.3966	0.3216	0.2385	0.1536	0.0862	0.0359	0.0065	0.27693292376340733
    '''

    pretrained_weights = r'./yolov3_weights_from_github_ultralytics.pth'
    image_dir = r'S:/datasets/coco2017/val/images'
    label_dir = r'S:/datasets/coco2017/val/labels'
    num_classes = 80
    input_size = 640
    conf_thres = 0.25
    iou_thres = 0.45
    anchors = anchor_fn()
    strides = [32, 16, 8]

    model = model_fn(pretrained_weights, anchors, num_classes=num_classes)
    model.cuda()

    metric = mean_average_precision(map_iou_values=np.linspace(0.5, 0.95, 10))
    map_50 = 0

    files= glob(os.path.join(image_dir, '*.jpg'))
    # np.random.seed(1)
    # np.random.shuffle(files)
    for file in tqdm.tqdm(files):
        image, ori_img, resize_ratio, padding_left, padding_top, ori_height, ori_width = preProcess(file, input_size)
        out = model(image.cuda())
        scores, categories, boxes = postProcess(out, strides, anchors, conf_thres, iou_thres, resize_ratio, padding_left, padding_top, ori_width, ori_height)

        base_name = os.path.basename(file)
        label_path = os.path.join(label_dir, f'{base_name.split(".")[0]}.txt')
        with open(label_path, 'r') as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            line = line.strip()
            gt_category_idx, gt_xmin, gt_ymin, gt_xmax, gt_ymax = line.split()
            labels.append([float(gt_category_idx), float(gt_xmin), float(gt_ymin), float(gt_xmax), float(gt_ymax)])
        labels = torch.tensor(labels, dtype=torch.int32).to(device=out[0].device).view(-1, 5)

        # metric.process_one(torch.cat([categories, scores, boxes], dim=1).to(dtype=torch.float32), labels)

        # if boxes.size(0) == 0:
        #     continue
        # for label in labels:
        #     label = label.to(dtype=torch.int32)
        #     gt_category_idx, gt_xmin, gt_ymin, gt_xmax, gt_ymax = label
        #     ori_img = draw_box_label(ori_img, (gt_xmin, gt_ymin, gt_xmax, gt_ymax), text=str(int(gt_category_idx)), line_color=int(gt_category_idx))

        for box, category, score in zip(boxes, categories, scores):
            box = box.to(dtype=torch.int32)
            xmin, ymin, xmax, ymax = box
            ori_img = draw_box_label(ori_img, (xmin, ymin, xmax, ymax), text=str(int(category)), line_color=int(category))
        cv2.imshow('img', ori_img)
        cv2.waitKey()

    map_each_iou, map_each_cls, unknow = metric.fetch()

    f = open(r'./metric.txt', 'a')
    title = 'map@50\tmap@55\tmap@60\tmap@65\tmap@70\tmap@75\tmap@80\tmap@85\tmap@90\tmap@95\tmap@.5:.95'
    f.write(f'{title}\n')
    print(title)

    for iou in map_each_iou.tolist():
        iou = round(iou, 4)
        print(f'{str(iou).ljust(6, "0")}\t', end='')
        f.write(f'{str(iou).ljust(6, "0")}\t')
    print(f'{str(np.mean(map_each_iou)).ljust(6, "0")}\t')

    f.write(f'{pretrained_weights}\n')
    f.write(f'{str(np.mean(map_each_iou)).ljust(6, "0")}\n')
    f.close()

    return map_each_iou.tolist()

Inference()