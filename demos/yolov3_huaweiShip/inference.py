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

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

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

def preProcess(img_path, input_size):

    try:
        ori_img = cv2.imread(img_path)
        h, w, c = ori_img.shape
    except:
        ori_img = Image.open(img_path)
        ori_img = ori_img.convert("RGB")
        ori_img = np.array(ori_img)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
        h, w, c = ori_img.shape


    ori_height, ori_width, _ = ori_img.shape

    resize_ratio = input_size / max(ori_height, ori_width)
    resize_width = int(ori_width * resize_ratio)
    resize_height = int(ori_height * resize_ratio)
    resize_image = cv2.resize(ori_img, (resize_width, resize_height))

    image, padding_top, padding_left = Padding(resize_image, input_size, fill_value=128)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = Compose([
        # PadIfNeeded(input_size, input_size, value=0, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

    image = transform(image=image)['image'].to(dtype=torch.float32)
    image = image / 255.

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

        grid_xy = grid(height=h, width=w, mode='xy').repeat(1, 1, 1, 1).unsqueeze(3).to(
            predict)  # torch.Size([1, 3, 19, 19, 85])
        anchor_wh = anchor.repeat(1, 1, 1, 1, 1)  # torch.Size([1, 1, 1, 3, 2])

        predict[..., 0:2] = (torch.sigmoid(predict[..., 0:2]) + grid_xy) * stride
        predict[..., 2:4] = (torch.exp(predict[..., 2:4]) * anchor_wh) * stride
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
    # single_dict = {}
    # for k, v in weights.items():
    #     single_dict[k[7:]] = v
        # single_dict[k] = v
    model.load_state_dict(weights, True)
    model.eval()

    return model

def anchor_fn():

    # anchors_small = torch.tensor([[116, 90], [156, 198], [373, 326]]) / 32
    # anchors_medium = torch.tensor([[30, 61], [62, 45], [59, 119]]) / 16
    # anchors_large = torch.tensor([[10, 13], [16, 30], [33, 23]]) / 8

    # 608
    anchors_small = torch.tensor([[462, 202], [340, 431], [522, 364]]) / 2 / 32
    anchors_medium = torch.tensor([[113, 270], [230, 134], [197, 397]]) / 2 / 16
    anchors_large = torch.tensor([[15, 24], [46, 63], [79, 134]]) / 2 / 8

    return (anchors_small.cuda(), anchors_medium.cuda(), anchors_large.cuda())

@torch.no_grad()
def Inference(pretrained_weights):

    # pretrained_weights = r'./epoch_5_loss_0.905585370792283.pth'
    image_dir = r'/home/ymluo/datasets1/huawei_det/V1/train/images'
    label_dir = r'/home/ymluo/datasets1/huawei_det/V1/train/labels'
    num_classes = 10
    input_size = 608 // 2 - 16
    conf_thres = 0.25
    iou_thres = 0.45

    anchors = anchor_fn()
    strides = [32, 16, 8]


    model = model_fn(pretrained_weights, anchors, num_classes=num_classes)
    model.cuda()

    metric = mean_average_precision(map_iou_values=np.linspace(0.5, 0.95, 10))
    map_50 = 0

    files= glob(os.path.join(image_dir, '*.jpg'))
    # np.random.seed(0)
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

        metric.process_one(torch.cat([categories, scores, boxes], dim=1).to(dtype=torch.float32), labels)

        # if boxes.size(0) == 0:
        #     continue
        # for label in labels:
        #     label = label.to(dtype=torch.int32)
        #     gt_category_idx, gt_xmin, gt_ymin, gt_xmax, gt_ymax = label
        #     ori_img = draw_box_label(ori_img, (gt_xmin, gt_ymin, gt_xmax, gt_ymax), text=str(int(gt_category_idx)), line_color=int(gt_category_idx))

        # for box, category, score in zip(boxes, categories, scores):
        #     box = box.to(dtype=torch.int32)
        #     xmin, ymin, xmax, ymax = box
        #     ori_img = draw_box_label(ori_img, (xmin, ymin, xmax, ymax), text=str(int(category)), line_color=int(category))
        # cv2.imshow('img', ori_img)
        # cv2.waitKey()

    map_each_iou, map_each_cls, unknow = metric.fetch()

    f = open(r'./metric.txt', 'a')
    f.write(f'{pretrained_weights}\n')

    title = 'map@50\tmap@55\tmap@60\tmap@65\tmap@70\tmap@75\tmap@80\tmap@85\tmap@90\tmap@95\tmap@.5:.95'
    f.write(f'{title}\n')
    print(title)

    for iou in map_each_iou.tolist():
        iou = round(iou, 4)
        print(f'{str(iou).ljust(6, "0")}\t', end='')
        f.write(f'{str(iou).ljust(6, "0")}\t')
    print(f'{str(np.mean(map_each_iou)).ljust(6, "0")}\t')

    f.write(f'{str(np.mean(map_each_iou)).ljust(6, "0")}\n')
    f.close()

    return map_each_iou.tolist()

already = set()
import time
best_iou = float('-inf')
best_model = ''
while True:

    # pth_files = ['1e-3-50-best.pth', '1e-3-50-epoch_50_loss_0.06590198433647552.pth']
    pth_files = glob(os.path.join('./', '*.submit'))
    pth_files.sort()

    for pretrained_weights in pth_files:
        if pretrained_weights in already:
            continue

        print(pretrained_weights)
        map_each_iou = Inference(pretrained_weights)
        if np.mean(map_each_iou) > best_iou:
            best_iou = np.mean(map_each_iou)
            best_model = pretrained_weights
        print(f'best iou : {best_iou}, best_model : {best_model}')
        print("||" * 50)

        already.add(pretrained_weights)

    time.sleep(5)