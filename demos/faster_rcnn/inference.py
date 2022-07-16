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
from utils.anchor_generator import get_base_anchor
from models.faster import Faster_Rcnn



os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def Padding(image, size, fill_value=128):
    '''
    :param image: np.ndarray [H, W, c] RGB
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

    return padding_image

def preProcess(img_path, input_size):

    ori_img = Image.open(img_path)
    ori_img = ori_img.convert("RGB")
    ori_img = np.array(ori_img)


    ori_height, ori_width, _ = ori_img.shape

    resize_ratio = input_size / max(ori_height, ori_width)
    resize_width = int(ori_width * resize_ratio)
    resize_height = int(ori_height * resize_ratio)
    image = cv2.resize(ori_img, (resize_width, resize_height))

    padding_left = (input_size - resize_width) // 2
    padding_top = (input_size - resize_height) // 2

    image = Padding(image, input_size, fill_value=128)

    transform = Compose([
        # PadIfNeeded(input_size, input_size, value=128, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

    image = transform(image=image)['image'].to(dtype=torch.float32)

    image = image.unsqueeze(0) / 255.

    return image, cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR), resize_ratio, padding_left, padding_top, ori_height, ori_width

def postProcess(proposals, args, resize_ratio, padding_left, padding_top, ori_width, ori_height):

    proposals[:, 0:4] = proposals[:, 0:4] * args.backbone_stride

    proposals[:, 0] = (proposals[:, 0] - padding_left) / resize_ratio
    proposals[:, 1] = (proposals[:, 1] - padding_top) / resize_ratio
    proposals[:, 2] = proposals[:, 2] / resize_ratio
    proposals[:, 3] = proposals[:, 3] / resize_ratio

    proposals[:, 0] = proposals[:, 0].clamp_(0, ori_width - 1)
    proposals[:, 1] = proposals[:, 1].clamp_(0, ori_height - 1)
    proposals[:, 2] = proposals[:, 2].clamp_(0, ori_width)
    proposals[:, 3] = proposals[:, 3].clamp_(0, ori_height)

    keep = (proposals[..., 2] > 5) & (proposals[..., 3] > 5)
    proposals = proposals[keep, ...]

    proposals[:, 0:4] = xywh2xyxy(proposals[:, 0:4])
    proposals[:, 0] = proposals[:, 0].clamp_(0, ori_width - 1)
    proposals[:, 1] = proposals[:, 1].clamp_(0, ori_height - 1)
    proposals[:, 2] = proposals[:, 2].clamp_(0, ori_width - 1)
    proposals[:, 3] = proposals[:, 3].clamp_(0, ori_height - 1)

    results = non_max_suppression(proposals, conf_thres=args.inference_conf_thres, iou_thres=args.inference_iou_thres, max_det=300)

    boxes = results[:, 0:4]
    categories = results[:, 4:5]
    scores = results[:, 5:6]

    return scores, categories, boxes


def model_fn(args, base_anchors):

    model = Faster_Rcnn(
        training=args.training,
        in_channels=args.in_channels,
        num_classes=args.num_classes,

        base_anchors=base_anchors,

        backbone_stride=args.backbone_stride,
        backbone_output_channels=args.backbone_output_channels,
        backbone_weights=args.backbone_weights,

        rpn_positive_iou_thres=args.rpn_positive_iou_thres,
        rpn_negative_iou_thres=args.rpn_negative_iou_thres,
        rpn_positives_per_image=args.rpn_positives_per_image,
        rpn_negatives_per_image=args.rpn_negatives_per_image,
        rpn_pre_nms_top_n=args.rpn_pre_nms_top_n,
        rpn_post_nms_top_n=args.rpn_post_nms_top_n,
        rpn_nms_thresh=args.rpn_nms_thresh,

        fast_multi_reg_head=args.fast_multi_reg_head,
        fast_positive_iou_thres=args.fast_positive_iou_thres,
        fast_negative_iou_thres=args.fast_negative_iou_thres,
        fast_positives_per_image=args.fast_positives_per_image,
        fast_negatives_per_image=args.fast_negatives_per_image,
        fast_roi_pool=args.fast_roi_pool,
    )

    weights = torch.load(args.inference_weights)
    model.load_state_dict(weights, True)

    model.eval()

    return model

def anchor_fn(scales, ratios):

    base_anchors = get_base_anchor(scales=scales, ratios=ratios)
    base_anchors = torch.from_numpy(base_anchors)  # torch.Size([9, 2]) [w, h]

    return base_anchors

@torch.no_grad()
def Inference(args):

    image_dir = r'S:\datasets\voc2012\test\images'
    label_dir = r'S:\datasets\voc2012\test\labels'

    # conf_thres = 0.001
    # iou_thres = 0.6
    base_anchors = anchor_fn(args.scales, args.ratios)

    args.num_classes = 20
    model = model_fn(args=args, base_anchors=base_anchors)
    # model.cuda()

    metric = mean_average_precision(map_iou_values=np.linspace(0.5, 0.95, 10))
    map_50 = 0

    files= glob(os.path.join(image_dir, '*.jpg'))
    # np.random.seed(1)
    # np.random.shuffle(files)
    for file in tqdm.tqdm(files):

        image, ori_img, resize_ratio, padding_left, padding_top, ori_height, ori_width = preProcess(file, args.input_size)
        predicts = model(image)
        scores, categories, boxes = postProcess(predicts[0], args, resize_ratio, padding_left, padding_top, ori_width, ori_height)

        # base_name = os.path.basename(file)
        # label_path = os.path.join(label_dir, f'{base_name.split(".")[0]}.txt')
        # with open(label_path, 'r') as f:
        #     lines = f.readlines()
        # labels = []
        # for line in lines:
        #     line = line.strip()
        #     gt_category_idx, gt_xmin, gt_ymin, gt_xmax, gt_ymax = line.split()
        #     labels.append([float(gt_category_idx), float(gt_xmin), float(gt_ymin), float(gt_xmax), float(gt_ymax)])
        # labels = torch.tensor(labels, dtype=torch.int32).to(device=predicts[0].device).view(-1, 5)
        #
        # metric.process_one(torch.cat([categories, scores, boxes], dim=1).to(dtype=torch.float32), labels)

        if boxes.size(0) == 0:
            continue
        # for label in labels:
        #     label = label.to(dtype=torch.int32)
        #     gt_category_idx, gt_xmin, gt_ymin, gt_xmax, gt_ymax = label
        #     ori_img = draw_box_label(ori_img, (gt_xmin, gt_ymin, gt_xmax, gt_ymax), text=str(int(gt_category_idx)), line_color=int(gt_category_idx))
            # ori_img = draw_box_label(ori_img, (gt_xmin, gt_ymin, gt_xmax, gt_ymax), text=str(int(gt_category_idx)), line_color=int(0))

        for box, category, score in zip(boxes, categories, scores):
            box = box.to(dtype=torch.int32)
            xmin, ymin, xmax, ymax = box
            ori_img = draw_box_label(ori_img, (xmin, ymin, xmax, ymax), text=str(int(category)) + str(round(float(score), 2)), line_color=int(category))
        cv2.imshow('img', ori_img)
        cv2.waitKey(1)

    map_each_iou, map_each_cls, unknow = metric.fetch()

    title = 'map@50\tmap@55\tmap@60\tmap@65\tmap@70\tmap@75\tmap@80\tmap@85\tmap@90\tmap@95\tmap@.5:.95'
    print(title)

    for iou in map_each_iou.tolist():
        iou = round(iou, 4)
        print(f'{str(iou).ljust(6, "0")}\t', end='')
    print(f'{str(np.mean(map_each_iou)).ljust(6, "0")}\t')

    return map_each_iou.tolist()