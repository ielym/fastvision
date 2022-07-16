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

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Resize,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop, PadIfNeeded,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize, RandomCrop
)
from albumentations.pytorch import ToTensorV2

from utils import non_max_suppression, non_max_suppression_batch, grid, draw_box_label, xywh2xyxy
from utils.anchor_generator import get_base_anchor
from models.faster import Faster_Rcnn



os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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

    transform = Compose([
        PadIfNeeded(input_size, input_size, value=128, border_mode=cv2.BORDER_CONSTANT, p=1.0),
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

        fast_positive_iou_thres=args.fast_positive_iou_thres,
        fast_negative_iou_thres=args.fast_negative_iou_thres,
        fast_positives_per_image=args.fast_positives_per_image,
        fast_negatives_per_image=args.fast_negatives_per_image,
        fast_roi_pool=args.fast_roi_pool,
    )

    weights = torch.load(args.inference_weights)
    # single_dict = {}
    # for k, v in weights.items():
    #     single_dict[k[7:]] = v
        # single_dict[k] = v
    model.load_state_dict(weights, True)
    model.eval()

    return model

def anchor_fn(scales, ratios):

    base_anchors = get_base_anchor(scales=scales, ratios=ratios)
    base_anchors = torch.from_numpy(base_anchors)  # torch.Size([9, 2]) [w, h]

    return base_anchors

def dataloader_fn(args):
    data_dict = yaml.safe_load(open(args.data_yaml, 'r'))

    num_classes = data_dict['num_classes']
    category_names = data_dict['categories']
    assert (num_classes == len(category_names)), f"num_classes {num_classes} must equal len(category_names) {len(category_names)}"
    args.num_classes = num_classes

    category_id_name_dict = {}
    for idx, name in enumerate(category_names):
        category_id_name_dict[idx] = name

    return category_id_name_dict

def prepare_folders(output_dir, year):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_dir = os.path.join(output_dir, 'results', f'VOC{year}', 'Main') # .../VOC2012/Main
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    return base_dir

def submit(predicts, output_dir, prefix='comp3_det_test_', year=2012):
    '''
    :param predicts: dict : {'car' : [(2009_000026, 0.949297, 172.000000, 233.000000, 191.000000, 248.000000), ...], 'person' : [(), (), ()]}
                            img_id, score, xmin, ymin, xmax, ymax
                            the xmin of image is 1, not 0
    :return:
    '''

    base_dir = prepare_folders(output_dir=output_dir, year=year)

    for category_name, predictions in predicts.items():
        file_path = os.path.join(base_dir, f'{prefix}{category_name}.txt')
        f = open(file_path, 'w')
        for obj in predictions:
            f.write(f'{obj[0]} {obj[1]} {obj[2]} {obj[3]} {obj[4]} {obj[5]}\n') # img_id, score, xmin, ymin, xmax, ymax
        f.close()

@torch.no_grad()
def Inference(args):

    image_dir = r'S:\datasets\voc2012\test\images'
    input_size = 608

    category_id_name_dict = dataloader_fn(args)
    print(category_id_name_dict)

    base_anchors = anchor_fn(args.scales, args.ratios)

    model = model_fn(args=args, base_anchors=base_anchors)
    # model.cuda()

    files= glob(os.path.join(image_dir, '*.jpg'))

    results = {}
    for file in tqdm.tqdm(files):

        image, ori_img, resize_ratio, padding_left, padding_top, ori_height, ori_width = preProcess(file, input_size)
        predicts = model(image)
        scores, categories, boxes = postProcess(predicts[0], args, resize_ratio, padding_left, padding_top, ori_width, ori_height)

        if boxes.size(0) == 0:
            continue

        img_id = os.path.basename(file).split('.')[0]
        for box, category, score in zip(boxes, categories, scores):

            xmin, ymin, xmax, ymax = box.cpu().numpy().tolist()
            category_name = category_id_name_dict[int(category)]
            score = float(score)

            if not category_name in results.keys():
                results[category_name] = []
            results[category_name].append((img_id, score, xmin, ymin, xmax, ymax))

            # ori_img = draw_box_label(ori_img, (int(xmin), int(ymin), int(xmax), int(ymax)), text=category_id_name_dict[int(category)], line_color=int(category))
        # cv2.imshow('img', ori_img)
        # cv2.waitKey()

    submit(results, output_dir='../../yolov4/pytorch-YOLOv4/', prefix='comp3_det_test_', year=2012)

