import os
from glob import glob
import yaml
import cv2

import numpy as np
import torch
import torch.nn as nn

from fastvision.datasets.detection_dataloader import create_dataloader, show_dataset
from fastvision.detection.tools import AnchorGenerator
from fastvision.utils.checkpoints import LoadStatedict
from fastvision.detection.tools import non_max_suppression
from fastvision.detection.plot import draw_box_label

def dataloader_fn(data_yaml, path):
    data_dict = yaml.safe_load(open(data_yaml, 'r'))

    num_classes = data_dict['num_classes']
    category_names = data_dict['categories']
    assert (num_classes == len(category_names)), f"num_classes {num_classes} must equal len(category_names) {len(category_names)}"

    if os.path.isdir(path):
        base_names = os.listdir(path)
        file_names = [os.path.join(path, name) for name in base_names]

    else:
        file_names = [path]

    return file_names, data_dict

def anchor_fn(data_loaders=None, num_anchors=None, num_workers=0, device=None, cache='', use_cache=False):

    if isinstance(data_loaders, type(None)):
        use_cache = True
        assert cache, "You should specify a cache path to load anchors, or you should definition data_loaders to generate anchors"

    anchor_generator = AnchorGenerator(data_loaders=data_loaders, k=num_anchors, iters=100, num_workers=num_workers, cache=cache, use_cache=use_cache, plot=True)

    anchors = anchor_generator.get_anchors()

    anchors = torch.from_numpy(anchors).float().view(-1, 2).to(device=device)

    return anchors

def model_fn(weights, in_channels, num_classes, num_anchors_per_level, device, anchors, DataParallel=False, DistributedDataParallel=False, SyncBatchNorm=False, training=True):
    from fastvision.classfication.models import darknet53
    from fastvision.detection.neck import yolov3neck
    from fastvision.detection.head import yolov3head
    from fastvision.detection.models import yolov3

    model = yolov3(backbone=darknet53, neck=yolov3neck, head=yolov3head, anchors=anchors, num_anchors_per_level=num_anchors_per_level, in_channels=in_channels, num_classes=num_classes, training=training)

    model = LoadStatedict(model=model, weights=weights, device=device, strict=True)

    if device.type == 'cuda':
        print('Model : using cuda')
        model = model.cuda()

    if device.type == 'cuda' and DataParallel:
        print('Model : using DataParallel')
        model = nn.DataParallel(model)

    model.half().float()
    return model

def preprocess(img_name):
    ori_img = cv2.imread(img_name)
    rgb_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, (608, 608))

    img_out = resized_img / 255.

    img_out = img_out.transpose([2, 0, 1])
    img_out = np.ascontiguousarray(img_out)
    img_out = img_out.astype(np.float32)
    img_out = torch.from_numpy(img_out)

    return resized_img, img_out

@torch.no_grad()
def Inference(args, device):

    file_names, data_dict = dataloader_fn(args.data_yaml, args.img_path)
    num_classes = data_dict['num_classes']
    category_names = data_dict['categories']
    category_id_name_map = {k : v for k, v in enumerate(category_names)}

    # ======================= Anchor Generator ============================
    anchors = anchor_fn(cache=args.cache_dir, use_cache=True)

    # ======================= Model ============================
    model = model_fn(args.inference_weights, args.in_channels, num_classes, args.num_anchors_per_level, anchors=anchors, training=args.training, device=device, DataParallel=args.DataParallel, DistributedDataParallel=args.DistributedDataParallel, SyncBatchNorm=args.SyncBatchNorm)

    for file_name in file_names:

        img, img_in = preprocess(file_name)

        head_out, results = model(img_in.unsqueeze(0))

        scores, categories, boxes = non_max_suppression(results, conf_thres=args.nms_conf_threshold, iou_thres=args.nms_iou_threshold, max_det=args.max_det)

        for box, category, score in zip(boxes, categories, scores):
            img = draw_box_label(img, box, f'{category_id_name_map[int(category)]} {float("%.2f" % score)}', bgr=True, line_color=int(category))

        cv2.imshow('img', img)
        cv2.waitKey(1000)