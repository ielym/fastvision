# -*- coding: utf-8 -*-
import argparse
import os
import random
import multiprocessing


import numpy as np
import torch
import torch.backends.cudnn as cudnn

from fastvision.utils.seed import set_random_seeds
from fastvision.utils.device import set_device

from train import Train
from inference import Inference

parser = argparse.ArgumentParser(description='FastVision')

parser.add_argument('--mode', default='Test', choices=['Train', 'Test'], type=str, help='')

# Data generation
parser.add_argument('--data_yaml', default=r'./data/voc.yaml', type=str, help='')
parser.add_argument('--input_size', default=608, type=int, help='')
parser.add_argument('--in_channels', default=3, type=int, help='')
parser.add_argument('--max_det', default=300, type=int, help='')
parser.add_argument('--nms_conf_threshold', default=0.25, type=float, help='')
parser.add_argument('--nms_iou_threshold', default=0.45, type=float, help='')


# YoLO
parser.add_argument('--pretrained_weights', default=r'P:\PythonWorkSpace\fastvision-ttt\yolov3.pth', type=str, help='')
parser.add_argument('--num_anchors_per_level', default=[3, 3, 3], type=list, help='')

# Loss
parser.add_argument('--iou_negative_thres', default=0.5, type=float, help='for selecting response/negative/ignore samples when compute loss')
parser.add_argument('--ratio_box', default=0.05, type=float, help='')
parser.add_argument('--ratio_conf', default=1.0, type=float, help='')
parser.add_argument('--ratio_cls', default=0.5, type=float, help='')


# Train
parser.add_argument('--device', default='cpu', type=str, help='cpu, or 0, 1 or 0')
parser.add_argument('--epochs', default=300, type=int, help='')
parser.add_argument('--batch_size', default=2, type=int, help='')
parser.add_argument('--initial_lr', default=1e-3, type=float, help='')
parser.add_argument('--last_lr', default=1e-5, type=float, help='')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='')
parser.add_argument('--num_workers', default=0.2, type=float, help='')
parser.add_argument('--seed', default=2021, type=int, help='0/1/2/... or None')
parser.add_argument('--DataParallel', default=True, type=bool, help='')
parser.add_argument('--DistributedDataParallel', default=False, type=bool, help='')
parser.add_argument('--SyncBatchNorm', default=True, type=bool, help='')

# Inference
parser.add_argument('--inference_weights', default=r'P:\PythonWorkSpace\last.pth', type=str, help='')
parser.add_argument('--img_path', default=r'S:\datasets\voc2012\val\images', type=str, help='absolute path of a img or a folder')

# cache
parser.add_argument('--cache_dir', default='./cache', type=str, help='')
parser.add_argument('--use_data_cache', default=True, type=bool, help='')
parser.add_argument('--use_anchor_cache', default=True, type=bool, help='')


args, unknown = parser.parse_known_args()

def main(args):

    set_random_seeds(args.seed)

    device = set_device(args.device)

    if args.num_workers >= 0 and args.num_workers <= 1:
        args.num_workers = min(int(multiprocessing.cpu_count() * args.num_workers), int(multiprocessing.cpu_count()))
        print(f'Num Workers : {args.num_workers}')
    else:
        raise Exception(f"num_works must be 0 or in range [0, 1]")


    if args.mode == 'Train':
        args.training = True
        Train(args=args, device=device)
    elif args.mode == 'Test':
        args.training = False
        Inference(args=args, device=device)

if __name__ == '__main__':
    main(args)
