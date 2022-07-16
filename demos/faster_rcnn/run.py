# -*- coding: utf-8 -*-
import argparse
import os
import random
import numpy as np
import torch
import warnings
import multiprocessing

import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--training', default=False, type=bool, help='')

parser.add_argument('--data_yaml', default=r'./data/voc.yaml', type=str, help='voc.yaml or coco.yaml')
parser.add_argument('--input_size', default=800, type=int, help='')
parser.add_argument('--in_channels', default=3, type=int, help='')

parser.add_argument('--batch_size', default=1, type=int, help='') # max 64 for 4 GPUs
parser.add_argument('--start_epoch', default=0, type=int, help='')
parser.add_argument('--total_epoch', default=300, type=int, help='')
parser.add_argument('--init_lr', default=1e-2, type=float, help='') # 1e-4 > 5e-4 > 1e-3 > 4e-3

parser.add_argument('--scales', default=[128, 256, 512], type=list, help='')
parser.add_argument('--ratios', default=[1, 1/2, 2], type=list, help='')

parser.add_argument('--backbone_weights', default=r'P:\PythonWorkSpace\zoos\vgg16.pth', type=str, help='')
parser.add_argument('--backbone_stride', default=16, type=int, help='')
parser.add_argument('--backbone_output_channels', default=512, type=int, help='')

parser.add_argument('--rpn_positive_iou_thres', default=0.7, type=float, help='')
parser.add_argument('--rpn_negative_iou_thres', default=0.3, type=float, help='')
parser.add_argument('--rpn_positives_per_image', default=128, type=int, help='')
parser.add_argument('--rpn_negatives_per_image', default=128, type=int, help='')
parser.add_argument('--rpn_pre_nms_top_n', default=6000, type=int, help='') # 12000 for train, 6000 for test
parser.add_argument('--rpn_post_nms_top_n', default=300, type=int, help='') # 2000 for train, 300 for test
parser.add_argument('--rpn_nms_thresh', default=0.7, type=float, help='')

parser.add_argument('--fast_multi_reg_head', default=True, type=bool, help='') # 每个类别都独立预测一个边界框，即 num_class * 4 个
parser.add_argument('--fast_positive_iou_thres', default=0.5, type=float, help='')
parser.add_argument('--fast_negative_iou_thres', default=0.5, type=float, help='')
parser.add_argument('--fast_positives_per_image', default=16, type=int, help='')
parser.add_argument('--fast_negatives_per_image', default=48, type=int, help='')
parser.add_argument('--fast_roi_pool', default=7, type=int, help='')

parser.add_argument('--seed', default=20220504, type=int, help='')
# parser.add_argument('--num_workers', default=int(multiprocessing.cpu_count() * 0.8), type=int, help='')
parser.add_argument('--num_workers', default=0, type=int, help='')

parser.add_argument('--inference_weights', default=r'./20.pth', type=str, help='')
parser.add_argument('--inference_conf_thres', default=0.5, type=float, help='')
parser.add_argument('--inference_iou_thres', default=0.3, type=float, help='')

args, unknown = parser.parse_known_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print('CUDA device count : {}'.format(torch.cuda.device_count()))

def check_args(args):
	print(args.data_local)
	if not os.path.exists(args.data_local):
		raise Exception('FLAGS.data_local_path: %s is not exist' % args.data_local)

def set_random_seeds(args):
	os.environ['PYTHONHASHSEED'] = str(args.seed)
	cudnn.deterministic = True
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
	print('You have chosen to seed training with seed {}.'.format(args.seed))

def main(args,**kwargs):
	# check_args(args)
	if args.seed != None:
		set_random_seeds(args)
	else:
		print('You have chosen to random seed.')

	if args.training:
		from train import train
		train(args=args)
	else:
		from inference import Inference
		# from inference_voc_test import Inference
		Inference(args=args)

if __name__ == '__main__':
	main(args)
