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

parser.add_argument('--data_yaml', default=r'./data/v1.yaml', type=str, help='voc.yaml or coco.yaml')
parser.add_argument('--in_channels', default=3, type=int, help='')
parser.add_argument('--input_size', default=608, type=int, help='')

parser.add_argument('--batch_size', default=32, type=int, help='') # max 64 for 4 GPUs

parser.add_argument('--start_epoch', default=0, type=int, help='')
parser.add_argument('--warmup_epoch', default=10, type=int, help='')
parser.add_argument('--no_aug_epoch', default=10, type=int, help='')
parser.add_argument('--total_epoch', default=250, type=int, help='')

parser.add_argument('--init_lr', default=1e-3, type=float, help='') # 1e-4 > 5e-4 > 1e-3 > 4e-3


parser.add_argument('--seed', default=20220504, type=int, help='')
parser.add_argument('--num_workers', default=int(multiprocessing.cpu_count() * 0.3), type=int, help='')
# parser.add_argument('--num_workers', default=0, type=int, help='')

parser.add_argument('--backbone_weights', default='./backbone.pth', type=str, help='')
parser.add_argument('--yolo_weights', default=r'./yolov3_weights_from_github_ultralytics.pth', type=str, help='')

args, unknown = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'
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

	from train import train
	train(args=args)

if __name__ == '__main__':
	main(args)
