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

parser.add_argument('--data_yaml', default=r'./data/coco.yaml', type=str, help='voc.yaml or coco.yaml')
parser.add_argument('--in_channels', default=3, type=int, help='')
parser.add_argument('--input_size', default=416, type=int, help='')

parser.add_argument('--batch_size', default=1, type=int, help='')
# 5e-5 small, 1e-e faster but saturated when 30 epochs
parser.add_argument('--learning_rate', default=1e-4, type=float, help='') # 1e-4 > 5e-4 > 1e-3 > 4e-3

parser.add_argument('--start_epoch', default=0, type=int, help='')
parser.add_argument('--max_epochs', default=99999999, type=int, help='')

parser.add_argument('--seed', default=20220504, type=int, help='')
# parser.add_argument('--num_workers', default=int(multiprocessing.cpu_count() * 0.8), type=int, help='')
parser.add_argument('--num_workers', default=0, type=int, help='')

# parser.add_argument('--backbone_weights', default='./epoch_66_acc1_71.61752537318638_acc5_90.56976583052655_loss_1.1456699687607435.pth', type=str, help='')
parser.add_argument('--backbone_weights', default=None, type=str, help='')
parser.add_argument('--yolo_weights', default=None, type=str, help='')

args, unknown = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
