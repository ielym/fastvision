import os
import yaml
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, distributed

from data_gen import create_dataset
from utils.anchor_generator import AnchorGenerator
from utils.map import mean_average_precision
from cfg._fit import Fit
from models.yolov3 import YoloV3

from utils.lossv3 import ComputeLoss
# from utils.lossv3_lambda import ComputeLoss


def dataloader_fn(data_yaml, batch_size, num_workers, input_size):
    data_dict = yaml.safe_load(open(data_yaml, 'r'))

    num_classes = data_dict['num_classes']
    category_names = data_dict['categories']
    assert (num_classes == len(category_names)), f"num_classes {num_classes} must equal len(category_names) {len(category_names)}"

    train_dir = os.path.join(data_dict['data_root'], data_dict['train_dir'])
    train_dataset = create_dataset(train_dir, input_size=input_size, mode='train')

    val_dataset = create_dataset(train_dir, input_size=input_size, mode='val')

    train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=False,
                num_workers=num_workers,
                collate_fn=train_dataset.collate_fn,
            )

    val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=False,
                num_workers=num_workers,
                collate_fn=train_dataset.collate_fn,
            )

    return train_loader, val_loader, data_dict

def anchor_fn(data_loaders=(), num_anchors=9, save_dir='./'):

    # anchor_generator = AnchorGenerator(data_loaders, k=num_anchors, iters=100, plot=True, save_dir=save_dir)
    # anchors = anchor_generator.get_anchors()
    # anchors = torch.from_numpy(anchors).float().view(-1, 2).to(device=device)

    anchors_small = torch.tensor([[462, 202], [340, 431], [522, 364]]) / 2 / 32
    anchors_medium = torch.tensor([[113, 270], [230, 134], [197, 397]]) / 2 / 16
    anchors_large = torch.tensor([[15, 24], [46, 63], [79, 134]]) / 2 / 8

    # anchors_small = torch.tensor([[116, 90], [156, 198], [373, 326]], dtype=torch.float32) / 32
    # anchors_medium = torch.tensor([[30, 61], [62, 45], [59, 119]], dtype=torch.float32) / 16
    # anchors_large = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32) / 8

    return (anchors_small.cuda(), anchors_medium.cuda(), anchors_large.cuda())

def optimizer_fn(model, lr, weight_decay):

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    optimizer = torch.optim.SGD(g0, lr=lr, momentum=0.937, nesterov=True)
    # optimizer = torch.optim.Adam(g0, lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum

    optimizer.add_param_group({'params': g1, 'weight_decay': weight_decay})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)

    return optimizer

def model_fn(backbone_weights, yolo_weights, in_channels, num_classes, anchors):

    model = YoloV3(in_channels=in_channels, num_classes=num_classes, anchors=anchors, backbone_weights=backbone_weights)

    if yolo_weights:
        pretrained_dict = torch.load(yolo_weights)
        pretrained_dict.pop('head.head_out_large.weight')
        pretrained_dict.pop('head.head_out_large.bias')
        pretrained_dict.pop('head.head_out_medium.weight')
        pretrained_dict.pop('head.head_out_medium.bias')
        pretrained_dict.pop('head.head_out_small.weight')
        pretrained_dict.pop('head.head_out_small.bias')
        model.load_state_dict(pretrained_dict, False)

    model = model.cuda()
    model = nn.DataParallel(model)

    for name, value in model.named_parameters():
        value.requires_grad = True

    for name, value in model.named_parameters():
        if 'neck' in name:
            value.requires_grad = True

    return model

def train(args):

    # ======================= Data Loader ============================
    train_loader, val_loader, data_dict = dataloader_fn(data_yaml=args.data_yaml, batch_size=args.batch_size, num_workers=args.num_workers, input_size=args.input_size)

    # ======================= Anchor Generator ============================
    anchors = anchor_fn()

    # ======================= Model ============================
    num_classes = data_dict['num_classes']
    model = model_fn(backbone_weights=args.backbone_weights, yolo_weights=args.yolo_weights, in_channels=args.in_channels, num_classes=num_classes, anchors=anchors)

    # ======================= Hyp Parameters ============================
    args.warmup_init_lr = args.init_lr * 0.1
    args.min_lr = args.init_lr * 0.01
    # args.min_lr = 0
    args.warmup_iters = args.warmup_epoch * len(train_loader)

    # ======================= Loss ============================
    yolo_loss = ComputeLoss().cuda()

    # ======================= metric ============================
    metric = mean_average_precision(np.linspace(0.5, 0.95, 10))

    # ======================= Optimizer ============================
    optimizer = optimizer_fn(model=model, lr=args.init_lr, weight_decay=5e-4)

    # def one_cycle(y1=0.0, y2=1.0, steps=100):
    #     return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1
    # lf = one_cycle(1, args.learning_rate * 0.1, args.max_epochs) # 1, final_lr (min lr), total epochs
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=args.total_epoch - args.warmup_epoch - args.no_aug_epoch, T_mult=2, eta_min=args.min_lr)

    Fit(
        model= model,
        args=args,
        optimizer=optimizer,
        criterion=yolo_loss,
        metric=metric,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader=val_loader
    )