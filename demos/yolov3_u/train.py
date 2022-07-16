import os
import yaml
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

    val_dir = os.path.join(data_dict['data_root'], data_dict['val_dir'])
    val_dataset = create_dataset(val_dir, input_size=input_size, mode='val')

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

    anchors_small = torch.tensor([[116, 90], [156, 198], [373, 326]], dtype=torch.float32) / 32
    anchors_medium = torch.tensor([[30, 61], [62, 45], [59, 119]], dtype=torch.float32) / 16
    anchors_large = torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32) / 8

    return (anchors_small.cuda(), anchors_medium.cuda(), anchors_large.cuda())

def optimizer_fn(model, lr, weight_decay):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.937, 0.999), weight_decay=weight_decay)  # adjust beta1 to momentum

    return optimizer

def model_fn(backbone_weights, yolo_weights, in_channels, num_classes, anchors):

    model = YoloV3(in_channels=in_channels, num_classes=num_classes, anchors=anchors, backbone_weights=backbone_weights)

    if yolo_weights:
        pretrained_dict = torch.load(yolo_weights)
        single_dict = {}
        for k, v in pretrained_dict.items():
            single_dict[k[7:]] = v
            # single_dict[k] = v
        model.load_state_dict(single_dict, True)

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

    # ======================= Loss ============================
    yolo_loss = ComputeLoss().cuda()

    # ======================= metric ============================
    metric = mean_average_precision(np.linspace(0.5, 0.95, 10))

    # ======================= Optimizer ============================
    optimizer = optimizer_fn(model=model, lr=args.learning_rate, weight_decay=5e-4)

    Fit(
        model= model,
        args=args,
        optimizer=optimizer,
        criterion=yolo_loss,
        metric=metric,
        train_loader=train_loader,
        validation_loader=val_loader
    )