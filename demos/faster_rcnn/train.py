import os
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, distributed


from models.faster import Faster_Rcnn
from data_gen import create_dataset
from utils.anchor_generator import get_base_anchor
from cfg._fit import Fit

def dataloader_fn(args):
    data_dict = yaml.safe_load(open(args.data_yaml, 'r'))

    num_classes = data_dict['num_classes']
    category_names = data_dict['categories']
    assert (num_classes == len(category_names)), f"num_classes {num_classes} must equal len(category_names) {len(category_names)}"
    args.num_classes = num_classes

    train_dir = os.path.join(data_dict['data_root'], data_dict['train_dir'])
    train_dataset = create_dataset(train_dir, input_size=args.input_size, mode='train')
    train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=args.num_workers,
                collate_fn=train_dataset.collate_fn,
            )

    val_dir = os.path.join(data_dict['data_root'], data_dict['train_dir'])
    val_dataset = create_dataset(val_dir, input_size=args.input_size, mode='val')
    val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=args.num_workers,
                collate_fn=train_dataset.collate_fn,
            )

    return train_loader, val_loader, data_dict

def anchor_fn(scales, ratios):
    base_anchors = get_base_anchor(scales=scales, ratios=ratios)
    base_anchors = torch.from_numpy(base_anchors)  # torch.Size([9, 2]) [w, h]

    return base_anchors

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

    model = model.cuda()
    model = nn.DataParallel(model)

    for name, value in model.named_parameters():
        value.requires_grad = True

    return model

def optimizer_fn(model, args):

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    optimizer = torch.optim.SGD(g1, lr=args.init_lr, momentum=0.937, nesterov=True)
    # optimizer = torch.optim.SGD(g0, lr=args.init_lr, momentum=0.937, nesterov=True)
    # optimizer = torch.optim.Adam(g0, lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum

    # optimizer.add_param_group({'params': g1, 'weight_decay': 5e-4})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2, 'lr':args.init_lr})  # add g2 (biases)

    return optimizer

def train(args):

    # ======================= Data Loader ============================
    train_loader, val_loader, data_dict = dataloader_fn(args)

    # ======================= Anchor Generator ============================
    base_anchors = anchor_fn(args.scales, args.ratios)

    # ======================= Model ============================
    num_classes = data_dict['num_classes']
    model = model_fn(args, base_anchors)

    # ======================= metric ============================
    # metric = mean_average_precision(np.linspace(0.5, 0.95, 10))

    # ======================= Optimizer ============================
    optimizer = optimizer_fn(model=model, args=args)

    Fit(
        model= model,
        args=args,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader
    )


