import os
import yaml

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from fastvision.datasets.detection_dataloader import create_dataloader, show_dataset
from fastvision.detection.tools import AnchorGenerator
from fastvision.utils.checkpoints import LoadFromSingle, LoadFromParrel
from fastvision.loss import Yolov3Loss
from fastvision.train import Fit

def dataloader_fn(data_yaml, batch_size, num_workers, input_size, max_det, device, cache, use_cache):
    data_dict = yaml.safe_load(open(data_yaml, 'r'))

    num_classes = data_dict['num_classes']
    category_names = data_dict['categories']
    assert (num_classes == len(category_names)), f"num_classes {num_classes} must equal len(category_names) {len(category_names)}"

    train_dir = os.path.join(data_dict['data_root'], data_dict['train_dir'])
    train_loader = create_dataloader(prefix='train', data_dir=train_dir, batch_size=batch_size, input_size=input_size, num_workers=num_workers, device=device, cache=cache, use_cache=use_cache, shuffle=True, pin_memory=False, drop_last=False, max_det=max_det)

    val_dir = os.path.join(data_dict['data_root'], data_dict['val_dir'])
    val_loader = create_dataloader(prefix='val', data_dir=val_dir, batch_size=batch_size, input_size=input_size, num_workers=num_workers, device=device, cache=cache, use_cache=use_cache, shuffle=True, pin_memory=False, drop_last=False, max_det=max_det)

    # show_dataset(prefix='train', data_dir=train_dir, category_names=category_names, num_workers=num_workers, cache=cache, use_cache=use_cache)

    return train_loader, val_loader, data_dict

def anchor_fn(data_loaders:list, num_anchors:int, num_workers, device, cache, use_cache):

    anchor_generator = AnchorGenerator(data_loaders=data_loaders, k=num_anchors, iters=100, num_workers=num_workers, cache=cache, use_cache=use_cache, plot=True)

    anchors = anchor_generator.get_anchors()

    anchors = torch.from_numpy(anchors).float().view(-1, 2).to(device=device)

    # anchors = torch.tensor([
    #     [116, 90, 156, 198, 373, 326],
    #     [30, 61, 62, 45, 59, 119],
    #     [10, 13, 16, 30, 33, 23],
    # ], dtype=torch.float).view(-1, 2)

    return anchors

def optimizer_fn(model, lr, weight_decay):

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    optimizer = torch.optim.Adam(g0, lr=lr, betas=(0.937, 0.999))  # adjust beta1 to momentum
    optimizer.add_param_group({'params': g1, 'weight_decay': weight_decay})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)

    return optimizer

def model_fn(weights, in_channels, num_classes, num_anchors_per_level, device, anchors, DataParallel=False, SyncBatchNorm=False, training=True):
    from fastvision.classfication.models import darknet53
    from fastvision.detection.neck import yolov3neck
    from fastvision.detection.head import yolov3head
    from fastvision.detection.models import yolov3

    model = yolov3(backbone=darknet53, neck=yolov3neck, head=yolov3head, anchors=anchors, num_anchors_per_level=num_anchors_per_level, in_channels=in_channels, num_classes=num_classes, training=training)

    if weights:
        model = LoadFromSingle(model=model, weights=weights, strict=False)

    if device.type == 'cuda':
        print('Model : using cuda')
        model = model.cuda()

    if device.type == 'cuda' and DataParallel:
        print('Model : using DataParallel')
        model = nn.DataParallel(model)

    if device.type == 'cuda' and SyncBatchNorm:
        print('Model : using SyncBatchNorm')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    for name, value in model.named_parameters():
        value.requires_grad = False

    for name, value in model.named_parameters():
        if 'neck' in name:
            value.requires_grad = True
        if 'head' in name:
            value.requires_grad = True

    model.half().float()
    return model

def Train(args, device):

    # ======================= Data Loader ============================
    train_loader, val_loader, data_dict = dataloader_fn(data_yaml=args.data_yaml, batch_size=args.batch_size, num_workers=args.num_workers, input_size=args.input_size, max_det=args.max_det, device=device, cache=args.cache_dir, use_cache=args.use_data_cache)

    # ======================= Anchor Generator ============================
    anchors = anchor_fn(data_loaders=[train_loader, val_loader], num_anchors=sum(args.num_anchors_per_level), num_workers=args.num_workers, device=device, cache=args.cache_dir, use_cache=args.use_anchor_cache)

    # ======================= Model ============================
    num_classes = data_dict['num_classes']
    model = model_fn(args.pretrained_weights, args.in_channels, num_classes, args.num_anchors_per_level, anchors=anchors, training=args.training, device=device)

    # ======================= Loss ============================
    loss = Yolov3Loss(model=model, iou_negative_thres=args.iou_negative_thres, ratio_box=args.ratio_box, ratio_conf=args.ratio_conf, ratio_cls=args.ratio_cls)

    # ======================= Optimizer ============================
    optimizer = optimizer_fn(model=model, lr=args.learning_rate, weight_decay=args.weight_decay)

    est = Fit(
                model=model,
                optimizer=optimizer,
                loss=loss,
                start_epoch=0,
                end_epoch=args.epochs,

                device = device,

                train_loader=train_loader,
                val_loader=val_loader,
                data_dict=data_dict,
        )

    est.run_epoches()