import torch
import torch.nn as nn

from fastvision.detection.tools import wh_iou_batch, wh_iou, cal_iou
from .classification_loss import CrossEntropyLoss, BiCrossEntropyLoss
from .iou_loss import IOULoss, GIOULoss, DIOULoss, CIOULoss

class Yolov3Loss(nn.Module):

    def __init__(self, model, ratio_box, ratio_conf, ratio_cls):
        super(Yolov3Loss, self).__init__()
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        self.anchor_levels = model.anchors_per_level
        self.backbone_stride_levels = model.backbone_strides_per_level

        self.levels = len(self.backbone_stride_levels)

        self.binary_cross_entropy_loss = BiCrossEntropyLoss(reduction='mean')
        self.iou_loss = CIOULoss(reduction='mean')

        self.ratio_box = ratio_box
        self.ratio_conf = ratio_conf
        self.ratio_cls = ratio_cls

    def forward(self, y_pred, y_true):
        '''
        :param y_pred:
        :param y_true:
        :return:
        '''
        gt_locations, gt_categories, gt_xywh, matched_anchors = self.build_target(y_pred, y_true)

        loss_cls, loss_box, loss_conf = torch.zeros(1).to(y_pred[0]), torch.zeros(1).to(y_pred[0]), torch.zeros(1).to(y_pred[0])

        for layer_idx, pre in enumerate(y_pred):

            # torch.Size([9]) torch.Size([9, 2]) torch.Size([9])
            select_target_batch_idx, select_target_grid_xy, select_target_anchor_idx = gt_locations[layer_idx]
            # torch.Size([9, 25])
            predict_corresponding_to_target = pre[select_target_batch_idx, select_target_anchor_idx, select_target_grid_xy[:, 1], select_target_grid_xy[:, 0], ...]
            # torch.Size([9, 2])
            anchors_corresponding_to_target = matched_anchors[layer_idx]

            targets_conf = torch.zeros_like(pre[..., 4:5]) # whether select_target_batch_idx.size(0), conf loss should be computed
            if select_target_batch_idx.size(0):
                predict_categories = predict_corresponding_to_target[..., 5:].sigmoid() # torch.Size([9, 20])
                targets_categories = gt_categories[layer_idx] # torch.Size([9])
                loss_cls += self.binary_cross_entropy_loss(predict_categories, targets_categories, already_sigmoid=True)

                predict_xy = predict_corresponding_to_target[..., 0:2].sigmoid()
                predict_wh = torch.exp(predict_corresponding_to_target[..., 2:4]) * anchors_corresponding_to_target
                predict_xywh = torch.cat([predict_xy, predict_wh], dim=1)
                targets_xywh = gt_xywh[layer_idx]
                loss_box += self.iou_loss(predict_xywh, targets_xywh, mode='xywh')

                iou_between_predict_and_targets = cal_iou(predict_xywh, targets_xywh, mode='xywh')
                targets_conf[select_target_batch_idx, select_target_anchor_idx, select_target_grid_xy[:, 1], select_target_grid_xy[:, 0], ...] = iou_between_predict_and_targets

            predict_conf = pre[..., 4:5].sigmoid()
            loss_conf += self.binary_cross_entropy_loss(predict_conf.view(-1, 1), targets_conf.view(-1, 1), already_sigmoid=True)

        loss_box *= self.ratio_box
        loss_conf *= self.ratio_conf
        loss_cls *= self.ratio_cls

        bs = y_pred[0].size(0)

        return (loss_box + loss_conf + loss_cls) * bs


    def build_target(self, y_pred, y_true):
        '''
        :param y_pred: list -> torch.Size([2, 3, 20, 15, 25]) torch.Size([2, 3, 20, 15, 25]) torch.Size([2, 3, 80, 60, 25]) image: (3, 640, 480)
        :param y_true: torch.Size([4, 6]) [batch_idx, category_idx, x_center, y_center, w, h] normalization size
        :return:
        '''

        gt_locations = []
        gt_categories = []
        gt_xywh = []
        matched_anchors = []

        for layer_idx, pre in enumerate(y_pred):
            anchors = self.anchor_levels[layer_idx].squeeze() # torch.Size([3, 2]) torch.Size([3, 2]) torch.Size([3, 2]) image size
            anchors = anchors / self.backbone_stride_levels[layer_idx] # feature size
            num_anchors = anchors.size(0)

            feature_whwh = torch.tensor(pre.size()).to(pre)[[3, 2, 3, 2]]

            target = y_true.clone() # torch.Size([4, 6])
            target[:, 2:] = y_true[:, 2:] * feature_whwh
            num_targets = target.size(0)

            wh_similarity = target[:, None, 4:] / anchors # torch.Size([4, 3, 2])
            similarity_mask = torch.max(wh_similarity, 1 / wh_similarity).max(2)[0] < 4 # torch.Size([4, 3])

            target_like_anchors = target.unsqueeze(1).repeat(1, num_anchors, 1) # torch.Size([4, 3, 6])
            anchors_idxs = torch.arange(num_anchors).unsqueeze(0).repeat(num_targets, 1).to(pre) # torch.Size([4, 3])
            target_with_anchors = torch.cat([target_like_anchors, anchors_idxs[:, :, None]], dim=2) # torch.Size([4, 3, 7])

            matches_target_anchors = target_with_anchors[similarity_mask, ...] # torch.Size([9, 7]) torch.Size([5, 7]) torch.Size([1, 7])

            select_target_batch_idx = matches_target_anchors[:, 0].long()
            select_target_category_idx = matches_target_anchors[:, 1].long()
            select_target_xy = matches_target_anchors[:, 2:4]
            select_target_wh = matches_target_anchors[:, 4:6]
            select_target_anchor_idx = matches_target_anchors[:, 6].long()

            select_target_grid_xy = torch.floor(select_target_xy).long()
            select_target_offset_xy = select_target_xy - select_target_grid_xy.float()

            select_target_grid_xy[:, 0] = select_target_grid_xy[:, 0].clamp_(0, feature_whwh[0] - 1)
            select_target_grid_xy[:, 1] = select_target_grid_xy[:, 1].clamp_(0, feature_whwh[1] - 1)

            gt_locations.append((select_target_batch_idx, select_target_grid_xy, select_target_anchor_idx))
            gt_categories.append(select_target_category_idx)
            gt_xywh.append(torch.cat([select_target_offset_xy, select_target_wh], dim=1))
            matched_anchors.append(anchors[select_target_anchor_idx])

        return gt_locations, gt_categories, gt_xywh, matched_anchors