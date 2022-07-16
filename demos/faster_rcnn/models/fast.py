import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision


class Fast(nn.Module):

    def __init__(self,
                 training = False,
                 fast_multi_reg_head = False,

                 module_after_roi=None,

                 in_channels=512,
                 num_classes=80,

                 fast_positive_iou_thres=0.5,
                 fast_negative_iou_thres=0.5,
                 fast_positives_per_image=16,
                 fast_negatives_per_image=48,
                 fast_roi_pool=7,
                 ):
        super(Fast, self).__init__()

        self.training = training
        self.fast_multi_reg_head = fast_multi_reg_head

        self.fast_positive_iou_thres = fast_positive_iou_thres
        self.fast_negative_iou_thres = fast_negative_iou_thres
        self.fast_positives_per_image = fast_positives_per_image
        self.fast_negatives_per_image = fast_negatives_per_image
        self.fast_roi_pool = fast_roi_pool

        mid_channels = 4096

        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=in_channels * (fast_roi_pool ** 2), out_features=mid_channels),
        #     nn.ReLU(),
        #     nn.Linear(mid_channels, out_features=mid_channels),
        #     nn.ReLU(),
        # )

        self.module_after_roi = module_after_roi

        self.classifier = nn.Linear(in_features=mid_channels, out_features=num_classes + 1)

        if self.fast_multi_reg_head:
            self.regressor = nn.Linear(in_features=mid_channels, out_features=(num_classes + 1) * 4)
        else:
            self.regressor = nn.Linear(in_features=mid_channels, out_features=4)

    def xywh2xyxy(self, xywh):
        xyxy = xywh.clone()
        xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2  # top left x
        xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2  # top left y
        xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2  # bottom right x
        xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2  # bottom right y
        return xyxy

    def batch_iou(self, xywh1, xywh2, eps=1e-7):
        xyxy1 = self.xywh2xyxy(xywh1)
        xyxy2 = self.xywh2xyxy(xywh2)

        area1 = (xyxy1[:, 2] - xyxy1[:, 0]) * (xyxy1[:, 3] - xyxy1[:, 1])
        area2 = (xyxy2[:, 2] - xyxy2[:, 0]) * (xyxy2[:, 3] - xyxy2[:, 1])

        try:
            inter = (torch.minimum(xyxy1[:, None, 2], xyxy2[:, 2]) - torch.maximum(xyxy1[:, None, 0],
                                                                                   xyxy2[:, 0])).clamp(0) * (
                            torch.minimum(xyxy1[:, None, 3], xyxy2[:, 3]) - torch.maximum(xyxy1[:, None, 1],
                                                                                          xyxy2[:, 1])).clamp(0)
        except:
            inter = (torch.min(xyxy1[:, None, 2], xyxy2[:, 2]) - torch.max(xyxy1[:, None, 0], xyxy2[:, 0])).clamp(0) * (
                    torch.min(xyxy1[:, None, 3], xyxy2[:, 3]) - torch.max(xyxy1[:, None, 1], xyxy2[:, 1])).clamp(0)

        union = area1[:, None] + area2 - inter + eps
        iou = inter / union

        return iou

    def xywh2dxdydwdh(self, target_xywh, anchor_xywh, eps=1e-7):
        dxdydwdh = target_xywh.clone()

        dxdydwdh[..., 0] = (target_xywh[..., 0] - anchor_xywh[..., 0]) / anchor_xywh[..., 2]
        dxdydwdh[..., 1] = (target_xywh[..., 1] - anchor_xywh[..., 1]) / anchor_xywh[..., 3]
        dxdydwdh[..., 2] = torch.log(target_xywh[..., 2] / anchor_xywh[..., 2] + eps)
        dxdydwdh[..., 3] = torch.log(target_xywh[..., 3] / anchor_xywh[..., 3] + eps)

        return dxdydwdh

    def dxdydwdh2xywh(self, dxdydwdh, anchor_xywh):
        xywh = dxdydwdh.clone()

        xywh[..., 0] = dxdydwdh[..., 0] * anchor_xywh[..., 2] + anchor_xywh[..., 0]
        xywh[..., 1] = dxdydwdh[..., 1] * anchor_xywh[..., 3] + anchor_xywh[..., 1]
        xywh[..., 2] = torch.exp(dxdydwdh[..., 2]) * anchor_xywh[..., 2]
        xywh[..., 3] = torch.exp(dxdydwdh[..., 2]) * anchor_xywh[..., 3]

        return xywh

    def select_positive_negative_samples(self, proposals, targets, device):

        all_positive = []
        all_negative = []

        bs = len(proposals)
        for batch_idx in range(bs):
            batch_proposals_xywh = proposals[batch_idx]  # torch.Size([595, 5])

            batch_target = targets[targets[..., 0] == batch_idx]  # torch.Size([7, 6])
            batch_target_cls = batch_target[..., 1:2]
            batch_target_xywh = batch_target[..., 2:]

            # 计算 proposals 和 gt bbox 的 iou
            iou_between_proposal_target = self.batch_iou(batch_proposals_xywh,batch_target_xywh)  # torch.Size([12996, 22])

            # 匹配 正负忽略 样本
            mask_positive_negative = torch.ones((iou_between_proposal_target.size(0), 2), dtype=torch.long).to(
                device=device) * -2  # torch.Size([12996, 2])
            mask_positive_negative[..., 0] = torch.arange(mask_positive_negative.size(0))  # 第一列用于记录的索引，第二列用于记录正负样本的标记
            # 如果一个 anchor 与任意一个 gt bbox 的 iou > fast_positive_iou_thres
            max_iou_values, max_iou_idx = torch.max(iou_between_proposal_target,dim=1)  # torch.Size([12996]) torch.Size([12996])
            matches = max_iou_values >= self.fast_positive_iou_thres
            mask_positive_negative[matches, 1] = max_iou_idx[matches]
            # 负样本
            matches = (max_iou_values < self.fast_negative_iou_thres) & (max_iou_values >= 0.1)
            mask_positive_negative[matches, 1] = -1

            # 挑选正负样本
            positives = mask_positive_negative[mask_positive_negative[:, 1] >= 0]
            negatives = mask_positive_negative[mask_positive_negative[:, 1] == -1]
            num_positive = min(positives.size(0), self.fast_positives_per_image)
            num_negative = min(negatives.size(0), max(self.fast_negatives_per_image,
                                                      self.fast_positives_per_image + self.fast_negatives_per_image - num_positive))
            sample_positive_idx = torch.randperm(positives.size(0), device=device)[:num_positive]  # torch.Size([13])
            sample_negative_idx = torch.randperm(negatives.size(0), device=device)[:num_negative]  # torch.Size([51])

            # 把 mask_positive_negative 拆分成 mask_positive 和 mask_negative
            mask_positive = mask_positive_negative[positives[sample_positive_idx, 0], ...]  # torch.Size([13, 2])
            mask_negative = mask_positive_negative[negatives[sample_negative_idx, 0], ...]  # torch.Size([51, 2])

            # 挑选出的正样本的 proposals 的 坐标 及其对应的 标签 txtytwth
            proposal_positive_xywh = batch_proposals_xywh[mask_positive[..., 0]]  # torch.Size([16, 4])
            targets_positive_xywh = batch_target_xywh[mask_positive[..., 1]]  # torch.Size([16, 4])
            targets_positive_txtytwth = self.xywh2dxdydwdh(targets_positive_xywh, proposal_positive_xywh)

            # 挑选出的正样本的 proposals 对应的 标签类别
            targets_positive_cls = batch_target_cls[mask_positive[..., 1]]  # torch.Size([16, 1])

            # 汇总正样本 batch_idx, proposal的 xywh, 标签的 txtytwth, 标签的类别索引
            positive_first_col = torch.ones_like(targets_positive_cls) * batch_idx
            batch_positive_xywh_cls = torch.cat(
                [positive_first_col, proposal_positive_xywh, targets_positive_txtytwth, targets_positive_cls],
                dim=1)  # torch.Size([16, 10])

            # 挑选出的负样本的 proposals 的 坐标
            proposal_negative_xywh = batch_proposals_xywh[mask_negative[..., 0]]  # torch.Size([51, 4])

            # 汇总负样本 batch_idx, proposal 的 xywh
            negative_first_col = torch.ones((proposal_negative_xywh.size(0), 1), device=device) * batch_idx
            batch_negative_xywh_cls = torch.cat([negative_first_col, proposal_negative_xywh],dim=1)  # torch.Size([51, 5])

            all_positive.append(batch_positive_xywh_cls)
            all_negative.append(batch_negative_xywh_cls)

        all_positive = torch.cat(all_positive, dim=0)
        all_negative = torch.cat(all_negative, dim=0)

        return all_positive, all_negative

    def compute_loss(self, predict_positive_cls, predict_negative_cls, predict_positive_dxdydwdh, target_txtytwth, target_cls):
        '''
        :param predict_positive_cls: torch.Size([7, 11])
        :param predict_negative_cls: torch.Size([249, 11])
        :param predict_positive_dxdydwdh: torch.Size([7, 4])
        :param target_txtytwth: torch.Size([7, 4])
        :param target_cls: torch.Size([7, 1])
        :return:
        '''

        if predict_positive_cls.size(0) == 0:
            loss_cls = torch.zeros(1).to(predict_positive_cls)
            loss_box = torch.zeros(1).to(predict_positive_dxdydwdh)
            return loss_cls, loss_box

        # loss box
        mean = torch.FloatTensor((0.0, 0.0, 0.0, 0.0)).to(target_txtytwth).expand_as(target_txtytwth)
        std = torch.FloatTensor((0.1, 0.1, 0.2, 0.2)).to(target_txtytwth).expand_as(target_txtytwth)
        target_txtytwth = (target_txtytwth - mean) / std
        loss_box = F.smooth_l1_loss(predict_positive_dxdydwdh, target_txtytwth, reduction='mean')

        # loss cls
        predict_all_cls = torch.cat([predict_positive_cls, predict_negative_cls], dim=0) # torch.Size([256, 11])
        targets_all_cls = torch.cat([target_cls + 1, torch.zeros((predict_negative_cls.size(0), 1)).to(target_cls)], dim=0)
        loss_cls = F.cross_entropy(predict_all_cls, targets_all_cls.view(-1).long(), reduction='mean')

        return loss_cls, loss_box

    def process_multi_reg_head(self, cls_idx, pred_dxdydwdh):
        pred_dxdydwdh = pred_dxdydwdh.view(pred_dxdydwdh.size(0), -1, 4)
        select_boxes = torch.zeros((pred_dxdydwdh.size(0), 4)).to(pred_dxdydwdh)

        for idx in range(pred_dxdydwdh.size(0)):
            select_boxes[idx, :] = pred_dxdydwdh[idx, cls_idx[idx].long(), :]

        return select_boxes


    def forward(self, feature_backbone, proposals, targets=None):
        bs, c, h, w = feature_backbone.size()
        device = feature_backbone.device

        if self.training:

            targets[..., 2:] = targets[..., 2:] * torch.tensor([w, h, w, h]).to(device=device)  # # torch.Size([31, 4])

            # 选择正负样本
            # positives : batch_idx, proposal_xywh, target_txtytwth, target_cls torch.Size([27, 10])
            # negatives : batch_idx, proposal_xywh torch.Size([229, 5])
            positives, negatives = self.select_positive_negative_samples(proposals, targets, device)
            positives[..., 1:5] = self.xywh2xyxy(positives[..., 1:5])
            negatives[..., 1:5] = self.xywh2xyxy(negatives[..., 1:5])

            # 正样本的 pool  torch.Size([27, 512, 7, 7])
            positive_roi_pool = torchvision.ops.roi_align(feature_backbone, positives[..., 0:5],
                                                          output_size=(self.fast_roi_pool, self.fast_roi_pool))
            # 负样本的 pool  torch.Size([27, 512, 7, 7])
            negative_roi_pool = torchvision.ops.roi_align(feature_backbone, negatives,
                                                          output_size=(self.fast_roi_pool, self.fast_roi_pool))
            # 计算正负样本的分类、回归
            positive_roi_features = torch.flatten(positive_roi_pool, 1)  # torch.Size([32, 25088])
            positive_roi_features = self.module_after_roi(positive_roi_features)
            positive_cls = self.classifier(positive_roi_features)  # torch.Size([32, 11])
            positive_box = self.regressor(positive_roi_features)  # torch.Size([32, 4]) or torch.Size([32, num_classes * 4])

            if self.fast_multi_reg_head:
                positive_box = self.process_multi_reg_head(positives[..., 9:10] + 1, positive_box)

            # 计算负样本的分类
            negative_roi_features = torch.flatten(negative_roi_pool, 1)  # torch.Size([227, 25088])
            negative_roi_features = self.module_after_roi(negative_roi_features)
            negative_cls = self.classifier(negative_roi_features)  # torch.Size([227, 11])

            loss_cls, loss_box = self.compute_loss(positive_cls, negative_cls, positive_box, positives[..., 5:9], positives[..., 9:10])

            return loss_cls, loss_box

        else:
            predicts = []
            for batch_idx in range(bs):
                batch_proposals_xywh = proposals[batch_idx]  # torch.Size([595, 4])

                batch_proposals_xyxy = self.xywh2xyxy(batch_proposals_xywh)
                first_col = torch.ones((batch_proposals_xyxy.size(0), 1), device=device) * batch_idx

                feature_roi_pool = torchvision.ops.roi_align(feature_backbone, torch.cat([first_col, batch_proposals_xyxy], dim=1), output_size=(self.fast_roi_pool, self.fast_roi_pool))

                roi_features = torch.flatten(feature_roi_pool, 1)  # torch.Size([227, 25088])
                roi_features = self.module_after_roi(roi_features)
                predict_cls = self.classifier(roi_features)  # torch.Size([32, 11])
                predict_box = self.regressor(roi_features)  # torch.Size([32, 11 * 4])

                if self.fast_multi_reg_head:
                    _, cls_max_idx = torch.max(predict_cls, dim=1)
                    predict_box = self.process_multi_reg_head(cls_max_idx, predict_box)

                mean = torch.FloatTensor((0.0, 0.0, 0.0, 0.0)).to(predict_box).expand_as(predict_box)
                std = torch.FloatTensor((0.1, 0.1, 0.2, 0.2)).to(predict_box).expand_as(predict_box)
                predict_box = predict_box * std + mean

                # post process
                predict_xywh = self.dxdydwdh2xywh(predict_box, batch_proposals_xywh) # torch.Size([419, 4])
                predict_scores = torch.softmax(predict_cls, dim=1) # torch.Size([419, 81])

                scores, categories = torch.max(predict_scores, dim=1) # torch.Size([419]) torch.Size([419])

                keep = categories > 0

                xywh = predict_xywh[keep]
                scores = scores[keep, None]
                categories = categories[keep, None] - 1 # 0 for bg

                predict = torch.cat([xywh, categories, scores], dim=1) # -1, 6
                predicts.append(predict)
            return predicts