import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5),
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class RPN(nn.Module):

    def __init__(self,
                        training=False,
                        base_anchors=None,

                        backbone_stride=16,
                        in_channels=512,

                        rpn_pre_nms_top_n=2000,
                        rpn_post_nms_top_n=2000,
                        rpn_nms_thresh=0.7,

                        rpn_positive_iou_thres=0.7,
                        rpn_negative_iou_thres=0.3,
                        rpn_positives_per_image=128,
                        rpn_negatives_per_image=128,
                 ):
        super(RPN, self).__init__()

        self.training = training

        # anchors
        self.base_anchors = base_anchors / backbone_stride # torch.Size([9, 2])
        num_anchors = base_anchors.size(0)

        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.rpn_nms_thresh = rpn_nms_thresh
        self.rpn_positive_iou_thres = rpn_positive_iou_thres
        self.rpn_negative_iou_thres = rpn_negative_iou_thres
        self.rpn_positives_per_image = rpn_positives_per_image
        self.rpn_negatives_per_image = rpn_negatives_per_image

        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)

        self.classifier = nn.Conv2d(in_channels=in_channels, out_channels=num_anchors * 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.regressor = nn.Conv2d(in_channels=in_channels, out_channels=num_anchors * 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

        self.focal_loss = FocalLoss(class_num=2)

    def dxdydwdh2xywh(self, dxdydwdh, anchor_xywh):
        xywh = dxdydwdh.clone()

        xywh[..., 0] = dxdydwdh[..., 0] * anchor_xywh[..., 2] + anchor_xywh[..., 0]
        xywh[..., 1] = dxdydwdh[..., 1] * anchor_xywh[..., 3] + anchor_xywh[..., 1]
        xywh[..., 2] = torch.exp(dxdydwdh[..., 2]) * anchor_xywh[..., 2]
        xywh[..., 3] = torch.exp(dxdydwdh[..., 2]) * anchor_xywh[..., 3]

        return xywh

    def xywh2dxdydwdh(self, target_xywh, anchor_xywh, eps=1e-7):
        dxdydwdh = target_xywh.clone()

        dxdydwdh[..., 0] = (target_xywh[..., 0] - anchor_xywh[..., 0]) / anchor_xywh[..., 2]
        dxdydwdh[..., 1] = (target_xywh[..., 1] - anchor_xywh[..., 1]) / anchor_xywh[..., 3]
        dxdydwdh[..., 2] = torch.log(target_xywh[..., 2] / anchor_xywh[..., 2] + eps)
        dxdydwdh[..., 3] = torch.log(target_xywh[..., 3] / anchor_xywh[..., 3] + eps)

        return dxdydwdh

    def xywh2xyxy(self, xywh):
        xyxy = xywh.clone()
        xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2  # top left x
        xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2  # top left y
        xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2  # bottom right x
        xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2  # bottom right y
        return xyxy

    def xyxy2xywh(self, xyxy):
        xywh = xyxy.clone()
        xywh[..., 0] = (xyxy[..., 0] + xyxy[..., 2]) / 2
        xywh[..., 1] = (xyxy[..., 1] + xyxy[..., 3]) / 2
        xywh[..., 2] = xyxy[..., 2] - xyxy[..., 0]
        xywh[..., 3] = xyxy[..., 3] - xyxy[..., 1]
        return xywh

    def grid(self, height, width, mode='xy'):
        ys = torch.arange(0, height)
        xs = torch.arange(0, width)

        offset_x, offset_y = torch.meshgrid(xs, ys)
        offset_yx = torch.stack([offset_x, offset_y]).permute(1, 2, 0)

        if mode == 'xy':
            offset_xy = offset_yx.permute(1, 0, 2)
            return offset_xy

        return offset_yx

    def make_anchors_xywh(self, feature_height, feature_width, device):
        anchors_wh = self.base_anchors.repeat(1, feature_height, feature_width, 1, 1).to(device=device) # torch.Size([1, 14, 14, 9, 2])
        anchors_xy = self.grid(feature_height, feature_width, mode='xy').to(device=device).unsqueeze(0).unsqueeze(3).expand_as(anchors_wh)

        anchor_xywh = torch.cat([anchors_xy, anchors_wh], dim=4)

        return anchor_xywh

    def filter_proposals(self, cls, dxdydwdh, anchor_xywh, feature_height, feature_width):
        proposal_dxdydwdh = dxdydwdh.clone() # torch.Size([2, 14, 14, 9, 4])
        proposal_cls = cls.clone() # torch.Size([2, 14, 14, 9, 2])

        proposal_xywh = self.dxdydwdh2xywh(proposal_dxdydwdh, anchor_xywh) # torch.Size([2, 14, 14, 9, 4])
        proposal_cls = torch.softmax(proposal_cls.clone(), dim=4) # torch.Size([2, 14, 14, 9, 2])

        proposal_score = proposal_cls[..., 1] # torch.Size([2, 14, 14, 9, 1]) 0 for bg and 1 for fg

        proposals = torch.cat([proposal_score[..., None], proposal_xywh], dim=4).detach() # torch.Size([2, 14, 14, 9, 5]) cls, xywh
        proposals = proposals.view(proposals.size(0), -1, 5) # torch.Size([2, 1764, 5]) # score, xywh

        # 裁剪边界
        proposals[..., 1:] = self.xywh2xyxy(proposals[..., 1:]) # score, xyxy
        proposals[..., 1] = proposals[..., 1].clamp(min=0, max=feature_width - 1)
        proposals[..., 2] = proposals[..., 2].clamp(min=0, max=feature_height - 1)
        proposals[..., 3] = proposals[..., 3].clamp(min=0, max=feature_width - 1)
        proposals[..., 4] = proposals[..., 4].clamp(min=0, max=feature_height - 1)

        # 每个图像bs单独自己一个做nms，得到维度可能不相等的 proposals
        batch_proposals = []
        for batch_idx in range(proposals.size(0)):
            proposal = proposals[batch_idx, ...] # torch.Size([1764, 6])

            # 按照置信度进行排序，挑选出 rpn_pre_nms_top_n 个proposals，
            rpn_pre_nms_top_n = min(self.rpn_pre_nms_top_n, proposal.size(0))
            _, top_k_idx = proposal[..., 0].topk(rpn_pre_nms_top_n, dim=-1) # torch.Size([1764])
            proposal = proposal[top_k_idx, ...]

            # nms
            nms_idx = torchvision.ops.nms(proposal[..., 1:], proposal[..., 0], self.rpn_nms_thresh)  # torch.Size([246])

            # 挑选出 rpn_post_nms_top_n 个 proposals
            rpn_post_nms_top_n = min(self.rpn_post_nms_top_n, nms_idx.size(0))
            nms_idx = nms_idx[:rpn_post_nms_top_n]
            proposal = proposal[nms_idx, ...]

            # 把 xyxy 转换成 xywh
            proposal[..., 1:] = self.xyxy2xywh(proposal[..., 1:])
            batch_proposals.append(proposal[..., 1:]) # xywh
        return batch_proposals

    def batch_iou(self, xywh1, xywh2, eps=1e-7):
        xyxy1 = self.xywh2xyxy(xywh1)
        xyxy2 = self.xywh2xyxy(xywh2)

        area1 = (xyxy1[:, 2] - xyxy1[:, 0]) * (xyxy1[:, 3] - xyxy1[:, 1])
        area2 = (xyxy2[:, 2] - xyxy2[:, 0]) * (xyxy2[:, 3] - xyxy2[:, 1])

        try:
            inter = (torch.minimum(xyxy1[:, None, 2], xyxy2[:, 2]) - torch.maximum(xyxy1[:, None, 0], xyxy2[:, 0])).clamp(0) * (torch.minimum(xyxy1[:, None, 3], xyxy2[:, 3]) - torch.maximum(xyxy1[:, None, 1], xyxy2[:, 1])).clamp(0)
        except:
            inter = (torch.min(xyxy1[:, None, 2], xyxy2[:, 2]) - torch.max(xyxy1[:, None, 0], xyxy2[:, 0])).clamp(0) * (torch.min(xyxy1[:, None, 3], xyxy2[:, 3]) - torch.max(xyxy1[:, None, 1], xyxy2[:, 1])).clamp(0)

        union = area1[:, None] + area2 - inter + eps
        iou = inter / union

        return iou

    def computet_loss(self, predict_cls, predict_dxdydwdh, anchor_xywh, targets):
        '''
        :param predict_cls: torch.Size([2, 38, 38, 9, 1])
        :param predict_xywh: torch.Size([2, 38, 38, 9, 4])
        :param predict_dxdydwdh: torch.Size([2, 38, 38, 9, 4])
        :param anchor_xywh: torch.Size([1, 38, 38, 9, 4])
        :param targets: torch.Size([12, 6]) batch_idx, cls_idx, xywh
        :return:
        '''

        predict_all_cls = []
        targets_all_cls = []

        predict_all_box = []
        targets_all_box = []

        device = predict_cls.device
        bs, feature_height, feature_width, num_anchors, _ = predict_cls.size()

        # 过滤掉超过边界的 anchors
        # anchor_xyxy = self.xywh2xyxy(anchor_xywh) # torch.Size([12996, 4])
        # keep = (anchor_xyxy[..., 0] >= 0) & (anchor_xyxy[..., 1] >= 0) & (
        #             anchor_xyxy[..., 2] < feature_width) & (anchor_xyxy[..., 3] < feature_height)
        # anchor_xywh = anchor_xywh[keep, ...].view(-1, 4)
        anchor_xywh = anchor_xywh.view(-1, 4)

        for batch_idx in range(bs):

            batch_cls = predict_cls[batch_idx, ...].reshape(-1, 2) # torch.Size([12996, 2])
            batch_dxdydwdh = predict_dxdydwdh[batch_idx, ...].reshape(-1, 4) # torch.Size([12996, 4])

            batch_target = targets[targets[..., 0] == batch_idx] # torch.Size([13, 6])
            target_xywh = batch_target[..., 2:] * torch.tensor([feature_width, feature_height, feature_width, feature_height]).to(device=device) # # torch.Size([31, 4])

            # 计算 anchor 和 gt bbox 的 iou
            iou_between_anchor_target = self.batch_iou(anchor_xywh, target_xywh) # torch.Size([12996, 22])

            # 匹配 正负忽略 样本
            mask_positive_negative = torch.ones((iou_between_anchor_target.size(0), 2), dtype=torch.long).to(device=device) * -2 # torch.Size([12996, 2])
            mask_positive_negative[..., 0] = torch.arange(mask_positive_negative.size(0)) # 第一列用于记录的索引，第二列用于记录正负样本的标记
            # 如果一个 anchor 与任意一个 gt bbox 的 iou > rpn_positive_iou_thres
            max_iou_values, max_iou_idx = torch.max(iou_between_anchor_target, dim=1) # torch.Size([12996]) torch.Size([12996])
            matches = max_iou_values > self.rpn_positive_iou_thres
            mask_positive_negative[matches, 1] = max_iou_idx[matches]
            # 负样本
            matches = max_iou_values < self.rpn_negative_iou_thres
            mask_positive_negative[matches, 1] = -1
            # 如果和一个 gt 的 iou 最大的 anchor
            max_iou_values, max_iou_idx = torch.max(iou_between_anchor_target, dim=0) # torch.Size([8]) torch.Size([8])
            for target_idx in range(max_iou_idx.size(0)):
                mask_positive_negative[max_iou_idx[target_idx], 1] = target_idx

            # 挑选正负样本
            positives = mask_positive_negative[mask_positive_negative[:, 1] >= 0]
            negatives = mask_positive_negative[mask_positive_negative[:, 1] == -1]

            num_positive = min(positives.size(0), self.rpn_positives_per_image)
            num_negative = min(negatives.size(0), max(self.rpn_negatives_per_image, self.rpn_positives_per_image + self.rpn_negatives_per_image - num_positive))
            sample_positive_idx = torch.randperm(positives.size(0), device=device)[:num_positive] # torch.Size([34])
            sample_negative_idx = torch.randperm(negatives.size(0), device=device)[:num_negative] # torch.Size([222])

            # 把 mask_positive_negative 拆分成 mask_positive 和 mask_negative
            mask_positive = mask_positive_negative[positives[sample_positive_idx, 0], ...] # torch.Size([34, 2])
            mask_negative = mask_positive_negative[negatives[sample_negative_idx, 0], ...] # torch.Size([222, 2])

            # 计算 cls loss 的 预测值和真实值
            predict_negative_cls = batch_cls[mask_negative[..., 0]] # torch.Size([222, 1])
            predict_positive_cls = batch_cls[mask_positive[..., 0]] # torch.Size([34, 1])
            targets_negative_cls = torch.zeros((predict_negative_cls.size(0))).to(predict_negative_cls)
            targets_positive_cls = torch.ones((predict_positive_cls.size(0))).to(predict_positive_cls)
            predict_all_cls.append(torch.cat([predict_negative_cls, predict_positive_cls], dim=0))
            targets_all_cls.append(torch.cat([targets_negative_cls, targets_positive_cls], dim=0))

            # 计算 bbox loss 的 预测值和真实值
            predict_positive_dxdydwdh = batch_dxdydwdh[mask_positive[..., 0]] # torch.Size([34, 4])
            anchors_positive_xywh = anchor_xywh[mask_positive[..., 0]] # torch.Size([34, 4])
            targets_positive_xywh = target_xywh[mask_positive[..., 1]] # torch.Size([34, 4])
            targets_positive_dxdydwdh = self.xywh2dxdydwdh(targets_positive_xywh, anchors_positive_xywh) # torch.Size([34, 4])
            predict_all_box.append(predict_positive_dxdydwdh)
            targets_all_box.append(targets_positive_dxdydwdh)

        # loss cls
        predict_all_cls = torch.cat(predict_all_cls, dim=0)
        targets_all_cls = torch.cat(targets_all_cls, dim=0)
        # loss_cls = F.cross_entropy(predict_all_cls, targets_all_cls.long(), reduction='mean')
        loss_cls = self.focal_loss(predict_all_cls, targets_all_cls.long())

        # loss box
        predict_all_box = torch.cat(predict_all_box, dim=0)
        targets_all_box = torch.cat(targets_all_box, dim=0)
        loss_box = F.smooth_l1_loss(predict_all_box, targets_all_box, reduction='mean')
        # loss_box = F.smooth_l1_loss(predict_all_box, targets_all_box, reduction='sum') / predict_all_cls.size(0)

        return loss_cls, loss_box


    def forward(self, feature_backbone, targets=None):
        bs, c, h, w = feature_backbone.size() # torch.Size([1, 512, 14, 14])
        device = feature_backbone.device

        anchor_xywh = self.make_anchors_xywh(h, w, device) # torch.Size([1, 14, 14, 9, 4])

        # rpn 提取特征
        feature_rpn = F.relu(self.conv3x3(feature_backbone))

        output_cls = self.classifier(feature_rpn) # torch.Size([2, 9, 14, 14])
        output_cls = output_cls.permute(0, 2, 3, 1).view(bs, h, w, -1, 2) # torch.Size([2, 14, 14, 9, 1])

        output_dxdydwdh = self.regressor(feature_rpn) # torch.Size([2, 36, 14, 14])
        output_dxdydwdh = output_dxdydwdh.permute(0, 2, 3, 1).view(bs, h, w, -1, 4) # torch.Size([2, 14, 14, 9, 4])

        # 获得 Proposals，需要注意，Proposal不（能）参与 loss 的计算，因为对边界做了修饰（限制、裁剪）
        proposals = self.filter_proposals(output_cls, output_dxdydwdh, anchor_xywh, h, w) # list [torch.Size([N, 5])] # score, xywh

        # 计算 rpn 的 loss
        if self.training:
            loss_cls, loss_box = self.computet_loss(output_cls, output_dxdydwdh, anchor_xywh, targets)
            return proposals, loss_cls, loss_box
        return proposals
