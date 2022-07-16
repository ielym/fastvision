import torch
import torch.nn as nn
import torch.functional as F

from .vgg import vgg16
from .rpn import RPN
from .fast import Fast

class Faster_Rcnn(nn.Module):

    def __init__(self,
                        training = False,
                        in_channels=3,
                        num_classes=80,

                        base_anchors=None,

                        backbone_stride=16,
                        backbone_output_channels=512,
                        backbone_weights = '',

                        rpn_positive_iou_thres=0.7,
                        rpn_negative_iou_thres=0.3,
                        rpn_positives_per_image=128,
                        rpn_negatives_per_image=128,
                        rpn_pre_nms_top_n=2000,
                        rpn_post_nms_top_n=2000,
                        rpn_nms_thresh=0.7,

                        fast_multi_reg_head=False,
                        fast_positive_iou_thres=0.5,
                        fast_negative_iou_thres=0.5,
                        fast_positives_per_image=16,
                        fast_negatives_per_image=48,
                        fast_roi_pool=7,
                 ):
        super(Faster_Rcnn, self).__init__()

        self.training = training

        # backbone
        self.backbone = vgg16(in_channels=in_channels)
        if backbone_weights:
            pretrained_backbone = torch.load(backbone_weights)
            matched_weights = {}
            not_matched_keys = []
            for k, v in pretrained_backbone.items():
                if k in self.backbone.state_dict().keys() and v.size() == pretrained_backbone[k].size():
                    matched_weights[k] = pretrained_backbone[k]
                else:
                    not_matched_keys.append(k)
            self.backbone.load_state_dict(matched_weights, True)
            print('Backbone not matched keys : ', not_matched_keys)

        # rpn
        self.rpn_positive_iou_thres = rpn_positive_iou_thres
        self.rpn_negative_iou_thres = rpn_negative_iou_thres
        self.rpn = RPN(
                        training=training,
                        base_anchors=base_anchors,

                        backbone_stride=backbone_stride,
                        in_channels=backbone_output_channels,

                        rpn_pre_nms_top_n=rpn_pre_nms_top_n,
                        rpn_post_nms_top_n=rpn_post_nms_top_n,
                        rpn_nms_thresh=rpn_nms_thresh,

                        rpn_positive_iou_thres = rpn_positive_iou_thres,
                        rpn_negative_iou_thres = rpn_negative_iou_thres,
                        rpn_positives_per_image = rpn_positives_per_image,
                        rpn_negatives_per_image = rpn_negatives_per_image,
                    )

        # fast
        self.fast = Fast(
                        training=training,
                        fast_multi_reg_head=fast_multi_reg_head,

                        module_after_roi= self.backbone.classifier,

                        in_channels=backbone_output_channels,
                        num_classes=num_classes,

                        fast_positive_iou_thres=fast_positive_iou_thres,
                        fast_negative_iou_thres=fast_negative_iou_thres,
                        fast_positives_per_image=fast_positives_per_image,
                        fast_negatives_per_image=fast_negatives_per_image,
                        fast_roi_pool=fast_roi_pool,
                    )


    def forward(self, images, targets=None):
        # batch_idx, cls_idx, xywh [N, 6]

        feature_backbone = self.backbone(images) # torch.Size([1, 512, 14, 14])

        if self.training:
            # proposals : list : score, xywh -> feature-size
            proposals, loss_rpn_cls, loss_rpn_box = self.rpn(feature_backbone, targets)
            loss_fast_cls, loss_fast_box = self.fast(feature_backbone, proposals, targets)
            return proposals, loss_rpn_cls, loss_rpn_box, loss_fast_cls, loss_fast_box
        else:
            # proposals : list : score, xywh -> feature-size
            proposals = self.rpn(feature_backbone)
            predicts = self.fast(feature_backbone, proposals)
            return predicts



