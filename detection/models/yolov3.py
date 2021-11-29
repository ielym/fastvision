import torch
import torch.nn as nn

from ..tools import offset

class Yolov3(nn.Module):
    def __init__(self, backbone, neck, head, anchors, num_anchors_per_level, in_channels=3, num_classes=80, training=False):
        super(Yolov3, self).__init__()

        self.training = training

        anchors = anchors.view(-1, 2)
        self.anchors_per_level = []
        start_idx = 0
        for idx in range(len(num_anchors_per_level)):
            self.anchors_per_level.append(anchors[start_idx : start_idx + num_anchors_per_level[idx]].view(num_anchors_per_level[idx], 1, 1, 2))
            start_idx += num_anchors_per_level[idx]

        self.num_classes = num_classes

        self.backbone = backbone(in_channels=in_channels, including_top=False)
        self.backbone_strides_per_level = self.backbone.backbone_strides_per_level()
        self.backbone_channels_per_level = self.backbone.backbone_channels_per_level()

        self.neck = neck(feature_channels=self.backbone_channels_per_level)

        self.head = head(feature_channels=self.backbone_channels_per_level, num_levels=len(self.backbone_channels_per_level), num_anchors_per_level=num_anchors_per_level, num_classes=num_classes)

    def forward(self, images):
        backbone_out = self.backbone(images)
        neck_out = self.neck(backbone_out)
        head_out = self.head(neck_out)

        results = []
        if not self.training:
            for i in range(len(head_out)):
                out = head_out[i].sigmoid()
                bs, num_anchors, height, width, _ = out.size()

                offset_level = torch.tensor(offset(height, width, mode='yx')).to(out)
                offset_level = offset_level.expand_as(out[..., 0:2])

                xy = (out[..., 0:2] * 2 - 0.5 + offset_level) * self.backbone_strides_per_level[i]  # xy
                wh = (out[..., 2:4] * 2) ** 2 * self.anchors_per_level[i].expand_as(out[..., 2:4])  # wh

                out = torch.cat((xy, wh, out[..., 4:]), -1)
                results.append(out.view(bs, -1, self.num_classes + 5))
            results = torch.cat(results, 1)
        return head_out if self.training else results


def yolov3(backbone=None, neck=None, head=None, anchors=None, num_anchors_per_level=None, in_channels=3, num_classes=80, training=False):

    if isinstance(backbone, type(None)):
        from fastvision.classfication.models import darknet53
        backbone = darknet53
    if isinstance(backbone, type(None)):
        from ..neck import yolov3neck
        neck = yolov3neck
    if isinstance(backbone, type(None)):
        from ..head import yolov3head
        head = yolov3head

    return Yolov3(backbone=backbone, neck=neck, head=head, anchors=anchors, num_anchors_per_level=num_anchors_per_level, in_channels=in_channels, num_classes=num_classes, training=training)
