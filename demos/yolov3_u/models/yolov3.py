import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import darknet53

def conv3x3(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=False):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

def conv1x1(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=False):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

def normalization(num_features):
    return nn.BatchNorm2d(num_features=num_features)

def activation():
    return nn.SiLU()

class ConvBlock3x3(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=False):
        super(ConvBlock3x3, self).__init__()

        self.conv = conv3x3(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = normalization(out_channels)
        self.act = activation()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ConvBlock1x1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=False):
        super(ConvBlock1x1, self).__init__()

        self.conv = conv1x1(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = normalization(out_channels)
        self.act = activation()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class NeckV3(nn.Module):

    def __init__(self, in_channels_small, in_channels_medium, in_channels_large):
        super(NeckV3, self).__init__()

        self.neck_small = nn.Sequential( # 1024 * 8 * 8
            ConvBlock1x1(in_channels=in_channels_small, out_channels=in_channels_small // 2),
            ConvBlock3x3(in_channels=in_channels_small // 2, out_channels=in_channels_small),
            ConvBlock1x1(in_channels=in_channels_small, out_channels=in_channels_small // 2),
            ConvBlock3x3(in_channels=in_channels_small // 2, out_channels=in_channels_small),
            ConvBlock1x1(in_channels=in_channels_small, out_channels=in_channels_small // 2),
        )
        self.neck_out_small = ConvBlock3x3(in_channels=in_channels_small // 2, out_channels=in_channels_small)

        self.up_sampling_small = nn.Sequential(
            ConvBlock1x1(in_channels=in_channels_small // 2, out_channels=in_channels_small // 4),
            nn.Upsample(None, 2, 'nearest'),
        )
        self.neck_medium = nn.Sequential( # 512, 14, 14
            ConvBlock1x1(in_channels=in_channels_medium + in_channels_small // 4, out_channels=in_channels_medium // 2),
            ConvBlock3x3(in_channels=in_channels_medium // 2, out_channels=in_channels_medium),
            ConvBlock1x1(in_channels=in_channels_medium, out_channels=in_channels_medium // 2),
            ConvBlock3x3(in_channels=in_channels_medium // 2, out_channels=in_channels_medium),
            ConvBlock1x1(in_channels=in_channels_medium, out_channels=in_channels_medium // 2),
        )
        self.neck_out_medium = ConvBlock3x3(in_channels=in_channels_medium // 2, out_channels=in_channels_medium)

        self.up_sampling_medium = nn.Sequential(
            ConvBlock1x1(in_channels=in_channels_medium // 2, out_channels=in_channels_medium // 4),
            nn.Upsample(None, 2, 'nearest'),
        )
        self.neck_large = nn.Sequential( # 256, 28, 28
            ConvBlock1x1(in_channels=in_channels_large + in_channels_medium // 4, out_channels=in_channels_large // 2),
            ConvBlock3x3(in_channels=in_channels_large // 2, out_channels=in_channels_large),
            ConvBlock1x1(in_channels=in_channels_large, out_channels=in_channels_large // 2),
            ConvBlock3x3(in_channels=in_channels_large // 2, out_channels=in_channels_large),
            ConvBlock1x1(in_channels=in_channels_large, out_channels=in_channels_large // 2),
        )
        self.neck_out_large = ConvBlock3x3(in_channels=in_channels_large // 2, out_channels=in_channels_large)


    def forward(self, x_small, x_medium, x_large):
        '''
        :param x_small:  torch.Size([1, 1024, 7, 7])
        :param x_medium:  torch.Size([1, 512, 14, 14])
        :param x_large:  torch.Size([1, 256, 28, 28])
        :return:
        '''

        neck_small = self.neck_small(x_small)
        neck_out_small = self.neck_out_small(neck_small)

        up_sampling_small = self.up_sampling_small(neck_small)
        neck_medium = self.neck_medium(torch.cat([x_medium, up_sampling_small], dim=1))
        neck_out_medium = self.neck_out_medium(neck_medium)

        up_sampling_medium = self.up_sampling_medium(neck_medium)
        neck_large = self.neck_large(torch.cat([x_large, up_sampling_medium], dim=1))
        neck_out_large = self.neck_out_large(neck_large)

        return neck_out_small, neck_out_medium, neck_out_large

class HeadV3(nn.Module):
    def __init__(self, in_channels_small, in_channels_medium, in_channels_large, anchors, num_classes):
        super(HeadV3, self).__init__()
        '''
            in_channels_small : torch.Size([1, 1024, 7, 7])
            in_channels_medium : torch.Size([1, 512, 14, 14])
            in_channels_large : torch.Size([1, 256, 28, 28])
            anchors : (anchors_small, anchors_medium, anchors_large)
        '''

        self.anchors_small = anchors[0] # torch.Size([3, 2])
        self.anchors_medium = anchors[1] # torch.Size([3, 2])
        self.anchors_large = anchors[2] # torch.Size([3, 2])

        self.head_out_large = nn.Conv2d(in_channels_large, self.anchors_large.size(0)*(5+num_classes), (1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.head_out_medium = nn.Conv2d(in_channels_medium, self.anchors_medium.size(0)*(5+num_classes), (1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.head_out_small = nn.Conv2d(in_channels_small, self.anchors_small.size(0)*(5+num_classes), (1, 1), stride=(1, 1), padding=(0, 0), bias=True)

    def forward(self, x_small, x_medium, x_large):
        '''
        :param x_small:  torch.Size([1, 1024, 7, 7])
        :param x_medium:  torch.Size([1, 512, 14, 14])
        :param x_large:  torch.Size([1, 256, 28, 28])
        :return:
        '''

        head_out_small = self.head_out_small(x_small)
        head_out_medium = self.head_out_medium(x_medium)
        head_out_large = self.head_out_large(x_large)

        return head_out_small, head_out_medium, head_out_large

class YoloV3(nn.Module):

    def __init__(self, in_channels=3, num_classes=80, anchors=(), backbone_weights=None):
        super(YoloV3, self).__init__()
        '''
            in_channels_small : torch.Size([1, 1024, 7, 7])
            in_channels_medium : torch.Size([1, 512, 14, 14])
            in_channels_large : torch.Size([1, 256, 28, 28])
            anchors : (anchors_small, anchors_medium, anchors_large)   /32, /16, /8
        '''

        self.anchors = anchors # ( torch.Size([3, 2]), torch.Size([3, 2]), torch.Size([3, 2]) )

        self.backbone = darknet53(in_channels=in_channels, num_classes=num_classes, including_top=False)
        if backbone_weights:
            pretrained_dict = torch.load(backbone_weights)
            single_dict = {}
            for k, v in pretrained_dict.items():
                single_dict[k[7:]] = v
                # single_dict[k] = v
            self.backbone.load_state_dict(single_dict, False)

        self.neck = NeckV3(1024, 512, 256)
        self.head = HeadV3(1024, 512, 256, anchors, num_classes)

    def forward(self, x):

        # torch.Size([1, 1024, 7, 7]) torch.Size([1, 512, 14, 14]) torch.Size([1, 256, 28, 28])
        backbone_small, backbone_medium, backbone_large = self.backbone(x)

        # torch.Size([1, 1024, 7, 7]) torch.Size([1, 512, 14, 14]) torch.Size([1, 256, 28, 28])
        neck_small, neck_medium, neck_large = self.neck(backbone_small, backbone_medium, backbone_large)

        # torch.Size([1, 255, 7, 7]) torch.Size([1, 255, 14, 14]) torch.Size([1, 255, 28, 28])
        head_small, head_medium, head_large = self.head(neck_small, neck_medium, neck_large)

        return (head_small, head_medium, head_large)

