import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=False):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

def conv1x1(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=False):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

def normalization(num_features):
    return nn.BatchNorm2d(num_features=num_features)

def activation():
    # return nn.LeakyReLU(inplace=True, negative_slope=0.1)
    return nn.SiLU()

class ConvBlock3x3(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=False):
        super(ConvBlock3x3, self).__init__()

        self.conv = conv3x3(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = normalization(out_channels)
        self.relu = activation()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvBlock1x1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, bias=False):
        super(ConvBlock1x1, self).__init__()

        self.conv = conv1x1(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = normalization(out_channels)
        self.relu = activation()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Yolov3Head(nn.Module):
    def __init__(self, feature_channels, num_levels, num_anchors_per_level, num_classes): # [1024, 512, 256]
        super(Yolov3Head, self).__init__()

        self.num_levels = num_levels
        self.num_anchors_per_level = num_anchors_per_level
        self.out_channels = num_classes + 5

        self.heads = nn.ModuleList(conv1x1(in_channels=channels, out_channels=self.out_channels * num_anchors, kernel_size=(1, 1), bias=True) for channels, num_anchors in zip(feature_channels, num_anchors_per_level))

    def forward(self, features:list):
        '''
        :param features: By Feature size : small, middle, large; By Detect: large, middle, small
        list(torch.Size([1, 1024, 13, 13]) torch.Size([1, 512, 26, 26]) torch.Size([1, 256, 52, 52]))
        :return:
        '''
        for i in range(self.num_levels):
            # print(features[i])
            # print(features[i].size())
            features[i] = self.heads[i](features[i])  # torch.Size([1, 255, 13, 13]) torch.Size([1, 255, 26, 26]) torch.Size([1, 255, 52, 52])
            bs, _, height, width = features[i].size()
            features[i] = features[i].view(bs, self.num_anchors_per_level[i], self.out_channels, height, width).permute(0, 1, 3, 4, 2).contiguous()
            #torch.Size([1, 3, 13, 13, 85])
            #torch.Size([1, 3, 26, 26, 85])
            #torch.Size([1, 3, 52, 52, 85])
        return features

def yolov3head(feature_channels, num_levels, num_anchors_per_level, num_classes):
    return Yolov3Head(feature_channels, num_levels, num_anchors_per_level, num_classes)