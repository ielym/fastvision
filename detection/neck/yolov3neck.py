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
    try:
        return nn.SiLU()
    except:
        from fastvision.layers import SILU
        return SILU()

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

class YoloBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YoloBlock, self).__init__()

        self.conv1 = ConvBlock1x1(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = ConvBlock3x3(in_channels=out_channels, out_channels=out_channels * 2)
        self.conv3 = ConvBlock1x1(in_channels=out_channels * 2, out_channels=out_channels)
        self.conv4 = ConvBlock3x3(in_channels=out_channels, out_channels=out_channels * 2)
        self.conv5 = ConvBlock1x1(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        return conv5

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpSampling, self).__init__()

        self.squeeze = ConvBlock1x1(in_channels=in_channels, out_channels=out_channels)
        self.upsampling = nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, x):
        return self.upsampling(self.squeeze(x))

class Yolov3Neck(nn.Module):
    def __init__(self, feature_channels): # [1024, 512, 256]
        super(Yolov3Neck, self).__init__()

        self.neck1 = YoloBlock(in_channels=feature_channels[0], out_channels=feature_channels[0] // 2)
        self.conv1 = ConvBlock3x3(in_channels=feature_channels[0] // 2, out_channels=feature_channels[0])
        self.up1 = UpSampling(in_channels=feature_channels[0] // 2, out_channels=feature_channels[0] // 4, scale_factor=2)

        self.neck2 = YoloBlock(in_channels=feature_channels[1] + feature_channels[0] // 4, out_channels=feature_channels[1] // 2)
        self.conv2 = ConvBlock3x3(in_channels=feature_channels[1] // 2, out_channels=feature_channels[1])
        self.up2 = UpSampling(in_channels=feature_channels[1] // 2, out_channels=feature_channels[1] // 4, scale_factor=2)

        self.neck3 = YoloBlock(in_channels=feature_channels[2] + feature_channels[1] // 4, out_channels=feature_channels[2] // 2)
        self.conv3 = ConvBlock3x3(in_channels=feature_channels[2] // 2, out_channels=feature_channels[2])


    def forward(self, features:list):
        '''
        :param features: By Feature size : small, middle, large; By Detect: large, middle, small
        list(torch.Size([1, 1024, 13, 13]) torch.Size([1, 512, 26, 26]) torch.Size([1, 256, 52, 52]))
        :return: The same to input features
        '''

        feature_small, feature_middle, feature_large = features # torch.Size([1, 1024, 13, 13]) torch.Size([1, 512, 26, 26]) torch.Size([1, 256, 52, 52])

        small_to_next = self.neck1(feature_small) # torch.Size([1, 512, 13, 13])
        up1 = self.up1(small_to_next) # torch.Size([1, 256, 26, 26])
        small_to_head = self.conv1(small_to_next) # torch.Size([1, 1024, 13, 13])

        middle_cat = torch.cat([up1, feature_middle], dim=1) # torch.Size([1, 768, 26, 26])
        middle_to_next = self.neck2(middle_cat) # torch.Size([1, 256, 26, 26])
        up2 = self.up2(middle_to_next) # torch.Size([1, 128, 52, 52])
        middle_to_head = self.conv2(middle_to_next) # torch.Size([1, 512, 26, 26])

        large_cat = torch.cat([up2, feature_large], dim=1) # torch.Size([1, 384, 52, 52])
        large_to_next = self.neck3(large_cat) # torch.Size([1, 128, 52, 52])
        large_to_head = self.conv3(large_to_next) # torch.Size([1, 256, 52, 52])

        return [small_to_head, middle_to_head, large_to_head]


def yolov3neck(feature_channels):
    return Yolov3Neck(feature_channels)

