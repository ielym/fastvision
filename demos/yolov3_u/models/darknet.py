
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvBlock1x1(in_channels=in_channels, out_channels=mid_channels)
        self.conv2 = ConvBlock3x3(in_channels=mid_channels, out_channels=mid_channels * 2)

    def forward(self, x):
        identity = x

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        return identity + conv2

class Darknet(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks, including_top=True):
        super(Darknet, self).__init__()

        self.including_top = including_top

        self.planes = 32

        self.conv0 = ConvBlock3x3(in_channels=in_channels, out_channels=self.planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv1 = ConvBlock3x3(in_channels=self.planes, out_channels=self.planes * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.planes *= 2
        self.res1 = self._make_layer(num_blocks=num_blocks[0], mid_channels=32)

        self.conv2 = ConvBlock3x3(in_channels=self.planes, out_channels=self.planes * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.planes *= 2
        self.res2 = self._make_layer(num_blocks=num_blocks[1], mid_channels=64)

        self.conv3 = ConvBlock3x3(in_channels=self.planes, out_channels=self.planes * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.planes *= 2
        self.res3 = self._make_layer(num_blocks=num_blocks[2], mid_channels=128)

        self.conv4 = ConvBlock3x3(in_channels=self.planes, out_channels=self.planes * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.planes *= 2
        self.res4 = self._make_layer(num_blocks=num_blocks[3], mid_channels=256)

        self.conv5 = ConvBlock3x3(in_channels=self.planes, out_channels=self.planes * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.planes *= 2
        self.res5 = self._make_layer(num_blocks=num_blocks[4], mid_channels=512)

        if self.including_top:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.planes, num_classes)

    def _make_layer(self, num_blocks, mid_channels):

        layers = []
        for _ in range(0, num_blocks):
            layers.append(ResidualBlock(in_channels=self.planes, mid_channels=mid_channels))
        return nn.Sequential(*layers)

    def backbone_strides_per_level(self):
        return [32, 16, 8]

    def backbone_channels_per_level(self):
        return [1024, 512, 256]

    def forward(self, x):

        conv0 = self.conv0(x) # torch.Size([1, 32, 256, 256])

        conv1 = self.conv1(conv0) # torch.Size([1, 64, 128, 128])
        res1 = self.res1(conv1) # torch.Size([1, 64, 128, 128])

        conv2 = self.conv2(res1) # torch.Size([1, 128, 64, 64])
        res2 = self.res2(conv2) # torch.Size([1, 128, 64, 64])

        conv3 = self.conv3(res2) # torch.Size([1, 256, 32, 32])
        res3 = self.res3(conv3) # torch.Size([1, 256, 32, 32])

        conv4 = self.conv4(res3) # torch.Size([1, 512, 16, 16])
        res4 = self.res4(conv4) # torch.Size([1, 512, 16, 16])

        conv5 = self.conv5(res4) # torch.Size([1, 1024, 8, 8])
        res5 = self.res5(conv5) # torch.Size([1, 1024, 8, 8])

        if self.including_top:
            out = self.gap(res5)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out

        return [res5, res4, res3]


def darknet53(in_channels=3, num_classes=1000, including_top=True):
    return Darknet(in_channels=in_channels, num_classes=num_classes, num_blocks=[1, 2, 8, 8, 4], including_top=including_top)