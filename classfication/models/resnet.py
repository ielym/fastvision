import torch
import torch.nn as nn
from fastvision.utils import initialize_weights

def conv3x3(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def conv1x1(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def normalization(num_features):
    return nn.BatchNorm2d(num_features=num_features)

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, mid_channels, downsample=False, downsample_stride=(1, 1)):
        super(Bottleneck, self).__init__()

        '''
                 |
                 x ---------
                 |          |
             conv1x1, 64    |
             conv3x3, 64    | (downsample)
             conv1x1, 256   |
                 |          | 
                 |----------
                 |
               x + f(x)
        '''

        self.down = downsample or in_channels != mid_channels * self.expansion

        self.conv1 = conv1x1(in_channels, mid_channels)
        self.bn1 = normalization(mid_channels)

        self.conv2 = conv3x3(mid_channels, mid_channels, stride=downsample_stride)
        self.bn2 = normalization(mid_channels)

        self.conv3 = conv1x1(mid_channels, mid_channels * self.expansion)
        self.bn3 = normalization(mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        if self.down or in_channels != mid_channels * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, mid_channels * self.expansion, stride=downsample_stride),
                normalization(mid_channels * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, mid_channels, downsample=False, downsample_stride=(1, 1)):
        super(BasicBlock, self).__init__()

        '''
                 |
                 x ---------
                 |          |
             conv3x3, 64    | (downsample)
             conv3x3, 64    | 
                 |          | 
                 |----------
                 |
               x + f(x)
        '''

        self.down = downsample

        self.conv1 = conv3x3(in_channels, mid_channels, stride=downsample_stride)
        self.bn1 = normalization(mid_channels)

        self.conv2 = conv3x3(mid_channels, mid_channels * self.expansion)
        self.bn2 = normalization(mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        if self.down:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, mid_channels * self.expansion, stride=downsample_stride),
                normalization(mid_channels * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks, basic_block, including_top=True):
        super(ResNet, self).__init__()

        self.including_top = including_top

        self.planes = 64

        self.conv1 = nn.Sequential(
            conv3x3(in_channels=in_channels, out_channels=self.planes, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            normalization(self.planes),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )

        self.res2 = self._make_layer(basic_block, mid_channels=64, num_blocks=num_blocks[0])
        self.res3 = self._make_layer(basic_block, mid_channels=128, num_blocks=num_blocks[1], downsample=True, downsample_stride=(2, 2))
        self.res4 = self._make_layer(basic_block, mid_channels=256, num_blocks=num_blocks[2], downsample=True, downsample_stride=(2, 2))
        self.res5 = self._make_layer(basic_block, mid_channels=512, num_blocks=num_blocks[3], downsample=True, downsample_stride=(2, 2))

        if self.including_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.planes, num_classes)

        initialize_weights(self)

    def _make_layer(self, basic_block, mid_channels, num_blocks, downsample=False, downsample_stride=(1, 1)):

        layers = []
        layers.append(basic_block(in_channels=self.planes, mid_channels=mid_channels, downsample=downsample, downsample_stride=downsample_stride))
        self.planes = mid_channels * basic_block.expansion
        for _ in range(1, num_blocks):
            layers.append(basic_block(in_channels=self.planes, mid_channels=mid_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = self.conv1(x) # torch.Size([1, 64, 112, 112])
        pool1 = self.pool1(conv1) # torch.Size([1, 64, 56, 56])

        res2 = self.res2(pool1) # torch.Size([1, 256, 56, 56])
        res3 = self.res3(res2) # torch.Size([1, 512, 28, 28])
        res4 = self.res4(res3) # torch.Size([1, 1024, 14, 14])
        res5 = self.res5(res4) # torch.Size([1, 2048, 7, 7])

        if self.including_top:
            out = self.avgpool(res5) # torch.Size([1, 2048, 1, 1])
            out = torch.flatten(out, 1) # torch.Size([1, 2048])
            out = self.fc(out) # torch.Size([1, 1000])
            return out

        return [res5, res4, res3]


def resnet18(in_channels=3, num_classes=1000, including_top=True):
    return ResNet(in_channels=in_channels, num_classes=num_classes, num_blocks=[2, 2, 2, 2], basic_block=BasicBlock, including_top=including_top)

def resnet34(in_channels=3, num_classes=1000, including_top=True):
    return ResNet(in_channels=in_channels, num_classes=num_classes, num_blocks=[3, 4, 6, 3], basic_block=BasicBlock, including_top=including_top)

def resnet50(in_channels=3, num_classes=1000, including_top=True):
    return ResNet(in_channels=in_channels, num_classes=num_classes, num_blocks=[3, 4, 6, 3], basic_block=Bottleneck, including_top=including_top)

def resnet101(in_channels=3, num_classes=1000, including_top=True):
    return ResNet(in_channels=in_channels, num_classes=num_classes, num_blocks=[3, 4, 23, 3], basic_block=Bottleneck, including_top=including_top)

def resnet152(in_channels=3, num_classes=1000, including_top=True):
    return ResNet(in_channels=in_channels, num_classes=num_classes, num_blocks=[3, 8, 36, 3], basic_block=Bottleneck, including_top=including_top)

