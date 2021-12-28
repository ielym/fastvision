import torch
import torch.nn as nn
from fastvision.utils import initialize_weights

def activation(inplace=True):
    return nn.ReLU(inplace=inplace)

def normalization(num_features):
    return nn.BatchNorm3d(num_features=num_features)


class Conv3x1x1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False):
        super(Conv3x1x1, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)

class Conv1x3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False):
        super(Conv1x3x3, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)

class Conv3x3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False):
        super(Conv3x3x3, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)

class Conv1x1x1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False):
        super(Conv1x1x1, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, mid_channels, downsample=False, downsample_stride=(1, 1, 1), tempral_size=1):
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

        if tempral_size == 1:
            self.conv1 = Conv1x1x1(in_channels, mid_channels)
        else:
            self.conv1 = Conv3x1x1(in_channels, mid_channels)
        self.bn1 = normalization(mid_channels)

        self.conv2 = Conv1x3x3(mid_channels, mid_channels, stride=downsample_stride)
        self.bn2 = normalization(mid_channels)

        self.conv3 = Conv1x1x1(mid_channels, mid_channels * self.expansion)
        self.bn3 = normalization(mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        if self.down or in_channels != mid_channels * self.expansion:
            self.downsample = nn.Sequential(
                Conv1x1x1(in_channels, mid_channels * self.expansion, stride=downsample_stride),
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

    def __init__(self, in_channels, mid_channels, downsample=False, downsample_stride=(1, 1), tempral_size=1):
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

        if tempral_size == 1:
            self.conv1 = Conv1x3x3(in_channels, mid_channels, stride=downsample_stride)
        else:
            self.conv1 = Conv3x3x3(in_channels, mid_channels, stride=downsample_stride)
        self.bn1 = normalization(mid_channels)

        self.conv2 = Conv1x3x3(mid_channels, mid_channels * self.expansion)
        self.bn2 = normalization(mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        if self.down:
            self.downsample = nn.Sequential(
                Conv1x1x1(in_channels, mid_channels * self.expansion, stride=downsample_stride),
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

class FastPathway(nn.Module):

    def __init__(self, in_channels, num_blocks, basic_block, alpha, beta):
        super(FastPathway, self).__init__()

        self.planes = int(64 * beta)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=self.planes, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            normalization(self.planes),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.lateral_pool1 = Conv3x1x1(in_channels=self.planes, out_channels=2 * self.planes, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))

        self.res2 = self._make_layer(basic_block, mid_channels=int(64 * beta), num_blocks=num_blocks[0], tempral_size=3)
        self.lateral_res2 = Conv3x1x1(in_channels=self.planes, out_channels=2 * self.planes, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))

        self.res3 = self._make_layer(basic_block, mid_channels=int(128 * beta), num_blocks=num_blocks[1], downsample=True, downsample_stride=(1, 2, 2), tempral_size=3)
        self.lateral_res3 = Conv3x1x1(in_channels=self.planes, out_channels=2 * self.planes, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))

        self.res4 = self._make_layer(basic_block, mid_channels=int(256 * beta), num_blocks=num_blocks[2], downsample=True, downsample_stride=(1, 2, 2), tempral_size=3)
        self.lateral_res4 = Conv3x1x1(in_channels=self.planes, out_channels=2 * self.planes, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))

        self.res5 = self._make_layer(basic_block, mid_channels=int(512 * beta), num_blocks=num_blocks[3], downsample=True, downsample_stride=(1, 2, 2), tempral_size=3)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, basic_block, mid_channels, num_blocks, downsample=False, downsample_stride=(1, 1, 1), tempral_size=3):

        layers = []
        layers.append(basic_block(in_channels=self.planes, mid_channels=mid_channels, downsample=downsample, downsample_stride=downsample_stride, tempral_size=tempral_size))
        self.planes = mid_channels * basic_block.expansion
        for _ in range(1, num_blocks):
            layers.append(basic_block(in_channels=self.planes, mid_channels=mid_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        :param x: torch.Size([1, 3, 32, 224, 224])
        :return:
        '''

        conv1 = self.conv1(x) # torch.Size([1, 8, 32, 112, 112])
        pool1 = self.pool1(conv1) # torch.Size([1, 8, 32, 56, 56])
        lateral_pool1 = self.lateral_pool1(pool1) # torch.Size([1, 16, 4, 56, 56])

        res2 = self.res2(pool1) # torch.Size([1, 32, 32, 56, 56])
        lateral_res2 = self.lateral_res2(res2) # torch.Size([1, 64, 4, 56, 56])

        res3 = self.res3(res2)  # torch.Size([1, 64, 32, 28, 28])
        lateral_res3 = self.lateral_res3(res3) # torch.Size([1, 128, 4, 28, 28])

        res4 = self.res4(res3)  # torch.Size([1, 128, 32, 14, 14])
        lateral_res4 = self.lateral_res4(res4) # torch.Size([1, 256, 4, 14, 14])

        res5 = self.res5(res4)  # torch.Size([1, 256, 32, 7, 7])

        out = self.avgpool(res5) # torch.Size([1, 256, 1, 1, 1])
        out = torch.flatten(out, 1) # torch.Size([1, 256])

        laterals = (lateral_pool1, lateral_res2, lateral_res3, lateral_res4)

        return laterals, out

class SlowPathway(nn.Module):

    def __init__(self, in_channels, num_blocks, basic_block, beta):
        super(SlowPathway, self).__init__()

        self.planes = 64

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=self.planes, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            normalization(self.planes),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.planes = self.planes + 2 * int(beta * self.planes)
        self.res2 = self._make_layer(basic_block, mid_channels=64, num_blocks=num_blocks[0], tempral_size=1)

        self.planes = self.planes + 2 * int(beta * self.planes)
        self.res3 = self._make_layer(basic_block, mid_channels=128, num_blocks=num_blocks[1], downsample=True, downsample_stride=(1, 2, 2), tempral_size=1)
        self.planes = self.planes + 2 * int(beta * self.planes)
        self.res4 = self._make_layer(basic_block, mid_channels=256, num_blocks=num_blocks[2], downsample=True, downsample_stride=(1, 2, 2), tempral_size=3)
        self.planes = self.planes + 2 * int(beta * self.planes)
        self.res5 = self._make_layer(basic_block, mid_channels=512, num_blocks=num_blocks[3], downsample=True, downsample_stride=(1, 2, 2), tempral_size=3)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, basic_block, mid_channels, num_blocks, downsample=False, downsample_stride=(1, 1, 1), tempral_size=3):

        layers = []
        layers.append(basic_block(in_channels=self.planes, mid_channels=mid_channels, downsample=downsample, downsample_stride=downsample_stride, tempral_size=tempral_size))
        self.planes = mid_channels * basic_block.expansion
        for _ in range(1, num_blocks):
            layers.append(basic_block(in_channels=self.planes, mid_channels=mid_channels))
        return nn.Sequential(*layers)

    def forward(self, x, laterals):
        '''
        :param x: torch.Size([1, 3, 4, 224, 224])
        :return:
        '''

        lateral_pool1, lateral_res2, lateral_res3, lateral_res4 = laterals

        conv1 = self.conv1(x) # torch.Size([1, 64, 4, 112, 112])
        pool1 = self.pool1(conv1) # torch.Size([1, 64, 4, 56, 56])
        pool1 = torch.cat([pool1, lateral_pool1], dim=1)

        res2 = self.res2(pool1) # torch.Size([1, 256, 4, 56, 56])
        res2 = torch.cat([res2, lateral_res2], dim=1)

        res3 = self.res3(res2)  # torch.Size([1, 512, 4, 28, 28])
        res3 = torch.cat([res3, lateral_res3], dim=1)

        res4 = self.res4(res3)  # torch.Size([1, 1024, 4, 14, 14])
        res4 = torch.cat([res4, lateral_res4], dim=1)

        res5 = self.res5(res4)  # torch.Size([1, 2048, 4, 7, 7])

        out = self.avgpool(res5) # torch.Size([1, 2048, 1, 1, 1])
        out = torch.flatten(out, 1) # torch.Size([1, 2048])

        return out


class SlowFast(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks, basic_block, alpha, beta):
        super(SlowFast, self).__init__()

        self.fast_pathway = FastPathway(in_channels=in_channels, num_blocks=num_blocks, basic_block=basic_block, alpha=alpha, beta=beta)
        self.slow_pathway = SlowPathway(in_channels=in_channels, num_blocks=num_blocks, basic_block=basic_block, beta=beta)

        self.fc = nn.Linear(2048 + int(2048 * beta), num_classes)

    def forward(self, slow_input, fast_input):
        '''
        :param slow_input: torch.Size([1, 3, 4, 224, 224])
        :param fast_input: torch.Size([1, 3, 32, 224, 224])
        :return:
        '''

        laterals, fast_out = self.fast_pathway(fast_input) # torch.Size([1, 256])
        slow_out = self.slow_pathway(slow_input, laterals) # torch.Size([1, 2048])

        out = torch.cat([fast_out, slow_out], dim=1)
        out = torch.flatten(out, 1) # torch.Size([1, 2048])
        out = self.fc(out) # torch.Size([1, 1000])
        return out

def slowfast_resnet50(in_channels=3, num_classes=1000, alpha=8, beta=1/8):
    '''
    :param in_channels:
    :param num_classes:
    :param alpha: = 8 for temporal, the sampling frequency.
    :param beta: = 1/ 8 for channel
    :param including_top:
    :return:
    '''
    return SlowFast(in_channels=in_channels, num_classes=num_classes, num_blocks=[3, 4, 6, 3], basic_block=Bottleneck, alpha=alpha, beta=beta)

def slowfast_resnet18(in_channels=3, num_classes=1000, alpha=8, beta=1/8):
    return SlowFast(in_channels=in_channels, num_classes=num_classes, num_blocks=[2, 2, 2, 2], basic_block=Bottleneck, alpha=alpha, beta=beta)

def slowfast_resnet34(in_channels=3, num_classes=1000, alpha=8, beta=1/8):
    return SlowFast(in_channels=in_channels, num_classes=num_classes, num_blocks=[3, 4, 6, 3], basic_block=Bottleneck, alpha=alpha, beta=beta)

def slowfast_resnet101(in_channels=3, num_classes=1000, alpha=8, beta=1/8):
    return SlowFast(in_channels=in_channels, num_classes=num_classes, num_blocks=[3, 4, 23, 3], basic_block=Bottleneck, alpha=alpha, beta=beta)

def slowfast_resnet152(in_channels=3, num_classes=1000, alpha=8, beta=1/8):
    return SlowFast(in_channels=in_channels, num_classes=num_classes, num_blocks=[3, 8, 36, 3], basic_block=Bottleneck, alpha=alpha, beta=beta)

# slow_frames = torch.zeros([1, 3, 4, 224, 224])
# fast_frames = torch.zeros([1, 3, 32, 224, 224])
# model = slowfast_resnet152()
#
# out = model(slow_frames, fast_frames)
# print(out.size())
