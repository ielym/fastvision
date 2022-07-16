import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=False):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

def normalization(num_features):
    return nn.BatchNorm2d(num_features=num_features)

class VGG(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks, channels, normal=False):
        super(VGG, self).__init__()

        self.in_channles = in_channels
        self.normal = normal

        self.vgg1 = self._make_layer(num_blocks[0], channels[0])
        self.vgg2 = self._make_layer(num_blocks[1], channels[1])
        self.vgg3 = self._make_layer(num_blocks[2], channels[2])
        self.vgg4 = self._make_layer(num_blocks[3], channels[3])
        self.vgg5 = self._make_layer(num_blocks[4], channels[4])

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # To Ensure the classifier receive a vector with fixed length 512 * 7 * 7
        # self.gmp = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
                                            nn.Linear(channels[3] * 7 * 7, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            # nn.Linear(4096, num_classes),
                            )

    def _make_layer(self, num_blocks, channels):

        layers = []
        for _ in range(0, num_blocks):
            layers.append(conv3x3(in_channels=self.in_channles, out_channels=channels, bias=True))
            if self.normal:
                layers.append(normalization(channels))
            layers.append(nn.ReLU(inplace=True))
            self.in_channles = channels
        return nn.Sequential(*layers)

    def forward(self, x):

        vgg1 = self.vgg1(x) # torch.Size([1, 64, 224, 224])
        vgg1 = self.maxpool(vgg1) # torch.Size([1, 64, 112, 112])

        vgg2 = self.vgg2(vgg1) # torch.Size([1, 128, 112, 112])
        vgg2 = self.maxpool(vgg2) # torch.Size([1, 128, 56, 56])

        vgg3 = self.vgg3(vgg2) # torch.Size([1, 256, 56, 56])
        vgg3 = self.maxpool(vgg3) # torch.Size([1, 256, 28, 28])

        vgg4 = self.vgg4(vgg3) # torch.Size([1, 512, 28, 28])
        vgg4 = self.maxpool(vgg4) # torch.Size([1, 512, 14, 14])

        vgg5 = self.vgg5(vgg4) # torch.Size([1, 512, 14, 14])
        # vgg5 = self.maxpool(vgg5) # torch.Size([1, 512, 7, 7])

        # out = self.gmp(vgg5) # torch.Size([1, 512, 7, 7])
        # out = torch.flatten(out, 1) # torch.Size([1, 25088])
        # out = self.classifier(out) # torch.Size([1, 1000])

        return vgg5

def vgg11(in_channels=3, num_classes=1000):
    channels = [64, 128, 256, 512, 512]
    return VGG(in_channels=in_channels, num_classes=num_classes, num_blocks=[1, 1, 2, 2, 2], channels=channels)

def vgg11_bn(in_channels=3, num_classes=1000):
    channels = [64, 128, 256, 512, 512]
    return VGG(in_channels=in_channels, num_classes=num_classes, num_blocks=[1, 1, 2, 2, 2], channels=channels, normal=True)

def vgg13(in_channels=3, num_classes=1000):
    channels = [64, 128, 256, 512, 512]
    return VGG(in_channels=in_channels, num_classes=num_classes, num_blocks=[2, 2, 2, 2, 2], channels=channels)

def vgg13_bn(in_channels=3, num_classes=1000):
    channels = [64, 128, 256, 512, 512]
    return VGG(in_channels=in_channels, num_classes=num_classes, num_blocks=[2, 2, 2, 2, 2], channels=channels, normal=True)

def vgg16(in_channels=3, num_classes=1000):
    channels = [64, 128, 256, 512, 512]
    return VGG(in_channels=in_channels, num_classes=num_classes, num_blocks=[2, 2, 3, 3, 3], channels=channels)

def vgg16_bn(in_channels=3, num_classes=1000):
    channels = [64, 128, 256, 512, 512]
    return VGG(in_channels=in_channels, num_classes=num_classes, num_blocks=[2, 2, 3, 3, 3], channels=channels, normal=True)

def vgg19(in_channels=3, num_classes=1000):
    channels = [64, 128, 256, 512, 512]
    return VGG(in_channels=in_channels, num_classes=num_classes, num_blocks=[2, 2, 4, 4, 4], channels=channels)

def vgg19_bn(in_channels=3, num_classes=1000):
    channels = [64, 128, 256, 512, 512]
    return VGG(in_channels=in_channels, num_classes=num_classes, num_blocks=[2, 2, 4, 4, 4], channels=channels, normal=True)
