import torch
import torch.nn as nn
from fastvision.utils import initialize_weights

def activation(inplace=True):
    return nn.ReLU(inplace=inplace)

def normalization(num_features):
    return nn.BatchNorm3d(num_features=num_features)


class Conv3x3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True):
        super(Conv3x3x3, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)

class C3D(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks, channels, including_top=True, normal=False):
        super(C3D, self).__init__()

        self.including_top = including_top
        self.normal = normal

        self.inplanes = in_channels

        self.layer1 = self._make_layer(num_blocks[0], channels[0])
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

        self.layer2 = self._make_layer(num_blocks[1], channels[1])
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

        self.layer3 = self._make_layer(num_blocks[2], channels[2])
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

        self.layer4 = self._make_layer(num_blocks[3], channels[3])
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

        self.layer5 = self._make_layer(num_blocks[4], channels[4])
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        if self.including_top:
            self.gap = nn.AdaptiveAvgPool3d((1, 4, 4))
            self.classifier = nn.Sequential(
                nn.Linear(self.inplanes * 4 * 4, 4096),
                activation(),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                activation(),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )


        self._initialize_weights()

    def _make_layer(self, num_blocks, channels):
        layers = []
        for _ in range(num_blocks):
            layers.append(Conv3x3x3(in_channels=self.inplanes, out_channels=channels))
            if self.normal:
                layers.append(normalization(channels))
            layers.append(activation())
            self.inplanes = channels
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        initialize_weights(self)

    def forward(self, x):
        '''
        :param x: torch.Size([1, 3, 16, 112, 112])
        :return:
        '''

        layer1 = self.layer1(x) # torch.Size([1, 64, 16, 112, 112])
        pool1 = self.pool1(layer1) # torch.Size([1, 64, 16, 56, 56])

        layer2 = self.layer2(pool1) # torch.Size([1, 128, 16, 56, 56])
        pool2 = self.pool2(layer2) # torch.Size([1, 128, 8, 28, 28])

        layer3 = self.layer3(pool2) # torch.Size([1, 256, 8, 28, 28])
        pool3 = self.pool3(layer3) # torch.Size([1, 256, 4, 14, 14])

        layer4 = self.layer4(pool3) # torch.Size([1, 512, 4, 14, 14])
        pool4 = self.pool4(layer4) # torch.Size([1, 512, 2, 7, 7])

        layer5 = self.layer5(pool4) # torch.Size([1, 512, 2, 7, 7])
        pool5 = self.pool5(layer5) # torch.Size([1, 512, 1, 4, 4])

        if self.including_top:
            out = self.gap(pool5)
            out = torch.flatten(out, 1)
            out = self.classifier(out)
            return out

        return pool5

def c3d(in_channels=3, num_classes=1000, including_top=True):
    '''
    pre-trained on Sports-1M with 487 categories
    '''
    channels = [64, 128, 256, 512, 512]
    return C3D(in_channels=in_channels, num_classes=num_classes, num_blocks=[1, 1, 2, 2, 2], channels=channels, including_top=including_top)

def c3d_bn(in_channels=3, num_classes=1000, including_top=True):
    channels = [64, 128, 256, 512, 512]
    return C3D(in_channels=in_channels, num_classes=num_classes, num_blocks=[1, 1, 2, 2, 2], channels=channels, including_top=including_top, normal=True)


# img = torch.zeros([1, 3, 16, 112, 112])
# model = c3d(in_channels=3, num_classes=487, including_top=True)
# out = model(img)
# print(out.size())

# model_state_dict = model.state_dict()
# pretrained_state_dict = torch.load(r'S:\ucf101-caffe.pth')

# print(pretrained_state_dict.keys())
# print(model_state_dict.keys())
#
# print(len(model_state_dict.keys()))
# print(len(pretrained_state_dict.keys()))

# from collections import OrderedDict
# mydict = OrderedDict()
#
# for idx in range(len(model_state_dict.keys())):
#     model_key = list(model_state_dict.keys())[idx]
#     model_value = list(model_state_dict.values())[idx]
#
#     pretrained_key = list(pretrained_state_dict.keys())[idx]
#     pretrained_value = list(pretrained_state_dict.values())[idx]
#
#     mydict[model_key] = pretrained_value
#
# model.load_state_dict(mydict, strict=True)
#
# torch.save(model.state_dict(), './c3d.pth')
