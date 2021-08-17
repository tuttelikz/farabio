"""AlexNet
Paper: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
Copyright 2021 | farabio
"""
import torch
import torch.nn as nn
from farabio.utils.helpers import get_num_parameters

__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Module):
    def __init__(self, n_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout()

        c1 = self._conv_bn(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2))
        c2 = self._conv_bn(nn.Conv2d(64, 192, kernel_size=5, padding=2))
        c3 = self._conv(nn.Conv2d(192, 384, kernel_size=3, padding=1))
        c4 = self._conv(nn.Conv2d(384, 256, kernel_size=3, padding=1))
        c5 = self._conv_bn(nn.Conv2d(256, 256, kernel_size=3, padding=1))

        self.conv = nn.Sequential(c1, c2, c3, c4, c5)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        l1 = self._fc(nn.Linear(256*6*6, 4096))
        l2 = self._fc(nn.Linear(4096, 4096))

        self.fc = nn.Sequential(l1, l2)
        self.final = nn.Linear(4096, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.final(x)

        return x

    def _conv_bn(self, conv):
        return nn.Sequential(
            conv,
            self.relu,
            self.maxpool
        )

    def _conv(self, conv):
        return nn.Sequential(
            conv,
            self.relu
        )

    def _fc(self, linear):
        return nn.Sequential(
            self.dropout,
            linear,
            self.relu
        )


def alexnet(**kwargs) -> AlexNet:
    model = AlexNet(**kwargs)
    return model


def test():
    x = torch.randn(1, 3, 224, 224)
    model = alexnet(n_classes=3)
    y = model(x)

    print("Trainable parameters: ", get_num_parameters(model))
    print("in shape: ", x.shape, ", out shape: ", y.shape)

# test()