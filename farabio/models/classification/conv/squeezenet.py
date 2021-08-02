"""SqueezeNet

Paper: https://arxiv.org/pdf/1602.07360.pdf
Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py

Copyright 2021 | farabio
"""
import torch
import torch.nn as nn
from typing import Any
from farabio.utils.helpers import get_num_parameters

__all__ = ['SqueezeNet', 'squeezenet']


class Fire(nn.Module):
    def __init__(
        self,
        in_planes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int
    ) -> None:
        super(Fire, self).__init__()

        self.in_planes = in_planes
        self.relu = nn.ReLU(inplace=True)

        self.squeeze = nn.Conv2d(in_planes, squeeze_planes, kernel_size=1)
        self.expand1x1 = nn.Conv2d(
            squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.squeeze(x))
        x = torch.cat([
            self.relu(self.expand1x1(x)),
            self.relu(self.expand3x3(x))
        ], 1)

        return x


class SqueezeNet(nn.Module):
    def __init__(
        self, n_classes=1000,
        init_weights: bool = True
    ) -> None:
        super(SqueezeNet, self).__init__()

        self.n_classes = n_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 28, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256)
        )

        self.conv = nn.Conv2d(512, self.n_classes, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            self.conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def squeezenet(**kwargs: Any) -> SqueezeNet:
    model = SqueezeNet(**kwargs)
    return model


def test():
    x = torch.randn(1, 3, 224, 224)

    model = SqueezeNet()
    y = model(x)

    print("Trainable parameters: ", get_num_parameters(model))
    print("in shape: ", x.shape, ", out shape: ", y.shape)


# test()
