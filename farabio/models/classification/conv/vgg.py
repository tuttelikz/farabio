"""VGG

Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
Paper: https://arxiv.org/pdf/1409.1556.pdf

Copyright 2021 | farabio
"""
import torch
import torch.nn as nn
from typing import Union, List, Any, cast
from farabio.utils.helpers import get_num_parameters

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, convnet: nn.Module, n_classes: int = 1000) -> None:
        super(VGG, self).__init__()
        self.conv = convnet
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

        l1 = self._fc(nn.Linear(512*7*7, 4096))
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

    def _fc(self, linear):
        return nn.Sequential(
            linear,
            self.relu,
            self.dropout
        )


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            v = cast(str, v)
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v

    return nn.Sequential(*layers)


def _vgg(cfg: str, batch_norm: bool, **kwargs) -> VGG:
    model = VGG(convnet=make_layers(cfg, batch_norm=batch_norm), **kwargs)
    return model


def vgg11(**kwargs) -> VGG:
    return _vgg(cfgs['A'], False, **kwargs)


def vgg11_bn(**kwargs) -> VGG:
    return _vgg(cfgs['A'], True, **kwargs)


def vgg13(**kwargs) -> VGG:
    return _vgg(cfgs['B'], False, **kwargs)


def vgg13_bn(**kwargs) -> VGG:
    return _vgg(cfgs['B'], True, **kwargs)


def vgg16(**kwargs) -> VGG:
    return _vgg(cfgs['D'], False, **kwargs)


def vgg16_bn(**kwargs) -> VGG:
    return _vgg(cfgs['D'], True, **kwargs)


def vgg19(**kwargs) -> VGG:
    return _vgg(cfgs['E'], False, **kwargs)


def vgg19_bn(**kwargs) -> VGG:
    return _vgg(cfgs['E'], True, **kwargs)


def test(convnet="vgg11"):
    x = torch.randn(1, 3, 224, 224)

    tests = {
        "vgg11": vgg11(),
        "vgg11_bn": vgg11_bn(),
        "vgg13": vgg13(),
        "vgg13_bn": vgg13_bn(),
        "vgg16": vgg16(),
        "vgg16_bn": vgg16_bn(),
        "vgg19": vgg19(),
        "vgg19_bn": vgg19_bn(n_classes=3)
    }

    model = tests[convnet]
    y = model(x)

    print("Trainable parameters: ", get_num_parameters(model))
    print("in shape: ", x.shape, ", out shape: ", y.shape)

#test("vgg19", n_classes = 3)
