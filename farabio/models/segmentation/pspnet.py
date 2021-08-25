"""PSPNet

Paper: https://arxiv.org/pdf/1612.01105
Adapted from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/pspnet/model.py

Copyright 2021 | farabio
"""
from typing import Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from farabio.models.segmentation.base import SegModel, SegmentationHead
from farabio.models.segmentation.backbones._backbones import get_backbone
from farabio.models.segmentation.blocks import Conv2dReLU
from farabio.utils.helpers import get_num_parameters

__all__ = [
    'PSPNet', 'pspnet_vgg11', 'pspnet_vgg11_bn', 'pspnet_vgg13', 'pspnet_vgg13_bn',
    'pspnet_vgg16', 'pspnet_vgg16_bn', 'pspnet_vgg19', 'pspnet_vgg19_bn', 'pspnet_mobilenetv2',
    'pspnet_resnet18', 'pspnet_resnet34', 'pspnet_resnet50', 'pspnet_resnet101', 'pspnet_resnet152'
]


class PSPNet(SegModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_name: str = "resnet34",
        encoder_depth: int = 3,
        psp_out_channels: int = 512,
        psp_use_bn: bool = True,
        psp_dropout: float = 0.2,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 8
    ):
        super().__init__()
        
        self.encoder = get_backbone(
            encoder_name,
            in_channels = in_channels,
            depth = encoder_depth,
        )
        
        self.decoder = PSPDecoder(
            encoder_channels = self.encoder.out_channels,
            use_bn = psp_use_bn,
            out_channels = psp_out_channels,
            dropout = psp_dropout
        )
        
        self.seg_head = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
            upsampling = upsampling
        )
        
        self.class_head = None
        self.name = "pspnet-{}".format(encoder_name)
        self.init()
        

class PSPDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        use_bn=True,
        out_channels=512,
        dropout=0.2
    ):
        super().__init__()
        
        self.psp = PSPModule(
            in_channels=encoder_channels[-1],
            sizes=(1,2,3,6),
            use_bn=use_bn
        )
        
        self.conv = Conv2dReLU(
            in_channels=encoder_channels[-1] * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_bn=use_bn
        )
        
        self.dropout = nn.Dropout2d(p=dropout)
        
    def forward(self, *features):
        x = features[-1]
        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)
        
        return x
    

class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1,2,3,6), use_bn=True):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, in_channels // len(sizes), size, use_bn=use_bn) for size in sizes
        ])
        
    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x
    

class PSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, use_bn=True):
        super().__init__()
        if pool_size == 1:
            use_bn = False
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            Conv2dReLU(in_channels, out_channels, (1,1), use_bn=use_bn)
        )
    
    def forward(self, x):
        h,w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h,w), mode='bilinear', align_corners=True)
        return x


def _pspnet(
    backbone: str = "resnet18",
    in_channels = 3,
    out_channels = 1,
    **kwargs: Any
) -> PSPNet:
    model = PSPNet(
        encoder_name=backbone,
        in_channels=in_channels,
        out_channels=out_channels
    )
    return model


def pspnet_vgg11(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="vgg11", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_vgg11_bn(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="vgg11_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_vgg13(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="vgg13", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_vgg13_bn(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="vgg13_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_vgg16(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="vgg16", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_vgg16_bn(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="vgg16_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_vgg19(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="vgg19", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_vgg19_bn(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="vgg19_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_mobilenetv2(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="mobilenet_v2", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_resnet18(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="resnet18", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_resnet34(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="resnet34", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_resnet50(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="resnet50", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_resnet101(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="resnet101", in_channels=in_channels, out_channels=out_channels, **kwargs)


def pspnet_resnet152(in_channels=3, out_channels=1, **kwargs: Any) -> PSPNet:
    return _pspnet(backbone="resnet152", in_channels=in_channels, out_channels=out_channels, **kwargs)


def test():
    x = torch.randn(4, 3, 256, 256)

    tests = {
        "pspnet_vgg11": pspnet_vgg11(),
        "pspnet_vgg11_bn": pspnet_vgg11_bn(),
        "pspnet_vgg13": pspnet_vgg13(),
        "pspnet_vgg13_bn": pspnet_vgg13_bn(),
        "pspnet_vgg16": pspnet_vgg16(),
        "pspnet_vgg16_bn": pspnet_vgg16_bn(),
        "pspnet_vgg19": pspnet_vgg19(),
        "pspnet_vgg19_bn": pspnet_vgg19_bn(),
        "pspnet_mobilenetv2": pspnet_mobilenetv2(),
        "pspnet_resnet18": pspnet_resnet18(),
        "pspnet_resnet34": pspnet_resnet34(),
        "pspnet_resnet50": pspnet_resnet50(),
        "pspnet_resnet101": pspnet_resnet101(),
        "pspnet_resnet152": pspnet_resnet152(),
    }
    
    for key, value in tests.items():
        model = tests[key]
        y = model(x)

        print("Model name: ", model.name)
        print("Trainable parameters: ", get_num_parameters(model))
        print("in shape: ", x.shape, ", out shape: ", y.shape)


# test()