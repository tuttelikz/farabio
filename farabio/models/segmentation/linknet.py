"""LinkNet

Paper: https://arxiv.org/pdf/1707.03718
Adapted from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/linknet/model.py

Copyright 2021 | farabio
"""
from typing import List, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from farabio.models.segmentation.base import SegModel, SegmentationHead
from farabio.models.segmentation.backbones._backbones import get_backbone
from farabio.models.segmentation.blocks import Conv2dReLU
from farabio.utils.helpers import get_num_parameters

__all__ = [
    'Linknet', 'linknet_vgg11', 'linknet_vgg11_bn', 'linknet_vgg13', 'linknet_vgg13_bn',
    'linknet_vgg16', 'linknet_vgg16_bn', 'linknet_vgg19', 'linknet_vgg19_bn', 'linknet_mobilenetv2',
    'linknet_resnet18', 'linknet_resnet34', 'linknet_resnet50', 'linknet_resnet101', 'linknet_resnet152'
]


class Linknet(SegModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        decoder_use_bn: bool = True,
        decoder_attention_type: Optional[str] = None,
        activation: Optional[Union[str, callable]] = None
    ):
        super().__init__()
        
        self.encoder = get_backbone(
            encoder_name,
            in_channels = in_channels,
            depth = encoder_depth,
        )
        
        self.decoder = LinknetDecoder(
            encoder_channels = self.encoder.out_channels,
            n_blocks = encoder_depth,
            prefinal_channels=32,
            use_bn = decoder_use_bn
        )
        
        self.seg_head = SegmentationHead(
            in_channels=32,
            out_channels=out_channels,
            activation=activation,
            kernel_size=1
        )
        
        self.class_head = None
        self.name = "linknet-{}".format(encoder_name)
        self.init()
        

class LinknetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        n_blocks=5,
        prefinal_channels=32,
        use_bn = True
    ):
        super().__init__()
        
        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        
        channels = list(encoder_channels) + [prefinal_channels]
        
        self.blocks = nn.ModuleList([
            DecoderBlock(channels[i], channels[i+1], use_bn=use_bn)
            for i in range(n_blocks)
        ])
        
    def forward(self, *features):
        features = features[1:]
        features = features[::-1]
        
        x = features[0]
        skips = features[1:]
        
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, in_channels // 4, kernel_size=1, use_bn=use_bn),
            TransposeX2(in_channels // 4, in_channels // 4, use_bn=use_bn),
            Conv2dReLU(in_channels // 4, out_channels, kernel_size=1, use_bn=use_bn)
        )
    
    def forward(self, x, skip=None):
        x = self.block(x)
        if skip is not None:
            x = x + skip
        return x


class TransposeX2(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ]
        
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        
        super().__init__(*layers)


def _linknet(
    backbone: str = "resnet18",
    in_channels = 3,
    out_channels = 1,
    **kwargs: Any
) -> Linknet:
    model = Linknet(
        encoder_name=backbone,
        in_channels=in_channels,
        out_channels=out_channels
    )
    return model


def linknet_vgg11(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="vgg11", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_vgg11_bn(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="vgg11_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_vgg13(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="vgg13", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_vgg13_bn(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="vgg13_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_vgg16(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="vgg16", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_vgg16_bn(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="vgg16_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_vgg19(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="vgg19", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_vgg19_bn(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="vgg19_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_mobilenetv2(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="mobilenet_v2", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_resnet18(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="resnet18", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_resnet34(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="resnet34", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_resnet50(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="resnet50", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_resnet101(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="resnet101", in_channels=in_channels, out_channels=out_channels, **kwargs)


def linknet_resnet152(in_channels=3, out_channels=1, **kwargs: Any) -> Linknet:
    return _linknet(backbone="resnet152", in_channels=in_channels, out_channels=out_channels, **kwargs)


def test():
    x = torch.randn(4, 3, 256, 256)

    tests = {
        "linknet_vgg11": linknet_vgg11(),
        "linknet_vgg11_bn": linknet_vgg11_bn(),
        "linknet_vgg13": linknet_vgg13(),
        "linknet_vgg13_bn": linknet_vgg13_bn(),
        "linknet_vgg16": linknet_vgg16(),
        "linknet_vgg16_bn": linknet_vgg16_bn(),
        "linknet_vgg19": linknet_vgg19(),
        "linknet_vgg19_bn": linknet_vgg19_bn(),
        "linknet_mobilenetv2": linknet_mobilenetv2(),
        "linknet_resnet18": linknet_resnet18(),
        "linknet_resnet34": linknet_resnet34(),
        "linknet_resnet50": linknet_resnet50(),
        "linknet_resnet101": linknet_resnet101(),
        "linknet_resnet152": linknet_resnet152(),
    }
    
    for key, value in tests.items():
        model = tests[key]
        y = model(x)

        print("Model name: ", model.name)
        print("Trainable parameters: ", get_num_parameters(model))
        print("in shape: ", x.shape, ", out shape: ", y.shape)


# test()