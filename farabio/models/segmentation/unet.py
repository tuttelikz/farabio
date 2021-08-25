"""U-Net

Paper: https://arxiv.org/pdf/1505.04597
Adapted from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/unet/model.py

Copyright 2021 | farabio
"""
from typing import List, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from farabio.models.segmentation.base import SegModel, SegmentationHead
from farabio.models.segmentation.backbones._backbones import get_backbone
from farabio.models.segmentation.blocks import Conv2dReLU, Attention
from farabio.utils.helpers import get_num_parameters

__all__ = [
    'Unet', 'unet_vgg11', 'unet_vgg11_bn', 'unet_vgg13', 'unet_vgg13_bn',
    'unet_vgg16', 'unet_vgg16_bn', 'unet_vgg19', 'unet_vgg19_bn', 'unet_mobilenetv2',
    'unet_resnet18', 'unet_resnet34', 'unet_resnet50', 'unet_resnet101', 'unet_resnet152'
]


class Unet(SegModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        decoder_use_bn: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        activation: Optional[Union[str, callable]] = None
    ):
        super().__init__()
        
        self.encoder = get_backbone(
            encoder_name,
            in_channels = in_channels,
            depth = encoder_depth,
        )
        
        self.decoder = UnetDecoder(
            encoder_channels = self.encoder.out_channels,
            decoder_channels = decoder_channels,
            n_blocks = encoder_depth,
            use_bn = decoder_use_bn,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type = decoder_attention_type
        )
        
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=out_channels,
            activation=activation,
            kernel_size=3
        )

        self.class_head = None
        self.name = "unet-{}".format(encoder_name)
        self.init()
        
        
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels,
        use_bn = True,
        attention_type = None
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_bn = use_bn
        )
        self.attention1 = Attention(attention_type, in_channels = in_channels+skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_bn=use_bn
        )
        self.attention2 = Attention(attention_type, in_channels = out_channels)
        
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
    

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_bn=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_bn=use_bn
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_bn=use_bn
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_bn=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()
        
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        
        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_bn=use_bn
            )
        else:
            self.center = nn.Identity()
        
        kwargs = dict(use_bn=use_bn, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, out_ch, skip_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        
    def forward(self, *features):
        features = features[1:]
        features = features[::-1]
        
        head = features[0]
        skips = features[1:]
        
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        
        return x


def _unet(
    backbone: str = "resnet18",
    in_channels = 3,
    out_channels = 1,
    **kwargs: Any
) -> Unet:
    model = Unet(
        encoder_name=backbone,
        in_channels=in_channels,
        out_channels=out_channels
    )
    return model


def unet_vgg11(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="vgg11", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_vgg11_bn(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="vgg11_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_vgg13(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="vgg13", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_vgg13_bn(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="vgg13_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_vgg16(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="vgg16", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_vgg16_bn(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="vgg16_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_vgg19(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="vgg19", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_vgg19_bn(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="vgg19_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_mobilenetv2(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="mobilenet_v2", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_resnet18(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="resnet18", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_resnet34(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="resnet34", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_resnet50(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="resnet50", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_resnet101(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="resnet101", in_channels=in_channels, out_channels=out_channels, **kwargs)


def unet_resnet152(in_channels=3, out_channels=1, **kwargs: Any) -> Unet:
    return _unet(backbone="resnet152", in_channels=in_channels, out_channels=out_channels, **kwargs)


def test():
    x = torch.randn(4, 3, 256, 256)

    tests = {
        "unet_vgg11": unet_vgg11(),
        "unet_vgg11_bn": unet_vgg11_bn(),
        "unet_vgg13": unet_vgg13(),
        "unet_vgg13_bn": unet_vgg13_bn(),
        "unet_vgg16": unet_vgg16(),
        "unet_vgg16_bn": unet_vgg16_bn(),
        "unet_vgg19": unet_vgg19(),
        "unet_vgg19_bn": unet_vgg19_bn(),
        "unet_mobilenetv2": unet_mobilenetv2(),
        "unet_resnet18": unet_resnet18(),
        "unet_resnet34": unet_resnet34(),
        "unet_resnet50": unet_resnet50(),
        "unet_resnet101": unet_resnet101(),
        "unet_resnet152": unet_resnet152(),
    }
    
    for key, value in tests.items():
        model = tests[key]
        y = model(x)

        print("Model name: ", model.name)
        print("Trainable parameters: ", get_num_parameters(model))
        print("in shape: ", x.shape, ", out shape: ", y.shape)


# test()