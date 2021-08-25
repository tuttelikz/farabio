"""FPN

Paper: http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf
Adapted from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/fpn/model.py

Copyright 2021 | farabio
"""
from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from farabio.models.segmentation.base import SegModel, SegmentationHead
from farabio.models.segmentation.backbones._backbones import get_backbone
from farabio.utils.helpers import get_num_parameters

__all__ = [
    'FPN', 'fpn_vgg11', 'fpn_vgg11_bn', 'fpn_vgg13', 'fpn_vgg13_bn',
    'fpn_vgg16', 'fpn_vgg16_bn', 'fpn_vgg19', 'fpn_vgg19_bn', 'fpn_mobilenetv2',
    'fpn_resnet18', 'fpn_resnet34', 'fpn_resnet50', 'fpn_resnet101', 'fpn_resnet152'
]


class FPN(SegModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        activation: Optional[str] = None,
        upsampling: int = 4
    ):
        super().__init__()
        
        self.encoder = get_backbone(
            encoder_name,
            in_channels = in_channels,
            depth = encoder_depth,
        )
        
        self.decoder = FPNDecoder(
            encoder_channels = self.encoder.out_channels, #_out_channels
            encoder_depth = encoder_depth,
            pyramid_channels = decoder_pyramid_channels,
            segmentation_channels = decoder_segmentation_channels,
            dropout = decoder_dropout,
            merge_policy = decoder_merge_policy
        )
        
        self.seg_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=out_channels,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling
        )
        
        self.class_head = None
        self.name = "fpn-{}".format(encoder_name)
        self.init()


class FPNDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth = 5,
        pyramid_channels = 256,
        segmentation_channels = 128,
        dropout = 0.2,
        merge_policy = "add"
    ):
        super().__init__()
        
        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]
        
        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])
        
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples) for n_upsamples in [3,2,1,0]
        ])
        
        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
    
    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
    
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x
    

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        
        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]
        
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))
            
        self.block = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.block(x)


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x
    

class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy
        
    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


def _fpn(
    backbone: str = "resnet18",
    in_channels = 3,
    out_channels = 1,
    **kwargs: Any
) -> FPN:
    model = FPN(
        encoder_name=backbone,
        in_channels=in_channels,
        out_channels=out_channels
    )
    return model


def fpn_vgg11(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="vgg11", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_vgg11_bn(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="vgg11_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_vgg13(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="vgg13", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_vgg13_bn(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="vgg13_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_vgg16(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="vgg16", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_vgg16_bn(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="vgg16_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_vgg19(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="vgg19", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_vgg19_bn(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="vgg19_bn", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_mobilenetv2(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="mobilenet_v2", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_resnet18(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="resnet18", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_resnet34(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="resnet34", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_resnet50(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="resnet50", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_resnet101(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="resnet101", in_channels=in_channels, out_channels=out_channels, **kwargs)


def fpn_resnet152(in_channels=3, out_channels=1, **kwargs: Any) -> FPN:
    return _fpn(backbone="resnet152", in_channels=in_channels, out_channels=out_channels, **kwargs)


def test():
    x = torch.randn(4, 3, 256, 256)

    tests = {
        "fpn_vgg11": fpn_vgg11(),
        "fpn_vgg11_bn": fpn_vgg11_bn(),
        "fpn_vgg13": fpn_vgg13(),
        "fpn_vgg13_bn": fpn_vgg13_bn(),
        "fpn_vgg16": fpn_vgg16(),
        "fpn_vgg16_bn": fpn_vgg16_bn(),
        "fpn_vgg19": fpn_vgg19(),
        "fpn_vgg19_bn": fpn_vgg19_bn(),
        "fpn_mobilenetv2": fpn_mobilenetv2(),
        "fpn_resnet18": fpn_resnet18(),
        "fpn_resnet34": fpn_resnet34(),
        "fpn_resnet50": fpn_resnet50(),
        "fpn_resnet101": fpn_resnet101(),
        "fpn_resnet152": fpn_resnet152(),
    }
    
    for key, value in tests.items():
        model = tests[key]
        y = model(x)

        print("Model name: ", model.name)
        print("Trainable parameters: ", get_num_parameters(model))
        print("in shape: ", x.shape, ", out shape: ", y.shape)


# test()