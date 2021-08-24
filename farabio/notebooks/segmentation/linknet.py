import sys
sys.path.append('.')
import torch.nn as nn
from segmodel import SegModel, SegmentationHead
from backbones import get_backbone
from base import *
from typing import List, Optional, Union

__all__ = ['Linknet']


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
            encoder_channels = self.encoder.out_channels, #out_channels
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