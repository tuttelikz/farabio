import sys
sys.path.append('.')
from segmodel import SegModel, SegmentationHead
from backbones import get_backbone
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import *


__all__ = ['PSPNet']


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
            encoder_channels = self.encoder.out_channels, #out_channels
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