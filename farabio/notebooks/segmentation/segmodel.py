"""segmodel.py

Base class for segmentation models

Copyright 2021 | farabio
"""
import sys
sys.path.append('.')
import torch.nn as nn
from base import Activation

__all__ = ['SegModel', 'SegmentationHead']


def init_decoder(block):
    for module in block.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


def init_head(block):
    for module in block.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class SegModel(nn.Module):
    def init(self):
        init_decoder(self.decoder)
        init_head(self.seg_head)
        if self.class_head is not None:
            init_head(self.class_head)
            
    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(*encoder_out)
        
        masks = self.seg_head(decoder_out)
        return masks
    
    def predict(self, x):
        if self.training:
            self.eval()
        
        with torch.no_grad():
            x = self.forward(x)
        
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)