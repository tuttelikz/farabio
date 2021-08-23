"""segmodel.py

Base class for segmentation models

Copyright 2021 | farabio
"""
import torch.nn as nn

__all__ = ['SegModel']


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
