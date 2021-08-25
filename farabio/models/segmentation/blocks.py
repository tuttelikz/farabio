import torch
import torch.nn as nn

__all__ = ['Conv2dReLU', 'SCSEModule', 'ArgMax', 'Activation', 'Attention', 'Flatten']


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_bn=True
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_bn),
        )
        relu = nn.ReLU(inplace=True)
        if use_bn:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()
        
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())
        
    def forward(self, x):
        return x * self.cSE(x) + x*self.sSE(x)
    

class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.argmax(x, dim=self.dim)
    

class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        
        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Not implemented: got {}'.format(name))

    def forward(self, x):
        return self.activation(x)
    

class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))
        
    def forward(self, x):
        return self.attention(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
