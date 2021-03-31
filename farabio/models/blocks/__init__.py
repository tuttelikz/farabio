import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvArithmetic:
    def __init__(self, i, s, k):
        self.i = i
        self.s = s
        self.k = k

    def padConv(self, mode="same"):
        if mode == "same":
            o = self.i
            p = int(((o-1)*self.s + self.k - self.i) / 2)
        return p

    def padTrans(self):
        o = 2 * self.i
        p = int((self.s*(self.i-1) + self.k - o) / 2)
        return p


class BareConv(nn.Module):
    def __init__(self, in_channels, out_channels, i, s=1, k=1):
        super(BareConv, self).__init__()

        p = ConvArithmetic(i, s, k).padConv("same")
        self.out = nn.Conv2d(in_channels, out_channels, k, s, p)

    def forward(self, X):
        return self.out(X)


class Conv(nn.Module):
    def __init__(self, ch_in, ch_out, i=None):
        super(Conv, self).__init__()

        s = 1
        k = 3
        if i is not None:
            p = ConvArithmetic(i, s, k).padConv("same")
        else:
            p = 1

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=k,
                      stride=s, padding=p, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=k,
                      stride=s, padding=p, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolConv(nn.Module):
    def __init__(self, in_channels, out_channels, i):
        super().__init__()

        self.poolconv = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(in_channels, out_channels, i)
        )

    def forward(self, X):
        return self.poolconv(X)


class TransConv(nn.Module):
    def __init__(self, in_channels, out_channels, i, in_, s=2, k=2):
        super().__init__()

        p = ConvArithmetic(i, s, k).padTrans()
        self.trans = nn.ConvTranspose2d(in_channels, out_channels, k, s, p)
        self.conv = Conv(in_channels, out_channels, in_)

    def forward(self, prev, skip):

        prev = self.trans(prev)
        x = torch.cat((prev, skip), dim=1)

        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()
