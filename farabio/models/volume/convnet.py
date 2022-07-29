import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.maxpool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.bn(self.maxpool(out))
        return out


class ConvNet3D(nn.Module):
    def __init__(self, block, num_classes, channels=None, latent=None):
        super(ConvNet3D, self).__init__()
        self.in_channels = 1
        if not channels:
            channels = [16, 64, 128, 256]
        if not latent:
            latent = 512
        self.layer1 = self.make_layer(block, channels[0])
        self.layer2 = self.make_layer(block, channels[1])
        self.layer3 = self.make_layer(block, channels[2])
        self.layer4 = self.make_layer(block, channels[3])

        self.gap = nn.AvgPool3d(kernel_size=2)
        self.dense = nn.Linear(channels[-1]*4*4*2, latent)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(latent, num_classes)

    def make_layer(self, block, out_channels):
        layer = block(self.in_channels, out_channels)
        self.in_channels = out_channels
        return layer

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        out = self.dropout(out)
        out = torch.sigmoid(self.fc(out))

        return out.view(-1)
