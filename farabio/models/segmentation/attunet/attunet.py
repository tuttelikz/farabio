import torch
import torch.nn as nn
from farabio.models.blocks import Conv, UpConv


class AttBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class AttUnet(nn.Module):
    """Attention U-Net from https://arxiv.org/abs/1804.03999

    Parameters
    ----------
    img_ch : int
        input image # channels
    out_ch : int
        output image # channels

    Methods
    -------
    __init__(self, img_ch=3, out_ch=1)
        Constructor for Unet class
    forward(self, x):
        Forward propagation
    """

    def __init__(self, img_ch=3, out_ch=1):
        """Constructor for AttUnet class

        Parameters
        ----------
        img_ch : int, optional
            # channels of input image, by default 3
        out_ch : int, optional
            # channels of output image, by default 1
        """
        super(AttUnet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        num_kernels = [64, 128, 256, 512, 1024]

        self.Conv1 = Conv(ch_in=img_ch, ch_out=64)
        self.Conv2 = Conv(ch_in=num_kernels[0], ch_out=num_kernels[1])
        self.Conv3 = Conv(ch_in=num_kernels[1], ch_out=num_kernels[2])
        self.Conv4 = Conv(ch_in=num_kernels[2], ch_out=num_kernels[3])
        self.Conv5 = Conv(ch_in=num_kernels[3], ch_out=num_kernels[4])

        self.Up5 = UpConv(ch_in=num_kernels[4], ch_out=num_kernels[3])
        self.Att5 = AttBlock(F_g=num_kernels[3], F_l=num_kernels[3],
                             F_int=num_kernels[2])
        self.Up_conv5 = Conv(ch_in=num_kernels[4], ch_out=num_kernels[3])

        self.Up4 = UpConv(ch_in=num_kernels[3], ch_out=num_kernels[2])
        self.Att4 = AttBlock(F_g=num_kernels[2], F_l=num_kernels[2],
                             F_int=num_kernels[1])
        self.Up_conv4 = Conv(ch_in=num_kernels[3], ch_out=num_kernels[2])

        self.Up3 = UpConv(ch_in=num_kernels[2], ch_out=num_kernels[1])
        self.Att3 = AttBlock(F_g=num_kernels[1], F_l=num_kernels[1],
                             F_int=num_kernels[1])
        self.Up_conv3 = Conv(ch_in=num_kernels[2], ch_out=num_kernels[1])

        self.Up2 = UpConv(ch_in=num_kernels[1], ch_out=num_kernels[0])
        self.Att2 = AttBlock(F_g=num_kernels[0], F_l=num_kernels[0],
                             F_int=num_kernels[0]//2)
        self.Up_conv2 = Conv(ch_in=num_kernels[1], ch_out=num_kernels[0])

        self.Conv_1x1 = nn.Conv2d(64, out_ch, kernel_size=1,
                                  stride=1, padding=0)

    def forward(self, x):
        """Forward propagation

        Parameters
        ----------
        X : torch.Tensor
            tensor to propagate

        Returns
        -------
        torch.Tensor
            tensor propagated through network, bare logit
        """
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
