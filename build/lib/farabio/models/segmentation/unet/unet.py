import torch
import torch.nn as nn
from farabio.models.blocks import Conv, PoolConv, TransConv, ConvArithmetic


class Final(nn.Module):
    def __init__(self, in_channels, out_channels, i, s=1, k=1):
        super(Final, self).__init__()

        p = ConvArithmetic(i, s, k).padConv("same")
        self.out = nn.Conv2d(in_channels, out_channels, k, s, p)

    def forward(self, X):
        return self.out(X)


class Unet(nn.Module):
    """U-Net from https://arxiv.org/abs/1505.04597

    Parameters
    ----------
    img_ch : int
        input image # channels
    out_ch : int
        output image # channels

    Methods
    -------
    __init__(self, img_ch, out_ch):
        Constructor for Unet class
    forward(self, X):
        Forward propagation
    """

    def __init__(self, img_ch=3, out_ch=1):
        """Constructor for Unet class

        Parameters
        ----------
        img_ch : int
            # channels of input image
        out_ch : int
            # channels of output image
        """
        super().__init__()

        num_kernels = [16, 32, 64, 128, 256]
        img_dims = [512, 256, 128, 64, 32]

        self.dummy_param = nn.Parameter(torch.empty(0))

        self.down1 = Conv(img_ch, num_kernels[0],
                          img_dims[0])
        self.down2 = PoolConv(num_kernels[0], num_kernels[1],
                              img_dims[1])
        self.down3 = PoolConv(num_kernels[1], num_kernels[2],
                              img_dims[2])
        self.down4 = PoolConv(num_kernels[2], num_kernels[3],
                              img_dims[3])

        self.bottle_neck = PoolConv(num_kernels[3], num_kernels[4],
                                    img_dims[4])

        self.up1 = TransConv(num_kernels[4], num_kernels[3],
                             img_dims[4], img_dims[3])
        self.up2 = TransConv(num_kernels[3], num_kernels[2],
                             img_dims[3], img_dims[2])
        self.up3 = TransConv(num_kernels[2], num_kernels[1],
                             img_dims[2], img_dims[1])
        self.up4 = TransConv(num_kernels[1], num_kernels[0],
                             img_dims[1], img_dims[0])

        self.final = Final(num_kernels[0], out_ch, img_dims[0])

    def forward(self, X):
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
        d1 = self.down1(X)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bottle_neck = self.bottle_neck(d4)

        # decoding + concat path
        u1 = self.up1(bottle_neck, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        out_logit = self.final(u4)

        return out_logit
