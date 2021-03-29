import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size, sigma):
    """Gaussian function
    """

    gauss = torch.Tensor([math.exp(-(x - window_size // 2) **
                                   2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Creates window
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """ssim function

    Parameters
    ----------
    img1 : [type]
        [description]
    img2 : [type]
        [description]
    window : [type]
        [description]
    window_size : [type]
        [description]
    channel : [type]
        [description]
    size_average : bool, optional
        [description], by default True

    Returns
    -------
    ssim
        float
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    """[summary]

    Parameters
    ----------
    img1 : [type]
        [description]
    img2 : [type]
        [description]
    window_size : int, optional
        [description], by default 11
    size_average : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class SegMetrics:
    """SegMetrics class evaluates segmentation results

    Methods
    -------
    iou(res, gdt)
        Calculates intersection of union
    dice(res, gdt)
        Calculates dice score of union
    """

    def iou(res, gdt):
        """Returns IOU score [0..1]

        Parameters
        ----------
        res : bool numpy.array
            resultant mask
        gdt : bool numpy.array
            ground truth mask

        Returns
        -------
        IOU: float
            IOU score, in the range [0..1]
        """
        return np.count_nonzero(res & gdt) / np.count_nonzero(res | gdt)

    def dice(res, gdt):
        """Returns Dice score [0..1]

        Parameters
        ----------
        res : bool numpy.array
            resultant mask
        gdt : bool numpy.array
            ground truth mask

        Returns
        -------
        Dice: float
            Dice score, in the range [0..1]
        """
        return 2 * np.count_nonzero(res & gdt) / (np.count_nonzero(res) + np.count_nonzero(gdt))


class SSIM(nn.Module):
    """SSIM Index
    """

    def __init__(self, window_size=11, size_average=True):
        """[summary]

        Parameters
        ----------
        window_size : int, optional
            [description], by default 11
        size_average : bool, optional
            [description], by default True
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """[summary]

        Parameters
        ----------
        img1 : [type]
            [description]
        img2 : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window,
                     self.window_size, channel, self.size_average)
