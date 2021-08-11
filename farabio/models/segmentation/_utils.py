"""Common utility for segmentation models

Adapted from: https://github.com/pytorch/vision/blob/183a722169421c83638e68ee2d8fc5bd3415c4b4/torchvision/models/_utils.py

Copyright 2021 | farabio
"""

from collections import OrderedDict

from torch import nn
from torch.nn import functional as F


class _SimpleSegmentationModel(nn.Module):

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        return result
