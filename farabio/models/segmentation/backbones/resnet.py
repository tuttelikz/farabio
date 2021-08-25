import torch.nn as nn
from farabio.models.classification.resnet import ResNet, BasicBlock, Bottleneck
from farabio.models.segmentation.backbones import BackboneExtension

__all__ = ['ResNetBackbone', 'resnet_backbones']


class ResNetBackbone(ResNet, BackboneExtension):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._in_channels = 3
        self._out_channels = out_channels
        self._depth = depth
        
        del self.fc
        del self.avgpool
    
    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()
        
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        
        return features


resnet_backbones = {
    "resnet18": {
        "backbone": ResNetBackbone,
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "backbone": ResNetBackbone,
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        "backbone": ResNetBackbone,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
        "backbone": ResNetBackbone,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "backbone": ResNetBackbone,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
}