from farabio.models.classification.resnet import ResNet, BasicBlock, Bottleneck


class ResNetBackbone(ResNet):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._in_channels = 3
        self._out_channels = out_channels
        self._depth = depth
        
        del self.fc
        del self.avgpool
        
    def get_stages(self):
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