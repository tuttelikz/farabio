import torch.nn as nn
from farabio.models.classification import MobileNetV2
from base import BackboneExtension


class MobileNetV2Backbone(MobileNetV2, BackboneExtension):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._in_channels = 3
        self._out_channels = out_channels
        self._depth = depth
        del self.classifier
        
    def get_stages(self):
        return [
            nn.Identity(),
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:]
        ]
    
    def forward(self, x):
        stages = self.get_stages()
        
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        
        return features


mobilenet_backbones = {
    "mobilenet_v2": {
        "backbone": MobileNetV2Backbone,
        "params": {
            "out_channels": (3, 16, 24, 32, 96, 1280),
        }
    }
}