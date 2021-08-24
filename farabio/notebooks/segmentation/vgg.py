import torch.nn as nn
from farabio.models.classification.vgg import VGG, make_layers, cfgs
from base import BackboneExtension


class VGGBackbone(VGG, BackboneExtension):
    def __init__(self, out_channels, config, batch_norm=False, depth=5, **kwargs):
        super().__init__(make_layers(config, batch_norm=batch_norm), **kwargs)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3
        
        del self.fc
        del self.final
    
    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("'VGG' models do not support dilated mode due to Max Pooling"
                         " operations for downsampling!")
        
    def get_stages(self):
        stages = []
        stage_modules = []
        for module in self.conv:
            if isinstance(module, nn.MaxPool2d):
                stages.append(nn.Sequential(*stage_modules))
                stage_modules = []
            stage_modules.append(module)
        stages.append(nn.Sequential(*stage_modules))
        return stages
    
    def forward(self, x):
        stages = self.get_stages()
        
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        
        return features


vgg_backbones = {
    "vgg11": {
        "backbone": VGGBackbone,
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfgs['A'],
            "batch_norm": False,
        }
    },
    "vgg11_bn": {
        "backbone": VGGBackbone,
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfgs['A'],
            "batch_norm": True,
        }
    },
    "vgg13": {
        "backbone": VGGBackbone,
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfgs['B'],
            "batch_norm": False,
        }
    },
    "vgg13_bn": {
        "backbone": VGGBackbone,
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfgs['B'],
            "batch_norm": True,
        }
    },
    "vgg16": {
        "backbone": VGGBackbone,
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfgs['D'],
            "batch_norm": False,
        }
    },
    "vgg16_bn": {
        "backbone": VGGBackbone,
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfgs['D'],
            "batch_norm": True,
        }
    },
    "vgg19": {
        "backbone": VGGBackbone,
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfgs['E'],
            "batch_norm": False,
        }
    },
    "vgg19_bn": {
        "backbone": VGGBackbone,
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfgs['E'],
            "batch_norm": True,
        }
    }
}