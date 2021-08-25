from farabio.models.segmentation.backbones.vgg import vgg_backbones
from farabio.models.segmentation.backbones.resnet import resnet_backbones
from farabio.models.segmentation.backbones.mobilenet import mobilenet_backbones

__all__ = ['get_backbone']


backbones = {}
backbones.update(vgg_backbones)
backbones.update(resnet_backbones)
backbones.update(mobilenet_backbones)


def get_backbone(name, in_channels=3, depth=5, output_stride=32, **kwargs):
    try:
        Backbone = backbones[name]["backbone"]
    except KeyError:
        raise KeyError("Wrong backbone name `{}`, supported backbones: {}".format(name, list(backbones.keys())))
        
    params = backbones[name]["params"]
    params.update(depth=depth)
    backbone = Backbone(**params)
    
    backbone.set_in_channels(in_channels)
    if output_stride != 32:
        backbone.make_dilated(output_stride)
    
    return backbone