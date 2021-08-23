from .resnet import resnet_backbones


backbones = {}
backbones.update(resnet_backbones)

def get_backbone(name, in_channels=3, depth=5, output_stride=32, **kwargs):
    try:
        Backbone = backbones[name]["backbone"]
    except KeyError:
        raise KeyError("Wrong backbone name `{}`, supported backbones: {}".format(name, list(backbones.keys())))
        
    params = backbones[name]["params"]
    params.update(depth=depth)
    backbone = Backbone(**params)
    
    # set_in_channels
    if output_stride != 32:
        backbone.make_dilated(output_stride)
    
    return backbone
    