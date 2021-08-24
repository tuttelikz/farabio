import torch
import torch.nn as nn

__all__ = ['patch_first_conv']


def patch_first_conv(model, new_in_channels, default_in_channels=3):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break
    
    weight = module.weight.detach()
    module.in_channels = new_in_channels
    
    module.weight = nn.parameter.Parameter(
        torch.Tensor(
            module.out_channels,
            new_in_channels // module.groups,
            *module.kernel_size
        )
    )
    module.reset_parameters()
    

def replace_strides_with_dilation(module, dilation_rate):
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1,1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)