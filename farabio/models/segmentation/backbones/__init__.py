import torch
import torch.nn as nn

__all__ = ['BackboneExtension']


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


class BackboneExtension:
    @property
    def out_channels(self):
        return self._out_channels[: self._depth + 1]
    
    def set_in_channels(self, in_channels):
        if in_channels == 3:
            return
        
        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])
        
        patch_first_conv(model=self, new_in_channels=in_channels)
    
    def get_stages(self):
        raise NotImplementedError
    
    def make_dilated(self, output_stride):
        if output_stride == 16:
            stage_list = [5,]
            dilation_list = [2,]
        
        elif output_stride == 8:
            stage_list = [4, 5]
            dilation_list = [2, 4]
        
        else:
            raise ValueError("Output stride should be 16 or 8, got {}.".format(output_stride))
            
        stages = self.get_stages()
        for stage_idx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module = stages[stage_idx],
                dilation_rate = dilation_rate
            )
