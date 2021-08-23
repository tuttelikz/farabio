from segmodel import SegModel
from typing import List, Optional, Union

__all__ = ['Unet']


class Unet(SegModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_name: str = "resnet34"
        encoder_depth: int = 5,
        decoder_use_bn: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        activation: Optional[Union[str, callable]] = None
    ):
        super().__init__()
        
        self.encoder = get_backbone(
            encoder_name,
            in_channels = in_channels,
            depth = encoder_depth,
        )
        
        self.decoder = UnetDecoder(
            encoder_channels = self.backbone.out_channels,
            decoder_channels = decoder_channels,
            n_blocks = encoder_depth,
            use_bn = decoder_use_bn,
            center=True if encoder
        )
        
        #########