"""ShuffleNetV2
Paper: https://arxiv.org/pdf/1807.11164.pdf
Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py
Copyright 2021 | farabio
"""
import torch
import torch.nn as nn
from typing import List, Callable, Any
from farabio.utils.helpers import get_num_parameters

__all__ = ['ShuffleNetV2',
           'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
           'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']


stages = {
    'x0.5': [[4, 8, 4], [24, 48, 96, 192, 1024]],
    'x1.0': [[4, 8, 4], [24, 116, 232, 464, 1024]],
    'x1.5': [[4, 8, 4], [24, 176, 352, 704, 1024]],
    'x2.0': [[4, 8, 4], [24, 244, 488, 976, 2048]],
}


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch_size, n_channels, h, w = x.size()
    channels_per_group = n_channels // groups

    x = x.view(batch_size, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(batch_size, -1, h, w)
    return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int
    ) -> None:
        super(InvertedResidual, self).__init__()

        self.stride = stride

        branch_features = oup // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3,
                                    stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features,
                                kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(
        i: int,
        o: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        stages_repeats: List[int],
        stages_out_channels: List[int],
        n_classes: int = 1000,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual
    ) -> None:
        super(ShuffleNetV2, self).__init__()

        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(
                    output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x


def _shufflenetv2(*args: Any, **kwargs: Any) -> ShuffleNetV2:
    model = ShuffleNetV2(*args, **kwargs)
    return model


def shufflenet_v2_x0_5(**kwargs: Any) -> ShuffleNetV2:
    return _shufflenetv2(stages['x0.5'][0], stages['x0.5'][-1])


def shufflenet_v2_x1_0(**kwargs: Any) -> ShuffleNetV2:
    return _shufflenetv2(stages['x1.0'][0], stages['x1.0'][-1])


def shufflenet_v2_x1_5(**kwargs: Any) -> ShuffleNetV2:
    return _shufflenetv2(stages['x1.5'][0], stages['x1.5'][-1])


def shufflenet_v2_x2_0(**kwargs: Any) -> ShuffleNetV2:
    return _shufflenetv2(stages['x2.0'][0], stages['x2.0'][-1])


def test():
    x = torch.randn(1, 3, 224, 224)

    model = shufflenet_v2_x0_5()
    y = model(x)

    print("Trainable parameters: ", get_num_parameters(model))
    print("in shape: ", x.shape, ", out shape: ", y.shape)


# test()
