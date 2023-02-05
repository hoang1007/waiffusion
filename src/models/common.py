import torch
from torch import nn
import torch.nn.functional as F


def Normalize(channels: int):
    return nn.GroupNorm(num_groups=32, num_channels=channels)


def ConvND(
    dim: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    *args,
    **kwargs,
):
    if dim == 1:
        return nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, *args, **kwargs
        )
    elif dim == 2:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, *args, **kwargs
        )
    elif dim == 3:
        return nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, *args, **kwargs
        )
    else:
        raise NotImplementedError(f"ConvND is not implemented for dim={dim}")


def AvgPoolND(
    dim: int, kernel_size: int, stride: int = 1, padding: int = 0, *args, **kwargs
):
    if dim == 1:
        return nn.AvgPool1d(kernel_size, stride, padding, *args, **kwargs)
    elif dim == 2:
        return nn.AvgPool2d(kernel_size, stride, padding, *args, **kwargs)
    elif dim == 3:
        return nn.AvgPool3d(kernel_size, stride, padding, *args, **kwargs)
    else:
        raise NotImplementedError(f"AvgPoolND is not implemented for dim={dim}")


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        use_conv: bool = True,
        dim: int = 2,
        padding: int = 1,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.use_conv = use_conv
        self.dim = dim

        if self.use_conv:
            self.conv = ConvND(dim, in_channels, out_channels, 3, padding=padding)

    def forward(self, x: torch.Tensor):
        if self.dim == 3:
            x = F.interpolate(
                x, (x.size(2), x.size(3) * 2, x.size(4) * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        use_conv: bool = True,
        dim: int = 2,
        padding: int = 1,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        stride = (1, 2, 2) if dim == 3 else 2
        self.use_conv = use_conv
        self.dim = dim

        if self.use_conv:
            self.op = ConvND(
                dim, in_channels, out_channels, 3, stride=stride, padding=padding
            )
        else:
            self.op = AvgPoolND(dim, kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor):
        return self.op(x)
