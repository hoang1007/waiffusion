from typing import List, Optional

import numpy as np
import torch
from torch import nn

from src.models.common import Normalize, Upsample, Downsample
from src.models.attention import AttentionType, SelfAttentionBlock
from src.models.unet.unet_blocks import ResBlock


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_channels: List[int] = [64],
        attention_levels: List[int] = [0],
        attn_type: AttentionType = "standard",
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
        num_res_blocks: int = 2,
        double_z: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList()

        # down
        out_channel = hidden_channels[0]
        for i in range(len(hidden_channels)):
            in_channel = out_channel
            out_channel = hidden_channels[i]
            add_downsample = i != len(hidden_channels) - 1
            add_attention = i in attention_levels

            self.down_blocks.append(
                DownBlock(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    add_downsample=add_downsample,
                    num_res_blocks=self.num_res_blocks,
                    add_attention=add_attention,
                    attn_type=attn_type,
                    num_attn_heads=num_attn_heads,
                    channels_per_head=channels_per_head,
                    dropout=dropout,
                )
            )
        
        self.mid_block = MidBlock(
            in_channels=hidden_channels[-1],
            attn_type=attn_type,
            num_attn_heads=num_attn_heads,
            channels_per_head=channels_per_head,
            dropout=dropout,
        )

        self.out_proj = nn.Sequential(
            Normalize(hidden_channels[-1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels[-1], 
                out_channels=2 * out_channels if double_z else out_channels,
                kernel_size=3,
                padding=1
            ),
        )

    def forward(self, x: torch.Tensor):
        h = self.conv_in(x)

        for down_block in self.down_blocks:
            h = down_block(h)

        h = self.mid_block(h)
    
        h = self.out_proj(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_channels: List[int] = [64],
        attention_levels: List[int] = [0],
        attn_type: AttentionType = "standard",
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels[-1],
            kernel_size=3,
            padding=1
        )

        self.up_blocks = nn.ModuleList()
        out_channel = hidden_channels[-1]
        for level, channels in zip(reversed(range(len(hidden_channels))), reversed(hidden_channels)):
            prev_out_channel = out_channel
            out_channel = channels
            add_upsample = level != 0
            add_attention = level in attention_levels

            self.up_blocks.append(UpBlock(
                in_channels=prev_out_channel,
                out_channels=out_channel,
                num_res_blocks=num_res_blocks + 1,
                add_upsample=add_upsample,
                add_attention=add_attention,
                attn_type=attn_type,
                num_attn_heads=num_attn_heads,
                channels_per_head=channels_per_head,
                dropout=dropout,
            ))

        self.mid_block = MidBlock(
            in_channels=hidden_channels[-1],
            attn_type=attn_type,
            num_attn_heads=num_attn_heads,
            channels_per_head=channels_per_head,
            dropout=dropout,
        )
        
        self.out_proj = nn.Sequential(
            Normalize(hidden_channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels[0],
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, z: torch.Tensor):
        h = self.conv_in(z)

        h = self.mid_block(h)

        for up_block in self.up_blocks:
            h = up_block(h)

        h = self.out_proj(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = torch.randn(
            self.mean.shape, generator=generator, device=self.parameters.device, dtype=self.parameters.dtype
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_res_blocks: int = 1,
        add_downsample: bool = True,
        use_skip_conv: bool = True,
        use_down_conv: bool = True,
        add_attention: bool = False,
        attn_type: AttentionType = "standard",
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            res_in_channels = in_channels if i == 0 else out_channels

            self.res_blocks.append(
                ResBlock(
                    in_channels=res_in_channels,
                    out_channels=out_channels,
                    use_conv=use_skip_conv,
                    dropout=dropout,
                )
            )

        if add_attention:
            self.attention_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                self.attention_blocks.append(
                    SelfAttentionBlock(
                        out_channels,
                        attn_type=attn_type,
                        num_attn_heads=num_attn_heads,
                        channels_per_head=channels_per_head,
                    )
                )
        else:
            self.attention_blocks = None

        if add_downsample:
            self.downsampler = Downsample(out_channels, use_conv=use_down_conv)
        else:
            self.downsampler = None

    def forward(self, x: torch.Tensor):
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)

            if self.attention_blocks is not None:
                x = self.attention_blocks[i](x)

        if self.downsampler is not None:
            x = self.downsampler(x)

        return x

    @property
    def num_layers(self):
        return len(self.res_blocks)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_res_blocks: int = 1,
        add_upsample: bool = True,
        use_skip_conv: bool = True,
        use_up_conv: bool = True,
        add_attention: bool = False,
        attn_type: AttentionType = "standard",
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            res_in_channels = in_channels if i == 0 else out_channels

            self.res_blocks.append(ResBlock(
                in_channels=res_in_channels,
                out_channels=out_channels,
                use_conv=use_skip_conv,
                dropout=dropout,
            ))

        if add_attention:
            self.attention_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                self.attention_blocks.append(
                    SelfAttentionBlock(
                        out_channels,
                        attn_type=attn_type,
                        num_attn_heads=num_attn_heads,
                        channels_per_head=channels_per_head,
                    )
                )
        else:
            self.attention_blocks = None

        if add_upsample:
            self.upsampler = Upsample(out_channels, use_conv=use_up_conv)
        else:
            self.upsampler = None

    def forward(self, x: torch.Tensor):
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)

            if self.attention_blocks is not None:
                x = self.attention_blocks[i](x)

        if self.upsampler is not None:
            x = self.upsampler(x)

        return x

    @property
    def num_layers(self):
        return len(self.res_blocks)
    

class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        dim: int = 2,
        num_layers: int = 1,
        use_skip_conv: bool = True,
        add_attention: bool = True,
        attn_type: AttentionType = "standard",
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                use_conv=use_skip_conv,
                dim=dim,
                dropout=dropout,
            )
        ])

        for _ in range(num_layers):
            self.res_blocks.append(
                ResBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    use_conv=use_skip_conv,
                    dim=dim,
                    dropout=dropout,
                )
            )

        
        self.attention_blocks = []
        for _ in range(num_layers):
            if add_attention:
                self.attention_blocks.append(
                    SelfAttentionBlock(
                        in_channels,
                        attn_type=attn_type,
                        num_attn_heads=num_attn_heads,
                        channels_per_head=channels_per_head,
                    )
                )
            else:
                self.attention_blocks.append(None)
        
        if add_attention:
            self.attention_blocks = nn.ModuleList(self.attention_blocks)

    def forward(self, x: torch.Tensor, embedding: Optional[torch.Tensor] = None):
        x = self.res_blocks[0](x, embedding)

        for attn, res_block in zip(self.attention_blocks, self.res_blocks[1:]):
            if attn is not None:
                x = attn(x)
            x = res_block(x, embedding)

        return x
