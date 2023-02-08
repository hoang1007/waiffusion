from copy import deepcopy
from typing import List, Optional, Tuple

import torch
from torch import nn

from src.models.attention import AttentionBlock, AttentionInterface
from src.models.common import ConvND, Downsample, Normalize, Upsample
from src.utils.module_utils import zero_module

from .modules import ResBlock, SinusoidalTimestepEmbedding, TimestepEmbedSequential


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int],
        num_res_blocks: int,
        attn: AttentionInterface,
        attention_levels: Tuple[int],
        dropout: float = 0.0,
        dim: int = 2,
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
    ):
        super().__init__()

        top_hidden_channels = hidden_channels[0]
        last_hidden_channels = hidden_channels[-1]
        time_embedding_dim = top_hidden_channels * 4

        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(top_hidden_channels),
            nn.Linear(top_hidden_channels, time_embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.encoder = UnetEncoder(
            in_channels,
            hidden_channels,
            time_embedding_dim,
            num_res_blocks,
            attn,
            attention_levels,
            dropout,
            dim,
            num_attn_heads,
            channels_per_head,
        )

        self.middle_blocks = TimestepEmbedSequential(
            ResBlock(last_hidden_channels, time_embedding_dim, dim=dim),
            AttentionBlock(
                last_hidden_channels, deepcopy(attn), num_attn_heads, channels_per_head
            ),
            ResBlock(last_hidden_channels, time_embedding_dim, dim=dim),
        )

        self.decoder = UnetDecoder(
            last_hidden_channels,
            hidden_channels[::-1],
            self.encoder.block_channels,
            time_embedding_dim,
            num_res_blocks,
            attn,
            attention_levels,
            dropout,
            dim,
            num_attn_heads,
            channels_per_head,
        )

        self.out_proj = nn.Sequential(
            Normalize(top_hidden_channels),
            nn.SiLU(inplace=True),
            zero_module(ConvND(dim, top_hidden_channels, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_steps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ):
        time_embedding = self.time_embedding(time_steps)

        h, states = self.encoder(x, time_embedding, context)
        h = self.middle_blocks(h, time_embedding, context)
        h = self.decoder(h, states, time_embedding, context)

        return self.out_proj(h)


class UnetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Tuple[int],
        embedding_dim: int,
        num_res_blocks: int,
        attn: AttentionInterface,
        attention_levels: Tuple[int],
        dropout: float = 0.0,
        dim: int = 2,
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self._block_channels = []

        cur_channels = hidden_channels[0]
        self.blocks.append(
            TimestepEmbedSequential(ConvND(dim, in_channels, cur_channels, 3, padding=1))
        )
        self._block_channels.append(cur_channels)

        for level, channels in enumerate(hidden_channels):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(cur_channels, embedding_dim, channels, dim=dim, dropout=dropout)
                ]

                cur_channels = channels

                if level in attention_levels:
                    layers.append(
                        AttentionBlock(channels, deepcopy(attn), num_attn_heads, channels_per_head)
                    )

                self.blocks.append(TimestepEmbedSequential(*layers))
                self._block_channels.append(cur_channels)

            if level != len(hidden_channels) - 1:  # if not last
                self.blocks.append(TimestepEmbedSequential(Downsample(cur_channels, dim=dim)))
                self._block_channels.append(cur_channels)

    @property
    def block_channels(self):
        return self._block_channels

    def forward(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ):
        states = []
        for module in self.blocks:
            x = module(x, embedding, context)
            states.append(x)

        return x, states


class UnetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Tuple[int],
        encoder_block_channels: List[int],
        embedding_dim: int,
        num_res_blocks: int,
        attn: AttentionInterface,
        attention_levels: Tuple[int],
        dropout: float = 0.0,
        dim: int = 2,
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        cur_channels = in_channels
        for level, channels in enumerate(hidden_channels):
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        cur_channels + encoder_block_channels.pop(),
                        embedding_dim,
                        channels,
                        dim=dim,
                        dropout=dropout,
                    )
                ]

                cur_channels = channels

                if level in attention_levels:
                    layers.append(
                        AttentionBlock(channels, deepcopy(attn), num_attn_heads, channels_per_head)
                    )

                if level != len(hidden_channels) - 1 and i == num_res_blocks:
                    layers.append(Upsample(cur_channels, dim=dim))

                self.blocks.append(TimestepEmbedSequential(*layers))

    def forward(
        self,
        x: torch.Tensor,
        enc_outputs: List[torch.Tensor],
        embedding: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ):
        for module in self.blocks:
            x = torch.cat((x, enc_outputs.pop()), dim=1)
            x = module(x, embedding, context)

        return x
