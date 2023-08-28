from copy import deepcopy
from typing import List, Optional, Tuple

import torch
from torch import nn

from src.models.attention import AttentionBlock, AttentionType
from src.models.common import ConvND, Normalize
from src.utils.module_utils import zero_module

from .unet_blocks import (
    DownBlock,
    ResBlock,
    MidBlock,
    SinusoidalTimestepEmbedding,
    TimestepEmbedSequential,
    UpBlock,
)


class Unet(nn.Module):
    def __init__(
        self,
        sample_size: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int],
        num_res_blocks: int,
        attention_levels: Tuple[int],
        attn_type: AttentionType = "standard",
        dropout: float = 0.0,
        dim: int = 2,
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
        num_class_embeds: Optional[int] = None,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.in_channels = in_channels

        top_hidden_channels = hidden_channels[0]
        last_hidden_channels = hidden_channels[-1]
        time_embedding_dim = top_hidden_channels * 4

        self.conv_in = ConvND(dim, in_channels, top_hidden_channels, 3, padding=1)

        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(top_hidden_channels),
            nn.Linear(top_hidden_channels, time_embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embedding_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList()
        self.mid_block = None
        self.up_blocks = nn.ModuleList()

        # down
        in_channels = hidden_channels[0]
        for level, channels in enumerate(hidden_channels):
            add_attention = level in attention_levels
            add_downsample = level != len(hidden_channels) - 1

            self.down_blocks.append(
                DownBlock(
                    in_channels=in_channels,
                    out_channels=channels,
                    embedding_channels=time_embedding_dim,
                    dropout=dropout,
                    dim=dim,
                    num_layers=num_res_blocks,
                    add_downsample=add_downsample,
                    add_attention=add_attention,
                    attn_type=attn_type,
                    num_attn_heads=num_attn_heads,
                    channels_per_head=channels_per_head,
                )
            )

            in_channels = channels

        # mid
        self.mid_block = MidBlock(
            in_channels=last_hidden_channels,
            embedding_channels=time_embedding_dim,
            dim=dim,
            num_layers=1,
            attn_type=attn_type,
            num_attn_heads=num_attn_heads,
            channels_per_head=channels_per_head,
            dropout=dropout
        )

        # up
        prev_out_channels = hidden_channels[-1]
        for level, channels in zip(
            reversed(range(len(hidden_channels))), reversed(hidden_channels)
        ):
            add_attention = level in attention_levels
            add_upsample = level != 0

            in_channels = hidden_channels[max(0, level - 1)]

            self.up_blocks.append(
                UpBlock(
                    in_channels=in_channels,
                    prev_out_channels=prev_out_channels,
                    out_channels=channels,
                    embedding_channels=time_embedding_dim,
                    dropout=dropout,
                    dim=dim,
                    num_layers=num_res_blocks + 1,
                    add_upsample=add_upsample,
                    add_attention=add_attention,
                    attn_type=attn_type,
                    num_attn_heads=num_attn_heads,
                    channels_per_head=channels_per_head,
                )
            )

            prev_out_channels = channels

        self.out_proj = nn.Sequential(
            Normalize(top_hidden_channels),
            nn.SiLU(inplace=True),
            zero_module(ConvND(dim, top_hidden_channels, out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_steps: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ):
        assert (
            tuple(x.shape[2:]) == tuple(self.sample_size),
            f"Sample does not match the shape {self.sample_size}. Got {x.shape[2:]}"
        )
        emb = self.time_embedding(time_steps)

        if class_labels is not None:
            assert self.class_embedding is not None, "Model does not have class embedding module"
            class_embedding = self.class_embedding(class_labels)
            emb = emb + class_embedding

        x = self.conv_in(x)

        hidden_states = [x]
        for down_block in self.down_blocks:
            x, hs = down_block(x, emb)
            hidden_states.extend(hs)

        x = self.mid_block(x, emb)

        for up_block in self.up_blocks:
            hs = hidden_states[-up_block.num_layers :]
            hidden_states = hidden_states[: -up_block.num_layers]

            x = up_block(x, hs, emb)

        x = self.out_proj(x)

        return x
