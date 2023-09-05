from abc import abstractmethod
from typing import List, Optional

import torch
from torch import nn

from src.models.attention import SelfAttentionBlock, SpatialTransformer, AttentionType
from src.models.common import ConvND, Downsample, Normalize, Upsample
from src.utils.module_utils import zero_module


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, output_dim: int, max_period: int = 10000):
        super().__init__()

        self.output_dim = output_dim
        self.max_period = max_period
        self.register_buffer("freqs", self._get_freqs())

    def _get_freqs(self):
        from math import log

        half = self.output_dim // 2

        freqs = torch.exp(
            -log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).unsqueeze_(0)

        return freqs

    def forward(self, timesteps: torch.Tensor):
        args = timesteps.unsqueeze(-1) * self.freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.output_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        dim: int = 2,
        num_layers: int = 1,
        add_downsample: bool = True,
        use_skip_conv: bool = True,
        use_down_conv: bool = True,
        add_attention: bool = False,
        attn_type: AttentionType = "standard",
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
        use_spatial_transformer: bool = False
    ):
        super().__init__()

        self.use_spatial_transformer = use_spatial_transformer

        self.res_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels

            self.res_blocks.append(
                ResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    embedding_channels=embedding_channels,
                    use_conv=use_skip_conv,
                    dim=dim,
                    dropout=dropout,
                )
            )

        if add_attention:
            self.attention_blocks = nn.ModuleList()
            for i in range(num_layers):
                self.attention_blocks.append(
                    SelfAttentionBlock(
                        out_channels,
                        attn_type=attn_type,
                        num_attn_heads=num_attn_heads,
                        channels_per_head=channels_per_head,
                    )
                    if not use_spatial_transformer else
                    SpatialTransformer(
                        out_channels,
                        context_dim=context_dim,
                        num_attn_heads=num_attn_heads,
                        dim_per_head=channels_per_head,
                        attn_type=attn_type,
                        dropout=dropout
                    )
                )
        else:
            self.attention_blocks = None

        if add_downsample:
            self.downsampler = Downsample(out_channels, use_conv=use_down_conv, dim=dim)
        else:
            self.downsampler = None

    def forward(
        self,
        x: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ):
        hidden_states = []

        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x, embedding)

            if self.attention_blocks is not None:
                if self.use_spatial_transformer:
                    x = self.attention_blocks[i](x, context=context)
                else:
                    x = self.attention_blocks[i](x)

            hidden_states.append(x)

        if self.downsampler is not None:
            x = self.downsampler(x)
            hidden_states.append(x)

        return x, hidden_states

    @property
    def num_layers(self):
        return len(self.res_blocks)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        embedding_channels: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        dim: int = 2,
        num_layers: int = 1,
        add_upsample: bool = True,
        use_skip_conv: bool = True,
        use_up_conv: bool = True,
        add_attention: bool = False,
        attn_type: AttentionType = "standard",
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
        use_spatial_transformer: bool = False
    ):
        super().__init__()

        self.use_spatial_transformer = use_spatial_transformer

        self.res_blocks = nn.ModuleList()
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            res_in_channels = prev_out_channels if i == 0 else out_channels

            self.res_blocks.append(
                ResBlock(
                    in_channels=res_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    embedding_channels=embedding_channels,
                    use_conv=use_skip_conv,
                    dim=dim,
                    dropout=dropout,
                )
            )

        if add_attention:
            self.attention_blocks = nn.ModuleList()
            for i in range(num_layers):
                self.attention_blocks.append(
                    SelfAttentionBlock(
                        out_channels,
                        attn_type=attn_type,
                        num_attn_heads=num_attn_heads,
                        channels_per_head=channels_per_head,
                    )
                    if not use_spatial_transformer else
                    SpatialTransformer(
                        out_channels,
                        context_dim=context_dim,
                        num_attn_heads=num_attn_heads,
                        dim_per_head=channels_per_head,
                        attn_type=attn_type,
                        dropout=dropout
                    )
                )
        else:
            self.attention_blocks = None

        if add_upsample:
            self.upsampler = Upsample(out_channels, use_conv=use_up_conv, dim=dim)
        else:
            self.upsampler = None

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: List[torch.Tensor],
        embedding: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ):
        for i, res_block in enumerate(self.res_blocks):
            hs = hidden_states.pop()
            x = torch.cat((x, hs), dim=1)
            x = res_block(x, embedding)

            if self.attention_blocks is not None:
                if self.use_spatial_transformer:
                    x = self.attention_blocks[i](x, context=context)
                else:
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
        embedding_channels: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        dim: int = 2,
        num_layers: int = 1,
        use_skip_conv: bool = True,
        add_attention: bool = True,
        attn_type: AttentionType = "standard",
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
        use_spatial_transformer: bool = False
    ):
        super().__init__()

        self.use_spatial_transformer = use_spatial_transformer

        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels=in_channels,
                embedding_channels=embedding_channels,
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
                    embedding_channels=embedding_channels,
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
                    if not use_spatial_transformer else
                    SpatialTransformer(
                        in_channels,
                        context_dim=context_dim,
                        num_attn_heads=num_attn_heads,
                        dim_per_head=channels_per_head,
                        attn_type=attn_type,
                        dropout=dropout
                    )
                )
            else:
                self.attention_blocks.append(None)
        
        if add_attention:
            self.attention_blocks = nn.ModuleList(self.attention_blocks)

    def forward(
        self,
        x: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ):
        x = self.res_blocks[0](x, embedding)

        for attn, res_block in zip(self.attention_blocks, self.res_blocks[1:]):
            if attn is not None:
                if self.use_spatial_transformer:
                    x = attn(x, context=context)
                else:
                    x = attn(x)
            x = res_block(x, embedding)

        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embedding_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        dim: int = 2,
        dropout: float = 0.0,
    ):
        """A residual block with a timestep embedding.

        Args:
            :param channels: The number of channels in the input.
            :param embedding_channels: The number of channels in the timestep embedding.
            :param out_channels: The number of channels in the output. If None, it will be the same as the input.
            :param use_conv: if `True` and out_channels is specified, use a spatial
                convolution instead of a smaller 1x1 convolution to change the
                channels in the skip connection.
            :param dim: The dimensionality of the input.
            :param dropout: The dropout rate.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.embedding_channels = embedding_channels

        self.in_layers = nn.Sequential(
            Normalize(self.in_channels),
            nn.SiLU(inplace=True),
            ConvND(dim, self.in_channels, self.out_channels, 3, padding=1),
        )

        if self.embedding_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.embedding_channels, self.out_channels)
            )

        self.out_layers = nn.Sequential(
            Normalize(self.out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            zero_module(ConvND(dim, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.in_channels == self.out_channels:
            self.skip_connector = nn.Identity()
        elif use_conv:
            self.skip_connector = ConvND(dim, self.in_channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connector = ConvND(dim, self.in_channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, embedding: Optional[torch.Tensor] = None):
        """
        Args:
            :param x: The input tensor.
            :param embedding: The timestep embedding.
        """
        h = self.in_layers(x)

        if embedding is not None:
            assert self.embedding_channels is not None, "Model does not have embedding module"
            emb_out = self.emb_layers(embedding).type_as(h)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out.unsqueeze_(-1)

            h = h + emb_out
        h = self.out_layers(h)

        return self.skip_connector(x) + h
