from abc import abstractmethod
from math import sqrt
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from src.utils.module_utils import zero_module

from .common import ConvND, Normalize

AttentionType = Literal["standard", "efficient", "flash"]


class IAttention(nn.Module):
    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (torch.Tensor): The tensor of query. Shape: (B, S, H, D)
            k (torch.Tensor): The tensor of key. Shape: (B, S', H, D)
            v (torch.Tensor): The tensor of value. Shape: (B, S', H, D)
            key_padding_mask (torch.BoolTensor): A bool tensor. `True` indices for a valid token. Shape: (B, S)

        Returns:
            (torch.Tensor) The tensor of attention output. Shape: (B, S, H, D)
            (torch.Tensor) The tensor of attention weights. Shape: (B, S, H, S)
        """
        pass


class StandardAttention(IAttention):
    def __init__(self, dropout: float = 0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, H, D = q.shape

        q = rearrange(q, "B T H D -> (B H) T D")
        k = rearrange(k, "B S H D -> (B H) S D")
        v = rearrange(v, "B S H D -> (B H) S D")

        scale = 1 / sqrt(sqrt(D))
        logits = einsum(q * scale, k * scale, "B T D, B S D -> B T S")

        if key_padding_mask is not None:
            logits = logits.masked_fill(~key_padding_mask, -torch.finfo(logits.dtype))

        # # Numberical stability
        # logits = logits - logits.max(dim=-1, keepdim=True).values

        probs = F.softmax(logits, dim=-1)
        probs = self.dropout(probs)

        attn = einsum(probs, v, "B T S, B S D -> B T D")
        attn = rearrange(attn, "(B H) T D -> B T H D", H=H)
        probs = rearrange(probs, "(B H) T S -> B T H S", H=H)

        return attn, probs


class EfficientAttention(IAttention):
    def __init__(self, dropout: float = 0.0):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            key_padding_mask is None
        ), "key_padding_mask is not supported for EfficientAttention"

        _, _, H, D = q.shape

        attn_vals = []
        for i in range(H):
            hk = F.softmax(k[:, :, i], dim=1)  # (B, S, D)
            hq = F.softmax(q[:, :, i], dim=2)  # (B, S, D)
            hv = v[:, :, i]  # (B, S, D)

            context = einsum(hk, hv, "B S DK, B S DV -> B DK DV")
            attn_val = einsum(context, hq, "B DK DV, B S DK-> B S DV")
            attn_vals.append(attn_val)

        # (B, S, H, D)
        attn_vals = torch.stack(attn_vals, dim=2)

        return attn_vals, None


class FlashAttention(IAttention):
    def __init__(self, dropout: float = 0.0):
        super().__init__()

        try:
            from flash_attn.flash_attention import FlashAttention as _FlashAttention

            self.attn = _FlashAttention(attention_dropout=dropout)
        except ImportError:
            raise ImportError("Please install flash_attn: pip install flash_attn")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv = torch.stack((q, k, v), dim=2)
        dtype = qkv.dtype
        return self.attn(qkv.type(torch.half), key_padding_mask).type(dtype)


def get_attention(attn_type: AttentionType, *args, **kwargs) -> IAttention:
    if attn_type == "standard":
        return StandardAttention(*args, **kwargs)
    elif attn_type == "efficient":
        return EfficientAttention(*args, **kwargs)
    elif attn_type == "flash":
        return FlashAttention(*args, **kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        attn_type: AttentionType = "standard",
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
    ):
        """Attention block for spatial input.

        Args:
            :param channels: The number of channels in the input tensor.
            :param attn: The attention module.
            :param num_attn_heads: The number of attention heads.
            :param channels_per_head: The number of channels per head. If specified, ignore `num_attn_heads` and instead use
                               a fixed channel width per attention head. Default: -1
        """
        super().__init__()

        self.channels = channels

        if channels_per_head != -1:
            assert (
                channels % channels_per_head == 0
            ), "The number of channels must be divisible by the number of channels per head"
            self.num_attn_heads = channels // channels_per_head
        else:
            assert (
                channels % num_attn_heads == 0
            ), "The number of channels must be divisible by the number of attention heads"
            self.num_attn_heads = num_attn_heads

        self.norm = Normalize(channels)
        self.qkv = ConvND(1, channels, channels * 3, 1)

        self.attn = get_attention(attn_type)
        self.proj_out = zero_module(ConvND(1, channels, channels, 1))

    def forward(self, x: torch.Tensor):
        b, c, *spatial = x.shape
        assert (
            c == self.channels
        ), f"Channels must be equals to the number of channels in the constructor (={self.channels})"

        x = x.reshape(b, c, -1)
        # (B, 3 * C, T)
        qkv = self.qkv(self.norm(x))
        qkv = rearrange(
            qkv, "B (three H C) T -> three B T H C", three=3, H=self.num_attn_heads
        )
        q, k, v = torch.unbind(qkv)

        # h.shape == (B, T, H, C // H)
        h, attn_weights = self.attn(q, k, v)
        h = rearrange(h, "B T H C -> B (H C) T")
        h = self.proj_out(h)

        out = (h + x).reshape(b, c, *spatial)
        return out


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        num_attn_heads: int = 8,
        dim_per_head: int = -1,
        attn_core: AttentionType = "standard",
        dropout: float = 0.0,
    ):
        super().__init__()

        if dim_per_head == -1:
            dim_per_head == query_dim // num_attn_heads

        hidden_dim = num_attn_heads * dim_per_head
        if context_dim is None:
            context_dim = query_dim
        self.num_attn_heads = num_attn_heads

        self.attn_core = get_attention(attn_core)

        self.q_proj = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, query_dim), nn.Dropout(p=dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        if context is None:
            context = x

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q, k, v = map(
            lambda t: rearrange(t, "B S (H D) -> B S H D", H=self.num_attn_heads),
            (q, k, v)
        )
        attn_out, attn_weights = self.attn_core(q, k, v, key_padding_mask)
        attn_out = rearrange(attn_out, "B S H D -> B S (H D)")
        out = self.out_proj(attn_out)

        return out


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        context_dim: Optional[int] = None,
        num_attn_heads: int = 8,
        dim_per_head: int = 64,
        gated_feedforward: bool = True,
        dropout: float = 0.0,
        attn_type: AttentionType = "standard",
    ):
        super().__init__()

        self.attn1 = CrossAttentionLayer(
            query_dim=input_dim,
            num_attn_heads=num_attn_heads,
            dim_per_head=dim_per_head,
            attn_core=attn_type,
            dropout=dropout
        )
        self.feedforward = FeedForward(input_dim, glu=gated_feedforward, dropout=dropout)
        self.attn2 = CrossAttentionLayer(
            query_dim=input_dim,
            context_dim=context_dim, 
            num_attn_heads=num_attn_heads,
            dim_per_head=dim_per_head,
            attn_core=attn_type,
            dropout=dropout
        )
    
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        x = self.attn1(self.norm1(x), key_padding_mask=key_padding_mask) + x
        x = self.attn2(self.norm2(x), context=context, key_padding_mask=key_padding_mask) + x
        x = self.feedforward(self.norm3(x)) + x

        return x


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        context_dim: Optional[int] = None,
        num_attn_heads: int = 8,
        dim_per_head: int = -1,
        num_layers: int = 1,
        attn_type: AttentionType = "standard",
        dropout: float = 0.0
    ):
        super().__init__()

        if dim_per_head == -1:
            dim_per_head = in_channels // num_attn_heads

        self.in_channels = in_channels
        hidden_dim = num_attn_heads * dim_per_head

        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                input_dim=hidden_dim,
                context_dim=context_dim,
                num_attn_heads=num_attn_heads,
                dim_per_head=dim_per_head,
                dropout=dropout,
                attn_type=attn_type
            ) for _ in range(num_layers)
        ])

        self.out_proj = zero_module(nn.Conv2d(hidden_dim, in_channels, kernel_size=1))
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        H, W = x.shape[2:]

        h = self.norm(x)
        h = self.proj_in(h)
        h = rearrange(h, 'B C H W -> B (H W) C')

        for block in self.transformer_blocks:
            h = block(h, context=context)
        
        h = rearrange(h, 'B (H W) C -> B C H W', H=H, W=W)
        h = self.out_proj(h)

        return h + x
