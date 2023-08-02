from abc import abstractmethod
from math import sqrt
from typing import Optional, Literal, Tuple

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
        self, qkv: torch.Tensor, key_padding_mask: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            :param qkv: The tensor of query, key, value. Shape: (B, S, 3, H, D)
            :param key_padding_mask: A bool tensor. `True` indices for a valid token. Shape: (B, S)

        Returns:
            :return output: The tensor of attention output. Shape: (B, S, H, D)
            :return attn: The tensor of attention weights. Shape: (B, S, H, S)
        """
        pass


class StandardAttention(IAttention):
    def __init__(self, dropout: float = 0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, qkv: torch.Tensor, key_padding_mask: Optional[torch.BoolTensor] = None
    ):
        q, k, v = torch.unbind(qkv, dim=2)
        _, _, H, D = q.shape

        q = rearrange(q, "B T H D -> (B H) T D")
        k = rearrange(k, "B S H D -> (B H) S D")
        v = rearrange(v, "B S H D -> (B H) S D")

        scale = 1 / sqrt(sqrt(D))
        logits = einsum(q * scale, k * scale, "B T D, B S D -> B T S")

        if key_padding_mask is not None:
            logits = logits.masked_fill(~key_padding_mask, -1e9)

        # Numberical stability
        logits = logits - logits.max(dim=-1, keepdim=True).values

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
        self, qkv: torch.Tensor, key_padding_mask: Optional[torch.BoolTensor] = None
    ):
        assert (
            key_padding_mask is None
        ), "key_padding_mask is not supported for EfficientAttention"

        q, k, v = torch.unbind(qkv, dim=2)
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
        self, qkv: torch.Tensor, key_padding_mask: Optional[torch.BoolTensor] = None
    ):
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


class AttentionBlock(nn.Module):
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
            qkv, "B (three H C) T -> B T three H C", three=3, H=self.num_attn_heads
        )

        # h.shape == (B, T, H, C // H)
        h, attn_weights = self.attn(qkv)
        h = rearrange(h, "B T H C -> B (H C) T")
        h = self.proj_out(h)

        out = (h + x).reshape(b, c, *spatial)
        return out
