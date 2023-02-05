import sys
sys.path.append('.')

from src.models.unet import Unet
from src.models.attention import FlashAttention, StandardAttention, EfficientAttention
import torch

torch.cuda.empty_cache()
device = "cuda"
model = Unet(3, 2, [64, 32], 2, EfficientAttention(), [1], channels_per_head=16).to(device)
out = model(
    torch.rand(2, 3, 32, 32, device=device),
    torch.tensor([0, 1], device=device)
)

print(out.shape)