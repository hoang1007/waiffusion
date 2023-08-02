import pytest

import torch
from src.models.unet import Unet

@pytest.mark.parametrize("batch_size", [4, 32])
def test_unet(batch_size):
    model = Unet(
        in_channels=3,
        out_channels=3,
        hidden_channels=[32, 64, 128],
        num_res_blocks=2,
        attention_levels=(1,),
    )

    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, 10, (batch_size,))
    out = model(x, t)

    assert out.shape == x.shape
