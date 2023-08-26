import pytest
import torch

from src.models.vae.autoencoder_kl import AutoencoderKL


@pytest.mark.parametrize("batch_size", [4, 32])
def test_unet(batch_size):
    model = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        hidden_channels=[32, 64],
        num_res_blocks=2,
        attention_levels=(1,),
    )

    x = torch.randn(batch_size, 3, 32, 32)
    out = model(x)

    assert out.shape == x.shape
