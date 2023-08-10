import math
from typing import Literal, Optional, TypeAlias, Union

import torch

BetaSchedule: TypeAlias = Literal["linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"]


def make_beta_schedule(
    num_timesteps: int,
    beta_schedule: BetaSchedule = "linear",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: Optional[Union[torch.device, str]] = None,
):
    if beta_schedule == "linear":
        betas = torch.linspace(
            beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device
        )
    elif beta_schedule == "scaled_linear":
        # this schedule is very specific to the latent diffusion model.
        betas = torch.linspace(
            beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32, device=device
        ).pow_(2.0)
    elif beta_schedule == "squaredcos_cap_v2":
        # Glide cosine schedule
        betas = betas_for_alpha_bar(num_timesteps)
    elif beta_schedule == "sigmoid":
        # GeoDiff sigmoid schedule
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"{beta_schedule} is not supported")

    return betas


def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)
