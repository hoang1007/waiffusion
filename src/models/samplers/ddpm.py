from typing import Literal, Optional

import torch

from src.models.schedule import BetaSchedule, make_beta_schedule
from src.utils.module_utils import exists

from .base import BaseSampler

VarianceType = Literal[
    "fixed_small", "fixed_small_log", "fixed_large", "fixed_large_log", "learned", "learned_range"
]


def expand_dim_like(x, y):
    while x.ndim < y.ndim:
        x = x.unsqueeze(-1)
    return x


class DDPMSampler(BaseSampler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: BetaSchedule = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        given_betas: Optional[torch.Tensor] = None,
        variance_type: VarianceType = "fixed_small",
        clip_denoised=True,
        parameterization="eps",
    ):
        super().__init__(num_train_timesteps=num_train_timesteps)

        self.clip_denoised = clip_denoised
        self.parameterization = parameterization
        self.variance_type = variance_type

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            num_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        self.set_timesteps(num_train_timesteps)

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                num_timesteps,
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
            )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        one = torch.ones(1, dtype=torch.float32)

        self.init_noise_sigma = 1.0
        self.timesteps = torch.arange(0, num_timesteps).flip(0).long()

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("final_alpha_cumprod", one)

    def set_timesteps(self, num_timesteps: int):
        super().set_timesteps(num_timesteps)

        step_ratio = self.num_train_timesteps // self.num_inference_timesteps
        self.timesteps = (
            (torch.arange(0, self.num_inference_timesteps) * step_ratio).flip(0).long()
        )

    def __get_variance(
        self, t, predicted_variance=None, variance_type: VarianceType = "fixed_small"
    ):
        prev_t = t - self.num_train_timesteps // self.num_inference_timesteps

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = torch.where(
            prev_t >= 0, self.alphas_cumprod[prev_t], self.final_alpha_cumprod
        )
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            variance = torch.clamp(variance, min=1e-20)
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_small_log":
            variance = torch.log(torch.clamp(variance, min=1e-20))
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = torch.log(current_beta_t)
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = torch.log(variance)
            max_log = torch.log(self.betas[t])
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log
        else:
            raise ValueError(f"Unknown variance type {variance_type}")

        return variance

    def step(self, sample: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        sqrt_alphas_prod = self.sqrt_alphas_cumprod[t].flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].flatten()

        sqrt_alphas_prod = expand_dim_like(sqrt_alphas_prod, sample)
        sqrt_one_minus_alpha_prod = expand_dim_like(sqrt_one_minus_alpha_prod, sample)

        noisy_sample = sqrt_alphas_prod * sample + sqrt_one_minus_alpha_prod * noise
        return noisy_sample

    @torch.no_grad()
    def reverse_step(self, model_output: torch.Tensor, t: torch.Tensor, sample: torch.Tensor):
        prev_t = t - self.num_train_timesteps // self.num_inference_timesteps

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in (
            "learned",
            "learned_range",
        ):
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = torch.where(
            prev_t >= 0, self.alphas_cumprod[prev_t], self.final_alpha_cumprod
        )

        alpha_prod_t = expand_dim_like(alpha_prod_t, sample)
        alpha_prod_t_prev = expand_dim_like(alpha_prod_t_prev, sample)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.parameterization == "eps":
            pred_org_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (
                0.5
            )
        elif self.parameterization == "x0":
            pred_org_sample = model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.parameterization} must be one of `epsilon`, `sample` or"
                " for the DDPMSampler."
            )

        if self.clip_denoised:
            pred_org_sample.clamp_(-1.0, 1.0)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
            pred_original_sample_coeff * pred_org_sample + current_sample_coeff * sample
        )

        # 6. Add noise
        variance = torch.zeros_like(model_output)
        noise = torch.randn_like(model_output)
        ids = t > 0
        v = expand_dim_like(
            self.__get_variance(t[ids], predicted_variance=predicted_variance), model_output
        )

        if self.variance_type == "fixed_small_log":
            variance[ids] = v * noise[ids]
        elif self.variance_type == "learned_range":
            variance[ids] = v
            variance = torch.exp(0.5 * variance) * noise
        else:
            variance[ids] = (v**0.5) * noise[ids]

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
