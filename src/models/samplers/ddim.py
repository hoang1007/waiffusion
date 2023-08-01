from typing import Literal, Optional

import torch

from src.models.schedule import BetaSchedule, make_beta_schedule
from src.utils.module_utils import exists

from .base import BaseSampler


def expand_dim_like(x, y):
    while x.ndim < y.ndim:
        x = x.unsqueeze(-1)
    return x


class DDIMSampler(BaseSampler):
    def __init__(
        self,
        num_train_timesteps: int,
        num_inference_timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        beta_schedule: BetaSchedule = "linear",
        given_betas: Optional[torch.Tensor] = None,
        set_alpha_to_one: bool = True,
        thresholding: bool = False,
        clip_denoised: bool = True,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        parameterization: Literal["eps", "x0"] = "eps",
    ):
        super().__init__(num_train_timesteps)

        self.clip_denoised = clip_denoised
        self.parameterization = parameterization
        self.thresholding = thresholding
        self.use_clipped_model_output = use_clipped_model_output
        self.eta = eta

        self.register_schedule(
            given_betas=given_betas,
            num_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            set_alpha_to_one=set_alpha_to_one,
        )

        self.set_timesteps(num_inference_timesteps)

    def register_schedule(
        self,
        given_betas=None,
        num_timesteps=1000,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=2e-2,
        set_alpha_to_one=True,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(num_timesteps, beta_schedule, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else alphas_cumprod[0]

        self.init_noise_sigma = 1.0

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("final_alpha_cumprod", final_alpha_cumprod)

    def __get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = torch.where(
            prev_timestep >= 0, self.alphas_cumprod[prev_timestep], self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    # # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    # def __threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
    #     """
    #     "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
    #     prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
    #     s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
    #     pixels from saturation at each step. We find that dynamic thresholding results in significantly better
    #     photorealism as well as better image-text alignment, especially when using very large guidance weights."

    #     https://arxiv.org/abs/2205.11487
    #     """
    #     dtype = sample.dtype
    #     batch_size, channels, height, width = sample.shape

    #     if dtype not in (torch.float32, torch.float64):
    #         sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

    #     # Flatten sample for doing quantile calculation along each image
    #     sample = sample.reshape(batch_size, channels * height * width)

    #     abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

    #     s = torch.quantile(abs_sample, self.dynamic_thresholding_ratio, dim=1)
    #     s = torch.clamp(
    #         s, min=1, max=self.config.sample_max_value
    #     )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]

    #     s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
    #     sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

    #     sample = sample.reshape(batch_size, channels, height, width)
    #     sample = sample.to(dtype)

    #     return sample

    def set_timesteps(self, num_inference_steps: int):
        super().set_timesteps(num_inference_steps)

        self.timesteps = (
            torch.linspace(0, self.num_train_timesteps - 1, num_inference_steps).flip(0).long()
        )

    def step(self, sample: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alphas_prod = self.sqrt_alphas_cumprod[t].flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].flatten()

        sqrt_alphas_prod = expand_dim_like(sqrt_alphas_prod, sample)
        sqrt_one_minus_alpha_prod = expand_dim_like(sqrt_one_minus_alpha_prod, sample)

        noisy_sample = sqrt_alphas_prod * sample + sqrt_one_minus_alpha_prod * noise
        return noisy_sample

    @torch.no_grad()
    def reverse_step(
        self, model_output: torch.Tensor, t: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        # 1. get previous step value (=t-1)
        eta = self.eta
        prev_t = t - self.num_train_timesteps // self.num_inference_timesteps

        # 2. compute alphas, betas
        sqrt_alpha_prod_t = self.sqrt_alphas_cumprod[t]
        alpha_prod_t_prev = torch.where(
            prev_t >= 0, self.alphas_cumprod[prev_t], self.final_alpha_cumprod
        )
        sqrt_alpha_prod_t_prev = torch.where(
            prev_t >= 0, self.sqrt_alphas_cumprod[prev_t], self.final_alpha_cumprod**0.5
        )
        sqrt_beta_prod_t = self.sqrt_one_minus_alphas_cumprod[t]

        sqrt_alpha_prod_t = expand_dim_like(sqrt_alpha_prod_t, sample)
        alpha_prod_t_prev = expand_dim_like(alpha_prod_t_prev, sample)
        sqrt_alpha_prod_t_prev = expand_dim_like(sqrt_alpha_prod_t_prev, sample)
        sqrt_beta_prod_t = expand_dim_like(sqrt_beta_prod_t, sample)

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.parameterization == "eps":
            pred_org_sample = (sample - sqrt_beta_prod_t * model_output) / sqrt_alpha_prod_t
            pred_epsilon = model_output
        elif self.parameterization == "x0":
            pred_org_sample = model_output
            pred_epsilon = (sample - sqrt_alpha_prod_t * pred_org_sample) / sqrt_beta_prod_t

        # 4. Clip or threshold "predicted x_0"
        # if self.thresholding:
        #     pred_org_sample = self.__threshold_sample(pred_org_sample)
        # elif self.clip_denoised:
        #     pred_org_sample.clamp_(-1.0, 1.0)
        if self.clip_denoised:
            pred_org_sample.clamp_(-1.0, 1.0)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = expand_dim_like(self.__get_variance(t, prev_t), sample)
        std_dev_t = eta * variance**0.5

        if self.use_clipped_model_output:
            pred_epsilon = (sample - sqrt_alpha_prod_t * pred_org_sample) / sqrt_beta_prod_t

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = sqrt_alpha_prod_t_prev * pred_org_sample + pred_sample_direction

        if eta > 0:
            variance_noise = torch.randn_like(model_output)
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        return prev_sample
