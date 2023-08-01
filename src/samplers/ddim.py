from typing import Literal, Optional
from .base import BaseSampler
from src.utils.module_utils import default, exists
import math
import torch


class DDIMSampler(BaseSampler):
    def __init__(
            self,
            num_timesteps: int,
            beta_start: float = 1e-4,
            beta_end: float = 2e-2,
            beta_schedule: Literal["linear", "scaled_linear", "squaredcos_cap_v2"] = "linear",
            given_betas: Optional[torch.Tensor] = None,
            set_alpha_to_one: bool = True,
            thresholding: bool = False,
            clip_denoised: bool = True,
            eta: float = 0.0,
            use_clipped_model_output: bool = False,
            parameterization: Literal["eps", "x0"] = "eps"
        ):
        super().__init__(num_timesteps)

        self.clip_denoised = clip_denoised
        self.parameterization = parameterization
        self.thresholding = thresholding
        self.use_clipped_model_output = use_clipped_model_output
        self.eta = eta

        self.register_schedule(
            given_betas=given_betas,
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            set_alpha_to_one=set_alpha_to_one
        )
    
    def register_schedule(
        self,
        given_betas=None,
        num_timesteps=1000,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=2e-2,
        set_alpha_to_one=True
    ):
        if exists(given_betas):
            betas= given_betas
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            betas = betas_for_alpha_bar(num_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not supported!")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else alphas_cumprod[0]

        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = torch.arange(0, num_timesteps)[::-1].long()

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("final_alpha_cumprod", final_alpha_cumprod)


    def __get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
    
    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def __threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, height, width = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * height * width)

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]

        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, height, width)
        sample = sample.to(dtype)

        return sample
    
    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        timesteps = torch.linspace(0, self.num_timesteps - 1, num_inference_steps)[::-1].long()

        self.timesteps = timesteps

    def step(self, sample: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alphas_prod = self.sqrt_alphas_cumprod[t].flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].flatten()

        while sqrt_alphas_prod.ndim < sample.ndim:
            sqrt_alphas_prod = sqrt_alphas_prod.unsqueeze(-1)

        while sqrt_one_minus_alpha_prod.ndim < sample.ndim:
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_sample = sqrt_alphas_prod * sample + sqrt_one_minus_alpha_prod * noise
        return noisy_sample

    @torch.no_grad()
    def reverse_step(self, model_output: torch.Tensor, t: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        assert self.num_inference_steps is not None, "Number of inference steps is not set"

        # 1. get previous step value (=t-1)
        eta = self.eta
        prev_t = t - self.num_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        sqrt_alpha_prod_t = self.sqrt_alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        sqrt_alpha_prod_t_prev = self.sqrt_alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod ** 0.5

        sqrt_beta_prod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.parameterization == "eps":
            pred_org_sample = (sample - sqrt_beta_prod_t * model_output) / sqrt_alpha_prod_t
            pred_epsilon = model_output
        elif self.parameterization == "x0":
            pred_org_sample = model_output
            pred_epsilon = (sample - sqrt_alpha_prod_t * pred_org_sample) / sqrt_beta_prod_t
        
        # 4. Clip or threshold "predicted x_0"
        if self.thresholding:
            pred_org_sample = self.__threshold_sample(pred_org_sample)
        elif self.clip_denoised:
            pred_org_sample.clamp_(-1.0, 1.0)   

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self.__get_variance(t, prev_t)
        std_dev_t = eta * variance ** 0.5

        if self.use_clipped_model_output:
            pred_epsilon = (sample - sqrt_alpha_prod_t * pred_org_sample) / sqrt_beta_prod_t
        
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = sqrt_alpha_prod_t_prev * pred_org_sample + pred_sample_direction

        if eta > 0:
            variance_noise = torch.randn_like(model_output)
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        return prev_sample


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
