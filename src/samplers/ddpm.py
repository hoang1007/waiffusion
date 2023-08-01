from functools import partial

import torch

from .base import BaseSampler

from src.utils.module_utils import default, exists
from src.models.diffusion.modules import (
    extract_into_tensor,
    make_beta_schedule,
    noise_like,
)


class DDPMSampler(BaseSampler):
    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        clip_denoised=True,
        v_posterior=0.0, # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        parameterization="eps",
    ):
        super().__init__(num_timesteps=timesteps)

        self.clip_denoised = clip_denoised
        self.parameterization = parameterization
        self.v_posterior = v_posterior

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # alphas_cumprod_prev = torch.append(1.0, alphas_cumprod[:-1])
        alphas_cumprod_prev = torch.cat((1.0, alphas_cumprod[:-1]))

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.asarray, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "sqrt_alphas_cumprod", to_torch(torch.sqrt(alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(torch.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(torch.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(torch.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(torch.sqrt(1.0 / alphas_cumprod - 1)),
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(torch.log(torch.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev)
                * torch.sqrt(alphas)
                / (1.0 - alphas_cumprod)
            ),
        )

        # if self.parameterization == "eps":
        #     lvlb_weights = self.betas**2 / (
        #         2
        #         * self.posterior_variance
        #         * to_torch(alphas)
        #         * (1 - self.alphas_cumprod)
        #     )
        # elif self.parameterization == "x0":
        #     lvlb_weights = (
        #         0.5
        #         * torch.sqrt(torch.Tensor(alphas_cumprod))
        #         / (2.0 * 1 - torch.Tensor(alphas_cumprod))
        #     )
        # else:
        #     raise NotImplementedError("mu not supported")
        # # TODO how to choose this term
        # lvlb_weights[0] = lvlb_weights[1]
        # self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        # assert not torch.isnan(self.lvlb_weights).all()

    def __get_variance(self, t, predicted_variance=None, variance_type=None):
        

    def __predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def __q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def step(self, sample: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        noise = default(noise, noise_like(sample.shape, sample.device))

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, sample.shape) * sample
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, sample.shape) * noise
        )
    
    @torch.no_grad()
    def reverse_step(self, model_output: torch.Tensor, t: torch.Tensor, sample: torch.Tensor):
        batch_size = model_output.size(0)
        
        if self.parameterization == "eps":
            x_recon = self.__predict_start_from_noise(sample, t=t, noise=model_output)
        elif self.parameterization == "x0":
            x_recon = model_output

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, _, model_log_variance = self.__q_posterior(
            x_start=x_recon, x_t=sample, t=t
        )
        
        noise = noise_like(sample.shape, sample.device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, *((1,) * (len(sample.shape) - 1)))

        pred_prev_sample = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_prev_sample
    
