import torch


class BaseSampler(torch.nn.Module):
    def __init__(self, num_timesteps: int):
        super().__init__()
        self.num_timesteps = num_timesteps

    def step(self, sample: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Perform a single step of the diffusion process.

        Args:
            sample (torch.Tensor): Sample with noiseless (Input).
            t (torch.Tensor): Timesteps in the diffusion chain.
            noise (torch.Tensor): The noise tensor for the current timestep.

        Returns:
            torch.Tensor: The noisy samples.
        """

    def reverse_step(self, model_output: torch.Tensor, t: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """
        Predict the sample at the previous timestep by reversing the SDE.

        Args:
            model_output (torch.Tensor): The output of the diffusion model.
            t (int): Timesteps in the diffusion chain.
            sample (torch.Tensor): Current instance of sample being created by diffusion process.

        Returns:
            torch.Tensor: Samples at the previous timesteps.
        """
        raise NotImplementedError
