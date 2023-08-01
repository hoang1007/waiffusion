import torch


class BaseSampler(torch.nn.Module):
    def __init__(self, num_train_timesteps: int):
        super().__init__()
        self.__num_train_timesteps = num_train_timesteps
        self.__num_inference_steps = None

    @property
    def num_train_timesteps(self):
        return self.__num_train_timesteps

    @property
    def num_inference_timesteps(self):
        assert (
            self.__num_inference_steps is not None
        ), "Number of inference steps is not set!"
        return self.__num_inference_steps

    def set_timesteps(self, num_timesteps: int):
        if num_timesteps > self.num_train_timesteps:
            raise ValueError(
                f"Number of inference steps ({num_timesteps}) cannot be large than number of train steps ({self.num_train_timesteps})"
            )
        else:
            self.__num_inference_steps = num_timesteps

    def step(
        self, sample: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a single step of the diffusion process.

        Args:
            sample (torch.Tensor): Sample with noiseless (Input).
            t (torch.Tensor): Timesteps in the diffusion chain.
            noise (torch.Tensor): The noise tensor for the current timestep.

        Returns:
            torch.Tensor: The noisy samples.
        """

    def reverse_step(
        self, model_output: torch.Tensor, t: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
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
