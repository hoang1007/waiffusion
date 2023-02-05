import torch


def zero_module(module: torch.nn.Module):
    """Zero the parameters of a module."""

    for param in module.parameters():
        param.data.zero_()

    return module
