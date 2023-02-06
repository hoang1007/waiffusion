import torch
from inspect import isfunction


def zero_module(module: torch.nn.Module):
    """Zero the parameters of a module."""

    for param in module.parameters():
        param.data.zero_()

    return module


def scale_module(module: torch.nn.Module, scale: float):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor: torch.Tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=tuple(range(1, len(tensor.shape))))


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
