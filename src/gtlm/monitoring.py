"""Small monitoring helpers for long training runs."""

from __future__ import annotations

import math

import torch


def is_invalid_number(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(torch.isnan(value).any().item() or torch.isinf(value).any().item())
    value = float(value)
    return math.isnan(value) or math.isinf(value)


def as_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def get_grad_norm(model_engine) -> float:
    if not hasattr(model_engine, "get_global_grad_norm"):
        return 0.0
    return as_float(model_engine.get_global_grad_norm())


def cuda_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def max_cuda_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9
