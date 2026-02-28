from __future__ import annotations

import math
import os
from typing import Union

import torch


def resolve_num_threads(num_threads: Union[int, str, None]) -> int:
    total_cpus = os.cpu_count() or 1
    if num_threads is None:
        return min(10, total_cpus)
    if num_threads == "max":
        return max(1, math.floor(total_cpus * 0.8))
    if not isinstance(num_threads, int):
        raise ValueError("num_threads must be an integer, 'max', or None")
    if num_threads < 1:
        raise ValueError("num_threads must be >= 1")
    return min(num_threads, total_cpus)


def configure_torch_threads(num_threads: Union[int, str, None]) -> int:
    resolved = resolve_num_threads(num_threads)
    torch.set_num_threads(resolved)
    return resolved


def resolve_device(device: str = "auto") -> torch.device:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device='cuda' was requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
