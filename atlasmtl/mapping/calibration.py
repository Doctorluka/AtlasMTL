from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class TemperatureScaling:
    """Per-head temperature scaling calibrator.

    Temperature scaling rescales logits as `logits / temperature` before softmax.
    A larger temperature yields softer (less confident) probabilities.
    """

    temperature: float

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        return logits / float(self.temperature)


def fit_temperature_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_iter: int = 100,
    lr: float = 0.05,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> TemperatureScaling:
    """Fit a single temperature scalar by minimizing NLL on a calibration set."""
    if logits.ndim != 2:
        raise ValueError("logits must be 2D: (n_samples, n_classes)")
    if targets.ndim != 1:
        raise ValueError("targets must be 1D: (n_samples,)")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have matching first dimension")
    if logits.shape[0] == 0:
        raise ValueError("cannot fit temperature scaling with empty calibration set")

    resolved_device = device or (logits.device if logits.is_cuda else torch.device("cpu"))
    logits = logits.detach().to(resolved_device)
    targets = targets.detach().to(resolved_device)

    log_temp = torch.nn.Parameter(torch.zeros((), device=resolved_device))
    opt = torch.optim.Adam([log_temp], lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(max_iter):
        opt.zero_grad(set_to_none=True)
        temp = torch.exp(log_temp).clamp_min(1e-6)
        loss = loss_fn(logits / temp, targets)
        loss.backward()
        opt.step()

    temperature = float(torch.exp(log_temp).clamp_min(1e-6).detach().cpu().item())
    return TemperatureScaling(temperature=temperature)

