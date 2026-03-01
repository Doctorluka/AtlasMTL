from __future__ import annotations

from typing import Dict, List, TypedDict


class ModelPreset(TypedDict, total=False):
    hidden_sizes: List[int]
    dropout_rate: float
    learning_rate: float


PRESETS: Dict[str, ModelPreset] = {
    "small": {"hidden_sizes": [128, 64], "dropout_rate": 0.2, "learning_rate": 1e-3},
    "default": {"hidden_sizes": [256, 128], "dropout_rate": 0.3, "learning_rate": 1e-3},
    "large": {"hidden_sizes": [512, 256], "dropout_rate": 0.3, "learning_rate": 5e-4},
}


def resolve_preset(name: str) -> ModelPreset:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {', '.join(sorted(PRESETS))}")
    return PRESETS[name]

