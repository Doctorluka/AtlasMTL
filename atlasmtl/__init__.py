"""atlasmtl: reference-aware multi-task single-cell annotation."""

from .version import __version__
from .core.api import build_model, predict, TrainedModel, PredictionResult

__all__ = [
    "__version__",
    "build_model",
    "predict",
    "TrainedModel",
    "PredictionResult",
]
