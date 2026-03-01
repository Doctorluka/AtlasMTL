"""atlasmtl: reference-aware multi-task single-cell annotation."""

from .version import __version__
from .core.api import build_model, predict, TrainedModel, PredictionResult
from .core.evaluate import evaluate_predictions
from .preprocess import FeaturePanel, PreprocessConfig, PreprocessReport, preprocess_query, preprocess_reference

__all__ = [
    "__version__",
    "build_model",
    "predict",
    "TrainedModel",
    "PredictionResult",
    "evaluate_predictions",
    "FeaturePanel",
    "PreprocessConfig",
    "PreprocessReport",
    "preprocess_reference",
    "preprocess_query",
]
