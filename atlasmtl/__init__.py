"""atlasmtl: reference-aware multi-task single-cell annotation."""

from .version import __version__
from .core.api import build_model, predict, TrainedModel, PredictionResult
from .core.evaluate import evaluate_predictions
from .preprocess import (
    FeaturePanel,
    PreprocessConfig,
    PreprocessReport,
    load_feature_panel,
    preprocess_query,
    preprocess_reference,
    save_feature_panel,
)

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
    "load_feature_panel",
    "preprocess_reference",
    "preprocess_query",
    "save_feature_panel",
]
