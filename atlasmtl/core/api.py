from .data import extract_matrix as _extract_matrix
from .predict import predict
from .train import build_model
from .types import PredictionResult, TrainedModel

__all__ = ["build_model", "predict", "TrainedModel", "PredictionResult", "_extract_matrix"]
