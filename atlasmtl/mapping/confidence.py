from __future__ import annotations

from typing import Tuple

import numpy as np


def mtl_confidence(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    max_prob = top2[:, 1]
    margin = top2[:, 1] - top2[:, 0]
    return max_prob, margin


def mtl_low_conf_mask(
    max_prob: np.ndarray,
    margin: np.ndarray,
    confidence_high: float,
    margin_threshold: float,
) -> np.ndarray:
    return (max_prob < confidence_high) | (margin < margin_threshold)


def closed_loop_unknown_mask(
    mtl_unknown: np.ndarray,
    mtl_low_conf: np.ndarray,
    used_knn: np.ndarray,
    knn_low_conf: np.ndarray,
) -> np.ndarray:
    return mtl_unknown | (used_knn & mtl_low_conf & knn_low_conf)
