from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def nn_distance_score(ref_coords: np.ndarray, query_coords: np.ndarray) -> np.ndarray:
    """Open-set score based on distance to the nearest reference neighbor."""
    if ref_coords.ndim != 2 or query_coords.ndim != 2:
        raise ValueError("ref_coords and query_coords must be 2D arrays")
    if ref_coords.shape[1] != query_coords.shape[1]:
        raise ValueError("ref_coords and query_coords must have the same feature dimension")
    if len(ref_coords) == 0:
        raise ValueError("ref_coords must be non-empty")
    if len(query_coords) == 0:
        return np.asarray([], dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(ref_coords)
    distances, _ = nn.kneighbors(query_coords, return_distance=True)
    return distances[:, 0].astype(np.float32, copy=False)


def prototype_distance_score(
    ref_coords: np.ndarray,
    query_coords: np.ndarray,
    ref_labels: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Open-set score based on distance to the nearest class prototype.

    Prototypes are computed as per-label centroids in the reference space.
    """
    if ref_coords.ndim != 2 or query_coords.ndim != 2:
        raise ValueError("ref_coords and query_coords must be 2D arrays")
    if ref_coords.shape[1] != query_coords.shape[1]:
        raise ValueError("ref_coords and query_coords must have the same feature dimension")
    if len(ref_coords) == 0:
        raise ValueError("ref_coords must be non-empty")
    if len(ref_coords) != len(ref_labels):
        raise ValueError("ref_labels must have same length as ref_coords")
    if len(query_coords) == 0:
        return np.asarray([], dtype=np.float32), 0

    labels = np.asarray(ref_labels).astype(object)
    unique = np.unique(labels)
    if unique.size == 0:
        raise ValueError("ref_labels must contain at least one label")

    centroids = np.zeros((unique.size, ref_coords.shape[1]), dtype=np.float32)
    for i, lab in enumerate(unique):
        mask = labels == lab
        centroids[i] = ref_coords[mask].mean(axis=0)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(centroids)
    distances, _ = nn.kneighbors(query_coords, return_distance=True)
    return distances[:, 0].astype(np.float32, copy=False), int(unique.size)


def openset_score(
    ref_coords: np.ndarray,
    query_coords: np.ndarray,
    *,
    method: str,
    ref_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[int]]:
    """Compute an open-set score (larger => more out-of-reference)."""
    if method == "nn_distance":
        return nn_distance_score(ref_coords, query_coords), None
    if method == "prototype":
        if ref_labels is None:
            raise ValueError("ref_labels is required for method='prototype'")
        scores, num_prototypes = prototype_distance_score(ref_coords, query_coords, ref_labels)
        return scores, num_prototypes
    raise ValueError("method must be one of: nn_distance, prototype")

