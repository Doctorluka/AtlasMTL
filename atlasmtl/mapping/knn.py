from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    from pynndescent import NNDescent
except Exception:  # pragma: no cover
    NNDescent = None


def knn_majority_vote(
    ref_coords: np.ndarray,
    ref_labels: np.ndarray,
    query_coords: np.ndarray,
    k: int,
    *,
    vote_mode: Literal["majority", "distance_weighted"] = "majority",
    index_mode: Literal["exact", "pynndescent"] = "exact",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_eff = min(int(k), len(ref_coords))
    if index_mode == "exact":
        nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        nn.fit(ref_coords)
        distances, idx = nn.kneighbors(query_coords, return_distance=True)
    elif index_mode == "pynndescent":
        if NNDescent is None:
            raise ValueError("index_mode='pynndescent' requires pynndescent to be installed")
        index = NNDescent(ref_coords, metric="euclidean", n_neighbors=k_eff, random_state=42)
        idx, distances = index.query(query_coords, k=k_eff)
    else:
        raise ValueError("index_mode must be one of: exact, pynndescent")

    labels = []
    vote_frac = []
    vote_margin = []
    eps = 1e-6
    for row_i, row in enumerate(idx):
        neigh_labels = ref_labels[row]
        if vote_mode == "majority":
            values, counts = np.unique(neigh_labels, return_counts=True)
            order = np.argsort(counts)[::-1]
            sorted_values = values[order]
            sorted_counts = counts[order]
            top_count = sorted_counts[0]
            second_count = sorted_counts[1] if len(sorted_counts) > 1 else 0
            labels.append(sorted_values[0])
            vote_frac.append(top_count / len(row))
            vote_margin.append((top_count - second_count) / len(row))
        elif vote_mode == "distance_weighted":
            weights = 1.0 / (distances[row_i] + eps)
            totals = {}
            for lab, w in zip(neigh_labels, weights):
                totals[lab] = totals.get(lab, 0.0) + float(w)
            items = sorted(totals.items(), key=lambda x: x[1], reverse=True)
            top_label, top_w = items[0]
            second_w = items[1][1] if len(items) > 1 else 0.0
            denom = float(sum(totals.values())) if totals else 1.0
            labels.append(top_label)
            vote_frac.append(float(top_w) / denom)
            vote_margin.append(float(top_w - second_w) / denom)
        else:
            raise ValueError("vote_mode must be one of: majority, distance_weighted")

    return (
        np.asarray(labels),
        np.asarray(vote_frac, dtype=np.float32),
        np.asarray(vote_margin, dtype=np.float32),
    )


def build_prototypes(ref_coords: np.ndarray, ref_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-label centroids as a compressed reference representation."""
    labels = np.asarray(ref_labels).astype(object)
    unique = np.unique(labels)
    if unique.size == 0:
        raise ValueError("ref_labels must contain at least one label")
    centroids = np.zeros((unique.size, ref_coords.shape[1]), dtype=np.float32)
    for i, lab in enumerate(unique):
        mask = labels == lab
        centroids[i] = ref_coords[mask].mean(axis=0)
    return centroids, unique.astype(object)
