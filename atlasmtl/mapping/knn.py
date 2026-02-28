from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_majority_vote(
    ref_coords: np.ndarray,
    ref_labels: np.ndarray,
    query_coords: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=min(k, len(ref_coords)), metric="euclidean")
    nn.fit(ref_coords)
    idx = nn.kneighbors(query_coords, return_distance=False)

    labels = []
    vote_frac = []
    vote_margin = []
    for row in idx:
        values, counts = np.unique(ref_labels[row], return_counts=True)
        order = np.argsort(counts)[::-1]
        sorted_values = values[order]
        sorted_counts = counts[order]
        top_count = sorted_counts[0]
        second_count = sorted_counts[1] if len(sorted_counts) > 1 else 0
        labels.append(sorted_values[0])
        vote_frac.append(top_count / len(row))
        vote_margin.append((top_count - second_count) / len(row))

    return (
        np.asarray(labels),
        np.asarray(vote_frac, dtype=np.float32),
        np.asarray(vote_margin, dtype=np.float32),
    )
