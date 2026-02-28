from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from anndata import AnnData


def apply_input_transform(X: np.ndarray, input_transform: str) -> np.ndarray:
    if input_transform == "binary":
        return (X > 0).astype(np.float32)
    if input_transform == "float":
        return X.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported input_transform: {input_transform}")


def extract_matrix(
    adata: AnnData,
    train_genes: Optional[List[str]] = None,
    input_transform: str = "binary",
) -> np.ndarray:
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if train_genes is None:
        return apply_input_transform(X, input_transform)

    gene_index = {g: i for i, g in enumerate(adata.var_names)}
    out = np.zeros((adata.n_obs, len(train_genes)), dtype=np.float32)
    for j, gene in enumerate(train_genes):
        idx = gene_index.get(gene)
        if idx is not None:
            out[:, j] = X[:, idx]
    return apply_input_transform(out, input_transform)


def compute_coord_stats(arr: np.ndarray) -> Dict[str, np.ndarray]:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std == 0] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def scale_coords(arr: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return ((arr - stats["mean"]) / stats["std"]).astype(np.float32)


def unscale_coords(arr: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (arr * stats["std"] + stats["mean"]).astype(np.float32)
