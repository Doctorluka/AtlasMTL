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
    if train_genes is None:
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        return apply_input_transform(X, input_transform)

    var_names = adata.var_names.astype(str)
    idx = var_names.get_indexer([str(g) for g in train_genes])
    present_mask = idx >= 0
    out = np.zeros((adata.n_obs, len(train_genes)), dtype=np.float32)
    if present_mask.any():
        src_cols = idx[present_mask]
        if isinstance(X, np.ndarray):
            tmp = X[:, src_cols]
            if tmp.dtype != np.float32:
                tmp = tmp.astype(np.float32)
            out[:, present_mask] = tmp
        else:
            # Avoid densifying the full matrix; densify only the selected columns.
            out[:, present_mask] = np.asarray(X[:, src_cols].toarray(), dtype=np.float32)
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
