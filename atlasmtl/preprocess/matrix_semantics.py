from __future__ import annotations

from typing import Literal

import numpy as np
from anndata import AnnData
from scipy import sparse


def is_count_like_matrix(
    data,
    *,
    n_obs: int = 100,
    n_vals: int = 20000,
    integer_tol: float = 1e-6,
) -> bool:
    if sparse.issparse(data):
        values = np.asarray(data.data)
    else:
        values = np.asarray(data)

    if values.size == 0:
        return True

    if values.ndim > 1:
        if values.shape[0] > n_obs:
            obs_idx = np.random.default_rng(0).choice(values.shape[0], size=n_obs, replace=False)
            values = values[obs_idx]
        values = values.reshape(-1)

    if values.size > n_vals:
        value_idx = np.random.default_rng(0).choice(values.size, size=n_vals, replace=False)
        values = values[value_idx]

    if np.any(values < 0):
        return False
    return bool(np.all(np.abs(values - np.rint(values)) <= integer_tol))


def detect_input_matrix_type(
    adata: AnnData,
    *,
    declared_type: str = "infer",
    n_obs: int = 100,
    n_vals: int = 20000,
    integer_tol: float = 1e-6,
) -> Literal["counts", "lognorm", "unknown"]:
    if declared_type != "infer":
        return str(declared_type)  # type: ignore[return-value]
    try:
        if is_count_like_matrix(adata.X, n_obs=n_obs, n_vals=n_vals, integer_tol=integer_tol):
            return "counts"
        return "lognorm"
    except Exception:
        return "unknown"
