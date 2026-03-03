from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from anndata import AnnData
from scipy import sparse


@dataclass
class MatrixSemanticsSummary:
    shape: tuple[int, ...]
    sampled_values: int
    nonzero_rate: float
    integer_like_fraction: float
    tiny_pos_rate_lt_1e8: float
    tiny_pos_rate_lt_1e6: float
    min_nonzero: float
    p1_nonzero: float
    p5_nonzero: float
    p50_nonzero: float
    p95_nonzero: float
    max_nonzero: float
    has_negative: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


def _sample_values(
    data,
    *,
    n_obs: int = 100,
    n_vals: int = 20000,
) -> np.ndarray:
    if sparse.issparse(data):
        values = np.asarray(data.data)
        if values.size > n_vals:
            idx = np.random.default_rng(0).choice(values.size, size=n_vals, replace=False)
            values = values[idx]
        return values

    values = np.asarray(data)
    if values.size == 0:
        return values.reshape(-1)

    if values.ndim > 1 and values.shape[0] > n_obs:
        obs_idx = np.random.default_rng(0).choice(values.shape[0], size=n_obs, replace=False)
        values = values[obs_idx]
    values = values.reshape(-1)
    if values.size > n_vals:
        value_idx = np.random.default_rng(0).choice(values.size, size=n_vals, replace=False)
        values = values[value_idx]
    return values


def summarize_matrix_semantics(
    data,
    *,
    n_obs: int = 100,
    n_vals: int = 20000,
    integer_tol: float = 1e-6,
    tiny_positive_tol: float = 1e-8,
) -> MatrixSemanticsSummary:
    values = _sample_values(data, n_obs=n_obs, n_vals=n_vals)
    if sparse.issparse(data):
        shape = tuple(int(dim) for dim in data.shape)
    else:
        shape = tuple(int(dim) for dim in np.asarray(data).shape)

    if values.size == 0:
        return MatrixSemanticsSummary(
            shape=shape,
            sampled_values=0,
            nonzero_rate=0.0,
            integer_like_fraction=1.0,
            tiny_pos_rate_lt_1e8=0.0,
            tiny_pos_rate_lt_1e6=0.0,
            min_nonzero=0.0,
            p1_nonzero=0.0,
            p5_nonzero=0.0,
            p50_nonzero=0.0,
            p95_nonzero=0.0,
            max_nonzero=0.0,
            has_negative=False,
        )

    has_negative = bool(np.any(values < 0))
    nonzero = values[values > 0]
    nonzero_rate = float(nonzero.size / values.size)
    if nonzero.size == 0:
        integer_like_fraction = 1.0
        tiny_pos_rate_lt_1e8 = 0.0
        tiny_pos_rate_lt_1e6 = 0.0
        min_nonzero = 0.0
        p1_nonzero = 0.0
        p5_nonzero = 0.0
        p50_nonzero = 0.0
        p95_nonzero = 0.0
        max_nonzero = 0.0
    else:
        integer_like_fraction = float(np.mean(np.abs(nonzero - np.rint(nonzero)) <= integer_tol))
        tiny_pos_rate_lt_1e8 = float(np.mean((nonzero > 0) & (nonzero < tiny_positive_tol)))
        tiny_pos_rate_lt_1e6 = float(np.mean((nonzero > 0) & (nonzero < 1e-6)))
        min_nonzero = float(np.min(nonzero))
        p1_nonzero = float(np.percentile(nonzero, 1))
        p5_nonzero = float(np.percentile(nonzero, 5))
        p50_nonzero = float(np.percentile(nonzero, 50))
        p95_nonzero = float(np.percentile(nonzero, 95))
        max_nonzero = float(np.max(nonzero))

    return MatrixSemanticsSummary(
        shape=shape,
        sampled_values=int(values.size),
        nonzero_rate=nonzero_rate,
        integer_like_fraction=integer_like_fraction,
        tiny_pos_rate_lt_1e8=tiny_pos_rate_lt_1e8,
        tiny_pos_rate_lt_1e6=tiny_pos_rate_lt_1e6,
        min_nonzero=min_nonzero,
        p1_nonzero=p1_nonzero,
        p5_nonzero=p5_nonzero,
        p50_nonzero=p50_nonzero,
        p95_nonzero=p95_nonzero,
        max_nonzero=max_nonzero,
        has_negative=has_negative,
    )


def classify_count_semantics(
    summary: MatrixSemanticsSummary,
    *,
    integer_fraction_threshold: float = 0.999,
) -> Literal["counts_confirmed", "counts_suspected", "not_counts"]:
    if summary.has_negative:
        return "not_counts"
    if summary.sampled_values == 0:
        return "counts_confirmed"
    if summary.integer_like_fraction >= integer_fraction_threshold and summary.tiny_pos_rate_lt_1e6 == 0.0:
        return "counts_confirmed"
    if summary.integer_like_fraction > 0.5:
        return "counts_suspected"
    return "not_counts"


def is_count_like_matrix(
    data,
    *,
    n_obs: int = 100,
    n_vals: int = 20000,
    integer_tol: float = 1e-6,
    tiny_positive_tol: float = 1e-8,
    counts_confirm_fraction: float = 0.999,
) -> bool:
    summary = summarize_matrix_semantics(
        data,
        n_obs=n_obs,
        n_vals=n_vals,
        integer_tol=integer_tol,
        tiny_positive_tol=tiny_positive_tol,
    )
    return classify_count_semantics(
        summary,
        integer_fraction_threshold=counts_confirm_fraction,
    ) == "counts_confirmed"


def detect_input_matrix_type(
    adata: AnnData,
    *,
    declared_type: str = "infer",
    n_obs: int = 100,
    n_vals: int = 20000,
    integer_tol: float = 1e-6,
    tiny_positive_tol: float = 1e-8,
    counts_confirm_fraction: float = 0.999,
) -> Literal["counts", "lognorm", "unknown"]:
    if declared_type != "infer":
        return str(declared_type)  # type: ignore[return-value]
    try:
        summary = summarize_matrix_semantics(
            adata.X,
            n_obs=n_obs,
            n_vals=n_vals,
            integer_tol=integer_tol,
            tiny_positive_tol=tiny_positive_tol,
        )
        decision = classify_count_semantics(summary, integer_fraction_threshold=counts_confirm_fraction)
        if decision == "counts_confirmed":
            return "counts"
        if decision in {"counts_suspected", "not_counts"} and not summary.has_negative:
            return "lognorm"
        return "unknown"
    except Exception:
        return "unknown"
