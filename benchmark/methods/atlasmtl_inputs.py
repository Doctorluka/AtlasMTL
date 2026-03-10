from __future__ import annotations

from typing import Any, Dict, List, Optional

from anndata import AnnData

from benchmark.methods.config import resolve_reference_query_layers

PHMAP_TASK_WEIGHTS = [0.3, 0.8, 1.5, 2.0]


def resolve_atlasmtl_layer_config(manifest: Dict[str, Any]) -> Dict[str, Optional[str]]:
    method_cfg = dict((manifest.get("method_configs") or {}).get("atlasmtl") or {})
    if not any(key in method_cfg for key in ("reference_layer", "query_layer", "counts_layer")) and "counts_layer" not in manifest:
        reference_layer, query_layer = None, None
    else:
        reference_layer, query_layer = resolve_reference_query_layers(manifest, method_cfg)
    return {
        "reference_layer": reference_layer,
        "query_layer": query_layer,
        "counts_layer": method_cfg.get("counts_layer") or manifest.get("counts_layer") or "counts",
    }


def adata_with_matrix_from_layer(adata: AnnData, *, layer_name: Optional[str]) -> AnnData:
    if not layer_name:
        return adata
    if layer_name not in adata.layers:
        raise ValueError(f"requested layer '{layer_name}' not found in AnnData.layers")
    out = adata.copy()
    out.X = adata.layers[layer_name].copy()
    return out


def matrix_source_label(layer_name: Optional[str]) -> str:
    return f"layers/{layer_name}" if layer_name else "X"


def resolve_task_weights(manifest: Dict[str, Any], label_columns: List[str]) -> List[float]:
    train_cfg = dict(manifest.get("train") or {})
    method_cfg = dict((manifest.get("method_configs") or {}).get("atlasmtl") or {})
    raw_weights = method_cfg.get("task_weights", train_cfg.get("task_weights"))
    if raw_weights is None:
        return [1.0 for _ in label_columns]
    weights = [float(value) for value in raw_weights]
    if len(weights) != len(label_columns):
        raise ValueError("task_weights length must match label_columns")
    return weights


def resolve_task_weight_candidates(
    manifest: Dict[str, Any],
    *,
    candidate_schedules: Dict[str, List[float]],
    label_columns: List[str],
) -> Dict[str, List[float]]:
    train_cfg = dict(manifest.get("train") or {})
    method_cfg = dict((manifest.get("method_configs") or {}).get("atlasmtl") or {})
    raw_candidates = method_cfg.get("task_weight_candidates", train_cfg.get("task_weight_candidates"))
    if raw_candidates is None:
        return {str(name): [float(x) for x in values] for name, values in candidate_schedules.items()}
    if not isinstance(raw_candidates, dict):
        raise ValueError("task_weight_candidates must be a mapping when provided")
    resolved: Dict[str, List[float]] = {}
    for name, values in raw_candidates.items():
        weights = [float(value) for value in values]
        if len(weights) != len(label_columns):
            raise ValueError("task_weight_candidates entries must match label_columns length")
        resolved[str(name)] = weights
    return resolved


def select_task_weight_candidate_from_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("candidate summary rows must be non-empty")

    def _sort_key(row: Dict[str, Any]) -> tuple:
        return (
            -float(row["full_path_accuracy"]),
            float(row["parent_correct_child_wrong_rate"]),
            -float(row["finest_macro_f1"]),
            0 if str(row["schedule_name"]).startswith("mild") else 1 if str(row["schedule_name"]).startswith("strong") else 2,
            str(row["schedule_name"]),
        )

    ranked = sorted(rows, key=_sort_key)
    return dict(ranked[0])


def task_weight_scheme_name(task_weights: List[float]) -> str:
    if all(abs(weight - 1.0) <= 1e-8 for weight in task_weights):
        return "uniform"
    if len(task_weights) == len(PHMAP_TASK_WEIGHTS) and all(
        abs(left - right) <= 1e-8 for left, right in zip(task_weights, PHMAP_TASK_WEIGHTS)
    ):
        return "phmap"
    return "custom"
