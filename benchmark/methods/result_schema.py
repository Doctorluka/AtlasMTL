from __future__ import annotations

from typing import Any, Dict, Optional


def build_input_contract(
    *,
    reference_matrix_source: str,
    query_matrix_source: str,
    feature_alignment: str,
    normalization_mode: str,
    label_scope: str,
    counts_layer: Optional[str] = None,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "reference_matrix_source": reference_matrix_source,
        "query_matrix_source": query_matrix_source,
        "counts_layer": counts_layer,
        "feature_alignment": feature_alignment,
        "normalization_mode": normalization_mode,
        "label_scope": label_scope,
        "backend": backend,
    }
