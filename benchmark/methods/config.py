from __future__ import annotations

from typing import Any, Dict, Tuple


def resolve_counts_layer(manifest: Dict[str, Any], method_cfg: Dict[str, Any]) -> str:
    return str(method_cfg.get("counts_layer") or manifest.get("counts_layer") or "counts")


def resolve_reference_query_layers(
    manifest: Dict[str, Any],
    method_cfg: Dict[str, Any],
) -> Tuple[str, str]:
    counts_layer = resolve_counts_layer(manifest, method_cfg)
    reference_layer = str(method_cfg.get("reference_layer") or counts_layer)
    query_layer = str(method_cfg.get("query_layer") or counts_layer)
    return reference_layer, query_layer
