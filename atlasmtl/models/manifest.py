from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def default_manifest_path(model_path: str) -> str:
    return model_path.replace(".pth", "_manifest.json")


def default_metadata_path(model_path: str) -> str:
    return model_path.replace(".pth", "_metadata.pkl")


def default_feature_panel_path(model_path: str) -> str:
    return model_path.replace(".pth", "_feature_panel.json")


def _relative_to_parent(path: Optional[str], parent: Path) -> Optional[str]:
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(parent.resolve()))
    except ValueError:
        return str(Path(path))


def build_manifest_payload(
    model_path: str,
    metadata_path: str,
    reference_storage: str,
    reference_path: Optional[str],
    input_transform: str,
    feature_panel_path: Optional[str] = None,
    preprocess: Optional[Dict[str, object]] = None,
    checksums: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    parent = Path(model_path).resolve().parent
    payload: Dict[str, object] = {
        "schema_version": 1,
        "model_path": _relative_to_parent(model_path, parent),
        "metadata_path": _relative_to_parent(metadata_path, parent),
        "reference_storage": reference_storage,
        "reference_path": _relative_to_parent(reference_path, parent),
        "input_transform": input_transform,
    }
    if feature_panel_path:
        payload["feature_panel_path"] = _relative_to_parent(feature_panel_path, parent)
    if preprocess:
        payload["preprocess"] = dict(preprocess)
    if checksums:
        payload["checksums"] = dict(checksums)
    return payload


def save_manifest(payload: Dict[str, object], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def load_manifest(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_manifest_paths(manifest_path: str) -> Dict[str, Optional[str]]:
    manifest = load_manifest(manifest_path)
    base_dir = Path(manifest_path).resolve().parent

    def _resolve(key: str) -> Optional[str]:
        value = manifest.get(key)
        if not value:
            return None
        return str((base_dir / str(value)).resolve())

    return {
        "model_path": _resolve("model_path"),
        "metadata_path": _resolve("metadata_path"),
        "reference_storage": manifest.get("reference_storage", "full"),
        "reference_path": _resolve("reference_path"),
        "feature_panel_path": _resolve("feature_panel_path"),
        "input_transform": manifest.get("input_transform", "binary"),
        "preprocess": manifest.get("preprocess"),
    }
