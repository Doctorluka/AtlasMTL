from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import torch

from ..core.model import AtlasMTLModel
from .manifest import (
    build_manifest_payload,
    default_manifest_path,
    default_metadata_path,
    resolve_manifest_paths,
    save_manifest,
)
from .reference_store import ReferenceData, load_reference_data, resolve_reference_path, save_reference_data


def save_trained_model_metadata(trained_model, path: str) -> None:
    trained_model.model.save(path)
    reference_storage = trained_model.reference_storage
    reference_path = trained_model.reference_path
    if reference_storage == "external":
        reference_path = resolve_reference_path(path, reference_path)
        save_reference_data(trained_model.reference_data, reference_path)

    reference_coords = trained_model.reference_data.coords if reference_storage == "full" else {}
    reference_labels = trained_model.reference_data.labels if reference_storage == "full" else {}
    meta_path = default_metadata_path(path)
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "label_columns": trained_model.label_columns,
                "label_encoders": trained_model.label_encoders,
                "train_genes": trained_model.train_genes,
                "coord_targets": trained_model.coord_targets,
                "coord_stats": trained_model.coord_stats,
                "reference_coords": reference_coords,
                "reference_labels": reference_labels,
                "reference_storage": reference_storage,
                "reference_path": reference_path,
                "latent_source": trained_model.latent_source,
                "input_transform": trained_model.input_transform,
                "train_config": trained_model.train_config,
            },
            f,
        )
    manifest_path = default_manifest_path(path)
    save_manifest(
        build_manifest_payload(
            model_path=path,
            metadata_path=meta_path,
            reference_storage=reference_storage,
            reference_path=reference_path,
            input_transform=trained_model.input_transform,
        ),
        manifest_path,
    )


def _resolve_artifact_paths(path: str) -> dict[str, Optional[str]]:
    if path.endswith(".json"):
        return resolve_manifest_paths(path)

    manifest_path = default_manifest_path(path)
    if Path(manifest_path).exists():
        resolved = resolve_manifest_paths(manifest_path)
        resolved.setdefault("model_path", str(Path(path).resolve()))
        return resolved

    return {
        "model_path": str(Path(path).resolve()),
        "metadata_path": str(Path(default_metadata_path(path)).resolve()),
        "reference_storage": "full",
        "reference_path": None,
        "input_transform": "binary",
    }


def load_trained_model_metadata(cls, path: str, device: Optional[torch.device] = None):
    artifact_paths = _resolve_artifact_paths(path)
    model_path = artifact_paths["model_path"]
    meta_path = artifact_paths["metadata_path"]
    if not model_path or not meta_path:
        raise ValueError("manifest must define both model_path and metadata_path")

    model = AtlasMTLModel.load(model_path, device=device)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    reference_storage = meta.get("reference_storage", artifact_paths.get("reference_storage", "full"))
    reference_path = artifact_paths.get("reference_path") or meta.get("reference_path")
    if reference_storage == "external":
        resolved_reference_path = resolve_reference_path(model_path, reference_path)
        reference_data = load_reference_data(resolved_reference_path)
        meta["reference_data"] = reference_data
        meta["reference_path"] = resolved_reference_path
    else:
        meta["reference_data"] = ReferenceData(
            coords=meta.get("reference_coords", {}),
            labels=meta.get("reference_labels", {}),
        )
        meta["reference_path"] = reference_path
    meta.pop("reference_coords", None)
    meta.pop("reference_labels", None)
    meta.setdefault("reference_storage", reference_storage)
    meta.setdefault("input_transform", artifact_paths.get("input_transform", "binary"))
    meta.setdefault("train_config", {})
    return cls(model=model, **meta)
