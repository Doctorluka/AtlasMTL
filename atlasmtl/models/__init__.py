"""Model registry and bundled model hooks."""

from .manifest import (
    build_manifest_payload,
    default_feature_panel_path,
    default_manifest_path,
    default_metadata_path,
    load_manifest,
    resolve_manifest_paths,
)
from .reference_store import ReferenceData
from .serialization import load_trained_model_metadata, save_trained_model_metadata
from .checksums import artifact_checksums, sha256_file

__all__ = [
    "ReferenceData",
    "build_manifest_payload",
    "default_feature_panel_path",
    "default_manifest_path",
    "default_metadata_path",
    "load_manifest",
    "resolve_manifest_paths",
    "save_trained_model_metadata",
    "load_trained_model_metadata",
    "sha256_file",
    "artifact_checksums",
]
