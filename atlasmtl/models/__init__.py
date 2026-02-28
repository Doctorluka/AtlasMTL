"""Model registry and bundled model hooks."""

from .manifest import default_manifest_path, default_metadata_path, load_manifest, resolve_manifest_paths
from .reference_store import ReferenceData
from .serialization import load_trained_model_metadata, save_trained_model_metadata

__all__ = [
    "ReferenceData",
    "default_manifest_path",
    "default_metadata_path",
    "load_manifest",
    "resolve_manifest_paths",
    "save_trained_model_metadata",
    "load_trained_model_metadata",
]
