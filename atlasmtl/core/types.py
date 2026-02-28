from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder

from ..io import select_prediction_frame, write_prediction_result
from ..models import ReferenceData, load_trained_model_metadata, save_trained_model_metadata
from .model import AtlasMTLModel


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _format_resource_usage(title: str, sections: List[tuple[str, Dict[str, object]]]) -> str:
    lines = [title]
    for section_title, payload in sections:
        items = [(key, _format_value(value)) for key, value in payload.items() if value is not None]
        if not items:
            continue
        width = max(len(key) for key, _ in items)
        lines.append("")
        lines.append(f"{section_title}:")
        for key, value in items:
            lines.append(f"  {key:<{width}} : {value}")
    return "\n".join(lines)


@dataclass
class TrainedModel:
    model: AtlasMTLModel
    label_columns: List[str]
    label_encoders: Dict[str, LabelEncoder]
    train_genes: List[str]
    coord_targets: Dict[str, str]
    coord_stats: Dict[str, Dict[str, np.ndarray]]
    reference_data: ReferenceData
    latent_source: str = "internal_preferred"
    input_transform: str = "binary"
    reference_storage: str = "external"
    reference_path: Optional[str] = None
    train_config: Dict[str, object] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save model weights, metadata, reference assets, and manifest.

        Parameters
        ----------
        path
            Target path for the main `.pth` model artifact. Saving also writes
            `*_metadata.pkl`, `*_manifest.json`, and, when
            `reference_storage="external"`, `*_reference.pkl`.
        """
        save_trained_model_metadata(self, path)

    def get_resource_usage(self) -> Dict[str, object]:
        """Return recorded training resource usage metadata."""
        usage: Dict[str, object] = {}
        usage.update(self.train_config.get("runtime_summary", {}))
        usage.update(
            {
                "device_used": self.train_config.get("device_used"),
                "device_requested": self.train_config.get("device_requested"),
                "num_threads_used": self.train_config.get("num_threads_used"),
                "coord_enabled": self.train_config.get("coord_enabled"),
            }
        )
        return usage

    def show_resource_usage(self) -> None:
        """Print recorded training resource usage to the terminal."""
        usage = self.get_resource_usage()
        summary = {
            "elapsed_seconds": usage.get("elapsed_seconds"),
            "items_per_second": usage.get("items_per_second"),
            "process_peak_rss_gb": usage.get("process_peak_rss_gb"),
            "gpu_peak_memory_gb": usage.get("gpu_peak_memory_gb"),
        }
        execution = {
            "device_used": usage.get("device_used"),
            "device_requested": usage.get("device_requested"),
            "num_threads_used": usage.get("num_threads_used"),
            "coord_enabled": usage.get("coord_enabled"),
        }
        print(
            _format_resource_usage(
                "atlasmtl training resource usage",
                [
                    ("Summary", summary),
                    ("Execution", execution),
                ],
            )
        )

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "TrainedModel":
        """Load a trained model from a `.pth` artifact or manifest file.

        Parameters
        ----------
        path
            Path to `model.pth` or `model_manifest.json`.
        device
            Optional torch device override for model weights.
        """
        return load_trained_model_metadata(cls, path, device=device)

    @property
    def reference_coords(self) -> Dict[str, np.ndarray]:
        return self.reference_data.coords

    @property
    def reference_labels(self) -> Dict[str, np.ndarray]:
        return self.reference_data.labels


class PredictionResult:
    """Container for full prediction outputs before optional export."""

    def __init__(self, predictions: pd.DataFrame, coordinates: Dict[str, np.ndarray], metadata: Dict):
        self.predictions = predictions
        self.coordinates = coordinates
        self.metadata = metadata

    def to_adata(
        self,
        adata: AnnData,
        mode: str = "standard",
        include_coords: bool = False,
        include_metadata: bool = True,
    ) -> AnnData:
        """Write selected prediction outputs back into an `AnnData` object.

        Parameters
        ----------
        adata
            Target `AnnData` object.
        mode
            Export level: `"minimal"`, `"standard"`, or `"full"`.
        include_coords
            Whether to write predicted coordinates into `adata.obsm`.
        include_metadata
            Whether to update `adata.uns["atlasmtl"]`.
        """
        return write_prediction_result(
            adata,
            self.predictions,
            self.coordinates,
            self.metadata,
            mode=mode,
            include_coords=include_coords,
            include_metadata=include_metadata,
        )

    def to_dataframe(self, mode: str = "standard") -> pd.DataFrame:
        """Return a mode-filtered prediction table indexed by cell name."""
        return select_prediction_frame(self.predictions, mode=mode)

    def to_csv(self, path: str | Path, mode: str = "standard", index: bool = True, **kwargs: object) -> None:
        """Write a mode-filtered prediction table to CSV.

        The exported table keeps `obs_names` as the row index by default, so it
        can be joined back into `adata.obs` later.
        """
        self.to_dataframe(mode=mode).to_csv(path, index=index, **kwargs)

    def get_resource_usage(self) -> Dict[str, object]:
        """Return recorded prediction resource usage metadata."""
        usage: Dict[str, object] = {}
        usage.update(self.metadata.get("prediction_runtime", {}))
        usage.update(
            {
                "device_used": self.metadata.get("device_used"),
                "device_requested": self.metadata.get("device_requested"),
                "num_threads_used": self.metadata.get("num_threads_used"),
            }
        )
        return usage

    def show_resource_usage(self) -> None:
        """Print recorded prediction resource usage to the terminal."""
        usage = self.get_resource_usage()
        summary = {
            "elapsed_seconds": usage.get("elapsed_seconds"),
            "items_per_second": usage.get("items_per_second"),
            "process_peak_rss_gb": usage.get("process_peak_rss_gb"),
            "gpu_peak_memory_gb": usage.get("gpu_peak_memory_gb"),
        }
        execution = {
            "device_used": usage.get("device_used"),
            "device_requested": usage.get("device_requested"),
            "num_threads_used": usage.get("num_threads_used"),
        }
        prediction = {
            "knn_correction": self.metadata.get("knn_correction"),
            "knn_space_used": self.metadata.get("knn_space_used"),
            "input_transform": self.metadata.get("input_transform"),
        }
        print(
            _format_resource_usage(
                "atlasmtl prediction resource usage",
                [
                    ("Summary", summary),
                    ("Execution", execution),
                    ("Prediction", prediction),
                ],
            )
        )
