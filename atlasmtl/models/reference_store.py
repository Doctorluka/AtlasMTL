from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class ReferenceData:
    coords: Dict[str, np.ndarray]
    labels: Dict[str, np.ndarray]


def default_reference_path(model_path: str) -> str:
    return model_path.replace(".pth", "_reference.pkl")


def save_reference_data(reference_data: ReferenceData, path: str) -> None:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wb") as f:
        pickle.dump(
            {
                "coords": reference_data.coords,
                "labels": reference_data.labels,
            },
            f,
        )


def load_reference_data(path: str) -> ReferenceData:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        payload = pickle.load(f)
    return ReferenceData(coords=payload["coords"], labels=payload["labels"])


def resolve_reference_path(model_path: str, reference_path: Optional[str] = None) -> str:
    return reference_path or default_reference_path(model_path)


def reference_path_exists(path: Optional[str]) -> bool:
    return bool(path) and Path(path).exists()
