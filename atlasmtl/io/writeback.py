from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from anndata import AnnData

CORE_SUFFIXES = ("_raw", "_knn")
STANDARD_PREFIXES = ("conf_", "margin_", "is_unknown_")
DEBUG_PREFIXES = ("is_low_conf_", "is_low_knn_conf_", "used_knn_", "knn_vote_frac_")


def get_prediction_columns(predictions: pd.DataFrame, mode: str) -> List[str]:
    if mode not in {"minimal", "standard", "full"}:
        raise ValueError("mode must be one of: minimal, standard, full")

    selected: List[str] = []
    for column in predictions.columns:
        if column.startswith("pred_") and not column.endswith(CORE_SUFFIXES):
            selected.append(column)
            continue
        if mode in {"standard", "full"} and column.startswith(STANDARD_PREFIXES):
            selected.append(column)
            continue
        if mode == "full" and (
            column.startswith(DEBUG_PREFIXES) or column.endswith(CORE_SUFFIXES)
        ):
            selected.append(column)
    return selected


def select_prediction_frame(predictions: pd.DataFrame, mode: str) -> pd.DataFrame:
    selected_columns = get_prediction_columns(predictions, mode)
    if not selected_columns:
        return predictions.iloc[:, 0:0].copy()
    return predictions.loc[:, selected_columns].copy()


def write_prediction_result(
    adata: AnnData,
    predictions: pd.DataFrame,
    coordinates: Dict[str, np.ndarray],
    metadata: Dict[str, object],
    mode: str = "standard",
    include_coords: bool = False,
    include_metadata: bool = True,
) -> AnnData:
    selected_columns = get_prediction_columns(predictions, mode)
    existing_prediction_columns = [column for column in predictions.columns if column in adata.obs.columns]
    if existing_prediction_columns:
        adata.obs = adata.obs.drop(columns=existing_prediction_columns)
    if selected_columns:
        adata.obs = adata.obs.join(predictions[selected_columns])

    if include_coords:
        for key, value in coordinates.items():
            adata.obsm[key] = value

    if include_metadata:
        adata.uns.setdefault("atlasmtl", {}).update(metadata)
    return adata
