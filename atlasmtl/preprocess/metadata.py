from __future__ import annotations

from typing import Optional

from anndata import AnnData

from .types import FeaturePanel, PreprocessConfig, PreprocessReport


def build_preprocess_metadata(
    *,
    config: PreprocessConfig,
    report: PreprocessReport,
    feature_panel: Optional[FeaturePanel] = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "config": config.to_dict(),
        "report": report.to_dict(),
    }
    if feature_panel is not None:
        payload["feature_panel"] = feature_panel.to_dict()
    return payload


def attach_preprocess_metadata(
    adata: AnnData,
    *,
    config: PreprocessConfig,
    report: PreprocessReport,
    feature_panel: Optional[FeaturePanel] = None,
) -> AnnData:
    adata.uns["atlasmtl_preprocess"] = build_preprocess_metadata(
        config=config,
        report=report,
        feature_panel=feature_panel,
    )
    return adata
