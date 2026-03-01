from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from atlasmtl.preprocess import FeaturePanel, PreprocessConfig
from atlasmtl.preprocess.features import align_query_to_feature_panel, select_reference_features


def test_select_reference_features_whole_keeps_all_genes():
    adata = AnnData(X=np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32), obs=pd.DataFrame(index=["a", "b"]))
    adata.var_names = ["ENSG1", "ENSG2"]
    adata.var["gene_symbol"] = ["GATA1", "CD3D"]
    cfg = PreprocessConfig(var_names_type="ensembl", species="human", feature_space="whole")

    selected, panel, report = select_reference_features(adata, cfg)

    assert selected.n_vars == 2
    assert panel.feature_space == "whole"
    assert panel.n_features == 2
    assert report.n_features_selected == 2


def test_align_query_to_feature_panel_zero_fills_missing():
    adata = AnnData(X=np.array([[5.0, 7.0]], dtype=np.float32), obs=pd.DataFrame(index=["q1"]))
    adata.var_names = ["ENSG2", "ENSG3"]
    panel = FeaturePanel(
        gene_ids=["ENSG1", "ENSG2"],
        gene_symbols=["GATA1", "CD3D"],
        feature_space="hvg",
        n_features=2,
        species="human",
    )
    cfg = PreprocessConfig(var_names_type="ensembl", species="human")

    aligned, report = align_query_to_feature_panel(adata, panel, cfg)

    assert list(aligned.var_names) == ["ENSG1", "ENSG2"]
    assert np.allclose(np.asarray(aligned.X), np.array([[0.0, 5.0]], dtype=np.float32))
    assert report.matched_feature_genes == 1
    assert report.missing_feature_genes == 1
