from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from atlasmtl import PreprocessConfig, build_model, predict, preprocess_query, preprocess_reference


def _mapping_table(tmp_path):
    path = tmp_path / "mapping.tsv"
    pd.DataFrame(
        {
            "human_ensembl_gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            "human_gene_symbol": ["GATA1", "CD3D", "MS4A1", "LYZ"],
            "human_gene_description": ["", "", "", ""],
            "mouse_ensembl_gene_id": ["ENSMUSG1", "ENSMUSG2", "ENSMUSG3", "ENSMUSG4"],
            "mouse_gene_symbol": ["Gata1", "Cd3d", "Ms4a1", "Lyz"],
            "rat_ensembl_gene_id": ["ENSRNOG1", "ENSRNOG2", "ENSRNOG3", "ENSRNOG4"],
            "rat_gene_symbol": ["Gata1", "Cd3d", "Ms4a1", "Lyz"],
        }
    ).to_csv(path, sep="\t", index=False)
    return path


def test_preprocess_reference_and_query_roundtrip(tmp_path):
    mapping = str(_mapping_table(tmp_path))
    ref = AnnData(
        X=np.array(
            [
                [3.0, 0.0, 0.0, 1.0],
                [2.0, 1.0, 0.0, 0.0],
                [0.0, 3.0, 1.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"anno_lv1": ["A", "A", "B", "B"]}, index=["r1", "r2", "r3", "r4"]),
    )
    ref.var_names = ["GATA1", "CD3D", "MS4A1", "LYZ"]
    ref.obsm["X_ref_latent"] = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]], dtype=np.float32)
    ref.obsm["X_umap"] = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]], dtype=np.float32)

    cfg = PreprocessConfig(
        var_names_type="symbol",
        species="human",
        gene_id_table=mapping,
        feature_space="whole",
    )
    ref_pp, feature_panel, ref_report = preprocess_reference(ref, cfg)
    assert ref_report.n_features_selected == ref_pp.n_vars
    assert "atlasmtl_preprocess" in ref_pp.uns

    model = build_model(
        adata=ref_pp,
        label_columns=["anno_lv1"],
        coord_targets={"latent": "X_ref_latent", "umap": "X_umap"},
        num_epochs=1,
        batch_size=2,
        hidden_sizes=[8],
        device="cpu",
    )
    assert "preprocess" in model.train_config

    query = AnnData(X=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), obs=pd.DataFrame(index=["q1", "q2"]))
    query.var_names = ["GATA1", "CD3D"]
    query_pp, query_report = preprocess_query(query, feature_panel, cfg)
    assert query_report.missing_feature_genes == 2

    result = predict(model, query_pp, knn_correction="off", batch_size=1, device="cpu")
    assert "preprocess" in result.metadata
