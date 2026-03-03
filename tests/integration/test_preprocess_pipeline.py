from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from atlasmtl import PreprocessConfig, build_model, ensure_counts_layer, predict, preprocess_query, preprocess_reference
from atlasmtl.models import default_feature_panel_path, default_manifest_path, load_manifest


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
    assert "counts" in ref_pp.layers
    assert ref_pp.uns["atlasmtl_preprocess"]["report"]["counts_layer_used"] == "counts"
    assert ref_pp.uns["atlasmtl_preprocess"]["report"]["counts_decision"] == "counts_confirmed"
    assert ref_pp.uns["atlasmtl_preprocess"]["feature_panel"]["counts_layer"] == "counts"
    assert ref_pp.uns["atlasmtl_preprocess"]["feature_panel"]["hvg_layer_used"] is None

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
    model_path = tmp_path / "model_preprocessed.pth"
    model.save(str(model_path))
    manifest = load_manifest(str(default_manifest_path(str(model_path))))
    assert manifest["feature_panel_path"] == "model_preprocessed_feature_panel.json"
    assert (tmp_path / "model_preprocessed_feature_panel.json").exists()
    assert default_feature_panel_path(str(model_path)).endswith("model_preprocessed_feature_panel.json")

    query = AnnData(X=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), obs=pd.DataFrame(index=["q1", "q2"]))
    query.var_names = ["GATA1", "CD3D"]
    query_pp, query_report = preprocess_query(query, feature_panel, cfg)
    assert query_report.missing_feature_genes == 2
    assert "counts" in query_pp.layers
    assert query_report.counts_decision == "counts_confirmed"

    result = predict(model, query_pp, knn_correction="off", batch_size=1, device="cpu")
    assert "preprocess" in result.metadata


def test_preprocess_reference_hvg_uses_counts_layer(tmp_path, monkeypatch):
    mapping = str(_mapping_table(tmp_path))
    ref = AnnData(
        X=np.array(
            [
                [10.0, 0.0, 1.0, 0.0],
                [11.0, 0.0, 1.0, 0.0],
                [0.0, 8.0, 0.0, 1.0],
                [0.0, 9.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"sample": ["s1", "s1", "s2", "s2"]}, index=["r1", "r2", "r3", "r4"]),
    )
    ref.var_names = ["GATA1", "CD3D", "MS4A1", "LYZ"]
    calls = {}

    def _fake_hvg(adata, **kwargs):
        calls.update(kwargs)
        return pd.DataFrame({"highly_variable": [True, True, False, False]}, index=adata.var_names)

    monkeypatch.setattr("atlasmtl.preprocess.features.sc.pp.highly_variable_genes", _fake_hvg)

    cfg = PreprocessConfig(
        var_names_type="symbol",
        species="human",
        gene_id_table=mapping,
        feature_space="hvg",
        n_top_genes=2,
        hvg_batch_key="sample",
    )
    ref_pp, feature_panel, ref_report = preprocess_reference(ref, cfg)

    assert ref_pp.n_vars == 2
    assert calls["layer"] == "counts"
    assert calls["flavor"] == "seurat_v3"
    assert feature_panel.hvg_layer_used == "counts"
    assert feature_panel.counts_layer == "counts"
    assert ref_report.hvg_layer_used == "counts"
    assert ref_pp.uns["atlasmtl_preprocess"]["report"]["hvg_layer_used"] == "counts"
    assert ref_report.counts_detection_summary is not None


def test_ensure_counts_layer_promotes_count_like_x(tmp_path):
    ref = AnnData(
        X=np.array(
            [
                [3.0, 0.0, 1.0],
                [0.0, 2.0, 0.0],
                [4.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(index=["r1", "r2", "r3"]),
    )
    ref.var_names = ["GATA1", "CD3D", "LYZ"]
    cfg = PreprocessConfig(
        var_names_type="symbol",
        species="human",
        gene_id_table=str(_mapping_table(tmp_path)),
        feature_space="whole",
    )

    with_counts, counts_meta = ensure_counts_layer(ref, cfg)

    assert "counts" in with_counts.layers
    assert counts_meta["counts_source_original"] == "X"
    assert counts_meta["counts_layer_materialized"] is True
    assert counts_meta["counts_decision"] == "counts_confirmed"
