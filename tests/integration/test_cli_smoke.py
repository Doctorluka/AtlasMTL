from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_atlasmtl.py"
PREDICT_SCRIPT = REPO_ROOT / "scripts" / "predict_atlasmtl.py"


def _write_test_h5ads(tmp_path: Path) -> tuple[Path, Path]:
    ref_obs = pd.DataFrame({"anno_lv1": ["A", "A", "B", "B"]}, index=["r1", "r2", "r3", "r4"])
    ref = AnnData(X=np.array([[1, 0], [1, 1], [0, 1], [0, 2]], dtype=np.float32), obs=ref_obs)
    ref.var_names = ["g1", "g2"]
    ref.obsm["X_ref_latent"] = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.1, 1.0]], dtype=np.float32)
    ref.obsm["X_umap"] = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.1]], dtype=np.float32)

    query = AnnData(X=np.array([[1, 0], [0, 1]], dtype=np.float32), obs=pd.DataFrame(index=["q1", "q2"]))
    query.var_names = ["g1", "g2"]

    ref_path = tmp_path / "ref.h5ad"
    query_path = tmp_path / "query.h5ad"
    ref.write_h5ad(ref_path)
    query.write_h5ad(query_path)
    return ref_path, query_path


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def test_train_cli_writes_model_artifacts(tmp_path):
    ref_path, _ = _write_test_h5ads(tmp_path)
    model_path = tmp_path / "model.pth"

    _run_cli(
        [
            str(TRAIN_SCRIPT),
            "--adata",
            str(ref_path),
            "--out",
            str(model_path),
            "--labels",
            "anno_lv1",
            "--hidden-sizes",
            "8",
            "--num-epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert model_path.exists()
    assert (tmp_path / "model_metadata.pkl").exists()
    assert (tmp_path / "model_reference.pkl").exists()
    assert (tmp_path / "model_manifest.json").exists()


def test_predict_cli_can_load_manifest_and_write_minimal_output(tmp_path):
    ref_path, query_path = _write_test_h5ads(tmp_path)
    model_path = tmp_path / "model.pth"
    output_path = tmp_path / "predicted.h5ad"

    _run_cli(
        [
            str(TRAIN_SCRIPT),
            "--adata",
            str(ref_path),
            "--out",
            str(model_path),
            "--labels",
            "anno_lv1",
            "--hidden-sizes",
            "8",
            "--num-epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )

    _run_cli(
        [
            str(PREDICT_SCRIPT),
            "--model",
            str(tmp_path / "model_manifest.json"),
            "--adata",
            str(query_path),
            "--out",
            str(output_path),
            "--write-mode",
            "minimal",
            "--knn-correction",
            "off",
            "--device",
            "cpu",
        ]
    )

    out = read_h5ad(output_path)
    assert "pred_anno_lv1" in out.obs.columns
    assert "conf_anno_lv1" not in out.obs.columns
    assert "atlasmtl" in out.uns


def test_train_cli_supports_no_coordinate_mode(tmp_path):
    ref_path, _ = _write_test_h5ads(tmp_path)
    ref = read_h5ad(ref_path)
    ref.obsm.clear()
    ref.write_h5ad(ref_path)

    model_path = tmp_path / "model_no_coords.pth"

    _run_cli(
        [
            str(TRAIN_SCRIPT),
            "--adata",
            str(ref_path),
            "--out",
            str(model_path),
            "--labels",
            "anno_lv1",
            "--hidden-sizes",
            "8",
            "--num-epochs",
            "1",
            "--batch-size",
            "2",
            "--no-coords",
            "--device",
            "cpu",
        ]
    )

    assert model_path.exists()
    assert (tmp_path / "model_no_coords_manifest.json").exists()


def test_train_and_predict_cli_support_preprocessing(tmp_path):
    mapping_path = tmp_path / "mapping.tsv"
    pd.DataFrame(
        {
            "human_ensembl_gene_id": ["ENSG1", "ENSG2"],
            "human_gene_symbol": ["GATA1", "CD3D"],
            "human_gene_description": ["", ""],
            "mouse_ensembl_gene_id": ["ENSMUSG1", "ENSMUSG2"],
            "mouse_gene_symbol": ["Gata1", "Cd3d"],
            "rat_ensembl_gene_id": ["ENSRNOG1", "ENSRNOG2"],
            "rat_gene_symbol": ["Gata1", "Cd3d"],
        }
    ).to_csv(mapping_path, sep="\t", index=False)

    ref_obs = pd.DataFrame({"anno_lv1": ["A", "A", "B", "B"]}, index=["r1", "r2", "r3", "r4"])
    ref = AnnData(X=np.array([[1, 0], [1, 1], [0, 1], [0, 2]], dtype=np.float32), obs=ref_obs)
    ref.var_names = ["GATA1", "CD3D"]
    ref.obsm["X_ref_latent"] = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.1, 1.0]], dtype=np.float32)
    ref.obsm["X_umap"] = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.1]], dtype=np.float32)

    query = AnnData(X=np.array([[1, 0], [0, 1]], dtype=np.float32), obs=pd.DataFrame(index=["q1", "q2"]))
    query.var_names = ["GATA1", "CD3D"]

    ref_path = tmp_path / "ref_symbol.h5ad"
    query_path = tmp_path / "query_symbol.h5ad"
    ref.write_h5ad(ref_path)
    query.write_h5ad(query_path)

    model_path = tmp_path / "model_preprocessed.pth"
    output_path = tmp_path / "predicted_preprocessed.h5ad"

    _run_cli(
        [
            str(TRAIN_SCRIPT),
            "--adata",
            str(ref_path),
            "--out",
            str(model_path),
            "--labels",
            "anno_lv1",
            "--hidden-sizes",
            "8",
            "--num-epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--var-names-type",
            "symbol",
            "--species",
            "human",
            "--gene-id-table",
            str(mapping_path),
            "--feature-space",
            "whole",
        ]
    )

    _run_cli(
        [
            str(PREDICT_SCRIPT),
            "--model",
            str(tmp_path / "model_preprocessed_manifest.json"),
            "--adata",
            str(query_path),
            "--out",
            str(output_path),
            "--write-mode",
            "minimal",
            "--knn-correction",
            "off",
            "--device",
            "cpu",
            "--var-names-type",
            "symbol",
            "--species",
            "human",
            "--gene-id-table",
            str(mapping_path),
        ]
    )

    out = read_h5ad(output_path)
    assert "pred_anno_lv1" in out.obs.columns
    assert "atlasmtl_preprocess" in out.uns
