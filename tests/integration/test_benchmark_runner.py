from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from anndata import AnnData, read_h5ad
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "benchmark" / "pipelines" / "run_benchmark.py"


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def _write_celltypist_model(model_path: Path, X: np.ndarray, y: np.ndarray, genes: list[str]) -> None:
    from celltypist import models

    classifier = LogisticRegression(max_iter=200)
    classifier.fit(X, y)
    classifier.features = np.asarray(genes, dtype=object)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X)
    model = models.Model(classifier, scaler, {"date": "2026-02-28", "details": "integration test model"})
    model.write(model_path)


def _write_mapping_table(path: Path) -> None:
    pd.DataFrame(
        {
            "human_ensembl_gene_id": ["ENSG1", "ENSG2", "ENSG3"],
            "human_gene_symbol": ["GATA1", "CD3D", "MS4A1"],
            "human_gene_description": ["", "", ""],
            "mouse_ensembl_gene_id": ["ENSMUSG1", "ENSMUSG2", "ENSMUSG3"],
            "mouse_gene_symbol": ["Gata1", "Cd3d", "Ms4a1"],
            "rat_ensembl_gene_id": ["ENSRNOG1", "ENSRNOG2", "ENSRNOG3"],
            "rat_gene_symbol": ["Gata1", "Cd3d", "Ms4a1"],
        }
    ).to_csv(path, sep="\t", index=False)


def test_benchmark_runner_produces_metrics_and_summary(tmp_path: Path):
    ref_obs = pd.DataFrame(
        {"anno_lv1": ["A", "A", "B", "B"], "anno_lv2": ["A1", "A1", "B1", "B1"]},
        index=["r1", "r2", "r3", "r4"],
    )
    ref = AnnData(X=np.array([[1, 0], [1, 1], [0, 1], [0, 2]], dtype=np.float32), obs=ref_obs)
    ref.var_names = ["g1", "g2"]
    ref.obsm["X_ref_latent"] = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.1, 1.0]], dtype=np.float32)
    ref.obsm["X_umap"] = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.1]], dtype=np.float32)

    query_obs = pd.DataFrame(
        {"anno_lv1": ["A", "B"], "anno_lv2": ["A1", "B1"]},
        index=["q1", "q2"],
    )
    query = AnnData(X=np.array([[1, 0], [0, 1]], dtype=np.float32), obs=query_obs)
    query.var_names = ["g1", "g2"]
    query.obsm["X_query_latent"] = np.array([[0.1, 0.0], [0.9, 1.0]], dtype=np.float32)
    query.obs["batch"] = ["b1", "b2"]

    ref_path = tmp_path / "ref.h5ad"
    query_path = tmp_path / "query.h5ad"
    ref.write_h5ad(ref_path)
    query.write_h5ad(query_path)

    manifest = {
        "dataset_name": "tiny",
        "version": 1,
        "protocol_version": 1,
        "random_seed": 2026,
        "split_name": "toy_split",
        "split_description": "tiny integration split",
        "query_subset": "heldout_query",
        "reference_h5ad": str(ref_path),
        "query_h5ad": str(query_path),
        "label_columns": ["anno_lv1", "anno_lv2"],
        "coord_targets": {"latent": "X_ref_latent", "umap": "X_umap"},
        "query_coord_targets": {"latent": "X_query_latent"},
        "domain_key": "batch",
        "train": {"num_epochs": 1, "batch_size": 2, "hidden_sizes": [8]},
        "predict": {
            "knn_correction": "all",
            "knn_vote_mode": "distance_weighted",
            "knn_reference_mode": "prototypes",
            "knn_index_mode": "exact",
            "enforce_hierarchy": True,
            "hierarchy_rules": {
                "anno_lv2": {
                    "parent_col": "anno_lv1",
                    "child_to_parent": {"A1": "A", "B1": "B"},
                }
            },
            "batch_size": 1,
        },
    }
    manifest_path = tmp_path / "dataset_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    out_dir = tmp_path / "out"
    _run_cli([str(RUNNER), "--dataset-manifest", str(manifest_path), "--output-dir", str(out_dir), "--device", "cpu"])

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "summary_by_domain.csv").exists()
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "atlasmtl_model.pth").exists()
    assert (out_dir / "atlasmtl_model_manifest.json").exists()

    # Basic sanity check: summary can be read and contains expected columns.
    summary = pd.read_csv(out_dir / "summary.csv")
    assert {"method", "dataset_name", "level", "accuracy", "coverage", "risk"}.issubset(summary.columns)
    assert {"unknown_rate", "knn_coverage", "knn_vote_mode", "knn_reference_mode", "enforce_hierarchy"}.issubset(
        summary.columns
    )
    assert set(summary["level"]) == {"anno_lv1", "anno_lv2"}
    assert set(summary["knn_vote_mode"]) == {"distance_weighted"}
    assert set(summary["knn_reference_mode"]) == {"prototypes"}
    summary_by_domain = pd.read_csv(out_dir / "summary_by_domain.csv")
    assert {"method", "dataset_name", "domain", "level", "accuracy", "unknown_rate", "knn_coverage"}.issubset(
        summary_by_domain.columns
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["protocol_version"] == 1
    assert metrics["protocol_context"]["random_seed"] == 2026
    assert metrics["protocol_context"]["split_name"] == "toy_split"
    assert len(metrics["results"]) == 1
    result = metrics["results"][0]
    assert result["protocol_version"] == 1
    assert result["model_source"] == "trained_in_runner"
    assert result["protocol_context"]["domain_key"] == "batch"
    assert "latent_rmse" in result["coordinate_metrics"]
    assert "latent_continuity" in result["coordinate_metrics"]
    assert "latent_neighbor_overlap" in result["coordinate_metrics"]
    assert "behavior_metrics" in result
    assert "anno_lv1" in result["behavior_metrics"]
    assert "hierarchy_metrics" in result
    assert "anno_lv2" in result["hierarchy_metrics"]["edges"]
    assert result["predict_config_used"]["knn_vote_mode"] == "distance_weighted"
    assert result["predict_config_used"]["knn_reference_mode"] == "prototypes"
    assert result["predict_config_used"]["enforce_hierarchy"] is True
    assert "artifact_checksums" in result
    assert result["artifact_checksums"]

    run_manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["protocol_version"] == 1
    assert run_manifest["split_name"] == "toy_split"
    assert run_manifest["random_seed"] == 2026
    assert run_manifest["domain_key"] == "batch"
    assert "artifact_checksums" in run_manifest

    # Ensure output AnnData files are still readable (runner touches only artifacts/metrics).
    _ = read_h5ad(ref_path)
    _ = read_h5ad(query_path)


def test_benchmark_runner_supports_preprocessing_manifest_fields(tmp_path: Path):
    mapping_path = tmp_path / "mapping.tsv"
    _write_mapping_table(mapping_path)

    ref_obs = pd.DataFrame({"anno_lv1": ["A", "A", "B", "B"]}, index=["r1", "r2", "r3", "r4"])
    ref = AnnData(X=np.array([[1, 0], [1, 1], [0, 1], [0, 2]], dtype=np.float32), obs=ref_obs)
    ref.var_names = ["GATA1", "CD3D"]
    ref.obsm["X_ref_latent"] = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.1, 1.0]], dtype=np.float32)

    query_obs = pd.DataFrame({"anno_lv1": ["A", "B"]}, index=["q1", "q2"])
    query = AnnData(X=np.array([[1, 0], [0, 1]], dtype=np.float32), obs=query_obs)
    query.var_names = ["GATA1", "CD3D"]

    ref_path = tmp_path / "ref_symbol.h5ad"
    query_path = tmp_path / "query_symbol.h5ad"
    ref.write_h5ad(ref_path)
    query.write_h5ad(query_path)

    manifest = {
        "dataset_name": "tiny_preprocess",
        "version": 1,
        "protocol_version": 1,
        "reference_h5ad": str(ref_path),
        "query_h5ad": str(query_path),
        "label_columns": ["anno_lv1"],
        "coord_targets": {"latent": "X_ref_latent"},
        "train": {"num_epochs": 1, "batch_size": 2, "hidden_sizes": [8]},
        "predict": {"knn_correction": "off", "batch_size": 1},
        "var_names_type": "symbol",
        "species": "human",
        "gene_id_table": str(mapping_path),
        "feature_space": "whole",
    }
    manifest_path = tmp_path / "dataset_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    out_dir = tmp_path / "out"
    _run_cli([str(RUNNER), "--dataset-manifest", str(manifest_path), "--output-dir", str(out_dir), "--device", "cpu"])

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "preprocess" in metrics
    assert metrics["preprocess"]["config"]["feature_space"] == "whole"
    result = metrics["results"][0]
    assert result["preprocess"]["config"]["var_names_type"] == "symbol"
    assert result["preprocess"]["feature_panel"]["gene_ids"] == ["ENSG1", "ENSG2"]
    run_manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["preprocess"]["config"]["species"] == "human"


def test_benchmark_runner_rejects_unknown_manifest_keys(tmp_path: Path):
    ref_obs = pd.DataFrame({"anno_lv1": ["A", "B"]}, index=["r1", "r2"])
    ref = AnnData(X=np.array([[1, 0], [0, 1]], dtype=np.float32), obs=ref_obs)
    ref.var_names = ["g1", "g2"]
    query = AnnData(X=np.array([[1, 0]], dtype=np.float32), obs=pd.DataFrame({"anno_lv1": ["A"]}, index=["q1"]))
    query.var_names = ["g1", "g2"]

    ref_path = tmp_path / "ref.h5ad"
    query_path = tmp_path / "query.h5ad"
    ref.write_h5ad(ref_path)
    query.write_h5ad(query_path)

    manifest = {
        "reference_h5ad": str(ref_path),
        "query_h5ad": str(query_path),
        "label_columns": ["anno_lv1"],
        "unexpected_key": True,
    }
    manifest_path = tmp_path / "bad_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--dataset-manifest", str(manifest_path), "--output-dir", str(tmp_path / "out")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "unsupported keys" in completed.stderr


def test_benchmark_runner_can_compare_atlasmtl_and_reference_knn(tmp_path: Path):
    ref_obs = pd.DataFrame({"anno_lv1": ["A", "A", "B", "B"]}, index=["r1", "r2", "r3", "r4"])
    ref = AnnData(X=np.array([[1, 0], [1, 1], [0, 1], [0, 2]], dtype=np.float32), obs=ref_obs)
    ref.var_names = ["g1", "g2"]
    ref.obsm["X_ref_latent"] = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.1, 1.0]], dtype=np.float32)

    query_obs = pd.DataFrame({"anno_lv1": ["A", "B"]}, index=["q1", "q2"])
    query = AnnData(X=np.array([[1, 0], [0, 1]], dtype=np.float32), obs=query_obs)
    query.var_names = ["g1", "g2"]

    ref_path = tmp_path / "ref.h5ad"
    query_path = tmp_path / "query.h5ad"
    ref.write_h5ad(ref_path)
    query.write_h5ad(query_path)

    manifest = {
        "dataset_name": "tiny_compare",
        "version": 1,
        "protocol_version": 1,
        "reference_h5ad": str(ref_path),
        "query_h5ad": str(query_path),
        "label_columns": ["anno_lv1"],
        "coord_targets": {"latent": "X_ref_latent"},
        "train": {"num_epochs": 1, "batch_size": 2, "hidden_sizes": [8]},
        "predict": {"knn_correction": "off", "batch_size": 1},
        "method_configs": {"reference_knn": {"k": 3, "input_transform": "binary"}},
    }
    manifest_path = tmp_path / "dataset_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    out_dir = tmp_path / "out"
    _run_cli(
        [
            str(RUNNER),
            "--dataset-manifest",
            str(manifest_path),
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--methods",
            "atlasmtl",
            "reference_knn",
        ]
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    methods = {item["method"] for item in metrics["results"]}
    assert methods == {"atlasmtl", "reference_knn"}

    summary = pd.read_csv(out_dir / "summary.csv")
    assert set(summary["method"]) == {"atlasmtl", "reference_knn"}


def test_benchmark_runner_can_compare_atlasmtl_and_celltypist(tmp_path: Path):
    ref_obs = pd.DataFrame({"anno_lv1": ["A", "A", "B", "B"]}, index=["r1", "r2", "r3", "r4"])
    ref = AnnData(X=np.array([[1, 0], [1, 1], [0, 1], [0, 2]], dtype=np.float32), obs=ref_obs)
    ref.var_names = ["g1", "g2"]

    query_obs = pd.DataFrame({"anno_lv1": ["A", "B"], "batch": ["b1", "b2"]}, index=["q1", "q2"])
    query = AnnData(X=np.array([[1, 0], [0, 1]], dtype=np.float32), obs=query_obs)
    query.var_names = ["g1", "g2"]

    ref_path = tmp_path / "ref.h5ad"
    query_path = tmp_path / "query.h5ad"
    ref.write_h5ad(ref_path)
    query.write_h5ad(query_path)

    celltypist_model_path = tmp_path / "celltypist_lv1.pkl"
    _write_celltypist_model(
        celltypist_model_path,
        X=np.asarray(ref.X, dtype=np.float32),
        y=ref.obs["anno_lv1"].astype(str).to_numpy(),
        genes=["g1", "g2"],
    )

    manifest = {
        "dataset_name": "tiny_celltypist",
        "version": 1,
        "protocol_version": 1,
        "reference_h5ad": str(ref_path),
        "query_h5ad": str(query_path),
        "label_columns": ["anno_lv1"],
        "domain_key": "batch",
        "train": {"num_epochs": 1, "batch_size": 2, "hidden_sizes": [8]},
        "predict": {"knn_correction": "off", "batch_size": 1},
        "method_configs": {
            "celltypist": {
                "model": str(celltypist_model_path),
                "target_label_column": "anno_lv1",
                "majority_voting": False,
            }
        },
    }
    manifest_path = tmp_path / "dataset_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    out_dir = tmp_path / "out"
    _run_cli(
        [
            str(RUNNER),
            "--dataset-manifest",
            str(manifest_path),
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--methods",
            "atlasmtl",
            "celltypist",
        ]
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    methods = {item["method"] for item in metrics["results"]}
    assert methods == {"atlasmtl", "celltypist"}

    celltypist_result = next(item for item in metrics["results"] if item["method"] == "celltypist")
    assert celltypist_result["label_columns"] == ["anno_lv1"]
    assert celltypist_result["model_source"] == "external_comparator"
    assert celltypist_result["artifact_checksums"]
    assert celltypist_result["behavior_metrics"]["anno_lv1"]["unknown_rate"] == 0.0
    assert "b1" in celltypist_result["metrics_by_domain"]
    assert celltypist_result["prediction_metadata"]["comparator_name"] == "celltypist"

    summary = pd.read_csv(out_dir / "summary.csv")
    assert set(summary["method"]) == {"atlasmtl", "celltypist"}


def test_benchmark_runner_can_compare_atlasmtl_and_scanvi(tmp_path: Path):
    rng = np.random.default_rng(0)
    ref_a = rng.poisson([5, 1, 1], size=(10, 3)).astype(np.float32)
    ref_b = rng.poisson([1, 5, 5], size=(10, 3)).astype(np.float32)
    ref_X = np.vstack([ref_a, ref_b])
    ref_obs = pd.DataFrame(
        {
            "anno_lv1": ["A"] * 10 + ["B"] * 10,
            "batch": ["b1"] * 5 + ["b2"] * 5 + ["b1"] * 5 + ["b2"] * 5,
        },
        index=[f"r{i}" for i in range(20)],
    )
    ref = AnnData(X=ref_X, obs=ref_obs)
    ref.var_names = ["g1", "g2", "g3"]

    query_X = np.vstack(
        [
            rng.poisson([5, 1, 1], size=(2, 3)),
            rng.poisson([1, 5, 5], size=(2, 3)),
        ]
    ).astype(np.float32)
    query_obs = pd.DataFrame(
        {
            "anno_lv1": ["A", "A", "B", "B"],
            "batch": ["b1", "b2", "b1", "b2"],
        },
        index=["q1", "q2", "q3", "q4"],
    )
    query = AnnData(X=query_X, obs=query_obs)
    query.var_names = ["g1", "g2", "g3"]

    ref_path = tmp_path / "ref.h5ad"
    query_path = tmp_path / "query.h5ad"
    ref.write_h5ad(ref_path)
    query.write_h5ad(query_path)

    manifest = {
        "dataset_name": "tiny_scanvi",
        "version": 1,
        "protocol_version": 1,
        "reference_h5ad": str(ref_path),
        "query_h5ad": str(query_path),
        "label_columns": ["anno_lv1"],
        "domain_key": "batch",
        "train": {"num_epochs": 1, "batch_size": 4, "hidden_sizes": [8]},
        "predict": {"knn_correction": "off", "batch_size": 2},
        "method_configs": {
            "scanvi": {
                "target_label_column": "anno_lv1",
                "batch_key": "batch",
                "n_latent": 2,
                "batch_size": 8,
                "scvi_max_epochs": 2,
                "scanvi_max_epochs": 2,
                "query_max_epochs": 2,
                "train_size": 0.9,
                "validation_size": 0.1,
                "save_model": False,
            }
        },
    }
    manifest_path = tmp_path / "dataset_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    out_dir = tmp_path / "out"
    _run_cli(
        [
            str(RUNNER),
            "--dataset-manifest",
            str(manifest_path),
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--methods",
            "atlasmtl",
            "scanvi",
        ]
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    methods = {item["method"] for item in metrics["results"]}
    assert methods == {"atlasmtl", "scanvi"}

    scanvi_result = next(item for item in metrics["results"] if item["method"] == "scanvi")
    assert scanvi_result["label_columns"] == ["anno_lv1"]
    assert scanvi_result["model_source"] == "trained_in_runner"
    assert scanvi_result["prediction_metadata"]["comparator_name"] == "scanvi"
    assert scanvi_result["prediction_metadata"]["latent_shape"] == [4, 2]
    assert "b1" in scanvi_result["metrics_by_domain"]
    assert scanvi_result["predict_config_used"]["target_label_column"] == "anno_lv1"

    summary = pd.read_csv(out_dir / "summary.csv")
    assert set(summary["method"]) == {"atlasmtl", "scanvi"}


def test_benchmark_runner_can_compare_atlasmtl_and_singler(tmp_path: Path):
    ref_obs = pd.DataFrame(
        {"anno_lv1": ["A", "A", "A", "B", "B", "B"], "batch": ["b1", "b1", "b2", "b1", "b2", "b2"]},
        index=[f"r{i}" for i in range(6)],
    )
    ref = AnnData(
        X=np.array(
            [
                [12, 1, 1],
                [11, 2, 1],
                [10, 1, 2],
                [1, 12, 10],
                [2, 11, 9],
                [1, 10, 11],
            ],
            dtype=np.float32,
        ),
        obs=ref_obs,
    )
    ref.var_names = ["g1", "g2", "g3"]
    ref.layers["counts"] = ref.X.copy()

    query_obs = pd.DataFrame(
        {"anno_lv1": ["A", "B"], "batch": ["b1", "b2"]},
        index=["q1", "q2"],
    )
    query = AnnData(
        X=np.array(
            [
                [11, 1, 1],
                [1, 11, 10],
            ],
            dtype=np.float32,
        ),
        obs=query_obs,
    )
    query.var_names = ["g1", "g2", "g3"]
    query.layers["counts"] = query.X.copy()

    ref_path = tmp_path / "ref.h5ad"
    query_path = tmp_path / "query.h5ad"
    ref.write_h5ad(ref_path)
    query.write_h5ad(query_path)

    manifest = {
        "dataset_name": "tiny_singler",
        "version": 1,
        "protocol_version": 1,
        "reference_h5ad": str(ref_path),
        "query_h5ad": str(query_path),
        "label_columns": ["anno_lv1"],
        "domain_key": "batch",
        "train": {"num_epochs": 1, "batch_size": 2, "hidden_sizes": [8]},
        "predict": {"knn_correction": "off", "batch_size": 1},
        "method_configs": {
            "singler": {
                "target_label_column": "anno_lv1",
                "reference_layer": "counts",
                "query_layer": "counts",
                "normalize_log1p": True,
                "use_pruned_labels": True,
                "fine_tune": True,
                "prune": True,
                "quantile": 0.8,
                "de_method": "classic",
                "save_raw_outputs": True,
            }
        },
    }
    manifest_path = tmp_path / "dataset_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    out_dir = tmp_path / "out"
    _run_cli(
        [
            str(RUNNER),
            "--dataset-manifest",
            str(manifest_path),
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--methods",
            "atlasmtl",
            "singler",
        ]
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    methods = {item["method"] for item in metrics["results"]}
    assert methods == {"atlasmtl", "singler"}

    singler_result = next(item for item in metrics["results"] if item["method"] == "singler")
    assert singler_result["label_columns"] == ["anno_lv1"]
    assert singler_result["model_source"] == "external_comparator"
    assert singler_result["prediction_metadata"]["comparator_name"] == "singler"
    assert "b1" in singler_result["metrics_by_domain"]
    assert singler_result["predict_config_used"]["target_label_column"] == "anno_lv1"
    assert singler_result["artifact_checksums"]

    summary = pd.read_csv(out_dir / "summary.csv")
    assert set(summary["method"]) == {"atlasmtl", "singler"}


def test_benchmark_runner_can_compare_atlasmtl_and_symphony(tmp_path: Path):
    rng = np.random.default_rng(1)
    n_noise = 27
    signal_a = rng.poisson([12, 1, 1], size=(12, 3))
    signal_b = rng.poisson([1, 12, 10], size=(12, 3))
    noise_a = rng.poisson(2, size=(12, n_noise))
    noise_b = rng.poisson(2, size=(12, n_noise))
    ref_a = np.hstack([signal_a, noise_a]).astype(np.float32)
    ref_b = np.hstack([signal_b, noise_b]).astype(np.float32)
    ref_obs = pd.DataFrame(
        {"anno_lv1": ["A"] * 12 + ["B"] * 12, "batch": ["b1", "b2", "b3"] * 8},
        index=[f"r{i}" for i in range(24)],
    )
    ref = AnnData(X=np.vstack([ref_a, ref_b]), obs=ref_obs)
    ref.var_names = [f"g{i}" for i in range(1, 31)]
    ref.layers["counts"] = ref.X.copy()

    query_obs = pd.DataFrame(
        {"anno_lv1": ["A", "A", "A", "B", "B", "B"], "batch": ["b1", "b2", "b3", "b1", "b2", "b3"]},
        index=["q1", "q2", "q3", "q4", "q5", "q6"],
    )
    query = AnnData(
        X=np.vstack(
            [
                np.hstack([rng.poisson([12, 1, 1], size=(3, 3)), rng.poisson(2, size=(3, n_noise))]),
                np.hstack([rng.poisson([1, 12, 10], size=(3, 3)), rng.poisson(2, size=(3, n_noise))]),
            ]
        ).astype(np.float32),
        obs=query_obs,
    )
    query.var_names = [f"g{i}" for i in range(1, 31)]
    query.layers["counts"] = query.X.copy()

    ref_path = tmp_path / "ref.h5ad"
    query_path = tmp_path / "query.h5ad"
    ref.write_h5ad(ref_path)
    query.write_h5ad(query_path)

    manifest = {
        "dataset_name": "tiny_symphony",
        "version": 1,
        "protocol_version": 1,
        "reference_h5ad": str(ref_path),
        "query_h5ad": str(query_path),
        "label_columns": ["anno_lv1"],
        "domain_key": "batch",
        "train": {"num_epochs": 1, "batch_size": 2, "hidden_sizes": [8]},
        "predict": {"knn_correction": "off", "batch_size": 1},
        "method_configs": {
            "symphony": {
                "target_label_column": "anno_lv1",
                "batch_key": "batch",
                "reference_layer": "counts",
                "query_layer": "counts",
                "do_normalize": True,
                "K": 2,
                "d": 2,
                "topn": 20,
                "vargenes_method": "vst",
                "sigma": 0.1,
                "knn_k": 3,
                "seed": 111,
                "save_raw_outputs": True,
            }
        },
    }
    manifest_path = tmp_path / "dataset_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    out_dir = tmp_path / "out"
    _run_cli(
        [
            str(RUNNER),
            "--dataset-manifest",
            str(manifest_path),
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--methods",
            "atlasmtl",
            "symphony",
        ]
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    methods = {item["method"] for item in metrics["results"]}
    assert methods == {"atlasmtl", "symphony"}

    symphony_result = next(item for item in metrics["results"] if item["method"] == "symphony")
    assert symphony_result["label_columns"] == ["anno_lv1"]
    assert symphony_result["model_source"] == "external_comparator"
    assert symphony_result["prediction_metadata"]["comparator_name"] == "symphony"
    assert "b1" in symphony_result["metrics_by_domain"]
    assert symphony_result["predict_config_used"]["target_label_column"] == "anno_lv1"
    assert symphony_result["artifact_checksums"]

    summary = pd.read_csv(out_dir / "summary.csv")
    assert set(summary["method"]) == {"atlasmtl", "symphony"}


def test_benchmark_runner_can_compare_atlasmtl_and_azimuth(tmp_path: Path):
    rng = np.random.default_rng(2)
    n_noise = 27
    ref_a = np.hstack([rng.poisson([12, 1, 1], size=(12, 3)), rng.poisson(2, size=(12, n_noise))]).astype(np.float32)
    ref_b = np.hstack([rng.poisson([1, 12, 10], size=(12, 3)), rng.poisson(2, size=(12, n_noise))]).astype(np.float32)
    ref_obs = pd.DataFrame(
        {"anno_lv1": ["A"] * 12 + ["B"] * 12, "batch": ["b1", "b2", "b3"] * 8},
        index=[f"r{i}" for i in range(24)],
    )
    ref = AnnData(X=np.vstack([ref_a, ref_b]), obs=ref_obs)
    ref.var_names = [f"g{i}" for i in range(1, 31)]
    ref.layers["counts"] = ref.X.copy()

    query_obs = pd.DataFrame(
        {"anno_lv1": ["A", "A", "A", "B", "B", "B"], "batch": ["b1", "b2", "b3", "b1", "b2", "b3"]},
        index=["q1", "q2", "q3", "q4", "q5", "q6"],
    )
    query = AnnData(
        X=np.vstack(
            [
                np.hstack([rng.poisson([12, 1, 1], size=(3, 3)), rng.poisson(2, size=(3, n_noise))]),
                np.hstack([rng.poisson([1, 12, 10], size=(3, 3)), rng.poisson(2, size=(3, n_noise))]),
            ]
        ).astype(np.float32),
        obs=query_obs,
    )
    query.var_names = [f"g{i}" for i in range(1, 31)]
    query.layers["counts"] = query.X.copy()

    ref_path = tmp_path / "ref.h5ad"
    query_path = tmp_path / "query.h5ad"
    ref.write_h5ad(ref_path)
    query.write_h5ad(query_path)

    manifest = {
        "dataset_name": "tiny_azimuth",
        "version": 1,
        "protocol_version": 1,
        "reference_h5ad": str(ref_path),
        "query_h5ad": str(query_path),
        "label_columns": ["anno_lv1"],
        "domain_key": "batch",
        "train": {"num_epochs": 1, "batch_size": 2, "hidden_sizes": [8]},
        "predict": {"knn_correction": "off", "batch_size": 1},
        "method_configs": {
            "azimuth": {
                "target_label_column": "anno_lv1",
                "batch_key": "batch",
                "reference_layer": "counts",
                "query_layer": "counts",
                "nfeatures": 20,
                "npcs": 5,
                "dims": [1, 2, 3, 4, 5],
                "k_anchor": 5,
                "k_score": 4,
                "k_weight": 3,
                "reference_k_param": 10,
                "mapping_score_k": 10,
                "n_trees": 5,
                "save_raw_outputs": True,
            }
        },
    }
    manifest_path = tmp_path / "dataset_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    out_dir = tmp_path / "out"
    _run_cli(
        [
            str(RUNNER),
            "--dataset-manifest",
            str(manifest_path),
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--methods",
            "atlasmtl",
            "azimuth",
        ]
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    methods = {item["method"] for item in metrics["results"]}
    assert methods == {"atlasmtl", "azimuth"}

    azimuth_result = next(item for item in metrics["results"] if item["method"] == "azimuth")
    assert azimuth_result["label_columns"] == ["anno_lv1"]
    assert azimuth_result["model_source"] == "external_comparator"
    assert azimuth_result["prediction_metadata"]["comparator_name"] == "azimuth"
    assert azimuth_result["prediction_metadata"]["implementation_backend"] in {
        "azimuth_native",
        "seurat_anchor_transfer_fallback",
    }
    assert "b1" in azimuth_result["metrics_by_domain"]
    assert azimuth_result["artifact_checksums"]

    summary = pd.read_csv(out_dir / "summary.csv")
    assert set(summary["method"]) == {"atlasmtl", "azimuth"}
