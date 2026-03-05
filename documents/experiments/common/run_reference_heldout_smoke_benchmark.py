#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "benchmark" / "pipelines" / "run_benchmark.py"
CELLTYPIST_TRAINER = (
    REPO_ROOT / "documents" / "experiments" / "2026-03-01_real_mapping_benchmark" / "scripts" / "train_celltypist_model.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--reference-h5ad", required=True)
    parser.add_argument("--query-h5ad", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--domain-key", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--reference-subset", required=True)
    parser.add_argument("--query-subset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "atlasmtl",
            "reference_knn",
            "celltypist",
            "scanvi",
            "singler",
            "symphony",
            "seurat_anchor_transfer",
        ],
    )
    return parser.parse_args()


def _run_command(cmd: List[str], *, env: Dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _log1p_norm_from_counts_layer(input_h5ad: Path, output_h5ad: Path, *, counts_layer: str = "counts") -> None:
    adata = ad.read_h5ad(str(input_h5ad))
    if counts_layer not in adata.layers:
        raise ValueError(f"missing counts layer for celltypist preparation: {counts_layer}")
    counts = adata.layers[counts_layer]
    if sparse.issparse(counts):
        counts = counts.tocsr(copy=True)
        cell_sums = np.asarray(counts.sum(axis=1)).ravel()
        scale = np.divide(1e4, cell_sums, out=np.zeros_like(cell_sums, dtype=np.float32), where=cell_sums > 0)
        norm = sparse.diags(scale.astype(np.float32)) @ counts
        norm.data = np.log1p(norm.data)
        adata.X = norm.astype(np.float32)
    else:
        counts_arr = np.asarray(counts, dtype=np.float32)
        cell_sums = counts_arr.sum(axis=1, keepdims=True)
        scale = np.divide(1e4, cell_sums, out=np.zeros_like(cell_sums, dtype=np.float32), where=cell_sums > 0)
        adata.X = np.log1p(counts_arr * scale).astype(np.float32)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(output_h5ad))


def _build_manifest(
    args: argparse.Namespace,
    *,
    celltypist_model: Path,
    reference_h5ad: Path,
    query_h5ad: Path,
) -> Dict[str, Any]:
    return {
        "dataset_name": args.dataset_name,
        "version": 1,
        "protocol_version": 1,
        "random_seed": 2026,
        "split_name": args.split_name,
        "split_description": "first-wave prepared smoke benchmark run",
        "reference_subset": args.reference_subset,
        "query_subset": args.query_subset,
        "reference_h5ad": str(reference_h5ad.resolve()),
        "query_h5ad": str(query_h5ad.resolve()),
        "label_columns": [args.label_column],
        "domain_key": args.domain_key,
        "counts_layer": "counts",
        "train": {
            "hidden_sizes": [256, 128],
            "dropout_rate": 0.2,
            "batch_size": 256,
            "num_epochs": 5,
            "learning_rate": 0.001,
            "input_transform": "binary",
            "val_fraction": 0.1,
            "early_stopping_patience": 2,
            "random_state": 2026,
            "reference_storage": "external",
        },
        "predict": {
            "knn_correction": "off",
            "batch_size": 256,
            "input_transform": "binary",
        },
        "method_configs": {
            "celltypist": {
                "target_label_column": args.label_column,
                "model": str(celltypist_model.resolve()),
            },
            "scanvi": {
                "target_label_column": args.label_column,
                "batch_key": args.domain_key,
                "scanvi_max_epochs": 5,
            },
            "singler": {
                "target_label_column": args.label_column,
                "reference_layer": "counts",
                "query_layer": "counts",
                "normalize_log1p": True,
                "use_pruned_labels": True,
            },
            "symphony": {
                "target_label_column": args.label_column,
                "batch_key": args.domain_key,
                "reference_layer": "counts",
                "query_layer": "counts",
                "topn": 2000,
                "d": 20,
                "K": 20,
            },
            "seurat_anchor_transfer": {
                "target_label_column": args.label_column,
                "batch_key": args.domain_key,
                "reference_layer": "counts",
                "query_layer": "counts",
                "nfeatures": 3000,
                "npcs": 30,
            },
            "reference_knn": {
                "input_transform": "binary",
            },
        },
    }


def _train_celltypist_model(
    args: argparse.Namespace,
    *,
    model_path: Path,
    method_cfg: Dict[str, Any],
    summary_json: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    trainer_backend = str(method_cfg.get("trainer_backend", "wrapped_logreg"))
    trainer_config = dict(method_cfg.get("trainer_config") or {})
    cmd = [
        sys.executable,
        str(CELLTYPIST_TRAINER),
        "--reference-h5ad",
        str(Path(args.reference_h5ad).resolve()),
        "--label-column",
        args.label_column,
        "--output-model",
        str(model_path),
        "--trainer-backend",
        trainer_backend,
        "--max-iter",
        str(int(trainer_config.get("max_iter", 200))),
        "--n-jobs",
        str(int(trainer_config.get("n_jobs", 1))),
        "--feature-selection",
        str(bool(trainer_config.get("feature_selection", False))).lower(),
        "--balance-cell-type",
        str(bool(trainer_config.get("balance_cell_type", False))).lower(),
        "--batch-size",
        str(int(trainer_config.get("batch_size", 1000))),
        "--top-genes",
        str(int(trainer_config.get("top_genes", 300))),
        "--use-gpu",
        str(bool(trainer_config.get("use_gpu", False))).lower(),
        "--with-mean",
        str(bool(trainer_config.get("with_mean", False))).lower(),
        "--min-cells-per-class",
        str(int(trainer_config.get("min_cells_per_class", 0))),
        "--summary-json",
        str(summary_json.resolve()),
    ]
    completed = _run_command(cmd, env=env, cwd=REPO_ROOT)
    trainer_summary = None
    if summary_json.exists():
        trainer_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    return {
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "command": cmd,
        "trainer_backend": trainer_backend,
        "trainer_summary": trainer_summary,
    }


def _collect_method_summary(method_dir: Path) -> Dict[str, Any]:
    summary_csv = method_dir / "summary.csv"
    metrics_json = method_dir / "metrics.json"
    if not summary_csv.exists() or not metrics_json.exists():
        return {}
    summary = pd.read_csv(summary_csv)
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    first_row = summary.iloc[0].to_dict() if not summary.empty else {}
    return {
        "summary_row": first_row,
        "result_count": len(payload.get("results", [])),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    celltypist_method_cfg: Dict[str, Any] = {}

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("NUMBA_CACHE_DIR", str((REPO_ROOT / ".tmp" / "numba_cache").resolve()))

    celltypist_inputs_dir = output_dir / "comparator_inputs" / "celltypist"
    celltypist_ref_h5ad = celltypist_inputs_dir / "reference_log1p_norm.h5ad"
    celltypist_query_h5ad = celltypist_inputs_dir / "query_log1p_norm.h5ad"
    _log1p_norm_from_counts_layer(Path(args.reference_h5ad), celltypist_ref_h5ad)
    _log1p_norm_from_counts_layer(Path(args.query_h5ad), celltypist_query_h5ad)

    celltypist_model = output_dir / "comparator_models" / f"celltypist_{args.label_column}.pkl"
    celltypist_model.parent.mkdir(parents=True, exist_ok=True)
    celltypist_training_summary = output_dir / "comparator_models" / f"celltypist_{args.label_column}_training_summary.json"
    celltypist_args = argparse.Namespace(**vars(args))
    celltypist_args.reference_h5ad = str(celltypist_ref_h5ad)
    celltypist_train = _train_celltypist_model(
        celltypist_args,
        model_path=celltypist_model,
        method_cfg=celltypist_method_cfg,
        summary_json=celltypist_training_summary,
        env=env,
    )
    if celltypist_train["returncode"] != 0:
        raise RuntimeError(
            "CellTypist model training failed before smoke benchmark run:\n"
            f"STDOUT:\n{celltypist_train['stdout']}\n"
            f"STDERR:\n{celltypist_train['stderr']}"
        )

    manifest_payload = _build_manifest(
        args,
        celltypist_model=celltypist_model,
        reference_h5ad=Path(args.reference_h5ad),
        query_h5ad=Path(args.query_h5ad),
    )
    manifest_path = output_dir / "runtime_manifest.yaml"
    _write_manifest(manifest_path, manifest_payload)
    celltypist_manifest_payload = _build_manifest(
        args,
        celltypist_model=celltypist_model,
        reference_h5ad=celltypist_ref_h5ad,
        query_h5ad=celltypist_query_h5ad,
    )
    celltypist_manifest_path = output_dir / "runtime_manifest_celltypist.yaml"
    _write_manifest(celltypist_manifest_path, celltypist_manifest_payload)

    statuses = []
    for method in args.methods:
        method_dir = output_dir / "runs" / method
        method_dir.mkdir(parents=True, exist_ok=True)
        active_manifest = celltypist_manifest_path if method == "celltypist" else manifest_path
        cmd = [
            sys.executable,
            str(RUNNER),
            "--dataset-manifest",
            str(active_manifest),
            "--output-dir",
            str(method_dir),
            "--methods",
            method,
            "--device",
            args.device,
        ]
        completed = _run_command(cmd, env=env, cwd=REPO_ROOT)
        status: Dict[str, Any] = {
            "method": method,
            "returncode": int(completed.returncode),
            "command": cmd,
            "stdout_path": str(method_dir / "stdout.log"),
            "stderr_path": str(method_dir / "stderr.log"),
        }
        (method_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (method_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode == 0:
            status["status"] = "success"
            status.update(_collect_method_summary(method_dir))
        else:
            status["status"] = "failed"
            status["stderr_tail"] = completed.stderr[-4000:]
        statuses.append(status)

    summary_payload = {
        "dataset_name": args.dataset_name,
        "label_column": args.label_column,
        "reference_h5ad": str(Path(args.reference_h5ad).resolve()),
        "query_h5ad": str(Path(args.query_h5ad).resolve()),
        "runtime_manifest": str(manifest_path),
        "runtime_manifest_celltypist": str(celltypist_manifest_path),
        "celltypist_model": str(celltypist_model),
        "celltypist_training_summary": str(celltypist_training_summary),
        "celltypist_training": celltypist_train.get("trainer_summary"),
        "methods": statuses,
    }
    (output_dir / "smoke_status.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
