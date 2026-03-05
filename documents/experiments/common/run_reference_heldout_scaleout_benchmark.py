#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
JOBLIB_SERIAL_FALLBACK_MARKER = "joblib will operate in serial mode"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
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


def _load_manifest(path: str) -> Dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be a YAML mapping")
    return payload


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _run_command(cmd: List[str], *, env: Dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


def _thread_policy_from_env(env: Dict[str, str]) -> Dict[str, Optional[str]]:
    return {
        "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": env.get("NUMEXPR_NUM_THREADS"),
    }


def _backend_path_for_method(method: str, *, metrics_payload: Dict[str, Any], celltypist_train: Optional[Dict[str, Any]]) -> str:
    if method == "celltypist":
        trainer = (celltypist_train or {}).get("trainer_summary") or {}
        return str(trainer.get("trainer_backend_path") or trainer.get("trainer_backend") or "unknown")
    if method == "seurat_anchor_transfer":
        results = list(metrics_payload.get("results") or [])
        if results:
            impl = str(((results[0].get("prediction_metadata") or {}).get("implementation_backend") or ""))
            if impl == "seurat_anchor_transfer_transferdata":
                return "TransferData-only"
            if impl == "seurat_anchor_transfer":
                return "MapQuery"
        return "unknown"
    if method == "scanvi":
        return "scanvi_native"
    if method == "atlasmtl":
        return "atlasmtl_native"
    if method == "singler":
        return "singler_native"
    if method == "symphony":
        return "symphony_native"
    if method == "reference_knn":
        return "reference_knn_native"
    return "unknown"


def _annotate_metrics_with_fairness(
    method_dir: Path,
    *,
    method: str,
    fairness_policy: str,
    thread_policy: Dict[str, Optional[str]],
    runtime_fairness_degraded: bool,
    degrade_reasons: List[str],
    device_requested: str,
    celltypist_train: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics_json = method_dir / "metrics.json"
    if not metrics_json.exists():
        return {}
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    results = list(payload.get("results") or [])
    if not results:
        return {}
    result = dict(results[0])
    train_usage = dict(result.get("train_usage") or {})
    predict_usage = dict(result.get("predict_usage") or {})
    device_used = str(predict_usage.get("device_used") or train_usage.get("device_used") or device_requested)
    method_backend_path = _backend_path_for_method(method, metrics_payload=payload, celltypist_train=celltypist_train)
    effective_threads_observed = (
        train_usage.get("num_threads_used")
        or predict_usage.get("num_threads_used")
        or train_usage.get("cpu_core_equiv_avg")
        or predict_usage.get("cpu_core_equiv_avg")
    )
    fairness_metadata = {
        "fairness_policy": fairness_policy,
        "thread_policy": thread_policy,
        "runtime_fairness_degraded": bool(runtime_fairness_degraded),
        "runtime_fairness_degraded_reasons": list(degrade_reasons),
        "effective_threads_observed": effective_threads_observed,
        "device_requested": device_requested,
        "device_used": device_used,
        "method_backend_path": method_backend_path,
    }
    result["fairness_metadata"] = fairness_metadata
    payload["results"] = [result]
    metrics_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return fairness_metadata


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


def _train_celltypist_model(
    *,
    reference_h5ad: Path,
    label_column: str,
    model_path: Path,
    method_cfg: Dict[str, Any],
    summary_json: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    trainer_backend = str(method_cfg.get("trainer_backend", "formal"))
    trainer_config = dict(method_cfg.get("trainer_config") or {})
    cmd = [
        sys.executable,
        str(CELLTYPIST_TRAINER),
        "--reference-h5ad",
        str(reference_h5ad.resolve()),
        "--label-column",
        label_column,
        "--output-model",
        str(model_path),
        "--trainer-backend",
        trainer_backend,
        "--max-iter",
        str(int(trainer_config.get("max_iter", 200))),
        "--n-jobs",
        str(int(trainer_config.get("n_jobs", 10))),
        "--feature-selection",
        str(bool(trainer_config.get("feature_selection", True))).lower(),
        "--balance-cell-type",
        str(bool(trainer_config.get("balance_cell_type", True))).lower(),
        "--batch-size",
        str(int(trainer_config.get("batch_size", 5000))),
        "--top-genes",
        str(int(trainer_config.get("top_genes", 500))),
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
    results = list(payload.get("results") or [])
    fairness_metadata = (results[0].get("fairness_metadata") if results else None)
    return {
        "summary_row": first_row,
        "result_count": len(payload.get("results", [])),
        "fairness_metadata": fairness_metadata,
    }


def _build_celltypist_manifest(
    manifest: Dict[str, Any],
    *,
    reference_h5ad: Path,
    query_h5ad: Path,
    model_path: Path,
) -> Dict[str, Any]:
    updated = dict(manifest)
    updated["reference_h5ad"] = str(reference_h5ad.resolve())
    updated["query_h5ad"] = str(query_h5ad.resolve())
    method_configs = dict(updated.get("method_configs") or {})
    celltypist_cfg = dict(method_configs.get("celltypist") or {})
    celltypist_cfg["model"] = str(model_path.resolve())
    method_configs["celltypist"] = celltypist_cfg
    updated["method_configs"] = method_configs
    return updated


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(args.dataset_manifest)
    celltypist_cfg = dict((manifest.get("method_configs") or {}).get("celltypist") or {})
    label_columns = list(manifest.get("label_columns") or [])
    if len(label_columns) != 1:
        raise ValueError("scale-out benchmark wrapper currently requires exactly one label column")
    label_column = str(label_columns[0])
    counts_layer = str(manifest.get("counts_layer", "counts"))
    reference_h5ad = Path(str(manifest["reference_h5ad"])).resolve()
    query_h5ad = Path(str(manifest["query_h5ad"])).resolve()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("NUMBA_CACHE_DIR", str((REPO_ROOT / ".tmp" / "numba_cache").resolve()))
    fairness_policy = str(env.get("ATLASMTL_FAIRNESS_POLICY") or ("cpu_only_strict" if args.device == "cpu" else "mixed_backend_labeled"))
    thread_policy = _thread_policy_from_env(env)

    base_manifest_path = output_dir / "runtime_manifest.yaml"
    _write_manifest(base_manifest_path, manifest)

    requested_methods = list(args.methods)
    run_celltypist = "celltypist" in requested_methods

    celltypist_model: Optional[Path] = None
    celltypist_training_summary: Optional[Path] = None
    celltypist_train: Optional[Dict[str, Any]] = None
    celltypist_manifest_path: Optional[Path] = None
    if run_celltypist:
        celltypist_inputs_dir = output_dir / "comparator_inputs" / "celltypist"
        celltypist_ref_h5ad = celltypist_inputs_dir / "reference_log1p_norm.h5ad"
        celltypist_query_h5ad = celltypist_inputs_dir / "query_log1p_norm.h5ad"
        _log1p_norm_from_counts_layer(reference_h5ad, celltypist_ref_h5ad, counts_layer=counts_layer)
        _log1p_norm_from_counts_layer(query_h5ad, celltypist_query_h5ad, counts_layer=counts_layer)

        celltypist_model = output_dir / "comparator_models" / f"celltypist_{label_column}.pkl"
        celltypist_model.parent.mkdir(parents=True, exist_ok=True)
        celltypist_training_summary = output_dir / "comparator_models" / f"celltypist_{label_column}_training_summary.json"
        celltypist_train = _train_celltypist_model(
            reference_h5ad=celltypist_ref_h5ad,
            label_column=label_column,
            model_path=celltypist_model,
            method_cfg=celltypist_cfg,
            summary_json=celltypist_training_summary,
            env=env,
        )
        if celltypist_train["returncode"] != 0:
            raise RuntimeError(
                "CellTypist model training failed before scale-out benchmark run:\n"
                f"STDOUT:\n{celltypist_train['stdout']}\n"
                f"STDERR:\n{celltypist_train['stderr']}"
            )

        celltypist_manifest = _build_celltypist_manifest(
            manifest,
            reference_h5ad=celltypist_ref_h5ad,
            query_h5ad=celltypist_query_h5ad,
            model_path=celltypist_model,
        )
        celltypist_manifest_path = output_dir / "runtime_manifest_celltypist.yaml"
        _write_manifest(celltypist_manifest_path, celltypist_manifest)

    statuses = []
    for method in requested_methods:
        method_dir = output_dir / "runs" / method
        method_dir.mkdir(parents=True, exist_ok=True)
        active_manifest = celltypist_manifest_path if (method == "celltypist" and celltypist_manifest_path is not None) else base_manifest_path
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
        degraded_reasons: List[str] = []
        if JOBLIB_SERIAL_FALLBACK_MARKER in str(completed.stderr):
            degraded_reasons.append("joblib_serial_fallback")
        if method == "celltypist" and celltypist_train and JOBLIB_SERIAL_FALLBACK_MARKER in str(celltypist_train.get("stderr", "")):
            degraded_reasons.append("joblib_serial_fallback_celltypist_train")
        status: Dict[str, Any] = {
            "method": method,
            "returncode": int(completed.returncode),
            "command": cmd,
            "stdout_path": str(method_dir / "stdout.log"),
            "stderr_path": str(method_dir / "stderr.log"),
            "fairness_policy": fairness_policy,
            "thread_policy": thread_policy,
            "runtime_fairness_degraded": bool(degraded_reasons),
            "runtime_fairness_degraded_reasons": degraded_reasons,
            "device_requested": args.device,
        }
        (method_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (method_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        fairness_metadata = _annotate_metrics_with_fairness(
            method_dir,
            method=method,
            fairness_policy=fairness_policy,
            thread_policy=thread_policy,
            runtime_fairness_degraded=bool(degraded_reasons),
            degrade_reasons=degraded_reasons,
            device_requested=args.device,
            celltypist_train=celltypist_train if method == "celltypist" else None,
        )
        if fairness_metadata:
            status["fairness_metadata"] = fairness_metadata
            status["method_backend_path"] = fairness_metadata.get("method_backend_path")
            status["device_used"] = fairness_metadata.get("device_used")
            status["effective_threads_observed"] = fairness_metadata.get("effective_threads_observed")
        if completed.returncode == 0:
            status["status"] = "success"
            status.update(_collect_method_summary(method_dir))
        else:
            status["status"] = "failed"
            status["stderr_tail"] = completed.stderr[-4000:]
        statuses.append(status)

    summary_payload = {
        "dataset_name": manifest.get("dataset_name"),
        "label_column": label_column,
        "reference_h5ad": str(reference_h5ad),
        "query_h5ad": str(query_h5ad),
        "fairness_policy": fairness_policy,
        "thread_policy": thread_policy,
        "runtime_manifest": str(base_manifest_path),
        "runtime_manifest_celltypist": None if celltypist_manifest_path is None else str(celltypist_manifest_path),
        "celltypist_model": None if celltypist_model is None else str(celltypist_model),
        "celltypist_training_summary": None if celltypist_training_summary is None else str(celltypist_training_summary),
        "celltypist_training": (celltypist_train or {}).get("trainer_summary"),
        "runtime_fairness_degraded": any(bool(item.get("runtime_fairness_degraded")) for item in statuses),
        "methods": statuses,
    }
    (output_dir / "scaleout_status.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
