#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUNNER = REPO_ROOT / "documents" / "experiments" / "common" / "run_reference_heldout_scaleout_benchmark.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage_a", "stage_b"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument(
        "--datasets-config",
        default="documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/configs/datasets.yaml",
    )
    parser.add_argument(
        "--grid-config",
        default="documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/configs/param_grid.yaml",
    )
    parser.add_argument(
        "--prepared-root",
        default="/tmp/atlasmtl_benchmarks/2026-03-07/reference_heldout",
    )
    parser.add_argument(
        "--output-root",
        default="/tmp/atlasmtl_benchmarks/2026-03-07/atlasmtl_param_lock",
    )
    parser.add_argument("--runner-script", default=str(DEFAULT_RUNNER))
    parser.add_argument(
        "--top-params-json",
        default="",
        help="optional; defaults to <output-root>/stage_a/<device>/top2_params.json for stage_b",
    )
    parser.add_argument("--dataset-names", default="", help="optional comma-separated dataset names")
    parser.add_argument("--param-ids", default="", help="optional comma-separated param_id values")
    parser.add_argument("--query-sizes", default="", help="optional comma-separated query sizes")
    parser.add_argument("--seed-override", type=int, default=-1)
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid yaml mapping: {path}")
    return payload


def _parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_hidden_sizes(value: Any) -> List[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, tuple):
        return [int(x) for x in value]
    if isinstance(value, str):
        tokens = [x for x in re.split(r"[\s,\[\]\(\)]+", value.strip()) if x]
        if not tokens:
            raise ValueError(f"invalid hidden_sizes string: {value!r}")
        return [int(x) for x in tokens]
    raise TypeError(f"unsupported hidden_sizes type: {type(value)!r}")


def _build_manifest(
    dataset: Dict[str, Any],
    *,
    prepared_root: Path,
    query_size: int,
    seed: int,
    split_tag: str,
    param: Dict[str, Any],
    fixed: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    dataset_name = str(dataset["dataset_name"])
    prepared_dir = prepared_root / dataset_name / "prepared" / "param_lock_train10k"
    reference_h5ad = prepared_dir / "reference_train_10k.h5ad"
    query_h5ad = prepared_dir / ("heldout_test_5k.h5ad" if query_size == 5000 else "heldout_test_10k.h5ad")
    if not reference_h5ad.exists():
        raise FileNotFoundError(f"missing prepared reference: {reference_h5ad}")
    if not query_h5ad.exists():
        raise FileNotFoundError(f"missing prepared query: {query_h5ad}")

    label_column = str(dataset["label_column"])
    domain_key = str(dataset["domain_key"])
    counts_layer = str(dataset.get("counts_layer", "counts"))
    input_matrix_type = str(dataset.get("input_matrix_type", "counts"))
    batch_size = int(param["batch_size"])

    hidden_sizes = _parse_hidden_sizes(param["hidden_sizes"])

    return {
        "dataset_name": dataset_name,
        "version": 1,
        "protocol_version": 1,
        "random_seed": int(seed),
        "split_name": f"atlasmtl_param_lock_{split_tag}",
        "split_description": (
            "atlasmtl pre-formal parameter lock run "
            f"({split_tag}, train10k, predict{query_size // 1000}k, device={device})"
        ),
        "reference_subset": "train_10k_prepared",
        "query_subset": f"heldout_test_{query_size // 1000}k_prepared",
        "reference_h5ad": str(reference_h5ad),
        "query_h5ad": str(query_h5ad),
        "label_columns": [label_column],
        "domain_key": domain_key,
        "input_matrix_type": input_matrix_type,
        "counts_layer": counts_layer,
        "train": {
            "hidden_sizes": hidden_sizes,
            "dropout_rate": 0.1,
            "batch_size": batch_size,
            "num_epochs": int(fixed["max_epochs"]),
            "learning_rate": float(param["learning_rate"]),
            "input_transform": str(fixed["input_transform"]),
            "val_fraction": float(fixed["val_fraction"]),
            "early_stopping_patience": int(fixed["early_stopping_patience"]),
            "early_stopping_min_delta": float(fixed["early_stopping_min_delta"]),
            "random_state": int(seed),
            "reference_storage": str(fixed["reference_storage"]),
        },
        "predict": {
            "knn_correction": str(fixed["predict_knn_correction"]),
            "batch_size": batch_size,
            "input_transform": str(fixed["input_transform"]),
        },
        "method_configs": {
            "atlasmtl": {
                "num_threads": int(fixed["num_threads"]),
                "device": device,
            }
        },
    }


def _read_result_row(
    run_dir: Path,
    *,
    stage: str,
    device: str,
    dataset_name: str,
    param_id: str,
    seed: int,
    query_size: int,
) -> Dict[str, Any]:
    summary_csv = run_dir / "runs" / "atlasmtl" / "summary.csv"
    metrics_json = run_dir / "runs" / "atlasmtl" / "metrics.json"
    row: Dict[str, Any] = {
        "stage": stage,
        "device": device,
        "dataset_name": dataset_name,
        "param_id": param_id,
        "seed": int(seed),
        "query_size": int(query_size),
        "success": False,
    }
    if not summary_csv.exists() or not metrics_json.exists():
        return row
    summary_df = pd.read_csv(summary_csv)
    if summary_df.empty:
        return row
    first = summary_df.iloc[0].to_dict()
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    result = (payload.get("results") or [{}])[0]
    train_usage = dict(result.get("train_usage") or {})
    predict_usage = dict(result.get("predict_usage") or {})
    fairness = dict(result.get("fairness_metadata") or {})
    row.update(
        {
            "success": True,
            "accuracy": float(first.get("accuracy", float("nan"))),
            "macro_f1": float(first.get("macro_f1", float("nan"))),
            "train_elapsed_seconds": float(train_usage.get("elapsed_seconds", float("nan"))),
            "predict_elapsed_seconds": float(predict_usage.get("elapsed_seconds", float("nan"))),
            "train_peak_rss_gb": float(train_usage.get("process_peak_rss_gb", float("nan"))),
            "predict_peak_rss_gb": float(predict_usage.get("process_peak_rss_gb", float("nan"))),
            "train_gpu_peak_memory_gb": float(train_usage.get("gpu_peak_memory_gb", float("nan"))),
            "predict_gpu_peak_memory_gb": float(predict_usage.get("gpu_peak_memory_gb", float("nan"))),
            "train_items_per_second": float(train_usage.get("items_per_second", float("nan"))),
            "predict_items_per_second": float(predict_usage.get("items_per_second", float("nan"))),
            "fairness_policy": fairness.get("fairness_policy"),
            "runtime_fairness_degraded": fairness.get("runtime_fairness_degraded"),
            "effective_threads_observed": fairness.get("effective_threads_observed"),
        }
    )
    return row


def _stage_b_top_params(
    *,
    args: argparse.Namespace,
    output_root: Path,
    device: str,
    stage_a_grid: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    top_json = (
        Path(args.top_params_json).resolve()
        if args.top_params_json
        else (output_root / "stage_a" / device / "top2_params.json")
    )
    if top_json.exists():
        payload = json.loads(top_json.read_text(encoding="utf-8"))
        items = list(payload.get("top_params") or [])
        if items:
            return items[:top_k]
    return stage_a_grid[:top_k]


def main() -> None:
    args = parse_args()
    datasets_cfg = _load_yaml(Path(args.datasets_config).resolve())
    grid_cfg = _load_yaml(Path(args.grid_config).resolve())
    datasets = list(datasets_cfg.get("datasets") or [])
    if not datasets:
        raise ValueError("datasets config has empty `datasets` list")
    dataset_filter = set(_parse_csv_list(args.dataset_names))
    if dataset_filter:
        datasets = [d for d in datasets if str(d.get("dataset_name")) in dataset_filter]
        if not datasets:
            raise ValueError(f"dataset filter matched nothing: {sorted(dataset_filter)}")

    output_root = Path(args.output_root).resolve()
    stage_root = output_root / args.stage / args.device
    stage_root.mkdir(parents=True, exist_ok=True)
    manifests_root = stage_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    prepared_root = Path(args.prepared_root).resolve()

    fixed = {
        "num_threads": int(grid_cfg.get("num_threads", 8)),
        "input_transform": str(grid_cfg.get("input_transform", "binary")),
        "reference_storage": str(grid_cfg.get("reference_storage", "external")),
        "max_epochs": int(grid_cfg.get("max_epochs", 50)),
        "val_fraction": float(grid_cfg.get("val_fraction", 0.1)),
        "early_stopping_patience": int(grid_cfg.get("early_stopping_patience", 5)),
        "early_stopping_min_delta": float(grid_cfg.get("early_stopping_min_delta", 0.0)),
        "predict_knn_correction": str(grid_cfg.get("predict_knn_correction", "off")),
    }
    stage_a_cfg = dict(grid_cfg.get("stage_a") or {})
    stage_b_cfg = dict(grid_cfg.get("stage_b") or {})
    stage_a_grid = list(stage_a_cfg.get("cpu_grid" if args.device == "cpu" else "gpu_grid") or [])
    if not stage_a_grid:
        raise ValueError(f"empty stage_a grid for device={args.device}")

    if args.stage == "stage_a":
        default_seed = int(stage_a_cfg.get("default_seed", 2026))
        seed_list = [int(args.seed_override)] if args.seed_override >= 0 else [default_seed]
        query_sizes = [int(stage_a_cfg.get("query_size", 5000))]
        param_grid = stage_a_grid
    else:
        top_k = int(stage_b_cfg.get("top_k", 2))
        seed_list = list(stage_b_cfg.get("seeds") or [17, 23])
        if args.seed_override >= 0:
            seed_list = [int(args.seed_override)]
        query_sizes = [int(x) for x in (stage_b_cfg.get("query_sizes") or [5000, 10000])]
        param_grid = _stage_b_top_params(
            args=args,
            output_root=output_root,
            device=args.device,
            stage_a_grid=stage_a_grid,
            top_k=top_k,
        )

    param_filter = set(_parse_csv_list(args.param_ids))
    if param_filter:
        param_grid = [p for p in param_grid if str(p.get("param_id")) in param_filter]
        if not param_grid:
            raise ValueError(f"param filter matched nothing: {sorted(param_filter)}")
    query_override = [int(x) for x in _parse_csv_list(args.query_sizes)]
    if query_override:
        query_sizes = query_override

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("NUMBA_CACHE_DIR", str((REPO_ROOT / ".tmp" / "numba_cache").resolve()))
    env["ATLASMTL_FAIRNESS_POLICY"] = "cpu_only_strict" if args.device == "cpu" else "mixed_backend_labeled"
    thread_env = str(fixed["num_threads"])
    env["OMP_NUM_THREADS"] = thread_env
    env["MKL_NUM_THREADS"] = thread_env
    env["OPENBLAS_NUM_THREADS"] = thread_env
    env["NUMEXPR_NUM_THREADS"] = thread_env

    runner_script = Path(args.runner_script).resolve()
    if not runner_script.exists():
        raise FileNotFoundError(f"runner script not found: {runner_script}")

    run_records: List[Dict[str, Any]] = []
    for dataset in datasets:
        dataset_name = str(dataset["dataset_name"])
        for param in param_grid:
            param_id = str(param["param_id"])
            for seed in seed_list:
                for query_size in query_sizes:
                    split_tag = f"{args.stage}_{args.device}_{param_id}_q{query_size // 1000}k_s{seed}"
                    run_name = f"{dataset_name}__{split_tag}"
                    run_dir = stage_root / "runs" / run_name
                    run_dir.mkdir(parents=True, exist_ok=True)

                    manifest_payload = _build_manifest(
                        dataset,
                        prepared_root=prepared_root,
                        query_size=int(query_size),
                        seed=int(seed),
                        split_tag=split_tag,
                        param=param,
                        fixed=fixed,
                        device=args.device,
                    )
                    manifest_path = manifests_root / f"{run_name}.yaml"
                    manifest_path.write_text(yaml.safe_dump(manifest_payload, sort_keys=False), encoding="utf-8")

                    cmd = [
                        sys.executable,
                        str(runner_script),
                        "--dataset-manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(run_dir),
                        "--methods",
                        "atlasmtl",
                        "--device",
                        args.device,
                    ]
                    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
                    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
                    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")

                    record = {
                        "run_name": run_name,
                        "stage": args.stage,
                        "device": args.device,
                        "dataset_name": dataset_name,
                        "param_id": param_id,
                        "query_size": int(query_size),
                        "seed": int(seed),
                        "returncode": int(completed.returncode),
                        "manifest_path": str(manifest_path),
                        "run_dir": str(run_dir),
                        "learning_rate": float(param["learning_rate"]),
                        "hidden_sizes": ",".join(str(x) for x in list(param["hidden_sizes"])),
                        "batch_size": int(param["batch_size"]),
                    }
                    record.update(
                        _read_result_row(
                            run_dir,
                            stage=args.stage,
                            device=args.device,
                            dataset_name=dataset_name,
                            param_id=param_id,
                            seed=int(seed),
                            query_size=int(query_size),
                        )
                    )
                    run_records.append(record)

    run_df = pd.DataFrame(run_records)
    run_csv = stage_root / "run_index.csv"
    run_json = stage_root / "run_index.json"
    run_df.to_csv(run_csv, index=False)
    run_json.write_text(json.dumps(run_records, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "stage": args.stage,
        "device": args.device,
        "num_runs": int(len(run_records)),
        "num_success": int(sum(1 for x in run_records if bool(x.get("success")) and int(x.get("returncode", 1)) == 0)),
        "num_failed": int(sum(1 for x in run_records if int(x.get("returncode", 1)) != 0)),
        "run_index_csv": str(run_csv),
        "run_index_json": str(run_json),
    }
    (stage_root / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
