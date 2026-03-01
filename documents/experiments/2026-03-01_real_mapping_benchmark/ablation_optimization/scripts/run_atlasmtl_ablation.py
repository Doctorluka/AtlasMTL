#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[5]
RUNNER = REPO_ROOT / "benchmark" / "pipelines" / "run_benchmark.py"
EXPORT_TABLES = REPO_ROOT / "benchmark" / "reports" / "export_paper_tables.py"
EXPORT_REPORT = REPO_ROOT / "benchmark" / "reports" / "generate_markdown_report.py"
CUDA_GATE = Path(__file__).resolve().with_name("check_cuda_gate.py")
PHMAP_TASK_WEIGHTS = [0.3, 0.8, 1.5, 2.0]
UNIFORM_TASK_WEIGHTS = [1.0, 1.0, 1.0, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AtlasMTL ablation grid on the real mapping benchmark.")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--feature-modes", nargs="+", default=["whole", "hvg3000", "hvg6000"])
    parser.add_argument("--input-transforms", nargs="+", default=["binary", "float"])
    parser.add_argument("--task-weight-schemes", nargs="+", default=["uniform", "phmap"])
    parser.add_argument("--target-label-column", default="anno_lv4")
    return parser.parse_args()


def _run_cli(args: List[str], *, cwd: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".tmp" / "numba_cache"))
    subprocess.run([sys.executable, *args], cwd=cwd, env=env, check=True, text=True)


def _load_manifest(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be a YAML mapping")
    return payload


def _feature_cfg(mode: str) -> Dict[str, Any]:
    if mode == "whole":
        return {"feature_space": "whole", "hvg_config": None}
    if mode == "hvg3000":
        return {"feature_space": "hvg", "hvg_config": {"method": "seurat_v3", "n_top_genes": 3000}}
    if mode == "hvg6000":
        return {"feature_space": "hvg", "hvg_config": {"method": "seurat_v3", "n_top_genes": 6000}}
    raise ValueError(f"unsupported feature mode: {mode}")


def _task_weights(scheme: str) -> List[float]:
    if scheme == "uniform":
        return list(UNIFORM_TASK_WEIGHTS)
    if scheme == "phmap":
        return list(PHMAP_TASK_WEIGHTS)
    raise ValueError(f"unsupported task-weight scheme: {scheme}")


def _variant_name(device: str, feature_mode: str, input_transform: str, weight_scheme: str) -> str:
    return f"atlasmtl_{device}_{feature_mode}_{input_transform}_{weight_scheme}"


def _iter_grid(devices: Iterable[str], feature_modes: Iterable[str], transforms: Iterable[str], weight_schemes: Iterable[str]):
    for device in devices:
        for feature_mode in feature_modes:
            for transform in transforms:
                for weight_scheme in weight_schemes:
                    yield {
                        "device": device,
                        "feature_mode": feature_mode,
                        "input_transform": transform,
                        "task_weight_scheme": weight_scheme,
                        "variant_name": _variant_name(device, feature_mode, transform, weight_scheme),
                    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_manifest_dir = output_dir / "generated_manifests"
    generated_manifest_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = _load_manifest(Path(args.dataset_manifest).resolve())

    gate_dir = output_dir / "cuda_gate"
    _run_cli([str(CUDA_GATE), "--output-dir", str(gate_dir)], cwd=REPO_ROOT)
    gate = json.loads((gate_dir / "cuda_gate.json").read_text(encoding="utf-8"))
    devices = list(args.devices)
    if "cuda" in devices and not gate.get("gate_passed"):
        devices = [device for device in devices if device != "cuda"]

    aggregate_results: List[Dict[str, Any]] = []
    for combo in _iter_grid(devices, args.feature_modes, args.input_transforms, args.task_weight_schemes):
        manifest = deepcopy(base_manifest)
        manifest["feature_space"] = _feature_cfg(combo["feature_mode"])["feature_space"]
        hvg_config = _feature_cfg(combo["feature_mode"])["hvg_config"]
        if hvg_config is None:
            manifest.pop("hvg_config", None)
        else:
            manifest["hvg_config"] = hvg_config
        manifest.setdefault("train", {})
        manifest["train"]["input_transform"] = combo["input_transform"]
        manifest["train"]["task_weights"] = _task_weights(combo["task_weight_scheme"])
        manifest.setdefault("method_configs", {})
        manifest["method_configs"]["atlasmtl"] = {
            "reference_layer": manifest.get("counts_layer", "counts"),
            "query_layer": manifest.get("counts_layer", "counts"),
            "task_weights": _task_weights(combo["task_weight_scheme"]),
        }

        manifest_path = generated_manifest_dir / f"{combo['variant_name']}.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        run_dir = runs_dir / combo["variant_name"]
        _run_cli(
            [
                str(RUNNER),
                "--dataset-manifest",
                str(manifest_path),
                "--output-dir",
                str(run_dir),
                "--methods",
                "atlasmtl",
                "--device",
                combo["device"],
            ],
            cwd=REPO_ROOT,
        )
        payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        result = dict(payload["results"][0])
        result.setdefault("variant_name", combo["variant_name"])
        result.setdefault("ablation_config", {})
        result["ablation_config"]["run_dir"] = str(run_dir)
        aggregate_results.append(result)

    aggregate_payload = {
        "protocol_version": base_manifest.get("protocol_version", 1),
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "dataset_name": base_manifest.get("dataset_name"),
        "dataset_version": base_manifest.get("version"),
        "cuda_gate": gate,
        "results": aggregate_results,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(aggregate_payload, indent=2, sort_keys=True), encoding="utf-8")
    _run_cli(
        [
            str(EXPORT_TABLES),
            "--metrics-json",
            str(metrics_path),
            "--output-dir",
            str(output_dir / "paper_tables"),
            "--target-label-column",
            args.target_label_column,
        ],
        cwd=REPO_ROOT,
    )
    _run_cli(
        [
            str(EXPORT_REPORT),
            "--metrics-json",
            str(metrics_path),
            "--output",
            str(output_dir / "benchmark_report.md"),
            "--target-label-column",
            args.target_label_column,
        ],
        cwd=REPO_ROOT,
    )
    manifest = {
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "output_dir": str(output_dir),
        "devices_requested": list(args.devices),
        "devices_run": devices,
        "feature_modes": list(args.feature_modes),
        "input_transforms": list(args.input_transforms),
        "task_weight_schemes": list(args.task_weight_schemes),
        "variants": [result.get("variant_name") for result in aggregate_results],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(metrics_path)


if __name__ == "__main__":
    main()
