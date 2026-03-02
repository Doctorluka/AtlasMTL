from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import anndata as ad
import yaml


REPO_ROOT = Path(__file__).resolve().parents[5]
RUNNER = REPO_ROOT / "benchmark" / "pipelines" / "run_benchmark.py"
EXPORT_TABLES = REPO_ROOT / "benchmark" / "reports" / "export_paper_tables.py"
EXPORT_REPORT = REPO_ROOT / "benchmark" / "reports" / "generate_markdown_report.py"
CUDA_GATE = Path(__file__).resolve().with_name("check_cuda_gate.py")
PHMAP_TASK_WEIGHTS = [0.3, 0.8, 1.5, 2.0]
UNIFORM_TASK_WEIGHTS = [1.0, 1.0, 1.0, 1.0]


def run_cli(args: List[str], *, cwd: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".tmp" / "numba_cache"))
    subprocess.run([sys.executable, *args], cwd=cwd, env=env, check=True, text=True)


def load_manifest(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be a YAML mapping")
    return payload


def resolve_data_path(value: str, *, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    manifest_relative = (manifest_path.parent / path).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return (REPO_ROOT / path).resolve()


def build_hierarchy_rules(reference_h5ad: Path) -> Dict[str, Any]:
    adata = ad.read_h5ad(reference_h5ad)
    rules: Dict[str, Any] = {}
    for child_col, parent_col in (("anno_lv2", "anno_lv1"), ("anno_lv3", "anno_lv2"), ("anno_lv4", "anno_lv3")):
        frame = adata.obs.loc[:, [child_col, parent_col]].dropna().copy()
        frame[child_col] = frame[child_col].astype(str)
        frame[parent_col] = frame[parent_col].astype(str)
        dedup = frame.drop_duplicates(subset=[child_col, parent_col], keep="first")
        rules[child_col] = {
            "parent_col": parent_col,
            "child_to_parent": dedup.set_index(child_col)[parent_col].to_dict(),
        }
    return rules


def run_cuda_gate(*, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cli([str(CUDA_GATE), "--output-dir", str(output_dir)], cwd=REPO_ROOT)
    return json.loads((output_dir / "cuda_gate.json").read_text(encoding="utf-8"))


def resolve_devices(requested_devices: Iterable[str], gate: Dict[str, Any]) -> List[str]:
    devices = list(requested_devices)
    if "cuda" in devices and not gate.get("gate_passed"):
        devices = [device for device in devices if device != "cuda"]
    return devices


def parse_feature_mode(mode: str) -> Dict[str, Any]:
    if mode == "whole":
        return {"feature_space": "whole", "hvg_config": None, "n_top_genes": None}
    match = re.fullmatch(r"hvg(\d+)", mode)
    if not match:
        raise ValueError(f"unsupported feature mode: {mode}")
    n_top_genes = int(match.group(1))
    return {
        "feature_space": "hvg",
        "hvg_config": {"method": "seurat_v3", "n_top_genes": n_top_genes},
        "n_top_genes": n_top_genes,
    }


def prepare_manifest(
    *,
    base_manifest: Dict[str, Any],
    dataset_manifest_path: Path,
    feature_mode: str,
    input_transform: str,
    task_weights: List[float],
) -> Dict[str, Any]:
    manifest = deepcopy(base_manifest)
    feature_cfg = parse_feature_mode(feature_mode)
    manifest["feature_space"] = feature_cfg["feature_space"]
    if feature_cfg["hvg_config"] is None:
        manifest.pop("hvg_config", None)
    else:
        manifest["hvg_config"] = feature_cfg["hvg_config"]
    manifest.setdefault("train", {})
    manifest["train"]["input_transform"] = input_transform
    manifest["train"]["task_weights"] = task_weights
    manifest.setdefault("predict", {})
    manifest["predict"]["hierarchy_rules"] = build_hierarchy_rules(
        resolve_data_path(str(base_manifest["reference_h5ad"]), manifest_path=dataset_manifest_path)
    )
    manifest.setdefault("method_configs", {})
    manifest["method_configs"]["atlasmtl"] = {
        "reference_layer": manifest.get("counts_layer", "counts"),
        "query_layer": manifest.get("counts_layer", "counts"),
        "task_weights": task_weights,
    }
    return manifest


def run_atlasmtl_variant(
    *,
    manifest_path: Path,
    output_dir: Path,
    device: str,
) -> Dict[str, Any]:
    run_cli(
        [
            str(RUNNER),
            "--dataset-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--methods",
            "atlasmtl",
            "--device",
            device,
        ],
        cwd=REPO_ROOT,
    )
    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    return dict(payload["results"][0])


def write_standard_outputs(
    *,
    metrics_path: Path,
    target_label_column: str,
) -> None:
    run_cli(
        [
            str(EXPORT_TABLES),
            "--metrics-json",
            str(metrics_path),
            "--output-dir",
            str(metrics_path.parent / "paper_tables"),
            "--target-label-column",
            target_label_column,
        ],
        cwd=REPO_ROOT,
    )
    run_cli(
        [
            str(EXPORT_REPORT),
            "--metrics-json",
            str(metrics_path),
            "--output",
            str(metrics_path.parent / "benchmark_report.md"),
            "--target-label-column",
            target_label_column,
        ],
        cwd=REPO_ROOT,
    )


def aggregate_runtime_peak(result: Dict[str, Any], key: str) -> Optional[float]:
    train_usage = result.get("train_usage") or {}
    predict_usage = result.get("predict_usage") or {}
    values = [train_usage.get(key), predict_usage.get(key)]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def first_hierarchy_edge_rate(result: Dict[str, Any]) -> Optional[float]:
    hierarchy_metrics = result.get("hierarchy_metrics") or {}
    edges = hierarchy_metrics.get("edges") or {}
    if not edges:
        return None
    return min(
        float((payload or {}).get("path_consistency_rate", 1.0))
        for payload in edges.values()
    )

