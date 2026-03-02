#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from ablation_common import (
    PHMAP_TASK_WEIGHTS,
    prepare_manifest,
    load_manifest,
    parse_feature_mode,
    resolve_devices,
    run_atlasmtl_variant,
    run_cuda_gate,
    write_standard_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AtlasMTL KNN correction ablation on the real mapping benchmark.")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--feature-mode", default="hvg6000")
    parser.add_argument("--input-transform", default="binary")
    parser.add_argument("--task-weights", nargs="+", type=float, default=list(PHMAP_TASK_WEIGHTS))
    parser.add_argument("--target-label-column", default="anno_lv4")
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--knn-conf-low", type=float, default=0.6)
    return parser.parse_args()


def _knn_variants() -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("knn_off", {"knn_correction": "off"}),
        ("knn_lowconf", {"knn_correction": "low_conf_only"}),
        ("knn_all", {"knn_correction": "all"}),
    ]


def _variant_name(device: str, feature_mode: str, input_transform: str, knn_variant: str) -> str:
    return f"atlasmtl_{device}_{feature_mode}_{input_transform}_{knn_variant}"


def _iter_grid(
    devices: Iterable[str],
    feature_mode: str,
    input_transform: str,
) -> Iterable[Dict[str, Any]]:
    for device in devices:
        for knn_variant, knn_cfg in _knn_variants():
            yield {
                "device": device,
                "feature_mode": feature_mode,
                "input_transform": input_transform,
                "knn_variant": knn_variant,
                "knn_cfg": knn_cfg,
                "variant_name": _variant_name(device, feature_mode, input_transform, knn_variant),
            }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_manifest_dir = output_dir / "generated_manifests"
    generated_manifest_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest_path = Path(args.dataset_manifest).resolve()
    base_manifest = load_manifest(dataset_manifest_path)

    gate = run_cuda_gate(output_dir=output_dir / "cuda_gate")
    devices = resolve_devices(args.devices, gate)
    feature_cfg = parse_feature_mode(args.feature_mode)

    results: List[Dict[str, Any]] = []
    for combo in _iter_grid(devices, args.feature_mode, args.input_transform):
        manifest = prepare_manifest(
            base_manifest=base_manifest,
            dataset_manifest_path=dataset_manifest_path,
            feature_mode=combo["feature_mode"],
            input_transform=combo["input_transform"],
            task_weights=list(args.task_weights),
        )
        manifest.setdefault("predict", {})
        manifest["predict"].update(
            {
                **dict(manifest["predict"]),
                **dict(combo["knn_cfg"]),
                "knn_k": int(args.knn_k),
                "knn_conf_low": float(args.knn_conf_low),
            }
        )
        manifest_path = generated_manifest_dir / f"{combo['variant_name']}.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

        run_dir = runs_dir / combo["variant_name"]
        result = run_atlasmtl_variant(manifest_path=manifest_path, output_dir=run_dir, device=combo["device"])
        result["variant_name"] = combo["variant_name"]
        result["ablation_config"] = {
            **dict(result.get("ablation_config") or {}),
            "variant_name": combo["variant_name"],
            "device": combo["device"],
            "feature_space": feature_cfg["feature_space"],
            "n_top_genes": feature_cfg["n_top_genes"],
            "input_transform": combo["input_transform"],
            "task_weights": list(args.task_weights),
            "feature_mode": combo["feature_mode"],
            "knn_variant": combo["knn_variant"],
            "knn_correction": combo["knn_cfg"]["knn_correction"],
            "knn_k": int(args.knn_k),
            "knn_conf_low": float(args.knn_conf_low),
            "search_family": "knn_eval",
            "run_dir": str(run_dir),
        }
        results.append(result)

    metrics_path = output_dir / "metrics.json"
    payload = {
        "protocol_version": base_manifest.get("protocol_version", 1),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_name": base_manifest.get("dataset_name"),
        "dataset_version": base_manifest.get("version"),
        "search_family": "knn_eval",
        "cuda_gate": gate,
        "results": results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_standard_outputs(metrics_path=metrics_path, target_label_column=args.target_label_column)
    print(metrics_path)


if __name__ == "__main__":
    main()
