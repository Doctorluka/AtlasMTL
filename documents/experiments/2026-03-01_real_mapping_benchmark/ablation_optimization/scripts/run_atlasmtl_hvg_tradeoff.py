#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from ablation_common import (
    PHMAP_TASK_WEIGHTS,
    load_manifest,
    parse_feature_mode,
    prepare_manifest,
    resolve_devices,
    run_atlasmtl_variant,
    run_cuda_gate,
    write_standard_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AtlasMTL HVG tradeoff search on the real mapping benchmark.")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=["whole", "hvg3000", "hvg4000", "hvg5000", "hvg6000", "hvg7000", "hvg8000"],
    )
    parser.add_argument("--input-transform", default="binary")
    parser.add_argument("--target-label-column", default="anno_lv4")
    return parser.parse_args()


def _variant_name(device: str, feature_mode: str, input_transform: str) -> str:
    return f"atlasmtl_{device}_{feature_mode}_{input_transform}_phmap"


def _iter_grid(devices: Iterable[str], feature_modes: Iterable[str], input_transform: str) -> Iterable[Dict[str, str]]:
    for device in devices:
        for feature_mode in feature_modes:
            yield {
                "device": device,
                "feature_mode": feature_mode,
                "input_transform": input_transform,
                "variant_name": _variant_name(device, feature_mode, input_transform),
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

    results: List[Dict[str, Any]] = []
    for combo in _iter_grid(devices, args.feature_modes, args.input_transform):
        manifest = prepare_manifest(
            base_manifest=base_manifest,
            dataset_manifest_path=dataset_manifest_path,
            feature_mode=combo["feature_mode"],
            input_transform=combo["input_transform"],
            task_weights=list(PHMAP_TASK_WEIGHTS),
        )
        manifest_path = generated_manifest_dir / f"{combo['variant_name']}.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        run_dir = runs_dir / combo["variant_name"]
        result = run_atlasmtl_variant(manifest_path=manifest_path, output_dir=run_dir, device=combo["device"])
        feature_cfg = parse_feature_mode(combo["feature_mode"])
        result.setdefault("variant_name", combo["variant_name"])
        result["ablation_config"] = {
            **dict(result.get("ablation_config") or {}),
            "device": combo["device"],
            "feature_space": feature_cfg["feature_space"],
            "n_top_genes": feature_cfg["n_top_genes"],
            "input_transform": combo["input_transform"],
            "task_weight_scheme": "phmap",
            "task_weights": list(PHMAP_TASK_WEIGHTS),
            "feature_mode": combo["feature_mode"],
            "search_family": "hvg_tradeoff",
            "run_dir": str(run_dir),
        }
        results.append(result)

    metrics_path = output_dir / "metrics.json"
    payload = {
        "protocol_version": base_manifest.get("protocol_version", 1),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_name": base_manifest.get("dataset_name"),
        "dataset_version": base_manifest.get("version"),
        "search_family": "hvg_tradeoff",
        "cuda_gate": gate,
        "results": results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_standard_outputs(metrics_path=metrics_path, target_label_column=args.target_label_column)

    run_manifest = {
        "dataset_manifest": str(dataset_manifest_path),
        "output_dir": str(output_dir),
        "devices_requested": list(args.devices),
        "devices_run": devices,
        "feature_modes": list(args.feature_modes),
        "input_transform": args.input_transform,
        "task_weights": list(PHMAP_TASK_WEIGHTS),
        "variants": [result.get("variant_name") for result in results],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(metrics_path)


if __name__ == "__main__":
    main()
