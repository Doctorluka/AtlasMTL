#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_atlasmtl_low_cost_optimization"
SOURCE_INDEX = (
    REPO_ROOT
    / "documents"
    / "experiments"
    / "2026-03-06_formal_third_wave_scaling"
    / "manifests"
    / "reference_heldout"
    / "manifest_index.json"
)

REPRESENTATIVE_POINTS = {
    ("build_scaling", 100000, 10000): "build_100000_eval10k",
    ("predict_scaling", 100000, 10000): "predict_100000_10000",
}

STAGE_DATASETS = {
    "stage_a": ["PHMap_Lung_Full_v43_light", "mTCA"],
    "stage_b": ["HLCA_Core", "PHMap_Lung_Full_v43_light", "mTCA", "DISCO_hPBMCs"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage_a", "stage_b"], required=True)
    parser.add_argument("--best-wd", choices=["1e-5", "5e-5", "1e-4"])
    parser.add_argument("--candidate-config", choices=["adamw_wd", "adamw_wd_plateau"])
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a YAML mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _point_name(item: Dict[str, Any]) -> Optional[str]:
    key = (item["track"], int(item["build_size"]), int(item["predict_size"]))
    return REPRESENTATIVE_POINTS.get(key)


def _config_rows(stage: str, best_wd: Optional[str], candidate_config: Optional[str]) -> List[Dict[str, Any]]:
    if stage == "stage_a":
        rows = [
            {"config_name": "baseline", "optimizer_name": "adam", "weight_decay": 0.0, "scheduler_name": None},
            {"config_name": "adamw_wd_1e5", "optimizer_name": "adamw", "weight_decay": 1e-5, "scheduler_name": None},
            {"config_name": "adamw_wd_5e5", "optimizer_name": "adamw", "weight_decay": 5e-5, "scheduler_name": None},
            {"config_name": "adamw_wd_1e4", "optimizer_name": "adamw", "weight_decay": 1e-4, "scheduler_name": None},
        ]
        if best_wd is not None:
            rows.append(
                {
                    "config_name": "adamw_bestwd_plateau",
                    "optimizer_name": "adamw",
                    "weight_decay": float(best_wd),
                    "scheduler_name": "reduce_lr_on_plateau",
                }
            )
        return rows

    if candidate_config is None:
        raise ValueError("--candidate-config is required for stage_b")
    candidate_scheduler = "reduce_lr_on_plateau" if candidate_config == "adamw_wd_plateau" else None
    if best_wd is None:
        raise ValueError("--best-wd is required for stage_b")
    return [
        {"config_name": "baseline", "optimizer_name": "adam", "weight_decay": 0.0, "scheduler_name": None},
        {
            "config_name": "candidate_default",
            "optimizer_name": "adamw",
            "weight_decay": float(best_wd),
            "scheduler_name": candidate_scheduler,
        },
    ]


def _atlasmtl_train_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "optimizer_name": config["optimizer_name"],
        "weight_decay": float(config["weight_decay"]),
        "scheduler_name": config["scheduler_name"],
    }
    if config["scheduler_name"] is not None:
        payload.update(
            {
                "scheduler_factor": 0.5,
                "scheduler_patience": 5,
                "scheduler_min_lr": 1e-6,
                "scheduler_monitor": "val_loss",
            }
        )
    return payload


def _iter_selected_index_rows(stage: str) -> Iterable[Dict[str, Any]]:
    datasets = set(STAGE_DATASETS[stage])
    for item in _load_json(SOURCE_INDEX):
        if item["dataset_name"] not in datasets:
            continue
        if _point_name(item) is None:
            continue
        yield item


def main() -> None:
    args = parse_args()
    out_root = ROUND_ROOT / "manifests" / args.stage
    configs = _config_rows(args.stage, args.best_wd, args.candidate_config)
    index_rows: List[Dict[str, Any]] = []

    for item in _iter_selected_index_rows(args.stage):
        source_manifest = Path(item["manifest_path"])
        source_payload = _load_yaml(source_manifest)
        point = _point_name(item)
        if point is None:
            continue

        for config in configs:
            payload = dict(source_payload)
            train_cfg = dict(payload.get("train") or {})
            train_cfg.update(_atlasmtl_train_overrides(config))
            payload["train"] = train_cfg
            payload["experiment_round"] = "2026-03-09_atlasmtl_low_cost_optimization"
            payload["optimization_stage"] = args.stage
            payload["config_name"] = config["config_name"]
            payload["source_formal_manifest"] = str(source_manifest.resolve())
            payload["generated_from_manifest_index"] = str(SOURCE_INDEX.resolve())

            output_path = (
                out_root
                / item["dataset_name"]
                / item["device_group"]
                / point
                / f"{config['config_name']}.yaml"
            )
            _write_yaml(output_path, payload)
            index_rows.append(
                {
                    "stage": args.stage,
                    "dataset_name": item["dataset_name"],
                    "device_group": item["device_group"],
                    "track": item["track"],
                    "build_size": int(item["build_size"]),
                    "predict_size": int(item["predict_size"]),
                    "point": point,
                    "config_name": config["config_name"],
                    "manifest_path": str(output_path.resolve()),
                    "source_formal_manifest": str(source_manifest.resolve()),
                }
            )

    (out_root / "manifest_index.json").write_text(
        json.dumps(index_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"stage": args.stage, "manifest_count": len(index_rows), "output_root": str(out_root)}, indent=2))


if __name__ == "__main__":
    main()
