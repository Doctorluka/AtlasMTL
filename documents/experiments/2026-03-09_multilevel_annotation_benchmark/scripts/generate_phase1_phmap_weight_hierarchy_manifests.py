#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_multilevel_annotation_benchmark"
SOURCE_ROOT = ROUND_ROOT / "manifests" / "multilevel" / "PHMap_Lung_Full_v43_light" / "gpu"
OUT_ROOT = ROUND_ROOT / "manifests" / "phase1_phmap_weight_hierarchy"

POINTS = ("build_100000_eval10k", "predict_100000_10000")
CONFIGS: Dict[str, List[float]] = {
    "uniform_control": [1.0, 1.0, 1.0, 1.0],
    "lv4strong_candidate": [0.2, 0.7, 1.5, 3.0],
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _source_manifest(point: str) -> Path:
    return SOURCE_ROOT / point / "atlasmtl_multilevel.yaml"


def _base_payload(point: str) -> Dict[str, Any]:
    return _load_yaml(_source_manifest(point))


def _configure_payload(payload: Dict[str, Any], *, config_name: str, task_weights: List[float], enforce_hierarchy: bool) -> Dict[str, Any]:
    train_cfg = dict(payload.get("train") or {})
    train_cfg.update(
        {
            "task_weights": list(task_weights),
            "num_epochs": 50,
            "learning_rate": 3e-4,
            "optimizer_name": "adamw",
            "weight_decay": 5e-5,
            "scheduler_name": None,
            "input_transform": "binary",
        }
    )
    payload["train"] = train_cfg

    predict_cfg = dict(payload.get("predict") or {})
    predict_cfg.update(
        {
            "knn_correction": "off",
            "enforce_hierarchy": enforce_hierarchy,
            "input_transform": "binary",
        }
    )
    payload["predict"] = predict_cfg

    payload["experiment_round"] = "2026-03-09_multilevel_annotation_benchmark"
    payload["optimization_stage"] = "phase1_phmap_weight_hierarchy"
    payload["config_name"] = config_name
    return payload


def main() -> None:
    train_rows = []
    predict_rows = []

    train_point = "build_100000_eval10k"
    for config_name, task_weights in CONFIGS.items():
        train_payload = _configure_payload(
            _base_payload(train_point),
            config_name=f"atlasmtl_phase1_{config_name}_train",
            task_weights=task_weights,
            enforce_hierarchy=True,
        )
        train_path = OUT_ROOT / "train" / config_name / "atlasmtl_phase1_train.yaml"
        _write_yaml(train_path, train_payload)
        train_rows.append(
            {
                "config_name": config_name,
                "train_manifest_path": train_path.resolve().as_posix(),
                "source_manifest": _source_manifest(train_point).resolve().as_posix(),
                "task_weights": list(task_weights),
                "reference_point": train_point,
            }
        )

        for point in POINTS:
            for hierarchy_setting, enforce_hierarchy in (("on", True), ("off", False)):
                predict_payload = _configure_payload(
                    _base_payload(point),
                    config_name=f"atlasmtl_phase1_{config_name}_{point}_hierarchy_{hierarchy_setting}",
                    task_weights=task_weights,
                    enforce_hierarchy=enforce_hierarchy,
                )
                predict_path = (
                    OUT_ROOT
                    / "predict"
                    / config_name
                    / point
                    / f"hierarchy_{hierarchy_setting}"
                    / "atlasmtl_phase1_predict.yaml"
                )
                _write_yaml(predict_path, predict_payload)
                predict_rows.append(
                    {
                        "config_name": config_name,
                        "point": point,
                        "hierarchy_setting": hierarchy_setting,
                        "enforce_hierarchy": enforce_hierarchy,
                        "predict_manifest_path": predict_path.resolve().as_posix(),
                        "source_manifest": _source_manifest(point).resolve().as_posix(),
                        "task_weights": list(task_weights),
                    }
                )

    if len(train_rows) != 2:
        raise ValueError(f"expected 2 training manifests, found {len(train_rows)}")
    if len(predict_rows) != 8:
        raise ValueError(f"expected 8 predict manifests, found {len(predict_rows)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "train_manifest_index.json").write_text(
        json.dumps(train_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (OUT_ROOT / "predict_manifest_index.json").write_text(
        json.dumps(predict_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (OUT_ROOT / "manifest_index.json").write_text(
        json.dumps({"train": train_rows, "predict": predict_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "train_manifest_count": len(train_rows),
                "predict_manifest_count": len(predict_rows),
                "output_root": OUT_ROOT.resolve().as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
