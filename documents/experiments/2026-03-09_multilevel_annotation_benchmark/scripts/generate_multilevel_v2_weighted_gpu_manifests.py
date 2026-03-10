#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_multilevel_annotation_benchmark"
SOURCE_INDEX = ROUND_ROOT / "manifests" / "multilevel" / "manifest_index.json"
OUT_ROOT = ROUND_ROOT / "manifests" / "multilevel_v2_weighted_gpu"
TASK_WEIGHTS = [0.2, 0.7, 1.5, 3.0]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    rows = []
    for item in _load_json(SOURCE_INDEX):
        if str(item["device_group"]) != "gpu":
            continue
        src_manifest = Path(str(item["manifest_path"])).resolve()
        payload = _load_yaml(src_manifest)
        train_cfg = dict(payload.get("train") or {})
        train_cfg.update(
            {
                "task_weights": list(TASK_WEIGHTS),
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
                "enforce_hierarchy": True,
                "input_transform": "binary",
            }
        )
        payload["predict"] = predict_cfg
        payload["experiment_round"] = "2026-03-09_multilevel_annotation_benchmark"
        payload["config_name"] = "atlasmtl_multilevel_v2_weighted_gpu"
        out_path = OUT_ROOT / str(item["dataset_name"]) / "gpu" / str(item["point"]) / "atlasmtl_multilevel_v2_weighted_gpu.yaml"
        _write_yaml(out_path, payload)
        rows.append(
            {
                "dataset_name": str(item["dataset_name"]),
                "device_group": "gpu",
                "track": str(item["track"]),
                "build_size": int(item["build_size"]),
                "predict_size": int(item["predict_size"]),
                "point": str(item["point"]),
                "config_name": "atlasmtl_multilevel_v2_weighted_gpu",
                "manifest_path": out_path.resolve().as_posix(),
                "source_v1_manifest": src_manifest.as_posix(),
                "task_weights": list(TASK_WEIGHTS),
            }
        )
    if len(rows) != 8:
        raise ValueError(f"expected 8 v2 weighted gpu manifests, found {len(rows)}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "manifest_index.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest_count": len(rows), "output_root": OUT_ROOT.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
