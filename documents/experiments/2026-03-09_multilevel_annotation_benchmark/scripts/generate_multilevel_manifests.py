#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from generate_hierarchy_rules import build_hierarchy_rules


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_multilevel_annotation_benchmark"
SOURCE_INDEX = (
    REPO_ROOT
    / "documents"
    / "experiments"
    / "2026-03-06_formal_third_wave_scaling"
    / "manifests"
    / "reference_heldout"
    / "manifest_index.json"
)

DATASET_LEVELS: Dict[str, List[str]] = {
    "HLCA_Core": ["ann_level_1", "ann_level_2", "ann_level_3", "ann_level_4", "ann_level_5"],
    "PHMap_Lung_Full_v43_light": ["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"],
    "DISCO_hPBMCs": ["cell_type", "cell_subtype"],
    "mTCA": ["Cell_type_level1", "Cell_type_level2", "Cell_type_level3"],
}


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


def _point_name(item: Dict[str, Any]) -> Optional[str]:
    track = str(item["track"])
    build_size = int(item["build_size"])
    predict_size = int(item["predict_size"])
    if track == "build_scaling" and build_size == 100000 and predict_size == 10000:
        return "build_100000_eval10k"
    if track == "predict_scaling" and build_size == 100000 and predict_size == 10000:
        return "predict_100000_10000"
    return None


def _device_group(value: str) -> str:
    return "cpu_core" if value == "cpu" else value


def _keep_row(item: Dict[str, Any]) -> bool:
    dataset = str(item["dataset_name"])
    if dataset not in DATASET_LEVELS:
        return False
    return _device_group(str(item["device_group"])) in {"cpu_core", "gpu"} and _point_name(item) is not None


def main() -> None:
    out_root = ROUND_ROOT / "manifests" / "multilevel"
    rules_root = ROUND_ROOT / "manifests" / "hierarchy_rules"
    out_root.mkdir(parents=True, exist_ok=True)
    rules_root.mkdir(parents=True, exist_ok=True)

    rows = []
    cached_rules: Dict[str, str] = {}
    for item in _load_json(SOURCE_INDEX):
        if not _keep_row(item):
            continue
        dataset = str(item["dataset_name"])
        levels = DATASET_LEVELS[dataset]
        point = _point_name(item)
        assert point is not None
        source_manifest = Path(str(item["manifest_path"])).resolve()
        payload = _load_yaml(source_manifest)
        rules_path = rules_root / f"{dataset}.json"
        if dataset not in cached_rules:
            rules = build_hierarchy_rules(Path(str(payload["reference_h5ad"])).resolve(), levels)
            rules_path.write_text(json.dumps(rules, indent=2, sort_keys=True), encoding="utf-8")
            cached_rules[dataset] = rules_path.resolve().as_posix()

        payload["label_columns"] = levels
        payload["experiment_round"] = "2026-03-09_multilevel_annotation_benchmark"
        payload["source_formal_manifest"] = source_manifest.as_posix()
        payload["generated_from_manifest_index"] = SOURCE_INDEX.resolve().as_posix()

        method_configs = dict(payload.get("method_configs") or {})
        atlas_cfg = dict(method_configs.get("atlasmtl") or {})
        atlas_cfg.update(
            {
                "reference_layer": str(payload.get("counts_layer", "counts")),
                "query_layer": str(payload.get("counts_layer", "counts")),
            }
        )
        method_configs["atlasmtl"] = atlas_cfg
        payload["method_configs"] = method_configs

        train_cfg = dict(payload.get("train") or {})
        train_cfg.update(
            {
                "input_transform": "binary",
                "optimizer_name": "adamw",
                "weight_decay": 5e-5,
                "scheduler_name": None,
            }
        )
        payload["train"] = train_cfg

        predict_cfg = dict(payload.get("predict") or {})
        predict_cfg.update(
            {
                "input_transform": "binary",
                "knn_correction": "off",
                "enforce_hierarchy": True,
                "hierarchy_rules": json.loads(Path(cached_rules[dataset]).read_text(encoding="utf-8")),
            }
        )
        payload["predict"] = predict_cfg

        device_group = _device_group(str(item["device_group"]))
        out_path = out_root / dataset / device_group / point / "atlasmtl_multilevel.yaml"
        _write_yaml(out_path, payload)
        rows.append(
            {
                "dataset_name": dataset,
                "device_group": device_group,
                "track": str(item["track"]),
                "build_size": int(item["build_size"]),
                "predict_size": int(item["predict_size"]),
                "point": point,
                "config_name": "atlasmtl_multilevel",
                "manifest_path": out_path.resolve().as_posix(),
                "source_formal_manifest": source_manifest.as_posix(),
                "hierarchy_rules_path": cached_rules[dataset],
                "label_columns": levels,
            }
        )
    if len(rows) != 16:
        raise ValueError(f"expected 16 multilevel manifests, found {len(rows)}")
    (out_root / "manifest_index.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_count": len(rows),
                "output_root": out_root.resolve().as_posix(),
                "rules_root": rules_root.resolve().as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
