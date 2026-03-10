#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_formal_atlasmtl_refresh"
SOURCE_INDEX = (
    REPO_ROOT
    / "documents"
    / "experiments"
    / "2026-03-06_formal_third_wave_scaling"
    / "manifests"
    / "reference_heldout"
    / "manifest_index.json"
)
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
    if track == "build_scaling" and build_size == 50000 and predict_size == 10000:
        return "build_50000_eval10k"
    if track == "predict_scaling" and build_size == 50000 and predict_size == 10000:
        return "predict_50000_10000"
    return None


def _keep_row(item: Dict[str, Any]) -> bool:
    ds = str(item["dataset_name"])
    dg = "cpu_core" if str(item["device_group"]) == "cpu" else str(item["device_group"])
    point = _point_name(item)
    if point is None:
        return False
    if ds in {"HLCA_Core", "PHMap_Lung_Full_v43_light", "mTCA", "DISCO_hPBMCs"}:
        return dg in {"cpu_core", "gpu"} and point in {"build_100000_eval10k", "predict_100000_10000"}
    if ds == "Vento":
        return dg in {"cpu_core", "gpu"} and point in {"build_50000_eval10k", "predict_50000_10000"}
    return False


def main() -> None:
    rows = []
    out_root = ROUND_ROOT / "manifests" / "refresh"
    for item in _load_json(SOURCE_INDEX):
        if not _keep_row(item):
            continue
        point = _point_name(item)
        assert point is not None
        source_manifest = Path(item["manifest_path"])
        payload = _load_yaml(source_manifest)
        train_cfg = dict(payload.get("train") or {})
        train_cfg.update(
            {
                "optimizer_name": "adamw",
                "weight_decay": 5e-5,
                "scheduler_name": None,
            }
        )
        payload["train"] = train_cfg
        payload["experiment_round"] = "2026-03-09_formal_atlasmtl_refresh"
        payload["config_name"] = "atlasmtl_refreshed_default"
        payload["source_formal_manifest"] = source_manifest.resolve().as_posix()
        payload["generated_from_manifest_index"] = SOURCE_INDEX.resolve().as_posix()

        device_group = "cpu_core" if str(item["device_group"]) == "cpu" else str(item["device_group"])
        out_path = out_root / str(item["dataset_name"]) / device_group / point / "atlasmtl_refreshed_default.yaml"
        _write_yaml(out_path, payload)
        rows.append(
            {
                "dataset_name": str(item["dataset_name"]),
                "device_group": device_group,
                "track": str(item["track"]),
                "build_size": int(item["build_size"]),
                "predict_size": int(item["predict_size"]),
                "point": point,
                "config_name": "atlasmtl_refreshed_default",
                "manifest_path": out_path.resolve().as_posix(),
                "source_formal_manifest": source_manifest.resolve().as_posix(),
                "generated_from_manifest_index": SOURCE_INDEX.resolve().as_posix(),
            }
        )
    if len(rows) != 20:
        raise ValueError(f"expected 20 refresh manifests, found {len(rows)}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest_index.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest_count": len(rows), "output_root": out_root.resolve().as_posix()}, indent=2))


if __name__ == "__main__":
    main()
