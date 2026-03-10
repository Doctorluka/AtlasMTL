#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
PHASE2_MANIFEST_ROOT = DOSSIER_ROOT / "manifests" / "phase2_seed" / "predict" / "lv4strong_plus_class_weight"
OUT_ROOT = DOSSIER_ROOT / "manifests" / "phase6c"
PHASE2_MODEL_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-09/phmap_study_split_phase2_seed/train/lv4strong_plus_class_weight")

SEEDS = (101, 17, 2026, 23, 47)
HOTSPOT_TOP6 = [
    "CD4+ T",
    "SMC de-differentiated",
    "Mph alveolar",
    "EC vascular",
    "Fibro adventitial",
    "CD8+ T",
]

CONFIGS: Dict[str, Dict[str, Any]] = {
    "correction_frozen_base_reranker_like": {
        "parent_conditioned_child_correction": {
            "parent_level": "anno_lv3",
            "target_level": "anno_lv4",
            "hotspot_parents": HOTSPOT_TOP6,
            "mode": "frozen_base",
            "feature_mode": "reranker_like",
            "base_lr_scale": 0.1,
            "loss_weight": 1.0,
            "rank_loss_weight": 0.2,
            "rank_margin": 0.2,
            "hidden_dim": 64,
            "residual_scale": 1.0,
        }
    },
}


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    train_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        source_manifest = PHASE2_MANIFEST_ROOT / f"seed_{seed}" / "build_100000_eval10k" / "atlasmtl_phase2_seed_predict.yaml"
        payload = yaml.safe_load(source_manifest.read_text(encoding="utf-8"))
        init_model_path = (
            PHASE2_MODEL_ROOT / f"seed_{seed}" / "runs" / "atlasmtl" / "atlasmtl_model_manifest.json"
        ).resolve().as_posix()
        for config_name, overrides in CONFIGS.items():
            out_payload = dict(payload)
            train_cfg = dict(out_payload.get("train") or {})
            train_cfg.update(overrides)
            train_cfg["init_model_path"] = init_model_path
            out_payload["train"] = train_cfg
            out_payload["optimization_stage"] = "phase6c_frozen_base_reranker_like"
            out_payload["config_name"] = f"atlasmtl_phase6c_{config_name}_seed_{seed}"
            out_payload["seed"] = seed
            out_path = OUT_ROOT / "train" / config_name / f"seed_{seed}" / "atlasmtl_phase6c_train.yaml"
            _write_yaml(out_path, out_payload)
            train_rows.append(
                {
                    "config_name": config_name,
                    "seed": seed,
                    "train_manifest_path": out_path.resolve().as_posix(),
                    "init_model_path": init_model_path,
                    "hotspot_parents": HOTSPOT_TOP6,
                }
            )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "train_manifest_index.json").write_text(json.dumps(train_rows, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "train_manifest_count": len(train_rows),
                "configs": list(CONFIGS.keys()),
                "seeds": list(SEEDS),
                "hotspot_top6": HOTSPOT_TOP6,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
