#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from anndata import read_h5ad


REPO_ROOT = Path(__file__).resolve().parents[4]
DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
OUT_ROOT = DOSSIER_ROOT / "manifests" / "phase1"
PREP_ROOT = Path(
    "/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation_study_split/PHMap_Lung_Full_v43_light/prepared/formal_split_v1"
)

LABEL_COLUMNS = ["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"]
CONFIGS: Dict[str, List[float]] = {
    "uniform_control": [1.0, 1.0, 1.0, 1.0],
    "lv4strong_candidate": [0.2, 0.7, 1.5, 3.0],
}
POINTS = ("build_100000_eval10k", "predict_100000_10000")


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _hierarchy_rules() -> Dict[str, Dict[str, object]]:
    ref_path = PREP_ROOT / "build_scaling" / "build_100000" / "reference_train_100000.h5ad"
    adata = read_h5ad(ref_path)
    rules: Dict[str, Dict[str, object]] = {}
    for parent_col, child_col in zip(LABEL_COLUMNS[:-1], LABEL_COLUMNS[1:]):
        frame = adata.obs.loc[:, [parent_col, child_col]].dropna().copy()
        frame[parent_col] = frame[parent_col].astype(str)
        frame[child_col] = frame[child_col].astype(str)
        dedup = frame.drop_duplicates(subset=[child_col, parent_col], keep="first")
        rules[str(child_col)] = {
            "parent_col": str(parent_col),
            "child_to_parent": dedup.set_index(child_col)[parent_col].to_dict(),
        }
    return rules


def _base_manifest(point: str) -> Dict[str, Any]:
    if point == "build_100000_eval10k":
        reference_h5ad = PREP_ROOT / "build_scaling" / "build_100000" / "reference_train_100000.h5ad"
        query_h5ad = PREP_ROOT / "build_scaling" / "build_100000" / "heldout_build_eval_10k.h5ad"
        split_name = "study_build_scaling_gpu_build100000_eval10k_v1"
        split_description = "PH-Map study-grouped build scaling manifest (gpu); build=100000, eval=10k"
        reference_subset = "reference_train_100000_prepared"
        query_subset = "heldout_build_eval_10k_prepared"
    else:
        reference_h5ad = PREP_ROOT / "build_scaling" / "build_100000" / "reference_train_100000.h5ad"
        query_h5ad = PREP_ROOT / "predict_scaling" / "fixed_build_100000" / "heldout_predict_10000.h5ad"
        split_name = "study_predict_scaling_gpu_build100000_predict10000_v1"
        split_description = "PH-Map study-grouped predict scaling manifest (gpu); build=100000, predict=10k"
        reference_subset = "reference_train_100000_prepared"
        query_subset = "heldout_predict_10000_prepared"
    return {
        "dataset_name": "PHMap_Lung_Full_v43_light",
        "version": 1,
        "protocol_version": 1,
        "random_seed": 2026,
        "split_name": split_name,
        "split_description": split_description,
        "reference_subset": reference_subset,
        "query_subset": query_subset,
        "reference_h5ad": str(reference_h5ad),
        "query_h5ad": str(query_h5ad),
        "label_columns": list(LABEL_COLUMNS),
        "domain_key": "study",
        "input_matrix_type": "infer",
        "counts_layer": "counts",
        "var_names_type": "ensembl",
        "species": "human",
        "canonical_target": "ensembl",
        "mapping_table_kind": "biomart_human_mouse_rat",
        "feature_space": "hvg",
        "hvg_config": {"method": "seurat_v3", "n_top_genes": 3000},
        "method_configs": {"atlasmtl": {"reference_layer": "counts", "query_layer": "counts"}},
    }


def _configured_manifest(point: str, config_name: str, task_weights: List[float], enforce_hierarchy: bool, rules: Dict[str, Any]) -> Dict[str, Any]:
    payload = _base_manifest(point)
    payload["train"] = {
        "hidden_sizes": [1024, 512],
        "dropout_rate": 0.2,
        "batch_size": 512,
        "num_epochs": 50,
        "learning_rate": 3e-4,
        "input_transform": "binary",
        "val_fraction": 0.1,
        "early_stopping_patience": 5,
        "random_state": 2026,
        "reference_storage": "external",
        "optimizer_name": "adamw",
        "weight_decay": 5e-5,
        "scheduler_name": None,
        "task_weights": list(task_weights),
    }
    payload["predict"] = {
        "knn_correction": "off",
        "batch_size": 512,
        "input_transform": "binary",
        "enforce_hierarchy": enforce_hierarchy,
        "hierarchy_rules": rules,
    }
    payload["experiment_round"] = "2026-03-09_phmap_study_split_validation"
    payload["optimization_stage"] = "phase1"
    payload["config_name"] = config_name
    return payload


def main() -> None:
    rules = _hierarchy_rules()
    train_rows = []
    predict_rows = []

    for config_name, task_weights in CONFIGS.items():
        train_payload = _configured_manifest(
            "build_100000_eval10k",
            f"atlasmtl_study_split_phase1_{config_name}_train",
            task_weights,
            True,
            rules,
        )
        train_path = OUT_ROOT / "train" / config_name / "atlasmtl_phase1_train.yaml"
        _write_yaml(train_path, train_payload)
        train_rows.append(
            {
                "config_name": config_name,
                "task_weights": list(task_weights),
                "train_manifest_path": train_path.resolve().as_posix(),
            }
        )
        for point in POINTS:
            for hierarchy_setting, enforce_hierarchy in (("on", True), ("off", False)):
                predict_payload = _configured_manifest(
                    point,
                    f"atlasmtl_study_split_phase1_{config_name}_{point}_hierarchy_{hierarchy_setting}",
                    task_weights,
                    enforce_hierarchy,
                    rules,
                )
                predict_path = OUT_ROOT / "predict" / config_name / point / f"hierarchy_{hierarchy_setting}" / "atlasmtl_phase1_predict.yaml"
                _write_yaml(predict_path, predict_payload)
                predict_rows.append(
                    {
                        "config_name": config_name,
                        "point": point,
                        "hierarchy_setting": hierarchy_setting,
                        "enforce_hierarchy": enforce_hierarchy,
                        "task_weights": list(task_weights),
                        "predict_manifest_path": predict_path.resolve().as_posix(),
                    }
                )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "train_manifest_index.json").write_text(json.dumps(train_rows, indent=2, sort_keys=True), encoding="utf-8")
    (OUT_ROOT / "predict_manifest_index.json").write_text(json.dumps(predict_rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"train_manifest_count": len(train_rows), "predict_manifest_count": len(predict_rows)}, indent=2))


if __name__ == "__main__":
    main()
