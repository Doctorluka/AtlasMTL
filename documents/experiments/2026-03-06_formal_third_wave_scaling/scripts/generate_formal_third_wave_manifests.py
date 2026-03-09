#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_TMP_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-config",
        default=str(
            REPO_ROOT
            / "documents"
            / "experiments"
            / "2026-03-06_formal_third_wave_scaling"
            / "configs"
            / "datasets.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT
            / "documents"
            / "experiments"
            / "2026-03-06_formal_third_wave_scaling"
            / "manifests"
            / "reference_heldout"
        ),
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"yaml root must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_list_override(dataset: Dict[str, Any], defaults: Dict[str, Any], key: str) -> List[int]:
    if key in dataset:
        value = dataset[key]
    else:
        value = defaults.get(key, [])
    return [int(x) for x in (value or [])]


def _build_common_method_configs(*, label_column: str, domain_key: str, include_scanvi: bool, gpu_manifest: bool) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "celltypist": {
            "target_label_column": label_column,
            "trainer_backend": "formal",
            "trainer_config": {
                "n_jobs": 8,
                "feature_selection": True,
                "balance_cell_type": True,
                "batch_size": 5000,
                "top_genes": 500,
                "use_gpu": False,
            },
        },
        "singler": {
            "target_label_column": label_column,
            "reference_layer": "counts",
            "query_layer": "counts",
            "normalize_log1p": True,
            "use_pruned_labels": True,
        },
        "symphony": {
            "target_label_column": label_column,
            "batch_key": domain_key,
            "reference_layer": "counts",
            "query_layer": "counts",
        },
        "seurat_anchor_transfer": {
            "target_label_column": label_column,
            "batch_key": domain_key,
            "reference_layer": "counts",
            "query_layer": "counts",
            "nfeatures": 3000,
        },
        "reference_knn": {
            "input_transform": "binary",
        },
    }
    if include_scanvi:
        cfg["scanvi"] = {
            "target_label_column": label_column,
            "batch_key": domain_key,
            "counts_layer": "counts",
            "scvi_max_epochs": 25,
            "scanvi_max_epochs": 25,
            "query_max_epochs": 20,
            "n_latent": 20,
            "batch_size": 256,
            "datasplitter_num_workers": 0,
            "use_gpu": gpu_manifest,
        }
    return cfg


def _atlas_train_config(*, device: str) -> Dict[str, Any]:
    if device == "cpu":
        return {
            "hidden_sizes": [256, 128],
            "dropout_rate": 0.2,
            "batch_size": 128,
            "num_epochs": 50,
            "learning_rate": 0.0003,
            "input_transform": "binary",
            "val_fraction": 0.1,
            "early_stopping_patience": 5,
            "random_state": 2026,
            "reference_storage": "external",
        }
    return {
        "hidden_sizes": [1024, 512],
        "dropout_rate": 0.2,
        "batch_size": 512,
        "num_epochs": 50,
        "learning_rate": 0.001,
        "input_transform": "binary",
        "val_fraction": 0.1,
        "early_stopping_patience": 5,
        "random_state": 2026,
        "reference_storage": "external",
    }


def _atlas_predict_config(*, device: str) -> Dict[str, Any]:
    return {
        "knn_correction": "off",
        "batch_size": 128 if device == "cpu" else 512,
        "input_transform": "binary",
    }


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = _load_yaml(Path(args.dataset_config).resolve())
    defaults = dict(config.get("defaults") or {})
    datasets = list(config.get("datasets") or [])
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_index: List[Dict[str, Any]] = []

    for dataset in datasets:
        dataset_name = str(dataset["dataset_name"])
        label_column = str(dataset["target_label"])
        domain_key = str(dataset["domain_key"])
        prep_manifest = _load_yaml(Path(str(dataset["prep_manifest"])).resolve())
        ceiling_path = ROUND_TMP_ROOT / dataset_name / "prepared" / "formal_split_v1" / "dataset_ceiling_summary.json"
        ceiling = _load_json(ceiling_path)
        build_sizes = list(ceiling["build_grid_feasible"])
        predict_sizes = list(ceiling["predict_grid_feasible"])
        fixed_build_size = int(ceiling["fixed_build_size_for_predict_scaling"])
        panel_type = str(ceiling["panel_type"])

        for device in ("cpu", "gpu"):
            include_scanvi = device == "gpu"
            method_configs = _build_common_method_configs(
                label_column=label_column,
                domain_key=domain_key,
                include_scanvi=include_scanvi,
                gpu_manifest=device == "gpu",
            )
            train_cfg = _atlas_train_config(device=device)
            predict_cfg = _atlas_predict_config(device=device)

            for build_size in build_sizes:
                ref_path = ROUND_TMP_ROOT / dataset_name / "prepared" / "formal_split_v1" / "build_scaling" / f"build_{build_size}" / f"reference_train_{build_size}.h5ad"
                query_path = ROUND_TMP_ROOT / dataset_name / "prepared" / "formal_split_v1" / "build_scaling" / f"build_{build_size}" / "heldout_build_eval_10k.h5ad"
                split_name = f"formal_build_scaling_{device}_build{build_size}_eval10k_v1"
                manifest_name = f"{dataset_name}__{label_column}__{split_name}.yaml"
                payload = {
                    "dataset_name": dataset_name,
                    "version": 1,
                    "protocol_version": 1,
                    "random_seed": int(defaults.get("seed", 2026)),
                    "split_name": split_name,
                    "split_description": f"formal third-wave build scaling manifest ({device}); build={build_size}, fixed query=10k, panel={panel_type}",
                    "reference_subset": f"reference_train_{build_size}_prepared",
                    "query_subset": "heldout_build_eval_10k_prepared",
                    "reference_h5ad": str(ref_path),
                    "query_h5ad": str(query_path),
                    "label_columns": [label_column],
                    "domain_key": domain_key,
                    "input_matrix_type": prep_manifest.get("input_matrix_type", "infer"),
                    "counts_layer": prep_manifest.get("counts_layer", "counts"),
                    "method_configs": method_configs,
                    "train": train_cfg,
                    "predict": predict_cfg,
                }
                out_path = output_dir / manifest_name
                _write_manifest(out_path, payload)
                manifest_index.append(
                    {
                        "dataset_name": dataset_name,
                        "panel_type": panel_type,
                        "device_group": device,
                        "track": "build_scaling",
                        "build_size": int(build_size),
                        "predict_size": 10000,
                        "manifest_path": str(out_path),
                    }
                )

            for predict_size in predict_sizes:
                ref_path = (
                    ROUND_TMP_ROOT
                    / dataset_name
                    / "prepared"
                    / "formal_split_v1"
                    / "build_scaling"
                    / f"build_{fixed_build_size}"
                    / f"reference_train_{fixed_build_size}.h5ad"
                )
                query_path = (
                    ROUND_TMP_ROOT
                    / dataset_name
                    / "prepared"
                    / "formal_split_v1"
                    / "predict_scaling"
                    / f"fixed_build_{fixed_build_size}"
                    / f"heldout_predict_{predict_size}.h5ad"
                )
                split_name = f"formal_predict_scaling_{device}_build{fixed_build_size}_predict{predict_size}_v1"
                manifest_name = f"{dataset_name}__{label_column}__{split_name}.yaml"
                payload = {
                    "dataset_name": dataset_name,
                    "version": 1,
                    "protocol_version": 1,
                    "random_seed": int(defaults.get("seed", 2026)),
                    "split_name": split_name,
                    "split_description": f"formal third-wave predict scaling manifest ({device}); fixed build={fixed_build_size}, query={predict_size}, panel={panel_type}",
                    "reference_subset": f"reference_train_{fixed_build_size}_prepared",
                    "query_subset": f"heldout_predict_{predict_size}_prepared",
                    "reference_h5ad": str(ref_path),
                    "query_h5ad": str(query_path),
                    "label_columns": [label_column],
                    "domain_key": domain_key,
                    "input_matrix_type": prep_manifest.get("input_matrix_type", "infer"),
                    "counts_layer": prep_manifest.get("counts_layer", "counts"),
                    "method_configs": method_configs,
                    "train": train_cfg,
                    "predict": predict_cfg,
                }
                out_path = output_dir / manifest_name
                _write_manifest(out_path, payload)
                manifest_index.append(
                    {
                        "dataset_name": dataset_name,
                        "panel_type": panel_type,
                        "device_group": device,
                        "track": "predict_scaling",
                        "build_size": int(fixed_build_size),
                        "predict_size": int(predict_size),
                        "manifest_path": str(out_path),
                    }
                )

    index_path = output_dir / "manifest_index.json"
    index_path.write_text(json.dumps(manifest_index, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest_count": len(manifest_index), "index_path": str(index_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
