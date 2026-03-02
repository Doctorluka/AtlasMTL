#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import anndata as ad
import yaml

from ablation_common import (
    PHMAP_TASK_WEIGHTS,
    load_manifest,
    parse_feature_mode,
    prepare_manifest,
    resolve_data_path,
    resolve_devices,
    run_atlasmtl_variant,
    run_cuda_gate,
    write_standard_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate AtlasMTL KNN geometry: predicted coord head vs internal latent.")
    parser.add_argument("--dataset-manifest", required=True, help="Base manifest used to derive the no-obsm query.")
    parser.add_argument("--coorddiag-manifest", required=True, help="Manifest used for coordinate regression diagnostics.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
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


def _geometry_modes() -> List[Tuple[str, Dict[str, Any]]]:
    return [
        (
            "predicted_scanvi_head",
            {
                "coord_targets": {"scanvi": "X_scANVI"},
                "knn_space": "scanvi",
            },
        ),
        (
            "latent_internal",
            {
                "coord_targets": None,
                "knn_space": "latent_internal",
            },
        ),
    ]


def _variant_name(
    device: str,
    feature_mode: str,
    input_transform: str,
    geometry_mode: str,
    knn_variant: str,
) -> str:
    return f"atlasmtl_{device}_{feature_mode}_{input_transform}_{geometry_mode}_{knn_variant}"


def _iter_grid(
    devices: Iterable[str],
    feature_mode: str,
    input_transform: str,
) -> Iterable[Dict[str, Any]]:
    for device in devices:
        for geometry_mode, geom_cfg in _geometry_modes():
            for knn_variant, knn_cfg in _knn_variants():
                yield {
                    "device": device,
                    "feature_mode": feature_mode,
                    "input_transform": input_transform,
                    "geometry_mode": geometry_mode,
                    "geom_cfg": geom_cfg,
                    "knn_variant": knn_variant,
                    "knn_cfg": knn_cfg,
                    "variant_name": _variant_name(
                        device=device,
                        feature_mode=feature_mode,
                        input_transform=input_transform,
                        geometry_mode=geometry_mode,
                        knn_variant=knn_variant,
                    ),
                }


def derive_no_obsm_query(*, query_source: Path, out_path: Path) -> None:
    adata = ad.read_h5ad(query_source)
    adata.obsm.clear()
    adata.obsp.clear()
    adata.varm.clear()
    adata.varp.clear()
    adata.uns["atlasmtl_geometry_eval"] = {
        "note": "Derived from query source by stripping obsm/obsp to simulate real deployment query without embeddings.",
        "source": str(query_source),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_manifest_dir = output_dir / "generated_manifests"
    generated_manifest_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    derived_dir = output_dir / "derived_inputs"
    derived_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest_path = Path(args.dataset_manifest).resolve()
    base_manifest = load_manifest(dataset_manifest_path)

    query_source = resolve_data_path(str(base_manifest["query_h5ad"]), manifest_path=dataset_manifest_path)
    derived_query = derived_dir / "query_3k_noobsm.h5ad"
    derive_no_obsm_query(query_source=query_source, out_path=derived_query)

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
        manifest["query_h5ad"] = str(derived_query)

        geom_cfg = dict(combo["geom_cfg"])
        if geom_cfg.get("coord_targets"):
            manifest["coord_targets"] = dict(geom_cfg["coord_targets"])
        else:
            manifest.pop("coord_targets", None)

        manifest.setdefault("predict", {})
        manifest["predict"].update(
            {
                **dict(manifest["predict"]),
                **dict(combo["knn_cfg"]),
                "knn_k": int(args.knn_k),
                "knn_conf_low": float(args.knn_conf_low),
                "knn_space": str(geom_cfg["knn_space"]),
            }
        )
        manifest["predict"].pop("knn_query_obsm_key", None)

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
            "geometry_mode": combo["geometry_mode"],
            "knn_variant": combo["knn_variant"],
            "knn_correction": combo["knn_cfg"]["knn_correction"],
            "knn_k": int(args.knn_k),
            "knn_conf_low": float(args.knn_conf_low),
            "query_obsm_available": False,
            "query_obsm_policy": "stripped_all_obsm_obsp",
            "search_family": "knn_geometry_eval",
            "run_dir": str(run_dir),
        }
        results.append(result)

    # Coordinate diagnostic run (A only) on the original query with obsm targets,
    # still without using knn_query_obsm_key.
    coorddiag_manifest_path = Path(args.coorddiag_manifest).resolve()
    coorddiag_manifest_base = load_manifest(coorddiag_manifest_path)
    coorddiag_out_dir = output_dir / "coorddiag"
    coorddiag_out_dir.mkdir(parents=True, exist_ok=True)
    coorddiag_manifest = prepare_manifest(
        base_manifest=coorddiag_manifest_base,
        dataset_manifest_path=coorddiag_manifest_path,
        feature_mode=args.feature_mode,
        input_transform=args.input_transform,
        task_weights=list(args.task_weights),
    )
    coorddiag_manifest.setdefault("predict", {})
    coorddiag_manifest["predict"].update(
        {
            **dict(coorddiag_manifest["predict"]),
            "knn_correction": "off",
            "knn_k": int(args.knn_k),
            "knn_conf_low": float(args.knn_conf_low),
            "knn_space": "scanvi",
        }
    )
    coorddiag_manifest["predict"].pop("knn_query_obsm_key", None)
    coorddiag_generated = generated_manifest_dir / "atlasmtl_coorddiag_predicted_scanvi_head_knn_off.yaml"
    coorddiag_generated.write_text(yaml.safe_dump(coorddiag_manifest, sort_keys=False), encoding="utf-8")
    coorddiag_result = run_atlasmtl_variant(
        manifest_path=coorddiag_generated,
        output_dir=coorddiag_out_dir,
        device=devices[0] if devices else "cpu",
    )
    coorddiag_result["variant_name"] = "atlasmtl_coorddiag_predicted_scanvi_head_knn_off"
    coorddiag_result["ablation_config"] = {
        **dict(coorddiag_result.get("ablation_config") or {}),
        "variant_name": coorddiag_result["variant_name"],
        "geometry_mode": "predicted_scanvi_head",
        "knn_variant": "knn_off",
        "knn_correction": "off",
        "query_obsm_available": True,
        "query_obsm_policy": "kept_for_coordinate_metrics_only",
        "search_family": "knn_geometry_eval",
        "run_dir": str(coorddiag_out_dir),
    }
    results.append(coorddiag_result)

    metrics_path = output_dir / "metrics.json"
    payload = {
        "protocol_version": base_manifest.get("protocol_version", 1),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_name": base_manifest.get("dataset_name"),
        "dataset_version": base_manifest.get("version"),
        "search_family": "knn_geometry_eval",
        "cuda_gate": gate,
        "derived_inputs": {"query_no_obsm_h5ad": str(derived_query)},
        "results": results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_standard_outputs(metrics_path=metrics_path, target_label_column=args.target_label_column)
    print(metrics_path)


if __name__ == "__main__":
    main()
