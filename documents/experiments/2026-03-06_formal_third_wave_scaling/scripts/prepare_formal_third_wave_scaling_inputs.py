#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anndata as ad
import numpy as np
import yaml

from atlasmtl.preprocess import PreprocessConfig, ensure_counts_layer, save_feature_panel
from atlasmtl.preprocess.features import align_query_to_feature_panel, select_reference_features
from atlasmtl.preprocess.gene_ids import canonicalize_gene_ids
from atlasmtl.preprocess.metadata import attach_preprocess_metadata
from atlasmtl.preprocess.split import make_group_split_plan, materialize_group_split_subsets
from atlasmtl.preprocess.types import PreprocessReport
from atlasmtl.utils.monitoring import SampledResourceTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--output-root", default="/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-candidates", type=int)
    parser.add_argument("--warning-build-label-min", type=int)
    parser.add_argument("--warning-predict-label-min", type=int)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"yaml root must be a mapping: {path}")
    return payload


def _build_preprocess_config(manifest: Dict[str, Any]) -> PreprocessConfig:
    hvg_cfg = dict(manifest.get("hvg_config") or {})
    return PreprocessConfig(
        var_names_type=str(manifest["var_names_type"]),
        species=str(manifest["species"]),
        gene_id_table=manifest.get("gene_id_table"),
        canonical_target=str(manifest.get("canonical_target", "ensembl")),
        ensembl_source_column=manifest.get("ensembl_source_column"),
        symbol_source_column=manifest.get("symbol_source_column"),
        prefer_existing_ensembl_column=bool(manifest.get("prefer_existing_ensembl_column", True)),
        mapping_table_kind=str(manifest.get("mapping_table_kind", "biomart_human_mouse_rat")),
        input_matrix_type=str(manifest.get("input_matrix_type", "infer")),
        counts_layer=str(manifest.get("counts_layer", "counts")),
        counts_required=True,
        counts_check_tiny_positive_tol=float(manifest.get("counts_check_tiny_positive_tol", 1e-8)),
        counts_confirm_fraction=float(manifest.get("counts_confirm_fraction", 0.999)),
        feature_space=str(manifest.get("feature_space", "hvg")),
        n_top_genes=int(hvg_cfg.get("n_top_genes", 3000)),
        hvg_method=str(hvg_cfg.get("method", "seurat_v3")),
        hvg_batch_key=hvg_cfg.get("batch_key"),
        duplicate_policy=str(manifest.get("duplicate_policy", "sum")),
        unmapped_policy=str(manifest.get("unmapped_policy", "drop")),
    )


def _compose_preprocess_report(
    *,
    config: PreprocessConfig,
    canonical_report: PreprocessReport,
    counts_meta: Dict[str, Any],
    feature_report: PreprocessReport,
    matched_feature_genes: int | None = None,
    missing_feature_genes: int | None = None,
) -> PreprocessReport:
    return PreprocessReport(
        n_input_genes=canonical_report.n_input_genes,
        n_canonical_genes=canonical_report.n_canonical_genes,
        n_duplicate_genes=canonical_report.n_duplicate_genes,
        n_unmapped_genes=canonical_report.n_unmapped_genes,
        n_features_selected=feature_report.n_features_selected,
        feature_space=config.feature_space,
        species=config.species,
        var_names_type=config.var_names_type,
        mapping_resource=canonical_report.mapping_resource,
        duplicate_policy=config.duplicate_policy,
        unmapped_policy=config.unmapped_policy,
        input_matrix_type_declared=config.input_matrix_type,
        input_matrix_type_detected=str(counts_meta["input_matrix_type_detected"]),
        counts_available=bool(counts_meta["counts_available"]),
        counts_layer_used=counts_meta["counts_layer_used"],
        counts_check_passed=bool(counts_meta["counts_check_passed"]),
        counts_decision=str(counts_meta["counts_decision"]),
        counts_detection_summary=dict(counts_meta["counts_detection_summary"]),
        counts_source_original=str(counts_meta["counts_source_original"]),
        counts_layer_materialized=bool(counts_meta["counts_layer_materialized"]),
        hvg_layer_used=feature_report.hvg_layer_used,
        matched_feature_genes=matched_feature_genes,
        missing_feature_genes=missing_feature_genes,
        ensembl_versions_stripped=canonical_report.ensembl_versions_stripped,
        gene_id_case=canonical_report.gene_id_case,
        canonical_source_column=canonical_report.canonical_source_column,
        n_genes_dropped_by_unmapped_policy=canonical_report.n_genes_dropped_by_unmapped_policy,
        n_duplicate_groups=canonical_report.n_duplicate_groups,
        duplicate_resolution_applied=canonical_report.duplicate_resolution_applied,
        preprocess_warnings=list(canonical_report.preprocess_warnings),
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_h5ad(adata: ad.AnnData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def _log(message: str) -> None:
    print(message, flush=True)


def _nested_index_sets(n_obs: int, sizes: List[int], seed: int) -> Dict[int, np.ndarray]:
    if not sizes:
        return {}
    max_size = max(sizes)
    if n_obs < max_size:
        raise ValueError(f"requested nested max size {max_size} exceeds available cells {n_obs}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_obs)
    output: Dict[int, np.ndarray] = {}
    for size in sorted(sizes):
        output[int(size)] = np.sort(perm[:size])
    return output


def _complement_indices(total_n: int, selected: np.ndarray) -> np.ndarray:
    mask = np.ones(total_n, dtype=bool)
    mask[selected] = False
    return np.flatnonzero(mask)


def _subset_adata(adata: ad.AnnData, indices: np.ndarray) -> ad.AnnData:
    return adata[np.sort(indices)].copy()


def _resolve_dataset(config: Dict[str, Any], dataset_name: str | None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    defaults = dict(config.get("defaults") or {})
    datasets = list(config.get("datasets") or [])
    if not datasets:
        raise ValueError("no datasets in config")
    if dataset_name is None:
        if len(datasets) != 1:
            raise ValueError("dataset-name is required when config contains multiple datasets")
        return defaults, dict(datasets[0])
    for row in datasets:
        if str(row.get("dataset_name")) == dataset_name:
            return defaults, dict(row)
    raise ValueError(f"dataset not found in config: {dataset_name}")


def _resolve_list_override(dataset: Dict[str, Any], defaults: Dict[str, Any], key: str) -> List[int]:
    if key in dataset:
        value = dataset[key]
    else:
        value = defaults.get(key, [])
    return [int(x) for x in (value or [])]


def _pick_split_plan(
    canonical_adata: ad.AnnData,
    *,
    split_key: str,
    target_label: str,
    build_grid: List[int],
    build_eval_size: int,
    predict_grid: List[int],
    predict_tail_optional: List[int],
    seed: int,
    n_candidates: int,
    warning_build_label_min: int,
    warning_predict_label_min: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    predict_grid = sorted({int(x) for x in predict_grid})
    predict_tail_optional = sorted({int(x) for x in predict_tail_optional})
    core_predict_max = max(predict_grid)
    required_heldout_targets = []
    if predict_tail_optional:
        required_heldout_targets.append(build_eval_size + max(predict_tail_optional))
    required_heldout_targets.append(build_eval_size + core_predict_max)

    attempts = []
    for heldout_total in required_heldout_targets:
        for build_size in sorted({int(x) for x in build_grid}, reverse=True):
            try:
                plan = make_group_split_plan(
                    canonical_adata,
                    split_key=split_key,
                    target_label=target_label,
                    build_size=build_size,
                    predict_size=heldout_total,
                    seed=seed,
                    n_candidates=n_candidates,
                    warning_build_label_min=warning_build_label_min,
                    warning_predict_label_min=warning_predict_label_min,
                )
            except Exception as exc:
                attempts.append({"heldout_total": heldout_total, "build_size": build_size, "status": "failed", "error": str(exc)})
                continue
            attempts.append({"heldout_total": heldout_total, "build_size": build_size, "status": "ok"})
            optional_tail_enabled = bool(predict_tail_optional and heldout_total >= build_eval_size + max(predict_tail_optional))
            return plan, {
                "selected_build_max": int(build_size),
                "selected_heldout_total": int(heldout_total),
                "core_predict_max": int(core_predict_max),
                "optional_tail_enabled": bool(optional_tail_enabled),
                "predict_tail_max": int(max(predict_tail_optional)) if optional_tail_enabled else None,
                "attempts": attempts,
            }
    raise ValueError(f"unable to construct formal split plan: {json.dumps(attempts, indent=2)}")


def _prepare_build_tiers(
    *,
    build_pool_canonical: ad.AnnData,
    build_eval_canonical: ad.AnnData,
    build_sizes: List[int],
    preprocess_config: PreprocessConfig,
    canonical_report: PreprocessReport,
    counts_meta: Dict[str, Any],
    out_root: Path,
    feature_panels: Dict[int, Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    build_sizes = sorted({int(x) for x in build_sizes})
    nested_indices = _nested_index_sets(build_pool_canonical.n_obs, build_sizes, seed=seed)
    summaries = []
    for size in build_sizes:
        tier_dir = out_root / f"build_scaling/build_{size}"
        reference_canonical = _subset_adata(build_pool_canonical, nested_indices[size])
        reference_prepared, feature_panel, reference_feature_report = select_reference_features(reference_canonical, preprocess_config)
        build_eval_prepared, build_eval_align_report = align_query_to_feature_panel(build_eval_canonical, feature_panel, preprocess_config)

        reference_report = _compose_preprocess_report(
            config=preprocess_config,
            canonical_report=canonical_report,
            counts_meta=counts_meta,
            feature_report=reference_feature_report,
        )
        build_eval_report = _compose_preprocess_report(
            config=preprocess_config,
            canonical_report=canonical_report,
            counts_meta=counts_meta,
            feature_report=build_eval_align_report,
            matched_feature_genes=build_eval_align_report.matched_feature_genes,
            missing_feature_genes=build_eval_align_report.missing_feature_genes,
        )
        attach_preprocess_metadata(reference_prepared, config=preprocess_config, report=reference_report, feature_panel=feature_panel)
        attach_preprocess_metadata(build_eval_prepared, config=preprocess_config, report=build_eval_report, feature_panel=feature_panel)

        _write_h5ad(reference_prepared, tier_dir / f"reference_train_{size}.h5ad")
        _write_h5ad(build_eval_prepared, tier_dir / "heldout_build_eval_10k.h5ad")
        save_feature_panel(feature_panel, str(tier_dir / "feature_panel.json"))
        _write_json(
            tier_dir / "preprocessing_summary.json",
            {
                "build_size": int(size),
                "reference_report": reference_report.to_dict(),
                "build_eval_report": build_eval_report.to_dict(),
                "reference_n_obs": int(reference_prepared.n_obs),
                "build_eval_n_obs": int(build_eval_prepared.n_obs),
            },
        )
        feature_panels[int(size)] = {
            "feature_panel": feature_panel,
            "reference_path": str((tier_dir / f"reference_train_{size}.h5ad").resolve()),
            "feature_panel_path": str((tier_dir / "feature_panel.json").resolve()),
        }
        summaries.append(
            {
                "build_size": int(size),
                "reference_path": str((tier_dir / f"reference_train_{size}.h5ad").resolve()),
                "build_eval_path": str((tier_dir / "heldout_build_eval_10k.h5ad").resolve()),
                "feature_panel_path": str((tier_dir / "feature_panel.json").resolve()),
            }
        )
    return {"build_tiers": summaries}


def _prepare_predict_tiers(
    *,
    predict_pool_canonical: ad.AnnData,
    predict_sizes: List[int],
    fixed_build_size: int,
    preprocess_config: PreprocessConfig,
    canonical_report: PreprocessReport,
    counts_meta: Dict[str, Any],
    out_root: Path,
    fixed_feature_panel: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    predict_sizes = sorted({int(x) for x in predict_sizes})
    nested_indices = _nested_index_sets(predict_pool_canonical.n_obs, predict_sizes, seed=seed)
    tier_dir = out_root / f"predict_scaling/fixed_build_{fixed_build_size}"
    feature_panel = fixed_feature_panel["feature_panel"]
    summaries = []
    for size in predict_sizes:
        query_canonical = _subset_adata(predict_pool_canonical, nested_indices[size])
        query_prepared, query_align_report = align_query_to_feature_panel(query_canonical, feature_panel, preprocess_config)
        query_report = _compose_preprocess_report(
            config=preprocess_config,
            canonical_report=canonical_report,
            counts_meta=counts_meta,
            feature_report=query_align_report,
            matched_feature_genes=query_align_report.matched_feature_genes,
            missing_feature_genes=query_align_report.missing_feature_genes,
        )
        attach_preprocess_metadata(query_prepared, config=preprocess_config, report=query_report, feature_panel=feature_panel)
        out_path = tier_dir / f"heldout_predict_{size}.h5ad"
        _write_h5ad(query_prepared, out_path)
        summaries.append(
            {
                "predict_size": int(size),
                "query_path": str(out_path.resolve()),
                "query_report": query_report.to_dict(),
            }
        )
    _write_json(
        tier_dir / "preprocessing_summary.json",
        {
            "fixed_build_size": int(fixed_build_size),
            "fixed_reference_path": fixed_feature_panel["reference_path"],
            "feature_panel_path": fixed_feature_panel["feature_panel_path"],
            "predict_tiers": summaries,
        },
    )
    return {"predict_tiers": summaries}


def main() -> None:
    args = parse_args()
    config_payload = _load_yaml(Path(args.dataset_config).resolve())
    defaults, dataset = _resolve_dataset(config_payload, args.dataset_name)

    seed = int(args.seed if args.seed is not None else defaults.get("seed", 2026))
    n_candidates = int(args.n_candidates if args.n_candidates is not None else defaults.get("n_candidates", 128))
    warning_build_label_min = int(
        args.warning_build_label_min if args.warning_build_label_min is not None else defaults.get("warning_build_label_min", 25)
    )
    warning_predict_label_min = int(
        args.warning_predict_label_min if args.warning_predict_label_min is not None else defaults.get("warning_predict_label_min", 10)
    )
    build_eval_size = int(defaults.get("build_eval_size", 10000))
    build_grid = _resolve_list_override(dataset, defaults, "build_grid")
    predict_grid = _resolve_list_override(dataset, defaults, "predict_grid")
    predict_tail_optional = _resolve_list_override(dataset, defaults, "predict_tail_optional")

    dataset_name = str(dataset["dataset_name"])
    split_key = str(dataset["split_key"])
    domain_key = str(dataset["domain_key"])
    target_label = str(dataset["target_label"])
    panel_type = str(dataset.get("panel_type", "main"))
    source_h5ad = Path(str(dataset["source_h5ad"])).expanduser().resolve()
    prep_manifest = _load_yaml(Path(str(dataset["prep_manifest"])).expanduser().resolve())
    preprocess_config = _build_preprocess_config(prep_manifest)

    out_root = Path(args.output_root).expanduser().resolve() / dataset_name / "prepared" / "formal_split_v1"
    out_root.mkdir(parents=True, exist_ok=True)
    _log(f"[{dataset_name}] output_root={out_root}")

    tracker = SampledResourceTracker(device="cpu")
    tracker.start()
    phase_seconds: Dict[str, float] = {}

    _log(f"[{dataset_name}] load source h5ad")
    t0 = time.perf_counter()
    raw_adata = ad.read_h5ad(str(source_h5ad))
    phase_seconds["load_h5ad_seconds"] = round(time.perf_counter() - t0, 4)

    _log(f"[{dataset_name}] ensure counts and canonicalize gene ids")
    t0 = time.perf_counter()
    with_counts, counts_meta = ensure_counts_layer(raw_adata, preprocess_config)
    canonical_adata, canonical_report = canonicalize_gene_ids(with_counts, preprocess_config)
    phase_seconds["counts_and_canonicalize_seconds"] = round(time.perf_counter() - t0, 4)

    _log(f"[{dataset_name}] search feasible split plan")
    t0 = time.perf_counter()
    split_plan, split_meta = _pick_split_plan(
        canonical_adata,
        split_key=split_key,
        target_label=target_label,
        build_grid=build_grid,
        build_eval_size=build_eval_size,
        predict_grid=predict_grid,
        predict_tail_optional=predict_tail_optional,
        seed=seed,
        n_candidates=n_candidates,
        warning_build_label_min=warning_build_label_min,
        warning_predict_label_min=warning_predict_label_min,
    )
    materialized = materialize_group_split_subsets(
        canonical_adata,
        split_plan,
        build_size=split_meta["selected_build_max"],
        predict_size=split_meta["selected_heldout_total"],
        seed=seed,
        warning_build_label_min=warning_build_label_min,
        warning_predict_label_min=warning_predict_label_min,
    )
    build_pool_canonical = materialized["reference_build_adata"]
    heldout_total_canonical = materialized["predict_adata"]
    phase_seconds["split_materialization_seconds"] = round(time.perf_counter() - t0, 4)

    _log(f"[{dataset_name}] partition heldout into build-eval and predict pool")
    t0 = time.perf_counter()
    build_eval_idx = _nested_index_sets(heldout_total_canonical.n_obs, [build_eval_size], seed + 17)[build_eval_size]
    predict_pool_idx = _complement_indices(heldout_total_canonical.n_obs, build_eval_idx)
    build_eval_canonical = _subset_adata(heldout_total_canonical, build_eval_idx)
    predict_pool_canonical = _subset_adata(heldout_total_canonical, predict_pool_idx)
    phase_seconds["heldout_partition_seconds"] = round(time.perf_counter() - t0, 4)

    feasible_build_sizes = [size for size in sorted(build_grid) if size <= build_pool_canonical.n_obs]
    feasible_predict_sizes = [size for size in sorted(predict_grid) if size <= predict_pool_canonical.n_obs]
    if split_meta["optional_tail_enabled"] and split_meta["predict_tail_max"] is not None:
        optional_tail_sizes = [size for size in predict_tail_optional if size <= predict_pool_canonical.n_obs]
        feasible_predict_sizes.extend(optional_tail_sizes)
    feasible_predict_sizes = sorted(set(feasible_predict_sizes))
    if panel_type == "supplementary" and 100000 not in feasible_build_sizes:
        fixed_build_size = max(feasible_build_sizes)
    elif 100000 in feasible_build_sizes:
        fixed_build_size = 100000
    else:
        fixed_build_size = max(feasible_build_sizes)

    split_summary = dict(materialized["split_summary"])
    split_summary.update(
        {
            "dataset_name": dataset_name,
            "panel_type": panel_type,
            "domain_key": domain_key,
            "source_h5ad": str(source_h5ad),
            "build_eval_size": build_eval_size,
            "predict_pool_cells": int(predict_pool_canonical.n_obs),
            "build_eval_cells": int(build_eval_canonical.n_obs),
            "predict_scaling_10k_distinct_from_build_eval_10k": True,
        }
    )
    _write_json(out_root / "split_plan.json", {**split_plan, **split_meta, "dataset_name": dataset_name, "panel_type": panel_type})
    _write_json(out_root / "split_summary.json", split_summary)

    _log(f"[{dataset_name}] materialize build tiers and fixed-query assets")
    t0 = time.perf_counter()
    feature_panels: Dict[int, Dict[str, Any]] = {}
    build_summary = _prepare_build_tiers(
        build_pool_canonical=build_pool_canonical,
        build_eval_canonical=build_eval_canonical,
        build_sizes=feasible_build_sizes,
        preprocess_config=preprocess_config,
        canonical_report=canonical_report,
        counts_meta=counts_meta,
        out_root=out_root,
        feature_panels=feature_panels,
        seed=seed + 31,
    )
    predict_summary = _prepare_predict_tiers(
        predict_pool_canonical=predict_pool_canonical,
        predict_sizes=feasible_predict_sizes,
        fixed_build_size=fixed_build_size,
        preprocess_config=preprocess_config,
        canonical_report=canonical_report,
        counts_meta=counts_meta,
        out_root=out_root,
        fixed_feature_panel=feature_panels[fixed_build_size],
        seed=seed + 53,
    )
    phase_seconds["tier_preprocessing_seconds"] = round(time.perf_counter() - t0, 4)

    _log(f"[{dataset_name}] write canonical subsets and summaries")
    _write_h5ad(build_eval_canonical, out_root / "canonical_subsets" / "heldout_build_eval_10k_canonical.h5ad")
    _write_h5ad(predict_pool_canonical, out_root / "canonical_subsets" / f"heldout_predict_pool_{predict_pool_canonical.n_obs}_canonical.h5ad")
    _write_h5ad(build_pool_canonical, out_root / "canonical_subsets" / f"reference_build_pool_{build_pool_canonical.n_obs}_canonical.h5ad")

    preprocessing_summary = {
        "dataset_name": dataset_name,
        "panel_type": panel_type,
        "prep_manifest": str(Path(str(dataset["prep_manifest"])).resolve()),
        "source_h5ad": str(source_h5ad),
        "preprocess_config": asdict(preprocess_config),
        "counts_meta": counts_meta,
        "canonical_report": canonical_report.to_dict(),
        "build_summary": build_summary,
        "predict_summary": predict_summary,
    }
    _write_json(out_root / "preprocessing_summary.json", preprocessing_summary)

    resource_snapshot = tracker.finish(
        phase="formal_prep",
        num_items=int(raw_adata.n_obs),
        num_threads_used=8,
    )
    total_elapsed = round(float(resource_snapshot.get("elapsed_seconds") or sum(phase_seconds.values())), 4)
    _write_json(
        out_root / "preparation_resource_summary.json",
        {
            "dataset_name": dataset_name,
            "device": "cpu",
            "phase_seconds": phase_seconds,
            "total_elapsed_seconds": total_elapsed,
            "average_rss_gb": resource_snapshot.get("process_avg_rss_gb"),
            "peak_rss_gb": resource_snapshot.get("process_peak_rss_gb"),
            "cpu_core_equiv_avg": resource_snapshot.get("cpu_core_equiv_avg"),
            "num_threads_used": resource_snapshot.get("num_threads_used"),
            "resource_tracker": resource_snapshot,
        },
    )

    _write_json(
        out_root / "dataset_ceiling_summary.json",
        {
            "dataset_name": dataset_name,
            "panel_type": panel_type,
            "build_grid_requested": build_grid,
            "build_grid_feasible": feasible_build_sizes,
            "predict_grid_requested": predict_grid,
            "predict_grid_feasible": feasible_predict_sizes,
            "predict_tail_optional_requested": predict_tail_optional,
            "optional_predict_tail_enabled": split_meta["optional_tail_enabled"],
            "selected_build_pool_size": int(build_pool_canonical.n_obs),
            "selected_heldout_total_size": int(heldout_total_canonical.n_obs),
            "build_eval_size": build_eval_size,
            "predict_pool_size": int(predict_pool_canonical.n_obs),
            "fixed_build_size_for_predict_scaling": int(fixed_build_size),
            "fixed_build_uses_ceiling_exception": bool(fixed_build_size != 100000 and panel_type == "main"),
        },
    )

    _log(f"[{dataset_name}] completed formal prep")
    print(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "panel_type": panel_type,
                "output_root": str(out_root),
                "build_grid_feasible": feasible_build_sizes,
                "predict_grid_feasible": feasible_predict_sizes,
                "fixed_build_size_for_predict_scaling": fixed_build_size,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
