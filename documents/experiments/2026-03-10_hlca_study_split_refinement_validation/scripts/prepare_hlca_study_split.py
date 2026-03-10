#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from atlasmtl.preprocess import (
    PreprocessConfig,
    ensure_counts_layer,
    make_group_split_plan,
    materialize_group_split_subsets,
    save_feature_panel,
)
from atlasmtl.preprocess.features import align_query_to_feature_panel, select_reference_features
from atlasmtl.preprocess.gene_ids import canonicalize_gene_ids
from atlasmtl.preprocess.metadata import attach_preprocess_metadata
from atlasmtl.preprocess.types import PreprocessReport
from atlasmtl.utils.monitoring import SampledResourceTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="documents/experiments/2026-03-10_hlca_study_split_refinement_validation/configs/hlca_study_split.yaml",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _write_adata(adata: ad.AnnData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def _build_preprocess_config(payload: Dict[str, Any]) -> PreprocessConfig:
    prep = dict(payload["preprocess"])
    hvg_cfg = dict(prep.get("hvg_config") or {})
    return PreprocessConfig(
        var_names_type=str(prep["var_names_type"]),
        species=str(prep["species"]),
        canonical_target=str(prep.get("canonical_target", "ensembl")),
        mapping_table_kind=str(prep.get("mapping_table_kind", "biomart_human_mouse_rat")),
        input_matrix_type=str(prep.get("input_matrix_type", "lognorm")),
        counts_layer=str(prep.get("counts_layer", "counts")),
        feature_space=str(prep.get("feature_space", "hvg")),
        n_top_genes=int(hvg_cfg.get("n_top_genes", 3000)),
        hvg_method=str(hvg_cfg.get("method", "seurat_v3")),
        duplicate_policy=str(prep.get("duplicate_policy", "sum")),
        unmapped_policy=str(prep.get("unmapped_policy", "drop")),
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


def _sample_subset(adata: ad.AnnData, *, size: int, seed: int) -> ad.AnnData:
    if adata.n_obs < size:
        raise ValueError(f"requested subset size {size} exceeds available cells {adata.n_obs}")
    if adata.n_obs == size:
        return adata.copy()
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(adata.n_obs, size=size, replace=False))
    return adata[chosen].copy()


def _sample_disjoint_heldout(
    heldout_total: ad.AnnData,
    *,
    build_eval_size: int,
    predict_pool_size: int,
    seed: int,
) -> Tuple[ad.AnnData, ad.AnnData]:
    if heldout_total.n_obs < build_eval_size + predict_pool_size:
        raise ValueError(
            "heldout_total is too small for disjoint build-eval and predict-pool subsets: "
            f"{heldout_total.n_obs} < {build_eval_size + predict_pool_size}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(heldout_total.n_obs)
    build_idx = np.sort(perm[:build_eval_size])
    predict_idx = np.sort(perm[build_eval_size : build_eval_size + predict_pool_size])
    return heldout_total[build_idx].copy(), heldout_total[predict_idx].copy()


def _write_label_coverage_summary(
    *,
    path: Path,
    target_label: str,
    dataset_total: ad.AnnData,
    build_pool: ad.AnnData,
    build_subset: ad.AnnData,
    heldout_total: ad.AnnData,
    build_eval: ad.AnnData,
    predict_pool: ad.AnnData,
    predict_subsets: Dict[int, ad.AnnData],
) -> None:
    rows: List[Dict[str, Any]] = []
    for split_name, adata in (
        ("dataset_total", dataset_total),
        ("build_pool", build_pool),
        ("build_subset_100000", build_subset),
        ("heldout_total", heldout_total),
        ("heldout_build_eval_10k", build_eval),
        ("heldout_predict_pool_50000", predict_pool),
    ):
        counts = adata.obs[target_label].astype(str).value_counts(dropna=False)
        for label, count in counts.items():
            rows.append(
                {
                    "split_name": split_name,
                    "label": str(label),
                    "count": int(count),
                }
            )
    for size, adata in predict_subsets.items():
        counts = adata.obs[target_label].astype(str).value_counts(dropna=False)
        for label, count in counts.items():
            rows.append(
                {
                    "split_name": f"heldout_predict_{size}",
                    "label": str(label),
                    "count": int(count),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_study_inventory(
    *,
    path: Path,
    dataset_total: ad.AnnData,
    build_groups: List[str],
    predict_groups: List[str],
) -> None:
    counts = dataset_total.obs["study"].astype(str).value_counts(dropna=False)
    rows: List[Dict[str, Any]] = []
    for study, count in counts.items():
        if study in build_groups:
            role = "build"
        elif study in predict_groups:
            role = "heldout"
        else:
            role = "unused"
        rows.append({"study": str(study), "n_cells": int(count), "assigned_role": role})
    pd.DataFrame(rows).sort_values(["assigned_role", "n_cells"], ascending=[True, False]).to_csv(path, index=False)


def _write_markdown_reports(
    *,
    repo_root: Path,
    config_path: Path,
    config: Dict[str, Any],
    split_plan: Dict[str, Any],
    split_summary: Dict[str, Any],
    prepared_root: Path,
) -> None:
    results_dir = repo_root / "documents" / "experiments" / "2026-03-10_hlca_study_split_refinement_validation" / "results_summary"
    results_dir.mkdir(parents=True, exist_ok=True)
    warning_lines = [f"  - `{value}`" for value in split_summary.get("warnings", [])]
    if not warning_lines:
        warning_lines = ["  - `none`"]
    report = [
        "# HLCA Study-Split Preparation Report",
        "",
        "- scenario: `HLCA_Core` study-grouped validation preparation",
        f"- source data: `{config['source_h5ad']}`",
        "- split key: `study`",
        "- output root:",
        f"  - `{prepared_root}`",
        "- chosen build groups:",
        *[f"  - `{value}`" for value in split_plan["build_groups"]],
        "- chosen heldout groups:",
        *[f"  - `{value}`" for value in split_plan["predict_groups"]],
        "- resulting prepared assets:",
        "  - `build_scaling/build_100000/reference_train_100000.h5ad`",
        "  - `build_scaling/build_100000/heldout_build_eval_10k.h5ad`",
        "  - `predict_scaling/fixed_build_100000/heldout_predict_10000.h5ad`",
        "  - `split_plan.json`",
        "  - `split_summary.json`",
        "  - `preprocessing_summary.json`",
        "  - `preparation_resource_summary.json`",
        "- key size outcome:",
        f"  - selected build pool: `{split_summary['build_subset_cells']}`",
        f"  - selected heldout total: `{split_summary['heldout_total_cells']}`",
        f"  - fixed build size for validation: `{config['build_size']}`",
        "- retained warnings:",
        *warning_lines,
        "",
        "Interpretation:",
        "",
        "- the stricter `study`-grouped validation path is feasible on HLCA",
        "- the split is intended to support dataset-specific weighting confirmation before reranker validation",
        "- this preparation should be treated as a new independent HLCA validation round",
        "",
    ]
    (results_dir / "study_split_preparation_report.md").write_text("\n".join(report), encoding="utf-8")

    record = [
        "# HLCA Study-Split Preparation Record",
        "",
        "- date: `2026-03-10`",
        "- stage: `preparation`",
        "- dataset: `HLCA_Core`",
        f"- source h5ad: `{config['source_h5ad']}`",
        f"- config: `{config_path}`",
        "- split key: `study`",
        "- domain key: `study`",
        "- target label: `ann_level_5`",
        f"- seed: `{config['seed']}`",
        f"- candidate count: `{config['n_candidates']}`",
        f"- selected build pool size: `{config['build_pool_size']}`",
        f"- selected heldout total size: `{config['heldout_total_size']}`",
        "- build groups:",
        *[f"  - `{value}`" for value in split_plan["build_groups"]],
        "- predict groups:",
        *[f"  - `{value}`" for value in split_plan["predict_groups"]],
        "- main outputs:",
        f"  - `{prepared_root / 'split_plan.json'}`",
        f"  - `{prepared_root / 'split_summary.json'}`",
        f"  - `{prepared_root / 'preprocessing_summary.json'}`",
        f"  - `{prepared_root / 'preparation_resource_summary.json'}`",
    ]
    if split_summary.get("warnings"):
        record.append("- warnings:")
        record.extend(f"  - `{value}`" for value in split_summary["warnings"])
    (results_dir / "study_split_preparation_record.md").write_text("\n".join(record) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    preprocess_config = _build_preprocess_config(config)
    source_h5ad = Path(config["source_h5ad"]).expanduser().resolve()
    prepared_root = Path(config["prepared_root"]).expanduser().resolve()

    tracker = SampledResourceTracker(device="cpu")
    tracker.start()
    phase_seconds: Dict[str, float] = {}

    t0 = time.perf_counter()
    raw_adata = ad.read_h5ad(str(source_h5ad))
    phase_seconds["load_h5ad_seconds"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    with_counts, full_counts_meta = ensure_counts_layer(raw_adata, preprocess_config)
    canonical_adata, canonical_report = canonicalize_gene_ids(with_counts, preprocess_config)
    phase_seconds["counts_and_canonicalize_seconds"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    split_plan = make_group_split_plan(
        canonical_adata,
        split_key=str(config["split_key"]),
        target_label=str(config["target_label"]),
        build_size=int(config["build_pool_size"]),
        predict_size=int(config["heldout_total_size"]),
        seed=int(config["seed"]),
        n_candidates=int(config["n_candidates"]),
        warning_build_label_min=int(config["warning_build_label_min"]),
        warning_predict_label_min=int(config["warning_predict_label_min"]),
    )
    split_plan["dataset_name"] = str(config["dataset_name"])
    split_plan["domain_key"] = str(config["domain_key"])
    split_plan["source_h5ad"] = str(source_h5ad)
    split_plan["build_size"] = int(config["build_size"])
    split_plan["build_pool_size"] = int(config["build_pool_size"])
    split_plan["heldout_total_size"] = int(config["heldout_total_size"])
    split_plan["build_eval_size"] = int(config["build_eval_size"])
    split_plan["predict_pool_size"] = int(config["predict_pool_size"])
    split_plan["predict_sizes"] = [int(value) for value in config["predict_sizes"]]

    materialized = materialize_group_split_subsets(
        canonical_adata,
        split_plan,
        build_size=int(config["build_pool_size"]),
        predict_size=int(config["heldout_total_size"]),
        seed=int(config["seed"]),
        warning_build_label_min=int(config["warning_build_label_min"]),
        warning_predict_label_min=int(config["warning_predict_label_min"]),
    )
    build_pool = materialized["reference_build_adata"]
    heldout_total = materialized["predict_adata"]
    build_subset = _sample_subset(build_pool, size=int(config["build_size"]), seed=int(config["seed"]) + 17)
    build_eval, predict_pool = _sample_disjoint_heldout(
        heldout_total,
        build_eval_size=int(config["build_eval_size"]),
        predict_pool_size=int(config["predict_pool_size"]),
        seed=int(config["seed"]) + 23,
    )
    predict_subsets = {
        int(size): _sample_subset(predict_pool, size=int(size), seed=int(config["seed"]) + int(size))
        for size in config["predict_sizes"]
    }
    phase_seconds["split_materialization_seconds"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    build_prepared, feature_panel, build_feature_report = select_reference_features(build_subset, preprocess_config)
    build_eval_prepared, build_eval_align_report = align_query_to_feature_panel(build_eval, feature_panel, preprocess_config)
    predict_prepared = {}
    predict_reports = {}
    for size, subset in predict_subsets.items():
        aligned, align_report = align_query_to_feature_panel(subset, feature_panel, preprocess_config)
        predict_prepared[size] = aligned
        predict_reports[size] = align_report
    phase_seconds["feature_selection_and_alignment_seconds"] = round(time.perf_counter() - t0, 4)

    build_report = _compose_preprocess_report(
        config=preprocess_config,
        canonical_report=canonical_report,
        counts_meta=full_counts_meta,
        feature_report=build_feature_report,
    )
    build_eval_report = _compose_preprocess_report(
        config=preprocess_config,
        canonical_report=canonical_report,
        counts_meta=full_counts_meta,
        feature_report=build_eval_align_report,
        matched_feature_genes=build_eval_align_report.matched_feature_genes,
        missing_feature_genes=build_eval_align_report.missing_feature_genes,
    )
    attach_preprocess_metadata(build_prepared, config=preprocess_config, report=build_report, feature_panel=feature_panel)
    attach_preprocess_metadata(build_eval_prepared, config=preprocess_config, report=build_eval_report, feature_panel=feature_panel)
    for size, aligned in predict_prepared.items():
        report = _compose_preprocess_report(
            config=preprocess_config,
            canonical_report=canonical_report,
            counts_meta=full_counts_meta,
            feature_report=predict_reports[size],
            matched_feature_genes=predict_reports[size].matched_feature_genes,
            missing_feature_genes=predict_reports[size].missing_feature_genes,
        )
        attach_preprocess_metadata(aligned, config=preprocess_config, report=report, feature_panel=feature_panel)

    t0 = time.perf_counter()
    build_dir = prepared_root / "build_scaling" / "build_100000"
    predict_dir = prepared_root / "predict_scaling" / "fixed_build_100000"
    canonical_dir = prepared_root / "canonical_subsets"
    _write_adata(build_pool, canonical_dir / f"reference_build_pool_{config['build_pool_size']}_canonical.h5ad")
    _write_adata(heldout_total, canonical_dir / f"heldout_total_{config['heldout_total_size']}_canonical.h5ad")
    _write_adata(build_eval, canonical_dir / "heldout_build_eval_10k_canonical.h5ad")
    _write_adata(predict_pool, canonical_dir / f"heldout_predict_pool_{config['predict_pool_size']}_canonical.h5ad")

    _write_adata(build_prepared, build_dir / "reference_train_100000.h5ad")
    _write_adata(build_eval_prepared, build_dir / "heldout_build_eval_10k.h5ad")
    save_feature_panel(feature_panel, str(build_dir / "feature_panel.json"))

    for size, aligned in predict_prepared.items():
        _write_adata(aligned, predict_dir / f"heldout_predict_{size}.h5ad")
    save_feature_panel(feature_panel, str(predict_dir / "feature_panel.json"))
    phase_seconds["write_outputs_seconds"] = round(time.perf_counter() - t0, 4)

    split_summary = dict(materialized["split_summary"])
    split_summary.update(
        {
            "dataset_name": str(config["dataset_name"]),
            "domain_key": str(config["domain_key"]),
            "source_h5ad": str(source_h5ad),
            "build_pool_cells": int(build_pool.n_obs),
            "build_subset_cells": int(build_subset.n_obs),
            "heldout_total_cells": int(heldout_total.n_obs),
            "build_eval_cells": int(build_eval.n_obs),
            "predict_pool_cells": int(predict_pool.n_obs),
            "predict_subset_cells": {str(size): int(adata.n_obs) for size, adata in predict_subsets.items()},
            "build_subset_label_counts": build_subset.obs[str(config["target_label"])].astype(str).value_counts(dropna=False).to_dict(),
            "build_eval_label_counts": build_eval.obs[str(config["target_label"])].astype(str).value_counts(dropna=False).to_dict(),
            "predict_pool_label_counts": predict_pool.obs[str(config["target_label"])].astype(str).value_counts(dropna=False).to_dict(),
        }
    )

    prep_usage = tracker.finish(
        phase="prepare_hlca_study_split",
        num_items=raw_adata.n_obs,
        num_batches=None,
        device_used="cpu",
        num_threads_used=None,
    )

    prepared_root.mkdir(parents=True, exist_ok=True)
    (prepared_root / "split_plan.json").write_text(
        json.dumps(_json_safe(split_plan), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (prepared_root / "split_summary.json").write_text(
        json.dumps(_json_safe(split_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (prepared_root / "preprocessing_summary.json").write_text(
        json.dumps(
            {
                "dataset_name": str(config["dataset_name"]),
                "source_h5ad": str(source_h5ad),
                "prepared_root": str(prepared_root),
                "label_columns": list(config["label_columns"]),
                "target_label": str(config["target_label"]),
                "preprocess_config": preprocess_config.to_dict(),
                "full_counts_meta": _json_safe(full_counts_meta),
                "canonical_report": canonical_report.to_dict(),
                "build_report": build_report.to_dict(),
                "build_eval_report": build_eval_report.to_dict(),
                "predict_reports": {str(size): report.to_dict() for size, report in predict_reports.items()},
                "feature_panel": feature_panel.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (prepared_root / "preparation_resource_summary.json").write_text(
        json.dumps(
            {
                "dataset_name": str(config["dataset_name"]),
                "source_h5ad": str(source_h5ad),
                "phase_seconds": phase_seconds,
                "resource_usage": _json_safe(prep_usage),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    _write_study_inventory(
        path=prepared_root / "study_inventory.csv",
        dataset_total=raw_adata,
        build_groups=list(split_plan["build_groups"]),
        predict_groups=list(split_plan["predict_groups"]),
    )
    _write_label_coverage_summary(
        path=prepared_root / "label_coverage_summary.csv",
        target_label=str(config["target_label"]),
        dataset_total=raw_adata,
        build_pool=build_pool,
        build_subset=build_subset,
        heldout_total=heldout_total,
        build_eval=build_eval,
        predict_pool=predict_pool,
        predict_subsets=predict_subsets,
    )
    _write_markdown_reports(
        repo_root=repo_root,
        config_path=config_path,
        config=config,
        split_plan=split_plan,
        split_summary=split_summary,
        prepared_root=prepared_root,
    )

    print(
        json.dumps(
            {
                "prepared_root": str(prepared_root),
                "build_groups": split_plan["build_groups"],
                "predict_groups": split_plan["predict_groups"],
                "build_subset_cells": int(build_subset.n_obs),
                "heldout_total_cells": int(heldout_total.n_obs),
                "predict_sizes": list(predict_subsets.keys()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
