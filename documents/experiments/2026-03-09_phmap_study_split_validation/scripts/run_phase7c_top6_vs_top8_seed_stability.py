#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
from anndata import read_h5ad

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atlasmtl import TrainedModel, predict
from atlasmtl.core.evaluate import evaluate_hierarchy_metrics, evaluate_prediction_behavior, evaluate_predictions
from atlasmtl.mapping import (
    build_parent_conditioned_refinement_plan,
    discover_hotspot_parents,
    fit_parent_conditioned_reranker,
)
from benchmark.methods.atlasmtl_inputs import adata_with_matrix_from_layer, resolve_atlasmtl_layer_config

DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
RESULTS_DIR = DOSSIER_ROOT / "results_summary"
ARTIFACT_ROOT = DOSSIER_ROOT / "artifacts" / "phase7c_top6_vs_top8"
RESULTS_SUBDIR = RESULTS_DIR / "phase7c_top6_vs_top8"
PHASE2_MANIFEST_INDEX = DOSSIER_ROOT / "manifests" / "phase2_seed" / "predict_manifest_index.json"
PHASE2_TRAIN_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-09/phmap_study_split_phase2_seed/train/lv4strong_plus_class_weight")

LABEL_COLUMNS = ["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"]
POINTS = ("build_100000_eval10k", "predict_100000_10000")
TARGET_POINT = "predict_100000_10000"
TARGET_CONFIG = "lv4strong_plus_class_weight"
UNKNOWN = "Unknown"
SELECTION_SPECS = (("reranker_top6", 6), ("reranker_top8", 8))
MIN_CELLS_PER_PARENT = 200
SEED_2026 = 2026


def _load_predict_index() -> List[Dict[str, Any]]:
    return json.loads(PHASE2_MANIFEST_INDEX.read_text(encoding="utf-8"))


def _seed_model_manifests() -> List[Tuple[int, Path]]:
    rows: List[Tuple[int, Path]] = []
    for seed_dir in sorted(p for p in PHASE2_TRAIN_ROOT.iterdir() if p.is_dir()):
        seed = int(seed_dir.name.replace("seed_", ""))
        manifest = seed_dir / "runs" / "atlasmtl" / "atlasmtl_model_manifest.json"
        if manifest.exists():
            rows.append((seed, manifest))
    return rows


def _load_manifest(seed: int, point: str) -> Dict[str, Any]:
    index = _load_predict_index()
    for row in index:
        if row["config_name"] == TARGET_CONFIG and int(row["seed"]) == seed and row["point"] == point:
            return yaml.safe_load(Path(row["predict_manifest_path"]).read_text(encoding="utf-8"))
    raise FileNotFoundError(f"predict manifest not found for seed={seed}, point={point}")


def _build_hierarchy_rules(reference_adata, label_columns: List[str]) -> Dict[str, Dict[str, Any]]:
    rules: Dict[str, Dict[str, Any]] = {}
    for parent_col, child_col in zip(label_columns[:-1], label_columns[1:]):
        pairs = (
            reference_adata.obs[[parent_col, child_col]]
            .dropna()
            .drop_duplicates()
            .astype(str)
            .to_records(index=False)
        )
        rules[str(child_col)] = {
            "parent_col": str(parent_col),
            "child_to_parent": {str(child): str(parent) for parent, child in pairs},
        }
    return rules


def _edge_breakdown(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    *,
    parent_col: str,
    child_col: str,
    child_to_parent: Dict[str, str],
) -> Dict[str, float]:
    parent_true = true_df[parent_col].astype(str)
    child_true = true_df[child_col].astype(str)
    parent_pred = pred_df[f"pred_{parent_col}"].astype(str)
    child_pred = pred_df[f"pred_{child_col}"].astype(str)
    implied_parent = child_pred.map(lambda x: child_to_parent.get(x) if x != UNKNOWN else UNKNOWN).fillna("MISSING")
    parent_correct = parent_true == parent_pred
    child_correct = child_true == child_pred
    path_break = (child_pred != UNKNOWN) & (parent_pred != UNKNOWN) & (implied_parent != parent_pred)
    return {
        "parent_correct_child_wrong_rate": float((parent_correct & ~child_correct).mean()),
        "path_break_rate": float(path_break.mean()),
    }


def _summarize_variant(
    *,
    seed: int,
    variant_name: str,
    point: str,
    selected_parents: List[str],
    result,
    truth_df: pd.DataFrame,
    hierarchy_rules: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    level_metrics = evaluate_predictions(result.predictions, truth_df, LABEL_COLUMNS)
    behavior = evaluate_prediction_behavior(result.predictions, truth_df, LABEL_COLUMNS)
    hierarchy = evaluate_hierarchy_metrics(result.predictions, truth_df, LABEL_COLUMNS, hierarchy_rules=hierarchy_rules)
    edge = _edge_breakdown(
        result.predictions,
        truth_df,
        parent_col="anno_lv3",
        child_col="anno_lv4",
        child_to_parent=hierarchy_rules["anno_lv4"]["child_to_parent"],
    )
    return {
        "seed": seed,
        "variant_name": variant_name,
        "point": point,
        "selected_parents": "|".join(selected_parents),
        "selected_parent_count": int(len(selected_parents)),
        "anno_lv4_macro_f1": level_metrics["anno_lv4"]["macro_f1"],
        "anno_lv4_balanced_accuracy": level_metrics["anno_lv4"]["balanced_accuracy"],
        "anno_lv4_coverage": level_metrics["anno_lv4"]["coverage"],
        "anno_lv4_unknown_rate": behavior["anno_lv4"]["unknown_rate"],
        "full_path_accuracy": hierarchy.get("full_path_accuracy", 0.0),
        "full_path_coverage": hierarchy.get("full_path_coverage", 0.0),
        "mean_path_consistency_rate": hierarchy["edges"]["anno_lv4"]["path_consistency_rate"],
        **edge,
    }


def _load_query(seed: int, point: str):
    manifest = _load_manifest(seed, point)
    layer_cfg = resolve_atlasmtl_layer_config(manifest)
    query_raw = read_h5ad(str(manifest["query_h5ad"]))
    query_model_input = adata_with_matrix_from_layer(query_raw, layer_name=layer_cfg["query_layer"])
    truth_df = query_raw.obs[LABEL_COLUMNS].astype(str).copy()
    return manifest, layer_cfg, query_raw, query_model_input, truth_df


def _discover_hotspots_for_seed(
    *,
    seed: int,
    model: TrainedModel,
    query_model_input,
    truth_df: pd.DataFrame,
    hierarchy_rules: Dict[str, Dict[str, Any]],
    artifact_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    baseline = predict(
        model,
        query_model_input,
        device="cpu",
        knn_correction="off",
        hierarchy_rules=hierarchy_rules,
        enforce_hierarchy=True,
        batch_size=512,
    )
    parent_label = "anno_lv3"
    child_label = "anno_lv4"
    parent_true = truth_df[parent_label].astype(str)
    child_true = truth_df[child_label].astype(str)
    parent_pred = baseline.predictions[f"pred_{parent_label}"].astype(str)
    child_pred = baseline.predictions[f"pred_{child_label}"].astype(str)
    rows: List[Dict[str, Any]] = []
    for parent in sorted(parent_true.unique().tolist()):
        mask = parent_true == str(parent)
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue
        parent_correct = parent_pred[mask] == parent_true[mask]
        child_correct = child_pred[mask] == child_true[mask]
        rows.append(
            {
                "parent_label": str(parent),
                "n_cells": n_cells,
                "parent_correct_child_wrong_rate": float((parent_correct & ~child_correct).mean()),
            }
        )
    subtree_df = pd.DataFrame(rows)
    hotspot_sets: Dict[str, List[str]] = {}
    ranking_payload: Dict[str, Any] = {"seed": seed, "selection_source": "seed-local baseline", "variants": {}}
    for variant_name, top_k in SELECTION_SPECS:
        ranking_df, selected_parents, summary = discover_hotspot_parents(
            subtree_df,
            selection_mode="topk",
            top_k=top_k,
            min_cells_per_parent=MIN_CELLS_PER_PARENT,
        )
        hotspot_sets[variant_name] = selected_parents
        ranking_payload["variants"][variant_name] = {
            "top_k": top_k,
            "selected_parents": selected_parents,
            "summary": summary,
            "ranking": ranking_df[
                ["parent_label", "n_cells", "parent_correct_child_wrong_rate", "selection_score", "cumulative_contribution"]
            ].to_dict(orient="records"),
        }
    (artifact_dir / "hotspot_ranking.json").write_text(
        json.dumps(ranking_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return subtree_df, hotspot_sets


def _variant_metrics_by_study(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    seed: int,
    variant_name: str,
    hierarchy_rules: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if "study" not in truth_df.columns:
        return rows
    for study, idx in truth_df.groupby("study").groups.items():
        truth_study = truth_df.loc[idx]
        pred_study = pred_df.loc[idx]
        if truth_study.empty:
            continue
        edge = _edge_breakdown(
            pred_study,
            truth_study,
            parent_col="anno_lv3",
            child_col="anno_lv4",
            child_to_parent=hierarchy_rules["anno_lv4"]["child_to_parent"],
        )
        level_metrics = evaluate_predictions(pred_study, truth_study, LABEL_COLUMNS)
        hierarchy = evaluate_hierarchy_metrics(
            pred_study,
            truth_study,
            LABEL_COLUMNS,
            hierarchy_rules=hierarchy_rules,
        )
        rows.append(
            {
                "seed": seed,
                "variant_name": variant_name,
                "study": str(study),
                "anno_lv4_macro_f1": level_metrics["anno_lv4"]["macro_f1"],
                "full_path_accuracy": hierarchy.get("full_path_accuracy", 0.0),
                "parent_correct_child_wrong_rate": edge["parent_correct_child_wrong_rate"],
            }
        )
    return rows


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_SUBDIR.mkdir(parents=True, exist_ok=True)

    comparison_rows: List[Dict[str, Any]] = []
    by_study_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []
    selected_by_seed: Dict[int, Dict[str, List[str]]] = {}
    seed2026_sets: Dict[str, List[str]] = {}

    seed_manifests = _seed_model_manifests()
    execution_payload = {
        "config_name": TARGET_CONFIG,
        "seeds": [seed for seed, _ in seed_manifests],
        "variants": ["baseline", "reranker_top6", "reranker_top8"],
        "points": list(POINTS),
    }

    for seed, model_manifest in seed_manifests:
        artifact_dir = ARTIFACT_ROOT / f"seed_{seed}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model = TrainedModel.load(str(model_manifest))

        target_manifest = _load_manifest(seed, TARGET_POINT)
        target_layer_cfg = resolve_atlasmtl_layer_config(target_manifest)
        ref_raw = read_h5ad(str(target_manifest["reference_h5ad"]))
        ref_model_input = adata_with_matrix_from_layer(ref_raw, layer_name=target_layer_cfg["reference_layer"])
        hierarchy_rules = _build_hierarchy_rules(ref_raw, LABEL_COLUMNS)

        _, _, target_query_raw, target_query_input, target_truth_df = _load_query(seed, TARGET_POINT)
        _, hotspot_sets = _discover_hotspots_for_seed(
            seed=seed,
            model=model,
            query_model_input=target_query_input,
            truth_df=target_truth_df,
            hierarchy_rules=hierarchy_rules,
            artifact_dir=artifact_dir,
        )
        selected_by_seed[seed] = {name: list(parents) for name, parents in hotspot_sets.items()}
        if seed == SEED_2026:
            seed2026_sets = {name: list(parents) for name, parents in hotspot_sets.items()}

        variant_artifacts: Dict[str, str] = {}
        for variant_name, top_k in SELECTION_SPECS:
            selected_parents = hotspot_sets[variant_name]
            artifact = fit_parent_conditioned_reranker(
                model,
                ref_model_input,
                parent_level="anno_lv3",
                child_level="anno_lv4",
                hotspot_parents=selected_parents,
                hierarchy_rules=hierarchy_rules,
                batch_size=512,
                device="cpu",
                selection_metadata={
                    "selection_source": "seed-local baseline",
                    "selection_point": f"{TARGET_POINT} + hierarchy_on",
                    "selection_score": "parent_correct_child_wrong_rate * n_cells",
                    "selection_mode": "topk",
                    "hotspot_topk": top_k,
                    "min_cells_per_parent": MIN_CELLS_PER_PARENT,
                    "seed": seed,
                },
            )
            artifact_path = artifact_dir / f"parent_conditioned_{variant_name}.pkl"
            artifact.save(artifact_path)
            pd.DataFrame(artifact.per_parent_summary).to_csv(
                artifact_dir / f"per_parent_reranker_summary_{variant_name}.csv",
                index=False,
            )
            plan = build_parent_conditioned_refinement_plan(
                parent_level="anno_lv3",
                child_level="anno_lv4",
                selection_source="seed-local baseline",
                selection_point=f"{TARGET_POINT} + hierarchy_on",
                selection_score="parent_correct_child_wrong_rate * n_cells",
                selection_mode="topk",
                selected_parents=selected_parents,
                artifact_path=str(artifact_path),
                top_k=top_k,
                min_cells_per_parent=MIN_CELLS_PER_PARENT,
                fallback_to_base=True,
                guardrail={
                    "point": f"{TARGET_POINT} + hierarchy_on",
                    "rules": [
                        "anno_lv4_macro_f1 >= base",
                        "full_path_accuracy >= base",
                        "parent_correct_child_wrong_rate <= base",
                    ],
                },
                ranking_path=str(artifact_dir / "hotspot_ranking.json"),
                per_parent_summary_path=str(artifact_dir / f"per_parent_reranker_summary_{variant_name}.csv"),
            )
            plan.save(artifact_dir / f"refinement_plan_{variant_name}.json")
            variant_artifacts[variant_name] = str(artifact_path)

        for point in POINTS:
            _, _, query_raw, query_input, truth_df = _load_query(seed, point)
            base_result = predict(
                model,
                query_input,
                device="cpu",
                knn_correction="off",
                hierarchy_rules=hierarchy_rules,
                enforce_hierarchy=True,
                batch_size=512,
            )
            comparison_rows.append(
                _summarize_variant(
                    seed=seed,
                    variant_name="baseline",
                    point=point,
                    selected_parents=[],
                    result=base_result,
                    truth_df=truth_df,
                    hierarchy_rules=hierarchy_rules,
                )
            )
            if point == TARGET_POINT:
                for row in _variant_metrics_by_study(
                    base_result.predictions,
                    query_raw.obs[LABEL_COLUMNS + ["study"]].astype(str),
                    seed,
                    "baseline",
                    hierarchy_rules,
                ):
                    by_study_rows.append(row)

            for variant_name, _ in SELECTION_SPECS:
                refined_result = predict(
                    model,
                    query_input,
                    device="cpu",
                    knn_correction="off",
                    hierarchy_rules=hierarchy_rules,
                    enforce_hierarchy=True,
                    batch_size=512,
                    refinement_config={
                        "method": "auto_parent_conditioned_reranker",
                        "plan_path": str(artifact_dir / f"refinement_plan_{variant_name}.json"),
                    },
                )
                selected_parents = hotspot_sets[variant_name]
                comparison_rows.append(
                    _summarize_variant(
                        seed=seed,
                        variant_name=variant_name,
                        point=point,
                        selected_parents=selected_parents,
                        result=refined_result,
                        truth_df=truth_df,
                        hierarchy_rules=hierarchy_rules,
                    )
                )
                if point == TARGET_POINT:
                    for row in _variant_metrics_by_study(
                        refined_result.predictions,
                        query_raw.obs[LABEL_COLUMNS + ["study"]].astype(str),
                        seed,
                        variant_name,
                        hierarchy_rules,
                    ):
                        by_study_rows.append(row)

    for seed, hotspot_sets in selected_by_seed.items():
        for variant_name, selected_parents in hotspot_sets.items():
            seed2026_parents = set(seed2026_sets.get(variant_name, []))
            current = set(selected_parents)
            overlap = current & seed2026_parents
            overlap_rows.append(
                {
                    "seed": seed,
                    "variant_name": variant_name,
                    "selected_parents": "|".join(selected_parents),
                    "selected_parent_count": len(selected_parents),
                    "overlap_with_seed2026_count": len(overlap),
                    "overlap_with_seed2026_fraction": 0.0
                    if len(current) == 0
                    else float(len(overlap) / len(current)),
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    baseline_cols = comparison_df[comparison_df["variant_name"] == "baseline"][
        ["seed", "point", "anno_lv4_macro_f1", "full_path_accuracy", "parent_correct_child_wrong_rate"]
    ].rename(
        columns={
            "anno_lv4_macro_f1": "baseline_anno_lv4_macro_f1",
            "full_path_accuracy": "baseline_full_path_accuracy",
            "parent_correct_child_wrong_rate": "baseline_parent_correct_child_wrong_rate",
        }
    )
    comparison_df = comparison_df.merge(baseline_cols, on=["seed", "point"], how="left")
    comparison_df["delta_vs_baseline_macro_f1"] = (
        comparison_df["anno_lv4_macro_f1"] - comparison_df["baseline_anno_lv4_macro_f1"]
    )
    comparison_df["delta_vs_baseline_full_path_accuracy"] = (
        comparison_df["full_path_accuracy"] - comparison_df["baseline_full_path_accuracy"]
    )
    comparison_df["delta_vs_baseline_parent_correct_child_wrong_rate"] = (
        comparison_df["parent_correct_child_wrong_rate"] - comparison_df["baseline_parent_correct_child_wrong_rate"]
    )
    comparison_df.to_csv(RESULTS_SUBDIR / "phase7c_top6_vs_top8_seed_stability.csv", index=False)

    summary_df = (
        comparison_df[comparison_df["variant_name"].isin(["reranker_top6", "reranker_top8"])]
        .groupby(["variant_name", "point"], dropna=False)
        .agg(
            anno_lv4_macro_f1_mean=("anno_lv4_macro_f1", "mean"),
            anno_lv4_macro_f1_std=("anno_lv4_macro_f1", "std"),
            full_path_accuracy_mean=("full_path_accuracy", "mean"),
            full_path_accuracy_std=("full_path_accuracy", "std"),
            parent_correct_child_wrong_rate_mean=("parent_correct_child_wrong_rate", "mean"),
            parent_correct_child_wrong_rate_std=("parent_correct_child_wrong_rate", "std"),
            delta_vs_baseline_macro_f1_mean=("delta_vs_baseline_macro_f1", "mean"),
            delta_vs_baseline_full_path_accuracy_mean=("delta_vs_baseline_full_path_accuracy", "mean"),
            delta_vs_baseline_parent_correct_child_wrong_rate_mean=(
                "delta_vs_baseline_parent_correct_child_wrong_rate",
                "mean",
            ),
        )
        .reset_index()
    )
    summary_df.to_csv(RESULTS_SUBDIR / "phase7c_top6_vs_top8_seed_summary.csv", index=False)

    by_study_df = pd.DataFrame(by_study_rows)
    by_study_df.to_csv(RESULTS_SUBDIR / "phase7c_top6_vs_top8_by_study.csv", index=False)

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(RESULTS_SUBDIR / "phase7c_top6_vs_top8_parent_overlap.csv", index=False)

    target_summary = summary_df[summary_df["point"] == TARGET_POINT].copy().set_index("variant_name")
    top6 = target_summary.loc["reranker_top6"]
    top8 = target_summary.loc["reranker_top8"]
    top8_better = (
        float(top8["anno_lv4_macro_f1_mean"]) >= float(top6["anno_lv4_macro_f1_mean"])
        and float(top8["full_path_accuracy_mean"]) > float(top6["full_path_accuracy_mean"])
        and float(top8["parent_correct_child_wrong_rate_mean"]) < float(top6["parent_correct_child_wrong_rate_mean"])
        and float(top8["anno_lv4_macro_f1_std"]) <= float(top6["anno_lv4_macro_f1_std"]) * 1.2
        and float(top8["full_path_accuracy_std"]) <= float(top6["full_path_accuracy_std"]) * 1.2
        and float(top8["parent_correct_child_wrong_rate_std"]) <= float(top6["parent_correct_child_wrong_rate_std"]) * 1.2
    )
    final_decision = (
        "Promote top8 to PH-Map default hotspot rule"
        if top8_better
        else "Keep top6 as stable default; retain top8 as optional wider-coverage mode"
    )

    summary_lines = [
        "# Phase 7C top6 vs top8 seed stability",
        "",
        "- base model: `lv4strong_plus_class_weight`",
        "- refinement method: `auto parent-conditioned reranker`",
        "- seeds: `101, 17, 2026, 23, 47`",
        "- primary point: `predict_100000_10000 + hierarchy_on`",
        "",
        "## Decision",
        "",
        f"- `{final_decision}`",
        "",
        "## Primary comparison",
        "",
        f"- `reranker_top6`: macro_f1 `{float(top6['anno_lv4_macro_f1_mean']):.6f} ± {float(top6['anno_lv4_macro_f1_std']):.6f}`, full_path `{float(top6['full_path_accuracy_mean']):.5f} ± {float(top6['full_path_accuracy_std']):.5f}`, parent_correct_child_wrong `{float(top6['parent_correct_child_wrong_rate_mean']):.5f} ± {float(top6['parent_correct_child_wrong_rate_std']):.5f}`",
        f"- `reranker_top8`: macro_f1 `{float(top8['anno_lv4_macro_f1_mean']):.6f} ± {float(top8['anno_lv4_macro_f1_std']):.6f}`, full_path `{float(top8['full_path_accuracy_mean']):.5f} ± {float(top8['full_path_accuracy_std']):.5f}`, parent_correct_child_wrong `{float(top8['parent_correct_child_wrong_rate_mean']):.5f} ± {float(top8['parent_correct_child_wrong_rate_std']):.5f}`",
    ]
    (RESULTS_SUBDIR / "phase7c_top6_vs_top8_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (RESULTS_SUBDIR / "phase7c_execution_record.md").write_text(
        json.dumps(execution_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
