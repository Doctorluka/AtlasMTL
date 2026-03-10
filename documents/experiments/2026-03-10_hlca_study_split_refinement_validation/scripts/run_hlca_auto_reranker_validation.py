#!/usr/bin/env python
from __future__ import annotations

import json
import sys
import time
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

DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-10_hlca_study_split_refinement_validation"
ARTIFACT_DIR = DOSSIER_ROOT / "artifacts" / "hlca_auto_reranker_validation"
RESULTS_DIR = DOSSIER_ROOT / "results_summary" / "hlca_auto_reranker_validation"
PREDICT_MANIFEST_INDEX = DOSSIER_ROOT / "manifests" / "weight_confirmation" / "predict_manifest_index.json"
BEST_CONFIG_PATH = DOSSIER_ROOT / "results_summary" / "hlca_weight_confirmation_best_config.json"
MODEL_MANIFEST = Path(
    "/tmp/atlasmtl_benchmarks/2026-03-10/hlca_study_split_weight_confirmation/train/"
    "uniform/runs/atlasmtl/atlasmtl_model_manifest.json"
)
POINTS = ("build_100000_eval10k", "predict_100000_10000")
LABEL_COLUMNS = ["ann_level_1", "ann_level_2", "ann_level_3", "ann_level_4", "ann_level_5"]
PARENT_LEVEL = "ann_level_4"
CHILD_LEVEL = "ann_level_5"
TARGET_POINT = "predict_100000_10000"
UNKNOWN = "Unknown"
HOTSPOT_TOPK = 6
MIN_PARENT_CELLS = 200


def _load_manifest(point: str) -> Dict[str, Any]:
    rows = json.loads(PREDICT_MANIFEST_INDEX.read_text(encoding="utf-8"))
    for row in rows:
        if row["config_name"] == "uniform" and row["point"] == point:
            return yaml.safe_load(Path(row["predict_manifest_path"]).read_text(encoding="utf-8"))
    raise FileNotFoundError(f"manifest not found for point={point}")


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
        child_to_parent = {str(child): str(parent) for parent, child in pairs}
        rules[str(child_col)] = {
            "parent_col": str(parent_col),
            "child_to_parent": child_to_parent,
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
    implied_parent = child_pred.map(lambda x: child_to_parent.get(x) if x != UNKNOWN else UNKNOWN).fillna("MISSING_PARENT")
    parent_correct = parent_true == parent_pred
    child_correct = child_true == child_pred
    path_break = (child_pred != UNKNOWN) & (parent_pred != UNKNOWN) & (implied_parent != parent_pred)
    return {
        "parent_accuracy": float((parent_true == parent_pred).mean()),
        "child_accuracy": float((child_true == child_pred).mean()),
        "parent_correct_child_wrong_rate": float((parent_correct & ~child_correct).mean()),
        "parent_wrong_child_correct_rate": float((~parent_correct & child_correct).mean()),
        "both_wrong_rate": float((~parent_correct & ~child_correct).mean()),
        "child_unknown_rate": float((child_pred == UNKNOWN).mean()),
        "parent_unknown_rate": float((parent_pred == UNKNOWN).mean()),
        "path_break_rate": float(path_break.mean()),
    }


def _subtree_breakdown(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    *,
    parent_col: str,
    child_col: str,
    child_to_parent: Dict[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for parent_label, parent_truth in true_df.groupby(parent_col):
        idx = parent_truth.index
        sub_true = true_df.loc[idx]
        sub_pred = pred_df.loc[idx]
        edge = _edge_breakdown(
            sub_pred,
            sub_true,
            parent_col=parent_col,
            child_col=child_col,
            child_to_parent=child_to_parent,
        )
        child_true = sub_true[child_col].astype(str)
        child_pred = sub_pred[f"pred_{child_col}"].astype(str)
        rows.append(
            {
                "parent_label": str(parent_label),
                "n_cells": int(len(idx)),
                "child_accuracy": float((child_true == child_pred).mean()),
                "child_unknown_rate": float((child_pred == UNKNOWN).mean()),
                **edge,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["parent_correct_child_wrong_rate", "n_cells", "parent_label"],
        ascending=[False, False, True],
    )


def _summarize_variant(
    *,
    variant_name: str,
    point: str,
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
        parent_col=PARENT_LEVEL,
        child_col=CHILD_LEVEL,
        child_to_parent=hierarchy_rules[CHILD_LEVEL]["child_to_parent"],
    )
    return {
        "variant_name": variant_name,
        "point": point,
        "ann_level_5_macro_f1": level_metrics[CHILD_LEVEL]["macro_f1"],
        "ann_level_5_balanced_accuracy": level_metrics[CHILD_LEVEL]["balanced_accuracy"],
        "ann_level_5_coverage": level_metrics[CHILD_LEVEL]["coverage"],
        "ann_level_5_unknown_rate": behavior[CHILD_LEVEL]["unknown_rate"],
        "full_path_accuracy": hierarchy.get("full_path_accuracy", 0.0),
        "full_path_coverage": hierarchy.get("full_path_coverage", 0.0),
        "mean_path_consistency_rate": hierarchy["edges"][CHILD_LEVEL]["path_consistency_rate"],
        **edge,
    }


def _by_study_rows(
    *,
    variant_name: str,
    point: str,
    result,
    truth_df: pd.DataFrame,
    hierarchy_rules: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for study, study_truth in truth_df.groupby("study"):
        idx = study_truth.index
        study_pred = result.predictions.loc[idx]
        metrics = evaluate_predictions(study_pred, study_truth[LABEL_COLUMNS], LABEL_COLUMNS)
        hierarchy = evaluate_hierarchy_metrics(
            study_pred,
            study_truth[LABEL_COLUMNS],
            LABEL_COLUMNS,
            hierarchy_rules=hierarchy_rules,
        )
        edge = _edge_breakdown(
            study_pred,
            study_truth,
            parent_col=PARENT_LEVEL,
            child_col=CHILD_LEVEL,
            child_to_parent=hierarchy_rules[CHILD_LEVEL]["child_to_parent"],
        )
        rows.append(
            {
                "variant_name": variant_name,
                "point": point,
                "study": str(study),
                "n_cells": int(len(idx)),
                "ann_level_5_macro_f1": metrics[CHILD_LEVEL]["macro_f1"],
                "full_path_accuracy": hierarchy.get("full_path_accuracy", 0.0),
                "parent_correct_child_wrong_rate": edge["parent_correct_child_wrong_rate"],
            }
        )
    return rows


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_target = _load_manifest(TARGET_POINT)
    layer_cfg_target = resolve_atlasmtl_layer_config(manifest_target)
    ref_raw = read_h5ad(str(manifest_target["reference_h5ad"]))
    ref_model_input = adata_with_matrix_from_layer(ref_raw, layer_name=layer_cfg_target["reference_layer"])
    query_target_raw = read_h5ad(str(manifest_target["query_h5ad"]))
    query_target_model_input = adata_with_matrix_from_layer(query_target_raw, layer_name=layer_cfg_target["query_layer"])
    query_target_truth = query_target_raw.obs[LABEL_COLUMNS + ["study"]].astype(str).copy()
    hierarchy_rules = _build_hierarchy_rules(ref_raw, LABEL_COLUMNS)

    best_payload = json.loads(BEST_CONFIG_PATH.read_text(encoding="utf-8"))
    model = TrainedModel.load(str(MODEL_MANIFEST))

    baseline_target = predict(
        model,
        query_target_model_input,
        device="cpu",
        knn_correction="off",
        hierarchy_rules=hierarchy_rules,
        enforce_hierarchy=True,
        batch_size=512,
    )

    subtree_df = _subtree_breakdown(
        baseline_target.predictions,
        query_target_truth,
        parent_col=PARENT_LEVEL,
        child_col=CHILD_LEVEL,
        child_to_parent=hierarchy_rules[CHILD_LEVEL]["child_to_parent"],
    )
    subtree_df.to_csv(RESULTS_DIR / "hlca_subtree_breakdown.csv", index=False)

    ranking_df, selected_parents, ranking_summary = discover_hotspot_parents(
        subtree_df,
        selection_mode="topk",
        top_k=HOTSPOT_TOPK,
        min_cells_per_parent=MIN_PARENT_CELLS,
    )
    ranking_payload = {
        "dataset": "HLCA_Core",
        "selection_source": "HLCA best base config",
        "selection_point": f"{TARGET_POINT} + hierarchy_on",
        "selection_score": "parent_correct_child_wrong_rate * n_cells",
        "selection_mode": "topk",
        "top_k": HOTSPOT_TOPK,
        "min_cells_per_parent": MIN_PARENT_CELLS,
        "best_base_config_name": best_payload["best_config_name"],
        "selected_parents": selected_parents,
        "summary": ranking_summary,
        "ranking": ranking_df[
            ["parent_label", "n_cells", "parent_correct_child_wrong_rate", "selection_score", "cumulative_contribution"]
        ].to_dict(orient="records"),
    }
    ranking_path = ARTIFACT_DIR / "hlca_hotspot_ranking.json"
    ranking_path.write_text(json.dumps(ranking_payload, indent=2, sort_keys=True), encoding="utf-8")

    fit_start = time.perf_counter()
    artifact = fit_parent_conditioned_reranker(
        model,
        ref_model_input,
        parent_level=PARENT_LEVEL,
        child_level=CHILD_LEVEL,
        hotspot_parents=selected_parents,
        hierarchy_rules=hierarchy_rules,
        batch_size=512,
        device="cpu",
        selection_metadata={
            "selection_source": "HLCA best base config",
            "selection_point": f"{TARGET_POINT} + hierarchy_on",
            "selection_score": "parent_correct_child_wrong_rate * n_cells",
            "selection_mode": "topk",
            "hotspot_topk": HOTSPOT_TOPK,
            "min_cells_per_parent": MIN_PARENT_CELLS,
            "best_base_config_name": best_payload["best_config_name"],
        },
    )
    fit_elapsed = time.perf_counter() - fit_start
    artifact_path = ARTIFACT_DIR / "hlca_parent_conditioned_reranker_top6.pkl"
    artifact.save(artifact_path)
    per_parent_summary_path = ARTIFACT_DIR / "hlca_per_parent_reranker_summary.csv"
    pd.DataFrame(artifact.per_parent_summary).to_csv(per_parent_summary_path, index=False)

    plan = build_parent_conditioned_refinement_plan(
        parent_level=PARENT_LEVEL,
        child_level=CHILD_LEVEL,
        selection_source="HLCA best base config",
        selection_point=f"{TARGET_POINT} + hierarchy_on",
        selection_score="parent_correct_child_wrong_rate * n_cells",
        selection_mode="topk",
        selected_parents=selected_parents,
        artifact_path=str(artifact_path),
        top_k=HOTSPOT_TOPK,
        min_cells_per_parent=MIN_PARENT_CELLS,
        fallback_to_base=True,
        guardrail={
            "point": f"{TARGET_POINT} + hierarchy_on",
            "rules": [
                "ann_level_5_macro_f1 >= base",
                "full_path_accuracy >= base",
                "parent_correct_child_wrong_rate <= base",
            ],
        },
        ranking_path=str(ranking_path),
        per_parent_summary_path=str(per_parent_summary_path),
    )
    plan_path = ARTIFACT_DIR / "hlca_refinement_plan.json"
    plan.save(plan_path)

    comparison_rows: List[Dict[str, Any]] = []
    by_study_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    for point in POINTS:
        manifest = _load_manifest(point)
        layer_cfg = resolve_atlasmtl_layer_config(manifest)
        query_raw = read_h5ad(str(manifest["query_h5ad"]))
        query_model_input = adata_with_matrix_from_layer(query_raw, layer_name=layer_cfg["query_layer"])
        truth_df = query_raw.obs[LABEL_COLUMNS + ["study"]].astype(str).copy()

        base_result = predict(
            model,
            query_model_input,
            device="cpu",
            knn_correction="off",
            hierarchy_rules=hierarchy_rules,
            enforce_hierarchy=True,
            batch_size=512,
        )
        apply_start = time.perf_counter()
        auto_result = predict(
            model,
            query_model_input,
            device="cpu",
            knn_correction="off",
            hierarchy_rules=hierarchy_rules,
            enforce_hierarchy=True,
            batch_size=512,
            refinement_config={
                "method": "auto_parent_conditioned_reranker",
                "plan_path": str(plan_path),
            },
        )
        apply_elapsed = time.perf_counter() - apply_start

        for variant_name, result in (("baseline", base_result), ("auto_parent_conditioned_reranker", auto_result)):
            summary = _summarize_variant(
                variant_name=variant_name,
                point=point,
                result=result,
                truth_df=truth_df[LABEL_COLUMNS],
                hierarchy_rules=hierarchy_rules,
            )
            summary["reranker_fit_seconds"] = fit_elapsed if variant_name != "baseline" else 0.0
            summary["reranker_apply_seconds"] = apply_elapsed if variant_name != "baseline" else 0.0
            comparison_rows.append(summary)
            edge_rows.append(
                {
                    "variant_name": variant_name,
                    "point": point,
                    "parent_col": PARENT_LEVEL,
                    "child_col": CHILD_LEVEL,
                    **{
                        key: summary[key]
                        for key in (
                            "parent_accuracy",
                            "child_accuracy",
                            "parent_correct_child_wrong_rate",
                            "parent_wrong_child_correct_rate",
                            "both_wrong_rate",
                            "child_unknown_rate",
                            "parent_unknown_rate",
                            "path_break_rate",
                        )
                    },
                }
            )
            by_study_rows.extend(
                _by_study_rows(
                    variant_name=variant_name,
                    point=point,
                    result=result,
                    truth_df=truth_df,
                    hierarchy_rules=hierarchy_rules,
                )
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["point", "variant_name"]).reset_index(drop=True)
    baseline_cols = ["ann_level_5_macro_f1", "full_path_accuracy", "parent_correct_child_wrong_rate"]
    baseline_df = comparison_df[comparison_df["variant_name"] == "baseline"][
        ["point"] + baseline_cols
    ].rename(columns={col: f"{col}_baseline" for col in baseline_cols})
    comparison_df = comparison_df.merge(baseline_df, on="point", how="left")
    comparison_df["delta_vs_baseline_ann_level_5_macro_f1"] = (
        comparison_df["ann_level_5_macro_f1"] - comparison_df["ann_level_5_macro_f1_baseline"]
    )
    comparison_df["delta_vs_baseline_full_path_accuracy"] = (
        comparison_df["full_path_accuracy"] - comparison_df["full_path_accuracy_baseline"]
    )
    comparison_df["delta_vs_baseline_parent_correct_child_wrong_rate"] = (
        comparison_df["parent_correct_child_wrong_rate"] - comparison_df["parent_correct_child_wrong_rate_baseline"]
    )
    comparison_df.to_csv(RESULTS_DIR / "hlca_main_comparison.csv", index=False)

    edge_df = pd.DataFrame(edge_rows).sort_values(["point", "variant_name"]).reset_index(drop=True)
    edge_df.to_csv(RESULTS_DIR / "hlca_error_decomposition.csv", index=False)

    by_study_df = pd.DataFrame(by_study_rows).sort_values(["point", "study", "variant_name"]).reset_index(drop=True)
    by_study_df.to_csv(RESULTS_DIR / "hlca_by_study.csv", index=False)

    target_base = comparison_df[
        (comparison_df["point"] == TARGET_POINT) & (comparison_df["variant_name"] == "baseline")
    ].iloc[0]
    target_auto = comparison_df[
        (comparison_df["point"] == TARGET_POINT) & (comparison_df["variant_name"] == "auto_parent_conditioned_reranker")
    ].iloc[0]
    guardrail_pass = bool(
        float(target_auto["ann_level_5_macro_f1"]) >= float(target_base["ann_level_5_macro_f1"])
        and float(target_auto["full_path_accuracy"]) >= float(target_base["full_path_accuracy"])
        and float(target_auto["parent_correct_child_wrong_rate"]) <= float(target_base["parent_correct_child_wrong_rate"])
    )
    guardrail_payload = {
        "passed": guardrail_pass,
        "target_point": f"{TARGET_POINT} + hierarchy_on",
        "baseline": {
            "ann_level_5_macro_f1": float(target_base["ann_level_5_macro_f1"]),
            "full_path_accuracy": float(target_base["full_path_accuracy"]),
            "parent_correct_child_wrong_rate": float(target_base["parent_correct_child_wrong_rate"]),
        },
        "candidate": {
            "ann_level_5_macro_f1": float(target_auto["ann_level_5_macro_f1"]),
            "full_path_accuracy": float(target_auto["full_path_accuracy"]),
            "parent_correct_child_wrong_rate": float(target_auto["parent_correct_child_wrong_rate"]),
        },
        "rules": plan.guardrail["rules"],
    }
    guardrail_path = ARTIFACT_DIR / "hlca_guardrail_decision.json"
    guardrail_path.write_text(json.dumps(guardrail_payload, indent=2, sort_keys=True), encoding="utf-8")

    summary_lines = [
        "# HLCA AutoHotspot Reranker Validation",
        "",
        f"- best base config: `{best_payload['best_config_name']}`",
        f"- target edge: `{PARENT_LEVEL} -> {CHILD_LEVEL}`",
        f"- selection rule: `parent_correct_child_wrong_rate * n_cells`, `topk={HOTSPOT_TOPK}`, `min_cells_per_parent={MIN_PARENT_CELLS}`",
        f"- selected hotspot parents: {', '.join(selected_parents) if selected_parents else '(none)'}",
        "",
        "## Target point",
        "",
        (
            f"- baseline: macro_f1 `{float(target_base['ann_level_5_macro_f1']):.6f}`, "
            f"full_path `{float(target_base['full_path_accuracy']):.4f}`, "
            f"parent_correct_child_wrong `{float(target_base['parent_correct_child_wrong_rate']):.4f}`"
        ),
        (
            f"- auto reranker: macro_f1 `{float(target_auto['ann_level_5_macro_f1']):.6f}`, "
            f"full_path `{float(target_auto['full_path_accuracy']):.4f}`, "
            f"parent_correct_child_wrong `{float(target_auto['parent_correct_child_wrong_rate']):.4f}`"
        ),
        f"- guardrail pass: `{guardrail_pass}`",
        "",
        "## Outputs",
        "",
        "- `hlca_main_comparison.csv`",
        "- `hlca_error_decomposition.csv`",
        "- `hlca_subtree_breakdown.csv`",
        "- `hlca_by_study.csv`",
        "- `hlca_hotspot_ranking.json`",
        "- `hlca_refinement_plan.json`",
        "- `hlca_guardrail_decision.json`",
    ]
    (RESULTS_DIR / "hlca_auto_reranker_validation.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
