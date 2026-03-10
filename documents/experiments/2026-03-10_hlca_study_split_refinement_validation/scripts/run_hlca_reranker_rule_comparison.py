#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from anndata import read_h5ad

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atlasmtl import TrainedModel, predict
from atlasmtl.core.evaluate import evaluate_hierarchy_metrics, evaluate_prediction_behavior, evaluate_predictions
from atlasmtl.mapping import build_parent_conditioned_refinement_plan, discover_hotspot_parents, fit_parent_conditioned_reranker
from benchmark.methods.atlasmtl_inputs import adata_with_matrix_from_layer, resolve_atlasmtl_layer_config

DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-10_hlca_study_split_refinement_validation"
ARTIFACT_ROOT = DOSSIER_ROOT / "artifacts" / "hlca_reranker_rule_comparison"
RESULTS_ROOT = DOSSIER_ROOT / "results_summary" / "hlca_reranker_rule_comparison"
PREDICT_MANIFEST_INDEX = DOSSIER_ROOT / "manifests" / "weight_confirmation" / "predict_manifest_index.json"
BEST_CONFIG_PATH = DOSSIER_ROOT / "results_summary" / "hlca_weight_confirmation_best_config.json"
MODEL_MANIFEST = Path(
    "/tmp/atlasmtl_benchmarks/2026-03-10/hlca_study_split_weight_confirmation/train/"
    "uniform/runs/atlasmtl/atlasmtl_model_manifest.json"
)
POINTS = ("build_100000_eval10k", "predict_100000_10000")
RULE_SPECS = (("reranker_top4", 4), ("reranker_top6", 6))
TARGET_POINT = "predict_100000_10000"
LABEL_COLUMNS = ["ann_level_1", "ann_level_2", "ann_level_3", "ann_level_4", "ann_level_5"]
PARENT_LEVEL = "ann_level_4"
CHILD_LEVEL = "ann_level_5"
MIN_PARENT_CELLS = 200
UNKNOWN = "Unknown"


def _load_manifest(point: str) -> Dict[str, Any]:
    rows = json.loads(PREDICT_MANIFEST_INDEX.read_text(encoding="utf-8"))
    for row in rows:
        if row["config_name"] == "uniform" and row["point"] == point:
            return yaml.safe_load(Path(row["predict_manifest_path"]).read_text(encoding="utf-8"))
    raise FileNotFoundError(f"manifest not found for point={point}")


def _build_hierarchy_rules(reference_adata) -> Dict[str, Dict[str, Any]]:
    rules: Dict[str, Dict[str, Any]] = {}
    for parent_col, child_col in zip(LABEL_COLUMNS[:-1], LABEL_COLUMNS[1:]):
        pairs = (
            reference_adata.obs[[parent_col, child_col]]
            .dropna()
            .drop_duplicates()
            .astype(str)
            .to_records(index=False)
        )
        child_to_parent = {str(child): str(parent) for parent, child in pairs}
        rules[str(child_col)] = {"parent_col": str(parent_col), "child_to_parent": child_to_parent}
    return rules


def _edge_breakdown(pred_df: pd.DataFrame, true_df: pd.DataFrame, *, child_to_parent: Dict[str, str]) -> Dict[str, float]:
    parent_true = true_df[PARENT_LEVEL].astype(str)
    child_true = true_df[CHILD_LEVEL].astype(str)
    parent_pred = pred_df[f"pred_{PARENT_LEVEL}"].astype(str)
    child_pred = pred_df[f"pred_{CHILD_LEVEL}"].astype(str)
    implied_parent = child_pred.map(lambda x: child_to_parent.get(x) if x != UNKNOWN else UNKNOWN).fillna("MISSING_PARENT")
    parent_correct = parent_true == parent_pred
    child_correct = child_true == child_pred
    path_break = (child_pred != UNKNOWN) & (parent_pred != UNKNOWN) & (implied_parent != parent_pred)
    return {
        "parent_correct_child_wrong_rate": float((parent_correct & ~child_correct).mean()),
        "path_break_rate": float(path_break.mean()),
    }


def _subtree_breakdown(pred_df: pd.DataFrame, true_df: pd.DataFrame, *, child_to_parent: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for parent_label, parent_truth in true_df.groupby(PARENT_LEVEL):
        idx = parent_truth.index
        edge = _edge_breakdown(pred_df.loc[idx], true_df.loc[idx], child_to_parent=child_to_parent)
        rows.append({"parent_label": str(parent_label), "n_cells": int(len(idx)), **edge})
    return pd.DataFrame(rows).sort_values(
        ["parent_correct_child_wrong_rate", "n_cells", "parent_label"],
        ascending=[False, False, True],
    )


def _summarize_variant(variant_name: str, point: str, result, true_df: pd.DataFrame, hierarchy_rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    level_metrics = evaluate_predictions(result.predictions, true_df[LABEL_COLUMNS], LABEL_COLUMNS)
    behavior = evaluate_prediction_behavior(result.predictions, true_df[LABEL_COLUMNS], LABEL_COLUMNS)
    hierarchy = evaluate_hierarchy_metrics(result.predictions, true_df[LABEL_COLUMNS], LABEL_COLUMNS, hierarchy_rules=hierarchy_rules)
    edge = _edge_breakdown(result.predictions, true_df[LABEL_COLUMNS], child_to_parent=hierarchy_rules[CHILD_LEVEL]["child_to_parent"])
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


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    best_payload = json.loads(BEST_CONFIG_PATH.read_text(encoding="utf-8"))
    target_manifest = _load_manifest(TARGET_POINT)
    target_layers = resolve_atlasmtl_layer_config(target_manifest)
    ref_raw = read_h5ad(str(target_manifest["reference_h5ad"]))
    ref_input = adata_with_matrix_from_layer(ref_raw, layer_name=target_layers["reference_layer"])
    hierarchy_rules = _build_hierarchy_rules(ref_raw)
    model = TrainedModel.load(str(MODEL_MANIFEST))

    query_target_raw = read_h5ad(str(target_manifest["query_h5ad"]))
    query_target_input = adata_with_matrix_from_layer(query_target_raw, layer_name=target_layers["query_layer"])
    target_truth = query_target_raw.obs[LABEL_COLUMNS + ["study"]].astype(str).copy()
    baseline_target = predict(
        model,
        query_target_input,
        device="cpu",
        knn_correction="off",
        hierarchy_rules=hierarchy_rules,
        enforce_hierarchy=True,
        batch_size=512,
    )
    baseline_subtree = _subtree_breakdown(
        baseline_target.predictions,
        target_truth,
        child_to_parent=hierarchy_rules[CHILD_LEVEL]["child_to_parent"],
    )
    baseline_subtree.to_csv(RESULTS_ROOT / "hlca_baseline_subtree_breakdown.csv", index=False)

    variant_artifacts: Dict[str, Dict[str, str]] = {}
    ranking_lookup: Dict[str, List[str]] = {}
    comparison_rows: List[Dict[str, Any]] = []
    by_study_rows: List[Dict[str, Any]] = []

    for variant_name, top_k in RULE_SPECS:
        variant_dir = ARTIFACT_ROOT / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        ranking_df, selected_parents, ranking_summary = discover_hotspot_parents(
            baseline_subtree,
            selection_mode="topk",
            top_k=top_k,
            min_cells_per_parent=MIN_PARENT_CELLS,
        )
        ranking_lookup[variant_name] = selected_parents
        ranking_path = variant_dir / "hotspot_ranking.json"
        ranking_path.write_text(
            json.dumps(
                {
                    "selection_mode": "topk",
                    "top_k": top_k,
                    "min_cells_per_parent": MIN_PARENT_CELLS,
                    "selected_parents": selected_parents,
                    "summary": ranking_summary,
                    "ranking": ranking_df.to_dict(orient="records"),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        artifact = fit_parent_conditioned_reranker(
            model,
            ref_input,
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
                "hotspot_topk": top_k,
                "min_cells_per_parent": MIN_PARENT_CELLS,
                "best_base_config_name": best_payload["best_config_name"],
            },
        )
        artifact_path = variant_dir / f"{variant_name}.pkl"
        artifact.save(artifact_path)
        per_parent_summary_path = variant_dir / "per_parent_reranker_summary.csv"
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
            top_k=top_k,
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
        plan_path = variant_dir / "refinement_plan.json"
        plan.save(plan_path)
        variant_artifacts[variant_name] = {
            "ranking_path": str(ranking_path),
            "artifact_path": str(artifact_path),
            "plan_path": str(plan_path),
            "per_parent_summary_path": str(per_parent_summary_path),
        }

    for point in POINTS:
        manifest = _load_manifest(point)
        layer_cfg = resolve_atlasmtl_layer_config(manifest)
        query_raw = read_h5ad(str(manifest["query_h5ad"]))
        query_input = adata_with_matrix_from_layer(query_raw, layer_name=layer_cfg["query_layer"])
        truth_df = query_raw.obs[LABEL_COLUMNS + ["study"]].astype(str).copy()

        baseline_result = predict(
            model,
            query_input,
            device="cpu",
            knn_correction="off",
            hierarchy_rules=hierarchy_rules,
            enforce_hierarchy=True,
            batch_size=512,
        )
        comparison_rows.append(_summarize_variant("baseline", point, baseline_result, truth_df, hierarchy_rules))

        for variant_name, _ in RULE_SPECS:
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
                    "plan_path": variant_artifacts[variant_name]["plan_path"],
                },
            )
            comparison_rows.append(_summarize_variant(variant_name, point, refined_result, truth_df, hierarchy_rules))
            for study, study_truth in truth_df.groupby("study"):
                idx = study_truth.index
                study_pred = refined_result.predictions.loc[idx]
                study_metrics = evaluate_predictions(study_pred, study_truth[LABEL_COLUMNS], LABEL_COLUMNS)
                study_hierarchy = evaluate_hierarchy_metrics(
                    study_pred,
                    study_truth[LABEL_COLUMNS],
                    LABEL_COLUMNS,
                    hierarchy_rules=hierarchy_rules,
                )
                study_edge = _edge_breakdown(study_pred, study_truth[LABEL_COLUMNS], child_to_parent=hierarchy_rules[CHILD_LEVEL]["child_to_parent"])
                by_study_rows.append(
                    {
                        "variant_name": variant_name,
                        "point": point,
                        "study": str(study),
                        "n_cells": int(len(idx)),
                        "ann_level_5_macro_f1": study_metrics[CHILD_LEVEL]["macro_f1"],
                        "full_path_accuracy": study_hierarchy.get("full_path_accuracy", 0.0),
                        "parent_correct_child_wrong_rate": study_edge["parent_correct_child_wrong_rate"],
                    }
                )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["point", "variant_name"]).reset_index(drop=True)
    baseline_df = comparison_df[comparison_df["variant_name"] == "baseline"][
        ["point", "ann_level_5_macro_f1", "full_path_accuracy", "parent_correct_child_wrong_rate"]
    ].rename(
        columns={
            "ann_level_5_macro_f1": "ann_level_5_macro_f1_baseline",
            "full_path_accuracy": "full_path_accuracy_baseline",
            "parent_correct_child_wrong_rate": "parent_correct_child_wrong_rate_baseline",
        }
    )
    comparison_df = comparison_df.merge(baseline_df, on="point", how="left")
    comparison_df["delta_vs_baseline_ann_level_5_macro_f1"] = comparison_df["ann_level_5_macro_f1"] - comparison_df["ann_level_5_macro_f1_baseline"]
    comparison_df["delta_vs_baseline_full_path_accuracy"] = comparison_df["full_path_accuracy"] - comparison_df["full_path_accuracy_baseline"]
    comparison_df["delta_vs_baseline_parent_correct_child_wrong_rate"] = (
        comparison_df["parent_correct_child_wrong_rate"] - comparison_df["parent_correct_child_wrong_rate_baseline"]
    )
    comparison_df.to_csv(RESULTS_ROOT / "hlca_reranker_rule_comparison.csv", index=False)
    pd.DataFrame(by_study_rows).sort_values(["point", "study", "variant_name"]).to_csv(
        RESULTS_ROOT / "hlca_reranker_rule_by_study.csv",
        index=False,
    )

    guardrail_rows: List[Dict[str, Any]] = []
    target_base = comparison_df[(comparison_df["point"] == TARGET_POINT) & (comparison_df["variant_name"] == "baseline")].iloc[0]
    for variant_name, _ in RULE_SPECS:
        target_variant = comparison_df[
            (comparison_df["point"] == TARGET_POINT) & (comparison_df["variant_name"] == variant_name)
        ].iloc[0]
        passed = bool(
            float(target_variant["ann_level_5_macro_f1"]) >= float(target_base["ann_level_5_macro_f1"])
            and float(target_variant["full_path_accuracy"]) >= float(target_base["full_path_accuracy"])
            and float(target_variant["parent_correct_child_wrong_rate"]) <= float(target_base["parent_correct_child_wrong_rate"])
        )
        payload = {
            "variant_name": variant_name,
            "passed": passed,
            "selected_parents": ranking_lookup[variant_name],
            "baseline": {
                "ann_level_5_macro_f1": float(target_base["ann_level_5_macro_f1"]),
                "full_path_accuracy": float(target_base["full_path_accuracy"]),
                "parent_correct_child_wrong_rate": float(target_base["parent_correct_child_wrong_rate"]),
            },
            "candidate": {
                "ann_level_5_macro_f1": float(target_variant["ann_level_5_macro_f1"]),
                "full_path_accuracy": float(target_variant["full_path_accuracy"]),
                "parent_correct_child_wrong_rate": float(target_variant["parent_correct_child_wrong_rate"]),
            },
        }
        guardrail_path = ARTIFACT_ROOT / variant_name / "guardrail_decision.json"
        guardrail_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        guardrail_rows.append(
            {
                "variant_name": variant_name,
                "guardrail_passed": passed,
                "selected_parent_count": len(ranking_lookup[variant_name]),
                "selected_parents": "|".join(ranking_lookup[variant_name]),
            }
        )
    pd.DataFrame(guardrail_rows).to_csv(RESULTS_ROOT / "hlca_reranker_rule_guardrails.csv", index=False)

    top4 = comparison_df[(comparison_df["point"] == TARGET_POINT) & (comparison_df["variant_name"] == "reranker_top4")].iloc[0]
    top6 = comparison_df[(comparison_df["point"] == TARGET_POINT) & (comparison_df["variant_name"] == "reranker_top6")].iloc[0]
    verdict = (
        "HLCA promoted to positive reranker validation case"
        if any(row["guardrail_passed"] for row in guardrail_rows)
        else "HLCA retained as mixed-evidence stress test"
    )
    summary_lines = [
        "# HLCA Reranker Rule Comparison",
        "",
        f"- best base config: `{best_payload['best_config_name']}`",
        f"- compared rules: `top4` vs `top6` on `{PARENT_LEVEL} -> {CHILD_LEVEL}`",
        f"- minimum parent cells: `{MIN_PARENT_CELLS}`",
        "",
        "## Target point",
        "",
        f"- baseline: macro_f1 `{float(target_base['ann_level_5_macro_f1']):.6f}`, full_path `{float(target_base['full_path_accuracy']):.4f}`, parent_correct_child_wrong `{float(target_base['parent_correct_child_wrong_rate']):.4f}`",
        f"- reranker_top4: macro_f1 `{float(top4['ann_level_5_macro_f1']):.6f}`, full_path `{float(top4['full_path_accuracy']):.4f}`, parent_correct_child_wrong `{float(top4['parent_correct_child_wrong_rate']):.4f}`",
        f"- reranker_top6: macro_f1 `{float(top6['ann_level_5_macro_f1']):.6f}`, full_path `{float(top6['full_path_accuracy']):.4f}`, parent_correct_child_wrong `{float(top6['parent_correct_child_wrong_rate']):.4f}`",
        "",
        "## Verdict",
        "",
        f"- {verdict}",
        "",
        "## Outputs",
        "",
        "- `hlca_reranker_rule_comparison.csv`",
        "- `hlca_reranker_rule_by_study.csv`",
        "- `hlca_reranker_rule_guardrails.csv`",
    ]
    (RESULTS_ROOT / "hlca_reranker_rule_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
