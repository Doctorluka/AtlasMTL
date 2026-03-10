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
from atlasmtl.mapping import discover_hotspot_parents, fit_parent_conditioned_reranker
from benchmark.methods.atlasmtl_inputs import adata_with_matrix_from_layer, resolve_atlasmtl_layer_config

DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
RESULTS_DIR = DOSSIER_ROOT / "results_summary" / "phase7b_hotspot_selection_comparison"
PHASE2_MANIFEST_INDEX = DOSSIER_ROOT / "manifests" / "phase2_seed" / "predict_manifest_index.json"
PHASE3_SUBTREE_PATH = DOSSIER_ROOT / "results_summary" / "phase3_tradeoff_subtree_breakdown.csv"
MODEL_MANIFEST = Path(
    "/tmp/atlasmtl_benchmarks/2026-03-09/phmap_study_split_phase2_seed/train/"
    "lv4strong_plus_class_weight/seed_2026/runs/atlasmtl/atlasmtl_model_manifest.json"
)
POINTS = ("build_100000_eval10k", "predict_100000_10000")
LABEL_COLUMNS = ["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"]
TARGET_POINT = "predict_100000_10000"
TARGET_CONFIG = "lv4strong_plus_class_weight"
TARGET_SEED = 2026
UNKNOWN = "Unknown"
SELECTION_SPECS: List[Tuple[str, Dict[str, Any]]] = [
    ("top4", {"selection_mode": "topk", "top_k": 4}),
    ("top6", {"selection_mode": "topk", "top_k": 6}),
    ("top8", {"selection_mode": "topk", "top_k": 8}),
    ("cumulative_50pct", {"selection_mode": "cumulative_contribution", "cumulative_target": 0.5}),
    ("cumulative_60pct", {"selection_mode": "cumulative_contribution", "cumulative_target": 0.6}),
]


def _load_manifest(point: str) -> Dict[str, Any]:
    index = json.loads(PHASE2_MANIFEST_INDEX.read_text(encoding="utf-8"))
    for row in index:
        if row["config_name"] == TARGET_CONFIG and int(row["seed"]) == TARGET_SEED and row["point"] == point:
            return yaml.safe_load(Path(row["predict_manifest_path"]).read_text(encoding="utf-8"))
    raise FileNotFoundError(f"manifest not found for config={TARGET_CONFIG}, seed={TARGET_SEED}, point={point}")


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


def _load_subtree_breakdown() -> pd.DataFrame:
    df = pd.read_csv(PHASE3_SUBTREE_PATH)
    return df[
        (df["config_name"] == "lv4strong_plus_class_weight")
        & (df["point"] == TARGET_POINT)
        & (df["hierarchy_setting"] == "on")
    ].copy()


def _edge_breakdown(pred_df: pd.DataFrame, true_df: pd.DataFrame, *, parent_col: str, child_col: str, child_to_parent):
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


def _summarize(point: str, result, truth_df: pd.DataFrame, hierarchy_rules) -> Dict[str, Any]:
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
        "point": point,
        "anno_lv4_macro_f1": level_metrics["anno_lv4"]["macro_f1"],
        "anno_lv4_balanced_accuracy": level_metrics["anno_lv4"]["balanced_accuracy"],
        "anno_lv4_coverage": level_metrics["anno_lv4"]["coverage"],
        "anno_lv4_unknown_rate": behavior["anno_lv4"]["unknown_rate"],
        "full_path_accuracy": hierarchy.get("full_path_accuracy", 0.0),
        "full_path_coverage": hierarchy.get("full_path_coverage", 0.0),
        "mean_path_consistency_rate": hierarchy["edges"]["anno_lv4"]["path_consistency_rate"],
        **edge,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = TrainedModel.load(str(MODEL_MANIFEST))
    target_manifest = _load_manifest(TARGET_POINT)
    target_layer_cfg = resolve_atlasmtl_layer_config(target_manifest)
    ref_raw = read_h5ad(str(target_manifest["reference_h5ad"]))
    ref_model_input = adata_with_matrix_from_layer(ref_raw, layer_name=target_layer_cfg["reference_layer"])
    hierarchy_rules = _build_hierarchy_rules(ref_raw, LABEL_COLUMNS)
    subtree_df = _load_subtree_breakdown()

    baseline_rows: List[Dict[str, Any]] = []
    for point in POINTS:
        manifest = _load_manifest(point)
        layer_cfg = resolve_atlasmtl_layer_config(manifest)
        query_raw = read_h5ad(str(manifest["query_h5ad"]))
        query_model_input = adata_with_matrix_from_layer(query_raw, layer_name=layer_cfg["query_layer"])
        truth_df = query_raw.obs[LABEL_COLUMNS].astype(str).copy()
        base_result = predict(
            model,
            query_model_input,
            device="cpu",
            knn_correction="off",
            hierarchy_rules=hierarchy_rules,
            enforce_hierarchy=True,
            batch_size=512,
        )
        baseline_rows.append({"variant_name": "baseline", "selected_parents": "", **_summarize(point, base_result, truth_df, hierarchy_rules)})
    baseline_df = pd.DataFrame(baseline_rows)

    result_rows: List[Dict[str, Any]] = []
    selection_manifest_rows: List[Dict[str, Any]] = []
    for variant_name, spec in SELECTION_SPECS:
        ranking_df, selected_parents, summary = discover_hotspot_parents(
            subtree_df,
            min_cells_per_parent=200,
            **spec,
        )
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
                "selection_source": "Phase 3 baseline",
                "selection_point": f"{TARGET_POINT} + hierarchy_on",
                "selection_score": "parent_correct_child_wrong_rate * n_cells",
                **summary,
            },
        )
        selection_manifest_rows.append(
            {
                "variant_name": variant_name,
                "selection_mode": summary["selection_mode"],
                "selected_parent_count": summary["selected_parent_count"],
                "selected_parents": "|".join(selected_parents),
                "selected_selection_score": summary["selected_selection_score"],
            }
        )
        for point in POINTS:
            manifest = _load_manifest(point)
            layer_cfg = resolve_atlasmtl_layer_config(manifest)
            query_raw = read_h5ad(str(manifest["query_h5ad"]))
            query_model_input = adata_with_matrix_from_layer(query_raw, layer_name=layer_cfg["query_layer"])
            truth_df = query_raw.obs[LABEL_COLUMNS].astype(str).copy()
            refined_result = predict(
                model,
                query_model_input,
                device="cpu",
                knn_correction="off",
                hierarchy_rules=hierarchy_rules,
                enforce_hierarchy=True,
                batch_size=512,
                refinement_config={
                    "method": "parent_conditioned_reranker",
                    "artifact": artifact,
                },
            )
            result_rows.append(
                {
                    "variant_name": variant_name,
                    "selected_parents": "|".join(selected_parents),
                    **_summarize(point, refined_result, truth_df, hierarchy_rules),
                }
            )

    result_df = pd.DataFrame(result_rows).merge(
        baseline_df[
            [
                "point",
                "anno_lv4_macro_f1",
                "full_path_accuracy",
                "parent_correct_child_wrong_rate",
            ]
        ].rename(
            columns={
                "anno_lv4_macro_f1": "baseline_anno_lv4_macro_f1",
                "full_path_accuracy": "baseline_full_path_accuracy",
                "parent_correct_child_wrong_rate": "baseline_parent_correct_child_wrong_rate",
            }
        ),
        on="point",
        how="left",
    )
    top6_target = result_df[
        (result_df["variant_name"] == "top6") & (result_df["point"] == TARGET_POINT)
    ].iloc[0]
    result_df["delta_vs_baseline_macro_f1"] = (
        result_df["anno_lv4_macro_f1"] - result_df["baseline_anno_lv4_macro_f1"]
    )
    result_df["delta_vs_baseline_full_path_accuracy"] = (
        result_df["full_path_accuracy"] - result_df["baseline_full_path_accuracy"]
    )
    result_df["delta_vs_baseline_parent_correct_child_wrong_rate"] = (
        result_df["parent_correct_child_wrong_rate"] - result_df["baseline_parent_correct_child_wrong_rate"]
    )
    result_df["delta_vs_top6_macro_f1"] = result_df["anno_lv4_macro_f1"] - float(top6_target["anno_lv4_macro_f1"])
    result_df["delta_vs_top6_full_path_accuracy"] = (
        result_df["full_path_accuracy"] - float(top6_target["full_path_accuracy"])
    )
    result_df["delta_vs_top6_parent_correct_child_wrong_rate"] = (
        result_df["parent_correct_child_wrong_rate"] - float(top6_target["parent_correct_child_wrong_rate"])
    )
    result_df.to_csv(RESULTS_DIR / "phase7b_hotspot_selection_comparison.csv", index=False)
    pd.DataFrame(selection_manifest_rows).to_csv(RESULTS_DIR / "phase7b_selection_manifest.csv", index=False)

    target_df = result_df[result_df["point"] == TARGET_POINT].copy()
    target_df = target_df.sort_values(
        ["full_path_accuracy", "parent_correct_child_wrong_rate", "anno_lv4_macro_f1", "variant_name"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    recommended = target_df.iloc[0]["variant_name"] if not target_df.empty else "top6"

    summary_lines = [
        "# Phase 7B Hotspot Selection Comparison",
        "",
        "- source model: `lv4strong_plus_class_weight`",
        "- selection source: `Phase 3 baseline`",
        "- scoring rule: `parent_correct_child_wrong_rate * n_cells`",
        f"- recommended default: `{recommended}`",
        "",
        "## Compared rules",
        "",
        "- `top4`",
        "- `top6`",
        "- `top8`",
        "- `cumulative_50pct`",
        "- `cumulative_60pct`",
    ]
    (RESULTS_DIR / "phase7b_hotspot_selection_comparison.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
