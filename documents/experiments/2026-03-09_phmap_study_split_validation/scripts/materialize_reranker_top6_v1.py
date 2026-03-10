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
from atlasmtl.mapping import fit_parent_conditioned_reranker
from benchmark.methods.atlasmtl_inputs import adata_with_matrix_from_layer, resolve_atlasmtl_layer_config

DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
ARTIFACT_DIR = DOSSIER_ROOT / "artifacts" / "reranker_top6_v1"
RESULTS_DIR = DOSSIER_ROOT / "results_summary" / "reranker_top6_v1"
PHASE2_MANIFEST_INDEX = DOSSIER_ROOT / "manifests" / "phase2_seed" / "predict_manifest_index.json"
PHASE3_SUBTREE_PATH = DOSSIER_ROOT / "results_summary" / "phase3_tradeoff_subtree_breakdown.csv"
MODEL_MANIFEST = Path(
    "/tmp/atlasmtl_benchmarks/2026-03-09/phmap_study_split_phase2_seed/train/"
    "lv4strong_plus_class_weight/seed_2026/runs/atlasmtl/atlasmtl_model_manifest.json"
)
POINTS = ("build_100000_eval10k", "predict_100000_10000")
LABEL_COLUMNS = ["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"]
TARGET_POINT = "predict_100000_10000"
TARGET_SEED = 2026
TARGET_CONFIG = "lv4strong_plus_class_weight"
UNKNOWN = "Unknown"


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
        child_to_parent = {str(child): str(parent) for parent, child in pairs}
        rules[str(child_col)] = {
            "parent_col": str(parent_col),
            "child_to_parent": child_to_parent,
        }
    return rules


def _load_hotspot_ranking() -> pd.DataFrame:
    df = pd.read_csv(PHASE3_SUBTREE_PATH)
    df = df[
        (df["config_name"] == "lv4strong_plus_class_weight")
        & (df["point"] == TARGET_POINT)
        & (df["hierarchy_setting"] == "on")
    ].copy()
    df["contribution"] = df["parent_correct_child_wrong_rate"] * df["n_cells"]
    return df.sort_values(
        ["contribution", "parent_correct_child_wrong_rate", "n_cells"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _edge_breakdown(pred_df: pd.DataFrame, true_df: pd.DataFrame, *, parent_col: str, child_col: str, child_to_parent):
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


def _summarize_variant(*, variant_name: str, result, truth_df: pd.DataFrame, hierarchy_rules):
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
        "variant_name": variant_name,
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
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ranking_df = _load_hotspot_ranking()
    hotspot_top6 = ranking_df["parent_label"].astype(str).head(6).tolist()
    hotspot_payload = {
        "selection_source": "Phase 3 baseline",
        "selection_point": f"{TARGET_POINT} + hierarchy_on",
        "selection_score": "parent_correct_child_wrong_rate * n_cells",
        "top6": hotspot_top6,
        "ranking": ranking_df[
            ["parent_label", "n_cells", "parent_correct_child_wrong_rate", "contribution"]
        ].to_dict(orient="records"),
    }
    (ARTIFACT_DIR / "hotspot_ranking.json").write_text(
        json.dumps(hotspot_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    model = TrainedModel.load(str(MODEL_MANIFEST))
    target_manifest = _load_manifest(TARGET_POINT)
    layer_cfg = resolve_atlasmtl_layer_config(target_manifest)
    ref_raw = read_h5ad(str(target_manifest["reference_h5ad"]))
    ref_model_input = adata_with_matrix_from_layer(ref_raw, layer_name=layer_cfg["reference_layer"])
    hierarchy_rules = _build_hierarchy_rules(ref_raw, LABEL_COLUMNS)

    artifact = fit_parent_conditioned_reranker(
        model,
        ref_model_input,
        parent_level="anno_lv3",
        child_level="anno_lv4",
        hotspot_parents=hotspot_top6,
        hierarchy_rules=hierarchy_rules,
        batch_size=512,
        device="cpu",
        selection_metadata={
            "selection_source": "Phase 3 baseline",
            "selection_point": f"{TARGET_POINT} + hierarchy_on",
            "selection_score": "parent_correct_child_wrong_rate * n_cells",
            "hotspot_topk": 6,
        },
    )
    artifact_path = ARTIFACT_DIR / "parent_conditioned_reranker_top6.pkl"
    artifact.save(artifact_path)
    pd.DataFrame(artifact.per_parent_summary).to_csv(ARTIFACT_DIR / "per_parent_reranker_summary.csv", index=False)

    comparison_rows: List[Dict[str, Any]] = []
    breakdown_rows: List[Dict[str, Any]] = []
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
                "artifact_path": str(artifact_path),
            },
        )
        for variant_name, result in (("baseline", base_result), ("reranker_top6_v1", refined_result)):
            summary = _summarize_variant(
                variant_name=variant_name,
                result=result,
                truth_df=truth_df,
                hierarchy_rules=hierarchy_rules,
            )
            summary["point"] = point
            comparison_rows.append(summary)
            breakdown_rows.append(
                {
                    "point": point,
                    "variant_name": variant_name,
                    "parent_col": "anno_lv3",
                    "child_col": "anno_lv4",
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

    comparison_df = pd.DataFrame(comparison_rows)
    baseline_df = comparison_df[comparison_df["variant_name"] == "baseline"][
        ["point", "anno_lv4_macro_f1", "full_path_accuracy", "parent_correct_child_wrong_rate"]
    ].rename(
        columns={
            "anno_lv4_macro_f1": "anno_lv4_macro_f1_baseline",
            "full_path_accuracy": "full_path_accuracy_baseline",
            "parent_correct_child_wrong_rate": "parent_correct_child_wrong_rate_baseline",
        }
    )
    comparison_df = comparison_df.merge(baseline_df, on="point", how="left")
    comparison_df["delta_vs_baseline_anno_lv4_macro_f1"] = (
        comparison_df["anno_lv4_macro_f1"] - comparison_df["anno_lv4_macro_f1_baseline"]
    )
    comparison_df["delta_vs_baseline_full_path_accuracy"] = (
        comparison_df["full_path_accuracy"] - comparison_df["full_path_accuracy_baseline"]
    )
    comparison_df["delta_vs_baseline_parent_correct_child_wrong_rate"] = (
        comparison_df["parent_correct_child_wrong_rate"] - comparison_df["parent_correct_child_wrong_rate_baseline"]
    )
    comparison_df.to_csv(RESULTS_DIR / "before_after_comparison.csv", index=False)
    pd.DataFrame(breakdown_rows).to_csv(RESULTS_DIR / "before_after_parent_child_breakdown.csv", index=False)

    summary_lines = [
        "# PH-Map reranker_top6 operational module v1",
        "",
        f"- model seed: `{TARGET_SEED}`",
        "- source model: `lv4strong_plus_class_weight`",
        "- refinement method: `parent_conditioned_reranker`",
        "- refinement edge: `anno_lv3 -> anno_lv4`",
        f"- hotspot parents: `{', '.join(hotspot_top6)}`",
        "- selection rule: `Phase 3 baseline`, `predict_100000_10000 + hierarchy_on`, `parent_correct_child_wrong_rate * n_cells`",
        "",
        "## Outputs",
        "",
        "- `hotspot_ranking.json`",
        "- `parent_conditioned_reranker_top6.pkl`",
        "- `parent_conditioned_reranker_top6.json`",
        "- `per_parent_reranker_summary.csv`",
        "- `before_after_comparison.csv`",
        "- `before_after_parent_child_breakdown.csv`",
    ]
    (RESULTS_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
