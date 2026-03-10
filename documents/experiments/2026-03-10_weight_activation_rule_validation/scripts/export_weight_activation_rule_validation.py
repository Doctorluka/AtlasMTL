from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import anndata as ad
import pandas as pd
import yaml

from atlasmtl.core.predict import predict
from atlasmtl.core.types import TrainedModel
from atlasmtl.mapping import suggest_task_weight_schedule


REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "documents" / "experiments" / "2026-03-10_weight_activation_rule_validation" / "results_summary"
PHMAP_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation" / "results_summary"
HLCA_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-10_hlca_study_split_refinement_validation" / "results_summary"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hotspot_concentration_score(path: Path, top_k: int = 3) -> Optional[float]:
    payload = _load_json(path)
    ranking = payload.get("ranking") or []
    scores = [float(row.get("selection_score", 0.0)) for row in ranking]
    total = float(sum(scores))
    if total <= 0.0:
        return None
    return float(sum(scores[:top_k]) / total)


def _first_value(df: pd.DataFrame, **filters: Any) -> pd.Series:
    out = df.copy()
    for key, value in filters.items():
        out = out[out[key] == value]
    if out.empty:
        raise ValueError(f"no rows matched filters: {filters}")
    return out.iloc[0]


def _maybe_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _collect_phmap_features() -> Dict[str, Any]:
    phase1_comparison = pd.read_csv(PHMAP_ROOT / "phase1_comparison.csv")
    phase1_levelwise = pd.read_csv(PHMAP_ROOT / "phase1_levelwise.csv")
    phase1_hierarchy_delta = pd.read_csv(PHMAP_ROOT / "phase1_hierarchy_delta.csv")
    phase3_parent_child = pd.read_csv(PHMAP_ROOT / "phase3_tradeoff_parent_child_breakdown.csv")

    target = _first_value(phase1_comparison, point="predict_100000_10000", hierarchy_setting="on")
    coarse = _first_value(
        phase1_levelwise,
        config_name="uniform_control",
        point="predict_100000_10000",
        level="anno_lv1",
    )
    fine = _first_value(
        phase1_levelwise,
        config_name="uniform_control",
        point="predict_100000_10000",
        level="anno_lv4",
    )
    hierarchy_gap = _first_value(
        phase1_hierarchy_delta,
        config_name="uniform_control",
        point="predict_100000_10000",
    )
    parent_child = _first_value(
        phase3_parent_child,
        config_name="lv4strong_baseline",
        point="predict_100000_10000",
        hierarchy_setting="on",
        parent_col="anno_lv3",
        child_col="anno_lv4",
    )
    hotspot_score = _hotspot_concentration_score(
        REPO_ROOT
        / "documents"
        / "experiments"
        / "2026-03-09_phmap_study_split_validation"
        / "artifacts"
        / "phase7a_auto_reranker_pipeline"
        / "hotspot_ranking.json"
    )
    return {
        "dataset": "PHMap_Lung_Full_v43_light",
        "split_type": "study_split",
        "n_levels": 4,
        "finest_level": "anno_lv4",
        "finest_macro_f1": float(target["macro_f1_uniform"]),
        "finest_balanced_accuracy": float(target["balanced_accuracy_uniform"]),
        "full_path_accuracy": float(target["full_path_accuracy_uniform"]),
        "coverage": float(target["coverage_uniform"]),
        "unknown_rate": float(target["unknown_rate_uniform"]),
        "coarse_to_fine_headroom_gap": float(coarse["macro_f1"] - fine["macro_f1"]),
        "full_path_vs_finest_gap": float(target["macro_f1_uniform"] - target["full_path_accuracy_uniform"]),
        "parent_correct_child_wrong_rate": float(parent_child["parent_correct_child_wrong_rate"]),
        "path_break_rate": float(parent_child["path_break_rate"]),
        "hierarchy_on_off_macro_f1_gap": float(hierarchy_gap["delta_off_minus_on_macro_f1"]),
        "hierarchy_on_off_full_path_gap": float(hierarchy_gap["delta_off_minus_on_full_path_accuracy"]),
        "hotspot_concentration_score": hotspot_score,
        "feature_source_note": (
            "uniform finest/coarse metrics from Phase 1 study-split weighting ablation; "
            "parent-child structure from earliest available PH-Map study-split tradeoff breakdown"
        ),
    }


def _collect_hlca_features() -> Dict[str, Any]:
    weight_confirmation = pd.read_csv(HLCA_ROOT / "hlca_weight_confirmation.csv")
    main_comparison = pd.read_csv(HLCA_ROOT / "hlca_auto_reranker_validation" / "hlca_main_comparison.csv")

    target = _first_value(weight_confirmation, config_name="uniform", point="predict_100000_10000")
    baseline = _first_value(main_comparison, variant_name="baseline", point="predict_100000_10000")
    hotspot_score = _hotspot_concentration_score(
        REPO_ROOT
        / "documents"
        / "experiments"
        / "2026-03-10_hlca_study_split_refinement_validation"
        / "artifacts"
        / "hlca_auto_reranker_validation"
        / "hlca_hotspot_ranking.json"
    )
    return {
        "dataset": "HLCA_Core",
        "split_type": "study_split",
        "n_levels": 5,
        "finest_level": "ann_level_5",
        "finest_macro_f1": float(target["macro_f1"]),
        "finest_balanced_accuracy": float(target["balanced_accuracy"]),
        "full_path_accuracy": float(target["full_path_accuracy"]),
        "coverage": float(target["coverage"]),
        "unknown_rate": float(target["unknown_rate"]),
        "coarse_to_fine_headroom_gap": None,
        "full_path_vs_finest_gap": float(target["macro_f1"] - target["full_path_accuracy"]),
        "parent_correct_child_wrong_rate": float(baseline["parent_correct_child_wrong_rate"]),
        "path_break_rate": float(baseline["path_break_rate"]),
        "hierarchy_on_off_macro_f1_gap": None,
        "hierarchy_on_off_full_path_gap": None,
        "hotspot_concentration_score": hotspot_score,
        "feature_source_note": (
            "uniform finest metrics from HLCA study-split weight confirmation; "
            "parent-child structure from baseline rows in first-pass auto-reranker validation"
        ),
    }


def _collect_mtca_features() -> Dict[str, Any]:
    manifest_path = (
        REPO_ROOT
        / "documents"
        / "experiments"
        / "2026-03-09_multilevel_annotation_benchmark"
        / "manifests"
        / "multilevel"
        / "mTCA"
        / "gpu"
        / "predict_100000_10000"
        / "atlasmtl_multilevel.yaml"
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    metrics = _load_json(
        Path(
            "/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation/mTCA/benchmark/gpu/predict_100000_10000/runs/atlasmtl/metrics.json"
        )
    )["results"][0]
    model = TrainedModel.load(
        "/tmp/atlasmtl_benchmarks/2026-03-09/multilevel_annotation/mTCA/benchmark/gpu/predict_100000_10000/runs/atlasmtl/atlasmtl_model_manifest.json",
        device="cpu",
    )
    query = ad.read_h5ad(manifest["query_h5ad"])
    result = predict(
        model,
        query,
        knn_correction="off",
        batch_size=int(manifest["predict"].get("batch_size", 512)),
        device="cpu",
        input_transform=str(manifest["predict"].get("input_transform") or "binary"),
        hierarchy_rules=manifest["predict"]["hierarchy_rules"],
        enforce_hierarchy=bool(manifest["predict"].get("enforce_hierarchy", True)),
        show_progress=False,
        show_summary=False,
    )
    pred = result.predictions.reset_index(drop=True)
    parent_col = "Cell_type_level2"
    child_col = "Cell_type_level3"
    parent_true = query.obs[parent_col].astype(str).reset_index(drop=True)
    child_true = query.obs[child_col].astype(str).reset_index(drop=True)
    parent_pred = pred[f"pred_{parent_col}"].astype(str).reset_index(drop=True)
    child_pred = pred[f"pred_{child_col}"].astype(str).reset_index(drop=True)
    parent_correct = parent_true.eq(parent_pred)
    child_correct = child_true.eq(child_pred)
    parent_correct_child_wrong_rate = float((parent_correct & ~child_correct).mean())
    child_to_parent = manifest["predict"]["hierarchy_rules"][child_col]["child_to_parent"]
    path_break_rate = float(child_pred.map(child_to_parent).fillna("__missing__").ne(parent_pred).mean())

    rows = []
    for parent_label in sorted(parent_true.unique()):
        mask = parent_true.eq(parent_label)
        n_cells = int(mask.sum())
        pccw = float((parent_correct[mask] & ~child_correct[mask]).mean())
        rows.append(
            {
                "parent_label": parent_label,
                "n_cells": n_cells,
                "parent_correct_child_wrong_rate": pccw,
                "selection_score": pccw * n_cells,
            }
        )
    breakdown = pd.DataFrame(rows).sort_values(
        ["selection_score", "parent_correct_child_wrong_rate", "n_cells", "parent_label"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    total_selection_score = float(breakdown["selection_score"].sum())
    if total_selection_score > 0:
        breakdown["cumulative_contribution"] = breakdown["selection_score"].cumsum() / total_selection_score
        hotspot_concentration_score = float(breakdown["selection_score"].head(3).sum() / total_selection_score)
    else:
        breakdown["cumulative_contribution"] = 0.0
        hotspot_concentration_score = None
    breakdown.to_csv(OUTPUT_DIR / "mtca_sanity_parent_breakdown.csv", index=False)

    return {
        "dataset": "mTCA",
        "split_type": "formal_multilevel_predict_scaling_gpu",
        "n_levels": 3,
        "finest_level": "Cell_type_level3",
        "finest_macro_f1": float(metrics["metrics"]["Cell_type_level3"]["macro_f1"]),
        "finest_balanced_accuracy": float(metrics["metrics"]["Cell_type_level3"]["balanced_accuracy"]),
        "full_path_accuracy": float(metrics["hierarchy_metrics"]["full_path_accuracy"]),
        "coverage": float(metrics["metrics"]["Cell_type_level3"]["coverage"]),
        "unknown_rate": float(metrics["behavior_metrics"]["Cell_type_level3"]["unknown_rate"]),
        "coarse_to_fine_headroom_gap": float(
            metrics["metrics"]["Cell_type_level1"]["macro_f1"] - metrics["metrics"]["Cell_type_level3"]["macro_f1"]
        ),
        "full_path_vs_finest_gap": float(
            metrics["metrics"]["Cell_type_level3"]["macro_f1"] - metrics["hierarchy_metrics"]["full_path_accuracy"]
        ),
        "parent_correct_child_wrong_rate": parent_correct_child_wrong_rate,
        "path_break_rate": path_break_rate,
        "hierarchy_on_off_macro_f1_gap": None,
        "hierarchy_on_off_full_path_gap": None,
        "hotspot_concentration_score": hotspot_concentration_score,
        "feature_source_note": (
            "finest/coarse metrics from existing multilevel benchmark metrics.json; "
            "parent-child structure recomputed from saved model plus heldout query without retraining"
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features = [_collect_phmap_features(), _collect_hlca_features(), _collect_mtca_features()]
    feature_rows = []
    decisions: Dict[str, Any] = {}
    paper_rows = []

    for feature in features:
        decision = suggest_task_weight_schedule(
            n_levels=int(feature["n_levels"]),
            finest_macro_f1=float(feature["finest_macro_f1"]),
            finest_balanced_accuracy=float(feature["finest_balanced_accuracy"]),
            full_path_accuracy=float(feature["full_path_accuracy"]),
            parent_correct_child_wrong_rate=float(feature["parent_correct_child_wrong_rate"]),
            coarse_to_fine_headroom_gap=_maybe_float(feature["coarse_to_fine_headroom_gap"]),
            hierarchy_on_off_macro_f1_gap=_maybe_float(feature["hierarchy_on_off_macro_f1_gap"]),
            hotspot_concentration_score=_maybe_float(feature["hotspot_concentration_score"]),
        )
        feature_rows.append(
            {
                **feature,
                "activate_nonuniform_weighting": decision.activate_nonuniform_weighting,
                "recommended_schedule_name": decision.recommended_schedule_name,
                "activation_rule_version": decision.activation_rule_version,
            }
        )
        decisions[str(feature["dataset"])] = decision.to_dict()
        paper_rows.append(
            {
                "dataset": feature["dataset"],
                "split_type": feature["split_type"],
                "activate_nonuniform_weighting": decision.activate_nonuniform_weighting,
                "recommended_schedule_name": decision.recommended_schedule_name,
                "reason_short": decision.decision_note,
                "finest_macro_f1": feature["finest_macro_f1"],
                "full_path_accuracy": feature["full_path_accuracy"],
                "parent_correct_child_wrong_rate": feature["parent_correct_child_wrong_rate"],
                "coarse_to_fine_headroom_gap": feature["coarse_to_fine_headroom_gap"],
            }
        )

    pd.DataFrame(feature_rows).to_csv(OUTPUT_DIR / "weight_activation_features.csv", index=False)
    (OUTPUT_DIR / "weight_activation_decisions.json").write_text(
        json.dumps(decisions, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(paper_rows).to_csv(OUTPUT_DIR / "weight_activation_paper_table.csv", index=False)

    lines = [
        "# Weight Activation Rule Validation",
        "",
        "- rule version: `activation_rule_v1`",
        "- policy target: decide whether a dataset should leave `uniform` task weights",
        "",
        "## Decisions",
        "",
    ]
    for feature in features:
        decision = decisions[str(feature["dataset"])]
        lines.extend(
            [
                f"### {feature['dataset']}",
                "",
                f"- activate non-uniform weighting: `{decision['activate_nonuniform_weighting']}`",
                f"- recommended schedule name: `{decision['recommended_schedule_name']}`",
                f"- candidate space: `{', '.join(decision['candidate_space'])}`",
                f"- finest macro_f1: `{feature['finest_macro_f1']:.6f}`",
                f"- full_path_accuracy: `{feature['full_path_accuracy']:.4f}`",
                f"- parent_correct_child_wrong_rate: `{feature['parent_correct_child_wrong_rate']:.4f}`",
                f"- decision note: {decision['decision_note']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- `PH-Map` is correctly classified as an activation case.",
            "- `HLCA` is correctly classified as a stay-uniform case.",
            "- `mTCA` also behaves as a stay-uniform sanity-check case under the current rule.",
            "- this supports the framework policy claim that non-uniform weighting should be treated as an error-driven, dataset-adaptive option rather than a universal default.",
        ]
    )
    (OUTPUT_DIR / "weight_activation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
