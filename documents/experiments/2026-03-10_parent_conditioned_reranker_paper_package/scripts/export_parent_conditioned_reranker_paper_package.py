#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
PHMAP_READY = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation" / "results_summary" / "paper_ready"
HLCA_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-10_hlca_study_split_refinement_validation"
OUT_ROOT = (
    REPO_ROOT
    / "documents"
    / "experiments"
    / "2026-03-10_parent_conditioned_reranker_paper_package"
    / "results_summary"
    / "paper_package"
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_size(path_str: str) -> int | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return int(path.stat().st_size)


def _build_main_panel_a() -> pd.DataFrame:
    phmap = _read_csv(PHMAP_READY / "paper_main_figure_panel_a.csv").copy()
    phmap["dataset_role"] = "primary_positive_case"
    phmap["guardrail_status"] = "pass"
    phmap["chapter_status"] = "final_operational_path"

    hlca_main = _read_csv(HLCA_ROOT / "results_summary" / "hlca_auto_reranker_validation" / "hlca_main_comparison.csv")
    rename_map = {
        "variant_name": "variant",
        "ann_level_5_macro_f1": "finest_level_macro_f1",
        "full_path_accuracy": "full_path_accuracy",
        "parent_correct_child_wrong_rate": "parent_correct_child_wrong_rate",
    }
    hlca = hlca_main.rename(columns=rename_map).copy()
    hlca["dataset"] = "HLCA_Core"
    hlca["display_name"] = hlca["variant"].map(
        {
            "baseline": "Best base config (uniform)",
            "auto_parent_conditioned_reranker": "Best base config + auto reranker top6",
        }
    )
    hlca["finest_level"] = "ann_level_5"
    hlca["finest_level_macro_f1_std"] = None
    hlca["full_path_accuracy_std"] = None
    hlca["parent_correct_child_wrong_rate_std"] = None
    hlca["split_type"] = "study"
    hlca["seed_mode"] = "single_seed_2026"
    hlca["dataset_role"] = "secondary_validation_case"
    hlca["chapter_status"] = hlca["variant"].map(
        {
            "baseline": "best_base_config",
            "auto_parent_conditioned_reranker": "first_pass_reranker",
        }
    )
    guardrail = _read_json(
        HLCA_ROOT / "artifacts" / "hlca_auto_reranker_validation" / "hlca_guardrail_decision.json"
    )
    hlca["guardrail_status"] = hlca["variant"].map(
        {
            "baseline": "not_applicable",
            "auto_parent_conditioned_reranker": "pass" if bool(guardrail["passed"]) else "fail",
        }
    )
    hlca = hlca[
        [
            "dataset",
            "point",
            "variant",
            "display_name",
            "finest_level",
            "finest_level_macro_f1",
            "finest_level_macro_f1_std",
            "full_path_accuracy",
            "full_path_accuracy_std",
            "parent_correct_child_wrong_rate",
            "parent_correct_child_wrong_rate_std",
            "split_type",
            "seed_mode",
            "dataset_role",
            "guardrail_status",
            "chapter_status",
        ]
    ]
    phmap = phmap[
        [
            "dataset",
            "point",
            "variant",
            "display_name",
            "finest_level",
            "finest_level_macro_f1",
            "finest_level_macro_f1_std",
            "full_path_accuracy",
            "full_path_accuracy_std",
            "parent_correct_child_wrong_rate",
            "parent_correct_child_wrong_rate_std",
            "split_type",
            "seed_mode",
            "dataset_role",
            "guardrail_status",
            "chapter_status",
        ]
    ]
    return pd.concat([phmap, hlca], ignore_index=True)


def _build_main_panel_b() -> pd.DataFrame:
    phmap = _read_csv(PHMAP_READY / "paper_main_figure_panel_b.csv").copy()
    phmap["dataset_role"] = "primary_positive_case"
    hlca_err = _read_csv(HLCA_ROOT / "results_summary" / "hlca_auto_reranker_validation" / "hlca_error_decomposition.csv")
    rows: List[Dict[str, Any]] = []
    for _, row in hlca_err.iterrows():
        for metric in ("parent_correct_child_wrong_rate", "path_break_rate"):
            rows.append(
                {
                    "dataset": "HLCA_Core",
                    "point": row["point"],
                    "variant": row["variant_name"],
                    "display_name": (
                        "Best base config (uniform)"
                        if row["variant_name"] == "baseline"
                        else "Best base config + auto reranker top6"
                    ),
                    "metric": metric,
                    "mean": float(row[metric]),
                    "std": None,
                    "dataset_role": "secondary_validation_case",
                }
            )
    return pd.concat([phmap, pd.DataFrame(rows)], ignore_index=True)


def _build_main_panel_d() -> pd.DataFrame:
    phmap = _read_csv(PHMAP_READY / "paper_main_figure_panel_d.csv").copy()
    phmap["dataset_role"] = "primary_positive_case"
    phmap["notes"] = "top6_vs_top8_n5_stability"

    hlca_weight = _read_csv(HLCA_ROOT / "results_summary" / "hlca_weight_confirmation.csv")
    hlca_uniform = hlca_weight[
        (hlca_weight["config_name"] == "uniform") & (hlca_weight["point"] == "predict_100000_10000")
    ].iloc[0]
    hlca_main = _read_csv(HLCA_ROOT / "results_summary" / "hlca_auto_reranker_validation" / "hlca_main_comparison.csv")
    hlca_reranker = hlca_main[
        (hlca_main["variant_name"] == "auto_parent_conditioned_reranker")
        & (hlca_main["point"] == "predict_100000_10000")
    ].iloc[0]
    rows = [
        {
            "dataset": "HLCA_Core",
            "variant": "uniform_base",
            "metric": "ann_level_5_macro_f1",
            "mean": float(hlca_uniform["macro_f1"]),
            "std": None,
            "dataset_role": "secondary_validation_case",
            "notes": "single_seed_2026_best_base",
        },
        {
            "dataset": "HLCA_Core",
            "variant": "uniform_base",
            "metric": "full_path_accuracy",
            "mean": float(hlca_uniform["full_path_accuracy"]),
            "std": None,
            "dataset_role": "secondary_validation_case",
            "notes": "single_seed_2026_best_base",
        },
        {
            "dataset": "HLCA_Core",
            "variant": "auto_reranker_top6",
            "metric": "ann_level_5_macro_f1",
            "mean": float(hlca_reranker["ann_level_5_macro_f1"]),
            "std": None,
            "dataset_role": "secondary_validation_case",
            "notes": "single_seed_2026_first_pass_guardrail_fail",
        },
        {
            "dataset": "HLCA_Core",
            "variant": "auto_reranker_top6",
            "metric": "full_path_accuracy",
            "mean": float(hlca_reranker["full_path_accuracy"]),
            "std": None,
            "dataset_role": "secondary_validation_case",
            "notes": "single_seed_2026_first_pass_guardrail_fail",
        },
    ]
    return pd.concat([phmap, pd.DataFrame(rows)], ignore_index=True)


def _build_s1_ablation() -> pd.DataFrame:
    phmap = _read_csv(PHMAP_READY / "paper_s1_ablation_ladder.csv").copy()
    phmap["dataset_role"] = "primary_positive_case"
    hlca_weight = _read_csv(HLCA_ROOT / "results_summary" / "hlca_weight_confirmation.csv").copy()
    hlca_weight = hlca_weight.assign(
        dataset="HLCA_Core",
        stage="weight_confirmation",
        variant=hlca_weight["config_name"],
        finest_level_macro_f1=hlca_weight["macro_f1"],
        full_path_accuracy=hlca_weight["full_path_accuracy"],
        coverage=hlca_weight["coverage"],
        parent_correct_child_wrong_rate=None,
        dataset_role="secondary_validation_case",
    )[
        [
            "dataset",
            "stage",
            "variant",
            "point",
            "finest_level_macro_f1",
            "full_path_accuracy",
            "coverage",
            "parent_correct_child_wrong_rate",
            "dataset_role",
        ]
    ]
    hlca_reranker = _read_csv(HLCA_ROOT / "results_summary" / "hlca_auto_reranker_validation" / "hlca_main_comparison.csv").copy()
    hlca_reranker = hlca_reranker.assign(
        dataset="HLCA_Core",
        stage="auto_reranker_validation",
        variant=hlca_reranker["variant_name"],
        finest_level_macro_f1=hlca_reranker["ann_level_5_macro_f1"],
        coverage=hlca_reranker["ann_level_5_coverage"],
        dataset_role="secondary_validation_case",
    )[
        [
            "dataset",
            "stage",
            "variant",
            "point",
            "finest_level_macro_f1",
            "full_path_accuracy",
            "coverage",
            "parent_correct_child_wrong_rate",
            "dataset_role",
        ]
    ]
    return pd.concat([phmap, hlca_weight, hlca_reranker], ignore_index=True)


def _build_s2_stability() -> pd.DataFrame:
    phmap = _read_csv(PHMAP_READY / "paper_s2_stability.csv").copy()
    phmap["dataset_role"] = "primary_positive_case"
    hlca_weight = _read_csv(HLCA_ROOT / "results_summary" / "hlca_weight_confirmation.csv")
    hlca_main = _read_csv(HLCA_ROOT / "results_summary" / "hlca_auto_reranker_validation" / "hlca_main_comparison.csv")
    target_uniform = hlca_weight[(hlca_weight["config_name"] == "uniform") & (hlca_weight["point"] == "predict_100000_10000")].iloc[0]
    target_reranker = hlca_main[
        (hlca_main["variant_name"] == "auto_parent_conditioned_reranker")
        & (hlca_main["point"] == "predict_100000_10000")
    ].iloc[0]
    rows = [
        {
            "dataset": "HLCA_Core",
            "comparison_family": "current_status",
            "variant": "uniform_base",
            "point": "predict_100000_10000",
            "finest_level_macro_f1_mean": float(target_uniform["macro_f1"]),
            "finest_level_macro_f1_std": None,
            "full_path_accuracy_mean": float(target_uniform["full_path_accuracy"]),
            "full_path_accuracy_std": None,
            "parent_correct_child_wrong_rate_mean": None,
            "parent_correct_child_wrong_rate_std": None,
            "dataset_role": "secondary_validation_case",
            "notes": "single_seed_2026_best_base",
        },
        {
            "dataset": "HLCA_Core",
            "comparison_family": "current_status",
            "variant": "auto_reranker_top6",
            "point": "predict_100000_10000",
            "finest_level_macro_f1_mean": float(target_reranker["ann_level_5_macro_f1"]),
            "finest_level_macro_f1_std": None,
            "full_path_accuracy_mean": float(target_reranker["full_path_accuracy"]),
            "full_path_accuracy_std": None,
            "parent_correct_child_wrong_rate_mean": float(target_reranker["parent_correct_child_wrong_rate"]),
            "parent_correct_child_wrong_rate_std": None,
            "dataset_role": "secondary_validation_case",
            "notes": "single_seed_2026_first_pass_guardrail_fail",
        },
    ]
    return pd.concat([phmap, pd.DataFrame(rows)], ignore_index=True)


def _build_s2_by_group() -> pd.DataFrame:
    phmap = _read_csv(PHMAP_READY / "paper_s2_by_group.csv").copy()
    phmap["dataset"] = "PHMap_Lung_Full_v43_light"
    phmap["group_type"] = "study"
    hlca = _read_csv(HLCA_ROOT / "results_summary" / "hlca_auto_reranker_validation" / "hlca_by_study.csv").copy()
    hlca["seed"] = 2026
    hlca = hlca.rename(
        columns={
            "study": "group",
            "ann_level_5_macro_f1": "finest_level_macro_f1",
            "parent_correct_child_wrong_rate": "parent_correct_child_wrong_rate",
        }
    )
    hlca["dataset"] = "HLCA_Core"
    hlca["group_type"] = "study"
    phmap = phmap.rename(columns={"study": "group", "anno_lv4_macro_f1": "finest_level_macro_f1"})
    common = ["dataset", "seed", "variant_name", "group_type", "group", "finest_level_macro_f1", "full_path_accuracy", "parent_correct_child_wrong_rate"]
    return pd.concat([phmap[common], hlca[common]], ignore_index=True)


def _build_s3_rule_comparison() -> pd.DataFrame:
    phmap_rules = _read_csv(PHMAP_READY / "paper_s3_hotspot_rule_comparison.csv").copy()
    phmap_rules["dataset"] = "PHMap_Lung_Full_v43_light"
    phmap_rules["rule_status"] = phmap_rules["variant_name"].map(
        lambda x: "default" if x == "top8" else ("current_previous_default" if x == "top6" else "ablation")
    )

    hlca_ranking = _read_json(HLCA_ROOT / "artifacts" / "hlca_auto_reranker_validation" / "hlca_hotspot_ranking.json")
    hlca_rows = []
    for row in hlca_ranking["ranking"]:
        hlca_rows.append(
            {
                "dataset": "HLCA_Core",
                "variant_name": "top6",
                "selected_parents": "|".join(hlca_ranking["selected_parents"]),
                "point": "predict_100000_10000",
                "ann_level_5_macro_f1": None,
                "full_path_accuracy": None,
                "parent_correct_child_wrong_rate": None,
                "selection_score": float(row["selection_score"]),
                "parent_label": row["parent_label"],
                "n_cells": int(row["n_cells"]),
                "cumulative_contribution": float(row["cumulative_contribution"]),
                "rule_status": "first_pass_candidate",
            }
        )
    return pd.concat([phmap_rules, pd.DataFrame(hlca_rows)], ignore_index=True, sort=False)


def _build_s4_internalization() -> pd.DataFrame:
    phmap = _read_csv(PHMAP_READY / "paper_s4_internalization_branch.csv").copy()
    phmap["dataset"] = "PHMap_Lung_Full_v43_light"
    phmap["branch_status"] = "research_branch"
    return phmap


def _build_operational_overhead() -> pd.DataFrame:
    phmap = _read_csv(PHMAP_READY / "paper_operational_overhead.csv").copy()
    hlca_main = _read_csv(HLCA_ROOT / "results_summary" / "hlca_auto_reranker_validation" / "hlca_main_comparison.csv")
    hlca_auto = hlca_main[
        (hlca_main["variant_name"] == "auto_parent_conditioned_reranker")
        & (hlca_main["point"] == "predict_100000_10000")
    ].iloc[0]
    hlca_artifact = HLCA_ROOT / "artifacts" / "hlca_auto_reranker_validation" / "hlca_parent_conditioned_reranker_top6.pkl"
    hlca_row = {
        "dataset": "HLCA_Core",
        "variant": "auto_parent_conditioned_reranker_top6_first_pass",
        "hotspot_ranking_path": str(HLCA_ROOT / "artifacts" / "hlca_auto_reranker_validation" / "hlca_hotspot_ranking.json"),
        "refinement_plan_path": str(HLCA_ROOT / "artifacts" / "hlca_auto_reranker_validation" / "hlca_refinement_plan.json"),
        "reranker_artifact_path": str(hlca_artifact),
        "guardrail_decision_path": str(HLCA_ROOT / "artifacts" / "hlca_auto_reranker_validation" / "hlca_guardrail_decision.json"),
        "fit_summary_path": str(HLCA_ROOT / "artifacts" / "hlca_auto_reranker_validation" / "hlca_per_parent_reranker_summary.csv"),
        "artifact_size_bytes": _artifact_size(str(hlca_artifact)),
        "reranker_fit_time_seconds": float(hlca_auto["reranker_fit_seconds"]),
        "reranker_apply_time_seconds": float(hlca_auto["reranker_apply_seconds"]),
        "delta_runtime_seconds": None,
        "delta_storage_bytes": None,
    }
    return pd.concat([phmap, pd.DataFrame([hlca_row])], ignore_index=True)


def _build_design_summary() -> str:
    return "\n".join(
        [
            "# Parent-Conditioned Reranker Chapter Design Summary",
            "",
            "## Chapter positioning",
            "",
            "This chapter is framed as an optional structured refinement module for difficult deep-hierarchy cases rather than as a second core AtlasMTL model.",
            "",
            "## Core claim under evaluation",
            "",
            "For difficult deep-hierarchy reference mapping, error-driven parent-conditioned hotspot reranking can improve finest-level annotation and full-path hierarchy recovery beyond the best base multi-level AtlasMTL configuration.",
            "",
            "## Experimental design",
            "",
            "### PH-Map",
            "",
            "- Rebuild the evaluation around a stricter `study`-isolated split from raw PH-Map reference data.",
            "- Confirm that finest-level task emphasis and finest-head class weighting are the highest-ROI base-model improvements.",
            "- Diagnose the remaining tradeoff through parent-child error decomposition.",
            "- Reject naive local fixes such as hotspot thresholding and shared temperature scaling.",
            "- Promote the auto parent-conditioned reranker to the operational path and select the default hotspot rule through multi-seed `top6` vs `top8` confirmation.",
            "",
            "### HLCA",
            "",
            "- Rebuild HLCA from raw `hlca_clean.h5ad` using a `study`-grouped split instead of reusing a legacy benchmark subset.",
            "- Because HLCA has five annotation levels, run a dataset-specific 5D weighting confirmation rather than inheriting PH-Map weights.",
            "- Use the winning HLCA base configuration to test whether the AutoHotspot reranker mechanism transfers to a second deep-hierarchy dataset.",
            "",
            "## Figure plan",
            "",
            "- Main Figure Panel A: final variant comparison.",
            "- Main Figure Panel B: error-mode comparison.",
            "- Main Figure Panel C: method and artifact schematic support.",
            "- Main Figure Panel D: stability/default-rule support.",
            "- Supplementary S1-S4: ablation ladder, stability/by-group, hotspot-rule comparison, train-time internalization branch.",
        ]
    ) + "\n"


def _build_current_summary() -> str:
    return "\n".join(
        [
            "# Current Results Summary For Expert Discussion",
            "",
            "## PH-Map",
            "",
            "- Final operational path is now fixed as `lv4strong + per-class weighting + auto parent-conditioned reranker_top8`.",
            "- PH-Map provides the finalized positive hard-case result.",
            "- On `predict_100000_10000 + hierarchy_on`, the paper-ready mean results are:",
            "  - base + class weighting: macro_f1 `0.587177 ± 0.005695`, full_path `0.43836 ± 0.01042`, parent_correct_child_wrong `0.12348 ± 0.00986`",
            "  - + auto reranker_top8: macro_f1 `0.588557 ± 0.002093`, full_path `0.47216 ± 0.00453`, parent_correct_child_wrong `0.08926 ± 0.00310`",
            "- `top8` has passed the default-rule confirmation and replaces `top6` as the PH-Map default hotspot rule.",
            "",
            "## HLCA",
            "",
            "- HLCA `study`-split preprocessing and weighting confirmation are complete.",
            "- HLCA does not inherit the PH-Map finest-level upweighting schedule; `uniform` is currently the best base configuration.",
            "- First-pass auto reranker validation on `ann_level_4 -> ann_level_5` yields mixed evidence.",
            "- On `predict_100000_10000 + hierarchy_on`:",
            "  - baseline uniform: macro_f1 `0.688732`, full_path `0.8239`, parent_correct_child_wrong `0.0334`",
            "  - + auto reranker_top6: macro_f1 `0.693015`, full_path `0.8200`, parent_correct_child_wrong `0.0371`",
            "- HLCA therefore improves finest-level macro-F1 but fails the PH-Map-style guardrail because full-path declines and the main error mode worsens.",
            "",
            "## Current paper interpretation",
            "",
            "- PH-Map is a strong positive case for the chapter claim.",
            "- HLCA currently supports the dataset-specific weighting claim, but only provides mixed first-pass evidence for reranker transfer.",
            "- At the moment, the chapter can robustly claim a positive operational module on PH-Map and a nontrivial second-dataset stress test on HLCA.",
            "- The remaining discussion question is whether HLCA needs another targeted reranker refinement round before it is presented as a second positive validation dataset.",
        ]
    ) + "\n"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    outputs = {
        "paper_main_figure_panel_a.csv": _build_main_panel_a(),
        "paper_main_figure_panel_b.csv": _build_main_panel_b(),
        "paper_main_figure_panel_d.csv": _build_main_panel_d(),
        "paper_s1_ablation_ladder.csv": _build_s1_ablation(),
        "paper_s2_stability.csv": _build_s2_stability(),
        "paper_s2_by_group.csv": _build_s2_by_group(),
        "paper_s3_hotspot_rule_comparison.csv": _build_s3_rule_comparison(),
        "paper_s4_internalization_branch.csv": _build_s4_internalization(),
        "paper_operational_overhead.csv": _build_operational_overhead(),
    }
    for name, df in outputs.items():
        df.to_csv(OUT_ROOT / name, index=False)

    design_summary = _build_design_summary()
    current_summary = _build_current_summary()
    (OUT_ROOT / "paper_experiment_design_summary.md").write_text(design_summary, encoding="utf-8")
    (OUT_ROOT / "paper_current_results_summary.md").write_text(current_summary, encoding="utf-8")

    package_manifest = {
        "package_dir": str(OUT_ROOT),
        "datasets": ["PHMap_Lung_Full_v43_light", "HLCA_Core"],
        "main_outputs": list(outputs.keys()),
        "supporting_markdown": [
            "paper_experiment_design_summary.md",
            "paper_current_results_summary.md",
        ],
        "phmap_status": "finalized_positive_operational_case",
        "hlca_status": "first_pass_mixed_validation_case",
    }
    (OUT_ROOT / "paper_package_manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
