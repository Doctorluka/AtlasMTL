#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
RESULTS_ROOT = DOSSIER_ROOT / "results_summary"
ARTIFACT_ROOT = DOSSIER_ROOT / "artifacts"
OUT_DIR = RESULTS_ROOT / "paper_ready"


def _load_csv(rel_path: str) -> pd.DataFrame:
    path = DOSSIER_ROOT / rel_path
    return pd.read_csv(path)


def _load_json(rel_path: str) -> Dict:
    path = DOSSIER_ROOT / rel_path
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_row(
    *,
    dataset: str,
    point: str,
    variant: str,
    display_name: str,
    finest_level: str,
    macro_f1: float,
    macro_f1_std: float | None,
    full_path_accuracy: float,
    full_path_accuracy_std: float | None,
    parent_correct_child_wrong_rate: float,
    parent_correct_child_wrong_rate_std: float | None,
    split_type: str,
    seed_mode: str,
) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "point": point,
        "variant": variant,
        "display_name": display_name,
        "finest_level": finest_level,
        "finest_level_macro_f1": macro_f1,
        "finest_level_macro_f1_std": macro_f1_std,
        "full_path_accuracy": full_path_accuracy,
        "full_path_accuracy_std": full_path_accuracy_std,
        "parent_correct_child_wrong_rate": parent_correct_child_wrong_rate,
        "parent_correct_child_wrong_rate_std": parent_correct_child_wrong_rate_std,
        "split_type": split_type,
        "seed_mode": seed_mode,
    }


def export_main_panels() -> None:
    phase6a = _load_csv("results_summary/phase6a_seed_summary.csv")
    phase7c = _load_csv("results_summary/phase7c_top6_vs_top8/phase7c_top6_vs_top8_seed_summary.csv")
    phase7c_seed = _load_csv("results_summary/phase7c_top6_vs_top8/phase7c_top6_vs_top8_seed_stability.csv")

    panel_a_rows: List[Dict[str, object]] = []
    for point in ("build_100000_eval10k", "predict_100000_10000"):
        base = phase6a[(phase6a["variant_name"] == "baseline") & (phase6a["point"] == point)].iloc[0]
        reranker = phase7c[(phase7c["variant_name"] == "reranker_top8") & (phase7c["point"] == point)].iloc[0]
        panel_a_rows.append(
            _variant_row(
                dataset="PHMap_Lung_Full_v43_light",
                point=point,
                variant="base_plus_class_weight",
                display_name="Base + class weighting",
                finest_level="anno_lv4",
                macro_f1=float(base["anno_lv4_macro_f1_mean"]),
                macro_f1_std=float(base["anno_lv4_macro_f1_std"]),
                full_path_accuracy=float(base["full_path_accuracy_mean"]),
                full_path_accuracy_std=float(base["full_path_accuracy_std"]),
                parent_correct_child_wrong_rate=float(base["parent_correct_child_wrong_rate_mean"]),
                parent_correct_child_wrong_rate_std=float(base["parent_correct_child_wrong_rate_std"]),
                split_type="study",
                seed_mode="seed_mean_n5",
            )
        )
        panel_a_rows.append(
            _variant_row(
                dataset="PHMap_Lung_Full_v43_light",
                point=point,
                variant="auto_reranker_top8",
                display_name="Base + class weighting + auto reranker top8",
                finest_level="anno_lv4",
                macro_f1=float(reranker["anno_lv4_macro_f1_mean"]),
                macro_f1_std=float(reranker["anno_lv4_macro_f1_std"]),
                full_path_accuracy=float(reranker["full_path_accuracy_mean"]),
                full_path_accuracy_std=float(reranker["full_path_accuracy_std"]),
                parent_correct_child_wrong_rate=float(reranker["parent_correct_child_wrong_rate_mean"]),
                parent_correct_child_wrong_rate_std=float(reranker["parent_correct_child_wrong_rate_std"]),
                split_type="study",
                seed_mode="seed_mean_n5",
            )
        )

    panel_a = pd.DataFrame(panel_a_rows)

    panel_b_base = phase7c_seed[
        (phase7c_seed["variant_name"] == "baseline")
        & (phase7c_seed["point"] == "predict_100000_10000")
    ].copy()
    panel_b_top8 = phase7c_seed[
        (phase7c_seed["variant_name"] == "reranker_top8")
        & (phase7c_seed["point"] == "predict_100000_10000")
    ].copy()
    panel_b_rows = []
    for variant_name, display_name, frame in (
        ("base_plus_class_weight", "Base + class weighting", panel_b_base),
        ("auto_reranker_top8", "Base + class weighting + auto reranker top8", panel_b_top8),
    ):
        panel_b_rows.append(
            {
                "dataset": "PHMap_Lung_Full_v43_light",
                "point": "predict_100000_10000",
                "variant": variant_name,
                "display_name": display_name,
                "metric": "parent_correct_child_wrong_rate",
                "mean": float(frame["parent_correct_child_wrong_rate"].mean()),
                "std": float(frame["parent_correct_child_wrong_rate"].std(ddof=1)),
            }
        )
        panel_b_rows.append(
            {
                "dataset": "PHMap_Lung_Full_v43_light",
                "point": "predict_100000_10000",
                "variant": variant_name,
                "display_name": display_name,
                "metric": "path_break_rate",
                "mean": float(frame["path_break_rate"].mean()),
                "std": float(frame["path_break_rate"].std(ddof=1)),
            }
        )
    panel_b = pd.DataFrame(panel_b_rows)

    panel_d = phase7c[
        (phase7c["point"] == "predict_100000_10000")
        & (phase7c["variant_name"].isin(["reranker_top6", "reranker_top8"]))
    ].copy()
    panel_d_rows = []
    for _, row in panel_d.iterrows():
        panel_d_rows.extend(
            [
                {
                    "dataset": "PHMap_Lung_Full_v43_light",
                    "variant": str(row["variant_name"]),
                    "metric": "anno_lv4_macro_f1",
                    "mean": float(row["anno_lv4_macro_f1_mean"]),
                    "std": float(row["anno_lv4_macro_f1_std"]),
                },
                {
                    "dataset": "PHMap_Lung_Full_v43_light",
                    "variant": str(row["variant_name"]),
                    "metric": "full_path_accuracy",
                    "mean": float(row["full_path_accuracy_mean"]),
                    "std": float(row["full_path_accuracy_std"]),
                },
            ]
        )
    panel_d = pd.DataFrame(panel_d_rows)

    panel_a.to_csv(OUT_DIR / "paper_main_figure_panel_a.csv", index=False)
    panel_b.to_csv(OUT_DIR / "paper_main_figure_panel_b.csv", index=False)
    panel_d.to_csv(OUT_DIR / "paper_main_figure_panel_d.csv", index=False)


def export_supplementary() -> None:
    phase2_screen = _load_csv("results_summary/phase2_screen_comparison.csv")
    phase4 = _load_csv("results_summary/phase4_hotspot_refinement_comparison.csv")
    phase5 = _load_csv("results_summary/phase5_parent_conditioned_refinement_comparison.csv")
    phase6b = _load_csv("results_summary/phase6b_comparison.csv")
    phase6c = _load_csv("results_summary/phase6c_comparison.csv")
    phase7b = _load_csv("results_summary/phase7b_hotspot_selection_comparison/phase7b_hotspot_selection_comparison.csv")
    phase7c_summary = _load_csv("results_summary/phase7c_top6_vs_top8/phase7c_top6_vs_top8_seed_summary.csv")
    phase7c_by_study = _load_csv("results_summary/phase7c_top6_vs_top8/phase7c_top6_vs_top8_by_study.csv")
    phase7c_overlap = _load_csv("results_summary/phase7c_top6_vs_top8/phase7c_top6_vs_top8_parent_overlap.csv")
    phase2_seed = _load_csv("results_summary/phase2_seed_summary.csv")

    ablation_rows = []
    target_point = "predict_100000_10000"
    for _, row in phase2_screen[phase2_screen["point"] == target_point].iterrows():
        ablation_rows.append(
            {
                "dataset": "PHMap_Lung_Full_v43_light",
                "stage": "phase2_screen",
                "variant": str(row["config_name"]),
                "point": target_point,
                "finest_level_macro_f1": float(row["macro_f1"]),
                "full_path_accuracy": float(row["full_path_accuracy"]),
                "coverage": float(row["coverage"]),
                "parent_correct_child_wrong_rate": None,
            }
        )
    for _, row in phase4[
        (phase4["point"] == target_point) & (phase4["hierarchy_setting"] == "on")
    ].iterrows():
        ablation_rows.append(
            {
                "dataset": "PHMap_Lung_Full_v43_light",
                "stage": "phase4_hotspot_refinement",
                "variant": str(row["variant_name"]),
                "point": target_point,
                "finest_level_macro_f1": float(row["anno_lv4_macro_f1"]),
                "full_path_accuracy": float(row["full_path_accuracy"]),
                "coverage": float(row["anno_lv4_coverage"]),
                "parent_correct_child_wrong_rate": float(row["parent_correct_child_wrong_rate"]),
            }
        )
    for _, row in phase5[
        (phase5["point"] == target_point) & (phase5["hierarchy_setting"] == "on")
    ].iterrows():
        ablation_rows.append(
            {
                "dataset": "PHMap_Lung_Full_v43_light",
                "stage": "phase5_parent_conditioned",
                "variant": str(row["variant_name"]),
                "point": target_point,
                "finest_level_macro_f1": float(row["anno_lv4_macro_f1"]),
                "full_path_accuracy": float(row["full_path_accuracy"]),
                "coverage": float(row["anno_lv4_coverage"]),
                "parent_correct_child_wrong_rate": float(row["parent_correct_child_wrong_rate"]),
            }
        )
    pd.DataFrame(ablation_rows).to_csv(OUT_DIR / "paper_s1_ablation_ladder.csv", index=False)

    stability_rows = []
    for _, row in phase2_seed[phase2_seed["point"] == target_point].iterrows():
        stability_rows.append(
            {
                "dataset": "PHMap_Lung_Full_v43_light",
                "comparison_family": "class_weight_seed_stability",
                "variant": str(row["config_name"]),
                "point": target_point,
                "finest_level_macro_f1_mean": float(row["macro_f1_mean"]),
                "finest_level_macro_f1_std": float(row["macro_f1_std"]),
                "full_path_accuracy_mean": float(row["full_path_accuracy_mean"]),
                "full_path_accuracy_std": float(row["full_path_accuracy_std"]),
                "parent_correct_child_wrong_rate_mean": None,
                "parent_correct_child_wrong_rate_std": None,
            }
        )
    for _, row in phase7c_summary[phase7c_summary["point"] == target_point].iterrows():
        stability_rows.append(
            {
                "dataset": "PHMap_Lung_Full_v43_light",
                "comparison_family": "top6_vs_top8_seed_stability",
                "variant": str(row["variant_name"]),
                "point": target_point,
                "finest_level_macro_f1_mean": float(row["anno_lv4_macro_f1_mean"]),
                "finest_level_macro_f1_std": float(row["anno_lv4_macro_f1_std"]),
                "full_path_accuracy_mean": float(row["full_path_accuracy_mean"]),
                "full_path_accuracy_std": float(row["full_path_accuracy_std"]),
                "parent_correct_child_wrong_rate_mean": float(row["parent_correct_child_wrong_rate_mean"]),
                "parent_correct_child_wrong_rate_std": float(row["parent_correct_child_wrong_rate_std"]),
            }
        )
    pd.DataFrame(stability_rows).to_csv(OUT_DIR / "paper_s2_stability.csv", index=False)
    phase7c_by_study.to_csv(OUT_DIR / "paper_s2_by_group.csv", index=False)

    phase7b.to_csv(OUT_DIR / "paper_s3_hotspot_rule_comparison.csv", index=False)
    phase7c_overlap.to_csv(OUT_DIR / "paper_s3_parent_overlap.csv", index=False)

    internalization = pd.concat(
        [
            phase6b[
                (phase6b["point"] == target_point)
                & (phase6b["hierarchy_setting"] == "on")
                & (
                    phase6b["variant_name"].isin(
                        ["baseline", "reranker_top6", "correction_frozen_base", "correction_joint"]
                    )
                )
            ].assign(source_round="phase6b"),
            phase6c[
                phase6c["point"] == target_point
            ].assign(source_round="phase6c"),
        ],
        ignore_index=True,
        sort=False,
    )
    internalization.to_csv(OUT_DIR / "paper_s4_internalization_branch.csv", index=False)


def export_operational_bundle() -> None:
    top8_artifact = DOSSIER_ROOT / "artifacts" / "phase7a_auto_reranker_pipeline"
    artifact_rows = [
        {
            "dataset": "PHMap_Lung_Full_v43_light",
            "variant": "auto_parent_conditioned_reranker_top8_candidate",
            "hotspot_ranking_path": str((top8_artifact / "hotspot_ranking.json").resolve()),
            "refinement_plan_path": str((top8_artifact / "refinement_plan.json").resolve()),
            "reranker_artifact_path": str((top8_artifact / "parent_conditioned_reranker_top6.pkl").resolve()),
            "guardrail_decision_path": str((top8_artifact / "guardrail_decision.json").resolve()),
            "fit_summary_path": str((top8_artifact / "per_parent_reranker_summary.csv").resolve()),
            "artifact_size_bytes": (
                (top8_artifact / "parent_conditioned_reranker_top6.pkl").stat().st_size
                if (top8_artifact / "parent_conditioned_reranker_top6.pkl").exists()
                else None
            ),
            "reranker_fit_time_seconds": None,
            "reranker_apply_time_seconds": None,
            "delta_runtime_seconds": None,
            "delta_storage_bytes": None,
        }
    ]
    pd.DataFrame(artifact_rows).to_csv(OUT_DIR / "paper_operational_overhead.csv", index=False)
    (OUT_DIR / "paper_operational_artifact_index.json").write_text(
        json.dumps(artifact_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def export_summary() -> None:
    summary = {
        "dataset": "PHMap_Lung_Full_v43_light",
        "default_operational_path": "lv4strong + class_weighting + auto_parent_conditioned_reranker_top8",
        "paper_ready_dir": str(OUT_DIR.resolve()),
        "generated_files": sorted(path.name for path in OUT_DIR.iterdir() if path.is_file()),
    }
    (OUT_DIR / "paper_ready_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_main_panels()
    export_supplementary()
    export_operational_bundle()
    export_summary()
    print(
        json.dumps(
            {
                "paper_ready_dir": str(OUT_DIR.resolve()),
                "file_count": len(list(OUT_DIR.glob("*"))),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
