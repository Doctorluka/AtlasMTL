#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from anndata import read_h5ad

from atlasmtl import TrainedModel
from atlasmtl.core.data import extract_matrix
from atlasmtl.core.evaluate import (
    evaluate_hierarchy_metrics,
    evaluate_prediction_behavior,
    evaluate_predictions,
)
from atlasmtl.core.predict_utils import append_level_predictions, run_model_in_batches
from atlasmtl.core.runtime import configure_torch_threads, resolve_device
from atlasmtl.mapping import enforce_parent_child_consistency
from benchmark.methods.atlasmtl_inputs import adata_with_matrix_from_layer, resolve_atlasmtl_layer_config


REPO_ROOT = Path(__file__).resolve().parents[4]
DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
RESULTS_DIR = DOSSIER_ROOT / "results_summary"
PHASE3_MANIFEST_ROOT = DOSSIER_ROOT / "manifests" / "phase3_tradeoff" / "predict" / "lv4strong_plus_class_weight"
MODEL_MANIFEST = Path(
    "/tmp/atlasmtl_benchmarks/2026-03-10/phmap_study_split_phase3_tradeoff/train/lv4strong_plus_class_weight/runs/atlasmtl/atlasmtl_model_manifest.json"
)
LABEL_COLUMNS = ["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"]
HOTSPOT_PARENTS = {"CD8+ T", "Mph alveolar", "Pericyte", "CD4+ T"}
POINTS = ("build_100000_eval10k", "predict_100000_10000")
HIERARCHY_SETTINGS = ("on", "off")
VARIANTS = (
    "baseline",
    "hotspot_thresholding",
    "hotspot_temperature_scaling",
    "hotspot_thresholding_plus_temperature",
)
UNKNOWN = "Unknown"
CONFIDENCE_HIGH = 0.7
CONFIDENCE_LOW = 0.4
MARGIN_THRESHOLD = 0.2
HOTSPOT_MAX_PROB_THRESHOLD = 0.55
HOTSPOT_MARGIN_THRESHOLD = 0.15
HOTSPOT_TEMPERATURE = 1.25


def _point_manifest(point: str) -> Dict[str, Any]:
    path = PHASE3_MANIFEST_ROOT / point / "hierarchy_on" / "atlasmtl_phase3_predict.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = log_probs / float(temperature)
    return _softmax(scaled)


def _base_probs(model: TrainedModel, query_model_input, *, device: str, batch_size: int) -> Dict[str, np.ndarray]:
    X = extract_matrix(query_model_input, model.train_genes, input_transform=model.input_transform)
    num_threads_used = configure_torch_threads(10)
    resolved_device = resolve_device(device)
    model.model.to(resolved_device)
    model.model.eval()
    logits, _, _, _ = run_model_in_batches(
        model,
        X,
        batch_size,
        resolved_device,
        show_progress=False,
        return_latent=False,
    )
    out = {}
    for idx, col in enumerate(model.label_columns):
        arr = logits[idx].numpy()
        calib = ((model.train_config or {}).get("calibration") or {}).get("temperatures", {})
        temp = calib.get(col)
        if temp:
            arr = arr / float(temp)
        out[col] = _softmax(arr)
    out["_num_threads_used"] = np.array([num_threads_used], dtype=np.int64)
    return out


def _build_pred_df(
    model: TrainedModel,
    probs_by_col: Dict[str, np.ndarray],
    *,
    variant_name: str,
    hierarchy_rules: Dict[str, Any],
    enforce_hierarchy: bool,
    obs_names,
) -> pd.DataFrame:
    pred_df = pd.DataFrame(index=obs_names)
    metadata: Dict[str, Any] = {}
    for col in LABEL_COLUMNS[:-1]:
        append_level_predictions(
            pred_df,
            metadata,
            column_name=col,
            probs=probs_by_col[col],
            label_encoder=model.label_encoders[col],
            confidence_high=CONFIDENCE_HIGH,
            confidence_low=CONFIDENCE_LOW,
            margin_threshold=MARGIN_THRESHOLD,
            knn_correction="off",
            knn_conf_low=0.6,
            knn_k=15,
            query_space=None,
            ref_space=None,
            ref_labels=model.reference_labels[col],
        )

    hotspot_mask = pred_df["pred_anno_lv3"].astype(str).isin(HOTSPOT_PARENTS).to_numpy()
    lv4_probs = probs_by_col["anno_lv4"].copy()
    if "temperature" in variant_name:
        lv4_probs[hotspot_mask] = _temperature_scale_probs(
            lv4_probs[hotspot_mask],
            HOTSPOT_TEMPERATURE,
        )

    append_level_predictions(
        pred_df,
        metadata,
        column_name="anno_lv4",
        probs=lv4_probs,
        label_encoder=model.label_encoders["anno_lv4"],
        confidence_high=CONFIDENCE_HIGH,
        confidence_low=CONFIDENCE_LOW,
        margin_threshold=MARGIN_THRESHOLD,
        knn_correction="off",
        knn_conf_low=0.6,
        knn_k=15,
        query_space=None,
        ref_space=None,
        ref_labels=model.reference_labels["anno_lv4"],
    )

    if "thresholding" in variant_name:
        conf = pred_df["conf_anno_lv4"].to_numpy(dtype=np.float32, copy=False)
        margin = pred_df["margin_anno_lv4"].to_numpy(dtype=np.float32, copy=False)
        trigger = hotspot_mask & (
            (conf < HOTSPOT_MAX_PROB_THRESHOLD)
            | (margin < HOTSPOT_MARGIN_THRESHOLD)
        )
        pred_df.loc[trigger, "is_unknown_anno_lv4"] = True
        pred_df.loc[trigger, "pred_anno_lv4"] = UNKNOWN

    if enforce_hierarchy:
        for child_col, rule in hierarchy_rules.items():
            pred_df, _ = enforce_parent_child_consistency(
                pred_df,
                parent_col=str(rule["parent_col"]),
                child_col=str(child_col),
                child_to_parent={str(k): str(v) for k, v in dict(rule["child_to_parent"]).items()},
            )
    return pred_df


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


def _subtree_rows(variant_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    child_col = "anno_lv4"
    parent_col = "anno_lv3"
    for item in variant_rows:
        pred_df = item["pred_df"]
        true_df = item["true_df"]
        hierarchy_rules = item["hierarchy_rules"]
        child_to_parent = {str(k): str(v) for k, v in dict(hierarchy_rules[child_col]["child_to_parent"]).items()}
        parent_true = true_df[parent_col].astype(str)
        child_true = true_df[child_col].astype(str)
        parent_pred = pred_df[f"pred_{parent_col}"].astype(str)
        child_pred = pred_df[f"pred_{child_col}"].astype(str)
        implied_parent = child_pred.map(lambda x: child_to_parent.get(x) if x != UNKNOWN else UNKNOWN).fillna("MISSING_PARENT")
        for parent_label, idx in true_df.groupby(parent_col, observed=False).groups.items():
            idx = list(idx)
            pt = parent_true.loc[idx]
            ct = child_true.loc[idx]
            pp = parent_pred.loc[idx]
            cp = child_pred.loc[idx]
            ip = implied_parent.loc[idx]
            parent_correct = pt == pp
            child_correct = ct == cp
            path_break = (cp != UNKNOWN) & (pp != UNKNOWN) & (ip != pp)
            rows.append(
                {
                    "variant_name": item["variant_name"],
                    "point": item["point"],
                    "hierarchy_setting": item["hierarchy_setting"],
                    "parent_label": str(parent_label),
                    "n_cells": int(len(idx)),
                    "lv4_accuracy": float((ct == cp).mean()),
                    "lv4_unknown_rate": float((cp == UNKNOWN).mean()),
                    "parent_correct_child_wrong_rate": float((parent_correct & ~child_correct).mean()),
                    "path_break_rate": float(path_break.mean()),
                }
            )
    return rows


def _by_study_rows(variant_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in variant_rows:
        pred_df = item["pred_df"]
        true_df = item["true_df"]
        for study, idx in true_df.groupby("study", observed=False).groups.items():
            idx = list(idx)
            sub_pred = pred_df.loc[idx]
            sub_true = true_df.loc[idx, LABEL_COLUMNS]
            metrics = evaluate_predictions(sub_pred, sub_true, LABEL_COLUMNS)
            behavior = evaluate_prediction_behavior(sub_pred, sub_true, LABEL_COLUMNS)
            hierarchy = evaluate_hierarchy_metrics(
                sub_pred,
                sub_true,
                LABEL_COLUMNS,
                hierarchy_rules=item["hierarchy_rules"],
            )
            rows.append(
                {
                    "variant_name": item["variant_name"],
                    "point": item["point"],
                    "hierarchy_setting": item["hierarchy_setting"],
                    "study": str(study),
                    "anno_lv4_macro_f1": (metrics.get("anno_lv4") or {}).get("macro_f1"),
                    "coverage": (metrics.get("anno_lv4") or {}).get("coverage"),
                    "unknown_rate": (behavior.get("anno_lv4") or {}).get("unknown_rate"),
                    "full_path_accuracy": hierarchy.get("full_path_accuracy"),
                }
            )
    return rows


def _winner(comp_df: pd.DataFrame) -> Dict[str, Any]:
    pred_on = comp_df[(comp_df["point"] == "predict_100000_10000") & (comp_df["hierarchy_setting"] == "on")].copy()
    pred_on = pred_on[pred_on["variant_name"] != "baseline"].copy()
    pred_on["guardrail_ok"] = (
        pred_on["delta_vs_baseline_anno_lv4_macro_f1"] >= -0.01
    ) & (
        pred_on["delta_vs_baseline_anno_lv4_coverage"] >= -0.08
    )
    viable = pred_on[pred_on["guardrail_ok"]].copy()
    if len(viable) == 0:
        return {"winner_variant": None, "status": "no_variant_passed_guardrail"}
    viable = viable.sort_values(
        by=[
            "full_path_accuracy",
            "anno_lv4_macro_f1",
            "delta_vs_baseline_parent_correct_child_wrong_rate",
            "delta_vs_baseline_path_break_rate",
        ],
        ascending=[False, False, True, True],
    )
    row = viable.iloc[0]
    return {
        "winner_variant": row["variant_name"],
        "status": "selected",
        "point": row["point"],
        "hierarchy_setting": row["hierarchy_setting"],
        "full_path_accuracy": row["full_path_accuracy"],
        "anno_lv4_macro_f1": row["anno_lv4_macro_f1"],
        "delta_vs_baseline_full_path_accuracy": row["delta_vs_baseline_full_path_accuracy"],
        "delta_vs_baseline_anno_lv4_macro_f1": row["delta_vs_baseline_anno_lv4_macro_f1"],
    }


def main() -> None:
    model = TrainedModel.load(str(MODEL_MANIFEST))
    variant_rows: List[Dict[str, Any]] = []
    levelwise_rows: List[Dict[str, Any]] = []
    hierarchy_rows: List[Dict[str, Any]] = []
    parent_child_rows: List[Dict[str, Any]] = []

    for point in POINTS:
        manifest = _point_manifest(point)
        layer_cfg = resolve_atlasmtl_layer_config(manifest)
        query = read_h5ad(str(manifest["query_h5ad"]))
        query_model_input = adata_with_matrix_from_layer(query, layer_name=layer_cfg["query_layer"])
        probs_by_col = _base_probs(
            model,
            query_model_input,
            device="cuda",
            batch_size=int((manifest.get("predict") or {}).get("batch_size", 512)),
        )
        hierarchy_rules = (manifest.get("predict") or {}).get("hierarchy_rules") or {}
        true_df = query.obs.loc[:, LABEL_COLUMNS + ["study"]].copy()
        for hierarchy_setting in HIERARCHY_SETTINGS:
            enforce_hierarchy = hierarchy_setting == "on"
            for variant_name in VARIANTS:
                pred_df = _build_pred_df(
                    model,
                    probs_by_col,
                    variant_name=variant_name,
                    hierarchy_rules=hierarchy_rules,
                    enforce_hierarchy=enforce_hierarchy,
                    obs_names=query.obs_names,
                )
                metrics = evaluate_predictions(pred_df, true_df[LABEL_COLUMNS], LABEL_COLUMNS)
                behavior = evaluate_prediction_behavior(pred_df, true_df[LABEL_COLUMNS], LABEL_COLUMNS)
                hierarchy = evaluate_hierarchy_metrics(
                    pred_df,
                    true_df[LABEL_COLUMNS],
                    LABEL_COLUMNS,
                    hierarchy_rules=hierarchy_rules,
                )
                lv4 = metrics["anno_lv4"]
                lv4_behavior = behavior["anno_lv4"]
                edge_rates = {child_col: (payload or {}).get("path_consistency_rate") for child_col, payload in (hierarchy.get("edges") or {}).items()}
                valid_rates = [float(v) for v in edge_rates.values() if v is not None]
                levelwise_rows.append(
                    {
                        "variant_name": variant_name,
                        "point": point,
                        "hierarchy_setting": hierarchy_setting,
                        "anno_lv4_accuracy": lv4.get("accuracy"),
                        "anno_lv4_macro_f1": lv4.get("macro_f1"),
                        "anno_lv4_balanced_accuracy": lv4.get("balanced_accuracy"),
                        "anno_lv4_coverage": lv4.get("coverage"),
                        "anno_lv4_covered_accuracy": lv4.get("covered_accuracy"),
                        "anno_lv4_unknown_rate": lv4_behavior.get("unknown_rate"),
                        "full_path_accuracy": hierarchy.get("full_path_accuracy"),
                        "full_path_coverage": hierarchy.get("full_path_coverage"),
                        "mean_path_consistency_rate": sum(valid_rates) / len(valid_rates) if valid_rates else None,
                    }
                )
                hierarchy_rows.append(
                    {
                        "variant_name": variant_name,
                        "point": point,
                        "hierarchy_setting": hierarchy_setting,
                        "full_path_accuracy": hierarchy.get("full_path_accuracy"),
                        "full_path_coverage": hierarchy.get("full_path_coverage"),
                        "full_path_covered_accuracy": hierarchy.get("full_path_covered_accuracy"),
                        "mean_path_consistency_rate": sum(valid_rates) / len(valid_rates) if valid_rates else None,
                    }
                )
                for child_col, rule in hierarchy_rules.items():
                    edge = _edge_breakdown(
                        pred_df,
                        true_df[LABEL_COLUMNS],
                        parent_col=str(rule["parent_col"]),
                        child_col=str(child_col),
                        child_to_parent={str(k): str(v) for k, v in dict(rule["child_to_parent"]).items()},
                    )
                    parent_child_rows.append(
                        {
                            "variant_name": variant_name,
                            "point": point,
                            "hierarchy_setting": hierarchy_setting,
                            "child_col": str(child_col),
                            "parent_col": str(rule["parent_col"]),
                            "path_consistency_rate": ((hierarchy.get("edges") or {}).get(str(child_col)) or {}).get("path_consistency_rate"),
                            **edge,
                        }
                    )
                variant_rows.append(
                    {
                        "variant_name": variant_name,
                        "point": point,
                        "hierarchy_setting": hierarchy_setting,
                        "pred_df": pred_df,
                        "true_df": true_df,
                        "hierarchy_rules": hierarchy_rules,
                    }
                )

    levelwise_df = pd.DataFrame(levelwise_rows)
    hierarchy_df = pd.DataFrame(hierarchy_rows)
    parent_child_df = pd.DataFrame(parent_child_rows)
    subtree_df = pd.DataFrame(_subtree_rows(variant_rows))
    study_df = pd.DataFrame(_by_study_rows(variant_rows))

    baseline_cols = [
        "anno_lv4_macro_f1",
        "anno_lv4_balanced_accuracy",
        "anno_lv4_coverage",
        "anno_lv4_unknown_rate",
        "full_path_accuracy",
        "full_path_coverage",
        "mean_path_consistency_rate",
    ]
    comp_df = levelwise_df.copy()
    edge_focus = parent_child_df[parent_child_df["child_col"] == "anno_lv4"][
        [
            "variant_name",
            "point",
            "hierarchy_setting",
            "parent_correct_child_wrong_rate",
            "path_break_rate",
        ]
    ]
    comp_df = comp_df.merge(edge_focus, on=["variant_name", "point", "hierarchy_setting"], how="left")
    baseline = comp_df[comp_df["variant_name"] == "baseline"][
        ["point", "hierarchy_setting"] + baseline_cols + ["parent_correct_child_wrong_rate", "path_break_rate"]
    ].rename(columns={col: f"{col}_baseline" for col in baseline_cols + ["parent_correct_child_wrong_rate", "path_break_rate"]})
    comp_df = comp_df.merge(baseline, left_on=["point", "hierarchy_setting"], right_on=["point", "hierarchy_setting"], how="left")
    for col in baseline_cols + ["parent_correct_child_wrong_rate", "path_break_rate"]:
        comp_df[f"delta_vs_baseline_{col}"] = comp_df[col] - comp_df[f"{col}_baseline"]

    winner = _winner(comp_df)

    summary_lines = [
        "# PH-Map Phase 4 Hotspot Refinement",
        "",
        f"- hotspot parents: `{', '.join(sorted(HOTSPOT_PARENTS))}`",
        f"- hotspot thresholds: `max_prob<{HOTSPOT_MAX_PROB_THRESHOLD}`, `margin<{HOTSPOT_MARGIN_THRESHOLD}`",
        f"- hotspot temperature: `T={HOTSPOT_TEMPERATURE}`",
        "",
        "## Comparison",
        "",
        comp_df[
            [
                "variant_name",
                "point",
                "hierarchy_setting",
                "anno_lv4_macro_f1",
                "delta_vs_baseline_anno_lv4_macro_f1",
                "full_path_accuracy",
                "delta_vs_baseline_full_path_accuracy",
                "anno_lv4_coverage",
                "delta_vs_baseline_anno_lv4_coverage",
                "parent_correct_child_wrong_rate",
                "delta_vs_baseline_parent_correct_child_wrong_rate",
                "path_break_rate",
                "delta_vs_baseline_path_break_rate",
            ]
        ].to_markdown(index=False),
        "",
        "## Winner",
        "",
        json.dumps(winner, indent=2, sort_keys=True),
        "",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(RESULTS_DIR / "phase4_hotspot_refinement_comparison.csv", index=False)
    parent_child_df.to_csv(RESULTS_DIR / "phase4_hotspot_refinement_parent_child_breakdown.csv", index=False)
    subtree_df.to_csv(RESULTS_DIR / "phase4_hotspot_refinement_subtree_breakdown.csv", index=False)
    study_df.to_csv(RESULTS_DIR / "phase4_hotspot_refinement_by_study.csv", index=False)
    (RESULTS_DIR / "phase4_hotspot_refinement_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (RESULTS_DIR / "phase4_execution_record.md").write_text(
        "\n".join(
            [
                "# Phase 4 Execution Record",
                "",
                "Date: `2026-03-10`",
                "",
                "This round applies hotspot-local child refinement on top of `lv4strong_plus_class_weight`.",
                "",
                f"- hotspot parents: `{', '.join(sorted(HOTSPOT_PARENTS))}`",
                f"- hotspot thresholds: `max_prob<{HOTSPOT_MAX_PROB_THRESHOLD}`, `margin<{HOTSPOT_MARGIN_THRESHOLD}`",
                f"- hotspot temperature: `T={HOTSPOT_TEMPERATURE}`",
                "",
                "Winner:",
                "",
                json.dumps(winner, indent=2, sort_keys=True),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "comparison_rows": len(comp_df),
                "parent_child_rows": len(parent_child_df),
                "subtree_rows": len(subtree_df),
                "study_rows": len(study_df),
                "winner": winner,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
