#!/usr/bin/env python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from anndata import read_h5ad
from sklearn.linear_model import LogisticRegression

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
HOTSPOT_PARENTS = ("CD8+ T", "Mph alveolar", "Pericyte", "CD4+ T")
POINTS = ("build_100000_eval10k", "predict_100000_10000")
HIERARCHY_SETTINGS = ("on", "off")
VARIANTS = ("baseline", "parent_conditioned_reranker")
UNKNOWN = "Unknown"
CONFIDENCE_HIGH = 0.7
CONFIDENCE_LOW = 0.4
MARGIN_THRESHOLD = 0.2


@dataclass
class HotspotReranker:
    parent_label: str
    child_names: List[str]
    child_full_indices: np.ndarray
    model: LogisticRegression | None
    constant_child_index: int | None = None

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.shape[0] == 0:
            return np.zeros((0, len(self.child_names)), dtype=np.float32)
        if self.constant_child_index is not None:
            out = np.zeros((features.shape[0], len(self.child_names)), dtype=np.float32)
            out[:, self.constant_child_index] = 1.0
            return out
        assert self.model is not None
        probs = self.model.predict_proba(features)
        return np.asarray(probs, dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _point_manifest(point: str) -> Dict[str, Any]:
    path = PHASE3_MANIFEST_ROOT / point / "hierarchy_on" / "atlasmtl_phase3_predict.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _base_outputs(model: TrainedModel, adata, *, device: str, batch_size: int) -> Dict[str, Dict[str, np.ndarray]]:
    X = extract_matrix(adata, model.train_genes, input_transform=model.input_transform)
    configure_torch_threads(10)
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
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    calibration = ((model.train_config or {}).get("calibration") or {}).get("temperatures", {})
    for idx, col in enumerate(model.label_columns):
        arr = logits[idx].numpy()
        temp = calibration.get(col)
        if temp:
            arr = arr / float(temp)
        outputs[col] = {
            "logits": np.asarray(arr, dtype=np.float32),
            "probs": np.asarray(_softmax(arr), dtype=np.float32),
        }
    return outputs


def _fit_hotspot_rerankers(
    model: TrainedModel,
    *,
    reference_h5ad: Path,
    layer_cfg: Dict[str, Any],
    hierarchy_rules: Dict[str, Any],
    batch_size: int,
) -> Dict[str, HotspotReranker]:
    ref = read_h5ad(str(reference_h5ad))
    ref_model_input = adata_with_matrix_from_layer(ref, layer_name=layer_cfg["reference_layer"])
    outputs = _base_outputs(model, ref_model_input, device="cuda", batch_size=batch_size)
    lv4_logits = outputs["anno_lv4"]["logits"]
    enc = model.label_encoders["anno_lv4"]
    child_to_parent = {
        str(k): str(v)
        for k, v in dict((hierarchy_rules.get("anno_lv4") or {}).get("child_to_parent") or {}).items()
    }
    child_names = np.asarray(enc.classes_, dtype=object)
    rerankers: Dict[str, HotspotReranker] = {}
    for parent_label in HOTSPOT_PARENTS:
        legal_mask = np.array([child_to_parent.get(str(name)) == parent_label for name in child_names], dtype=bool)
        legal_indices = np.where(legal_mask)[0]
        legal_names = child_names[legal_indices].tolist()
        ref_mask = ref.obs["anno_lv3"].astype(str).to_numpy() == parent_label
        X_parent = lv4_logits[ref_mask][:, legal_indices]
        y_parent_names = ref.obs.loc[ref_mask, "anno_lv4"].astype(str).to_numpy()
        present_names = [name for name in legal_names if np.any(y_parent_names == name)]
        if len(present_names) == 0:
            continue
        present_indices = np.array([legal_names.index(name) for name in present_names], dtype=np.int64)
        X_parent = X_parent[:, present_indices]
        full_indices = legal_indices[present_indices]
        y_parent = np.array([present_names.index(name) for name in y_parent_names], dtype=np.int64)
        unique = np.unique(y_parent)
        if unique.size == 1:
            rerankers[parent_label] = HotspotReranker(
                parent_label=parent_label,
                child_names=present_names,
                child_full_indices=full_indices,
                model=None,
                constant_child_index=int(unique[0]),
            )
            continue
        clf = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=2026,
        )
        clf.fit(X_parent, y_parent)
        rerankers[parent_label] = HotspotReranker(
            parent_label=parent_label,
            child_names=present_names,
            child_full_indices=full_indices,
            model=clf,
            constant_child_index=None,
        )
    return rerankers


def _build_pred_df(
    model: TrainedModel,
    outputs_by_col: Dict[str, Dict[str, np.ndarray]],
    *,
    variant_name: str,
    hierarchy_rules: Dict[str, Any],
    enforce_hierarchy: bool,
    obs_names,
    rerankers: Dict[str, HotspotReranker],
) -> pd.DataFrame:
    pred_df = pd.DataFrame(index=obs_names)
    metadata: Dict[str, Any] = {}
    for col in LABEL_COLUMNS[:-1]:
        append_level_predictions(
            pred_df,
            metadata,
            column_name=col,
            probs=outputs_by_col[col]["probs"],
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

    lv4_probs = outputs_by_col["anno_lv4"]["probs"].copy()
    if variant_name == "parent_conditioned_reranker":
        lv4_logits = outputs_by_col["anno_lv4"]["logits"]
        parent_pred = pred_df["pred_anno_lv3"].astype(str).to_numpy()
        for parent_label, reranker in rerankers.items():
            mask = parent_pred == parent_label
            if not np.any(mask):
                continue
            parent_probs = reranker.predict_proba(lv4_logits[mask][:, reranker.child_full_indices])
            full_probs = np.zeros((parent_probs.shape[0], lv4_probs.shape[1]), dtype=np.float32)
            full_probs[:, reranker.child_full_indices] = parent_probs
            lv4_probs[mask] = full_probs

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
    pred_on = comp_df[
        (comp_df["point"] == "predict_100000_10000")
        & (comp_df["hierarchy_setting"] == "on")
        & (comp_df["variant_name"] != "baseline")
    ].copy()
    pred_on["guardrail_ok"] = (
        pred_on["delta_vs_baseline_anno_lv4_macro_f1"] >= -0.005
    ) & (
        pred_on["delta_vs_baseline_anno_lv4_balanced_accuracy"] >= -0.005
    )
    viable = pred_on[pred_on["guardrail_ok"]].copy()
    if len(viable) == 0:
        return {"winner_variant": None, "status": "no_variant_passed_guardrail"}
    viable = viable.sort_values(
        by=[
            "full_path_accuracy",
            "delta_vs_baseline_parent_correct_child_wrong_rate",
            "delta_vs_baseline_path_break_rate",
            "anno_lv4_macro_f1",
        ],
        ascending=[False, True, True, False],
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
    anchor_manifest = _point_manifest("predict_100000_10000")
    layer_cfg = resolve_atlasmtl_layer_config(anchor_manifest)
    hierarchy_rules = (anchor_manifest.get("predict") or {}).get("hierarchy_rules") or {}
    rerankers = _fit_hotspot_rerankers(
        model,
        reference_h5ad=Path(anchor_manifest["reference_h5ad"]),
        layer_cfg=layer_cfg,
        hierarchy_rules=hierarchy_rules,
        batch_size=int((anchor_manifest.get("predict") or {}).get("batch_size", 512)),
    )

    variant_rows: List[Dict[str, Any]] = []
    levelwise_rows: List[Dict[str, Any]] = []
    hierarchy_rows: List[Dict[str, Any]] = []
    parent_child_rows: List[Dict[str, Any]] = []

    for point in POINTS:
        manifest = _point_manifest(point)
        point_hierarchy_rules = (manifest.get("predict") or {}).get("hierarchy_rules") or {}
        point_layer_cfg = resolve_atlasmtl_layer_config(manifest)
        query = read_h5ad(str(manifest["query_h5ad"]))
        query_model_input = adata_with_matrix_from_layer(query, layer_name=point_layer_cfg["query_layer"])
        outputs_by_col = _base_outputs(
            model,
            query_model_input,
            device="cuda",
            batch_size=int((manifest.get("predict") or {}).get("batch_size", 512)),
        )
        true_df = query.obs.loc[:, LABEL_COLUMNS + ["study"]].copy()
        for hierarchy_setting in HIERARCHY_SETTINGS:
            enforce_hierarchy = hierarchy_setting == "on"
            for variant_name in VARIANTS:
                pred_df = _build_pred_df(
                    model,
                    outputs_by_col,
                    variant_name=variant_name,
                    hierarchy_rules=point_hierarchy_rules,
                    enforce_hierarchy=enforce_hierarchy,
                    obs_names=query.obs_names,
                    rerankers=rerankers,
                )
                metrics = evaluate_predictions(pred_df, true_df[LABEL_COLUMNS], LABEL_COLUMNS)
                behavior = evaluate_prediction_behavior(pred_df, true_df[LABEL_COLUMNS], LABEL_COLUMNS)
                hierarchy = evaluate_hierarchy_metrics(
                    pred_df,
                    true_df[LABEL_COLUMNS],
                    LABEL_COLUMNS,
                    hierarchy_rules=point_hierarchy_rules,
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
                for child_col, rule in point_hierarchy_rules.items():
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
                        "hierarchy_rules": point_hierarchy_rules,
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
        ["variant_name", "point", "hierarchy_setting", "parent_correct_child_wrong_rate", "path_break_rate"]
    ]
    comp_df = comp_df.merge(edge_focus, on=["variant_name", "point", "hierarchy_setting"], how="left")
    baseline = comp_df[comp_df["variant_name"] == "baseline"][
        ["point", "hierarchy_setting"] + baseline_cols + ["parent_correct_child_wrong_rate", "path_break_rate"]
    ].rename(columns={col: f"{col}_baseline" for col in baseline_cols + ["parent_correct_child_wrong_rate", "path_break_rate"]})
    comp_df = comp_df.merge(baseline, on=["point", "hierarchy_setting"], how="left")
    for col in baseline_cols + ["parent_correct_child_wrong_rate", "path_break_rate"]:
        comp_df[f"delta_vs_baseline_{col}"] = comp_df[col] - comp_df[f"{col}_baseline"]

    winner = _winner(comp_df)

    summary_lines = [
        "# PH-Map Phase 5 Parent-Conditioned Child Refinement",
        "",
        f"- hotspot parents: `{', '.join(HOTSPOT_PARENTS)}`",
        f"- fitted rerankers: `{', '.join(sorted(rerankers.keys()))}`",
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
                "anno_lv4_balanced_accuracy",
                "delta_vs_baseline_anno_lv4_balanced_accuracy",
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
    comp_df.to_csv(RESULTS_DIR / "phase5_parent_conditioned_refinement_comparison.csv", index=False)
    parent_child_df.to_csv(RESULTS_DIR / "phase5_parent_conditioned_parent_child_breakdown.csv", index=False)
    subtree_df.to_csv(RESULTS_DIR / "phase5_parent_conditioned_subtree_breakdown.csv", index=False)
    study_df.to_csv(RESULTS_DIR / "phase5_parent_conditioned_by_study.csv", index=False)
    (RESULTS_DIR / "phase5_parent_conditioned_refinement_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (RESULTS_DIR / "phase5_execution_record.md").write_text(
        "\n".join(
            [
                "# Phase 5 Execution Record",
                "",
                "Date: `2026-03-10`",
                "",
                "This round applies parent-conditioned hotspot child reranking on top of `lv4strong_plus_class_weight`.",
                "",
                f"- hotspot parents: `{', '.join(HOTSPOT_PARENTS)}`",
                f"- fitted rerankers: `{', '.join(sorted(rerankers.keys()))}`",
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
