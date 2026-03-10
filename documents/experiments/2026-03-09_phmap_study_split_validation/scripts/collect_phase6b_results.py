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
from atlasmtl.core.evaluate import evaluate_hierarchy_metrics, evaluate_prediction_behavior, evaluate_predictions
from atlasmtl.core.predict_utils import append_level_predictions, run_model_in_batches
from atlasmtl.core.runtime import configure_torch_threads, resolve_device
from atlasmtl.mapping import enforce_parent_child_consistency
from benchmark.methods.atlasmtl_inputs import adata_with_matrix_from_layer, resolve_atlasmtl_layer_config


REPO_ROOT = Path(__file__).resolve().parents[4]
DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
RESULTS_DIR = DOSSIER_ROOT / "results_summary"
PHASE2_MANIFEST_ROOT = DOSSIER_ROOT / "manifests" / "phase2_seed" / "predict" / "lv4strong_plus_class_weight"
PHASE2_MODEL_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-09/phmap_study_split_phase2_seed/train/lv4strong_plus_class_weight")
PHASE6B_MODEL_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-10/phmap_study_split_phase6b/train")

LABEL_COLUMNS = ["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"]
POINTS = ("build_100000_eval10k", "predict_100000_10000")
HIERARCHY_SETTINGS = ("on", "off")
SEEDS = (101, 17, 2026, 23, 47)
HOTSPOT_TOP6 = [
    "CD4+ T",
    "SMC de-differentiated",
    "Mph alveolar",
    "EC vascular",
    "Fibro adventitial",
    "CD8+ T",
]
UNKNOWN = "Unknown"
CONFIDENCE_HIGH = 0.7
CONFIDENCE_LOW = 0.4
MARGIN_THRESHOLD = 0.2
VARIANTS = (
    "baseline",
    "reranker_top6",
    "correction_joint",
    "correction_frozen_base",
)


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
        return np.asarray(self.model.predict_proba(features), dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _point_manifest(seed: int, point: str) -> Dict[str, Any]:
    path = PHASE2_MANIFEST_ROOT / f"seed_{seed}" / point / "atlasmtl_phase2_seed_predict.yaml"
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
    child_names = np.asarray(enc.classes_, dtype=object)
    child_to_parent = {
        str(k): str(v)
        for k, v in dict((hierarchy_rules.get("anno_lv4") or {}).get("child_to_parent") or {}).items()
    }
    rerankers: Dict[str, HotspotReranker] = {}
    for parent_label in HOTSPOT_TOP6:
        legal_mask = np.array([child_to_parent.get(str(name)) == parent_label for name in child_names], dtype=bool)
        legal_indices = np.where(legal_mask)[0]
        legal_names = child_names[legal_indices].tolist()
        ref_mask = ref.obs["anno_lv3"].astype(str).to_numpy() == str(parent_label)
        X_parent = lv4_logits[ref_mask][:, legal_indices]
        y_parent_names = ref.obs.loc[ref_mask, "anno_lv4"].astype(str).to_numpy()
        present_names = [name for name in legal_names if np.any(y_parent_names == name)]
        if not present_names:
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
        clf = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=2026)
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
    if rerankers:
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


def _edge_breakdown(pred_df: pd.DataFrame, true_df: pd.DataFrame, hierarchy_rules: Dict[str, Any]) -> Dict[str, float]:
    child_to_parent = {str(k): str(v) for k, v in dict(hierarchy_rules["anno_lv4"]["child_to_parent"]).items()}
    parent_true = true_df["anno_lv3"].astype(str)
    child_true = true_df["anno_lv4"].astype(str)
    parent_pred = pred_df["pred_anno_lv3"].astype(str)
    child_pred = pred_df["pred_anno_lv4"].astype(str)
    implied_parent = child_pred.map(lambda x: child_to_parent.get(x) if x != UNKNOWN else UNKNOWN).fillna("MISSING_PARENT")
    parent_correct = parent_true == parent_pred
    child_correct = child_true == child_pred
    path_break = (child_pred != UNKNOWN) & (parent_pred != UNKNOWN) & (implied_parent != parent_pred)
    return {
        "parent_correct_child_wrong_rate": float((parent_correct & ~child_correct).mean()),
        "path_break_rate": float(path_break.mean()),
    }


def _variant_models(seed: int) -> Dict[str, TrainedModel]:
    return {
        "baseline": TrainedModel.load(
            str(PHASE2_MODEL_ROOT / f"seed_{seed}" / "runs" / "atlasmtl" / "atlasmtl_model_manifest.json")
        ),
        "correction_joint": TrainedModel.load(
            str(PHASE6B_MODEL_ROOT / "correction_joint" / f"seed_{seed}" / "runs" / "atlasmtl" / "atlasmtl_model_manifest.json")
        ),
        "correction_frozen_base": TrainedModel.load(
            str(PHASE6B_MODEL_ROOT / "correction_frozen_base" / f"seed_{seed}" / "runs" / "atlasmtl" / "atlasmtl_model_manifest.json")
        ),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_rows: List[Dict[str, Any]] = []
    study_rows: List[Dict[str, Any]] = []
    parent_child_rows: List[Dict[str, Any]] = []

    for seed in SEEDS:
        models = _variant_models(seed)
        anchor_manifest = _point_manifest(seed, "predict_100000_10000")
        anchor_layer_cfg = resolve_atlasmtl_layer_config(anchor_manifest)
        anchor_rules = (anchor_manifest.get("predict") or {}).get("hierarchy_rules") or {}
        rerankers = _fit_hotspot_rerankers(
            models["baseline"],
            reference_h5ad=Path(anchor_manifest["reference_h5ad"]),
            layer_cfg=anchor_layer_cfg,
            hierarchy_rules=anchor_rules,
            batch_size=int((anchor_manifest.get("predict") or {}).get("batch_size", 512)),
        )

        for point in POINTS:
            manifest = _point_manifest(seed, point)
            hierarchy_rules = (manifest.get("predict") or {}).get("hierarchy_rules") or {}
            layer_cfg = resolve_atlasmtl_layer_config(manifest)
            query = read_h5ad(str(manifest["query_h5ad"]))
            query_model_input = adata_with_matrix_from_layer(query, layer_name=layer_cfg["query_layer"])
            outputs = {name: _base_outputs(model, query_model_input, device="cuda", batch_size=int((manifest.get("predict") or {}).get("batch_size", 512))) for name, model in models.items()}

            true_df = query.obs.loc[:, LABEL_COLUMNS + ["study"]].copy()
            for hierarchy_setting in HIERARCHY_SETTINGS:
                enforce_hierarchy = hierarchy_setting == "on"
                variant_pred_dfs: Dict[str, pd.DataFrame] = {}
                variant_pred_dfs["baseline"] = _build_pred_df(
                    models["baseline"],
                    outputs["baseline"],
                    hierarchy_rules=hierarchy_rules,
                    enforce_hierarchy=enforce_hierarchy,
                    obs_names=query.obs_names,
                    rerankers={},
                )
                variant_pred_dfs["reranker_top6"] = _build_pred_df(
                    models["baseline"],
                    outputs["baseline"],
                    hierarchy_rules=hierarchy_rules,
                    enforce_hierarchy=enforce_hierarchy,
                    obs_names=query.obs_names,
                    rerankers=rerankers,
                )
                for config_name in ("correction_joint", "correction_frozen_base"):
                    variant_pred_dfs[config_name] = _build_pred_df(
                        models[config_name],
                        outputs[config_name],
                        hierarchy_rules=hierarchy_rules,
                        enforce_hierarchy=enforce_hierarchy,
                        obs_names=query.obs_names,
                        rerankers={},
                    )

                baseline_metrics = None
                reranker_metrics = None
                for variant_name in VARIANTS:
                    pred_df = variant_pred_dfs[variant_name]
                    metrics = evaluate_predictions(pred_df, true_df[LABEL_COLUMNS], LABEL_COLUMNS)
                    behavior = evaluate_prediction_behavior(pred_df, true_df[LABEL_COLUMNS], LABEL_COLUMNS)
                    hierarchy = evaluate_hierarchy_metrics(pred_df, true_df[LABEL_COLUMNS], LABEL_COLUMNS, hierarchy_rules=hierarchy_rules)
                    edge = _edge_breakdown(pred_df, true_df[LABEL_COLUMNS], hierarchy_rules)
                    lv4 = metrics["anno_lv4"]
                    lv4_behavior = behavior["anno_lv4"]
                    rate_payload = ((hierarchy.get("edges") or {}).get("anno_lv4") or {})
                    row = {
                        "seed": seed,
                        "point": point,
                        "hierarchy_setting": hierarchy_setting,
                        "variant_name": variant_name,
                        "anno_lv4_macro_f1": lv4.get("macro_f1"),
                        "anno_lv4_balanced_accuracy": lv4.get("balanced_accuracy"),
                        "anno_lv4_coverage": lv4.get("coverage"),
                        "anno_lv4_unknown_rate": lv4_behavior.get("unknown_rate"),
                        "full_path_accuracy": hierarchy.get("full_path_accuracy"),
                        "full_path_coverage": hierarchy.get("full_path_coverage"),
                        "mean_path_consistency_rate": rate_payload.get("path_consistency_rate"),
                        "parent_correct_child_wrong_rate": edge["parent_correct_child_wrong_rate"],
                        "path_break_rate": edge["path_break_rate"],
                    }
                    if variant_name == "baseline":
                        baseline_metrics = row
                    if variant_name == "reranker_top6":
                        reranker_metrics = row
                    for col in (
                        "anno_lv4_macro_f1",
                        "anno_lv4_balanced_accuracy",
                        "anno_lv4_coverage",
                        "anno_lv4_unknown_rate",
                        "full_path_accuracy",
                        "full_path_coverage",
                        "mean_path_consistency_rate",
                        "parent_correct_child_wrong_rate",
                        "path_break_rate",
                    ):
                        row[f"delta_vs_baseline_{col}"] = row[col] - baseline_metrics[col]
                        if reranker_metrics is not None:
                            row[f"delta_vs_reranker_{col}"] = row[col] - reranker_metrics[col]
                    comparison_rows.append(row)

                    if point == "predict_100000_10000" and hierarchy_setting == "on":
                        for study, idx in true_df.groupby("study", observed=False).groups.items():
                            idx = list(idx)
                            sub_pred = pred_df.loc[idx]
                            sub_true = true_df.loc[idx, LABEL_COLUMNS]
                            sub_metrics = evaluate_predictions(sub_pred, sub_true, LABEL_COLUMNS)
                            sub_hierarchy = evaluate_hierarchy_metrics(sub_pred, sub_true, LABEL_COLUMNS, hierarchy_rules=hierarchy_rules)
                            sub_edge = _edge_breakdown(sub_pred, sub_true, hierarchy_rules)
                            study_rows.append(
                                {
                                    "seed": seed,
                                    "variant_name": variant_name,
                                    "study": str(study),
                                    "anno_lv4_macro_f1": (sub_metrics.get("anno_lv4") or {}).get("macro_f1"),
                                    "full_path_accuracy": sub_hierarchy.get("full_path_accuracy"),
                                    "parent_correct_child_wrong_rate": sub_edge["parent_correct_child_wrong_rate"],
                                }
                            )
                    parent_child_rows.append(
                        {
                            "seed": seed,
                            "point": point,
                            "hierarchy_setting": hierarchy_setting,
                            "variant_name": variant_name,
                            **edge,
                        }
                    )

    comp_df = pd.DataFrame(comparison_rows)
    study_df = pd.DataFrame(study_rows)
    parent_child_df = pd.DataFrame(parent_child_rows)
    seed_summary = (
        comp_df.groupby(["variant_name", "point", "hierarchy_setting"], as_index=False)
        .agg(
            anno_lv4_macro_f1_mean=("anno_lv4_macro_f1", "mean"),
            anno_lv4_macro_f1_std=("anno_lv4_macro_f1", "std"),
            anno_lv4_balanced_accuracy_mean=("anno_lv4_balanced_accuracy", "mean"),
            anno_lv4_balanced_accuracy_std=("anno_lv4_balanced_accuracy", "std"),
            full_path_accuracy_mean=("full_path_accuracy", "mean"),
            full_path_accuracy_std=("full_path_accuracy", "std"),
            parent_correct_child_wrong_rate_mean=("parent_correct_child_wrong_rate", "mean"),
            parent_correct_child_wrong_rate_std=("parent_correct_child_wrong_rate", "std"),
            delta_vs_reranker_anno_lv4_macro_f1_mean=("delta_vs_reranker_anno_lv4_macro_f1", "mean"),
            delta_vs_reranker_full_path_accuracy_mean=("delta_vs_reranker_full_path_accuracy", "mean"),
            delta_vs_reranker_parent_correct_child_wrong_rate_mean=("delta_vs_reranker_parent_correct_child_wrong_rate", "mean"),
        )
    )
    study_summary = (
        study_df.groupby(["variant_name", "study"], as_index=False)
        .agg(
            anno_lv4_macro_f1_mean=("anno_lv4_macro_f1", "mean"),
            full_path_accuracy_mean=("full_path_accuracy", "mean"),
            parent_correct_child_wrong_rate_mean=("parent_correct_child_wrong_rate", "mean"),
        )
    )
    winner_rows = seed_summary[
        (seed_summary["point"] == "predict_100000_10000")
        & (seed_summary["hierarchy_setting"] == "on")
        & seed_summary["variant_name"].isin(["correction_joint", "correction_frozen_base"])
    ].copy()
    winner_rows["passes_reranker_gate"] = (
        (winner_rows["delta_vs_reranker_anno_lv4_macro_f1_mean"] >= 0.0)
        & (winner_rows["delta_vs_reranker_full_path_accuracy_mean"] >= 0.0)
        & (winner_rows["delta_vs_reranker_parent_correct_child_wrong_rate_mean"] <= 0.0)
    )
    viable = winner_rows[winner_rows["passes_reranker_gate"]].copy()
    if len(viable):
        viable = viable.sort_values(
            ["full_path_accuracy_mean", "parent_correct_child_wrong_rate_mean", "anno_lv4_macro_f1_mean"],
            ascending=[False, True, False],
        )
        winner = viable.iloc[0].to_dict()
    else:
        winner = {"variant_name": None, "status": "no_variant_passed_reranker_gate"}

    comp_df.to_csv(RESULTS_DIR / "phase6b_comparison.csv", index=False)
    seed_summary.to_csv(RESULTS_DIR / "phase6b_seed_summary.csv", index=False)
    study_df.to_csv(RESULTS_DIR / "phase6b_by_study.csv", index=False)
    parent_child_df.to_csv(RESULTS_DIR / "phase6b_parent_child_breakdown.csv", index=False)
    (RESULTS_DIR / "phase6b_summary.md").write_text(
        "\n".join(
            [
                "# PH-Map Phase 6B Train-Time Child Correction",
                "",
                f"- hotspot_top6: `{', '.join(HOTSPOT_TOP6)}`",
                "",
                seed_summary.to_markdown(index=False),
                "",
                "## Winner",
                "",
                "```json",
                json.dumps(winner, indent=2, sort_keys=True),
                "```",
            ]
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "phase6b_execution_record.md").write_text(
        "\n".join(
            [
                "# Phase 6B Execution Record",
                "",
                "Date: `2026-03-10`",
                "",
                "Compared baseline, reranker_top6, correction_joint, and correction_frozen_base",
                "using the PH-Map study-split seed bank.",
                "",
                "```json",
                json.dumps(winner, indent=2, sort_keys=True),
                "```",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"winner": winner, "comparison_rows": len(comp_df), "seed_summary_rows": len(seed_summary)}, indent=2))


if __name__ == "__main__":
    main()
