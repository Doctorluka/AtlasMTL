#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from anndata import read_h5ad

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atlasmtl import TrainedModel  # noqa: E402
from atlasmtl.core.data import extract_matrix  # noqa: E402
from atlasmtl.core.evaluate import (  # noqa: E402
    evaluate_hierarchy_metrics,
    evaluate_prediction_behavior,
    evaluate_predictions,
)
from atlasmtl.core.predict_utils import append_level_predictions, run_model_in_batches  # noqa: E402
from atlasmtl.core.runtime import configure_torch_threads, resolve_device  # noqa: E402
from atlasmtl.mapping import enforce_parent_child_consistency  # noqa: E402
from benchmark.methods.atlasmtl_inputs import adata_with_matrix_from_layer, resolve_atlasmtl_layer_config  # noqa: E402


DOSSIER_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_phmap_study_split_validation"
RESULTS_DIR = DOSSIER_ROOT / "results_summary"
PHASE2_MANIFEST_ROOT = DOSSIER_ROOT / "manifests" / "phase2_seed" / "predict" / "lv4strong_plus_class_weight"
PHASE2_MODEL_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-09/phmap_study_split_phase2_seed/train/lv4strong_plus_class_weight")
PHASE6C_MODEL_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-10/phmap_study_split_phase6c/train")

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
    "correction_frozen_base_reranker_like",
)


class HotspotReranker:
    def __init__(self, parent_label: str, child_names: List[str], child_full_indices: np.ndarray, model, constant_child_index=None):
        self.parent_label = parent_label
        self.child_names = child_names
        self.child_full_indices = child_full_indices
        self.model = model
        self.constant_child_index = constant_child_index

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.shape[0] == 0:
            return np.zeros((0, len(self.child_names)), dtype=np.float32)
        if self.constant_child_index is not None:
            out = np.zeros((features.shape[0], len(self.child_names)), dtype=np.float32)
            out[:, self.constant_child_index] = 1.0
            return out
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
        outputs[col] = {"logits": np.asarray(arr, dtype=np.float32), "probs": np.asarray(_softmax(arr), dtype=np.float32)}
    return outputs


def _fit_hotspot_rerankers(model: TrainedModel, *, reference_h5ad: Path, layer_cfg: Dict[str, Any], hierarchy_rules: Dict[str, Any], batch_size: int):
    from sklearn.linear_model import LogisticRegression

    ref = read_h5ad(str(reference_h5ad))
    ref_model_input = adata_with_matrix_from_layer(ref, layer_name=layer_cfg["reference_layer"])
    outputs = _base_outputs(model, ref_model_input, device="cuda", batch_size=batch_size)
    lv4_logits = outputs["anno_lv4"]["logits"]
    enc = model.label_encoders["anno_lv4"]
    child_names = np.asarray(enc.classes_, dtype=object)
    child_to_parent = {str(k): str(v) for k, v in dict((hierarchy_rules.get("anno_lv4") or {}).get("child_to_parent") or {}).items()}
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
            rerankers[parent_label] = HotspotReranker(parent_label, present_names, full_indices, None, int(unique[0]))
            continue
        clf = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=2026)
        clf.fit(X_parent, y_parent)
        rerankers[parent_label] = HotspotReranker(parent_label, present_names, full_indices, clf)
    return rerankers


def _build_pred_df(model: TrainedModel, outputs_by_col: Dict[str, Dict[str, np.ndarray]], *, hierarchy_rules: Dict[str, Any], enforce_hierarchy: bool, obs_names, rerankers) -> pd.DataFrame:
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
        "baseline": TrainedModel.load(str(PHASE2_MODEL_ROOT / f"seed_{seed}" / "runs" / "atlasmtl" / "atlasmtl_model_manifest.json")),
        "correction_frozen_base_reranker_like": TrainedModel.load(
            str(PHASE6C_MODEL_ROOT / "correction_frozen_base_reranker_like" / f"seed_{seed}" / "runs" / "atlasmtl" / "atlasmtl_model_manifest.json")
        ),
    }


def _by_study_rows(
    seed: int,
    variant_name: str,
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    hierarchy_rules: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for study, idx in true_df.groupby("study", observed=False).groups.items():
        sub_pred = pred_df.loc[idx]
        sub_true = true_df.loc[idx]
        level = evaluate_predictions(sub_pred, sub_true, ["anno_lv4"])["anno_lv4"]
        hierarchy = evaluate_hierarchy_metrics(sub_pred, sub_true, LABEL_COLUMNS, hierarchy_rules=hierarchy_rules)
        edge = {
            "parent_correct_child_wrong_rate": float(
                ((sub_true["anno_lv3"].astype(str) == sub_pred["pred_anno_lv3"].astype(str))
                 & (sub_true["anno_lv4"].astype(str) != sub_pred["pred_anno_lv4"].astype(str))).mean()
            )
        }
        rows.append(
            {
                "seed": seed,
                "variant_name": variant_name,
                "study": str(study),
                "anno_lv4_macro_f1": level["macro_f1"],
                "full_path_accuracy": hierarchy.get("full_path_accuracy", 0.0),
                "parent_correct_child_wrong_rate": edge["parent_correct_child_wrong_rate"],
            }
        )
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    by_study_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        models = _variant_models(seed)
        for point in POINTS:
            manifest = _point_manifest(seed, point)
            layer_cfg = resolve_atlasmtl_layer_config(manifest)
            ref = read_h5ad(str(manifest["reference_h5ad"]))
            query = read_h5ad(str(manifest["query_h5ad"]))
            ref_model_input = adata_with_matrix_from_layer(ref, layer_name=layer_cfg["reference_layer"])
            query_model_input = adata_with_matrix_from_layer(query, layer_name=layer_cfg["query_layer"])
            hierarchy_rules = {}
            for parent_col, child_col in zip(LABEL_COLUMNS[:-1], LABEL_COLUMNS[1:]):
                pairs = ref.obs[[parent_col, child_col]].dropna().drop_duplicates().astype(str).to_records(index=False)
                hierarchy_rules[child_col] = {
                    "parent_col": parent_col,
                    "child_to_parent": {str(child): str(parent) for parent, child in pairs},
                }
            rerankers = _fit_hotspot_rerankers(
                models["baseline"],
                reference_h5ad=Path(manifest["reference_h5ad"]),
                layer_cfg=layer_cfg,
                hierarchy_rules=hierarchy_rules,
                batch_size=512,
            )
            outputs_cache = {name: _base_outputs(bundle, query_model_input, device="cuda", batch_size=512) for name, bundle in models.items()}
            truth_df = query.obs[LABEL_COLUMNS + ["study"]].astype(str).copy()
            for hierarchy_setting in HIERARCHY_SETTINGS:
                enforce_hierarchy = hierarchy_setting == "on"
                pred_frames = {
                    "baseline": _build_pred_df(
                        models["baseline"],
                        outputs_cache["baseline"],
                        hierarchy_rules=hierarchy_rules,
                        enforce_hierarchy=enforce_hierarchy,
                        obs_names=query.obs_names,
                        rerankers={},
                    ),
                    "reranker_top6": _build_pred_df(
                        models["baseline"],
                        outputs_cache["baseline"],
                        hierarchy_rules=hierarchy_rules,
                        enforce_hierarchy=enforce_hierarchy,
                        obs_names=query.obs_names,
                        rerankers=rerankers,
                    ),
                    "correction_frozen_base_reranker_like": _build_pred_df(
                        models["correction_frozen_base_reranker_like"],
                        outputs_cache["correction_frozen_base_reranker_like"],
                        hierarchy_rules=hierarchy_rules,
                        enforce_hierarchy=enforce_hierarchy,
                        obs_names=query.obs_names,
                        rerankers={},
                    ),
                }
                for variant_name in VARIANTS:
                    pred_df = pred_frames[variant_name]
                    level_metrics = evaluate_predictions(pred_df, truth_df, LABEL_COLUMNS)
                    behavior = evaluate_prediction_behavior(pred_df, truth_df, LABEL_COLUMNS)
                    hierarchy = evaluate_hierarchy_metrics(pred_df, truth_df, LABEL_COLUMNS, hierarchy_rules=hierarchy_rules)
                    edge = _edge_breakdown(pred_df, truth_df, hierarchy_rules)
                    rows.append(
                        {
                            "seed": seed,
                            "point": point,
                            "hierarchy_setting": hierarchy_setting,
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
                    )
                    if point == "predict_100000_10000" and hierarchy_setting == "on":
                        by_study_rows.extend(_by_study_rows(seed, variant_name, pred_df, truth_df, hierarchy_rules))

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(RESULTS_DIR / "phase6c_comparison.csv", index=False)
    pd.DataFrame(by_study_rows).to_csv(RESULTS_DIR / "phase6c_by_study.csv", index=False)

    summary_df = (
        raw_df[raw_df["hierarchy_setting"] == "on"]
        .groupby(["variant_name", "point"], as_index=False)
        .agg(
            anno_lv4_macro_f1_mean=("anno_lv4_macro_f1", "mean"),
            anno_lv4_macro_f1_std=("anno_lv4_macro_f1", "std"),
            anno_lv4_balanced_accuracy_mean=("anno_lv4_balanced_accuracy", "mean"),
            anno_lv4_balanced_accuracy_std=("anno_lv4_balanced_accuracy", "std"),
            full_path_accuracy_mean=("full_path_accuracy", "mean"),
            full_path_accuracy_std=("full_path_accuracy", "std"),
            parent_correct_child_wrong_rate_mean=("parent_correct_child_wrong_rate", "mean"),
            parent_correct_child_wrong_rate_std=("parent_correct_child_wrong_rate", "std"),
        )
    )
    baseline = summary_df[summary_df["variant_name"] == "baseline"][
        ["point", "anno_lv4_macro_f1_mean", "full_path_accuracy_mean", "parent_correct_child_wrong_rate_mean"]
    ].rename(
        columns={
            "anno_lv4_macro_f1_mean": "baseline_anno_lv4_macro_f1_mean",
            "full_path_accuracy_mean": "baseline_full_path_accuracy_mean",
            "parent_correct_child_wrong_rate_mean": "baseline_parent_correct_child_wrong_rate_mean",
        }
    )
    reranker = summary_df[summary_df["variant_name"] == "reranker_top6"][
        ["point", "anno_lv4_macro_f1_mean", "full_path_accuracy_mean", "parent_correct_child_wrong_rate_mean"]
    ].rename(
        columns={
            "anno_lv4_macro_f1_mean": "reranker_anno_lv4_macro_f1_mean",
            "full_path_accuracy_mean": "reranker_full_path_accuracy_mean",
            "parent_correct_child_wrong_rate_mean": "reranker_parent_correct_child_wrong_rate_mean",
        }
    )
    summary_df = summary_df.merge(baseline, on="point", how="left").merge(reranker, on="point", how="left")
    summary_df["delta_vs_baseline_anno_lv4_macro_f1_mean"] = (
        summary_df["anno_lv4_macro_f1_mean"] - summary_df["baseline_anno_lv4_macro_f1_mean"]
    )
    summary_df["delta_vs_baseline_full_path_accuracy_mean"] = (
        summary_df["full_path_accuracy_mean"] - summary_df["baseline_full_path_accuracy_mean"]
    )
    summary_df["delta_vs_baseline_parent_correct_child_wrong_rate_mean"] = (
        summary_df["parent_correct_child_wrong_rate_mean"] - summary_df["baseline_parent_correct_child_wrong_rate_mean"]
    )
    summary_df["delta_vs_reranker_full_path_accuracy_mean"] = (
        summary_df["full_path_accuracy_mean"] - summary_df["reranker_full_path_accuracy_mean"]
    )
    summary_df["delta_vs_reranker_parent_correct_child_wrong_rate_mean"] = (
        summary_df["parent_correct_child_wrong_rate_mean"] - summary_df["reranker_parent_correct_child_wrong_rate_mean"]
    )
    summary_df.to_csv(RESULTS_DIR / "phase6c_seed_summary.csv", index=False)

    target = summary_df[(summary_df["point"] == "predict_100000_10000") & (summary_df["variant_name"] == "correction_frozen_base_reranker_like")]
    if target.empty:
        winner = "missing_phase6c_target"
    else:
        target_row = target.iloc[0]
        if (
            target_row["anno_lv4_macro_f1_mean"] >= target_row["reranker_anno_lv4_macro_f1_mean"]
            and target_row["full_path_accuracy_mean"] >= target_row["reranker_full_path_accuracy_mean"]
            and target_row["parent_correct_child_wrong_rate_mean"] <= target_row["reranker_parent_correct_child_wrong_rate_mean"]
        ):
            winner = "matched_or_beat_reranker"
        else:
            winner = "gap_not_closed"

    summary_lines = [
        "# PH-Map Phase 6C",
        "",
        "- base path: `lv4strong_plus_class_weight`",
        "- train-time mode: `frozen_base` only",
        "- correction feature mode: `reranker_like`",
        "- additional loss: local pairwise rank loss",
        "- hotspot parents: `" + ", ".join(HOTSPOT_TOP6) + "`",
        f"- decision: `{winner}`",
        "",
        "## Key table",
        "",
        summary_df.to_markdown(index=False),
    ]
    (RESULTS_DIR / "phase6c_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (RESULTS_DIR / "phase6c_execution_record.md").write_text(
        json.dumps(
            {
                "phase": "6c",
                "variants": list(VARIANTS),
                "seeds": list(SEEDS),
                "hotspot_top6": HOTSPOT_TOP6,
                "winner_status": winner,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
