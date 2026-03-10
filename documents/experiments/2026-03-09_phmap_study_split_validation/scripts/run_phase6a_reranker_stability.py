#!/usr/bin/env python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
PHASE2_MANIFEST_ROOT = DOSSIER_ROOT / "manifests" / "phase2_seed" / "predict" / "lv4strong_plus_class_weight"
PHASE2_TMP_ROOT = Path("/tmp/atlasmtl_benchmarks/2026-03-09/phmap_study_split_phase2_seed")
PHASE3_SUBTREE_PATH = RESULTS_DIR / "phase3_tradeoff_subtree_breakdown.csv"

LABEL_COLUMNS = ["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"]
POINTS = ("build_100000_eval10k", "predict_100000_10000")
HOTSPOT_COUNTS = (2, 4, 6)
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
        return np.asarray(self.model.predict_proba(features), dtype=np.float32)


def _seed_dirs() -> List[Tuple[int, Path]]:
    root = PHASE2_TMP_ROOT / "train" / "lv4strong_plus_class_weight"
    rows: List[Tuple[int, Path]] = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        seed = int(path.name.replace("seed_", ""))
        model_manifest = path / "runs" / "atlasmtl" / "atlasmtl_model_manifest.json"
        if model_manifest.exists():
            rows.append((seed, model_manifest))
    return rows


def _point_manifest(seed: int, point: str) -> Dict[str, Any]:
    path = PHASE2_MANIFEST_ROOT / f"seed_{seed}" / point / "atlasmtl_phase2_seed_predict.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


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


def _load_hotspot_sets() -> Dict[str, List[str]]:
    df = pd.read_csv(PHASE3_SUBTREE_PATH)
    df = df[
        (df["config_name"] == "lv4strong_plus_class_weight")
        & (df["point"] == "predict_100000_10000")
        & (df["hierarchy_setting"] == "on")
    ].copy()
    df["contribution"] = df["parent_correct_child_wrong_rate"] * df["n_cells"]
    ranked = (
        df.sort_values(["contribution", "parent_correct_child_wrong_rate", "n_cells"], ascending=[False, False, False])[
            "parent_label"
        ]
        .astype(str)
        .tolist()
    )
    hotspot_sets: Dict[str, List[str]] = {}
    for count in HOTSPOT_COUNTS:
        hotspot_sets[f"reranker_top{count}"] = ranked[:count]
    return hotspot_sets


def _fit_hotspot_rerankers(
    model: TrainedModel,
    *,
    reference_h5ad: Path,
    layer_cfg: Dict[str, Any],
    hierarchy_rules: Dict[str, Any],
    batch_size: int,
    hotspot_parents: Iterable[str],
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
    for parent_label in hotspot_parents:
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
            rerankers[str(parent_label)] = HotspotReranker(
                parent_label=str(parent_label),
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
        rerankers[str(parent_label)] = HotspotReranker(
            parent_label=str(parent_label),
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
        "parent_correct_child_wrong_rate": float((parent_correct & ~child_correct).mean()),
        "path_break_rate": float(path_break.mean()),
    }


def _variant_order() -> List[str]:
    return ["baseline"] + [f"reranker_top{count}" for count in HOTSPOT_COUNTS]


def _summarize_by_study(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["variant_name", "study"], as_index=False)
        .agg(
            anno_lv4_macro_f1_mean=("anno_lv4_macro_f1", "mean"),
            full_path_accuracy_mean=("full_path_accuracy", "mean"),
            parent_correct_child_wrong_rate_mean=("parent_correct_child_wrong_rate", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    baseline = grouped[grouped["variant_name"] == "baseline"][
        ["study", "anno_lv4_macro_f1_mean", "full_path_accuracy_mean", "parent_correct_child_wrong_rate_mean"]
    ].rename(
        columns={
            "anno_lv4_macro_f1_mean": "anno_lv4_macro_f1_mean_baseline",
            "full_path_accuracy_mean": "full_path_accuracy_mean_baseline",
            "parent_correct_child_wrong_rate_mean": "parent_correct_child_wrong_rate_mean_baseline",
        }
    )
    merged = grouped.merge(baseline, on="study", how="left")
    merged["delta_macro_f1"] = merged["anno_lv4_macro_f1_mean"] - merged["anno_lv4_macro_f1_mean_baseline"]
    merged["delta_full_path_accuracy"] = merged["full_path_accuracy_mean"] - merged["full_path_accuracy_mean_baseline"]
    merged["delta_parent_correct_child_wrong_rate"] = (
        merged["parent_correct_child_wrong_rate_mean"] - merged["parent_correct_child_wrong_rate_mean_baseline"]
    )
    return merged.sort_values(["variant_name", "study"]).reset_index(drop=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hotspot_sets = _load_hotspot_sets()
    seed_models = _seed_dirs()

    comparison_rows: List[Dict[str, Any]] = []
    study_rows: List[Dict[str, Any]] = []
    parent_child_rows: List[Dict[str, Any]] = []

    for seed, model_manifest in seed_models:
        model = TrainedModel.load(str(model_manifest))
        anchor_manifest = _point_manifest(seed, "predict_100000_10000")
        anchor_layer_cfg = resolve_atlasmtl_layer_config(anchor_manifest)
        anchor_rules = (anchor_manifest.get("predict") or {}).get("hierarchy_rules") or {}
        reranker_library = {
            name: _fit_hotspot_rerankers(
                model,
                reference_h5ad=Path(anchor_manifest["reference_h5ad"]),
                layer_cfg=anchor_layer_cfg,
                hierarchy_rules=anchor_rules,
                batch_size=int((anchor_manifest.get("predict") or {}).get("batch_size", 512)),
                hotspot_parents=parents,
            )
            for name, parents in hotspot_sets.items()
        }

        for point in POINTS:
            manifest = _point_manifest(seed, point)
            hierarchy_rules = (manifest.get("predict") or {}).get("hierarchy_rules") or {}
            layer_cfg = resolve_atlasmtl_layer_config(manifest)
            query = read_h5ad(str(manifest["query_h5ad"]))
            query_model_input = adata_with_matrix_from_layer(query, layer_name=layer_cfg["query_layer"])
            outputs_by_col = _base_outputs(
                model,
                query_model_input,
                device="cuda",
                batch_size=int((manifest.get("predict") or {}).get("batch_size", 512)),
            )
            true_df = query.obs.loc[:, LABEL_COLUMNS + ["study"]].copy()
            variant_payloads = {"baseline": {}}
            variant_payloads.update(reranker_library)

            baseline_record: Dict[str, Any] | None = None
            for variant_name in _variant_order():
                rerankers = variant_payloads.get(variant_name, {})
                pred_df = _build_pred_df(
                    model,
                    outputs_by_col,
                    hierarchy_rules=hierarchy_rules,
                    enforce_hierarchy=True,
                    obs_names=query.obs_names,
                    rerankers=rerankers,
                )
                metrics = evaluate_predictions(pred_df, true_df[LABEL_COLUMNS], LABEL_COLUMNS)
                behavior = evaluate_prediction_behavior(pred_df, true_df[LABEL_COLUMNS], LABEL_COLUMNS)
                hierarchy = evaluate_hierarchy_metrics(
                    pred_df,
                    true_df[LABEL_COLUMNS],
                    LABEL_COLUMNS,
                    hierarchy_rules=hierarchy_rules,
                )
                edge = _edge_breakdown(
                    pred_df,
                    true_df[LABEL_COLUMNS],
                    parent_col="anno_lv3",
                    child_col="anno_lv4",
                    child_to_parent={str(k): str(v) for k, v in dict(hierarchy_rules["anno_lv4"]["child_to_parent"]).items()},
                )
                lv4 = metrics["anno_lv4"]
                lv4_behavior = behavior["anno_lv4"]
                mean_path_consistency = ((hierarchy.get("edges") or {}).get("anno_lv4") or {}).get("path_consistency_rate")
                row = {
                    "seed": seed,
                    "point": point,
                    "hierarchy_setting": "on",
                    "variant_name": variant_name,
                    "hotspot_parents": "|".join(hotspot_sets.get(variant_name, [])),
                    "anno_lv4_macro_f1": lv4.get("macro_f1"),
                    "anno_lv4_balanced_accuracy": lv4.get("balanced_accuracy"),
                    "anno_lv4_coverage": lv4.get("coverage"),
                    "anno_lv4_unknown_rate": lv4_behavior.get("unknown_rate"),
                    "full_path_accuracy": hierarchy.get("full_path_accuracy"),
                    "full_path_coverage": hierarchy.get("full_path_coverage"),
                    "mean_path_consistency_rate": mean_path_consistency,
                    "parent_correct_child_wrong_rate": edge["parent_correct_child_wrong_rate"],
                    "path_break_rate": edge["path_break_rate"],
                }
                if baseline_record is None:
                    baseline_record = row
                for key in (
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
                    row[f"{key}_baseline"] = baseline_record[key]
                    row[f"delta_vs_baseline_{key}"] = row[key] - baseline_record[key]
                comparison_rows.append(row)

                if point == "predict_100000_10000":
                    for study, idx in true_df.groupby("study", observed=False).groups.items():
                        idx = list(idx)
                        sub_pred = pred_df.loc[idx]
                        sub_true = true_df.loc[idx, LABEL_COLUMNS]
                        sub_metrics = evaluate_predictions(sub_pred, sub_true, LABEL_COLUMNS)
                        sub_hierarchy = evaluate_hierarchy_metrics(
                            sub_pred,
                            sub_true,
                            LABEL_COLUMNS,
                            hierarchy_rules=hierarchy_rules,
                        )
                        sub_edge = _edge_breakdown(
                            sub_pred,
                            sub_true,
                            parent_col="anno_lv3",
                            child_col="anno_lv4",
                            child_to_parent={str(k): str(v) for k, v in dict(hierarchy_rules["anno_lv4"]["child_to_parent"]).items()},
                        )
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
                        "variant_name": variant_name,
                        "parent_col": "anno_lv3",
                        "child_col": "anno_lv4",
                        **edge,
                    }
                )

    comp_df = pd.DataFrame(comparison_rows)
    summary_df = (
        comp_df.groupby(["variant_name", "point"], as_index=False)
        .agg(
            anno_lv4_macro_f1_mean=("anno_lv4_macro_f1", "mean"),
            anno_lv4_macro_f1_std=("anno_lv4_macro_f1", "std"),
            anno_lv4_balanced_accuracy_mean=("anno_lv4_balanced_accuracy", "mean"),
            anno_lv4_balanced_accuracy_std=("anno_lv4_balanced_accuracy", "std"),
            full_path_accuracy_mean=("full_path_accuracy", "mean"),
            full_path_accuracy_std=("full_path_accuracy", "std"),
            anno_lv4_coverage_mean=("anno_lv4_coverage", "mean"),
            anno_lv4_coverage_std=("anno_lv4_coverage", "std"),
            parent_correct_child_wrong_rate_mean=("parent_correct_child_wrong_rate", "mean"),
            parent_correct_child_wrong_rate_std=("parent_correct_child_wrong_rate", "std"),
            path_break_rate_mean=("path_break_rate", "mean"),
            path_break_rate_std=("path_break_rate", "std"),
            delta_vs_baseline_anno_lv4_macro_f1_mean=("delta_vs_baseline_anno_lv4_macro_f1", "mean"),
            delta_vs_baseline_full_path_accuracy_mean=("delta_vs_baseline_full_path_accuracy", "mean"),
            delta_vs_baseline_parent_correct_child_wrong_rate_mean=(
                "delta_vs_baseline_parent_correct_child_wrong_rate",
                "mean",
            ),
        )
    )
    hotspot_df = summary_df[summary_df["point"] == "predict_100000_10000"].copy()
    study_df = pd.DataFrame(study_rows)
    study_summary_df = _summarize_by_study(study_df)
    parent_child_df = pd.DataFrame(parent_child_rows)

    stable_candidates = hotspot_df[hotspot_df["variant_name"] != "baseline"].copy()
    stable_candidates["passes_gate"] = (
        (stable_candidates["delta_vs_baseline_anno_lv4_macro_f1_mean"] > 0.0)
        & (stable_candidates["delta_vs_baseline_full_path_accuracy_mean"] > 0.0)
        & (stable_candidates["delta_vs_baseline_parent_correct_child_wrong_rate_mean"] < 0.0)
    )
    selected = stable_candidates[stable_candidates["passes_gate"]].copy()
    if len(selected):
        selected = selected.sort_values(
            [
                "full_path_accuracy_mean",
                "delta_vs_baseline_parent_correct_child_wrong_rate_mean",
                "anno_lv4_macro_f1_mean",
            ],
            ascending=[False, True, False],
        )
        winner = selected.iloc[0].to_dict()
    else:
        winner = {"variant_name": None, "status": "no_variant_passed_gate"}

    comp_df.to_csv(RESULTS_DIR / "phase6a_seed_comparison.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "phase6a_seed_summary.csv", index=False)
    hotspot_df.to_csv(RESULTS_DIR / "phase6a_hotspot_sensitivity.csv", index=False)
    study_df.to_csv(RESULTS_DIR / "phase6a_by_study.csv", index=False)
    parent_child_df.to_csv(RESULTS_DIR / "phase6a_parent_child_breakdown.csv", index=False)

    (RESULTS_DIR / "phase6a_seed_stability.md").write_text(
        "\n".join(
            [
                "# PH-Map Phase 6A Seed Stability",
                "",
                f"- seeds: `{', '.join(str(seed) for seed, _ in seed_models)}`",
                f"- hotspot variants: `{', '.join(_variant_order())}`",
                "",
                summary_df.to_markdown(index=False),
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
    (RESULTS_DIR / "phase6a_hotspot_sensitivity.md").write_text(
        "\n".join(
            [
                "# PH-Map Phase 6A Hotspot Sensitivity",
                "",
                "Hotspot parents were ranked from Phase 3 baseline `predict_100000_10000 + hierarchy_on` by `parent_correct_child_wrong_rate * n_cells`.",
                "",
                "```json",
                json.dumps(hotspot_sets, indent=2, sort_keys=True),
                "```",
                "",
                hotspot_df.to_markdown(index=False),
            ]
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "phase6a_by_study_summary.md").write_text(
        "\n".join(
            [
                "# PH-Map Phase 6A By-Study Stability",
                "",
                study_summary_df.to_markdown(index=False),
            ]
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "phase6a_execution_record.md").write_text(
        "\n".join(
            [
                "# Phase 6A Execution Record",
                "",
                "Date: `2026-03-10`",
                "",
                "This round reuses the completed Phase 2 `lv4strong_plus_class_weight` seed models and evaluates",
                "parent-conditioned rerankers for `top2`, `top4`, and `top6` hotspot parent sets.",
                "",
                "## Hotspot Sets",
                "",
                "```json",
                json.dumps(hotspot_sets, indent=2, sort_keys=True),
                "```",
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
    print(
        json.dumps(
            {
                "seed_count": len(seed_models),
                "comparison_rows": len(comp_df),
                "study_rows": len(study_df),
                "hotspot_variants": list(hotspot_sets.keys()),
                "winner": winner,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
