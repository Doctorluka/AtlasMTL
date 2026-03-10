from __future__ import annotations

import json
import pickle
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.linear_model import LogisticRegression

from ..core.data import extract_matrix
from ..core.predict_utils import run_model_in_batches
from ..core.runtime import configure_torch_threads, resolve_device
from ..core.types import TrainedModel


@dataclass
class ParentConditionedReranker:
    parent_label: str
    child_names: List[str]
    child_full_indices: np.ndarray
    model: LogisticRegression | None
    constant_child_index: int | None = None
    train_size: int = 0

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.shape[0] == 0:
            return np.zeros((0, len(self.child_names)), dtype=np.float32)
        if self.constant_child_index is not None:
            out = np.zeros((features.shape[0], len(self.child_names)), dtype=np.float32)
            out[:, self.constant_child_index] = 1.0
            return out
        assert self.model is not None
        return np.asarray(self.model.predict_proba(features), dtype=np.float32)


@dataclass
class ParentConditionedRerankerArtifact:
    parent_level: str
    child_level: str
    hotspot_parents: List[str]
    child_classes: List[str]
    hierarchy_child_to_parent: Dict[str, str]
    rerankers: Dict[str, ParentConditionedReranker]
    selection_metadata: Dict[str, Any]
    per_parent_summary: List[Dict[str, Any]]
    hierarchy_child_to_parent_hash: Optional[str] = None
    label_space_hash: Optional[str] = None
    selection_metadata_version: str = "selection_metadata_v1"

    def __post_init__(self) -> None:
        if self.hierarchy_child_to_parent_hash is None:
            self.hierarchy_child_to_parent_hash = _stable_hash_payload(self.hierarchy_child_to_parent)
        if self.label_space_hash is None:
            self.label_space_hash = _stable_hash_payload(
                {
                    "child_classes": list(self.child_classes),
                    "hierarchy_child_to_parent_hash": self.hierarchy_child_to_parent_hash,
                }
            )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            pickle.dump(self, handle)
        manifest = {
            "artifact_path": str(target.resolve()),
            "method": "parent_conditioned_reranker",
            "parent_level": self.parent_level,
            "child_level": self.child_level,
            "hotspot_parents": list(self.hotspot_parents),
            "selection_metadata": self.selection_metadata,
            "selection_metadata_version": self.selection_metadata_version,
            "label_columns": [self.parent_level, self.child_level],
            "child_classes": list(self.child_classes),
            "hierarchy_child_to_parent": dict(self.hierarchy_child_to_parent),
            "hierarchy_child_to_parent_hash": self.hierarchy_child_to_parent_hash,
            "label_space_hash": self.label_space_hash,
            "per_parent_summary": self.per_parent_summary,
        }
        target.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ParentConditionedRerankerArtifact":
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def apply(
        self,
        *,
        child_logits: np.ndarray,
        parent_pred_labels: np.ndarray,
        child_classes: List[str],
        hierarchy_child_to_parent: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if list(child_classes) != list(self.child_classes):
            return _softmax(child_logits), {
                "applied": False,
                "fallback_to_base": True,
                "fallback_reason": "child_class_mismatch",
                "num_refined_cells": 0,
            }
        if hierarchy_child_to_parent is not None:
            input_hash = _stable_hash_payload({str(k): str(v) for k, v in hierarchy_child_to_parent.items()})
            if input_hash != self.hierarchy_child_to_parent_hash:
                return _softmax(child_logits), {
                    "applied": False,
                    "fallback_to_base": True,
                    "fallback_reason": "hierarchy_child_to_parent_hash_mismatch",
                    "num_refined_cells": 0,
                }
        base_probs = _softmax(child_logits)
        refined_probs = base_probs.copy()
        refined_cells = 0
        used_parents: List[str] = []
        for parent_label, reranker in self.rerankers.items():
            mask = np.asarray(parent_pred_labels, dtype=object) == str(parent_label)
            if not np.any(mask):
                continue
            parent_probs = reranker.predict_proba(child_logits[mask][:, reranker.child_full_indices])
            full_probs = np.zeros((parent_probs.shape[0], refined_probs.shape[1]), dtype=np.float32)
            full_probs[:, reranker.child_full_indices] = parent_probs
            refined_probs[mask] = full_probs
            refined_cells += int(mask.sum())
            used_parents.append(parent_label)
        return refined_probs, {
            "applied": True,
            "fallback_to_base": False,
            "fallback_reason": None,
            "num_refined_cells": refined_cells,
            "used_parent_count": len(used_parents),
            "used_parents": used_parents,
        }


@dataclass
class ParentConditionedRefinementPlan:
    enabled: bool
    method: str
    parent_level: str
    child_level: str
    selection_source: str
    selection_point: str
    selection_score: str
    selection_mode: str
    selected_parents: List[str]
    artifact_path: Optional[str]
    top_k: Optional[int] = None
    cumulative_target: Optional[float] = None
    min_cells_per_parent: int = 0
    fallback_to_base: bool = True
    guardrail: Optional[Dict[str, Any]] = None
    ranking_path: Optional[str] = None
    per_parent_summary_path: Optional[str] = None
    selection_metadata_version: str = "selection_metadata_v1"
    hierarchy_child_to_parent_hash: Optional[str] = None
    label_space_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "method": str(self.method),
            "parent_level": str(self.parent_level),
            "child_level": str(self.child_level),
            "selection_source": str(self.selection_source),
            "selection_point": str(self.selection_point),
            "selection_score": str(self.selection_score),
            "selection_mode": str(self.selection_mode),
            "selected_parents": [str(x) for x in self.selected_parents],
            "artifact_path": self.artifact_path,
            "top_k": self.top_k,
            "cumulative_target": self.cumulative_target,
            "min_cells_per_parent": int(self.min_cells_per_parent),
            "fallback_to_base": bool(self.fallback_to_base),
            "guardrail": dict(self.guardrail or {}),
            "ranking_path": self.ranking_path,
            "per_parent_summary_path": self.per_parent_summary_path,
            "selection_metadata_version": str(self.selection_metadata_version),
            "hierarchy_child_to_parent_hash": self.hierarchy_child_to_parent_hash,
            "label_space_hash": self.label_space_hash,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ParentConditionedRefinementPlan":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            enabled=bool(payload.get("enabled", True)),
            method=str(payload.get("method") or "auto_parent_conditioned_reranker"),
            parent_level=str(payload["parent_level"]),
            child_level=str(payload["child_level"]),
            selection_source=str(payload.get("selection_source") or ""),
            selection_point=str(payload.get("selection_point") or ""),
            selection_score=str(payload.get("selection_score") or ""),
            selection_mode=str(payload.get("selection_mode") or "topk"),
            selected_parents=[str(x) for x in list(payload.get("selected_parents") or [])],
            artifact_path=payload.get("artifact_path"),
            top_k=None if payload.get("top_k") is None else int(payload["top_k"]),
            cumulative_target=None
            if payload.get("cumulative_target") is None
            else float(payload["cumulative_target"]),
            min_cells_per_parent=int(payload.get("min_cells_per_parent", 0)),
            fallback_to_base=bool(payload.get("fallback_to_base", True)),
            guardrail=dict(payload.get("guardrail") or {}),
            ranking_path=payload.get("ranking_path"),
            per_parent_summary_path=payload.get("per_parent_summary_path"),
            selection_metadata_version=str(payload.get("selection_metadata_version", "selection_metadata_v1")),
            hierarchy_child_to_parent_hash=payload.get("hierarchy_child_to_parent_hash"),
            label_space_hash=payload.get("label_space_hash"),
        )


def _stable_hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_refinement_guardrail_profile(name: str) -> Dict[str, Any]:
    if name == "phmap_v1":
        return {
            "name": "phmap_v1",
            "version": 1,
            "thresholds": {
                "child_macro_f1_delta_min": 0.0,
                "full_path_accuracy_delta_min": 0.0,
                "parent_correct_child_wrong_rate_delta_max": 0.0,
            },
            "rules": [
                "child_macro_f1 >= base",
                "full_path_accuracy >= base",
                "parent_correct_child_wrong_rate <= base",
            ],
        }
    if name == "none":
        return {
            "name": "none",
            "version": 1,
            "thresholds": {},
            "rules": [],
        }
    raise ValueError("refinement_guardrail_profile must be 'phmap_v1' or 'none'")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def discover_hotspot_parents(
    subtree_breakdown: pd.DataFrame,
    *,
    parent_label_col: str = "parent_label",
    rate_col: str = "parent_correct_child_wrong_rate",
    n_cells_col: str = "n_cells",
    selection_mode: str = "topk",
    top_k: int = 6,
    cumulative_target: float = 0.6,
    min_cells_per_parent: int = 0,
    max_selected_parents: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    if parent_label_col not in subtree_breakdown.columns:
        raise ValueError(f"missing parent_label_col: {parent_label_col}")
    if rate_col not in subtree_breakdown.columns:
        raise ValueError(f"missing rate_col: {rate_col}")
    if n_cells_col not in subtree_breakdown.columns:
        raise ValueError(f"missing n_cells_col: {n_cells_col}")
    if selection_mode not in {"topk", "cumulative_contribution"}:
        raise ValueError("selection_mode must be 'topk' or 'cumulative_contribution'")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not (0.0 < cumulative_target <= 1.0):
        raise ValueError("cumulative_target must be in (0, 1]")
    if max_selected_parents is not None and int(max_selected_parents) <= 0:
        raise ValueError("max_selected_parents must be > 0 when provided")

    df = subtree_breakdown.copy()
    df[parent_label_col] = df[parent_label_col].astype(str)
    df[n_cells_col] = pd.to_numeric(df[n_cells_col], errors="coerce").fillna(0).astype(float)
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0).astype(float)
    df = df[df[n_cells_col] >= float(min_cells_per_parent)].copy()
    df["selection_score"] = df[rate_col] * df[n_cells_col]
    df = df.sort_values(
        ["selection_score", rate_col, n_cells_col, parent_label_col],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    total = float(df["selection_score"].sum())
    if total > 0:
        df["cumulative_contribution"] = df["selection_score"].cumsum() / total
    else:
        df["cumulative_contribution"] = 0.0

    if selection_mode == "topk":
        selected_df = df.head(int(top_k)).copy()
    else:
        selected_df = df[df["cumulative_contribution"] <= float(cumulative_target)].copy()
        if selected_df.empty and not df.empty:
            selected_df = df.head(1).copy()
        if not df.empty:
            next_row = df.iloc[[len(selected_df)]].copy() if len(selected_df) < len(df) else pd.DataFrame()
            if not next_row.empty and float(selected_df["cumulative_contribution"].max() if not selected_df.empty else 0.0) < float(cumulative_target):
                selected_df = pd.concat([selected_df, next_row], ignore_index=True)
        if max_selected_parents is not None:
            selected_df = selected_df.head(int(max_selected_parents)).copy()

    selected_parents = selected_df[parent_label_col].astype(str).tolist()
    summary = {
        "selection_mode": selection_mode,
        "top_k": int(top_k),
        "cumulative_target": float(cumulative_target),
        "min_cells_per_parent": int(min_cells_per_parent),
        "max_selected_parents": None if max_selected_parents is None else int(max_selected_parents),
        "candidate_parent_count": int(len(df)),
        "selected_parent_count": int(len(selected_parents)),
        "total_selection_score": total,
        "selected_selection_score": float(selected_df["selection_score"].sum()) if not selected_df.empty else 0.0,
        "enabled": bool(len(selected_parents) > 0),
        "fallback_to_base": bool(len(selected_parents) == 0),
        "fallback_reason": None if len(selected_parents) > 0 else "no_hotspot_parents_after_filtering",
    }
    return df, selected_parents, summary


def build_parent_conditioned_refinement_plan(
    *,
    parent_level: str,
    child_level: str,
    selection_source: str,
    selection_point: str,
    selection_score: str,
    selection_mode: str,
    selected_parents: List[str],
    artifact_path: Optional[str],
    top_k: Optional[int] = None,
    cumulative_target: Optional[float] = None,
    min_cells_per_parent: int = 0,
    fallback_to_base: bool = True,
    guardrail: Optional[Dict[str, Any]] = None,
    ranking_path: Optional[str] = None,
    per_parent_summary_path: Optional[str] = None,
    selection_metadata_version: str = "selection_metadata_v1",
    hierarchy_child_to_parent_hash: Optional[str] = None,
    label_space_hash: Optional[str] = None,
) -> ParentConditionedRefinementPlan:
    return ParentConditionedRefinementPlan(
        enabled=True,
        method="auto_parent_conditioned_reranker",
        parent_level=parent_level,
        child_level=child_level,
        selection_source=selection_source,
        selection_point=selection_point,
        selection_score=selection_score,
        selection_mode=selection_mode,
        selected_parents=[str(x) for x in selected_parents],
        artifact_path=artifact_path,
        top_k=top_k,
        cumulative_target=cumulative_target,
        min_cells_per_parent=min_cells_per_parent,
        fallback_to_base=fallback_to_base,
        guardrail=dict(guardrail or {}),
        ranking_path=ranking_path,
        per_parent_summary_path=per_parent_summary_path,
        selection_metadata_version=selection_metadata_version,
        hierarchy_child_to_parent_hash=hierarchy_child_to_parent_hash,
        label_space_hash=label_space_hash,
    )


def _base_outputs(model: TrainedModel, adata: AnnData, *, batch_size: int, device: str) -> Dict[str, Dict[str, np.ndarray]]:
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
    calibration = ((model.train_config or {}).get("calibration") or {}).get("temperatures", {})
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    for idx, col in enumerate(model.label_columns):
        arr = logits[idx].numpy()
        temp = calibration.get(col)
        if temp:
            arr = arr / float(temp)
        outputs[col] = {"logits": np.asarray(arr, dtype=np.float32), "probs": np.asarray(_softmax(arr), dtype=np.float32)}
    return outputs


def fit_parent_conditioned_reranker(
    model: TrainedModel,
    reference_adata: AnnData,
    *,
    parent_level: str,
    child_level: str,
    hotspot_parents: List[str],
    hierarchy_rules: Dict[str, Dict[str, Any]],
    batch_size: int = 512,
    device: str = "auto",
    selection_metadata: Optional[Dict[str, Any]] = None,
) -> ParentConditionedRerankerArtifact:
    if parent_level not in model.label_columns or child_level not in model.label_columns:
        raise ValueError("parent_level and child_level must exist in model.label_columns")
    outputs = _base_outputs(model, reference_adata, batch_size=batch_size, device=device)
    child_logits = outputs[child_level]["logits"]
    child_encoder = model.label_encoders[child_level]
    child_classes = np.asarray(child_encoder.classes_, dtype=object)
    child_to_parent = {
        str(k): str(v)
        for k, v in dict((hierarchy_rules.get(child_level) or {}).get("child_to_parent") or {}).items()
    }
    rerankers: Dict[str, ParentConditionedReranker] = {}
    summaries: List[Dict[str, Any]] = []
    for parent_label in hotspot_parents:
        legal_mask = np.array([child_to_parent.get(str(name)) == str(parent_label) for name in child_classes], dtype=bool)
        legal_indices = np.where(legal_mask)[0]
        legal_names = child_classes[legal_indices].tolist()
        ref_mask = reference_adata.obs[parent_level].astype(str).to_numpy() == str(parent_label)
        X_parent = child_logits[ref_mask][:, legal_indices]
        y_parent_names = reference_adata.obs.loc[ref_mask, child_level].astype(str).to_numpy()
        present_names = [name for name in legal_names if np.any(y_parent_names == name)]
        if not present_names:
            summaries.append(
                {
                    "parent_label": str(parent_label),
                    "train_size": 0,
                    "status": "skipped_no_observed_children",
                    "legal_child_count": len(legal_names),
                }
            )
            continue
        present_indices = np.array([legal_names.index(name) for name in present_names], dtype=np.int64)
        X_parent = X_parent[:, present_indices]
        full_indices = legal_indices[present_indices]
        y_parent = np.array([present_names.index(name) for name in y_parent_names], dtype=np.int64)
        unique = np.unique(y_parent)
        if unique.size == 1:
            reranker = ParentConditionedReranker(
                parent_label=str(parent_label),
                child_names=present_names,
                child_full_indices=full_indices,
                model=None,
                constant_child_index=int(unique[0]),
                train_size=int(len(y_parent)),
            )
            status = "constant_single_child"
        else:
            clf = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=2026)
            clf.fit(X_parent, y_parent)
            reranker = ParentConditionedReranker(
                parent_label=str(parent_label),
                child_names=present_names,
                child_full_indices=full_indices,
                model=clf,
                constant_child_index=None,
                train_size=int(len(y_parent)),
            )
            status = "fit_ok"
        rerankers[str(parent_label)] = reranker
        summaries.append(
            {
                "parent_label": str(parent_label),
                "train_size": int(len(y_parent)),
                "status": status,
                "legal_child_count": int(len(legal_names)),
                "active_child_count": int(len(present_names)),
            }
        )

    return ParentConditionedRerankerArtifact(
        parent_level=parent_level,
        child_level=child_level,
        hotspot_parents=[str(x) for x in hotspot_parents],
        child_classes=[str(x) for x in child_classes.tolist()],
        hierarchy_child_to_parent=child_to_parent,
        rerankers=rerankers,
        selection_metadata=dict(selection_metadata or {}),
        per_parent_summary=summaries,
        selection_metadata_version="selection_metadata_v1",
    )
