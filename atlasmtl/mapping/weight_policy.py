from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


DEFAULT_MILD_LV4_SCHEDULE = [0.7, 0.8, 1.0, 1.8]
DEFAULT_STRONG_LV4_SCHEDULE = [0.2, 0.7, 1.5, 3.0]
DEFAULT_MILD_LV3_SCHEDULE = [0.8, 1.0, 1.8]
DEFAULT_STRONG_LV3_SCHEDULE = [0.5, 1.0, 3.0]
DEFAULT_MILD_LV5_SCHEDULE = [0.7, 0.8, 1.0, 1.2, 2.0]
DEFAULT_STRONG_LV5_SCHEDULE = [0.4, 0.6, 0.9, 1.3, 3.0]


@dataclass
class TaskWeightActivationDecision:
    activate_nonuniform_weighting: bool
    recommended_schedule_name: str
    recommended_schedule: Optional[List[float]]
    candidate_space: List[str]
    candidate_schedules: Dict[str, List[float]]
    decision_note: str
    activation_features: Dict[str, Any]
    activation_rule_version: str = "activation_rule_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activate_nonuniform_weighting": bool(self.activate_nonuniform_weighting),
            "recommended_schedule_name": str(self.recommended_schedule_name),
            "recommended_schedule": None
            if self.recommended_schedule is None
            else [float(x) for x in self.recommended_schedule],
            "candidate_space": [str(x) for x in self.candidate_space],
            "candidate_schedules": {
                str(name): [float(x) for x in schedule]
                for name, schedule in self.candidate_schedules.items()
            },
            "decision_note": str(self.decision_note),
            "activation_features": dict(self.activation_features),
            "activation_rule_version": str(self.activation_rule_version),
        }


@dataclass
class ParentConditionedRefinementActivationDecision:
    activate_refinement: bool
    recommended_selection_mode: str
    recommended_top_k: Optional[int]
    recommended_cumulative_target: Optional[float]
    recommended_guardrail_profile: str
    decision_note: str
    activation_features: Dict[str, Any]
    activation_rule_version: str = "refinement_activation_rule_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activate_refinement": bool(self.activate_refinement),
            "recommended_selection_mode": str(self.recommended_selection_mode),
            "recommended_top_k": None if self.recommended_top_k is None else int(self.recommended_top_k),
            "recommended_cumulative_target": None
            if self.recommended_cumulative_target is None
            else float(self.recommended_cumulative_target),
            "recommended_guardrail_profile": str(self.recommended_guardrail_profile),
            "decision_note": str(self.decision_note),
            "activation_features": dict(self.activation_features),
            "activation_rule_version": str(self.activation_rule_version),
        }


def _default_candidate_schedules(n_levels: int) -> Dict[str, List[float]]:
    if n_levels == 3:
        return {
            "uniform": [1.0, 1.0, 1.0],
            "mild_lv3": list(DEFAULT_MILD_LV3_SCHEDULE),
            "strong_lv3": list(DEFAULT_STRONG_LV3_SCHEDULE),
        }
    if n_levels == 4:
        return {
            "uniform": [1.0, 1.0, 1.0, 1.0],
            "mild_lv4": list(DEFAULT_MILD_LV4_SCHEDULE),
            "strong_lv4": list(DEFAULT_STRONG_LV4_SCHEDULE),
        }
    if n_levels == 5:
        return {
            "uniform": [1.0, 1.0, 1.0, 1.0, 1.0],
            "mild_lv5": list(DEFAULT_MILD_LV5_SCHEDULE),
            "strong_lv5": list(DEFAULT_STRONG_LV5_SCHEDULE),
        }
    raise ValueError("n_levels must currently be 3, 4, or 5")


def suggest_task_weight_schedule(
    *,
    n_levels: int,
    finest_macro_f1: float,
    finest_balanced_accuracy: float,
    full_path_accuracy: float,
    parent_correct_child_wrong_rate: float,
    coarse_to_fine_headroom_gap: Optional[float] = None,
    hierarchy_on_off_macro_f1_gap: Optional[float] = None,
    hotspot_concentration_score: Optional[float] = None,
    activation_rule_version: str = "activation_rule_v1",
) -> TaskWeightActivationDecision:
    core_metrics = {
        "finest_macro_f1": finest_macro_f1,
        "finest_balanced_accuracy": finest_balanced_accuracy,
        "full_path_accuracy": full_path_accuracy,
        "parent_correct_child_wrong_rate": parent_correct_child_wrong_rate,
    }
    missing = [name for name, value in core_metrics.items() if value is None]
    if missing:
        raise ValueError(f"missing required activation metrics: {', '.join(missing)}")

    candidate_schedules = _default_candidate_schedules(n_levels)
    finest_macro_f1 = float(finest_macro_f1)
    finest_balanced_accuracy = float(finest_balanced_accuracy)
    full_path_accuracy = float(full_path_accuracy)
    parent_correct_child_wrong_rate = float(parent_correct_child_wrong_rate)
    full_path_vs_finest_gap = finest_macro_f1 - full_path_accuracy
    coarse_gap = None if coarse_to_fine_headroom_gap is None else float(coarse_to_fine_headroom_gap)
    hierarchy_gap = (
        None if hierarchy_on_off_macro_f1_gap is None else float(hierarchy_on_off_macro_f1_gap)
    )
    hotspot_score = None if hotspot_concentration_score is None else float(hotspot_concentration_score)

    difficulty_reasons: List[str] = []
    if coarse_gap is not None and coarse_gap >= 0.10:
        difficulty_reasons.append(f"coarse_to_fine_headroom_gap={coarse_gap:.4f}")
    if finest_balanced_accuracy <= 0.70:
        difficulty_reasons.append(f"finest_balanced_accuracy={finest_balanced_accuracy:.4f}")
    if finest_macro_f1 <= 0.70:
        difficulty_reasons.append(f"finest_macro_f1={finest_macro_f1:.4f}")

    structural_reasons: List[str] = []
    if parent_correct_child_wrong_rate >= 0.08:
        structural_reasons.append(
            f"parent_correct_child_wrong_rate={parent_correct_child_wrong_rate:.4f}"
        )
    if full_path_vs_finest_gap >= 0.10:
        structural_reasons.append(f"full_path_vs_finest_gap={full_path_vs_finest_gap:.4f}")
    if hierarchy_gap is not None and hierarchy_gap >= 0.01:
        structural_reasons.append(f"hierarchy_on_off_macro_f1_gap={hierarchy_gap:.4f}")
    if hotspot_score is not None and hotspot_score >= 0.45 and parent_correct_child_wrong_rate >= 0.05:
        structural_reasons.append(
            f"hotspot_concentration_score={hotspot_score:.4f} with parent_correct_child_wrong_rate={parent_correct_child_wrong_rate:.4f}"
        )

    activate = bool(difficulty_reasons and structural_reasons)
    if activate:
        recommended_schedule_name = "needs_candidate_test"
        recommended_schedule = None
        decision_note = (
            "activate non-uniform weighting because baseline shows fine-level difficulty ("
            + "; ".join(difficulty_reasons)
            + ") and structural tradeoff ("
            + "; ".join(structural_reasons)
            + ")"
        )
        candidate_space = [name for name in candidate_schedules.keys()]
    else:
        recommended_schedule_name = "uniform"
        recommended_schedule = list(candidate_schedules["uniform"])
        reason_parts: List[str] = []
        if not difficulty_reasons:
            reason_parts.append("no strong fine-level difficulty trigger")
        if not structural_reasons:
            reason_parts.append("no strong structural tradeoff trigger")
        decision_note = "keep uniform weighting because " + " and ".join(reason_parts)
        candidate_space = ["uniform"]

    activation_features = {
        "n_levels": int(n_levels),
        "finest_macro_f1": finest_macro_f1,
        "finest_balanced_accuracy": finest_balanced_accuracy,
        "full_path_accuracy": full_path_accuracy,
        "parent_correct_child_wrong_rate": parent_correct_child_wrong_rate,
        "coarse_to_fine_headroom_gap": coarse_gap,
        "full_path_vs_finest_gap": full_path_vs_finest_gap,
        "hierarchy_on_off_macro_f1_gap": hierarchy_gap,
        "hotspot_concentration_score": hotspot_score,
        "difficulty_reasons": list(difficulty_reasons),
        "structural_reasons": list(structural_reasons),
    }

    return TaskWeightActivationDecision(
        activate_nonuniform_weighting=activate,
        recommended_schedule_name=recommended_schedule_name,
        recommended_schedule=recommended_schedule,
        candidate_space=candidate_space,
        candidate_schedules=candidate_schedules,
        decision_note=decision_note,
        activation_features=activation_features,
        activation_rule_version=activation_rule_version,
    )


def suggest_parent_conditioned_refinement(
    *,
    n_levels: int,
    finest_macro_f1: float,
    full_path_accuracy: float,
    parent_correct_child_wrong_rate: float,
    path_break_rate: float,
    hotspot_concentration_score: float,
    activation_rule_version: str = "refinement_activation_rule_v1",
) -> ParentConditionedRefinementActivationDecision:
    core_metrics = {
        "finest_macro_f1": finest_macro_f1,
        "full_path_accuracy": full_path_accuracy,
        "parent_correct_child_wrong_rate": parent_correct_child_wrong_rate,
        "path_break_rate": path_break_rate,
        "hotspot_concentration_score": hotspot_concentration_score,
    }
    missing = [name for name, value in core_metrics.items() if value is None]
    if missing:
        raise ValueError(f"missing required refinement activation metrics: {', '.join(missing)}")

    finest_macro_f1 = float(finest_macro_f1)
    full_path_accuracy = float(full_path_accuracy)
    parent_correct_child_wrong_rate = float(parent_correct_child_wrong_rate)
    path_break_rate = float(path_break_rate)
    hotspot_concentration_score = float(hotspot_concentration_score)
    full_path_vs_finest_gap = finest_macro_f1 - full_path_accuracy

    base_trigger = (
        parent_correct_child_wrong_rate >= 0.05 and hotspot_concentration_score >= 0.40
    )
    secondary_reasons: List[str] = []
    if full_path_vs_finest_gap >= 0.05:
        secondary_reasons.append(f"full_path_vs_finest_gap={full_path_vs_finest_gap:.4f}")
    if finest_macro_f1 <= 0.75:
        secondary_reasons.append(f"finest_macro_f1={finest_macro_f1:.4f}")

    activate = bool(base_trigger and secondary_reasons)
    if activate:
        recommended_selection_mode = "topk"
        recommended_top_k = 6 if int(n_levels) == 4 else 4
        recommended_cumulative_target = None
        recommended_guardrail_profile = "phmap_v1"
        decision_note = (
            "activate parent-conditioned refinement because baseline shows concentrated "
            f"parent-correct/child-wrong errors (parent_correct_child_wrong_rate={parent_correct_child_wrong_rate:.4f}, "
            f"hotspot_concentration_score={hotspot_concentration_score:.4f}) and secondary trigger(s): "
            + "; ".join(secondary_reasons)
        )
    else:
        recommended_selection_mode = "none"
        recommended_top_k = None
        recommended_cumulative_target = None
        recommended_guardrail_profile = "none"
        reason_parts: List[str] = []
        if not base_trigger:
            reason_parts.append(
                "baseline does not show sufficient concentrated parent-conditioned child errors"
            )
        if not secondary_reasons:
            reason_parts.append("secondary refinement trigger not met")
        decision_note = "skip parent-conditioned refinement because " + " and ".join(reason_parts)

    activation_features = {
        "n_levels": int(n_levels),
        "finest_macro_f1": finest_macro_f1,
        "full_path_accuracy": full_path_accuracy,
        "full_path_vs_finest_gap": full_path_vs_finest_gap,
        "parent_correct_child_wrong_rate": parent_correct_child_wrong_rate,
        "path_break_rate": path_break_rate,
        "hotspot_concentration_score": hotspot_concentration_score,
        "base_trigger": bool(base_trigger),
        "secondary_reasons": list(secondary_reasons),
    }

    return ParentConditionedRefinementActivationDecision(
        activate_refinement=activate,
        recommended_selection_mode=recommended_selection_mode,
        recommended_top_k=recommended_top_k,
        recommended_cumulative_target=recommended_cumulative_target,
        recommended_guardrail_profile=recommended_guardrail_profile,
        decision_note=decision_note,
        activation_features=activation_features,
        activation_rule_version=activation_rule_version,
    )
