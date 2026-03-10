from __future__ import annotations

import pytest

from atlasmtl.mapping import suggest_parent_conditioned_refinement, suggest_task_weight_schedule


def test_suggest_task_weight_schedule_activates_for_phmap_like_case() -> None:
    decision = suggest_task_weight_schedule(
        n_levels=4,
        finest_macro_f1=0.549254,
        finest_balanced_accuracy=0.550158,
        full_path_accuracy=0.4548,
        parent_correct_child_wrong_rate=0.12348,
        coarse_to_fine_headroom_gap=0.401,
        hierarchy_on_off_macro_f1_gap=0.0201,
        hotspot_concentration_score=0.6333,
    )

    assert decision.activate_nonuniform_weighting is True
    assert decision.recommended_schedule_name == "needs_candidate_test"
    assert decision.recommended_schedule is None
    assert decision.candidate_space == ["uniform", "mild_lv4", "strong_lv4"]


def test_suggest_task_weight_schedule_keeps_uniform_for_hlca_like_case() -> None:
    decision = suggest_task_weight_schedule(
        n_levels=5,
        finest_macro_f1=0.688732,
        finest_balanced_accuracy=0.682491,
        full_path_accuracy=0.8239,
        parent_correct_child_wrong_rate=0.0334,
        coarse_to_fine_headroom_gap=None,
        hierarchy_on_off_macro_f1_gap=None,
        hotspot_concentration_score=0.8425,
    )

    assert decision.activate_nonuniform_weighting is False
    assert decision.recommended_schedule_name == "uniform"
    assert decision.recommended_schedule == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert decision.candidate_space == ["uniform"]


def test_suggest_task_weight_schedule_rejects_missing_required_metrics() -> None:
    with pytest.raises(ValueError, match="missing required activation metrics"):
        suggest_task_weight_schedule(
            n_levels=4,
            finest_macro_f1=None,  # type: ignore[arg-type]
            finest_balanced_accuracy=0.7,
            full_path_accuracy=0.6,
            parent_correct_child_wrong_rate=0.1,
        )


def test_suggest_task_weight_schedule_returns_level_specific_candidate_space() -> None:
    decision = suggest_task_weight_schedule(
        n_levels=5,
        finest_macro_f1=0.55,
        finest_balanced_accuracy=0.52,
        full_path_accuracy=0.40,
        parent_correct_child_wrong_rate=0.12,
        coarse_to_fine_headroom_gap=0.20,
        hierarchy_on_off_macro_f1_gap=0.03,
        hotspot_concentration_score=0.6,
    )

    assert decision.activate_nonuniform_weighting is True
    assert decision.candidate_space == ["uniform", "mild_lv5", "strong_lv5"]
    assert decision.candidate_schedules["strong_lv5"] == [0.4, 0.6, 0.9, 1.3, 3.0]


def test_suggest_task_weight_schedule_supports_three_level_negative_case() -> None:
    decision = suggest_task_weight_schedule(
        n_levels=3,
        finest_macro_f1=0.848143,
        finest_balanced_accuracy=0.84609,
        full_path_accuracy=0.9334,
        parent_correct_child_wrong_rate=0.0299,
        coarse_to_fine_headroom_gap=0.142097,
        hierarchy_on_off_macro_f1_gap=None,
        hotspot_concentration_score=0.765886,
    )

    assert decision.activate_nonuniform_weighting is False
    assert decision.recommended_schedule_name == "uniform"
    assert decision.recommended_schedule == [1.0, 1.0, 1.0]


def test_suggest_parent_conditioned_refinement_activates_for_phmap_like_case() -> None:
    decision = suggest_parent_conditioned_refinement(
        n_levels=4,
        finest_macro_f1=0.587355,
        full_path_accuracy=0.46664,
        parent_correct_child_wrong_rate=0.09504,
        path_break_rate=0.0,
        hotspot_concentration_score=0.63,
    )

    assert decision.activate_refinement is True
    assert decision.recommended_selection_mode == "topk"
    assert decision.recommended_top_k == 6
    assert decision.recommended_guardrail_profile == "phmap_v1"


def test_suggest_parent_conditioned_refinement_stays_off_for_hlca_like_case() -> None:
    decision = suggest_parent_conditioned_refinement(
        n_levels=5,
        finest_macro_f1=0.688732,
        full_path_accuracy=0.8239,
        parent_correct_child_wrong_rate=0.0334,
        path_break_rate=0.0,
        hotspot_concentration_score=0.84,
    )

    assert decision.activate_refinement is False
    assert decision.recommended_selection_mode == "none"
