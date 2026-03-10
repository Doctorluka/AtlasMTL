import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from atlasmtl.core.predict import predict
from atlasmtl.mapping.reranker import (
    ParentConditionedRefinementPlan,
    ParentConditionedReranker,
    ParentConditionedRerankerArtifact,
    discover_hotspot_parents,
)
from atlasmtl.models import ReferenceData


class DummyEncoder:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes)

    def inverse_transform(self, idx):
        return self.classes_[idx]


class DummyMultiLevelModelModule:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        rows = x.shape[0]
        parent_logits = np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float32)[:rows]
        child_logits = np.array([[4.0, 1.0, 0.5], [0.5, 0.5, 4.0]], dtype=np.float32)[:rows]
        logits = [x.new_tensor(parent_logits), x.new_tensor(child_logits)]
        return logits, {}, x


class DummyMultiLevelModel:
    def __init__(self):
        self.model = DummyMultiLevelModelModule()
        self.label_columns = ["anno_lv3", "anno_lv4"]
        self.label_encoders = {
            "anno_lv3": DummyEncoder(["P1", "P2"]),
            "anno_lv4": DummyEncoder(["P1_c1", "P1_c2", "P2_c1"]),
        }
        self.train_genes = ["g1", "g2"]
        self.coord_targets = {}
        self.coord_stats = {}
        self.reference_data = ReferenceData(
            coords={},
            labels={
                "anno_lv3": np.array(["P1", "P2"], dtype=object),
                "anno_lv4": np.array(["P1_c1", "P2_c1"], dtype=object),
            },
        )
        self.latent_source = "internal_preferred"
        self.input_transform = "binary"
        self.reference_storage = "external"
        self.reference_path = None
        self.train_config = {"input_transform": "binary"}

    @property
    def reference_coords(self):
        return self.reference_data.coords

    @property
    def reference_labels(self):
        return self.reference_data.labels


def test_parent_conditioned_reranker_artifact_save_load_and_fallback(tmp_path):
    artifact = ParentConditionedRerankerArtifact(
        parent_level="anno_lv3",
        child_level="anno_lv4",
        hotspot_parents=["P1"],
        child_classes=["P1_c1", "P1_c2", "P2_c1"],
        hierarchy_child_to_parent={"P1_c1": "P1", "P1_c2": "P1", "P2_c1": "P2"},
        rerankers={
            "P1": ParentConditionedReranker(
                parent_label="P1",
                child_names=["P1_c1", "P1_c2"],
                child_full_indices=np.array([0, 1], dtype=np.int64),
                model=None,
                constant_child_index=1,
                train_size=4,
            )
        },
        selection_metadata={"hotspot_topk": 1},
        per_parent_summary=[{"parent_label": "P1", "train_size": 4, "status": "constant_single_child"}],
    )

    out_path = tmp_path / "reranker.pkl"
    artifact.save(out_path)
    loaded = ParentConditionedRerankerArtifact.load(out_path)
    assert loaded.hotspot_parents == ["P1"]
    assert loaded.selection_metadata["hotspot_topk"] == 1
    assert out_path.with_suffix(".json").exists()

    probs, meta = loaded.apply(
        child_logits=np.array([[3.0, 1.0, 0.5]], dtype=np.float32),
        parent_pred_labels=np.array(["P1"], dtype=object),
        child_classes=["other_a", "other_b", "other_c"],
    )
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(1, dtype=np.float32))
    assert meta["fallback_to_base"] is True
    assert meta["fallback_reason"] == "child_class_mismatch"


def test_predict_supports_parent_conditioned_reranker_refinement():
    model = DummyMultiLevelModel()
    query = AnnData(X=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), obs=pd.DataFrame(index=["q1", "q2"]))
    query.var_names = ["g1", "g2"]

    artifact = ParentConditionedRerankerArtifact(
        parent_level="anno_lv3",
        child_level="anno_lv4",
        hotspot_parents=["P1"],
        child_classes=["P1_c1", "P1_c2", "P2_c1"],
        hierarchy_child_to_parent={"P1_c1": "P1", "P1_c2": "P1", "P2_c1": "P2"},
        rerankers={
            "P1": ParentConditionedReranker(
                parent_label="P1",
                child_names=["P1_c1", "P1_c2"],
                child_full_indices=np.array([0, 1], dtype=np.int64),
                model=None,
                constant_child_index=1,
                train_size=4,
            )
        },
        selection_metadata={"hotspot_topk": 1},
        per_parent_summary=[],
    )

    result = predict(
        model,
        query,
        knn_correction="off",
        confidence_low=-1.0,
        batch_size=2,
        device="cpu",
        refinement_config={
            "method": "parent_conditioned_reranker",
            "artifact": artifact,
        },
    )

    pred = result.predictions
    assert pred.loc["q1", "pred_anno_lv3"] == "P1"
    assert pred.loc["q2", "pred_anno_lv3"] == "P2"
    assert pred.loc["q1", "pred_anno_lv4"] == "P1_c2"
    assert pred.loc["q2", "pred_anno_lv4"] == "P2_c1"
    assert result.metadata["refinement"]["method"] == "parent_conditioned_reranker"
    assert result.metadata["refinement"]["status"]["num_refined_cells"] == 1


def test_predict_rejects_unknown_refinement_method():
    model = DummyMultiLevelModel()
    query = AnnData(X=np.array([[1.0, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["q1"]))
    query.var_names = ["g1", "g2"]

    with pytest.raises(ValueError, match="parent_conditioned_reranker"):
        predict(
            model,
            query,
            device="cpu",
            refinement_config={"method": "unknown"},
        )


def test_discover_hotspot_parents_supports_topk_and_cumulative():
    df = pd.DataFrame(
        {
            "parent_label": ["A", "B", "C", "D"],
            "parent_correct_child_wrong_rate": [0.30, 0.20, 0.10, 0.05],
            "n_cells": [100, 120, 50, 10],
        }
    )

    ranked_topk, selected_topk, summary_topk = discover_hotspot_parents(
        df,
        selection_mode="topk",
        top_k=2,
        min_cells_per_parent=20,
    )
    assert selected_topk == ["A", "B"]
    assert summary_topk["selected_parent_count"] == 2
    assert "D" not in ranked_topk["parent_label"].tolist()

    ranked_cum, selected_cum, summary_cum = discover_hotspot_parents(
        df,
        selection_mode="cumulative_contribution",
        cumulative_target=0.6,
        min_cells_per_parent=20,
    )
    assert ranked_cum.iloc[0]["parent_label"] == "A"
    assert selected_cum == ["A", "B"]
    assert summary_cum["selected_parent_count"] == 2


def test_refinement_plan_roundtrip_and_auto_predict(tmp_path):
    model = DummyMultiLevelModel()
    query = AnnData(X=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), obs=pd.DataFrame(index=["q1", "q2"]))
    query.var_names = ["g1", "g2"]

    artifact = ParentConditionedRerankerArtifact(
        parent_level="anno_lv3",
        child_level="anno_lv4",
        hotspot_parents=["P1"],
        child_classes=["P1_c1", "P1_c2", "P2_c1"],
        hierarchy_child_to_parent={"P1_c1": "P1", "P1_c2": "P1", "P2_c1": "P2"},
        rerankers={
            "P1": ParentConditionedReranker(
                parent_label="P1",
                child_names=["P1_c1", "P1_c2"],
                child_full_indices=np.array([0, 1], dtype=np.int64),
                model=None,
                constant_child_index=1,
                train_size=4,
            )
        },
        selection_metadata={"hotspot_topk": 1},
        per_parent_summary=[],
    )
    artifact_path = tmp_path / "reranker.pkl"
    artifact.save(artifact_path)

    plan = ParentConditionedRefinementPlan(
        enabled=True,
        method="auto_parent_conditioned_reranker",
        parent_level="anno_lv3",
        child_level="anno_lv4",
        selection_source="test",
        selection_point="predict",
        selection_score="rate*n",
        selection_mode="topk",
        selected_parents=["P1"],
        artifact_path=str(artifact_path),
        top_k=1,
        min_cells_per_parent=0,
    )
    plan_path = tmp_path / "plan.json"
    plan.save(plan_path)

    loaded_plan = ParentConditionedRefinementPlan.load(plan_path)
    assert loaded_plan.selected_parents == ["P1"]
    assert loaded_plan.artifact_path == str(artifact_path)

    result = predict(
        model,
        query,
        knn_correction="off",
        confidence_low=-1.0,
        batch_size=2,
        device="cpu",
        refinement_config={
            "method": "auto_parent_conditioned_reranker",
            "plan_path": str(plan_path),
        },
    )

    pred = result.predictions
    assert pred.loc["q1", "pred_anno_lv4"] == "P1_c2"
    assert pred.loc["q2", "pred_anno_lv4"] == "P2_c1"
    assert result.metadata["refinement"]["method"] == "auto_parent_conditioned_reranker"
    assert result.metadata["refinement"]["plan"]["selected_parents"] == ["P1"]
    assert result.metadata["refinement"]["status"]["num_refined_cells"] == 1
