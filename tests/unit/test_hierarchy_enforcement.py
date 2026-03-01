import numpy as np
import pandas as pd
from anndata import AnnData

from atlasmtl.core.api import predict
from atlasmtl.models import ReferenceData


class DummyEncoder:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes)

    def inverse_transform(self, idx):
        return self.classes_[idx]


class HierDummyModelModule:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        rows = x.shape[0]
        # Two heads: parent predicts P1, child predicts C_bad (inconsistent with P1).
        parent_logits = np.array([[10.0, 0.0]], dtype=np.float32)
        child_logits = np.array([[10.0, 0.0]], dtype=np.float32)
        logits = [
            x.new_tensor(np.repeat(parent_logits, rows, axis=0)),
            x.new_tensor(np.repeat(child_logits, rows, axis=0)),
        ]
        coords = {"latent": x.new_tensor(np.zeros((rows, 2), dtype=np.float32))}
        return logits, coords, x


class DummyHierModel:
    def __init__(self):
        self.model = HierDummyModelModule()
        self.label_columns = ["parent", "child"]
        self.label_encoders = {
            "parent": DummyEncoder(["P1", "P2"]),
            "child": DummyEncoder(["C_bad", "C_good"]),
        }
        self.train_genes = ["g1"]
        self.coord_targets = {"latent": "X_ref_latent"}
        self.coord_stats = {"latent": {"mean": np.zeros(2, dtype=np.float32), "std": np.ones(2, dtype=np.float32)}}
        self.reference_data = ReferenceData(
            coords={"X_ref_latent": np.zeros((2, 2), dtype=np.float32)},
            labels={"parent": np.array(["P1", "P2"], dtype=object), "child": np.array(["C_bad", "C_good"], dtype=object)},
        )
        self.latent_source = "internal_preferred"
        self.input_transform = "binary"
        self.reference_storage = "full"
        self.reference_path = None
        self.train_config = {"input_transform": "binary"}

    @property
    def reference_coords(self):
        return self.reference_data.coords

    @property
    def reference_labels(self):
        return self.reference_data.labels


def test_predict_can_enforce_hierarchy_by_unknowning_inconsistent_child():
    model = DummyHierModel()
    adata = AnnData(X=np.array([[1.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1"]

    rules = {"child": {"parent_col": "parent", "child_to_parent": {"C_good": "P1", "C_bad": "P2"}}}
    result = predict(
        model,
        adata,
        knn_correction="off",
        confidence_low=-1.0,
        enforce_hierarchy=True,
        hierarchy_rules=rules,
        device="cpu",
    )

    assert result.predictions.loc["c1", "pred_parent"] == "P1"
    assert result.predictions.loc["c1", "pred_child_raw"] == "C_bad"
    assert result.predictions.loc["c1", "pred_child"] == "Unknown"
    assert result.metadata["hierarchy_enforced"] is True
