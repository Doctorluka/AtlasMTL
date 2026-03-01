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


class OpensetDummyModelModule:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        rows = x.shape[0]
        # High-confidence logits for class A.
        logits_template = np.array([[10.0, 0.0]], dtype=np.float32)
        logits = [x.new_tensor(np.repeat(logits_template, rows, axis=0))]
        coords = {"latent": x.new_tensor(np.repeat(np.array([[10.0, 10.0]], dtype=np.float32), rows, axis=0))}
        return logits, coords, x


class DummyTrainedModelForOpenset:
    def __init__(self):
        self.model = OpensetDummyModelModule()
        self.label_columns = ["celltype"]
        self.label_encoders = {"celltype": DummyEncoder(["A", "B"])}
        self.train_genes = ["g1", "g2"]
        self.coord_targets = {"latent": "X_ref_latent"}
        self.coord_stats = {"latent": {"mean": np.zeros(2, dtype=np.float32), "std": np.ones(2, dtype=np.float32)}}
        self.reference_data = ReferenceData(
            coords={"X_ref_latent": np.array([[0.0, 0.0], [0.1, 0.1]], dtype=np.float32)},
            labels={"celltype": np.array(["A", "B"], dtype=object)},
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


def test_predict_openset_nn_distance_forces_unknown_and_records_metadata():
    model = DummyTrainedModelForOpenset()
    adata = AnnData(X=np.array([[1.0, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1", "g2"]

    result = predict(
        model,
        adata,
        knn_correction="off",
        confidence_low=-1.0,
        openset_method="nn_distance",
        openset_threshold=1.0,
        device="cpu",
    )

    assert result.predictions.loc["c1", "pred_celltype_raw"] == "A"
    assert result.predictions.loc["c1", "pred_celltype"] == "Unknown"
    assert result.metadata["openset_method"] == "nn_distance"
    assert result.metadata["openset_threshold"] == 1.0
    assert result.metadata["openset_space_used"] in {"latent", "umap", "none"}
    assert result.metadata["openset_unknown_rate"] == 1.0

