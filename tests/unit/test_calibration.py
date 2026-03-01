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


class CalibDummyModelModule:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        rows = x.shape[0]
        logits_template = np.array([[2.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        logits = [x.new_tensor(logits_template[:rows])]
        coords = {"latent": x.new_tensor(np.zeros((rows, 2), dtype=np.float32))}
        return logits, coords, x


class DummyTrainedModelWithCalibration:
    def __init__(self, temperature: float):
        self.model = CalibDummyModelModule()
        self.label_columns = ["celltype"]
        self.label_encoders = {"celltype": DummyEncoder(["A", "B"])}
        self.train_genes = ["g1", "g2"]
        self.coord_targets = {"latent": "X_ref_latent"}
        self.coord_stats = {"latent": {"mean": np.zeros(2, dtype=np.float32), "std": np.ones(2, dtype=np.float32)}}
        self.reference_data = ReferenceData(
            coords={"X_ref_latent": np.zeros((2, 2), dtype=np.float32)},
            labels={"celltype": np.array(["A", "B"], dtype=object)},
        )
        self.latent_source = "internal_preferred"
        self.input_transform = "binary"
        self.reference_storage = "full"
        self.reference_path = None
        self.train_config = {
            "input_transform": "binary",
            "calibration": {
                "method": "temperature_scaling",
                "split": "val",
                "temperatures": {"celltype": float(temperature)},
            },
        }

    @property
    def reference_coords(self):
        return self.reference_data.coords

    @property
    def reference_labels(self):
        return self.reference_data.labels


def test_predict_temperature_scaling_softens_confidence_and_records_metadata():
    model = DummyTrainedModelWithCalibration(temperature=2.0)
    adata = AnnData(X=np.array([[1.0, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1", "g2"]

    uncal = predict(model, adata, knn_correction="off", confidence_low=-1.0, apply_calibration=False, device="cpu")
    cal = predict(model, adata, knn_correction="off", confidence_low=-1.0, apply_calibration=True, device="cpu")

    assert uncal.metadata["calibration_applied"] is False
    assert cal.metadata["calibration_applied"] is True
    assert cal.metadata["calibration_method"] == "temperature_scaling"
    assert cal.metadata["temperature_celltype"] == 2.0

    # With logits [2,0], T=2 makes probabilities less peaked.
    assert cal.predictions.loc["c1", "conf_celltype"] < uncal.predictions.loc["c1", "conf_celltype"]
    assert cal.predictions.loc["c1", "pred_celltype"] == uncal.predictions.loc["c1", "pred_celltype"]

