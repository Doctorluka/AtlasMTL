import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from atlasmtl.core.api import _extract_matrix, predict
from atlasmtl.models import ReferenceData
from atlasmtl.io import get_prediction_columns


class DummyEncoder:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes)

    def inverse_transform(self, idx):
        return self.classes_[idx]


class DummyModelModule:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        rows = x.shape[0]
        logits_template = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        coords_template = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        logits = [
            x.new_tensor(logits_template[:rows])
        ]
        coords = {
            "latent": x.new_tensor(coords_template[:rows])
        }
        return logits, coords, x


class DummyTrainedModel:
    def __init__(self):
        self.model = DummyModelModule()
        self.label_columns = ["celltype"]
        self.label_encoders = {"celltype": DummyEncoder(["A", "B"])}
        self.train_genes = ["g1", "g2"]
        self.coord_targets = {"latent": "X_ref_latent"}
        self.coord_stats = {"latent": {"mean": np.zeros(2, dtype=np.float32), "std": np.ones(2, dtype=np.float32)}}
        self.reference_data = ReferenceData(
            coords={"X_ref_latent": np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0]], dtype=np.float32)},
            labels={"celltype": np.array(["A", "B", "B"], dtype=object)},
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


def test_extract_matrix_binary_transform():
    adata = AnnData(X=np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32))
    adata.var_names = ["g1", "g2"]
    out = _extract_matrix(adata, input_transform="binary")
    assert np.array_equal(out, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))


def test_predict_knn_off_keeps_raw_labels():
    model = DummyTrainedModel()
    adata = AnnData(X=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1", "c2"]))
    adata.var_names = ["g1", "g2"]
    result = predict(model, adata, knn_correction="off", confidence_low=-1.0, device="cpu")
    pred = result.predictions
    assert not pred["used_knn_celltype"].any()
    assert np.array_equal(pred["pred_celltype_raw"].values, pred["pred_celltype"].values)


def test_predict_closed_loop_unknown_uses_knn_confidence():
    model = DummyTrainedModel()
    adata = AnnData(X=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1", "c2"]))
    adata.var_names = ["g1", "g2"]
    result = predict(
        model,
        adata,
        knn_correction="low_conf_only",
        confidence_high=0.9,
        confidence_low=-1.0,
        margin_threshold=0.9,
        knn_k=2,
        knn_conf_low=0.75,
        knn_index_mode="pynndescent",
        device="cpu",
    )
    pred = result.predictions
    assert pred["used_knn_celltype"].all()
    assert pred.loc["c1", "pred_celltype"] == "Unknown"
    assert pred.loc["c2", "pred_celltype"] == "B"


def test_to_adata_records_metadata():
    model = DummyTrainedModel()
    adata = AnnData(X=np.array([[1.0, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1", "g2"]
    result = predict(model, adata, knn_correction="off", confidence_low=-1.0, device="cpu")
    out = result.to_adata(adata)
    assert out.uns["atlasmtl"]["input_transform"] == "binary"
    assert "knn_space_used" in out.uns["atlasmtl"]
    assert "prediction_runtime" in out.uns["atlasmtl"]
    assert out.uns["atlasmtl"]["prediction_runtime"]["phase"] == "predict"


def test_get_prediction_columns_by_mode():
    predictions = pd.DataFrame(
        {
            "pred_celltype": ["A"],
            "pred_celltype_raw": ["A"],
            "pred_celltype_knn": ["A"],
            "conf_celltype": [0.9],
            "margin_celltype": [0.8],
            "is_unknown_celltype": [False],
            "is_low_conf_celltype": [False],
            "used_knn_celltype": [False],
            "knn_vote_frac_celltype": [np.nan],
        }
    )

    assert get_prediction_columns(predictions, "minimal") == ["pred_celltype"]
    assert get_prediction_columns(predictions, "standard") == [
        "pred_celltype",
        "conf_celltype",
        "margin_celltype",
        "is_unknown_celltype",
    ]
    assert get_prediction_columns(predictions, "full") == list(predictions.columns)


def test_to_adata_minimal_writes_only_final_labels():
    model = DummyTrainedModel()
    adata = AnnData(X=np.array([[1.0, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1", "g2"]
    result = predict(model, adata, knn_correction="off", confidence_low=-1.0, device="cpu")
    out = result.to_adata(adata, mode="minimal", include_metadata=False)
    assert "pred_celltype" in out.obs.columns
    assert "conf_celltype" not in out.obs.columns
    assert "pred_celltype_raw" not in out.obs.columns
    assert "atlasmtl" not in out.uns


def test_to_adata_standard_excludes_debug_and_coords():
    model = DummyTrainedModel()
    adata = AnnData(X=np.array([[1.0, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1", "g2"]
    result = predict(model, adata, knn_correction="off", confidence_low=-1.0, device="cpu")
    out = result.to_adata(adata, mode="standard")
    assert "pred_celltype" in out.obs.columns
    assert "conf_celltype" in out.obs.columns
    assert "is_unknown_celltype" in out.obs.columns
    assert "pred_celltype_raw" not in out.obs.columns
    assert "X_pred_latent" not in out.obsm


def test_to_adata_full_can_write_coords():
    model = DummyTrainedModel()
    adata = AnnData(X=np.array([[1.0, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1", "g2"]
    result = predict(model, adata, knn_correction="off", confidence_low=-1.0, device="cpu")
    out = result.to_adata(adata, mode="full", include_coords=True)
    assert "pred_celltype_raw" in out.obs.columns
    assert "pred_celltype_knn" in out.obs.columns
    assert "X_pred_latent" in out.obsm


def test_to_adata_rewrites_columns_when_mode_changes():
    model = DummyTrainedModel()
    adata = AnnData(X=np.array([[1.0, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1", "g2"]
    result = predict(model, adata, knn_correction="off", confidence_low=-1.0, device="cpu")
    out = result.to_adata(adata, mode="full")
    assert "pred_celltype_raw" in out.obs.columns
    out = result.to_adata(out, mode="minimal", include_metadata=False)
    assert "pred_celltype" in out.obs.columns
    assert "pred_celltype_raw" not in out.obs.columns
    assert "conf_celltype" not in out.obs.columns


@pytest.mark.parametrize(
    ("mode", "required_columns"),
    [
        ("minimal", {"pred_celltype"}),
        ("standard", {"pred_celltype", "conf_celltype", "margin_celltype", "is_unknown_celltype"}),
        (
            "full",
            {
                "pred_celltype",
                "pred_celltype_raw",
                "pred_celltype_knn",
                "conf_celltype",
                "margin_celltype",
                "is_unknown_celltype",
                "is_low_conf_celltype",
                "used_knn_celltype",
                "knn_vote_frac_celltype",
            },
        ),
    ],
)
def test_to_dataframe_matches_mode_selection(mode, required_columns):
    model = DummyTrainedModel()
    adata = AnnData(X=np.array([[1.0, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1", "g2"]
    result = predict(model, adata, knn_correction="off", confidence_low=-1.0, device="cpu")

    frame = result.to_dataframe(mode=mode)

    assert list(frame.columns) == get_prediction_columns(result.predictions, mode)
    assert required_columns.issubset(frame.columns)
    assert list(frame.index) == ["c1"]


def test_to_csv_writes_selected_columns(tmp_path):
    model = DummyTrainedModel()
    adata = AnnData(X=np.array([[1.0, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1", "g2"]
    result = predict(model, adata, knn_correction="off", confidence_low=-1.0, device="cpu")

    output_path = tmp_path / "predictions.csv"
    result.to_csv(output_path, mode="minimal")

    exported = pd.read_csv(output_path, index_col=0)
    assert list(exported.columns) == ["pred_celltype"]
    assert exported.loc["c1", "pred_celltype"] in {"A", "B", "Unknown"}


def test_prediction_result_resource_usage_helpers(capsys):
    model = DummyTrainedModel()
    adata = AnnData(X=np.array([[1.0, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1", "g2"]
    result = predict(model, adata, knn_correction="off", confidence_low=-1.0, device="cpu")

    usage = result.get_resource_usage()
    assert usage["phase"] == "predict"
    assert usage["device_used"] == "cpu"

    result.show_resource_usage()
    captured = capsys.readouterr()
    assert "atlasmtl prediction resource usage" in captured.out
    assert "Summary:" in captured.out
    assert "Execution:" in captured.out
    assert "Prediction:" in captured.out
    assert "device_used" in captured.out
    assert "cpu" in captured.out


def test_predict_can_print_summary_automatically(capsys):
    model = DummyTrainedModel()
    adata = AnnData(X=np.array([[1.0, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1", "g2"]

    predict(
        model,
        adata,
        knn_correction="off",
        confidence_low=-1.0,
        device="cpu",
        show_summary=True,
    )

    captured = capsys.readouterr()
    assert "atlasmtl prediction resource usage" in captured.out
