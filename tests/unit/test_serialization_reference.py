from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from atlasmtl.core.model import AtlasMTLModel
from atlasmtl.core.types import TrainedModel
from atlasmtl.models import ReferenceData, default_feature_panel_path, default_manifest_path, load_manifest


def _make_model(reference_storage="external"):
    encoder = LabelEncoder()
    encoder.fit(["A", "B"])
    return TrainedModel(
        model=AtlasMTLModel(input_size=2, num_classes=[2], hidden_sizes=[4], coord_dims={"latent": 2}),
        label_columns=["celltype"],
        label_encoders={"celltype": encoder},
        train_genes=["g1", "g2"],
        coord_targets={"latent": "X_ref_latent"},
        coord_stats={"latent": {"mean": np.zeros(2, dtype=np.float32), "std": np.ones(2, dtype=np.float32)}},
        reference_data=ReferenceData(
            coords={"X_ref_latent": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)},
            labels={"celltype": np.array(["A", "B"], dtype=object)},
        ),
        input_transform="binary",
        reference_storage=reference_storage,
        train_config={},
    )


def test_trained_model_external_reference_roundtrip(tmp_path):
    model = _make_model(reference_storage="external")
    model_path = tmp_path / "model.pth"
    model.save(str(model_path))

    assert (tmp_path / "model_reference.pkl").exists()
    assert Path(default_manifest_path(str(model_path))).exists()
    loaded = TrainedModel.load(str(model_path))
    assert loaded.reference_storage == "external"
    assert "X_ref_latent" in loaded.reference_coords
    assert loaded.reference_labels["celltype"][0] == "A"


def test_trained_model_full_reference_roundtrip(tmp_path):
    model = _make_model(reference_storage="full")
    model_path = tmp_path / "model.pth"
    model.save(str(model_path))

    assert not (tmp_path / "model_reference.pkl").exists()
    loaded = TrainedModel.load(str(model_path))
    assert loaded.reference_storage == "full"
    assert "X_ref_latent" in loaded.reference_coords


def test_manifest_records_artifact_layout_and_supports_loading(tmp_path):
    model = _make_model(reference_storage="external")
    model_path = tmp_path / "model.pth"
    model.save(str(model_path))

    manifest_path = Path(default_manifest_path(str(model_path)))
    manifest = load_manifest(str(manifest_path))

    assert manifest["schema_version"] == 1
    assert manifest["model_path"] == "model.pth"
    assert manifest["metadata_path"] == "model_metadata.pkl"
    assert manifest["reference_path"] == "model_reference.pkl"
    assert manifest["reference_storage"] == "external"

    loaded = TrainedModel.load(str(manifest_path))
    assert loaded.reference_storage == "external"
    assert loaded.reference_labels["celltype"][1] == "B"


def test_feature_panel_is_persisted_as_independent_artifact(tmp_path):
    model = _make_model(reference_storage="external")
    model.train_config = {
        "preprocess": {
            "config": {
                "var_names_type": "ensembl",
                "species": "human",
                "feature_space": "whole",
                "n_top_genes": 3000,
            },
            "feature_panel": {
                "gene_ids": ["g1", "g2"],
                "gene_symbols": ["G1", "G2"],
                "feature_space": "whole",
                "n_features": 2,
                "species": "human",
                "var_names_type_original": "ensembl",
            },
        }
    }
    model_path = tmp_path / "model.pth"
    model.save(str(model_path))

    manifest = load_manifest(str(default_manifest_path(str(model_path))))
    assert manifest["feature_panel_path"] == "model_feature_panel.json"
    assert Path(default_feature_panel_path(str(model_path))).exists()

    loaded = TrainedModel.load(str(default_manifest_path(str(model_path))))
    assert loaded.train_config["preprocess"]["feature_panel"]["gene_ids"] == ["g1", "g2"]
