import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from atlasmtl import build_model


def _tiny_reference() -> AnnData:
    obs = pd.DataFrame({"anno_lv1": ["A", "A", "B", "B"]}, index=["r1", "r2", "r3", "r4"])
    adata = AnnData(X=np.array([[1, 0], [1, 1], [0, 1], [0, 2]], dtype=np.float32), obs=obs)
    adata.var_names = ["g1", "g2"]
    return adata


def test_build_model_records_default_optimizer_config():
    model = build_model(
        adata=_tiny_reference(),
        label_columns=["anno_lv1"],
        coord_targets={},
        num_epochs=1,
        batch_size=2,
        hidden_sizes=[8],
        device="cpu",
    )

    assert model.train_config["optimizer_name"] == "adam"
    assert model.train_config["weight_decay"] == 0.0
    assert model.train_config["scheduler_name"] is None


def test_build_model_records_adamw_optimizer_config():
    model = build_model(
        adata=_tiny_reference(),
        label_columns=["anno_lv1"],
        coord_targets={},
        num_epochs=1,
        batch_size=2,
        hidden_sizes=[8],
        optimizer_name="adamw",
        weight_decay=1e-5,
        val_fraction=0.5,
        scheduler_name="reduce_lr_on_plateau",
        device="cpu",
    )

    assert model.train_config["optimizer_name"] == "adamw"
    assert model.train_config["weight_decay"] == 1e-5
    assert model.train_config["scheduler_name"] == "reduce_lr_on_plateau"
    assert model.train_config["scheduler_monitor"] == "val_loss"
    assert model.train_config["final_learning_rate"] <= model.train_config["learning_rate"]


def test_build_model_rejects_invalid_optimizer_name():
    with pytest.raises(ValueError, match="optimizer_name"):
        build_model(
            adata=_tiny_reference(),
            label_columns=["anno_lv1"],
            coord_targets={},
            num_epochs=1,
            batch_size=2,
            hidden_sizes=[8],
            optimizer_name="sgd",
            device="cpu",
        )


def test_build_model_rejects_invalid_scheduler_name():
    with pytest.raises(ValueError, match="scheduler_name"):
        build_model(
            adata=_tiny_reference(),
            label_columns=["anno_lv1"],
            coord_targets={},
            num_epochs=1,
            batch_size=2,
            hidden_sizes=[8],
            scheduler_name="cosine",
            val_fraction=0.5,
            device="cpu",
        )


def test_build_model_requires_validation_for_scheduler():
    with pytest.raises(ValueError, match="val_fraction > 0"):
        build_model(
            adata=_tiny_reference(),
            label_columns=["anno_lv1"],
            coord_targets={},
            num_epochs=1,
            batch_size=2,
            hidden_sizes=[8],
            optimizer_name="adamw",
            weight_decay=1e-5,
            scheduler_name="reduce_lr_on_plateau",
            val_fraction=0.0,
            device="cpu",
        )
