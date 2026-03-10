import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from atlasmtl import build_model


def _imbalanced_reference() -> AnnData:
    obs = pd.DataFrame(
        {
            "anno_lv1": ["A", "A", "A", "A", "B", "B"],
            "anno_lv4": ["major", "major", "major", "major", "minor", "minor"],
        },
        index=[f"r{i}" for i in range(6)],
    )
    adata = AnnData(
        X=np.array(
            [
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [0, 1],
                [0, 2],
            ],
            dtype=np.float32,
        ),
        obs=obs,
    )
    adata.var_names = ["g1", "g2"]
    return adata


def test_build_model_records_class_imbalance_controls():
    model = build_model(
        adata=_imbalanced_reference(),
        label_columns=["anno_lv1", "anno_lv4"],
        coord_targets={},
        task_weights=[0.5, 2.0],
        class_weighting={"label_column": "anno_lv4", "mode": "balanced"},
        class_balanced_sampling={"label_column": "anno_lv4", "mode": "balanced"},
        num_epochs=1,
        batch_size=2,
        hidden_sizes=[8],
        device="cpu",
    )

    class_weighting = model.train_config["class_weighting"]
    class_balanced_sampling = model.train_config["class_balanced_sampling"]

    assert class_weighting["label_column"] == "anno_lv4"
    assert class_weighting["mode"] == "balanced"
    assert class_weighting["class_counts"] == {"major": 4, "minor": 2}
    assert class_weighting["class_weights"]["minor"] > class_weighting["class_weights"]["major"]

    assert class_balanced_sampling["label_column"] == "anno_lv4"
    assert class_balanced_sampling["mode"] == "balanced"
    assert class_balanced_sampling["replacement"] is True
    assert class_balanced_sampling["class_counts"] == {"major": 4, "minor": 2}


def test_build_model_rejects_unknown_class_imbalance_label():
    with pytest.raises(ValueError, match="label_column"):
        build_model(
            adata=_imbalanced_reference(),
            label_columns=["anno_lv1", "anno_lv4"],
            coord_targets={},
            class_weighting={"label_column": "anno_lv3", "mode": "balanced"},
            num_epochs=1,
            batch_size=2,
            hidden_sizes=[8],
            device="cpu",
        )
