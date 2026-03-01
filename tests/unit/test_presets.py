import numpy as np
import pandas as pd
from anndata import AnnData

from atlasmtl import build_model


def test_build_model_preset_sets_hidden_sizes_when_not_provided():
    obs = pd.DataFrame({"anno_lv1": ["A", "A"]}, index=["r1", "r2"])
    ref = AnnData(X=np.array([[1, 0], [0, 1]], dtype=np.float32), obs=obs)
    ref.var_names = ["g1", "g2"]
    model = build_model(
        adata=ref,
        label_columns=["anno_lv1"],
        coord_targets={},
        num_epochs=1,
        batch_size=2,
        device="cpu",
        preset="small",
    )
    assert model.train_config["preset"] == "small"
    assert model.train_config["hidden_sizes"] == [128, 64]

