import numpy as np
import pandas as pd
from anndata import AnnData

from atlasmtl import build_model


def test_build_model_records_domain_key_and_domains():
    obs = pd.DataFrame(
        {
            "anno_lv1": ["A", "A", "B", "B"],
            "batch": ["b1", "b1", "b2", "b2"],
        },
        index=["r1", "r2", "r3", "r4"],
    )
    ref = AnnData(X=np.array([[1, 0], [1, 1], [0, 1], [0, 2]], dtype=np.float32), obs=obs)
    ref.var_names = ["g1", "g2"]
    ref.obsm["X_ref_latent"] = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.1, 1.0]], dtype=np.float32)
    ref.obsm["X_umap"] = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.1]], dtype=np.float32)

    model = build_model(
        adata=ref,
        label_columns=["anno_lv1"],
        coord_targets={"latent": "X_ref_latent", "umap": "X_umap"},
        num_epochs=1,
        batch_size=2,
        hidden_sizes=[8],
        device="cpu",
        domain_key="batch",
        domain_loss_weight=0.1,
    )

    assert model.train_config["domain_key"] == "batch"
    assert model.train_config["domain_loss_weight"] == 0.1
    assert model.train_config["domain_loss_method"] == "mean"
    assert model.train_config["domains"] == ["b1", "b2"]

