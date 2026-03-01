import numpy as np
import pandas as pd
from anndata import AnnData

from atlasmtl import build_model


def test_build_model_accepts_topology_loss_and_records_config():
    obs = pd.DataFrame({"anno_lv1": ["A", "A", "B", "B"]}, index=["r1", "r2", "r3", "r4"])
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
        topology_loss_weight=0.5,
        topology_k=1,
        topology_coord="latent",
    )

    assert model.train_config["topology_loss_weight"] == 0.5
    assert model.train_config["topology_k"] == 1
    assert model.train_config["topology_coord"] == "latent"

