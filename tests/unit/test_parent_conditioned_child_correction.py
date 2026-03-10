import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from atlasmtl import TrainedModel, build_model


def _toy_reference() -> AnnData:
    obs = pd.DataFrame(
        {
            "anno_lv3": [
                "P1",
                "P1",
                "P1",
                "P1",
                "P2",
                "P2",
                "P2",
                "P2",
            ],
            "anno_lv4": [
                "P1_c1",
                "P1_c1",
                "P1_c2",
                "P1_c2",
                "P2_c1",
                "P2_c1",
                "P2_c2",
                "P2_c2",
            ],
        },
        index=[f"r{i}" for i in range(8)],
    )
    adata = AnnData(
        X=np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 0],
                [0, 2, 0],
                [0, 1, 1],
                [0, 2, 1],
            ],
            dtype=np.float32,
        ),
        obs=obs,
    )
    adata.var_names = ["g1", "g2", "g3"]
    return adata


def test_build_model_records_parent_conditioned_child_correction(tmp_path):
    model = build_model(
        adata=_toy_reference(),
        label_columns=["anno_lv3", "anno_lv4"],
        coord_targets={},
        num_epochs=1,
        batch_size=2,
        hidden_sizes=[8],
        device="cpu",
        parent_conditioned_child_correction={
            "parent_level": "anno_lv3",
            "target_level": "anno_lv4",
            "hotspot_parents": ["P1"],
            "mode": "frozen_base",
            "base_lr_scale": 0.1,
            "loss_weight": 1.0,
        },
    )

    cfg = model.train_config["parent_conditioned_child_correction"]
    assert cfg["parent_level"] == "anno_lv3"
    assert cfg["target_level"] == "anno_lv4"
    assert cfg["hotspot_parents"] == ["P1"]
    assert cfg["mode"] == "frozen_base"
    assert cfg["hotspot_child_indices"]

    out_path = tmp_path / "toy_model.pth"
    model.save(str(out_path))
    loaded = TrainedModel.load(str(out_path.with_name("toy_model_manifest.json")))
    assert loaded.train_config["parent_conditioned_child_correction"]["hotspot_parents"] == ["P1"]
    assert loaded.model.has_parent_conditioned_child_correction()


def test_parent_conditioned_child_correction_leaves_non_hotspot_rows_unchanged():
    model = build_model(
        adata=_toy_reference(),
        label_columns=["anno_lv3", "anno_lv4"],
        coord_targets={},
        num_epochs=1,
        batch_size=2,
        hidden_sizes=[8],
        device="cpu",
        parent_conditioned_child_correction={
            "parent_level": "anno_lv3",
            "target_level": "anno_lv4",
            "hotspot_parents": ["P1"],
            "mode": "joint",
            "base_lr_scale": 0.1,
            "loss_weight": 1.0,
        },
    )

    latent = torch.randn(3, model.model.latent_dim)
    parent_logits = torch.tensor(
        [
            [5.0, 0.0],  # hotspot P1
            [0.0, 5.0],  # non-hotspot P2
            [0.0, 5.0],  # non-hotspot P2
        ]
    )
    child_logits = torch.randn(3, len(model.label_encoders["anno_lv4"].classes_))

    for module in model.model.child_correction_modules.values():
        final = module[-1]
        assert isinstance(final, torch.nn.Linear)
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.constant_(final.bias, 1.0)

    corrected, active = model.model.apply_parent_conditioned_child_correction(
        latent,
        [parent_logits.clone(), child_logits.clone()],
        parent_indices_override=torch.tensor([0, 1, 1]),
    )

    assert bool(active[0].item()) is True
    assert bool(active[1].item()) is False
    assert bool(active[2].item()) is False
    np.testing.assert_allclose(corrected[1][1].detach().numpy(), child_logits[1].detach().numpy())
    np.testing.assert_allclose(corrected[1][2].detach().numpy(), child_logits[2].detach().numpy())
    assert not np.allclose(corrected[1][0].detach().numpy(), child_logits[0].detach().numpy())


def test_parent_conditioned_child_correction_supports_reranker_like_mode():
    model = build_model(
        adata=_toy_reference(),
        label_columns=["anno_lv3", "anno_lv4"],
        coord_targets={},
        num_epochs=1,
        batch_size=2,
        hidden_sizes=[8],
        device="cpu",
        parent_conditioned_child_correction={
            "parent_level": "anno_lv3",
            "target_level": "anno_lv4",
            "hotspot_parents": ["P1"],
            "mode": "frozen_base",
            "feature_mode": "reranker_like",
            "rank_loss_weight": 0.2,
            "rank_margin": 0.1,
            "base_lr_scale": 0.1,
            "loss_weight": 1.0,
        },
    )

    cfg = model.train_config["parent_conditioned_child_correction"]
    assert cfg["feature_mode"] == "reranker_like"
    assert cfg["rank_loss_weight"] == 0.2
    assert cfg["rank_margin"] == 0.1

    latent = torch.randn(2, model.model.latent_dim)
    parent_logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    child_logits = torch.tensor([[3.0, 1.0, 0.5, 0.0], [0.5, 0.5, 3.0, 0.0]])[:, : len(model.label_encoders["anno_lv4"].classes_)]

    corrected, active = model.model.apply_parent_conditioned_child_correction(
        latent,
        [parent_logits.clone(), child_logits.clone()],
        parent_indices_override=torch.tensor([0, 1]),
    )

    assert bool(active[0].item()) is True
    assert bool(active[1].item()) is False
    np.testing.assert_allclose(corrected[1][1].detach().numpy(), child_logits[1].detach().numpy())
