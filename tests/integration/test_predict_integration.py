import numpy as np
import pandas as pd
from anndata import AnnData

from atlasmtl import build_model, predict


def test_train_predict_roundtrip_standard_mode_writes_core_outputs():
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
    )

    query = AnnData(X=np.array([[1, 0], [0, 1]], dtype=np.float32), obs=pd.DataFrame(index=["q1", "q2"]))
    query.var_names = ["g1", "g2"]

    result = predict(model, query, batch_size=1, device="cpu")
    out = result.to_adata(query)

    assert "pred_anno_lv1" in out.obs.columns
    assert "conf_anno_lv1" in out.obs.columns
    assert "is_unknown_anno_lv1" in out.obs.columns
    assert "pred_anno_lv1_knn" not in out.obs.columns
    assert "X_pred_latent" not in out.obsm
    assert "X_pred_umap" not in out.obsm
    assert "atlasmtl" in out.uns


def test_train_predict_roundtrip_full_mode_can_write_coords():
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
    )

    query = AnnData(X=np.array([[1, 0], [0, 1]], dtype=np.float32), obs=pd.DataFrame(index=["q1", "q2"]))
    query.var_names = ["g1", "g2"]

    result = predict(model, query, batch_size=1, device="cpu")
    out = result.to_adata(query, mode="full", include_coords=True)

    assert "pred_anno_lv1_knn" in out.obs.columns
    assert "pred_anno_lv1_raw" in out.obs.columns
    assert "X_pred_latent" in out.obsm
    assert "X_pred_umap" in out.obsm


def test_train_predict_roundtrip_without_coordinates_works_with_knn_off():
    obs = pd.DataFrame({"anno_lv1": ["A", "A", "B", "B"]}, index=["r1", "r2", "r3", "r4"])
    ref = AnnData(X=np.array([[1, 0], [1, 1], [0, 1], [0, 2]], dtype=np.float32), obs=obs)
    ref.var_names = ["g1", "g2"]

    model = build_model(
        adata=ref,
        label_columns=["anno_lv1"],
        coord_targets={},
        num_epochs=1,
        batch_size=2,
        hidden_sizes=[8],
        device="cpu",
    )

    assert model.coord_targets == {}
    assert model.train_config["coord_enabled"] is False
    assert model.train_config["resource_summary"]["num_coord_heads"] == 0
    assert model.train_config["resource_summary"]["num_threads_used"] == 10
    assert model.train_config["device_used"] == "cpu"
    assert model.train_config["train_seconds"] >= 0.0
    assert model.train_config["runtime_summary"]["phase"] == "train"

    query = AnnData(X=np.array([[1, 0], [0, 1]], dtype=np.float32), obs=pd.DataFrame(index=["q1", "q2"]))
    query.var_names = ["g1", "g2"]

    result = predict(model, query, knn_correction="off", batch_size=1, device="cpu")
    out = result.to_adata(query, mode="standard")

    assert "pred_anno_lv1" in out.obs.columns
    assert "conf_anno_lv1" in out.obs.columns
    assert "X_pred_latent" not in out.obsm
    assert out.uns["atlasmtl"]["train_config"]["coord_enabled"] is False
    assert out.uns["atlasmtl"]["device_used"] == "cpu"
    assert out.uns["atlasmtl"]["prediction_runtime"]["phase"] == "predict"


def test_trained_model_resource_usage_helpers(capsys):
    obs = pd.DataFrame({"anno_lv1": ["A", "A", "B", "B"]}, index=["r1", "r2", "r3", "r4"])
    ref = AnnData(X=np.array([[1, 0], [1, 1], [0, 1], [0, 2]], dtype=np.float32), obs=obs)
    ref.var_names = ["g1", "g2"]

    model = build_model(
        adata=ref,
        label_columns=["anno_lv1"],
        coord_targets=None,
        num_epochs=1,
        batch_size=2,
        hidden_sizes=[8],
        device="cpu",
    )

    usage = model.get_resource_usage()
    assert usage["phase"] == "train"
    assert usage["device_used"] == "cpu"

    model.show_resource_usage()
    captured = capsys.readouterr()
    assert "atlasmtl training resource usage" in captured.out
    assert "Summary:" in captured.out
    assert "Execution:" in captured.out
    assert "device_used" in captured.out
    assert "cpu" in captured.out


def test_build_model_can_print_summary_automatically(capsys):
    obs = pd.DataFrame({"anno_lv1": ["A", "A", "B", "B"]}, index=["r1", "r2", "r3", "r4"])
    ref = AnnData(X=np.array([[1, 0], [1, 1], [0, 1], [0, 2]], dtype=np.float32), obs=obs)
    ref.var_names = ["g1", "g2"]

    build_model(
        adata=ref,
        label_columns=["anno_lv1"],
        coord_targets=None,
        num_epochs=1,
        batch_size=2,
        hidden_sizes=[8],
        device="cpu",
        show_summary=True,
    )

    captured = capsys.readouterr()
    assert "atlasmtl training resource usage" in captured.out
