from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from benchmark.methods.atlasmtl_inputs import (
    PHMAP_TASK_WEIGHTS,
    adata_with_matrix_from_layer,
    matrix_source_label,
    resolve_atlasmtl_layer_config,
    resolve_task_weights,
    task_weight_scheme_name,
)


def test_resolve_atlasmtl_layer_config_uses_method_overrides() -> None:
    manifest = {
        "counts_layer": "counts",
        "method_configs": {
            "atlasmtl": {
                "reference_layer": "raw_counts",
                "query_layer": "raw_counts_query",
            }
        },
    }
    resolved = resolve_atlasmtl_layer_config(manifest)
    assert resolved["reference_layer"] == "raw_counts"
    assert resolved["query_layer"] == "raw_counts_query"
    assert resolved["counts_layer"] == "counts"


def test_resolve_atlasmtl_layer_config_defaults_to_x_when_no_layer_requested() -> None:
    resolved = resolve_atlasmtl_layer_config({"method_configs": {}})
    assert resolved["reference_layer"] is None
    assert resolved["query_layer"] is None
    assert resolved["counts_layer"] == "counts"


def test_adata_with_matrix_from_layer_replaces_x() -> None:
    adata = AnnData(X=np.array([[0.1, 0.2]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1", "g2"]
    adata.layers["counts"] = np.array([[2.0, 0.0]], dtype=np.float32)
    out = adata_with_matrix_from_layer(adata, layer_name="counts")
    assert np.array_equal(np.asarray(out.X), np.array([[2.0, 0.0]], dtype=np.float32))
    assert np.array_equal(np.asarray(adata.X), np.array([[0.1, 0.2]], dtype=np.float32))


def test_adata_with_matrix_from_layer_rejects_missing_layer() -> None:
    adata = AnnData(X=np.array([[1.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["g1"]
    with pytest.raises(ValueError, match="requested layer"):
        adata_with_matrix_from_layer(adata, layer_name="counts")


def test_resolve_task_weights_prefers_method_config() -> None:
    manifest = {
        "train": {"task_weights": [1.0, 1.0]},
        "method_configs": {"atlasmtl": {"task_weights": [0.3, 2.0]}},
    }
    assert resolve_task_weights(manifest, ["lv1", "lv2"]) == [0.3, 2.0]


def test_task_weight_scheme_name_identifies_phmap() -> None:
    assert task_weight_scheme_name(PHMAP_TASK_WEIGHTS) == "phmap"
    assert task_weight_scheme_name([1.0, 1.0, 1.0, 1.0]) == "uniform"
    assert task_weight_scheme_name([0.2, 0.8, 1.5, 2.0]) == "custom"


def test_matrix_source_label() -> None:
    assert matrix_source_label("counts") == "layers/counts"
    assert matrix_source_label(None) == "X"
