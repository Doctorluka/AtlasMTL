from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from atlasmtl import PreprocessConfig, preprocess_reference
from atlasmtl.preprocess.matrix_semantics import detect_input_matrix_type, is_count_like_matrix


def _config(**kwargs) -> PreprocessConfig:
    return PreprocessConfig(var_names_type="ensembl", species="human", feature_space="whole", **kwargs)


def test_is_count_like_matrix_accepts_dense_counts():
    assert is_count_like_matrix(np.array([[1, 0], [2, 3]], dtype=np.float32))


def test_is_count_like_matrix_accepts_sparse_counts():
    matrix = sparse.csr_matrix(np.array([[1, 0], [0, 4]], dtype=np.float32))
    assert is_count_like_matrix(matrix)


def test_detect_input_matrix_type_marks_non_integer_as_lognorm():
    adata = AnnData(X=np.array([[0.1, 1.5], [2.2, 0.0]], dtype=np.float32), obs=pd.DataFrame(index=["a", "b"]))
    assert detect_input_matrix_type(adata) == "lognorm"


def test_preprocess_reference_requires_counts_layer_when_x_is_not_counts():
    adata = AnnData(X=np.array([[0.1, 1.1], [0.0, 2.3]], dtype=np.float32), obs=pd.DataFrame(index=["a", "b"]))
    adata.var_names = ["ENSG1", "ENSG2"]

    with pytest.raises(ValueError, match="raw counts must be provided"):
        preprocess_reference(adata, _config())


def test_preprocess_reference_rejects_invalid_counts_layer():
    adata = AnnData(X=np.array([[0.1, 1.1], [0.0, 2.3]], dtype=np.float32), obs=pd.DataFrame(index=["a", "b"]))
    adata.var_names = ["ENSG1", "ENSG2"]
    adata.layers["counts"] = np.array([[0.1, 1.1], [0.0, 2.3]], dtype=np.float32)

    with pytest.raises(ValueError, match="counts layer exists but is not count-like"):
        preprocess_reference(adata, _config())


def test_preprocess_reference_copies_count_x_to_counts_layer():
    adata = AnnData(X=np.array([[1, 0], [2, 3]], dtype=np.float32), obs=pd.DataFrame(index=["a", "b"]))
    adata.var_names = ["ENSG1", "ENSG2"]

    ref_pp, _, report = preprocess_reference(adata, _config())

    assert "counts" in ref_pp.layers
    np.testing.assert_array_equal(np.asarray(ref_pp.layers["counts"]), np.asarray(ref_pp.X))
    assert report.counts_layer_used == "counts"
    assert report.counts_check_passed is True
