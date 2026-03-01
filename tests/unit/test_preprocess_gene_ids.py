from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from atlasmtl.preprocess import PreprocessConfig
from atlasmtl.preprocess.gene_ids import canonicalize_gene_ids


def _mapping_table(tmp_path):
    path = tmp_path / "mapping.tsv"
    pd.DataFrame(
        {
            "human_ensembl_gene_id": ["ENSG1", "ENSG2", "ENSG3"],
            "human_gene_symbol": ["GATA1", "CD3D", "MS4A1"],
            "human_gene_description": ["", "", ""],
            "mouse_ensembl_gene_id": ["ENSMUSG1", "ENSMUSG2", "ENSMUSG3"],
            "mouse_gene_symbol": ["Gata1", "Cd3d", "Ms4a1"],
            "rat_ensembl_gene_id": ["ENSRNOG1", "ENSRNOG2", "ENSRNOG3"],
            "rat_gene_symbol": ["Gata1", "Cd3d", "Ms4a1"],
        }
    ).to_csv(path, sep="\t", index=False)
    return path


def test_canonicalize_symbols_sums_duplicates_and_drops_unmapped(tmp_path):
    adata = AnnData(X=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["GATA1", "GATA1", "CD3D", "UNKNOWN"]
    cfg = PreprocessConfig(
        var_names_type="symbol",
        species="human",
        gene_id_table=str(_mapping_table(tmp_path)),
        duplicate_policy="sum",
        unmapped_policy="drop",
    )

    out, report = canonicalize_gene_ids(adata, cfg)

    assert list(out.var_names) == ["ENSG1", "ENSG2"]
    assert out.var.loc["ENSG1", "gene_symbol"] == "GATA1"
    assert np.allclose(np.asarray(out.X), np.array([[3.0, 3.0]], dtype=np.float32))
    assert report.n_duplicate_genes == 1
    assert report.n_unmapped_genes == 1


def test_canonicalize_ensembl_strips_versions(tmp_path):
    adata = AnnData(X=np.array([[1.0, 2.0]], dtype=np.float32), obs=pd.DataFrame(index=["c1"]))
    adata.var_names = ["ENSG1.7", "ENSG2.2"]
    cfg = PreprocessConfig(
        var_names_type="ensembl",
        species="human",
        gene_id_table=str(_mapping_table(tmp_path)),
    )

    out, report = canonicalize_gene_ids(adata, cfg)

    assert list(out.var_names) == ["ENSG1", "ENSG2"]
    assert report.ensembl_versions_stripped == 2
    assert list(out.var["gene_symbol"]) == ["GATA1", "CD3D"]
