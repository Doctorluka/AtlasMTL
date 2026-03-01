from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import anndata as ad
import numpy as np
import scipy.sparse as sp

from atlasmtl.preprocess import PreprocessConfig, preprocess_query, preprocess_reference, save_feature_panel
from atlasmtl.preprocess.gene_ids import canonicalize_gene_ids, default_gene_id_table_path
from atlasmtl.preprocess.matrix_semantics import is_count_like_matrix


def _sample_min_max(matrix: Any) -> Dict[str, float | None]:
    if sp.issparse(matrix):
        data = matrix.data
    else:
        data = np.asarray(matrix).ravel()
    if data.size == 0:
        return {"min": None, "max": None}
    return {"min": float(np.min(data)), "max": float(np.max(data))}


def _adata_audit(adata: ad.AnnData, *, counts_layer: str) -> Dict[str, Any]:
    return {
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "obs_columns": [str(col) for col in adata.obs.columns],
        "var_columns": [str(col) for col in adata.var.columns],
        "layers": [str(key) for key in adata.layers.keys()],
        "obsm_keys": [str(key) for key in adata.obsm.keys()],
        "var_names_head": [str(item) for item in adata.var_names[:10]],
        "X_count_like": bool(is_count_like_matrix(adata.X)),
        "X_sample_range": _sample_min_max(adata.X),
        "counts_layer_present": bool(counts_layer in adata.layers),
        "counts_layer_count_like": bool(is_count_like_matrix(adata.layers[counts_layer])) if counts_layer in adata.layers else False,
        "counts_layer_sample_range": _sample_min_max(adata.layers[counts_layer]) if counts_layer in adata.layers else {"min": None, "max": None},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-h5ad", required=True)
    parser.add_argument("--query-h5ad", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--gene-id-table", default=default_gene_id_table_path())
    parser.add_argument("--feature-space", default="whole", choices=["whole", "hvg"])
    parser.add_argument("--n-top-genes", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_raw = ad.read_h5ad(args.reference_h5ad)
    query_raw = ad.read_h5ad(args.query_h5ad)

    preprocess_config = PreprocessConfig(
        var_names_type="symbol",
        species="human",
        gene_id_table=args.gene_id_table,
        input_matrix_type="lognorm",
        counts_layer=args.counts_layer,
        feature_space=args.feature_space,
        n_top_genes=args.n_top_genes,
        duplicate_policy="sum",
        unmapped_policy="drop",
    )

    ref_pp, feature_panel, ref_report = preprocess_reference(ref_raw, preprocess_config)
    query_pp, query_report = preprocess_query(query_raw, feature_panel, preprocess_config)

    ref_out = output_dir / "reference_preprocessed.h5ad"
    query_out = output_dir / "query_preprocessed.h5ad"
    panel_out = output_dir / "feature_panel.json"
    audit_out = output_dir / "preprocessing_audit.json"

    ref_pp.write_h5ad(ref_out)
    query_pp.write_h5ad(query_out)
    save_feature_panel(feature_panel, str(panel_out))

    ref_map_preview, ref_map_report = canonicalize_gene_ids(ref_raw[:, : min(ref_raw.n_vars, 2000)].copy(), preprocess_config)
    preview_rows = min(15, ref_map_preview.n_vars)
    payload = {
        "raw_reference": _adata_audit(ref_raw, counts_layer=args.counts_layer),
        "raw_query": _adata_audit(query_raw, counts_layer=args.counts_layer),
        "preprocess_config": preprocess_config.to_dict(),
        "reference_report": ref_report.to_dict(),
        "query_report": query_report.to_dict(),
        "feature_panel": feature_panel.to_dict(),
        "mapping_preview": {
            "n_preview_genes": int(preview_rows),
            "canonical_var_names_head": [str(item) for item in ref_map_preview.var_names[:preview_rows]],
            "canonical_gene_symbol_head": [str(item) for item in ref_map_preview.var["gene_symbol"].astype(str).head(preview_rows).tolist()],
            "canonical_ensembl_gene_id_head": [str(item) for item in ref_map_preview.var["ensembl_gene_id"].astype(str).head(preview_rows).tolist()],
            "preview_report": ref_map_report.to_dict(),
        },
        "outputs": {
            "reference_preprocessed_h5ad": str(ref_out),
            "query_preprocessed_h5ad": str(query_out),
            "feature_panel_json": str(panel_out),
        },
    }
    audit_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
