# atlasmtl

`atlasmtl` is a generic single-cell reference mapping package with multi-task label prediction, coordinate regression, gated KNN correction, and Unknown-cell abstention.

## Design goals
- AnnData in, AnnData out.
- Multi-level annotation with shared representation.
- Optional coordinate prediction (`X_pred_latent`, `X_pred_umap`).
- Low-confidence-only KNN correction to save compute.

## Quick start
```python
import atlasmtl
model = atlasmtl.build_model(
    adata=adata_ref,
    label_columns=["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"],
    coord_targets={"latent": "X_ref_latent", "umap": "X_umap"},
)
result = atlasmtl.predict(model, adata_query)
adata_query = result.to_adata(adata_query)
```

## Project layout
- `atlasmtl/`: package source code
- `benchmark/`: benchmarking pipelines and reports
- `documents/`: design docs and paper assets
- `notebooks/`: reproducible examples
- `vendor/phmap_snapshot/`: read-only source reference
