# Comparator Benchmark Result Template

Date: `YYYY-MM-DD`

This template can be filled manually or generated semi-automatically from
`metrics.json` using:

`benchmark/reports/generate_markdown_report.py`

For paper-oriented CSV / Markdown exports, use:

`benchmark/reports/export_paper_tables.py`

## Run identity

- Dataset:
- Dataset version:
- Split name:
- Split description:
- Reference file:
- Query file:
- Target label column:
- Protocol version:
- Python env:
- `NUMBA_CACHE_DIR`:
- R library paths:
- Output directory:

## Methods included

| Method | Backend | Runtime env | Notes |
|---|---|---|---|
| `atlasmtl` |  |  |  |
| `reference_knn` |  |  |  |
| `celltypist` |  |  |  |
| `scanvi` |  |  |  |
| `singler` |  |  |  |
| `symphony` |  |  |  |
| `seurat_anchor_transfer` |  |  |  |

## Main comparison table

Fill this table for the shared `target_label_column` only.

| Method | Accuracy | Macro-F1 | Balanced accuracy | Coverage | Reject rate | ECE | Brier | AURC | Backend actually used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `atlasmtl` |  |  |  |  |  |  |  |  |  |
| `reference_knn` |  |  |  |  |  |  |  |  |  |
| `celltypist` |  |  |  |  |  |  |  |  |  |
| `scanvi` |  |  |  |  |  |  |  |  |  |
| `singler` |  |  |  |  |  |  |  |  |  |
| `symphony` |  |  |  |  |  |  |  |  |  |
| `seurat_anchor_transfer` |  |  |  |  |  |  |  |  |  |

## Domain-wise table

Use only when `domain_key` is defined.

| Method | Domain | Accuracy | Macro-F1 | Coverage | Reject rate | Unknown rate | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `atlasmtl` |  |  |  |  |  |  |  |
| `reference_knn` |  |  |  |  |  |  |  |
| `celltypist` |  |  |  |  |  |  |  |
| `scanvi` |  |  |  |  |  |  |  |
| `singler` |  |  |  |  |  |  |  |
| `symphony` |  |  |  |  |  |  |  |
| `seurat_anchor_transfer` |  |  |  |  |  |  |  |

## atlasmtl-specific analysis

These metrics should be reported as method-specific analysis rather than forced
onto all comparators.

| Metric | Value | Interpretation |
|---|---:|---|
| Unknown rate |  |  |
| KNN coverage |  |  |
| KNN rescue rate |  |  |
| KNN harm rate |  |  |
| Full-path accuracy |  |  |
| Path consistency rate |  |  |

## Coordinate and topology diagnostics

Use only when coordinate heads and query coordinate targets are available.

| Method | RMSE | Trustworthiness | Continuity | Neighbor overlap | Notes |
|---|---:|---:|---:|---:|---|
| `atlasmtl` |  |  |  |  |  |

## Fairness checklist

- Same reference/query split used for all methods
- Same `target_label_column` used for all external comparators
- Same held-out truth labels used for scoring
- `seurat_anchor_transfer` should be reported under its own name and backend
  rather than being relabeled as `Azimuth`
- Single-level comparators not described as full multi-level methods
- atlasmtl-specific hierarchy/KNN/open-set metrics kept in method-specific analysis

## Environment record

### Python

- Main env:
  `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env`
- Exact package snapshot:

### R

- `Seurat v5` library:
  `/home/data/fhz/seurat_v5`
- Repo-local comparator R library:
  `/home/data/fhz/project/phmap_package/atlasmtl/.r_libs`
- Additional R package notes:

## Interpretation summary

### Main finding

### Where atlasmtl wins

### Where atlasmtl is similar or weaker

### Important caveats

### Recommended next benchmark step
