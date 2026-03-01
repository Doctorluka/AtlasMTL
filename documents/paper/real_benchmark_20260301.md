# Real Benchmark Summary (2026-03-01)

This note summarizes the sampled real-data benchmark run stored in:

- repo dossier:
  `documents/experiments/2026-03-01_real_mapping_benchmark/`
- runtime artifacts:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/`

## 1. Dataset and protocol

Reference and query:

- reference:
  `data/test_adatas/sampled_adata_10k.h5ad`
- query:
  `data/test_adatas/sampled_adata_3000.h5ad`

Shared properties:

- raw gene namespace at load time:
  symbol
- canonical internal namespace used in preprocessing:
  versionless Ensembl
- authoritative mapping resource:
  `atlasmtl/resources/gene_id/biomart_human_mouse_rat.tsv.gz`
- raw counts source:
  `layers["counts"]`

Preprocessing outcome:

- input genes:
  `21977`
- canonical genes retained:
  `21510`
- unmapped genes dropped:
  `467`
- duplicate canonical IDs collapsed:
  `0`

Single-level shared benchmark target:

- `anno_lv4`

Multi-level AtlasMTL labels:

- `anno_lv1`
- `anno_lv2`
- `anno_lv3`
- `anno_lv4`

KNN scope in this run:

- disabled
- reason:
  no coordinate targets were available in `obsm`

## 2. Single-level comparator results

Final all-method benchmark bundle:

- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/single_level_benchmark/all_methods_final_v2/`

Headline results on `anno_lv4`:

| method | accuracy | macro_f1 | balanced_accuracy |
| --- | ---: | ---: | ---: |
| celltypist | 0.7900 | 0.7093 | 0.7104 |
| atlasmtl | 0.7467 | 0.6381 | 0.6068 |
| azimuth | 0.7343 | 0.5888 | 0.5854 |
| singler | 0.6850 | 0.5990 | 0.6360 |
| symphony | 0.6160 | 0.4838 | 0.4875 |
| scanvi | 0.5773 | 0.3161 | 0.3409 |
| reference_knn | 0.3630 | 0.2917 | 0.2606 |

Interpretation:

- `celltypist` is strongest on this sampled setup under its prepared-expression
  contract.
- `atlasmtl` is competitive and clearly ahead of `scanvi` and
  `reference_knn`.
- `singler` remains a strong single-level baseline.
- `azimuth` completed in fallback mode, not strict native mode.

Azimuth backend note:

- this run used:
  `seurat_anchor_transfer_fallback`
- therefore the result should be labeled explicitly in any manuscript table

## 3. Multi-level AtlasMTL results

Runtime bundle:

- `~/tmp/atlasmtl_real_mapping_benchmark_20260301/multilevel_atlasmtl/`

Per-level results:

| level | accuracy | macro_f1 | balanced_accuracy |
| --- | ---: | ---: | ---: |
| anno_lv1 | 0.9877 | 0.9855 | 0.9858 |
| anno_lv2 | 0.9353 | 0.8569 | 0.8822 |
| anno_lv3 | 0.8700 | 0.7473 | 0.7477 |
| anno_lv4 | 0.7437 | 0.6274 | 0.5990 |

Hierarchy metrics:

- `anno_lv2 -> anno_lv1` path consistency:
  `1.0000`
- `anno_lv3 -> anno_lv2` path consistency:
  `1.0000`
- `anno_lv4 -> anno_lv3` path consistency:
  `1.0000`
- full-path accuracy:
  `0.7330`
- full-path coverage:
  `0.8993`
- covered full-path accuracy:
  `0.8150`

Interpretation:

- the expected coarse-to-fine degradation pattern is present
- hierarchy-aware prediction is functioning and auditable
- this run supports the claim that AtlasMTL should be evaluated as a
  multi-level label-transfer method, not only as a flat classifier

## 4. Runtime and resource recording

Yes, this run did record runtime information, and AtlasMTL also recorded
partial resource information.

Recorded fields are stored in:

- single-level benchmark:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/single_level_benchmark/all_methods_final_v2/metrics.json`
- multi-level AtlasMTL:
  `~/tmp/atlasmtl_real_mapping_benchmark_20260301/multilevel_atlasmtl/metrics.json`

AtlasMTL single-level:

- train elapsed seconds:
  `11.0855`
- predict elapsed seconds:
  `0.2809`
- train peak RSS:
  `4.1933 GB`
- predict peak RSS:
  `4.1933 GB`
- threads used:
  `10`

AtlasMTL multi-level:

- train elapsed seconds:
  `12.7103`
- predict elapsed seconds:
  `0.2340`
- train peak RSS:
  `4.1910 GB`
- predict peak RSS:
  `4.1910 GB`
- threads used:
  `10`

Comparator runtime fields were also recorded, but with weaker resource detail:

- `reference_knn`
  - train:
    `1.5815 s`
  - predict:
    `4.8005 s`
- `celltypist`
  - model load:
    `0.0031 s`
  - predict:
    `6.9385 s`
- `scanvi`
  - train:
    `64.9547 s`
  - predict:
    `6.4463 s`
- `singler`
  - annotate/predict:
    `149.5625 s`
- `symphony`
  - map/predict:
    `33.6403 s`
- `azimuth`
  - transfer/predict:
    `208.4910 s`

Current limitation:

- for most comparators, `process_peak_rss_gb` is still `null`
- so the repository currently records elapsed time consistently across methods,
  but peak memory usage is only reliably populated for AtlasMTL

This means:

- runtime comparison is already usable
- memory comparison is not yet fully comparator-complete

## 5. Paper-facing conclusion

This sampled run is sufficient to support three manuscript-level points:

1. AtlasMTL runs successfully on real reference/query data after bundled
   BioMart-based canonicalization.
2. AtlasMTL is competitive on the shared single-level `anno_lv4` task while
   providing additional multi-level structure not captured by flat baselines.
3. AtlasMTL's multi-level hierarchy-aware path is functioning and measurable on
   real data.

What this run does not yet support:

- formal KNN correction analysis
- coordinate-regression analysis
- strict native-only Azimuth reporting
- fully standardized cross-method peak-memory comparison
