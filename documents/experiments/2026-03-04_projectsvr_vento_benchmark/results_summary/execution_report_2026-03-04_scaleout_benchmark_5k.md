# Vento second-wave scale-out benchmark (`5k` heldout)

- scenario: `Vento` reference-heldout nested `5k` scale-out benchmark
- manifests:
  - `documents/experiments/2026-03-04_projectsvr_vento_benchmark/manifests/reference_heldout/Vento__annotation__scaleout_runtime_5k_v1.yaml`
- output root: `/home/data/fhz/tmp/atlasmtl_benchmarks/2026-03-04/reference_heldout/Vento/benchmark/group_split_v2_train50k_test5k_nested/all_methods_v1/`
- methods attempted: `atlasmtl`, `reference_knn`, `celltypist`, `scanvi`, `singler`, `symphony`, `seurat_anchor_transfer`
- run status: `7/7 success`

## Metric summary

- `scanvi`: `accuracy=0.9224`, `macro_f1=0.8774`
- `atlasmtl`: `accuracy=0.9056`, `macro_f1=0.8670`, `coverage=0.9912`, `unknown_rate=0.0088`
- `celltypist`: `accuracy=0.8566`, `macro_f1=0.7934`
- `singler`: `accuracy=0.8562`, `macro_f1=0.8251`, `unknown_rate=0.0220`
- `reference_knn`: `accuracy=0.7404`, `macro_f1=0.5759`
- `symphony`: `accuracy=0.7310`, `macro_f1=0.5817`
- `seurat_anchor_transfer`: `accuracy=0.5632`, `macro_f1=0.4043`, backend=`seurat_anchor_transfer_transferdata`

## Benchmark resource summary

| Method | Train time (s) | Predict time (s) | Train peak RSS (GB) | Predict peak RSS (GB) |
| --- | ---: | ---: | ---: | ---: |
| `atlasmtl` | `9.5542` | `0.0434` | `3.1956` | `3.5485` |
| `reference_knn` | `1.1545` | `3.0314` | `2.7979` | `2.7979` |
| `celltypist` | `0.0011` | `0.2717` | `0.9405` | `1.1012` |
| `scanvi` | `458.9876` | `17.1991` | `2.2776` | `2.2776` |
| `singler` | `262.6317` | `262.6317` | `0.0` | `0.0` |
| `symphony` | `34.7795` | `34.7795` | `0.0` | `0.0` |
| `seurat_anchor_transfer` | `1146.0187` | `1146.0187` | `0.0` | `0.0` |

Additional comparator-training note:

- `celltypist` also required wrapper-side comparator model training before the
  benchmark run itself: `166.1846s` on the `50k` prepared reference

## Discussion

- the nested `5k` run preserved the same ranking pattern seen on the `10k`
  heldout run: `scanvi` led on raw accuracy, `atlasmtl` stayed extremely close
  and retained the best macro-F1 among the methods with abstention behavior
- `atlasmtl` improved slightly on macro-F1 relative to the `10k` run while
  keeping low reject rate
- `celltypist` and `singler` remained competitive second-tier baselines on this
  single-label setting
- `reference_knn`, `symphony`, and especially `seurat_anchor_transfer` remained
  materially weaker than the leading methods
- external-comparator RSS accounting remains incomplete for `singler`,
  `symphony`, and `seurat_anchor_transfer`; the `0.0` values should be treated
  as monitoring gaps rather than true zero-memory runs
