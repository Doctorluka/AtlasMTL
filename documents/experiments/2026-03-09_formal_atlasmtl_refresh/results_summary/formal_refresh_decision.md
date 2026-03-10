# Formal Refresh Decision

- main-panel rows: `16`
- main-panel rows with refresh results: `16`
- gpu headline improvements: `4/8`
- main mean delta_macro_f1: `-0.005250`
- main min delta_macro_f1: `-0.054224`
- gpu median delta_train_gpu_peak_memory_gb: `0.003550`
- main median delta_train_process_peak_rss_gb: `0.002100`

Recommended expression:

The fifth-round formal refresh does not justify replacing the retained
third-wave manuscript-grade AtlasMTL baseline rows.

The refreshed configuration (`AdamW + weight_decay=5e-5`) remains the software
default because it is still acceptable as a lightweight training default, but
the formal refresh evidence is not strong enough for manuscript-table
replacement:

- across the `16` main-panel rows, the mean `delta_macro_f1` is `-0.005250`
- GPU headline improvements are only `4/8`
- there is a substantial regression at
  `DISCO_hPBMCs / gpu / predict_100000_10000` with `delta_macro_f1 = -0.054224`

Formal decision:

- `keep_formal_atlasmtl_baseline_rows`
- formal comparison tables should continue to use the retained third-wave
  AtlasMTL baseline rows
- the refreshed configuration remains the software default, not the manuscript
  replacement row set

Supplementary Vento note:

- `cpu_core / build_50000_eval10k` delta_macro_f1 = `0.014249`
- `cpu_core / predict_50000_10000` delta_macro_f1 = `0.002898`
- `gpu / build_50000_eval10k` delta_macro_f1 = `0.003299`
- `gpu / predict_50000_10000` delta_macro_f1 = `-0.031901`
