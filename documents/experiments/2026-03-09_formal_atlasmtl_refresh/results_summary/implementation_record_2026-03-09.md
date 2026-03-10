# Formal Refresh Implementation Record (`2026-03-09`)

## Scope

This round executed the fifth-round formal AtlasMTL refresh:

- AtlasMTL only
- formal third-wave contract retained
- refreshed train config fixed to:
  - `optimizer_name="adamw"`
  - `weight_decay=5e-5`
  - `scheduler_name=None`

Covered points:

- main panel `16` points
  - `HLCA_Core`
  - `PHMap_Lung_Full_v43_light`
  - `mTCA`
  - `DISCO_hPBMCs`
  - tracks: `cpu_core`, `gpu`
  - points: `build_100000_eval10k`, `predict_100000_10000`
- supplementary `4` points
  - `Vento`
  - tracks: `cpu_core`, `gpu`
  - points: `build_50000_eval10k`, `predict_50000_10000`

## Execution notes

Preparation:

- froze the old formal AtlasMTL baseline into
  `atlasmtl_formal_baseline_anchor.csv`
- generated `20` refresh manifests from the retained formal third-wave manifest
  index
- corrected the refresh manifest generator after the first dry run exposed
  unsupported top-level manifest keys

Run commands:

```bash
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python \
  documents/experiments/2026-03-09_formal_atlasmtl_refresh/scripts/freeze_formal_atlasmtl_baseline.py

/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python \
  documents/experiments/2026-03-09_formal_atlasmtl_refresh/scripts/generate_formal_refresh_manifests.py

bash documents/experiments/2026-03-09_formal_atlasmtl_refresh/scripts/run_formal_refresh_cpu.sh

bash documents/experiments/2026-03-09_formal_atlasmtl_refresh/scripts/run_formal_refresh_gpu.sh

/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python \
  documents/experiments/2026-03-09_formal_atlasmtl_refresh/scripts/collect_formal_refresh_results.py
```

Execution mode:

- CPU refresh ran in unrestricted shell mode
- GPU refresh ran in unrestricted shell mode with CUDA available
- observed GPU preflight: `NVIDIA GeForce RTX 4090`

Completion status:

- CPU: `10/10` refresh points succeeded
- GPU: `10/10` refresh points succeeded
- total: `20/20` refresh points succeeded

## Result summary

Main panel:

- rows completed: `16/16`
- mean `delta_macro_f1`: `-0.005250`
- GPU positive deltas: `4/8`
- worst main-panel delta: `-0.054224`
  - `DISCO_hPBMCs / gpu / predict_100000_10000`
- GPU median `delta_train_gpu_peak_memory_gb`: `0.003550`
- main median `delta_train_process_peak_rss_gb`: `0.002100`

Supplementary `Vento`:

- `cpu_core / build_50000_eval10k`: `+0.014249`
- `cpu_core / predict_50000_10000`: `+0.002898`
- `gpu / build_50000_eval10k`: `+0.003299`
- `gpu / predict_50000_10000`: `-0.031901`

## Decision

Formal refresh decision:

- `keep_formal_atlasmtl_baseline_rows`

Interpretation:

- the promoted code default `AdamW + weight_decay=5e-5` remains the software
  default
- this refresh did not provide sufficiently strong formal evidence to replace
  the retained third-wave AtlasMTL manuscript rows
- manuscript-grade formal comparison tables should therefore continue to use
  the retained third-wave AtlasMTL baseline rows unless a later formal refresh
  produces a stronger replacement result

Recommended expression:

The fifth-round formal refresh does not justify replacing the retained
third-wave manuscript-grade AtlasMTL baseline rows. The refreshed
configuration (`AdamW + weight_decay=5e-5`) remains the software default
because it is still acceptable as a lightweight training default, but the
formal refresh evidence is not strong enough for manuscript-table replacement:
across the `16` main-panel rows, the mean `delta_macro_f1` is `-0.005250`, GPU
headline improvements are only `4/8`, and there is a substantial regression at
`DISCO_hPBMCs / gpu / predict_100000_10000` (`-0.054224`). Therefore, formal
comparison tables should continue to use the retained third-wave AtlasMTL
baseline rows.
