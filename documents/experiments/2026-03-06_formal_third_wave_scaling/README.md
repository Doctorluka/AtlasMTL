# Formal third-wave scaling dossier

This dossier owns the round-level preparation assets for the formal third-wave
scaling benchmark.

Use this dossier for:

- dataset-audit configs
- split/preprocessing scripts
- round-level preparation notes
- links to tmp output roots

Primary protocol and plan:

- `plan/2026-03-06_formal_third_wave_scaling_plan.md`
- `documents/protocols/formal_third_wave_scaling_protocol.md`

Round-level tmp root:

- `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/`

This preparation round materializes:

- nested build subsets
- standalone `build_eval_fixed_10k`
- nested predict-scaling subsets
- dataset ceiling summaries
- formal benchmark manifests

Main panel:

- `HLCA_Core`
- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`

Supplementary:

- `Vento`

Key phase-1 outputs:

- `results_summary/phase1_preparation_report_2026-03-06.md`
- `results_summary/execution_report_2026-03-06_formal_prep_start.md`
- `manifests/reference_heldout/manifest_index.json`

Current execution note:

- `HLCA_Core` formal sanity execution has started.
- GPU sanity finished successfully for the two `100k` sanity points.
- CPU sanity was intentionally stopped after a long-running
  `seurat_anchor_transfer` step on `build100k -> eval10k`.
- See `results_summary/execution_report_2026-03-06_formal_hlca_sanity_start.md`.
