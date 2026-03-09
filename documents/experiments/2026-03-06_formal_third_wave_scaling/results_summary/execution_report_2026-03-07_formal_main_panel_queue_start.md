# Execution report: formal main-panel queue start

Date: `2026-03-07`

This report records the start of the remaining main-panel formal execution
queues after `HLCA_Core` established the validated execution template.

## Target datasets

Main-panel datasets launched in this queue:

- `PHMap_Lung_Full_v43_light`
- `mTCA`
- `DISCO_hPBMCs`

Supplementary dataset excluded from the default queue:

- `Vento`

Reason:

- `Vento` remains a supplementary reduced-ceiling dataset and should not be
  mixed into the default main-panel execution queue

## Queue scripts

- `scripts/run_formal_round_cpu_core_queue.sh`
- `scripts/run_formal_round_gpu_queue.sh`
- `scripts/run_formal_round_cpu_seurat_queue.sh`

## Queue policy

- `cpu_core` runs the remaining main-panel datasets in sequence
- `gpu` runs the remaining main-panel datasets in sequence from a direct
  non-sandbox shell session
- `cpu_seurat` runs in an isolated restricted policy mode:
  - build scaling only
  - `10k / 20k / 30k / 50k`
  - no CPU seurat predict scaling

## Record rule

All issues discovered during these queues should be appended to:

- `results_summary/formal_experiment_report.md`

## Background monitoring

A lightweight background monitor was enabled for the remaining formal queues.

Policy:

- write one progress snapshot every `10 minutes`
- record per-dataset / per-track completed-point counts
- record timeout-file counts
- record a compact active-process snapshot

Monitoring assets:

- script:
  - `scripts/monitor_formal_round_progress.sh`
- log:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/logs/formal_round_progress_monitor.log`
- pid:
  - `/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/pids/formal_round_progress_monitor.pid`

## 2026-03-08 queue interruption and restart

The queued main-panel runs were interrupted before the remaining
`PHMap_Lung_Full_v43_light`, `mTCA`, and `DISCO_hPBMCs` workloads finished.

Before restart, the on-disk checkpoint state was:

- `PHMap_Lung_Full_v43_light`
  - `cpu_core`: `12` completed points, latest `predict_100000_8000`
  - `gpu`: `14` completed points, latest `predict_100000_8000`
  - `cpu_seurat`: `4` completed points, latest `build_50000_eval10k`
- `mTCA`
  - `cpu_core`: `0` completed points
  - `gpu`: `13` completed points, latest `predict_100000_8000`
  - `cpu_seurat`: `3` completed points, latest `build_30000_eval10k`
- `DISCO_hPBMCs`
  - `cpu_core`: `0` completed points
  - `gpu`: `13` completed points, latest `predict_100000_8000`
  - `cpu_seurat`: `0` completed points

Restart policy:

- reuse the existing queue scripts
- skip points only when `scaleout_status.json` shows all required methods
  succeeded
- continue all unfinished tracks from the stored checkpoint state
