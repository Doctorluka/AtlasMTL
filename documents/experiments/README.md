# Experiment Dossiers

This directory stores repo-tracked benchmark and optimization dossiers.

Current rule for the paper-grade benchmark rollout:

- each reference dataset gets its own dossier
- each dossier owns its own manifests, scripts, notes, and summaries
- fairness is enforced within each scenario, not by forcing all references
  through one identical execution bundle

## Active dossiers

- `2026-03-01_real_mapping_benchmark`
  - early real-data benchmark and ablation baseline
- `2026-03-02_mtl_classifier_optimization`
  - AtlasMTL classifier optimization track
- `2026-03-03_phmap_benchmark`
  - pilot dossier for `PHMap_Lung_Full_v43_light`
- `2026-03-03_hlca_benchmark`
  - pilot dossier for `HLCA_Core`
- `2026-03-03_projectsvr_disco_benchmark`
  - pilot dossier for `DISCO_hPBMCs`
- `2026-03-04_projectsvr_mtca_benchmark`
  - second-wave scale-out dossier for `mTCA`
- `2026-03-04_projectsvr_cd4_benchmark`
  - second-wave scale-out dossier for `cd4`
- `2026-03-04_projectsvr_cd8_benchmark`
  - second-wave scale-out dossier for `cd8`
- `2026-03-04_projectsvr_vento_benchmark`
  - second-wave scale-out dossier for `Vento`

## Pilot rollout

The current first-wave pilots are:

- `PH-Map`
- `HLCA`
- `DISCO_hPBMCs`

Each pilot should first ship:

- one reference-heldout protocol
- one initial manifest draft
- one output-root convention
- one results-summary skeleton

External query validation can be added after the heldout protocol is reviewed.

Shared pre-execution checklist:

- `documents/protocols/pilot_benchmark_review_checklist.md`

Shared first-wave preprocessing contract:

- `documents/protocols/preprocessing_contract_first_wave.md`

## Second-wave scale-out

The current scale-out round expands from smoke validation to:

- all reference datasets
- larger heldout reference runs
- explicit preparation resource summaries
- nested `10k` / `5k` heldout outputs

Latest round-close status (`2026-03-05`):

- completed: `PHMap_Lung_Full_v43_light`, `DISCO_hPBMCs`, `mTCA`,
  `HLCA_Core`, `Vento`
- excluded in this round: `cd4`, `cd8` (raw-count contract blockers)

Execution plan:

- `plan/2026-03-04_second_wave_scaleout_benchmark_plan.md`
- `documents/experiments/2026-03-04_second_wave_round_status.md`
- `documents/experiments/2026-03-04_second_wave_execution_template.md`
- `documents/protocols/third_wave_fairness_protocol.md`

## Runtime caveats

When running comparator benchmarks in restricted/sandboxed environments, some
methods may silently degrade to serial execution even if `n_jobs > 1` is
configured. The most common signal is:

- `joblib will operate in serial mode` (`Errno 13: Permission denied`)

Implications:

- label-metric comparability is usually still valid
- runtime/throughput/core-usage comparisons are not strictly fair

For formal runtime fairness checks, rerun on an unrestricted server shell and
record effective thread/device usage in the run report.
