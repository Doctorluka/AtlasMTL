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
