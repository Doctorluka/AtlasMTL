# Benchmark scenario registry

Date: `2026-03-03`

This file is the working scenario registry for the next benchmark execution
round. It separates **formal reference-heldout benchmark scenarios** from
**external query validation scenarios** so the execution plan does not blur
accuracy claims with deployment-style biological review.

Operational rule for this round:

- each reference dataset should be executed as an independent experiment track
- code, manifests, outputs, and summaries should be organized per reference
- fairness is enforced within each scenario, not by forcing all references into
  one identical execution bundle

## Status legend

- `READY_TO_MATERIALIZE`
  - dataset contract is sufficiently clear to create the first manifest draft
- `WAITING_FOR_SPLIT`
  - needs actual train/validation/test split materialization before the
    manifest can be finalized
- `WAITING_FOR_POLICY`
  - blocked by an unresolved policy choice, such as target labels, domain key,
    or truth-label approval

## 1) Reference-heldout scenarios

| Scenario ID | Dataset | Target label(s) | Split field | Candidate training grid | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `phmap_ref_heldout_lv4_v1` | `PHMap_Lung_Full_v43_light` | `anno_lv4` | `sample` | `1k,3k,6k,10k,20k,30k,40k,50k` | `WAITING_FOR_SPLIT` | primary fine-label quantitative benchmark |
| `phmap_ref_heldout_hier_v1` | `PHMap_Lung_Full_v43_light` | `anno_lv1..anno_lv4` | `sample` | `1k,3k,6k,10k,20k,30k,40k,50k` | `WAITING_FOR_SPLIT` | hierarchy metrics and full-path analysis |
| `hlca_ref_heldout_lv5_v1` | `HLCA_Core` | `ann_level_5` | `donor_id` | `1k,3k,6k,10k,20k,30k,40k,50k` | `WAITING_FOR_SPLIT` | primary HLCA fine-label benchmark |
| `hlca_ref_heldout_hier_v1` | `HLCA_Core` | `ann_level_1..ann_level_5` | `donor_id` | `1k,3k,6k,10k,20k,30k,40k,50k` | `WAITING_FOR_SPLIT` | hierarchy benchmark; no `ann_finest_level` |
| `mtca_ref_heldout_lv3_v1` | `mTCA` | `Cell_type_level3` | `orig.ident` | `1k,3k,6k,10k,20k,30k,40k,50k` | `WAITING_FOR_SPLIT` | count-in-`adata.X` validation path |
| `disco_ref_heldout_subtype_v1` | `DISCO_hPBMCs` | `cell_subtype` | `sample` | `1k,3k,6k,10k,20k,30k,40k,50k` | `WAITING_FOR_SPLIT` | recommended ProjectSVR pilot |
| `cd4_ref_heldout_subtype_v1` | `cd4` | `cell_subtype` | `sample` | `1k,3k,6k,10k,20k,30k,40k,50k` | `WAITING_FOR_SPLIT` | smaller-feature T-cell atlas |
| `cd8_ref_heldout_subtype_v1` | `cd8` | `cell_subtype` | `sample` | `1k,3k,6k,10k,20k,30k,40k,50k` | `WAITING_FOR_SPLIT` | smaller-feature T-cell atlas |
| `vento_ref_heldout_annotation_v1` | `Vento` | `annotation` | `orig.ident` | `TBD after split` | `WAITING_FOR_SPLIT` | single-label only; likely reduced upper ceiling |

## 2) External query validation scenarios

| Scenario ID | Reference | Query | Target label(s) | Domain / grouping key | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `phmap_to_query_ph_marker_v1` | `PHMap_Lung_Full_v43_light` | `query_PH` | `anno_lv2`, `anno_lv4` | `sample` or `group` | `READY_TO_MATERIALIZE` | marker review; query has `layers["counts"]` |
| `hlca_to_gse302339_marker_v1` | `HLCA_Core` | `hlca_query_GSE302339` | `ann_level_3`, `ann_level_5` | `GSE_id` or `Subject group` | `WAITING_FOR_POLICY` | `Gold` not formal truth by default |
| `disco_to_pbmc_marker_v1` | `DISCO_hPBMCs` | `pbmc_query` | `cell_type`, `cell_subtype` | `donor` or `orig.ident` | `READY_TO_MATERIALIZE` | count-in-`adata.X` query path |
| `vento_to_decidua_marker_v1` | `Vento` | `decidua_query` | `annotation` | `orig.ident` | `READY_TO_MATERIALIZE` | Sankey + marker review only |

## 3) Immediate execution order

Recommended order for the next execution wave:

1. `phmap_ref_heldout_lv4_v1`
2. `hlca_ref_heldout_lv5_v1`
3. `disco_ref_heldout_subtype_v1`
4. `phmap_to_query_ph_marker_v1`

This order keeps the first wave intentionally heterogeneous:

- one Symbol-based reference with explicit `layers["counts"]`
- one Ensembl-based reference with log-normalized `adata.X`
- one ProjectSVR reference validating count handling in `adata.X`
- one external query validation scenario for visualization outputs

## 4) Recommended dossier layout

Recommended reference-specific dossier roots:

- `documents/experiments/<date>_phmap_benchmark/`
- `documents/experiments/<date>_hlca_benchmark/`
- `documents/experiments/<date>_projectsvr_mtca_benchmark/`
- `documents/experiments/<date>_projectsvr_disco_benchmark/`
- `documents/experiments/<date>_projectsvr_cd4_benchmark/`
- `documents/experiments/<date>_projectsvr_cd8_benchmark/`
- `documents/experiments/<date>_projectsvr_vento_benchmark/`

Each dossier should own:

- scenario-specific manifests
- dataset-specific scripts
- run notes and exclusions
- compact results summaries

Current pilot dossier roots already created:

- `documents/experiments/2026-03-03_phmap_benchmark/`
- `documents/experiments/2026-03-03_hlca_benchmark/`
- `documents/experiments/2026-03-03_projectsvr_disco_benchmark/`

Second-wave scale-out dossier roots:

- `documents/experiments/2026-03-04_projectsvr_mtca_benchmark/`
- `documents/experiments/2026-03-04_projectsvr_cd4_benchmark/`
- `documents/experiments/2026-03-04_projectsvr_cd8_benchmark/`
- `documents/experiments/2026-03-04_projectsvr_vento_benchmark/`

Current first-pass heldout manifest drafts:

- `documents/experiments/2026-03-03_phmap_benchmark/manifests/reference_heldout/PHMap_Lung_Full_v43_light__anno_lv4__group_split_v1.yaml`
- `documents/experiments/2026-03-03_hlca_benchmark/manifests/reference_heldout/HLCA_Core__ann_level_5__group_split_v1.yaml`
- `documents/experiments/2026-03-03_projectsvr_disco_benchmark/manifests/reference_heldout/DISCO_hPBMCs__cell_subtype__group_split_v1.yaml`

Shared execution review checklist:

- `documents/protocols/pilot_benchmark_review_checklist.md`

Shared first-wave preprocessing contract:

- `documents/protocols/preprocessing_contract_first_wave.md`
