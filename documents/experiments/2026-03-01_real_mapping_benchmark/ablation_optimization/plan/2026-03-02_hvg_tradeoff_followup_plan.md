# AtlasMTL HVG Tradeoff Follow-up Plan

## Goal

The next round should not optimize for the single highest accuracy point. The
goal is to identify a defensible operational setting that balances fine-level
label quality against runtime and memory cost.

## Locked decision rule

- primary quality targets:
  - `anno_lv4` accuracy
  - `anno_lv4` macro-F1
- required resource targets:
  - train elapsed seconds
  - predict elapsed seconds
  - peak RSS
  - peak GPU memory when applicable
- selection rule:
  - find the best observed quality run on the dataset
  - define a near-optimal quality band around that run
  - choose the lowest-resource candidate inside that band

## Follow-up benchmark scope

Keep fixed:

- counts source: `layers["counts"]`
- gene canonicalization: bundled BioMart mapping
- hierarchy: enabled
- KNN correction: disabled for this dataset
- input transform baseline: `binary`
- task-weight baseline: `phmap = [0.3, 0.8, 1.5, 2.0]`

Vary:

- feature space:
  - `whole`
  - `hvg3000`
  - `hvg4000`
  - `hvg5000`
  - `hvg6000`
  - `hvg7000`
  - `hvg8000`

Optional secondary checks:

- `uniform` task weights as a stability control
- `float` input only when re-checking the binary decision on new datasets
- `cpu` and `cuda` as separate AtlasMTL variants when the CUDA gate passes

## Execution order

1. Reuse the current sampled dataset and confirm the local HVG curve around
   `6000`.
2. Export an HVG-only tradeoff table with quality and resource fields.
3. Promote one dataset-level recommendation.
4. Repeat the same grid on at least one additional dataset.
5. Compare whether the recommended HVG count is stable or dataset-dependent.

## Deliverables

- `atlasmtl_hvg_tradeoff.csv`
- `atlasmtl_hvg_tradeoff.md`
- dataset-level recommendation note
- cross-dataset comparison note

## Success criteria

- the project can justify a recommended HVG setting without claiming a
  universal optimum
- the recommendation is based on both quality and resource evidence
- `whole` remains available as the strong baseline in formal reports
