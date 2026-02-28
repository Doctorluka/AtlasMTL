# atlasmtl Benchmark Protocol

## Goal
Benchmark `atlasmtl` against published tools under a fair comparison protocol.

## Locked decisions
- `atlasmtl` default input transform is `binary`.
- `phmap` is treated as a methodological predecessor, not a benchmark baseline.
- Primary comparison targets are published tools such as `scANVI`, `CellTypist`, `SingleR`, and `ProjectSVR`.

## Preprocessing policy
- `atlasmtl` uses the repository-configured input transform and threshold settings.
- Comparator tools use their recommended preprocessing pipelines.
- Any tool-specific preprocessing differences must be recorded in the benchmark report.

## Output normalization
- Multi-level outputs should be aligned to the available dataset label hierarchy.
- If a comparator only supports a single level, report that level separately.
- If a comparator has no abstention mode, coverage is treated as `1.0`.

## Metrics
- Primary:
  - macro-F1
  - balanced accuracy
  - coverage
  - risk
- Secondary:
  - runtime
  - memory
  - calibration metrics where supported

## Reporting requirements
- Record dataset name, version, split definition, and label mapping.
- Record the exact config used for `atlasmtl`.
- Record comparator package versions and invocation settings.
