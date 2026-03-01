# 2026-03-01 Benchmark Resource Accounting Update

This change extends the benchmark reporting layer so runtime/resource comparison
is treated as a first-class benchmark output instead of an incidental note.

## Added

- `runtime_resources` paper-table export in
  `benchmark/reports/export_paper_tables.py`
- `Runtime Resources` section in
  `benchmark/reports/generate_markdown_report.py`
- structured resource fields for:
  - elapsed seconds
  - throughput
  - average RSS
  - peak RSS
  - average GPU memory
  - peak GPU memory
  - average CPU utilization / core-equivalent usage
  - device used
  - thread count when available

## Improved

- comparator wrappers now collect richer runtime/resource payloads instead of
  elapsed time only
- formal benchmark protocol and benchmark README now state that resource
  benchmarking is part of the formal benchmark contract
- CPU-mode and GPU-mode AtlasMTL runs are now explicitly documented as separate
  benchmark variants for future formal runs

## Current limitation

- external comparator wrappers still vary in how completely they expose device
  and thread information
- GPU average/peak memory is only meaningful when the comparator itself is run
  on CUDA and the wrapper can observe the same process
