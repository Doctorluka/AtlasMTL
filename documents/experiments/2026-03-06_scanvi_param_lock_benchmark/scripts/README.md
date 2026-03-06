# Scripts (`scanvi param lock benchmark`)

- `prepare_scanvi_param_lock_inputs.py`
  - materialize `reference_train_10k.h5ad` for all target datasets
  - copy prepared `heldout_test_5k.h5ad` and `heldout_test_10k.h5ad`
- `run_scanvi_param_sweep.py`
  - execute `scanvi` benchmark runs for `stage_a` or `stage_b`
  - wraps `documents/experiments/common/run_reference_heldout_scaleout_benchmark.py`
- `aggregate_scanvi_param_sweep.py`
  - aggregate raw run outputs
  - export stage-A ranking and stage-B stability
  - write `top2_params.json` for stage-B reruns
- `render_scanvi_param_tables.py`
  - render markdown tables from aggregated csv outputs

## Typical command sequence

```bash
python documents/experiments/2026-03-06_scanvi_param_lock_benchmark/scripts/prepare_scanvi_param_lock_inputs.py
python documents/experiments/2026-03-06_scanvi_param_lock_benchmark/scripts/run_scanvi_param_sweep.py --stage stage_a
python documents/experiments/2026-03-06_scanvi_param_lock_benchmark/scripts/aggregate_scanvi_param_sweep.py
python documents/experiments/2026-03-06_scanvi_param_lock_benchmark/scripts/run_scanvi_param_sweep.py --stage stage_b
python documents/experiments/2026-03-06_scanvi_param_lock_benchmark/scripts/aggregate_scanvi_param_sweep.py
python documents/experiments/2026-03-06_scanvi_param_lock_benchmark/scripts/render_scanvi_param_tables.py
```
