# Scripts (`atlasmtl param lock benchmark`)

- `prepare_atlasmtl_param_lock_inputs.py`
  - materialize `reference_train_10k.h5ad` and copy `heldout_test_5k/10k.h5ad`
- `run_atlasmtl_param_sweep.py`
  - run stage-A or stage-B sweeps for one device track (`cpu` or `cuda`)
- `aggregate_atlasmtl_param_sweep.py`
  - aggregate stage outputs, rank configs, generate top-2 per track
- `render_atlasmtl_param_tables.py`
  - render markdown tables from aggregated csv outputs

## Typical execution order

```bash
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/prepare_atlasmtl_param_lock_inputs.py
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/run_atlasmtl_param_sweep.py --stage stage_a --device cpu
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/run_atlasmtl_param_sweep.py --stage stage_a --device cuda
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/aggregate_atlasmtl_param_sweep.py
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/run_atlasmtl_param_sweep.py --stage stage_b --device cpu
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/run_atlasmtl_param_sweep.py --stage stage_b --device cuda
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/aggregate_atlasmtl_param_sweep.py
python documents/experiments/2026-03-07_atlasmtl_param_lock_benchmark/scripts/render_atlasmtl_param_tables.py
```
