# atlasmtl

`atlasmtl` is a generic single-cell reference mapping package with multi-task label prediction, coordinate regression, gated KNN correction, and Unknown-cell abstention.

## Design goals
- AnnData in, AnnData out.
- Multi-level annotation with shared representation.
- Optional coordinate prediction (`X_pred_latent`, `X_pred_umap`).
- Low-confidence-only KNN correction to save compute.

## Quick start
```python
import atlasmtl
model = atlasmtl.build_model(
    adata=adata_ref,
    label_columns=["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"],
    coord_targets={"latent": "X_ref_latent", "umap": "X_umap"},
    device="auto",
)
model.save("model.pth")
model = atlasmtl.TrainedModel.load("model_manifest.json")

result = atlasmtl.predict(model, adata_query)
adata_query = result.to_adata(adata_query)
labels = result.to_dataframe(mode="minimal")
result.to_csv("predictions.csv", mode="minimal")
model.show_resource_usage()
result.show_resource_usage()
```

`show_resource_usage()` now prints a compact sectioned summary. Training and
prediction also support `show_summary=True` to print this summary automatically
when the run finishes. When `show_summary` is left as `None`, atlasmtl prints
the summary automatically only in interactive terminals.

Model storage:

- `build_model()` defaults to `reference_storage="external"`.
- Recommended artifact layout:
  - `model.pth`: network weights
  - `model_metadata.pkl`: model metadata and config
  - `model_reference.pkl`: external KNN reference data
  - `model_manifest.json`: light-weight artifact manifest and stable load entry
- This keeps the main model artifact lighter than embedding full reference data into metadata.

You can load from either `model.pth` or `model_manifest.json`. The manifest is the more stable entry point when moving model artifacts between directories or automation steps.

Core tuning parameters:

- `build_model(...)`
  - `hidden_sizes`: shared encoder widths, default effective value `[256, 128]`
  - `dropout_rate`: encoder dropout, default `0.3`
  - `batch_size`: train batch size, default `256`
  - `num_epochs`: max epochs, default `40`
  - `learning_rate`: Adam learning rate, default `1e-3`
  - `input_transform`: `"binary"` or `"float"`, default `"binary"`
  - `coord_targets`: `None` means no-coordinate mode; pass an explicit `obsm` mapping to enable coordinate heads
  - `val_fraction`: optional validation split, default `0.0`
  - `early_stopping_patience`: disabled when `None`
  - `reference_storage`: `"external"` or `"full"`, default `"external"`
  - `num_threads`: default `10`; `"max"` uses up to 80% of CPUs
  - `device`: `"auto" | "cpu" | "cuda"`, default `"auto"`
  - `show_progress`: auto-detected by terminal; displays epoch progress and ETA during training
  - `show_summary`: auto-detected by terminal; prints a compact training summary when finished
- `predict(...)`
  - `knn_correction`: `"off" | "low_conf_only" | "all"`, default `"low_conf_only"`
  - `confidence_high`: low-confidence gating threshold, default `0.7`
  - `confidence_low`: Unknown threshold, default `0.4`
  - `margin_threshold`: top1-top2 margin threshold, default `0.2`
  - `knn_k`: reference neighbors, default `15`
  - `knn_conf_low`: minimum KNN vote fraction for closed-loop acceptance, default `0.6`
  - `num_threads`: default `10`; `"max"` uses up to 80% of CPUs
  - `device`: `"auto" | "cpu" | "cuda"`, default `"auto"`
  - `show_progress`: auto-detected by terminal; displays batch progress and ETA during inference
  - `show_summary`: auto-detected by terminal; prints a compact prediction summary when finished

Training metadata:

- `model.train_config["train_seconds"]`: elapsed wall-clock training time
- `model.train_config["resource_summary"]`: training cell/gene counts, device type, CPU count, GPU name, and completed epochs
- `model.train_config["runtime_summary"]`: elapsed time, throughput, process peak RSS, and GPU peak memory
- `result.metadata["prediction_runtime"]`: inference elapsed time, throughput, process peak RSS, and GPU peak memory
- `model.show_resource_usage()` / `result.show_resource_usage()`: print a compact runtime summary directly to the terminal

Writeback modes:

- `result.to_adata(adata, mode="minimal")`: write only `pred_<level>` plus `uns["atlasmtl"]` by default.
- `result.to_adata(adata, mode="standard")`: write `pred_<level>`, `conf_<level>`, `margin_<level>`, `is_unknown_<level>`.
- `result.to_adata(adata, mode="full", include_coords=True)`: write all prediction columns and predicted coordinates.
- `result.to_dataframe(mode="standard")`: export the same mode-filtered prediction table without modifying `AnnData`.
- `result.to_csv("predictions.csv", mode="minimal")`: save a light-weight label table directly to disk.

Coordinates are never written by default. Set `include_coords=True` if you want `obsm["X_pred_*"]`.
If you do not need `AnnData` writeback, prefer `to_dataframe()` or `to_csv()` to keep result handling lighter.

For a stable summary of the public interfaces, artifact layout, and writeback/export behavior, see `documents/design/api_contract.md`.

## Development environment

This project is developed against the mamba env:

- `/home/data/fhz/.local/share/mamba/envs/atlasmtl-env`

Verified (2026-02-28):

- Python 3.11.14
- phmap 0.1.1
- torch 2.10.0
- numpy 2.4.2
- pandas 2.3.3
- scanpy 1.11.5
- anndata 0.12.10
- scikit-learn 1.8.0
- numba 0.64.0

Dev tools (install if missing):

```bash
/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/pip install -e ".[dev]"
```

If `import scanpy` fails due to numba cache locator / write restrictions, set a writable cache dir:

```bash
export NUMBA_CACHE_DIR=/tmp/numba_cache
```

## Project layout
- `atlasmtl/`: package source code
- `benchmark/`: benchmarking pipelines and reports
- `documents/`: design docs and paper assets
- `notebooks/`: reproducible examples
- `vendor/phmap_snapshot/`: read-only source reference
