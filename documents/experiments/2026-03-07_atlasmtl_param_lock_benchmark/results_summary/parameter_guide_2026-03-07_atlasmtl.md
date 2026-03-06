# AtlasMTL parameter guide (`2026-03-07`)

This guide summarizes the benchmark-facing atlasmtl parameter scope.

## Fixed in this lock round

- `num_threads=8`
- `input_transform=binary`
- `max_epochs=50`
- `val_fraction=0.1`
- `early_stopping_patience=5`
- `early_stopping_min_delta=0.0`
- `reference_storage=external`

## Core tuned in this lock round

- `learning_rate`
- `hidden_sizes`
- `batch_size`

## CPU grid

- `learning_rate`: `1e-3`, `3e-4`, `1e-4`
- `hidden_sizes`: `[256,128]`, `[512,256]`
- `batch_size`: `128`, `256`

## GPU grid

- `learning_rate`: `1e-3`, `3e-4`, `1e-4`
- `hidden_sizes`: `[256,128]`, `[512,256]`, `[1024,512]`
- `batch_size`: `256`, `512`

## Deferred to ablation

- `domain_loss_weight`
- `topology_loss_weight`
- `calibration_method`
