# Plan

## Goal

Use experimental evidence to compare two KNN geometry strategies under the real
deployment assumption ("query has no embeddings"):

- `predicted_scanvi_head`
- `latent_internal`

## Grid

- geometry_mode ∈ {`predicted_scanvi_head`, `latent_internal`}
- knn_variant ∈ {`knn_off`, `knn_lowconf`, `knn_all`}
- device ∈ {`cpu`} (optionally `cuda` as an ablation)

## Fixed defaults

- `feature_mode=hvg6000`
- `input_transform=binary`
- `task_weights=[0.3, 0.8, 1.5, 2.0]`
- `knn_k=15`
- `knn_conf_low=0.6`
- `enforce_hierarchy=true`

## Outputs

- per-level label metrics
- coverage / Unknown rate
- KNN rescue/harm/change metrics
- resource usage (train + predict)
- coordinate regression metrics for A (diagnostic-only)

