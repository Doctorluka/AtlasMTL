# atlasmtl Model Artifact Layout

## Recommended default

Use `reference_storage="external"` when building models.

## Artifact files

- `model.pth`
  - neural network weights and architecture payload
- `model_metadata.pkl`
  - label encoders
  - train genes
  - coordinate statistics
  - training config
  - reference storage mode
  - optional reference path
- `model_reference.pkl`
  - external KNN reference coordinates
  - external KNN reference labels
- `model_manifest.json`
  - light-weight artifact index
  - records model path, metadata path, reference path, storage mode, and schema version

## Why this layout

- Keeps the main model artifact smaller.
- Makes reference-heavy KNN assets explicit.
- Improves portability when users want to version model weights and reference data separately.
- Provides a stable load entry for automation and benchmark runners without relying on filename guessing.

## Storage modes

- `external`
  - recommended default
  - stores reference data in `*_reference.pkl`
- `full`
  - stores reference data in metadata
  - useful for compact distribution at small scale

## Runtime behavior

- `predict(knn_correction="off")` does not require KNN reference data.
- `predict(knn_correction="low_conf_only" | "all")` requires reference data to be available.
- `TrainedModel.load()` accepts either `model.pth` or `model_manifest.json`.
- If no manifest is present, loading falls back to the legacy filename convention.
