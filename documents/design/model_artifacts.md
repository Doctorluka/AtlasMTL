# atlasmtl Model Artifact Layout

## Recommended default

Use `reference_storage="external"` when building models. This keeps the main
model file smaller and makes KNN-heavy assets explicit.

## Main artifact bundle

Saving a `TrainedModel` writes these files by default:

- `model.pth`
  - neural network weights and architecture payload
- `model_metadata.pkl`
  - label encoders
  - train genes
  - coordinate targets and scaling statistics
  - training configuration and runtime summary
  - reference storage mode and optional reference path
- `model_reference.pkl`
  - external KNN reference coordinates and labels when
    `reference_storage="external"`
- `model_manifest.json`
  - stable automation entry point
  - stores relative paths to the model, metadata, and reference assets
  - stores `reference_storage`
  - stores `input_transform`
  - stores optional SHA-256 checksums

The manifest is generated from `atlasmtl/models/manifest.py` and is the
recommended load entry point for scripts and benchmark runs.

## Storage modes

### `external`

- recommended default
- stores reference data outside metadata
- default external filename is `*_reference.pkl`
- custom reference paths are supported
- `.pkl.gz` paths are also supported for compressed reference storage

### `full`

- embeds reference coordinates and labels into metadata
- useful for small self-contained bundles
- avoids a separate reference file at the cost of larger metadata size

## Path resolution and loading behavior

Manifest paths are stored relative to the manifest directory when possible.
Loading uses this order:

1. if the input path is a manifest, resolve artifact paths from the manifest
2. if the input path is a `.pth` file and a paired manifest exists, use the
   manifest
3. otherwise fall back to the legacy filename convention

This keeps `TrainedModel.load()` compatible with both explicit manifest-driven
loading and older file layouts.

## Checksums

When saving a model, atlasmtl computes best-effort SHA-256 checksums for:

- `model_path`
- `metadata_path`
- `reference_path` when external reference storage is used

Those values are written under `model_manifest.json["checksums"]`.

Checksums are intended for:

- artifact traceability
- auditability in benchmark runs
- safer artifact relocation and automation

They are not yet a full experiment reproducibility protocol by themselves. Split
definitions and dataset manifests still need to be tracked separately.

## Run manifests

In addition to the model artifact bundle, atlasmtl currently writes run-level
manifests:

- `train_run_manifest.json`
  - written by `scripts/train_atlasmtl.py`
  - records the output path, labels, device request, and `train_config`
- `predict_run_manifest.json`
  - written by `scripts/predict_atlasmtl.py`
  - records the model path, output path, device request, and prediction
    metadata
- `run_manifest.json`
  - written by `benchmark/pipelines/run_benchmark.py`
  - records the dataset manifest path, methods run, Python executable, and
    benchmark-written artifact checksums when available

These files improve traceability, but they do not yet fully standardize split
recording or all seed metadata across every workflow.

## Runtime behavior

- `predict(knn_correction="off")` does not require KNN reference data
- `predict(knn_correction="low_conf_only" | "all")` requires reference data to
  be available
- `TrainedModel.load()` accepts either `model.pth` or `model_manifest.json`
- external reference data can be loaded from plain pickle or gzip-compressed
  pickle

## Known gaps against the roadmap

The current artifact layer already supports:

- explicit manifests
- artifact checksums
- compressed external reference storage

The remaining roadmap work is protocol-level rather than serialization-level:

- standard split recording
- standard seed recording
- benchmark-level artifact/result provenance rules
- benchmark evidence for preset and compressed-reference tradeoffs
