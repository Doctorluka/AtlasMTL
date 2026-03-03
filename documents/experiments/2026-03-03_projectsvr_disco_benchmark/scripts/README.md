# DISCO scripts

Put only DISCO pilot scripts here.

Keep the “count-like `adata.X` without `layers[\"counts\"]`” contract explicit.

Current first-wave entrypoint:

- `run_prepare_reference_heldout.sh`
  - materializes the `5k/1k` DISCO heldout preparation assets
  - validates count-like `adata.X`, then standardizes it into `layers["counts"]`
  - uses `sample` as both `split_key` and `domain_key`
  - writes prepared outputs to the manifest-declared `~/tmp/...` location
