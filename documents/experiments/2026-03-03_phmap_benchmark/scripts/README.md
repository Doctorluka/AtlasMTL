# PH-Map scripts

Put only PH-Map pilot scripts here.

Do not share ad hoc dataset-specific assumptions across dossiers without
explicitly copying and reviewing them.

Current first-wave entrypoint:

- `run_prepare_reference_heldout.sh`
  - materializes the `5k/1k` PH-Map heldout preparation assets
  - uses `sample` as both `split_key` and `domain_key`
  - writes prepared outputs to the manifest-declared `~/tmp/...` location
