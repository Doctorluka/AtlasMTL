# Error-fix note (`2026-03-06`, scanvi param lock)

## Recorded issue

- During early smoke/prototype checks in sandboxed execution, `scanvi` GPU runs failed with:
  - `MisconfigurationException: No supported gpu backend found!`

## Root cause

- sandbox runtime cannot reliably expose CUDA devices to `torch/lightning/scvi`.

## Resolution

- moved official stage-A and stage-B runs to non-sandbox GPU execution.
- retained sandbox only for lightweight pipeline smoke or file-shape checks.

## Impact on final evidence run

- none; final stage-A and stage-B official runs are complete (`80/80` total, all success).
