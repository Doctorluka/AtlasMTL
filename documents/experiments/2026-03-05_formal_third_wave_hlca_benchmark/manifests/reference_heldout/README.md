# Formal runtime manifests (`HLCA_Core`)

This directory stores formal third-wave runtime manifests for:

- `train10k` reference build
- `test5k` heldout prediction
- separated CPU and GPU method groups

Manifest status:

- `*_v1.yaml`
  - historical pilot manifests used for the `2026-03-05` HLCA formal pilot
  - kept unchanged for auditability
- `*_v2.yaml`
  - rerun-ready manifests aligned to the locked formal defaults from:
    - `scanvi` lock dossier (`2026-03-06`)
    - `atlasmtl` lock dossier (`2026-03-07`)
