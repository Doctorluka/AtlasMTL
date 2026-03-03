# 2026-03-03 Seurat anchor transfer comparator refactor

- Formal comparator definition changed from `azimuth` to `seurat_anchor_transfer`.
- The previous runnable path was already a Seurat anchor-transfer fallback in
  most real smoke runs, so the method name and the executed backend were not
  aligned.
- The new comparator is implemented as a Seurat anchor-transfer plus
  `MapQuery` workflow and no longer attempts native Azimuth reference-package
  construction.
- The reference flow is aligned with the ProjectSVR Seurat benchmark structure,
  but the feature policy remains the atlasmtl benchmark-wide `HVG 3000`
  contract instead of inheriting a `2000`-feature default.
- README, benchmark protocol, pilot manifests, and smoke orchestration were
  updated to use `seurat_anchor_transfer`.
