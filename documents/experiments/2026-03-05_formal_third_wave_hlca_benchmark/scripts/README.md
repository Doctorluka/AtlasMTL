# Scripts (`formal third-wave HLCA`)

- `prepare_formal_hlca_train10k_test5k.py`
  - materialize `reference_train_10k.h5ad` from existing HLCA second-wave prepared reference
  - keep heldout at `heldout_test_5k.h5ad`
- `record_environment_versions.sh`
  - write key package table and full environment snapshots
- `export_formal_tables.py`
  - generate unified CPU+GPU performance/resource tables from method `metrics.json`
- `run_formal_hlca_cpu_group.sh`
  - run CPU group (`atlasmtl`, `reference_knn`, `celltypist`, `singler`, `symphony`, `seurat_anchor_transfer`)
  - exports formal fairness policy and fixed thread env vars
- `run_formal_hlca_gpu_group.sh`
  - run GPU group (`atlasmtl`, `scanvi`)
  - performs CUDA preflight before launching benchmark
