# CUDA execution discrepancy

Observed state as of 2026-03-01:

- user interactive shell in `atlasmtl-env` reports:
  - `torch.cuda.is_available() == True`
  - device `NVIDIA GeForce RTX 4090`
- benchmark agent / non-interactive runner previously reported:
  - `torch.cuda.is_available() == False`
  - `cudaGetDeviceCount()` error 304
  - NVML initialization warning

Implication:

- GPU support should not be assumed from code alone
- every formal AtlasMTL GPU benchmark must first pass the benchmark-entry CUDA
  gate in `scripts/check_cuda_gate.py`
- if the gate fails, CPU remains the formal result and GPU is documented as
  environment-unverified rather than method-unsupported
