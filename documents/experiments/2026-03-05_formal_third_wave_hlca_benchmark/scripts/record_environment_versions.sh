#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOSSIER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${DOSSIER_DIR}/results_summary"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python}"

"${PYTHON_BIN}" - <<'PY' > "${OUT_DIR}/environment_versions_2026-03-05.md"
import importlib
from datetime import datetime

pkgs = [
    "atlasmtl",
    "phmap",
    "scanpy",
    "anndata",
    "scvi",
    "celltypist",
    "sklearn",
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "pytorch_lightning",
    "joblib",
]

print("# Environment versions (formal HLCA third-wave)")
print("")
print(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
print("")
print("| package | version |")
print("| --- | --- |")
for name in pkgs:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
    except Exception as exc:
        version = f"ERROR:{type(exc).__name__}:{exc}"
    print(f"| {name} | {version} |")
PY

"${PYTHON_BIN}" -m pip freeze > "${OUT_DIR}/pip_freeze_2026-03-05.txt"
Rscript -e 'sessionInfo()' > "${OUT_DIR}/r_sessioninfo_2026-03-05.txt"
Rscript -e 'x <- as.data.frame(installed.packages()[, c("Package","Version")]); write.csv(x, file="documents/experiments/2026-03-05_formal_third_wave_hlca_benchmark/results_summary/r_packages_2026-03-05.csv", row.names=FALSE)'
