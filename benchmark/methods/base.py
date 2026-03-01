from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .azimuth import run_azimuth
from .celltypist import run_celltypist
from .reference_knn import run_reference_knn
from .scanvi import run_scanvi
from .singler import run_singler
from .symphony import run_symphony


def run_method(
    method: str,
    manifest: Dict[str, Any],
    *,
    output_dir: Path,
    atlasmtl_model: Optional[str],
    device: str,
) -> Dict[str, Any]:
    if method == "reference_knn":
        return run_reference_knn(manifest, output_dir=output_dir)
    if method == "azimuth":
        return run_azimuth(manifest, output_dir=output_dir)
    if method == "celltypist":
        return run_celltypist(manifest, output_dir=output_dir)
    if method == "scanvi":
        return run_scanvi(manifest, output_dir=output_dir, device=device)
    if method == "singler":
        return run_singler(manifest, output_dir=output_dir)
    if method == "symphony":
        return run_symphony(manifest, output_dir=output_dir)
    raise ValueError(f"method not implemented yet: {method}")
