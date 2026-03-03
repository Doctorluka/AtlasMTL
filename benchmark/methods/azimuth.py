from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .seurat_anchor_transfer import run_seurat_anchor_transfer


def run_azimuth(
    manifest: Dict[str, Any],
    *,
    output_dir: Path,
) -> Dict[str, Any]:
    return run_seurat_anchor_transfer(manifest, output_dir=output_dir)
