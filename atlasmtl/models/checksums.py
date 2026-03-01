from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_checksums(paths: Dict[str, Optional[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, p in paths.items():
        if not p:
            continue
        if not isinstance(p, (str, Path)):
            continue
        path = Path(p)
        if not path.exists():
            continue
        out[key] = sha256_file(str(path))
    return out
