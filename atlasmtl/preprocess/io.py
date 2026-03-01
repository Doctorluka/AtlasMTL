from __future__ import annotations

import json
from pathlib import Path

from .types import FeaturePanel


def save_feature_panel(feature_panel: FeaturePanel, path: str) -> None:
    Path(path).write_text(
        json.dumps(feature_panel.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_feature_panel(path: str) -> FeaturePanel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FeaturePanel.from_dict(payload)
