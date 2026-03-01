#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[5] / ".tmp" / "numba_cache"))

from atlasmtl import build_model, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CUDA availability for AtlasMTL benchmark runs.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _tiny_adata() -> AnnData:
    obs = pd.DataFrame({"anno_lv1": ["A", "A", "B", "B"]}, index=[f"c{i}" for i in range(4)])
    adata = AnnData(
        X=np.array([[2, 0], [3, 1], [0, 2], [0, 3]], dtype=np.float32),
        obs=obs,
    )
    adata.var_names = ["g1", "g2"]
    return adata


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": None,
        "train_predict_smoke_passed": False,
        "error": None,
    }
    if report["cuda_available"]:
        report["device_name"] = torch.cuda.get_device_name(0)
        try:
            ref = _tiny_adata()
            query = ref[:2].copy()
            model = build_model(
                ref,
                label_columns=["anno_lv1"],
                hidden_sizes=[8],
                batch_size=2,
                num_epochs=1,
                input_transform="float",
                device="cuda",
                show_progress=False,
                show_summary=False,
            )
            _ = predict(
                model,
                query,
                knn_correction="off",
                input_transform="float",
                batch_size=2,
                device="cuda",
                show_progress=False,
                show_summary=False,
            )
            report["train_predict_smoke_passed"] = True
        except Exception as exc:  # pragma: no cover - environment-specific
            report["error"] = f"{type(exc).__name__}: {exc}"

    passed = bool(report["cuda_available"] and report["device_count"] > 0 and report["train_predict_smoke_passed"])
    report["gate_passed"] = passed
    (output_dir / "cuda_gate.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown = [
        "# CUDA Gate Report",
        "",
        f"- torch: `{report['torch_version']}`",
        f"- torch cuda: `{report['torch_cuda_version']}`",
        f"- cuda_available: `{report['cuda_available']}`",
        f"- device_count: `{report['device_count']}`",
        f"- device_name: `{report['device_name']}`",
        f"- train_predict_smoke_passed: `{report['train_predict_smoke_passed']}`",
        f"- gate_passed: `{report['gate_passed']}`",
    ]
    if report["error"]:
        markdown.extend(["", "## Error", "", f"`{report['error']}`"])
    (output_dir / "cuda_gate.md").write_text("\n".join(markdown), encoding="utf-8")
    print(output_dir / "cuda_gate.json")


if __name__ == "__main__":
    main()
