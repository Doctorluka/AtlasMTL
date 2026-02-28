#!/usr/bin/env python
"""Benchmark runner skeleton for atlasmtl and published comparator tools."""

from __future__ import annotations

import argparse
from pathlib import Path

from atlasmtl.models import resolve_manifest_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--atlasmtl-model",
        help="Path to atlasmtl `model.pth` or `model_manifest.json` for benchmark runs.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["atlasmtl"],
        help="Benchmark methods to run. Comparator wrappers are added incrementally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Benchmark runner skeleton")
    print(f"Dataset manifest: {args.dataset_manifest}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Output dir: {output_dir}")
    if args.atlasmtl_model:
        model_path = Path(args.atlasmtl_model)
        if model_path.suffix == ".json":
            artifacts = resolve_manifest_paths(str(model_path))
            print(f"atlasmtl manifest: {model_path}")
            print(f"  resolved model: {artifacts['model_path']}")
            print(f"  resolved metadata: {artifacts['metadata_path']}")
            print(f"  resolved reference: {artifacts['reference_path']}")
        else:
            print(f"atlasmtl model: {model_path}")
    print("Comparator execution is not implemented yet.")


if __name__ == "__main__":
    main()
