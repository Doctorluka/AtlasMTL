#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import anndata as ad
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
ROUND_ROOT = REPO_ROOT / "documents" / "experiments" / "2026-03-09_multilevel_annotation_benchmark"
RESULTS_DIR = ROUND_ROOT / "results_summary"

DATASETS: Dict[str, Dict[str, object]] = {
    "HLCA_Core": {
        "label_columns": ["ann_level_1", "ann_level_2", "ann_level_3", "ann_level_4", "ann_level_5"],
        "hierarchy_depth_class": "deep_multilevel",
        "reference_h5ad": "/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/HLCA_Core/prepared/formal_split_v1/build_scaling/build_100000/reference_train_100000.h5ad",
    },
    "PHMap_Lung_Full_v43_light": {
        "label_columns": ["anno_lv1", "anno_lv2", "anno_lv3", "anno_lv4"],
        "hierarchy_depth_class": "deep_multilevel",
        "reference_h5ad": "/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/PHMap_Lung_Full_v43_light/prepared/formal_split_v1/build_scaling/build_100000/reference_train_100000.h5ad",
    },
    "DISCO_hPBMCs": {
        "label_columns": ["cell_type", "cell_subtype"],
        "hierarchy_depth_class": "shallow_multilevel",
        "reference_h5ad": "/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/DISCO_hPBMCs/prepared/formal_split_v1/build_scaling/build_100000/reference_train_100000.h5ad",
    },
    "mTCA": {
        "label_columns": ["Cell_type_level1", "Cell_type_level2", "Cell_type_level3"],
        "hierarchy_depth_class": "mid_multilevel",
        "reference_h5ad": "/tmp/atlasmtl_benchmarks/2026-03-06/formal_third_wave/mTCA/prepared/formal_split_v1/build_scaling/build_100000/reference_train_100000.h5ad",
    },
}


def _class_counts(adata: ad.AnnData, columns: List[str]) -> List[int]:
    return [int(adata.obs[col].dropna().astype(str).nunique()) for col in columns]


def main() -> None:
    rows = []
    for dataset, spec in DATASETS.items():
        reference_h5ad = Path(str(spec["reference_h5ad"])).resolve()
        adata = ad.read_h5ad(reference_h5ad)
        label_columns = list(spec["label_columns"])
        rows.append(
            {
                "dataset": dataset,
                "reference_h5ad": reference_h5ad.as_posix(),
                "level_columns": json.dumps(label_columns),
                "num_levels": len(label_columns),
                "coarsest_label_column": label_columns[0],
                "finest_label_column": label_columns[-1],
                "n_classes_by_level": json.dumps(_class_counts(adata, label_columns)),
                "hierarchy_depth_class": str(spec["hierarchy_depth_class"]),
            }
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "dataset_hierarchy_inventory.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(json.dumps({"row_count": len(rows), "output_csv": out_path.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
