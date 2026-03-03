from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from atlasmtl.preprocess.split import make_group_split_plan, materialize_group_split_subsets


def _make_adata() -> AnnData:
    obs_rows = []
    for sample_idx in range(8):
        sample = f"s{sample_idx}"
        label = "A" if sample_idx < 4 else "B"
        for cell_idx in range(300):
            obs_rows.append({"sample": sample, "label": label, "cell_id": f"{sample}_{cell_idx}"})
    obs = pd.DataFrame(obs_rows).set_index("cell_id")
    adata = AnnData(X=np.ones((len(obs_rows), 2), dtype=np.float32), obs=obs)
    adata.var_names = ["ENSG1", "ENSG2"]
    return adata


def test_make_group_split_plan_is_group_aware_and_deterministic():
    adata = _make_adata()
    plan_a = make_group_split_plan(
        adata,
        split_key="sample",
        target_label="label",
        build_size=1200,
        predict_size=600,
        seed=2026,
        n_candidates=32,
    )
    plan_b = make_group_split_plan(
        adata,
        split_key="sample",
        target_label="label",
        build_size=1200,
        predict_size=600,
        seed=2026,
        n_candidates=32,
    )

    assert plan_a["build_groups"] == plan_b["build_groups"]
    assert plan_a["predict_groups"] == plan_b["predict_groups"]
    assert set(plan_a["build_groups"]).isdisjoint(set(plan_a["predict_groups"]))
    assert plan_a["build_pool_cells"] >= 1200
    assert plan_a["predict_pool_cells"] >= 600


def test_materialize_group_split_subsets_hits_requested_sizes():
    adata = _make_adata()
    plan = make_group_split_plan(
        adata,
        split_key="sample",
        target_label="label",
        build_size=1200,
        predict_size=600,
        seed=2026,
        n_candidates=32,
    )
    materialized = materialize_group_split_subsets(
        adata,
        plan,
        build_size=1200,
        predict_size=600,
        seed=2026,
    )

    build = materialized["reference_build_adata"]
    predict = materialized["predict_adata"]
    assert build.n_obs == 1200
    assert predict.n_obs == 600
    assert set(build.obs["sample"].astype(str)).isdisjoint(set(predict.obs["sample"].astype(str)))
    assert materialized["split_summary"]["build_subset_cells"] == 1200
    assert materialized["split_summary"]["predict_subset_cells"] == 600
