from __future__ import annotations

from collections import Counter
from typing import Any, Dict

import numpy as np
import pandas as pd
from anndata import AnnData


def _label_counts(adata: AnnData, label_column: str) -> dict[str, int]:
    values = adata.obs[label_column].astype(str)
    return {str(key): int(val) for key, val in values.value_counts(dropna=False).to_dict().items()}


def _materialize_subset(
    adata: AnnData,
    *,
    allowed_groups: set[str],
    split_key: str,
    subset_size: int,
    seed: int,
) -> AnnData:
    mask = adata.obs[split_key].astype(str).isin(sorted(allowed_groups)).to_numpy()
    pool = adata[mask].copy()
    if pool.n_obs < subset_size:
        raise ValueError(f"pool size {pool.n_obs} is smaller than requested subset size {subset_size}")
    if pool.n_obs == subset_size:
        return pool
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(pool.n_obs, size=subset_size, replace=False))
    return pool[chosen].copy()


def make_group_split_plan(
    adata: AnnData,
    *,
    split_key: str,
    target_label: str,
    build_size: int,
    predict_size: int,
    seed: int,
    n_candidates: int = 128,
) -> Dict[str, Any]:
    if split_key not in adata.obs.columns:
        raise ValueError(f"split_key not found in adata.obs: {split_key}")
    if target_label not in adata.obs.columns:
        raise ValueError(f"target_label not found in adata.obs: {target_label}")

    groups = adata.obs[split_key].astype(str)
    labels = adata.obs[target_label].astype(str)
    group_counts = groups.value_counts().to_dict()
    group_labels = {
        str(group): dict(Counter(labels.loc[groups == group].tolist()))
        for group in groups.unique().tolist()
    }

    candidate_summaries = []
    rejected_reasons: Counter[str] = Counter()
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(groups.unique().tolist()), dtype=object)

    for candidate_id in range(n_candidates):
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        build_groups: list[str] = []
        predict_groups: list[str] = []
        build_total = 0
        predict_total = 0

        for group in shuffled.tolist():
            size = int(group_counts[group])
            if build_total < build_size:
                build_groups.append(group)
                build_total += size
            elif predict_total < predict_size:
                predict_groups.append(group)
                predict_total += size
            elif build_total <= predict_total:
                build_groups.append(group)
                build_total += size
            else:
                predict_groups.append(group)
                predict_total += size

        build_set = set(build_groups)
        predict_set = set(predict_groups)
        if build_set & predict_set:
            rejected_reasons["group_leakage"] += 1
            continue
        if build_total < build_size:
            rejected_reasons["build_too_small"] += 1
            continue
        if predict_total < predict_size:
            rejected_reasons["predict_too_small"] += 1
            continue

        build_mask = groups.isin(build_set)
        predict_mask = groups.isin(predict_set)
        build_label_counts = dict(Counter(labels.loc[build_mask].tolist()))
        predict_label_counts = dict(Counter(labels.loc[predict_mask].tolist()))
        if len(build_label_counts) < 2:
            rejected_reasons["build_label_collapse"] += 1
            continue
        if len(predict_label_counts) < 2:
            rejected_reasons["predict_label_collapse"] += 1
            continue

        score = (
            -(abs(build_total - build_size) + abs(predict_total - predict_size)),
            len(predict_label_counts),
            min(predict_label_counts.values()),
            len(build_label_counts),
            -abs(build_total - predict_total),
        )
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "build_groups": sorted(build_set),
                "predict_groups": sorted(predict_set),
                "build_pool_cells": int(build_total),
                "predict_pool_cells": int(predict_total),
                "build_label_counts": {str(k): int(v) for k, v in build_label_counts.items()},
                "predict_label_counts": {str(k): int(v) for k, v in predict_label_counts.items()},
                "score": score,
            }
        )

    if not candidate_summaries:
        raise ValueError(f"no valid split candidate found; rejected reasons: {dict(rejected_reasons)}")

    best = sorted(
        candidate_summaries,
        key=lambda row: (
            -row["score"][0],
            -row["score"][1],
            -row["score"][2],
            -row["score"][3],
            row["score"][4],
            tuple(row["build_groups"]),
            tuple(row["predict_groups"]),
        ),
    )[0]

    warnings = []
    if min(best["predict_label_counts"].values()) < 5:
        warnings.append("heldout_pool_has_label_with_lt5_cells")
    if min(best["build_label_counts"].values()) < 10:
        warnings.append("build_pool_has_label_with_lt10_cells")

    return {
        "split_key": split_key,
        "target_label": target_label,
        "seed": int(seed),
        "n_candidates": int(n_candidates),
        "candidate_count_valid": int(len(candidate_summaries)),
        "candidate_rejections": {str(k): int(v) for k, v in rejected_reasons.items()},
        "chosen_candidate_id": int(best["candidate_id"]),
        "build_groups": list(best["build_groups"]),
        "predict_groups": list(best["predict_groups"]),
        "build_pool_cells": int(best["build_pool_cells"]),
        "predict_pool_cells": int(best["predict_pool_cells"]),
        "build_pool_label_counts": dict(best["build_label_counts"]),
        "predict_pool_label_counts": dict(best["predict_label_counts"]),
        "warnings": warnings,
    }


def materialize_group_split_subsets(
    adata: AnnData,
    plan: Dict[str, Any],
    *,
    build_size: int,
    predict_size: int,
    seed: int,
) -> Dict[str, Any]:
    split_key = str(plan["split_key"])
    target_label = str(plan["target_label"])
    build = _materialize_subset(
        adata,
        allowed_groups=set(plan["build_groups"]),
        split_key=split_key,
        subset_size=build_size,
        seed=seed,
    )
    predict = _materialize_subset(
        adata,
        allowed_groups=set(plan["predict_groups"]),
        split_key=split_key,
        subset_size=predict_size,
        seed=seed + 1,
    )
    warnings = list(plan.get("warnings") or [])
    build_counts = _label_counts(build, target_label)
    predict_counts = _label_counts(predict, target_label)
    if min(build_counts.values()) < 10:
        warnings.append("build_subset_has_label_with_lt10_cells")
    if min(predict_counts.values()) < 5:
        warnings.append("predict_subset_has_label_with_lt5_cells")
    return {
        "reference_build_adata": build,
        "predict_adata": predict,
        "split_summary": {
            **dict(plan),
            "build_subset_cells": int(build.n_obs),
            "predict_subset_cells": int(predict.n_obs),
            "build_subset_label_counts": build_counts,
            "predict_subset_label_counts": predict_counts,
            "warnings": warnings,
        },
    }
