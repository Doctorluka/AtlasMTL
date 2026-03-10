from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from anndata import AnnData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data import compute_coord_stats, extract_matrix, scale_coords
from .model import AtlasMTLModel
from .runtime import configure_torch_threads, resolve_device
from ..mapping.calibration import fit_temperature_scaling
from ..models import ReferenceData
from ..models.presets import resolve_preset
from .types import TrainedModel
from ..utils import RuntimeMonitor, progress_iter, resolve_show_summary


def _encode_internal_latent(
    model: AtlasMTLModel,
    X: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    show_progress: Optional[bool] = None,
) -> np.ndarray:
    model.eval()
    latents: List[np.ndarray] = []
    loader = DataLoader(X, batch_size=batch_size, shuffle=False)
    batch_iterator = progress_iter(
        loader,
        total=len(loader),
        desc="atlasmtl encode latent",
        show_progress=show_progress,
    )
    with torch.no_grad():
        for batch in batch_iterator:
            _, _, z = model(batch.to(device))
            latents.append(z.detach().cpu().numpy())
    if not latents:
        raise ValueError("No cells available for latent encoding")
    return np.asarray(np.concatenate(latents, axis=0), dtype=np.float32)


def _domain_mean_penalty(z: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    """Lightweight domain alignment penalty based on mean embedding matching."""
    unique = torch.unique(domains)
    if unique.numel() <= 1:
        return torch.zeros((), device=z.device)
    means = []
    for d in unique:
        mask = domains == d
        if torch.any(mask):
            means.append(z[mask].mean(dim=0))
    if len(means) <= 1:
        return torch.zeros((), device=z.device)
    penalty = torch.zeros((), device=z.device)
    pairs = 0
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            diff = means[i] - means[j]
            penalty = penalty + torch.mean(diff * diff)
            pairs += 1
    return penalty / float(max(pairs, 1))


def _topology_loss(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    """Neighborhood distance preservation loss within a minibatch."""
    if k <= 0:
        return torch.zeros((), device=pred_coords.device)
    n = int(pred_coords.shape[0])
    if n <= 1:
        return torch.zeros((), device=pred_coords.device)
    k_eff = min(int(k), n - 1)
    with torch.no_grad():
        dist_target = torch.cdist(target_coords, target_coords)
        dist_target.fill_diagonal_(float("inf"))
        nn_idx = torch.topk(dist_target, k_eff, largest=False).indices
    dist_pred = torch.cdist(pred_coords, pred_coords)
    row = torch.arange(n, device=pred_coords.device).unsqueeze(1).expand(n, k_eff)
    d_t = dist_target[row, nn_idx]
    d_p = dist_pred[row, nn_idx]
    return torch.mean((d_p - d_t) ** 2)


def _resolve_balanced_label_config(
    config: Optional[Dict[str, Any]],
    *,
    label_columns: List[str],
    config_name: str,
) -> Optional[Tuple[int, str, str]]:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError(f"{config_name} must be a dict when provided")
    label_column = str(config.get("label_column") or label_columns[-1])
    if label_column not in label_columns:
        raise ValueError(f"{config_name}.label_column must be one of label_columns")
    mode = str(config.get("mode", "balanced")).lower()
    if mode != "balanced":
        raise ValueError(f"{config_name}.mode must be 'balanced'")
    return label_columns.index(label_column), label_column, mode


def _balanced_class_weights(
    y: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    if np.any(present):
        weights[present] = float(len(y)) / (float(np.sum(present)) * counts[present])
    return weights


def _class_distribution_metadata(
    *,
    classes: np.ndarray,
    counts: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    out = {
        "class_counts": {
            str(cls): int(count)
            for cls, count in zip(classes.tolist(), counts.tolist())
        }
    }
    if weights is not None:
        out["class_weights"] = {
            str(cls): float(weight)
            for cls, weight in zip(classes.tolist(), weights.tolist())
        }
    return out


def _resolve_parent_conditioned_child_correction_config(
    config: Optional[Dict[str, Any]],
    *,
    adata: AnnData,
    label_columns: List[str],
    label_encoders: Dict[str, LabelEncoder],
) -> Optional[Dict[str, Any]]:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("parent_conditioned_child_correction must be a dict when provided")
    parent_level = str(config.get("parent_level") or "")
    target_level = str(config.get("target_level") or "")
    hotspot_parents = [str(x) for x in list(config.get("hotspot_parents") or [])]
    mode = str(config.get("mode", "joint"))
    if parent_level not in label_columns:
        raise ValueError("parent_conditioned_child_correction.parent_level must be one of label_columns")
    if target_level not in label_columns:
        raise ValueError("parent_conditioned_child_correction.target_level must be one of label_columns")
    parent_head_index = label_columns.index(parent_level)
    child_head_index = label_columns.index(target_level)
    if child_head_index != parent_head_index + 1:
        raise ValueError("parent_conditioned_child_correction currently requires adjacent parent and target levels")
    if not hotspot_parents:
        raise ValueError("parent_conditioned_child_correction.hotspot_parents must be non-empty")
    if mode not in {"joint", "frozen_base"}:
        raise ValueError("parent_conditioned_child_correction.mode must be 'joint' or 'frozen_base'")
    base_lr_scale = float(config.get("base_lr_scale", 0.1))
    if not (0.0 <= base_lr_scale <= 1.0):
        raise ValueError("parent_conditioned_child_correction.base_lr_scale must be in [0, 1]")
    loss_weight = float(config.get("loss_weight", 1.0))
    if loss_weight < 0.0:
        raise ValueError("parent_conditioned_child_correction.loss_weight must be >= 0")
    hidden_dim = int(config.get("hidden_dim", 64))
    if hidden_dim <= 0:
        raise ValueError("parent_conditioned_child_correction.hidden_dim must be > 0")
    residual_scale = float(config.get("residual_scale", 1.0))
    if residual_scale < 0.0:
        raise ValueError("parent_conditioned_child_correction.residual_scale must be >= 0")
    feature_mode = str(config.get("feature_mode", "standard"))
    if feature_mode not in {"standard", "reranker_like"}:
        raise ValueError("parent_conditioned_child_correction.feature_mode must be 'standard' or 'reranker_like'")
    rank_loss_weight = float(config.get("rank_loss_weight", 0.0))
    if rank_loss_weight < 0.0:
        raise ValueError("parent_conditioned_child_correction.rank_loss_weight must be >= 0")
    rank_margin = float(config.get("rank_margin", 0.2))
    if rank_margin < 0.0:
        raise ValueError("parent_conditioned_child_correction.rank_margin must be >= 0")

    parent_encoder = label_encoders[parent_level]
    child_encoder = label_encoders[target_level]
    observed = adata.obs.loc[:, [parent_level, target_level]].dropna().copy()
    observed[parent_level] = observed[parent_level].astype(str)
    observed[target_level] = observed[target_level].astype(str)
    legal_child_names: Dict[str, List[str]] = {}
    hotspot_parent_indices: List[int] = []
    hotspot_child_indices: Dict[str, List[int]] = {}
    for parent_label in hotspot_parents:
        if parent_label not in parent_encoder.classes_:
            raise ValueError(
                f"parent_conditioned_child_correction hotspot parent not found in {parent_level}: {parent_label}"
            )
        child_names = sorted(observed.loc[observed[parent_level] == parent_label, target_level].unique().tolist())
        if not child_names:
            raise ValueError(
                f"parent_conditioned_child_correction hotspot parent has no observed {target_level} children: {parent_label}"
            )
        parent_idx = int(parent_encoder.transform([parent_label])[0])
        child_indices = [int(child_encoder.transform([child_name])[0]) for child_name in child_names]
        hotspot_parent_indices.append(parent_idx)
        hotspot_child_indices[str(parent_idx)] = child_indices
        legal_child_names[parent_label] = child_names
    return {
        "enabled": True,
        "parent_level": parent_level,
        "target_level": target_level,
        "parent_head_index": parent_head_index,
        "child_head_index": child_head_index,
        "hotspot_parents": hotspot_parents,
        "hotspot_parent_indices": hotspot_parent_indices,
        "hotspot_child_indices": hotspot_child_indices,
        "legal_child_names": legal_child_names,
        "mode": mode,
        "base_lr_scale": base_lr_scale,
        "loss_weight": loss_weight,
        "hidden_dim": hidden_dim,
        "residual_scale": residual_scale,
        "feature_mode": feature_mode,
        "rank_loss_weight": rank_loss_weight,
        "rank_margin": rank_margin,
    }


def _compute_parent_conditioned_correction_loss(
    corrected_logits: List[torch.Tensor],
    targets: List[torch.Tensor],
    correction_cfg: Dict[str, Any],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    parent_head_index = int(correction_cfg["parent_head_index"])
    child_head_index = int(correction_cfg["child_head_index"])
    hotspot_child_indices = dict(correction_cfg["hotspot_child_indices"])
    parent_targets = targets[parent_head_index]
    child_targets = targets[child_head_index]
    child_logits = corrected_logits[child_head_index]
    total = torch.zeros((), device=device)
    total_items = 0
    for parent_idx_str, child_indices in hotspot_child_indices.items():
        parent_idx = int(parent_idx_str)
        mask = parent_targets == parent_idx
        if not torch.any(mask):
            continue
        subset_indices = torch.tensor(child_indices, dtype=torch.long, device=device)
        subset_logits = child_logits[mask].index_select(1, subset_indices)
        target_subset = child_targets[mask]
        remapped = torch.full_like(target_subset, fill_value=-100)
        for local_idx, global_idx in enumerate(child_indices):
            remapped[target_subset == int(global_idx)] = int(local_idx)
        valid = remapped >= 0
        if not torch.any(valid):
            continue
        total = total + F.cross_entropy(subset_logits[valid], remapped[valid], reduction="sum")
        total_items += int(valid.sum().item())
    if total_items == 0:
        return torch.zeros((), device=device), 0
    return total / float(total_items), total_items


def _compute_parent_conditioned_ranking_loss(
    corrected_logits: List[torch.Tensor],
    targets: List[torch.Tensor],
    correction_cfg: Dict[str, Any],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    parent_head_index = int(correction_cfg["parent_head_index"])
    child_head_index = int(correction_cfg["child_head_index"])
    hotspot_child_indices = dict(correction_cfg["hotspot_child_indices"])
    parent_targets = targets[parent_head_index]
    child_targets = targets[child_head_index]
    child_logits = corrected_logits[child_head_index]
    margin = float(correction_cfg.get("rank_margin", 0.2))
    total = torch.zeros((), device=device)
    total_items = 0
    for parent_idx_str, child_indices in hotspot_child_indices.items():
        parent_idx = int(parent_idx_str)
        mask = parent_targets == parent_idx
        if not torch.any(mask) or len(child_indices) < 2:
            continue
        subset_indices = torch.tensor(child_indices, dtype=torch.long, device=device)
        subset_logits = child_logits[mask].index_select(1, subset_indices)
        target_subset = child_targets[mask]
        remapped = torch.full_like(target_subset, fill_value=-100)
        for local_idx, global_idx in enumerate(child_indices):
            remapped[target_subset == int(global_idx)] = int(local_idx)
        valid = remapped >= 0
        if not torch.any(valid):
            continue
        valid_logits = subset_logits[valid]
        valid_targets = remapped[valid]
        row_idx = torch.arange(valid_logits.shape[0], device=device)
        true_logits = valid_logits[row_idx, valid_targets]
        competitor_logits = valid_logits.clone()
        competitor_logits[row_idx, valid_targets] = float("-inf")
        top_other = competitor_logits.max(dim=1).values
        losses = torch.relu(torch.tensor(margin, device=device) - (true_logits - top_other))
        total = total + losses.sum()
        total_items += int(valid.sum().item())
    if total_items == 0:
        return torch.zeros((), device=device), 0
    return total / float(total_items), total_items


def _build_resource_summary(
    adata: AnnData,
    device: torch.device,
    *,
    train_size: int,
    val_size: int,
    num_epochs: int,
    num_coord_heads: int,
    num_threads_used: int,
    device_requested: str,
    runtime_summary: Dict[str, object],
) -> Dict[str, object]:
    gpu_name = None
    gpu_total_memory_gb = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu_name = props.name
        gpu_total_memory_gb = round(props.total_memory / (1024 ** 3), 2)

    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "num_epochs_completed": int(num_epochs),
        "num_coord_heads": int(num_coord_heads),
        "device_requested": device_requested,
        "device_type": device.type,
        "gpu_name": gpu_name,
        "gpu_total_memory_gb": gpu_total_memory_gb,
        "cpu_count": os.cpu_count(),
        "num_threads_used": int(num_threads_used),
        "runtime_summary": runtime_summary,
    }


def build_model(
    adata: AnnData,
    label_columns: List[str],
    coord_targets: Optional[Dict[str, str]] = None,
    task_weights: Optional[List[float]] = None,
    coord_loss_weights: Optional[Dict[str, float]] = None,
    latent_source: str = "internal_preferred",
    knn_reference_obsm_key: Optional[str] = None,
    knn_space: Optional[str] = None,
    hidden_sizes: Optional[List[int]] = None,
    dropout_rate: float = 0.3,
    batch_size: int = 256,
    num_epochs: int = 40,
    learning_rate: float = 1e-3,
    optimizer_name: str = "adamw",
    weight_decay: float = 5e-5,
    scheduler_name: Optional[str] = None,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 5,
    scheduler_min_lr: float = 1e-6,
    scheduler_monitor: str = "val_loss",
    input_transform: str = "binary",
    val_fraction: float = 0.0,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    random_state: int = 42,
    preset: Optional[str] = None,
    domain_key: Optional[str] = None,
    domain_loss_weight: float = 0.0,
    domain_loss_method: str = "mean",
    topology_loss_weight: float = 0.0,
    topology_k: int = 10,
    topology_coord: str = "latent",
    calibration_method: Optional[str] = None,
    calibration_max_iter: int = 100,
    calibration_lr: float = 0.05,
    class_weighting: Optional[Dict[str, Any]] = None,
    class_balanced_sampling: Optional[Dict[str, Any]] = None,
    parent_conditioned_child_correction: Optional[Dict[str, Any]] = None,
    reference_storage: str = "external",
    reference_path: Optional[str] = None,
    init_model_path: Optional[str] = None,
    num_threads: Union[int, str, None] = 10,
    device: str = "auto",
    show_progress: Optional[bool] = None,
    show_summary: Optional[bool] = None,
) -> TrainedModel:
    """Train an `atlasmtl` model from a reference `AnnData`.

    Parameters
    ----------
    adata
        Reference dataset. `adata.X` provides the input expression matrix,
        `adata.obs[label_columns]` provides hierarchical labels, and
        `adata.obsm[coord_targets[*]]` provides coordinate regression targets.
    label_columns
        Ordered annotation columns to predict. Each column becomes one
        classification head.
    coord_targets
        Mapping from internal coordinate head name to `adata.obsm` key.
        When `None`, atlasmtl trains in no-coordinate mode. Pass an explicit
        dict such as `{"latent": "X_ref_latent", "umap": "X_umap"}` to enable
        coordinate heads.
    task_weights
        Per-label loss weights for the classification heads. Must match
        `label_columns` length. Defaults to uniform weights.
    coord_loss_weights
        Per-coordinate regression loss weights. Defaults to
        `{"latent": 0.5, "umap": 0.2}`.
    latent_source
        Metadata flag describing how latent coordinates should be interpreted
        downstream. Currently stored on the model for later prediction metadata.
    knn_reference_obsm_key
        Optional reference-space override stored for KNN correction only (no
        regression head is trained). Example: `"X_scANVI"`. When provided,
        atlasmtl stores the array in `reference_data.coords` for later lookup.
    knn_space
        Optional explicit KNN space name used for storage and later lookup,
        stored as `reference_data.coords["X_ref_{knn_space}"]`. If not provided,
        it is derived from `knn_reference_obsm_key` by stripping the `"X_"`
        prefix and lowercasing.
    hidden_sizes
        Hidden layer widths for the shared encoder. Defaults to `[256, 128]`
        when not provided.
    dropout_rate
        Dropout applied in the shared encoder.
    batch_size
        Mini-batch size used for training and optional validation.
    num_epochs
        Maximum number of training epochs.
    learning_rate
        Adam learning rate.
    optimizer_name
        Optimizer to use. Supported values are `"adam"` and `"adamw"`.
    weight_decay
        Weight decay passed to the selected optimizer.
    scheduler_name
        Optional learning-rate scheduler. Supported values are `None` and
        `"reduce_lr_on_plateau"`.
    scheduler_factor
        Multiplicative factor for `ReduceLROnPlateau`.
    scheduler_patience
        Plateau patience in epochs for `ReduceLROnPlateau`.
    scheduler_min_lr
        Minimum learning rate for `ReduceLROnPlateau`.
    scheduler_monitor
        Quantity monitored by the scheduler. Only `"val_loss"` is supported.
    input_transform
        Input preprocessing applied to `adata.X`. Supported values are
        `"binary"` and `"float"`. `"binary"` is the default and recommended
        mode for phmap-consistent behavior.
    val_fraction
        Fraction of cells held out for validation. Set to `0.0` to disable
        the validation split.
    early_stopping_patience
        Number of validation epochs without improvement allowed before stopping.
        Disabled when `None`.
    early_stopping_min_delta
        Minimum validation-loss improvement required to reset early stopping.
    random_state
        Random seed used when creating the optional validation split.
    class_weighting
        Optional per-head class weighting configuration. Current supported form
        is `{"label_column": "<target>", "mode": "balanced"}`.
    class_balanced_sampling
        Optional weighted sampling configuration. Current supported form is
        `{"label_column": "<target>", "mode": "balanced"}`.
    parent_conditioned_child_correction
        Optional local train-time child correction configuration. Current
        supported form is `{"parent_level": "<parent>", "target_level":
        "<child>", "hotspot_parents": [...], "mode": "joint" |
        "frozen_base"}`.
    reference_storage
        How KNN reference data is stored with the trained model. Supported
        values are `"external"` and `"full"`. `"external"` is recommended.
    reference_path
        Optional custom path for external reference storage. Ignored when
        `reference_storage="full"`.
    init_model_path
        Optional path to an existing trained model artifact or manifest used to
        initialize matching weights before training.
    num_threads
        Number of CPU threads made available to PyTorch. Default is `10`.
        Pass `"max"` to use up to 80% of available CPUs.
    device
        Execution device: `"auto"`, `"cpu"`, or `"cuda"`. `"auto"` uses CUDA
        when available and otherwise falls back to CPU.
    show_progress
        Whether to display a training progress bar with ETA. Defaults to
        auto-detection based on whether stderr is attached to a terminal.
    show_summary
        Whether to print a compact post-training resource summary. Defaults to
        auto-detection based on whether stdout is attached to a terminal.

    Returns
    -------
    TrainedModel
        Trained model bundle with weights, label encoders, coordinate scaling
        statistics, reference data, and training metadata including resource
        summary and elapsed training time.
    """
    coord_targets = dict(coord_targets or {})
    coord_loss_weights = coord_loss_weights or {"latent": 0.5, "umap": 0.2}
    if preset is not None:
        cfg = resolve_preset(preset)
        if hidden_sizes is None and "hidden_sizes" in cfg:
            hidden_sizes = cfg["hidden_sizes"]
        if dropout_rate == 0.3 and "dropout_rate" in cfg:
            dropout_rate = float(cfg["dropout_rate"])
        if learning_rate == 1e-3 and "learning_rate" in cfg:
            learning_rate = float(cfg["learning_rate"])

    for col in label_columns:
        if col not in adata.obs.columns:
            raise ValueError(f"Missing label column: {col}")
    for name, key in coord_targets.items():
        if key not in adata.obsm:
            raise ValueError(f"Missing coordinate target in adata.obsm: {key} for {name}")
    if knn_reference_obsm_key is not None and knn_reference_obsm_key not in adata.obsm:
        raise ValueError(f"Missing knn_reference_obsm_key in adata.obsm: {knn_reference_obsm_key}")
    if domain_key is not None and domain_key not in adata.obs.columns:
        raise ValueError(f"Missing domain_key column in adata.obs: {domain_key}")
    if domain_loss_method != "mean":
        raise ValueError("domain_loss_method must be 'mean'")
    optimizer_name = str(optimizer_name).lower()
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("optimizer_name must be 'adam' or 'adamw'")
    if weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    scheduler_name = None if scheduler_name is None else str(scheduler_name).lower()
    if scheduler_name not in {None, "reduce_lr_on_plateau"}:
        raise ValueError("scheduler_name must be None or 'reduce_lr_on_plateau'")
    if scheduler_monitor != "val_loss":
        raise ValueError("scheduler_monitor must be 'val_loss'")
    if domain_loss_weight < 0:
        raise ValueError("domain_loss_weight must be >= 0")
    if topology_loss_weight < 0:
        raise ValueError("topology_loss_weight must be >= 0")
    if topology_k < 0:
        raise ValueError("topology_k must be >= 0")

    preprocess_metadata = dict(adata.uns.get("atlasmtl_preprocess", {})) if "atlasmtl_preprocess" in adata.uns else None
    X = extract_matrix(adata, input_transform=input_transform)
    y_arrays: List[np.ndarray] = []
    label_encoders: Dict[str, LabelEncoder] = {}
    num_classes: List[int] = []
    for col in label_columns:
        le = LabelEncoder()
        y = le.fit_transform(adata.obs[col].astype(str).values)
        label_encoders[col] = le
        y_arrays.append(y)
        num_classes.append(len(le.classes_))
    parent_conditioned_child_correction_cfg = _resolve_parent_conditioned_child_correction_config(
        parent_conditioned_child_correction,
        adata=adata,
        label_columns=label_columns,
        label_encoders=label_encoders,
    )

    coord_stats: Dict[str, Dict[str, np.ndarray]] = {}
    coord_data: Dict[str, np.ndarray] = {}
    coord_dims: Dict[str, int] = {}
    for name, key in coord_targets.items():
        raw = np.asarray(adata.obsm[key], dtype=np.float32)
        stats = compute_coord_stats(raw)
        coord_stats[name] = stats
        coord_data[name] = scale_coords(raw, stats)
        coord_dims[name] = raw.shape[1]

    model = AtlasMTLModel(
        input_size=X.shape[1],
        num_classes=num_classes,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        coord_dims=coord_dims,
        parent_conditioned_child_correction=parent_conditioned_child_correction_cfg,
    )
    num_threads_used = configure_torch_threads(num_threads)
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    if init_model_path is not None:
        init_bundle = TrainedModel.load(str(init_model_path), device=resolved_device)
        current_state = model.state_dict()
        init_state = init_bundle.model.state_dict()
        compatible = {
            key: value
            for key, value in init_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)

    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = [torch.tensor(y, dtype=torch.long) for y in y_arrays]
    c_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in coord_data.items()}
    d_t = None
    domain_encoder = None
    if domain_key is not None:
        domain_encoder = LabelEncoder()
        domains = domain_encoder.fit_transform(adata.obs[domain_key].astype(str).values)
        d_t = torch.tensor(domains, dtype=torch.long)

    task_weights = task_weights or [1.0 for _ in label_columns]
    if len(task_weights) != len(label_columns):
        raise ValueError("task_weights length must match label_columns")
    class_weighting_spec = _resolve_balanced_label_config(
        class_weighting,
        label_columns=label_columns,
        config_name="class_weighting",
    )
    class_sampling_spec = _resolve_balanced_label_config(
        class_balanced_sampling,
        label_columns=label_columns,
        config_name="class_balanced_sampling",
    )

    huber = torch.nn.HuberLoss()
    optimizer_cls = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    if parent_conditioned_child_correction_cfg is not None:
        correction_names = {
            name
            for name, _ in model.named_parameters()
            if name.startswith("child_correction_modules.")
        }
        if parent_conditioned_child_correction_cfg["mode"] == "frozen_base":
            for name, param in model.named_parameters():
                param.requires_grad = name in correction_names
            opt = optimizer_cls(
                [param for param in model.parameters() if param.requires_grad],
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            base_params = []
            correction_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name in correction_names:
                    correction_params.append(param)
                else:
                    base_params.append(param)
            opt = optimizer_cls(
                [
                    {
                        "params": base_params,
                        "lr": learning_rate * float(parent_conditioned_child_correction_cfg["base_lr_scale"]),
                    },
                    {"params": correction_params, "lr": learning_rate},
                ],
                weight_decay=weight_decay,
            )
    else:
        opt = optimizer_cls(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    coord_names = list(coord_targets.keys())
    train_indices = np.arange(adata.n_obs)
    val_indices = np.array([], dtype=np.int64)
    if val_fraction > 0:
        train_indices, val_indices = train_test_split(
            np.arange(adata.n_obs),
            test_size=val_fraction,
            random_state=random_state,
            shuffle=True,
        )

    ce_losses: List[torch.nn.CrossEntropyLoss] = []
    class_weighting_summary = None
    for head_index, column in enumerate(label_columns):
        weight_tensor = None
        if class_weighting_spec is not None and head_index == class_weighting_spec[0]:
            class_weights = _balanced_class_weights(
                y_arrays[head_index][train_indices],
                num_classes=num_classes[head_index],
            )
            weight_tensor = torch.tensor(
                class_weights,
                dtype=torch.float32,
                device=resolved_device,
            )
            class_counts = np.bincount(
                y_arrays[head_index][train_indices],
                minlength=num_classes[head_index],
            )
            class_weighting_summary = {
                "label_column": column,
                "mode": class_weighting_spec[2],
                **_class_distribution_metadata(
                    classes=label_encoders[column].classes_,
                    counts=class_counts,
                    weights=class_weights,
                ),
            }
        ce_losses.append(torch.nn.CrossEntropyLoss(weight=weight_tensor))

    train_tensors = [
        x_t[train_indices],
        *[tensor[train_indices] for tensor in y_t],
        *[c_t[name][train_indices] for name in coord_names],
    ]
    if d_t is not None:
        train_tensors.append(d_t[train_indices])
    train_dataset = TensorDataset(*train_tensors)
    train_sampler = None
    class_balanced_sampling_summary = None
    if class_sampling_spec is not None:
        sample_head_index, sample_label_column, sample_mode = class_sampling_spec
        sample_targets = y_arrays[sample_head_index][train_indices]
        sample_class_weights = _balanced_class_weights(
            sample_targets,
            num_classes=num_classes[sample_head_index],
        )
        sample_weights = sample_class_weights[sample_targets]
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_indices),
            replacement=True,
            generator=torch.Generator().manual_seed(int(random_state)),
        )
        sample_counts = np.bincount(
            sample_targets,
            minlength=num_classes[sample_head_index],
        )
        class_balanced_sampling_summary = {
            "label_column": sample_label_column,
            "mode": sample_mode,
            "replacement": True,
            **_class_distribution_metadata(
                classes=label_encoders[sample_label_column].classes_,
                counts=sample_counts,
                weights=sample_class_weights,
            ),
        }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )

    val_loader = None
    if len(val_indices) > 0:
        val_tensors = [
            x_t[val_indices],
            *[tensor[val_indices] for tensor in y_t],
            *[c_t[name][val_indices] for name in coord_names],
        ]
        if d_t is not None:
            val_tensors.append(d_t[val_indices])
        val_dataset = TensorDataset(*val_tensors)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if scheduler_name == "reduce_lr_on_plateau" and val_loader is None:
        raise ValueError("scheduler_name='reduce_lr_on_plateau' requires val_fraction > 0")

    scheduler = None
    if scheduler_name == "reduce_lr_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(scheduler_factor),
            patience=int(scheduler_patience),
            min_lr=float(scheduler_min_lr),
        )

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    epochs_completed = 0
    last_train_loss = None
    last_val_loss = None
    runtime_monitor = RuntimeMonitor(phase="train", device=resolved_device)
    epoch_iterator = progress_iter(
        range(num_epochs),
        total=num_epochs,
        desc="atlasmtl train",
        show_progress=show_progress,
    )
    for _ in epoch_iterator:
        epochs_completed += 1
        model.train()
        epoch_loss_total = 0.0
        epoch_items = 0
        for batch in train_loader:
            bx = batch[0].to(resolved_device)
            by = [batch[i + 1].to(resolved_device) for i in range(len(label_columns))]
            bc = {
                coord_names[i]: batch[1 + len(label_columns) + i].to(resolved_device)
                for i in range(len(coord_names))
            }
            domains_batch = None
            if d_t is not None:
                domains_batch = batch[1 + len(label_columns) + len(coord_names)].to(resolved_device)
            logits, coords, z = model(bx)
            loss_cls = sum(task_weights[i] * ce_losses[i](logits[i], by[i]) for i in range(len(by)))
            loss_corr = torch.zeros((), device=resolved_device)
            loss_rank = torch.zeros((), device=resolved_device)
            if parent_conditioned_child_correction_cfg is not None:
                corrected_logits, _ = model.apply_parent_conditioned_child_correction(
                    z,
                    logits,
                    parent_indices_override=by[parent_conditioned_child_correction_cfg["parent_head_index"]],
                )
                loss_corr, _ = _compute_parent_conditioned_correction_loss(
                    corrected_logits,
                    by,
                    parent_conditioned_child_correction_cfg,
                    device=resolved_device,
                )
                loss_corr = loss_corr * float(parent_conditioned_child_correction_cfg["loss_weight"])
                if float(parent_conditioned_child_correction_cfg.get("rank_loss_weight", 0.0)) > 0:
                    loss_rank, _ = _compute_parent_conditioned_ranking_loss(
                        corrected_logits,
                        by,
                        parent_conditioned_child_correction_cfg,
                        device=resolved_device,
                    )
                    loss_rank = loss_rank * float(parent_conditioned_child_correction_cfg["rank_loss_weight"])
            loss_coord = sum(
                coord_loss_weights.get(name, 0.0) * huber(coords[name], bc[name])
                for name in coord_names
            )
            loss_domain = torch.zeros((), device=resolved_device)
            if domains_batch is not None and domain_loss_weight > 0:
                loss_domain = _domain_mean_penalty(z, domains_batch) * float(domain_loss_weight)
            loss_topo = torch.zeros((), device=resolved_device)
            if topology_loss_weight > 0 and topology_coord in coord_names:
                loss_topo = _topology_loss(
                    coords[topology_coord],
                    bc[topology_coord],
                    k=int(topology_k),
                ) * float(topology_loss_weight)
            loss = loss_cls + loss_corr + loss_rank + loss_coord + loss_domain + loss_topo
            if not loss.requires_grad:
                epoch_loss_total += loss.item() * len(bx)
                epoch_items += len(bx)
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss_total += loss.item() * len(bx)
            epoch_items += len(bx)

        last_train_loss = epoch_loss_total / max(epoch_items, 1)

        if val_loader is None or early_stopping_patience is None:
            if hasattr(epoch_iterator, "set_postfix"):
                epoch_iterator.set_postfix(train_loss=f"{last_train_loss:.4f}")
            continue

        model.eval()
        total_val_loss = 0.0
        total_val_items = 0
        with torch.no_grad():
            for batch in val_loader:
                bx = batch[0].to(resolved_device)
                by = [batch[i + 1].to(resolved_device) for i in range(len(label_columns))]
                bc = {
                    coord_names[i]: batch[1 + len(label_columns) + i].to(resolved_device)
                    for i in range(len(coord_names))
                }
                logits, coords, z = model(bx)
                loss_cls = sum(task_weights[i] * ce_losses[i](logits[i], by[i]) for i in range(len(by)))
                loss_coord = sum(
                    coord_loss_weights.get(name, 0.0) * huber(coords[name], bc[name])
                    for name in coord_names
                )
                loss_domain = torch.zeros((), device=resolved_device)
                if d_t is not None and domain_loss_weight > 0:
                    domains_batch = batch[1 + len(label_columns) + len(coord_names)].to(resolved_device)
                    loss_domain = _domain_mean_penalty(z, domains_batch) * float(domain_loss_weight)
                loss_corr = torch.zeros((), device=resolved_device)
                loss_rank = torch.zeros((), device=resolved_device)
                if parent_conditioned_child_correction_cfg is not None:
                    corrected_logits, _ = model.apply_parent_conditioned_child_correction(
                        z,
                        logits,
                        parent_indices_override=by[parent_conditioned_child_correction_cfg["parent_head_index"]],
                    )
                    loss_corr, _ = _compute_parent_conditioned_correction_loss(
                        corrected_logits,
                        by,
                        parent_conditioned_child_correction_cfg,
                        device=resolved_device,
                    )
                    loss_corr = loss_corr * float(parent_conditioned_child_correction_cfg["loss_weight"])
                    if float(parent_conditioned_child_correction_cfg.get("rank_loss_weight", 0.0)) > 0:
                        loss_rank, _ = _compute_parent_conditioned_ranking_loss(
                            corrected_logits,
                            by,
                            parent_conditioned_child_correction_cfg,
                            device=resolved_device,
                        )
                        loss_rank = loss_rank * float(parent_conditioned_child_correction_cfg["rank_loss_weight"])
                loss_topo = torch.zeros((), device=resolved_device)
                if topology_loss_weight > 0 and topology_coord in coord_names:
                    loss_topo = _topology_loss(
                        coords[topology_coord],
                        bc[topology_coord],
                        k=int(topology_k),
                    ) * float(topology_loss_weight)
                batch_loss = (loss_cls + loss_corr + loss_rank + loss_coord + loss_domain + loss_topo).item()
                total_val_loss += batch_loss * len(bx)
                total_val_items += len(bx)

        mean_val_loss = total_val_loss / max(total_val_items, 1)
        last_val_loss = mean_val_loss
        if scheduler is not None:
            scheduler.step(mean_val_loss)
        if hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(train_loss=f"{last_train_loss:.4f}", val_loss=f"{mean_val_loss:.4f}")
        if mean_val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = mean_val_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break
    train_runtime = runtime_monitor.finish(num_items=len(train_indices) * max(epochs_completed, 1), num_batches=len(train_loader) * max(epochs_completed, 1))
    train_seconds = train_runtime["elapsed_seconds"]

    if best_state is not None:
        model.load_state_dict(best_state)

    calibration_payload: Optional[Dict[str, object]] = None
    if calibration_method is not None:
        if calibration_method != "temperature_scaling":
            raise ValueError("calibration_method must be None or 'temperature_scaling'")
        if val_loader is None:
            raise ValueError("temperature scaling requires val_fraction > 0 to fit calibration temperatures")
        model.eval()
        val_logits_parts: Optional[List[List[torch.Tensor]]] = None
        val_targets_parts: List[List[torch.Tensor]] = [[] for _ in label_columns]
        with torch.no_grad():
            for batch in val_loader:
                bx = batch[0].to(resolved_device)
                by = [batch[i + 1].to(resolved_device) for i in range(len(label_columns))]
                logits_out, _, z_out = model(bx)
                if parent_conditioned_child_correction_cfg is not None:
                    logits_out, _ = model.apply_parent_conditioned_child_correction(
                        z_out,
                        logits_out,
                        parent_indices_override=by[parent_conditioned_child_correction_cfg["parent_head_index"]],
                    )
                if val_logits_parts is None:
                    val_logits_parts = [[] for _ in range(len(logits_out))]
                for i, logit in enumerate(logits_out):
                    val_logits_parts[i].append(logit.detach())
                    val_targets_parts[i].append(by[i].detach())

        if val_logits_parts is None:
            raise ValueError("no validation batches available for calibration")
        temperatures: Dict[str, float] = {}
        for i, col in enumerate(label_columns):
            head_logits = torch.cat(val_logits_parts[i], dim=0)
            head_targets = torch.cat(val_targets_parts[i], dim=0)
            calibrator = fit_temperature_scaling(
                head_logits,
                head_targets,
                max_iter=calibration_max_iter,
                lr=calibration_lr,
                device=resolved_device,
            )
            temperatures[col] = calibrator.temperature
        calibration_payload = {
            "method": calibration_method,
            "split": "val",
            "temperatures": temperatures,
            "max_iter": int(calibration_max_iter),
            "lr": float(calibration_lr),
            "val_fraction": float(val_fraction),
            "random_state": int(random_state),
        }

    ref_internal_latent = _encode_internal_latent(
        model,
        x_t,
        batch_size=batch_size,
        device=resolved_device,
        show_progress=show_progress,
    )
    ref_coords = {
        f"X_ref_{k}": np.asarray(adata.obsm[v], dtype=np.float32) for k, v in coord_targets.items()
    }
    if knn_reference_obsm_key is not None:
        space_name = str(knn_space or knn_reference_obsm_key).removeprefix("X_").replace("-", "_").replace(" ", "_").lower()
        ref_coords[f"X_ref_{space_name}"] = np.asarray(adata.obsm[knn_reference_obsm_key], dtype=np.float32)
    ref_coords["X_ref_latent_internal"] = ref_internal_latent
    ref_labels = {col: adata.obs[col].astype(str).values.copy() for col in label_columns}
    resource_summary = _build_resource_summary(
        adata,
        resolved_device,
        train_size=len(train_indices),
        val_size=len(val_indices),
        num_epochs=epochs_completed,
        num_coord_heads=len(coord_names),
        num_threads_used=num_threads_used,
        device_requested=device,
        runtime_summary=train_runtime,
    )
    train_config = {
        "preset": preset,
        "hidden_sizes": hidden_sizes or [256, 128],
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "epochs_completed": epochs_completed,
        "learning_rate": learning_rate,
        "optimizer_name": optimizer_name,
        "weight_decay": float(weight_decay),
        "scheduler_name": scheduler_name,
        "scheduler_factor": float(scheduler_factor),
        "scheduler_patience": int(scheduler_patience),
        "scheduler_min_lr": float(scheduler_min_lr),
        "scheduler_monitor": scheduler_monitor,
        "final_learning_rate": float(opt.param_groups[0]["lr"]),
        "coord_targets": coord_targets,
        "coord_loss_weights": coord_loss_weights,
        "task_weights": task_weights,
        "input_transform": input_transform,
        "val_fraction": val_fraction,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "random_state": random_state,
        "domain_key": domain_key,
        "domain_loss_weight": float(domain_loss_weight),
        "domain_loss_method": domain_loss_method,
        "domains": None if domain_encoder is None else [str(x) for x in domain_encoder.classes_],
        "topology_loss_weight": float(topology_loss_weight),
        "topology_k": int(topology_k),
        "topology_coord": topology_coord,
        "calibration": calibration_payload,
        "class_weighting": class_weighting_summary,
        "class_balanced_sampling": class_balanced_sampling_summary,
        "parent_conditioned_child_correction": parent_conditioned_child_correction_cfg,
        "reference_storage": reference_storage,
        "init_model_path": init_model_path,
        "coord_enabled": bool(coord_targets),
        "resolved_coord_targets": coord_targets,
        "knn_reference_obsm_key": knn_reference_obsm_key,
        "knn_space": knn_space,
        "num_threads_requested": num_threads,
        "num_threads_used": num_threads_used,
        "device_requested": device,
        "device_used": resolved_device.type,
        "train_seconds": train_seconds,
        "runtime_summary": train_runtime,
        "last_train_loss": None if last_train_loss is None else round(last_train_loss, 6),
        "last_val_loss": None if last_val_loss is None else round(last_val_loss, 6),
        "resource_summary": resource_summary,
    }
    if preprocess_metadata is not None:
        train_config["preprocess"] = preprocess_metadata

    trained_model = TrainedModel(
        model=model,
        label_columns=label_columns,
        label_encoders=label_encoders,
        train_genes=list(adata.var_names),
        coord_targets=coord_targets,
        coord_stats=coord_stats,
        reference_data=ReferenceData(coords=ref_coords, labels=ref_labels),
        latent_source=latent_source,
        input_transform=input_transform,
        reference_storage=reference_storage,
        reference_path=reference_path,
        train_config=train_config,
    )
    if resolve_show_summary(show_summary):
        trained_model.show_resource_usage()
    return trained_model
