from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from anndata import AnnData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from .data import compute_coord_stats, extract_matrix, scale_coords
from .model import AtlasMTLModel
from .runtime import configure_torch_threads, resolve_device
from ..models import ReferenceData
from .types import TrainedModel
from ..utils import RuntimeMonitor, progress_iter, resolve_show_summary


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
    hidden_sizes: Optional[List[int]] = None,
    dropout_rate: float = 0.3,
    batch_size: int = 256,
    num_epochs: int = 40,
    learning_rate: float = 1e-3,
    input_transform: str = "binary",
    val_fraction: float = 0.0,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    random_state: int = 42,
    reference_storage: str = "external",
    reference_path: Optional[str] = None,
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
    reference_storage
        How KNN reference data is stored with the trained model. Supported
        values are `"external"` and `"full"`. `"external"` is recommended.
    reference_path
        Optional custom path for external reference storage. Ignored when
        `reference_storage="full"`.
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

    for col in label_columns:
        if col not in adata.obs.columns:
            raise ValueError(f"Missing label column: {col}")
    for name, key in coord_targets.items():
        if key not in adata.obsm:
            raise ValueError(f"Missing coordinate target in adata.obsm: {key} for {name}")

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
    )
    num_threads_used = configure_torch_threads(num_threads)
    resolved_device = resolve_device(device)
    model.to(resolved_device)

    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = [torch.tensor(y, dtype=torch.long) for y in y_arrays]
    c_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in coord_data.items()}

    task_weights = task_weights or [1.0 for _ in label_columns]
    if len(task_weights) != len(label_columns):
        raise ValueError("task_weights length must match label_columns")

    ce = torch.nn.CrossEntropyLoss()
    huber = torch.nn.HuberLoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    train_dataset = TensorDataset(
        x_t[train_indices],
        *[tensor[train_indices] for tensor in y_t],
        *[c_t[name][train_indices] for name in coord_names],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if len(val_indices) > 0:
        val_dataset = TensorDataset(
            x_t[val_indices],
            *[tensor[val_indices] for tensor in y_t],
            *[c_t[name][val_indices] for name in coord_names],
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
            logits, coords, _ = model(bx)
            loss_cls = sum(task_weights[i] * ce(logits[i], by[i]) for i in range(len(by)))
            loss_coord = sum(
                coord_loss_weights.get(name, 0.0) * huber(coords[name], bc[name])
                for name in coord_names
            )
            loss = loss_cls + loss_coord
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
                logits, coords, _ = model(bx)
                loss_cls = sum(task_weights[i] * ce(logits[i], by[i]) for i in range(len(by)))
                loss_coord = sum(
                    coord_loss_weights.get(name, 0.0) * huber(coords[name], bc[name])
                    for name in coord_names
                )
                batch_loss = (loss_cls + loss_coord).item()
                total_val_loss += batch_loss * len(bx)
                total_val_items += len(bx)

        mean_val_loss = total_val_loss / max(total_val_items, 1)
        last_val_loss = mean_val_loss
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

    ref_coords = {
        f"X_ref_{k}": np.asarray(adata.obsm[v], dtype=np.float32) for k, v in coord_targets.items()
    }
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
        "hidden_sizes": hidden_sizes or [256, 128],
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "epochs_completed": epochs_completed,
        "learning_rate": learning_rate,
        "coord_targets": coord_targets,
        "coord_loss_weights": coord_loss_weights,
        "task_weights": task_weights,
        "input_transform": input_transform,
        "val_fraction": val_fraction,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "random_state": random_state,
        "reference_storage": reference_storage,
        "coord_enabled": bool(coord_targets),
        "resolved_coord_targets": coord_targets,
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
