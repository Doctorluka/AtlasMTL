from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from .model import AtlasMTLModel


@dataclass
class TrainedModel:
    model: AtlasMTLModel
    label_columns: List[str]
    label_encoders: Dict[str, LabelEncoder]
    train_genes: List[str]
    coord_targets: Dict[str, str]
    coord_stats: Dict[str, Dict[str, np.ndarray]]
    reference_coords: Dict[str, np.ndarray]
    reference_labels: Dict[str, np.ndarray]
    latent_source: str = "internal_preferred"

    def save(self, path: str) -> None:
        self.model.save(path)
        meta_path = path.replace(".pth", "_metadata.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "label_columns": self.label_columns,
                    "label_encoders": self.label_encoders,
                    "train_genes": self.train_genes,
                    "coord_targets": self.coord_targets,
                    "coord_stats": self.coord_stats,
                    "reference_coords": self.reference_coords,
                    "reference_labels": self.reference_labels,
                    "latent_source": self.latent_source,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "TrainedModel":
        model = AtlasMTLModel.load(path, device=device)
        meta_path = path.replace(".pth", "_metadata.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return cls(model=model, **meta)


class PredictionResult:
    def __init__(self, predictions: pd.DataFrame, coordinates: Dict[str, np.ndarray], metadata: Dict):
        self.predictions = predictions
        self.coordinates = coordinates
        self.metadata = metadata

    def to_adata(self, adata: sc.AnnData) -> sc.AnnData:
        adata.obs = adata.obs.join(self.predictions)
        for key, value in self.coordinates.items():
            adata.obsm[key] = value
        adata.uns.setdefault("atlasmtl", {}).update(self.metadata)
        return adata


def _extract_matrix(adata: sc.AnnData, train_genes: Optional[List[str]] = None) -> np.ndarray:
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if train_genes is None:
        return X
    gene_index = {g: i for i, g in enumerate(adata.var_names)}
    out = np.zeros((adata.n_obs, len(train_genes)), dtype=np.float32)
    for j, gene in enumerate(train_genes):
        idx = gene_index.get(gene)
        if idx is not None:
            out[:, j] = X[:, idx]
    return out


def _compute_coord_stats(arr: np.ndarray) -> Dict[str, np.ndarray]:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std == 0] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _scale_coords(arr: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return ((arr - stats["mean"]) / stats["std"]).astype(np.float32)


def _unscale_coords(arr: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (arr * stats["std"] + stats["mean"]).astype(np.float32)


def build_model(
    adata: sc.AnnData,
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
) -> TrainedModel:
    coord_targets = coord_targets or {"latent": "X_ref_latent", "umap": "X_umap"}
    coord_loss_weights = coord_loss_weights or {"latent": 0.5, "umap": 0.2}

    for col in label_columns:
        if col not in adata.obs.columns:
            raise ValueError(f"Missing label column: {col}")
    for name, key in coord_targets.items():
        if key not in adata.obsm:
            raise ValueError(f"Missing coordinate target in adata.obsm: {key} for {name}")

    X = _extract_matrix(adata)
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
        stats = _compute_coord_stats(raw)
        coord_stats[name] = stats
        coord_data[name] = _scale_coords(raw, stats)
        coord_dims[name] = raw.shape[1]

    model = AtlasMTLModel(
        input_size=X.shape[1],
        num_classes=num_classes,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        coord_dims=coord_dims,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = [torch.tensor(y, dtype=torch.long) for y in y_arrays]
    c_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in coord_data.items()}
    dataset = TensorDataset(x_t, *y_t, *[c_t[k] for k in coord_targets.keys()])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    task_weights = task_weights or [1.0 for _ in label_columns]
    if len(task_weights) != len(label_columns):
        raise ValueError("task_weights length must match label_columns")

    ce = torch.nn.CrossEntropyLoss()
    huber = torch.nn.HuberLoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    coord_names = list(coord_targets.keys())
    for _ in range(num_epochs):
        model.train()
        for batch in loader:
            bx = batch[0].to(device)
            by = [batch[i + 1].to(device) for i in range(len(label_columns))]
            bc = {
                coord_names[i]: batch[1 + len(label_columns) + i].to(device)
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

    model.eval()
    ref_coords = {f"X_ref_{k}": np.asarray(adata.obsm[v], dtype=np.float32) for k, v in coord_targets.items()}
    ref_labels = {col: adata.obs[col].astype(str).values.copy() for col in label_columns}

    return TrainedModel(
        model=model,
        label_columns=label_columns,
        label_encoders=label_encoders,
        train_genes=list(adata.var_names),
        coord_targets=coord_targets,
        coord_stats=coord_stats,
        reference_coords=ref_coords,
        reference_labels=ref_labels,
        latent_source=latent_source,
    )


def _softmax(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def _top1_margin(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    max_prob = top2[:, 1]
    margin = top2[:, 1] - top2[:, 0]
    return max_prob, margin


def _knn_predict_subset(
    ref_coords: np.ndarray,
    ref_labels: np.ndarray,
    query_coords: np.ndarray,
    k: int,
) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k, len(ref_coords)), metric="euclidean")
    nn.fit(ref_coords)
    idx = nn.kneighbors(query_coords, return_distance=False)
    out = []
    for row in idx:
        votes = pd.Series(ref_labels[row]).value_counts()
        out.append(votes.index[0])
    return np.asarray(out)


def predict(
    model: TrainedModel,
    adata: sc.AnnData,
    knn_correction: str = "low_conf_only",
    confidence_high: float = 0.7,
    confidence_low: float = 0.4,
    margin_threshold: float = 0.2,
    knn_k: int = 15,
) -> PredictionResult:
    X = _extract_matrix(adata, model.train_genes)
    x_t = torch.tensor(X, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    model.model.eval()

    with torch.no_grad():
        logits, coords_scaled, _ = model.model(x_t.to(device))

    pred_df = pd.DataFrame(index=adata.obs_names)

    pred_coords: Dict[str, np.ndarray] = {}
    for name, coord in coords_scaled.items():
        pred_coords[f"X_pred_{name}"] = _unscale_coords(
            coord.detach().cpu().numpy(), model.coord_stats[name]
        )

    for i, col in enumerate(model.label_columns):
        probs = _softmax(logits[i])
        max_prob, margin = _top1_margin(probs)
        raw_idx = probs.argmax(axis=1)
        raw_label = model.label_encoders[col].inverse_transform(raw_idx)

        pred_df[f"pred_{col}_raw"] = raw_label
        pred_df[f"conf_{col}"] = max_prob
        pred_df[f"margin_{col}"] = margin

        is_low = (max_prob < confidence_high) | (margin < margin_threshold)
        pred_df[f"is_low_conf_{col}"] = is_low

        final_label = raw_label.copy()
        if knn_correction in {"low_conf_only", "all"}:
            apply_mask = np.ones_like(is_low, dtype=bool) if knn_correction == "all" else is_low
            if apply_mask.any():
                query_space = pred_coords["X_pred_latent"] if "X_pred_latent" in pred_coords else pred_coords["X_pred_umap"]
                ref_space = model.reference_coords["X_ref_latent"] if "X_ref_latent" in model.reference_coords else model.reference_coords["X_ref_umap"]
                knn_labels = _knn_predict_subset(
                    ref_coords=ref_space,
                    ref_labels=model.reference_labels[col],
                    query_coords=query_space[apply_mask],
                    k=knn_k,
                )
                final_label[apply_mask] = knn_labels

        is_unknown = max_prob < confidence_low
        final_label = final_label.astype(object)
        final_label[is_unknown] = "Unknown"

        pred_df[f"is_unknown_{col}"] = is_unknown
        pred_df[f"pred_{col}_final"] = final_label

    metadata = {
        "knn_correction": knn_correction,
        "confidence_high": confidence_high,
        "confidence_low": confidence_low,
        "margin_threshold": margin_threshold,
        "knn_k": knn_k,
        "latent_source": model.latent_source,
    }

    return PredictionResult(predictions=pred_df, coordinates=pred_coords, metadata=metadata)
