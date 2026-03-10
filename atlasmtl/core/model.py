from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class AtlasMTLModel(nn.Module):
    """Shared encoder + multi-label classification heads + coordinate heads."""

    def __init__(
        self,
        input_size: int,
        num_classes: List[int],
        hidden_sizes: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        coord_dims: Optional[Dict[str, int]] = None,
        parent_conditioned_child_correction: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 128]
        if coord_dims is None:
            coord_dims = {"latent": 16, "umap": 2}

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.coord_dims = coord_dims
        self.parent_conditioned_child_correction = dict(parent_conditioned_child_correction or {})

        layers: List[nn.Module] = []
        prev = input_size
        for hs in hidden_sizes:
            layers.extend([nn.Linear(prev, hs), nn.ReLU(), nn.Dropout(dropout_rate)])
            prev = hs
        self.latent_dim = prev
        self.encoder = nn.Sequential(*layers)

        self.label_heads = nn.ModuleList([nn.Linear(prev, n) for n in num_classes])
        self.coord_heads = nn.ModuleDict({k: nn.Linear(prev, d) for k, d in coord_dims.items()})
        self.child_correction_modules = nn.ModuleDict()
        self._init_parent_conditioned_child_correction()

    def _correction_feature_dim(self, *, child_dim: int, parent_dim: int) -> int:
        cfg = self.parent_conditioned_child_correction
        feature_mode = str(cfg.get("feature_mode", "standard"))
        if feature_mode == "reranker_like":
            # restricted child logits + restricted child probs + top1/top2/margin + parent logits
            return (2 * child_dim) + 3 + parent_dim
        return self.latent_dim + parent_dim + child_dim

    def _build_correction_features(
        self,
        *,
        z: torch.Tensor,
        parent_logits: torch.Tensor,
        child_subset_logits: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.parent_conditioned_child_correction
        feature_mode = str(cfg.get("feature_mode", "standard"))
        if feature_mode == "reranker_like":
            child_probs = torch.softmax(child_subset_logits, dim=1)
            topk = min(2, child_probs.shape[1])
            top_vals = torch.topk(child_probs, k=topk, dim=1).values
            top1 = top_vals[:, :1]
            if topk >= 2:
                top2 = top_vals[:, 1:2]
            else:
                top2 = torch.zeros_like(top1)
            margin = top1 - top2
            return torch.cat([child_subset_logits, child_probs, top1, top2, margin, parent_logits], dim=1)
        return torch.cat([z, parent_logits, child_subset_logits], dim=1)

    def _init_parent_conditioned_child_correction(self) -> None:
        cfg = self.parent_conditioned_child_correction
        if not cfg.get("enabled", False):
            return
        parent_head_index = int(cfg["parent_head_index"])
        child_head_index = int(cfg["child_head_index"])
        hidden_dim = int(cfg.get("hidden_dim", 64))
        hotspot_child_indices = dict(cfg.get("hotspot_child_indices") or {})
        input_parent_dim = int(self.num_classes[parent_head_index])
        for parent_idx_str, child_indices in hotspot_child_indices.items():
            child_dim = int(len(child_indices))
            feature_dim = self._correction_feature_dim(child_dim=child_dim, parent_dim=input_parent_dim)
            module = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, child_dim),
            )
            final_linear = module[-1]
            assert isinstance(final_linear, nn.Linear)
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)
            self.child_correction_modules[str(parent_idx_str)] = module

    def has_parent_conditioned_child_correction(self) -> bool:
        return bool(self.parent_conditioned_child_correction.get("enabled", False))

    def apply_parent_conditioned_child_correction(
        self,
        z: torch.Tensor,
        cls_logits: List[torch.Tensor],
        *,
        parent_indices_override: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if not self.has_parent_conditioned_child_correction():
            active = torch.zeros(cls_logits[0].shape[0], dtype=torch.bool, device=cls_logits[0].device)
            return cls_logits, active

        cfg = self.parent_conditioned_child_correction
        parent_head_index = int(cfg["parent_head_index"])
        child_head_index = int(cfg["child_head_index"])
        residual_scale = float(cfg.get("residual_scale", 1.0))
        hotspot_child_indices = dict(cfg.get("hotspot_child_indices") or {})

        parent_logits = cls_logits[parent_head_index]
        child_logits = cls_logits[child_head_index]
        corrected_child_logits = child_logits.clone()
        if parent_indices_override is None:
            parent_indices = parent_logits.argmax(dim=1)
        else:
            parent_indices = parent_indices_override.to(parent_logits.device)

        active_mask = torch.zeros(parent_indices.shape[0], dtype=torch.bool, device=parent_indices.device)
        for parent_idx_str, child_indices in hotspot_child_indices.items():
            parent_idx = int(parent_idx_str)
            mask = parent_indices == parent_idx
            if not torch.any(mask):
                continue
            active_mask = active_mask | mask
            child_idx_tensor = torch.tensor(child_indices, dtype=torch.long, device=child_logits.device)
            child_subset_logits = child_logits[mask].index_select(1, child_idx_tensor)
            features = self._build_correction_features(
                z=z[mask],
                parent_logits=parent_logits[mask],
                child_subset_logits=child_subset_logits,
            )
            residual = self.child_correction_modules[str(parent_idx_str)](features) * residual_scale
            updated = corrected_child_logits[mask]
            updated[:, child_idx_tensor] = updated[:, child_idx_tensor] + residual
            corrected_child_logits[mask] = updated

        out = list(cls_logits)
        out[child_head_index] = corrected_child_logits
        return out, active_mask

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        z = self.encoder(x)
        cls_logits = [head(z) for head in self.label_heads]
        coords = {name: head(z) for name, head in self.coord_heads.items()}
        return cls_logits, coords, z

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_size": self.input_size,
                "num_classes": self.num_classes,
                "hidden_sizes": self.hidden_sizes,
                "dropout_rate": self.dropout_rate,
                "coord_dims": self.coord_dims,
                "parent_conditioned_child_correction": self.parent_conditioned_child_correction,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "AtlasMTLModel":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device)
        model = cls(
            input_size=ckpt["input_size"],
            num_classes=ckpt["num_classes"],
            hidden_sizes=ckpt["hidden_sizes"],
            dropout_rate=ckpt["dropout_rate"],
            coord_dims=ckpt.get("coord_dims", {"latent": 16, "umap": 2}),
            parent_conditioned_child_correction=ckpt.get("parent_conditioned_child_correction"),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
