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
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 128]
        coord_dims = coord_dims or {"latent": 16, "umap": 2}

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.coord_dims = coord_dims

        layers: List[nn.Module] = []
        prev = input_size
        for hs in hidden_sizes:
            layers.extend([nn.Linear(prev, hs), nn.ReLU(), nn.Dropout(dropout_rate)])
            prev = hs
        self.encoder = nn.Sequential(*layers)

        self.label_heads = nn.ModuleList([nn.Linear(prev, n) for n in num_classes])
        self.coord_heads = nn.ModuleDict({k: nn.Linear(prev, d) for k, d in coord_dims.items()})

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
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
