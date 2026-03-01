from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional


@dataclass
class PreprocessConfig:
    var_names_type: Literal["symbol", "ensembl"]
    species: Literal["human", "mouse", "rat"]
    gene_id_table: Optional[str] = None
    strip_ensembl_version: bool = True
    input_matrix_type: Literal["infer", "counts", "lognorm"] = "infer"
    counts_layer: str = "counts"
    counts_required: bool = True
    counts_check_n_obs: int = 100
    counts_check_n_vals: int = 20000
    counts_check_integer_tol: float = 1e-6
    feature_space: Literal["hvg", "whole"] = "hvg"
    n_top_genes: int = 3000
    hvg_method: Literal["seurat_v3"] = "seurat_v3"
    hvg_batch_key: Optional[str] = None
    hvg_input_layer: Literal["auto", "counts", "X"] = "auto"
    duplicate_policy: Literal["sum", "mean", "first", "error"] = "sum"
    unmapped_policy: Literal["drop", "keep_original", "error"] = "drop"
    gene_symbol_column: str = "gene_symbol"
    canonical_gene_id_column: str = "ensembl_gene_id"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class FeaturePanel:
    gene_ids: List[str]
    gene_symbols: List[str] = field(default_factory=list)
    feature_space: str = "hvg"
    n_features: int = 0
    species: str = "human"
    var_names_type_original: str = "symbol"
    gene_id_table: Optional[str] = None
    hvg_method: Optional[str] = None
    n_top_genes: Optional[int] = None
    hvg_batch_key: Optional[str] = None
    counts_layer: Optional[str] = None
    hvg_layer_used: Optional[str] = None
    reference_dataset_name: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["n_features"] = int(self.n_features or len(self.gene_ids))
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "FeaturePanel":
        return cls(
            gene_ids=list(payload.get("gene_ids", [])),
            gene_symbols=list(payload.get("gene_symbols", [])),
            feature_space=str(payload.get("feature_space", "hvg")),
            n_features=int(payload.get("n_features", len(payload.get("gene_ids", [])))),
            species=str(payload.get("species", "human")),
            var_names_type_original=str(payload.get("var_names_type_original", "symbol")),
            gene_id_table=payload.get("gene_id_table"),
            hvg_method=payload.get("hvg_method"),
            n_top_genes=payload.get("n_top_genes"),
            hvg_batch_key=payload.get("hvg_batch_key"),
            counts_layer=payload.get("counts_layer"),
            hvg_layer_used=payload.get("hvg_layer_used"),
            reference_dataset_name=payload.get("reference_dataset_name"),
        )


@dataclass
class PreprocessReport:
    n_input_genes: int
    n_canonical_genes: int
    n_duplicate_genes: int
    n_unmapped_genes: int
    n_features_selected: int
    feature_space: str
    species: str
    var_names_type: str
    mapping_resource: Optional[str]
    duplicate_policy: str
    unmapped_policy: str
    input_matrix_type_declared: str = "infer"
    input_matrix_type_detected: Optional[str] = None
    counts_available: bool = False
    counts_layer_used: Optional[str] = None
    counts_check_passed: bool = False
    hvg_layer_used: Optional[str] = None
    matched_feature_genes: Optional[int] = None
    missing_feature_genes: Optional[int] = None
    ensembl_versions_stripped: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
