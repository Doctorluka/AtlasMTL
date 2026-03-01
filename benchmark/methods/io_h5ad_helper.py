from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite


def _write_matrix(path: Path, matrix) -> None:
    if sparse.issparse(matrix):
        mmwrite(str(path), matrix.tocoo())
        return
    mmwrite(str(path), sparse.coo_matrix(np.asarray(matrix)))


def _export_h5ad(args: argparse.Namespace) -> None:
    adata = ad.read_h5ad(args.input)
    layer_name = None if args.layer in (None, "", "X") else args.layer
    if layer_name and layer_name in adata.layers:
        matrix = adata.layers[layer_name]
        matrix_source = f"layers/{layer_name}"
    else:
        matrix = adata.X
        matrix_source = "X"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_matrix(output_dir / "matrix.mtx", matrix)
    adata.obs.to_csv(output_dir / "obs.csv")
    adata.var.to_csv(output_dir / "var.csv")

    obsm_written: list[str] = []
    for key in args.obsm or []:
        candidates = [key]
        if not key.startswith("X_"):
            candidates.insert(0, f"X_{key}")
        found = next((candidate for candidate in candidates if candidate in adata.obsm), None)
        if found is None:
            continue
        frame = pd.DataFrame(np.asarray(adata.obsm[found]), index=adata.obs_names)
        frame.to_csv(output_dir / f"obsm__{key}.csv")
        obsm_written.append(key)

    payload = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "matrix_source": matrix_source,
        "obs_columns": list(map(str, adata.obs.columns)),
        "var_columns": list(map(str, adata.var.columns)),
        "obsm_written": obsm_written,
    }
    (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_matrix(path: Path):
    matrix = mmread(str(path))
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    return matrix.tocsr()


def _import_h5ad(args: argparse.Namespace) -> None:
    bundle_dir = Path(args.input_dir)
    matrix = _read_matrix(bundle_dir / "matrix.mtx")
    obs = pd.read_csv(bundle_dir / "obs.csv", index_col=0)
    var = pd.read_csv(bundle_dir / "var.csv", index_col=0)
    adata = ad.AnnData(X=matrix.T.tocsr(), obs=obs, var=var)

    for path in sorted(bundle_dir.glob("obsm__*.csv")):
        key = path.stem.split("__", 1)[1]
        frame = pd.read_csv(path, index_col=0)
        adata.obsm[f"X_{key}"] = frame.loc[adata.obs_names].to_numpy(dtype=np.float32)

    layer_name = args.layer_name or "counts"
    adata.layers[layer_name] = adata.X.copy()
    adata.write_h5ad(args.output, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-h5ad")
    export_parser.add_argument("--input", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--layer", default="counts")
    export_parser.add_argument("--obsm", nargs="*")
    export_parser.set_defaults(func=_export_h5ad)

    import_parser = subparsers.add_parser("import-h5ad")
    import_parser.add_argument("--input-dir", required=True)
    import_parser.add_argument("--output", required=True)
    import_parser.add_argument("--layer-name", default="counts")
    import_parser.set_defaults(func=_import_h5ad)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
