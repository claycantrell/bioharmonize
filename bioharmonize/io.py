from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import anndata


def read_obs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    if path.suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", index_col=0, encoding="utf-8-sig")
    if path.suffix in {".h5ad", ".h5"}:
        return read_h5ad(path).obs
    raise ValueError(f"Unsupported metadata file format: {path.suffix}")


def read_h5ad(path: str | Path) -> anndata.AnnData:
    """Read an h5ad file and return the AnnData object.

    Requires the ``anndata`` package. Install with::

        pip install bioharmonize[anndata]
    """
    try:
        import anndata as _anndata
    except ImportError:
        raise ImportError(
            "anndata is required to read .h5ad files. "
            "Install with: pip install bioharmonize[anndata]"
        ) from None
    return _anndata.read_h5ad(Path(path))
