from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_obs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    if path.suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", index_col=0, encoding="utf-8-sig")
    raise ValueError(f"Unsupported metadata file format: {path.suffix}")
