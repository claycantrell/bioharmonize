from __future__ import annotations

import re


def normalize_col_name(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())
