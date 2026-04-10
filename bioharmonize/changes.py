from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Change:
    kind: Literal[
        "rename_column", "normalize_value", "coerce_dtype", "drop_column", "fill_missing"
    ]
    column: str
    before: Any
    after: Any
    count: int | None = None
