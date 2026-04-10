from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: Literal["string", "categorical", "integer", "float", "boolean"]
    description: str
    required: bool = False
    nullable: bool = True
    allowed_values: tuple[str, ...] | None = None
