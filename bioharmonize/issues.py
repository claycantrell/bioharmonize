from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Issue:
    severity: Literal["info", "warning", "error"]
    code: str
    column: str | None
    message: str
    suggestion: str | None = None
    row_count: int | None = None
