from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .changes import Change
from .issues import Issue

if TYPE_CHECKING:
    import anndata


@dataclass
class Report:
    cleaned: pd.DataFrame
    issues: list[Issue] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)
    profile_name: str = ""
    validation_level: str = "standard"
    adata: anndata.AnnData | None = field(default=None, repr=False)

    def summary(self) -> str:
        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        infos = [i for i in self.issues if i.severity == "info"]

        lines = [
            f"bioharmonize report — profile: {self.profile_name}, level: {self.validation_level}",
            f"  shape: {self.cleaned.shape[0]} rows x {self.cleaned.shape[1]} columns",
            f"  changes: {len(self.changes)}",
            f"  issues: {len(errors)} error(s), {len(warnings)} warning(s), {len(infos)} info(s)",
        ]

        if self.changes:
            lines.append("")
            lines.append("  changes:")
            for c in self.changes:
                if c.kind == "rename_column":
                    lines.append(f"    rename: {c.before} -> {c.after}")
                elif c.kind == "normalize_value":
                    count_str = f" ({c.count} rows)" if c.count else ""
                    lines.append(
                        f"    normalize [{c.column}]: {c.before!r} -> {c.after!r}{count_str}"
                    )
                elif c.kind == "coerce_dtype":
                    lines.append(f"    coerce [{c.column}]: {c.before} -> {c.after}")
                else:
                    lines.append(f"    {c.kind} [{c.column}]: {c.before} -> {c.after}")

        if errors or warnings:
            lines.append("")
            lines.append("  issues:")
            for issue in errors + warnings:
                col_str = f" [{issue.column}]" if issue.column else ""
                lines.append(f"    {issue.severity.upper()}{col_str}: {issue.message}")
                if issue.suggestion:
                    lines.append(f"      suggestion: {issue.suggestion}")

        return "\n".join(lines)

    def issues_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(
                columns=["severity", "code", "column", "message", "suggestion", "row_count"]
            )
        return pd.DataFrame(
            [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "column": i.column,
                    "message": i.message,
                    "suggestion": i.suggestion,
                    "row_count": i.row_count,
                }
                for i in self.issues
            ]
        )

    def changes_frame(self) -> pd.DataFrame:
        if not self.changes:
            return pd.DataFrame(columns=["kind", "column", "before", "after", "count"])
        return pd.DataFrame(
            [
                {
                    "kind": c.kind,
                    "column": c.column,
                    "before": c.before,
                    "after": c.after,
                    "count": c.count,
                }
                for c in self.changes
            ]
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.cleaned.to_csv(path / "cleaned.csv")
        self.issues_frame().to_csv(path / "issues.csv", index=False)
        self.changes_frame().to_csv(path / "changes.csv", index=False)
        (path / "summary.txt").write_text(self.summary())
