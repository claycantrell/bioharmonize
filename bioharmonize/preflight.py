"""Preflight task profiles for downstream analysis readiness checks.

Each task profile defines what metadata columns and data properties a dataset
needs before it can be used for a specific bioinformatics task (clustering,
differential expression, integration, cell type annotation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .issues import Issue


@dataclass(frozen=True)
class TaskProfile:
    """Defines preflight requirements for a downstream analysis task."""

    name: str
    description: str
    required_columns: tuple[str, ...]
    recommended_columns: tuple[str, ...]
    checks: tuple[Callable[[pd.DataFrame], list[Issue]], ...] = ()


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def _check_condition_has_groups(df: pd.DataFrame) -> list[Issue]:
    """Condition column must have at least 2 unique values for comparison."""
    if "condition" not in df.columns:
        return []
    n_groups = df["condition"].dropna().nunique()
    if n_groups < 2:
        return [
            Issue(
                severity="error",
                code="INSUFFICIENT_GROUPS",
                column="condition",
                message=f"Differential expression requires >= 2 condition groups, found {n_groups}",
                suggestion="Ensure the condition column encodes the groups to compare.",
            )
        ]
    return []


def _check_replicates_per_condition(df: pd.DataFrame) -> list[Issue]:
    """Warn if any condition group has only 1 biological replicate."""
    if "condition" not in df.columns or "sample_id" not in df.columns:
        return []
    groups = df.groupby("condition")["sample_id"].nunique()
    low = groups[groups < 2]
    if len(low) > 0:
        names = sorted(low.index.tolist())
        return [
            Issue(
                severity="warning",
                code="LOW_REPLICATES",
                column="sample_id",
                message=(
                    f"Condition group(s) {names} have < 2 biological replicates "
                    f"(unique sample_id values) — statistical power will be limited"
                ),
                suggestion="Pseudo-bulk DE methods need >= 2 samples per condition.",
            )
        ]
    return []


def _check_batch_has_variation(df: pd.DataFrame) -> list[Issue]:
    """Batch variable must have >= 2 unique values to integrate across."""
    if "batch_id" not in df.columns:
        return []
    n_batches = df["batch_id"].dropna().nunique()
    if n_batches < 2:
        return [
            Issue(
                severity="error",
                code="SINGLE_BATCH",
                column="batch_id",
                message=f"Integration requires >= 2 batches, found {n_batches}",
                suggestion="If all data is from one batch, integration is unnecessary.",
            )
        ]
    return []


def _check_cell_type_exists(df: pd.DataFrame) -> list[Issue]:
    """Warn if cell_type column already exists (annotation will overwrite)."""
    if "cell_type" in df.columns:
        n_unique = df["cell_type"].dropna().nunique()
        return [
            Issue(
                severity="info",
                code="EXISTING_ANNOTATION",
                column="cell_type",
                message=f"cell_type column already exists with {n_unique} unique value(s) — annotation may overwrite it",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Built-in task profiles
# ---------------------------------------------------------------------------

CLUSTERING = TaskProfile(
    name="clustering",
    description="Unsupervised cell clustering (Leiden, Louvain, k-means)",
    required_columns=(),
    recommended_columns=("batch_id", "sample_id"),
    checks=(
        _check_batch_has_variation,
    ),
)

DIFFERENTIAL_EXPRESSION = TaskProfile(
    name="differential_expression",
    description="Differential expression analysis between conditions",
    required_columns=("condition",),
    recommended_columns=("sample_id", "donor_id"),
    checks=(
        _check_condition_has_groups,
        _check_replicates_per_condition,
    ),
)

INTEGRATION = TaskProfile(
    name="integration",
    description="Cross-batch or cross-sample data integration (Harmony, scVI, BBKNN)",
    required_columns=("batch_id",),
    recommended_columns=("sample_id", "assay"),
    checks=(
        _check_batch_has_variation,
    ),
)

CELL_TYPE_ANNOTATION = TaskProfile(
    name="cell_type_annotation",
    description="Automated or manual cell type annotation",
    required_columns=(),
    recommended_columns=("tissue", "species"),
    checks=(
        _check_cell_type_exists,
    ),
)

_BUILTIN_TASKS: dict[str, TaskProfile] = {
    "clustering": CLUSTERING,
    "differential_expression": DIFFERENTIAL_EXPRESSION,
    "integration": INTEGRATION,
    "cell_type_annotation": CELL_TYPE_ANNOTATION,
}


def resolve_task(task: str | TaskProfile) -> TaskProfile:
    """Look up a built-in task profile by name, or pass through a TaskProfile."""
    if isinstance(task, TaskProfile):
        return task
    if task in _BUILTIN_TASKS:
        return _BUILTIN_TASKS[task]
    raise ValueError(
        f"Unknown task profile: {task!r}. Available: {sorted(_BUILTIN_TASKS)}"
    )


def list_tasks() -> list[str]:
    """Return the names of all built-in task profiles."""
    return sorted(_BUILTIN_TASKS)


def run_preflight(
    df: pd.DataFrame,
    task: str | TaskProfile,
) -> list[Issue]:
    """Run preflight checks for a downstream task against a DataFrame.

    Returns a list of Issues describing missing requirements, recommendations,
    and task-specific data quality concerns.
    """
    tp = resolve_task(task)
    issues: list[Issue] = []

    # Check required columns
    for col in tp.required_columns:
        if col not in df.columns:
            issues.append(
                Issue(
                    severity="error",
                    code="PREFLIGHT_MISSING_REQUIRED",
                    column=col,
                    message=f"Task '{tp.name}' requires column {col!r} but it is missing",
                )
            )

    # Check recommended columns
    for col in tp.recommended_columns:
        if col not in df.columns:
            issues.append(
                Issue(
                    severity="warning",
                    code="PREFLIGHT_MISSING_RECOMMENDED",
                    column=col,
                    message=f"Task '{tp.name}' recommends column {col!r} for best results",
                )
            )

    # Run task-specific checks
    for check_fn in tp.checks:
        issues.extend(check_fn(df))

    return issues
