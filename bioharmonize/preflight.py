"""Preflight task profiles for downstream analysis readiness checks.

Each task profile defines what metadata columns and data properties a dataset
needs before it can be used for a specific bioinformatics task (clustering,
differential expression, integration, cell type annotation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from .issues import Issue

if TYPE_CHECKING:
    import anndata


@dataclass(frozen=True)
class TaskProfile:
    """Defines preflight requirements for a downstream analysis task."""

    name: str
    description: str
    required_columns: tuple[str, ...]
    recommended_columns: tuple[str, ...]
    checks: tuple[Callable[[pd.DataFrame], list[Issue]], ...] = ()
    adata_checks: tuple[Callable[[Any], list[Issue]], ...] = ()


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
# AnnData-level check helpers
# ---------------------------------------------------------------------------


def _x_is_likely_raw_counts(X: Any) -> bool:
    """Heuristic: True if X looks like raw counts (non-negative integers)."""
    try:
        from scipy import sparse

        if sparse.issparse(X):
            data = X.data
            if len(data) == 0:
                return True
            sample = data[: min(1000, len(data))]
            return bool(np.all(sample >= 0) and np.allclose(sample, np.round(sample)))
    except ImportError:
        pass
    arr = np.asarray(X)
    if arr.size == 0:
        return True
    flat = arr.flat[: min(1000, arr.size)]
    return bool(np.all(flat >= 0) and np.allclose(flat, np.round(flat)))


# ---------------------------------------------------------------------------
# AnnData-level check functions
# ---------------------------------------------------------------------------


def _check_x_exists(adata: anndata.AnnData) -> list[Issue]:
    """X matrix must exist and be non-empty."""
    if adata.X is None:
        return [
            Issue(
                severity="error",
                code="MISSING_X_MATRIX",
                column=None,
                message="AnnData has no X matrix — expression data is required",
                suggestion="Load or assign expression data to adata.X.",
            )
        ]
    if adata.X.shape[0] == 0 or adata.X.shape[1] == 0:
        return [
            Issue(
                severity="error",
                code="EMPTY_X_MATRIX",
                column=None,
                message=f"X matrix is empty (shape {adata.X.shape})",
                suggestion="Ensure the dataset contains cells and features.",
            )
        ]
    return []


def _check_x_is_normalized(adata: anndata.AnnData) -> list[Issue]:
    """Warn if X appears to be raw counts — clustering/annotation need normalized data."""
    if adata.X is None:
        return []
    if _x_is_likely_raw_counts(adata.X):
        return [
            Issue(
                severity="warning",
                code="X_APPEARS_RAW",
                column=None,
                message=(
                    "X matrix appears to contain raw counts (non-negative integers) — "
                    "this task typically expects normalized/log-transformed data"
                ),
                suggestion="Run sc.pp.normalize_total() and sc.pp.log1p() first.",
            )
        ]
    return []


def _check_x_is_raw_counts(adata: anndata.AnnData) -> list[Issue]:
    """Warn if X doesn't look like raw counts — DE methods need counts."""
    if adata.X is None:
        return []
    if not _x_is_likely_raw_counts(adata.X):
        # Check if a counts layer exists as fallback
        if "counts" in adata.layers:
            return [
                Issue(
                    severity="info",
                    code="X_NOT_COUNTS_BUT_LAYER_EXISTS",
                    column=None,
                    message=(
                        "X matrix appears normalized but a 'counts' layer exists — "
                        "DE methods can use the counts layer"
                    ),
                )
            ]
        return [
            Issue(
                severity="warning",
                code="X_NOT_RAW_COUNTS",
                column=None,
                message=(
                    "X matrix does not appear to contain raw counts — "
                    "most DE methods (DESeq2, edgeR) require integer counts"
                ),
                suggestion="Store raw counts in adata.X or adata.layers['counts'].",
            )
        ]
    return []


def _check_counts_layer_exists(adata: anndata.AnnData) -> list[Issue]:
    """Recommend a counts layer for DE workflows."""
    if adata.X is None:
        return []
    has_counts_layer = "counts" in adata.layers
    x_is_counts = _x_is_likely_raw_counts(adata.X)
    if not has_counts_layer and not x_is_counts:
        return [
            Issue(
                severity="warning",
                code="NO_COUNTS_LAYER",
                column=None,
                message=(
                    "No 'counts' layer found and X is not raw counts — "
                    "DE analysis requires access to raw count data"
                ),
                suggestion="Store raw counts: adata.layers['counts'] = raw_counts.",
            )
        ]
    return []


def _check_x_sparsity(adata: anndata.AnnData) -> list[Issue]:
    """Report X matrix sparsity as informational."""
    if adata.X is None:
        return []
    try:
        from scipy import sparse

        if sparse.issparse(adata.X):
            nnz = adata.X.nnz
            total = adata.X.shape[0] * adata.X.shape[1]
            if total > 0:
                sparsity = 1.0 - (nnz / total)
                return [
                    Issue(
                        severity="info",
                        code="X_SPARSITY",
                        column=None,
                        message=f"X matrix is sparse ({sparsity:.0%} zeros, {nnz:,} non-zero entries)",
                    )
                ]
        else:
            arr = np.asarray(adata.X)
            total = arr.size
            if total > 0:
                n_zero = int((arr == 0).sum())
                sparsity = n_zero / total
                fmt = "dense"
                return [
                    Issue(
                        severity="info",
                        code="X_SPARSITY",
                        column=None,
                        message=f"X matrix is {fmt} ({sparsity:.0%} zeros)",
                    )
                ]
    except ImportError:
        arr = np.asarray(adata.X)
        total = arr.size
        if total > 0:
            n_zero = int((arr == 0).sum())
            sparsity = n_zero / total
            return [
                Issue(
                    severity="info",
                    code="X_SPARSITY",
                    column=None,
                    message=f"X matrix is dense ({sparsity:.0%} zeros)",
                )
            ]
    return []


def _check_min_cells(adata: anndata.AnnData, *, min_cells: int = 50) -> list[Issue]:
    """Warn if dataset has very few cells."""
    n_cells = adata.n_obs
    if n_cells < min_cells:
        return [
            Issue(
                severity="warning",
                code="LOW_CELL_COUNT",
                column=None,
                message=f"Dataset has only {n_cells} cells (recommend >= {min_cells})",
                suggestion="Consider whether results will be meaningful with so few cells.",
                row_count=n_cells,
            )
        ]
    return []


def _check_cells_per_condition(adata: anndata.AnnData) -> list[Issue]:
    """Check minimum cell count per condition group for DE."""
    if "condition" not in adata.obs.columns:
        return []
    group_counts = adata.obs["condition"].value_counts()
    small_groups = group_counts[group_counts < 3]
    if len(small_groups) > 0:
        details = ", ".join(
            f"{name!r}: {count}" for name, count in small_groups.items()
        )
        return [
            Issue(
                severity="warning",
                code="LOW_CELLS_PER_CONDITION",
                column="condition",
                message=f"Condition group(s) with < 3 cells: {details}",
                suggestion="DE results may be unreliable with very few cells per group.",
                row_count=int(small_groups.sum()),
            )
        ]
    return []


def _check_cells_per_batch(adata: anndata.AnnData) -> list[Issue]:
    """Check minimum cell count per batch for integration."""
    if "batch_id" not in adata.obs.columns:
        return []
    group_counts = adata.obs["batch_id"].value_counts()
    small_groups = group_counts[group_counts < 10]
    if len(small_groups) > 0:
        details = ", ".join(
            f"{name!r}: {count}" for name, count in small_groups.items()
        )
        return [
            Issue(
                severity="warning",
                code="LOW_CELLS_PER_BATCH",
                column="batch_id",
                message=f"Batch(es) with < 10 cells: {details}",
                suggestion="Integration may be unreliable with very few cells per batch.",
                row_count=int(small_groups.sum()),
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
    adata_checks=(
        _check_x_exists,
        _check_x_is_normalized,
        _check_x_sparsity,
        _check_min_cells,
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
    adata_checks=(
        _check_x_exists,
        _check_x_is_raw_counts,
        _check_counts_layer_exists,
        _check_cells_per_condition,
        _check_x_sparsity,
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
    adata_checks=(
        _check_x_exists,
        _check_x_sparsity,
        _check_cells_per_batch,
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
    adata_checks=(
        _check_x_exists,
        _check_x_is_normalized,
        _check_x_sparsity,
        _check_min_cells,
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
    adata: anndata.AnnData | None = None,
) -> list[Issue]:
    """Run preflight checks for a downstream task against a DataFrame.

    Parameters
    ----------
    df
        The obs DataFrame to check for required/recommended columns.
    task
        Task name or TaskProfile object.
    adata
        Optional AnnData object. When provided, matrix-level checks
        (X properties, layer presence, cell counts per group) are run
        in addition to the column-level checks.

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

    # Run task-specific obs-level checks
    for check_fn in tp.checks:
        issues.extend(check_fn(df))

    # Run AnnData-level checks when an AnnData object is available
    if adata is not None:
        for check_fn in tp.adata_checks:
            issues.extend(check_fn(adata))

    return issues
