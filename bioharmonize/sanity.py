"""Dataset sanity checks for AnnData objects.

Structural integrity checks that go beyond obs-level metadata validation:
shape consistency, duplicate cell IDs, missing features, all-zero columns,
layer presence, categorical encoding, and index hygiene.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .issues import Issue

if TYPE_CHECKING:
    import anndata


ALL_CHECKS = (
    "obs_matrix_shape",
    "duplicate_cell_ids",
    "missing_features",
    "all_zero_columns",
    "layer_presence",
    "categorical_encoding",
    "index_hygiene",
)

# String columns with more unique values than this are flagged for categorical encoding
_CATEGORICAL_THRESHOLD = 0.5  # ratio of unique/total


def check_dataset(
    adata: anndata.AnnData,
    *,
    checks: list[str] | None = None,
) -> list[Issue]:
    """Run sanity checks on an AnnData dataset.

    Parameters
    ----------
    adata
        The AnnData object to check.
    checks
        List of check names to run. ``None`` runs all checks.
        Valid names: ``obs_matrix_shape``, ``duplicate_cell_ids``,
        ``missing_features``, ``all_zero_columns``, ``layer_presence``,
        ``categorical_encoding``, ``index_hygiene``.

    Returns
    -------
    list[Issue]
        Issues found during the checks.
    """
    if checks is None:
        checks = list(ALL_CHECKS)

    dispatch = {
        "obs_matrix_shape": _check_obs_matrix_shape,
        "duplicate_cell_ids": _check_duplicate_cell_ids,
        "missing_features": _check_missing_features,
        "all_zero_columns": _check_all_zero_columns,
        "layer_presence": _check_layer_presence,
        "categorical_encoding": _check_categorical_encoding,
        "index_hygiene": _check_index_hygiene,
    }

    issues: list[Issue] = []
    for name in checks:
        fn = dispatch.get(name)
        if fn is None:
            raise ValueError(
                f"Unknown check {name!r}. Valid checks: {', '.join(ALL_CHECKS)}"
            )
        issues.extend(fn(adata))
    return issues


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_obs_matrix_shape(adata: anndata.AnnData) -> list[Issue]:
    """Verify obs row count matches the expression matrix row count."""
    issues: list[Issue] = []
    if adata.X is not None:
        n_obs = adata.obs.shape[0]
        n_x_rows = adata.X.shape[0]
        if n_obs != n_x_rows:
            issues.append(
                Issue(
                    severity="error",
                    code="OBS_MATRIX_SHAPE_MISMATCH",
                    column=None,
                    message=(
                        f"obs has {n_obs} rows but expression matrix has {n_x_rows} rows"
                    ),
                )
            )
        if adata.var.shape[0] != adata.X.shape[1]:
            issues.append(
                Issue(
                    severity="error",
                    code="VAR_MATRIX_SHAPE_MISMATCH",
                    column=None,
                    message=(
                        f"var has {adata.var.shape[0]} entries but expression matrix "
                        f"has {adata.X.shape[1]} columns"
                    ),
                )
            )
    return issues


def _check_duplicate_cell_ids(adata: anndata.AnnData) -> list[Issue]:
    """Check for duplicate cell IDs in the obs index."""
    issues: list[Issue] = []
    if adata.obs.index.duplicated().any():
        n_dupes = int(adata.obs.index.duplicated().sum())
        issues.append(
            Issue(
                severity="error",
                code="DUPLICATE_CELL_IDS",
                column=None,
                message=f"obs index has {n_dupes} duplicate cell ID(s)",
                suggestion="Ensure each cell has a unique barcode or identifier.",
                row_count=n_dupes,
            )
        )
    return issues


def _check_missing_features(adata: anndata.AnnData) -> list[Issue]:
    """Check for missing or empty feature names in var."""
    issues: list[Issue] = []
    var_index = adata.var.index

    # Check for NaN/None in var index
    n_null = int(pd.isna(var_index).sum())
    if n_null > 0:
        issues.append(
            Issue(
                severity="error",
                code="NULL_FEATURE_NAMES",
                column=None,
                message=f"var index has {n_null} null feature name(s)",
                suggestion="All features should have non-null names (gene symbols or IDs).",
                row_count=n_null,
            )
        )

    # Check for empty strings
    str_index = var_index.astype(str)
    n_empty = int((str_index.str.strip() == "").sum())
    if n_empty > 0:
        issues.append(
            Issue(
                severity="error",
                code="EMPTY_FEATURE_NAMES",
                column=None,
                message=f"var index has {n_empty} empty feature name(s)",
                suggestion="All features should have non-empty names.",
                row_count=n_empty,
            )
        )

    # Check for duplicate feature names
    if var_index.duplicated().any():
        n_dupes = int(var_index.duplicated().sum())
        issues.append(
            Issue(
                severity="warning",
                code="DUPLICATE_FEATURE_NAMES",
                column=None,
                message=f"var index has {n_dupes} duplicate feature name(s)",
                suggestion="Consider using make_names_unique() or switching to Ensembl IDs.",
                row_count=n_dupes,
            )
        )
    return issues


def _check_all_zero_columns(adata: anndata.AnnData) -> list[Issue]:
    """Detect features (columns in X) that are entirely zero."""
    issues: list[Issue] = []
    if adata.X is None or adata.X.shape[1] == 0:
        return issues

    X = adata.X
    try:
        from scipy import sparse

        if sparse.issparse(X):
            # For sparse matrices, check which columns have no nonzero entries
            col_nnz = np.diff(X.tocsc().indptr)
            n_zero_cols = int((col_nnz == 0).sum())
        else:
            n_zero_cols = int((np.asarray(X) == 0).all(axis=0).sum())
    except ImportError:
        n_zero_cols = int((np.asarray(X) == 0).all(axis=0).sum())

    if n_zero_cols > 0:
        issues.append(
            Issue(
                severity="warning",
                code="ALL_ZERO_FEATURES",
                column=None,
                message=f"{n_zero_cols} feature(s) have all-zero expression across all cells",
                suggestion="Consider filtering with sc.pp.filter_genes(min_cells=1).",
                row_count=n_zero_cols,
            )
        )
    return issues


def _check_layer_presence(adata: anndata.AnnData) -> list[Issue]:
    """Check that layers exist and have consistent shapes."""
    issues: list[Issue] = []
    if not adata.layers:
        return issues

    expected_shape = adata.X.shape if adata.X is not None else (adata.n_obs, adata.n_vars)
    for name, layer in adata.layers.items():
        if layer.shape != expected_shape:
            issues.append(
                Issue(
                    severity="error",
                    code="LAYER_SHAPE_MISMATCH",
                    column=name,
                    message=(
                        f"Layer {name!r} has shape {layer.shape} but expected {expected_shape}"
                    ),
                )
            )
    return issues


def _check_categorical_encoding(adata: anndata.AnnData) -> list[Issue]:
    """Flag string columns with low cardinality that could be categorical."""
    issues: list[Issue] = []
    n_obs = adata.n_obs
    if n_obs == 0:
        return issues

    for col in adata.obs.columns:
        series = adata.obs[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            # Check for unused categories (always, regardless of dataset size)
            if hasattr(series.cat, "categories"):
                used = series.dropna().unique()
                unused = set(series.cat.categories) - set(used)
                if unused:
                    issues.append(
                        Issue(
                            severity="info",
                            code="UNUSED_CATEGORIES",
                            column=col,
                            message=(
                                f"Column {col!r} has {len(unused)} unused "
                                f"categor{'y' if len(unused) == 1 else 'ies'}: "
                                f"{sorted(str(c) for c in unused)}"
                            ),
                            suggestion="Remove unused categories with .cat.remove_unused_categories().",
                        )
                    )
        elif n_obs >= 50 and (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
            # Only suggest categorical encoding on datasets large enough for it to matter
            n_unique = series.nunique()
            if n_unique > 0 and (n_unique / n_obs) <= _CATEGORICAL_THRESHOLD:
                issues.append(
                    Issue(
                        severity="info",
                        code="COULD_BE_CATEGORICAL",
                        column=col,
                        message=(
                            f"Column {col!r} has {n_unique} unique values across "
                            f"{n_obs} cells ({n_unique/n_obs:.0%} unique) — "
                            f"consider categorical encoding"
                        ),
                        suggestion=f"Convert with: adata.obs['{col}'] = adata.obs['{col}'].astype('category')",
                    )
                )
    return issues


def _check_index_hygiene(adata: anndata.AnnData) -> list[Issue]:
    """Check obs index for common problems."""
    issues: list[Issue] = []
    idx = adata.obs.index

    # Check for null values in index
    n_null = int(pd.isna(idx).sum())
    if n_null > 0:
        issues.append(
            Issue(
                severity="error",
                code="NULL_INDEX_VALUES",
                column=None,
                message=f"obs index has {n_null} null value(s)",
                suggestion="Cell IDs should never be null.",
                row_count=n_null,
            )
        )

    # Check for default integer index (likely unset)
    if pd.api.types.is_integer_dtype(idx):
        issues.append(
            Issue(
                severity="warning",
                code="DEFAULT_INTEGER_INDEX",
                column=None,
                message="obs index is integer-based (likely a default RangeIndex)",
                suggestion="Set meaningful cell IDs: adata.obs_names = cell_barcodes",
            )
        )

    # Check for whitespace in index values
    str_idx = idx.astype(str)
    has_leading_trailing = (str_idx != str_idx.str.strip()).any()
    if has_leading_trailing:
        n_ws = int((str_idx != str_idx.str.strip()).sum())
        issues.append(
            Issue(
                severity="warning",
                code="INDEX_WHITESPACE",
                column=None,
                message=f"{n_ws} cell ID(s) have leading or trailing whitespace",
                suggestion="Strip whitespace: adata.obs_names = adata.obs_names.str.strip()",
                row_count=n_ws,
            )
        )

    return issues
