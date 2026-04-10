from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd

from .changes import Change
from .issues import Issue
from .preflight import TaskProfile, resolve_task, run_preflight
from .profiles import Profile, resolve_profile
from .report import Report
from .sanity import check_dataset
from .validators import run_validation

if TYPE_CHECKING:
    import anndata

#: Accepted input types for the top-level API.
ObsData = Union["anndata.AnnData", pd.DataFrame]


def _normalize_col_name(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _extract_obs(data: ObsData) -> tuple[pd.DataFrame, Any]:
    """Return (obs_dataframe, adata_or_none)."""
    try:
        import anndata as _anndata

        if isinstance(data, _anndata.AnnData):
            return data.obs, data
    except ImportError:
        pass
    if isinstance(data, pd.DataFrame):
        return data, None
    raise TypeError(f"Expected AnnData or DataFrame, got {type(data).__name__}")


def _guess_data_state(adata: Any) -> str:
    """Heuristic to guess whether X holds raw counts, normalized, or log-transformed data."""
    if adata.X is None or adata.X.shape[0] == 0 or adata.X.shape[1] == 0:
        return "empty"

    X = adata.X
    try:
        from scipy import sparse

        if sparse.issparse(X):
            sample = np.asarray(X[:min(500, X.shape[0])].todense())
        else:
            sample = np.asarray(X[:min(500, X.shape[0])])
    except ImportError:
        sample = np.asarray(X[:min(500, X.shape[0])])

    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return "empty (all NaN/Inf)"

    has_negative = bool(finite.min() < 0)
    max_val = float(finite.max())
    is_integer = bool(np.allclose(finite, np.round(finite)))

    if has_negative:
        return "scaled (negative values present)"
    if is_integer and max_val > 0:
        return "raw counts (integer values)"
    if max_val <= 20:
        return "log-normalized (small positive floats)"
    return "normalized or transformed"


def _dataset_diagnostics(adata: Any) -> list[Issue]:
    """Generate informational diagnostics for an AnnData object."""
    issues: list[Issue] = []

    # Shape
    issues.append(
        Issue(
            severity="info",
            code="DATASET_SHAPE",
            column=None,
            message=f"{adata.n_obs} cells \u00d7 {adata.n_vars} features",
        )
    )

    # Layers
    layer_names = list(adata.layers.keys()) if adata.layers else []
    if layer_names:
        issues.append(
            Issue(
                severity="info",
                code="DATASET_LAYERS",
                column=None,
                message=f"Layers: {', '.join(sorted(layer_names))}",
            )
        )
    else:
        issues.append(
            Issue(
                severity="info",
                code="DATASET_LAYERS",
                column=None,
                message="No additional layers (X only)",
            )
        )

    # Key obs columns
    obs_cols = list(adata.obs.columns)
    if obs_cols:
        issues.append(
            Issue(
                severity="info",
                code="OBS_COLUMNS",
                column=None,
                message=f"obs columns ({len(obs_cols)}): {', '.join(obs_cols)}",
            )
        )
    else:
        issues.append(
            Issue(
                severity="info",
                code="OBS_COLUMNS",
                column=None,
                message="obs has no annotation columns",
            )
        )

    # Key var columns
    var_cols = list(adata.var.columns)
    if var_cols:
        issues.append(
            Issue(
                severity="info",
                code="VAR_COLUMNS",
                column=None,
                message=f"var columns ({len(var_cols)}): {', '.join(var_cols)}",
            )
        )

    # Missingness stats
    obs_df = adata.obs
    if not obs_df.empty and len(obs_df.columns) > 0:
        missing = obs_df.isnull().sum()
        cols_with_missing = missing[missing > 0]
        if not cols_with_missing.empty:
            n_rows = len(obs_df)
            parts = [
                f"{col}: {count}/{n_rows} ({count / n_rows:.0%})"
                for col, count in cols_with_missing.items()
            ]
            issues.append(
                Issue(
                    severity="info",
                    code="OBS_MISSINGNESS",
                    column=None,
                    message=f"Missing values in obs \u2014 {', '.join(parts)}",
                )
            )

    # Likely data state
    state = _guess_data_state(adata)
    issues.append(
        Issue(
            severity="info",
            code="DATA_STATE",
            column=None,
            message=f"Likely data state: {state}",
        )
    )

    return issues


def _build_rename_map(
    df: pd.DataFrame,
    prof: Profile,
    column_map: dict[str, str] | None = None,
) -> tuple[dict[str, str], list[Issue]]:
    """Build the column rename map and collect conflict issues."""
    rename_map: dict[str, str] = {}
    issues: list[Issue] = []

    alias_lookup: dict[str, str] = {
        _normalize_col_name(k): v for k, v in prof.column_aliases.items()
    }

    # Explicit user mappings first (highest priority)
    if column_map:
        for src, dst in column_map.items():
            if src in df.columns:
                rename_map[src] = dst

    canonical_names = set(prof.canonical_columns)
    already_targeted = set(rename_map.values()) | (set(df.columns) & canonical_names)

    for col in df.columns:
        if col in rename_map:
            continue
        normed = _normalize_col_name(col)
        if normed in canonical_names and normed == col:
            continue
        if normed in canonical_names and normed != col:
            target = normed
        else:
            target = alias_lookup.get(normed)
        if target is None:
            target = alias_lookup.get(col.lower().strip())
        if target is not None and target not in already_targeted:
            rename_map[col] = target
            already_targeted.add(target)
        elif target is not None and target in already_targeted:
            issues.append(
                Issue(
                    severity="error",
                    code="COLUMN_CONFLICT",
                    column=col,
                    message=(
                        f"Column {col!r} maps to {target!r} but that target is already "
                        f"present or mapped from another column"
                    ),
                    suggestion="Use column_map to resolve the conflict explicitly.",
                )
            )

    return rename_map, issues


def _compute_normalizations(
    df: pd.DataFrame,
    prof: Profile,
    value_maps: dict[str, dict[str, str]] | None = None,
) -> list[Change]:
    """Compute value normalization changes without modifying the DataFrame."""
    changes: list[Change] = []
    all_normalizers = dict(prof.value_normalizers)
    if value_maps:
        from .normalizers import make_map_normalizer

        for col_name, vmap in value_maps.items():
            all_normalizers[col_name] = make_map_normalizer(vmap)

    for col_name, normalizer in all_normalizers.items():
        if col_name not in df.columns:
            continue
        if df.columns.duplicated().any() and isinstance(df[col_name], pd.DataFrame):
            continue
        series = df[col_name]
        mask = series.notna()
        if not mask.any():
            continue

        for val in series[mask].unique():
            normalized = normalizer(str(val))
            if normalized is not None and normalized != str(val):
                count = int((series == val).sum())
                changes.append(
                    Change(
                        kind="normalize_value",
                        column=col_name,
                        before=str(val),
                        after=normalized,
                        count=count,
                    )
                )

    return changes


def _apply_normalizations(
    df: pd.DataFrame,
    prof: Profile,
    value_maps: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, list[Change]]:
    """Apply value normalizations to the DataFrame, returning changes."""
    changes: list[Change] = []
    all_normalizers = dict(prof.value_normalizers)
    if value_maps:
        from .normalizers import make_map_normalizer

        for col_name, vmap in value_maps.items():
            all_normalizers[col_name] = make_map_normalizer(vmap)

    for col_name, normalizer in all_normalizers.items():
        if col_name not in df.columns:
            continue
        if df.columns.duplicated().any() and isinstance(df[col_name], pd.DataFrame):
            continue
        series = df[col_name]
        mask = series.notna()
        if not mask.any():
            continue

        new_values = series.copy()
        for val in series[mask].unique():
            normalized = normalizer(str(val))
            if normalized is not None and normalized != str(val):
                val_mask = series == val
                count = int(val_mask.sum())
                new_values[val_mask] = normalized
                changes.append(
                    Change(
                        kind="normalize_value",
                        column=col_name,
                        before=str(val),
                        after=normalized,
                        count=count,
                    )
                )
        df[col_name] = new_values

    return df, changes


def _apply_dtype_coercions(
    df: pd.DataFrame, prof: Profile
) -> tuple[pd.DataFrame, list[Change]]:
    """Apply dtype coercions and return changes."""
    changes: list[Change] = []
    for col_name, spec in prof.canonical_columns.items():
        if col_name not in df.columns:
            continue
        if df.columns.duplicated().any() and isinstance(df[col_name], pd.DataFrame):
            continue
        series = df[col_name]
        old_dtype = str(series.dtype)

        if spec.dtype in ("string", "categorical") and not pd.api.types.is_string_dtype(series):
            if pd.api.types.is_numeric_dtype(series):
                df[col_name] = series.apply(
                    lambda x: (
                        str(int(x))
                        if pd.notna(x) and float(x) == int(x)
                        else (str(x) if pd.notna(x) else x)
                    )
                )
                changes.append(
                    Change(kind="coerce_dtype", column=col_name, before=old_dtype, after="string")
                )
        elif spec.dtype == "integer" and not pd.api.types.is_integer_dtype(series):
            try:
                df[col_name] = pd.to_numeric(series, errors="raise").astype("Int64")
                changes.append(
                    Change(kind="coerce_dtype", column=col_name, before=old_dtype, after="Int64")
                )
            except (ValueError, TypeError):
                pass
        elif spec.dtype == "float" and not pd.api.types.is_float_dtype(series):
            try:
                df[col_name] = pd.to_numeric(series, errors="raise").astype("Float64")
                changes.append(
                    Change(kind="coerce_dtype", column=col_name, before=old_dtype, after="Float64")
                )
            except (ValueError, TypeError):
                pass

    return df, changes


def _repair_index(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Change]]:
    """Fix structural index issues: whitespace, integer index, duplicates."""
    changes: list[Change] = []

    # Step 1: Strip leading/trailing whitespace from index values
    str_idx = df.index.astype(str)
    stripped = str_idx.str.strip()
    n_ws = int((str_idx != stripped).sum())
    if n_ws > 0:
        df.index = pd.Index(stripped, name=df.index.name)
        changes.append(
            Change(
                kind="strip_index",
                column="(index)",
                before="whitespace in cell IDs",
                after="stripped",
                count=n_ws,
            )
        )

    # Step 2: Replace default integer / unnamed index with generated cell IDs
    # Check for actual integer dtype OR string representations of sequential integers
    # (AnnData converts integer indices to strings like '0', '1', '2')
    _is_integer_index = pd.api.types.is_integer_dtype(df.index)
    if not _is_integer_index and len(df) > 0:
        try:
            as_ints = [int(v) for v in df.index]
            _is_integer_index = as_ints == list(range(len(df)))
        except (ValueError, TypeError):
            pass
    if _is_integer_index:
        n = len(df)
        new_index = pd.Index([f"cell_{i}" for i in range(n)])
        df.index = new_index
        changes.append(
            Change(
                kind="generate_index",
                column="(index)",
                before="integer index",
                after="cell_0..cell_{0}".format(n - 1) if n > 0 else "cell_0",
                count=n,
            )
        )

    # Step 3: Deduplicate cell IDs by appending -1, -2, ... suffixes
    if df.index.duplicated().any():
        n_dupes = int(df.index.duplicated().sum())
        seen: dict[str, int] = {}
        new_ids: list[str] = []
        for val in df.index:
            s = str(val)
            if s in seen:
                seen[s] += 1
                new_ids.append(f"{s}-{seen[s]}")
            else:
                seen[s] = 0
                new_ids.append(s)
        df.index = pd.Index(new_ids, name=df.index.name)
        changes.append(
            Change(
                kind="deduplicate_index",
                column="(index)",
                before=f"{n_dupes} duplicate cell ID(s)",
                after="unique suffixes appended",
                count=n_dupes,
            )
        )

    return df, changes


def _wrap_adata(
    cleaned_obs: pd.DataFrame,
    source_adata: Any,
    copy: bool,
) -> Any:
    """Wrap repaired obs back into an AnnData object."""
    if source_adata is None:
        return None
    if copy:
        new_ad = source_adata.copy()
        new_ad.obs = cleaned_obs
        return new_ad
    else:
        source_adata.obs = cleaned_obs
        return source_adata


# ── Top-level API ────────────────────────────────────────────────────────────


def inspect(
    data: ObsData,
    *,
    profile: str | Profile = "single_cell_human",
    column_map: dict[str, str] | None = None,
    value_maps: dict[str, dict[str, str]] | None = None,
) -> Report:
    """Analyse metadata without modifying it.

    Returns a Report whose ``changes`` list shows what *would* be renamed or
    normalised (but nothing is applied) and whose ``issues`` list contains
    minimal-level validation findings plus informational diagnostics.
    """
    prof = resolve_profile(profile)
    obs, source_adata = _extract_obs(data)
    df = obs.copy()

    # Planned renames
    rename_map, conflict_issues = _build_rename_map(df, prof, column_map)
    planned_changes: list[Change] = [
        Change(kind="rename_column", column=dst, before=src, after=dst)
        for src, dst in rename_map.items()
    ]

    # Preview normalizations on renamed copy
    preview = df.rename(columns=rename_map)
    norm_changes = _compute_normalizations(preview, prof, value_maps)
    planned_changes.extend(norm_changes)

    # Minimal validation on original data
    issues: list[Issue] = list(conflict_issues)
    issues.extend(run_validation(df, prof, level="minimal"))

    # AnnData-specific diagnostics: shape, layers, missingness, data state
    if source_adata is not None:
        issues.extend(_dataset_diagnostics(source_adata))

    return Report(
        cleaned=obs.copy(),
        issues=issues,
        changes=planned_changes,
        profile_name=prof.name,
        validation_level="minimal",
        adata=source_adata,
    )


def validate(
    data: ObsData,
    *,
    profile: str | Profile = "single_cell_human",
    level: str = "standard",
) -> Report:
    """Validate metadata without modifying it.

    Runs validation rules at the requested level and returns issues.
    """
    prof = resolve_profile(profile)
    obs, source_adata = _extract_obs(data)
    df = obs.copy()
    issues = run_validation(df, prof, level=level)

    # Flag columns that look like known aliases but weren't renamed.
    # Also detect conflicts where multiple columns map to the same target.
    alias_lookup = {_normalize_col_name(k): v for k, v in prof.column_aliases.items()}
    alias_hits: dict[str, list[str]] = {}  # target -> list of source columns
    for col in df.columns:
        normed = _normalize_col_name(col)
        target = alias_lookup.get(normed) or alias_lookup.get(col.lower().strip())
        if target and target not in df.columns:
            alias_hits.setdefault(target, []).append(col)
            issues.append(
                Issue(
                    severity="warning",
                    code="POSSIBLE_ALIAS",
                    column=col,
                    message=(
                        f"Column {col!r} looks like an alias for canonical column {target!r}. "
                        f"Use repair() to rename automatically."
                    ),
                    suggestion=f"Run bh.repair(data) to rename {col!r} → {target!r}.",
                )
            )

    # Flag conflicts where multiple columns compete for the same canonical name
    for target, sources in alias_hits.items():
        if len(sources) > 1:
            issues.append(
                Issue(
                    severity="error",
                    code="ALIAS_CONFLICT",
                    column=target,
                    message=(
                        f"Multiple columns map to {target!r}: {sources!r}. "
                        f"Ambiguous — use column_map to resolve the conflict explicitly."
                    ),
                    suggestion=f"Pass column_map={{'{sources[0]}': '{target}'}} to choose which column to keep.",
                )
            )

    # Run structural sanity checks when input is AnnData
    if source_adata is not None:
        issues.extend(check_dataset(source_adata))

    return Report(
        cleaned=df,
        issues=issues,
        changes=[],
        profile_name=prof.name,
        validation_level=level,
        adata=source_adata,
    )


def repair(
    data: ObsData,
    *,
    profile: str | Profile = "single_cell_human",
    column_map: dict[str, str] | None = None,
    value_maps: dict[str, dict[str, str]] | None = None,
    validation: str = "standard",
    copy: bool = True,
    inplace: bool | None = None,
) -> "Report | tuple[Any, Report]":
    """Rename columns, normalise values, coerce dtypes, then validate.

    When *data* is an ``AnnData``, returns ``(repaired_adata, report)`` tuple.
    When *data* is a ``DataFrame``, returns a ``Report``.

    The ``inplace`` parameter is an alias for ``not copy`` (pandas/AnnData
    convention). If both are specified, ``inplace`` takes precedence.
    """
    if inplace is not None:
        copy = not inplace
    prof = resolve_profile(profile)
    obs, source_adata = _extract_obs(data)
    result = obs.copy() if copy else obs
    changes: list[Change] = []
    issues: list[Issue] = []

    # step 0: structural index repairs (whitespace, integer index, duplicates)
    result, index_changes = _repair_index(result)
    changes.extend(index_changes)

    # step 1: column renames
    rename_map, conflict_issues = _build_rename_map(result, prof, column_map)
    issues.extend(conflict_issues)
    if rename_map:
        result = result.rename(columns=rename_map)
        for src, dst in rename_map.items():
            changes.append(Change(kind="rename_column", column=dst, before=src, after=dst))

    # step 2: value normalizations
    result, norm_changes = _apply_normalizations(result, prof, value_maps)
    changes.extend(norm_changes)

    # step 3: dtype coercions
    result, dtype_changes = _apply_dtype_coercions(result, prof)
    changes.extend(dtype_changes)

    # step 4: validation
    issues.extend(run_validation(result, prof, level=validation))

    repaired_adata = _wrap_adata(result, source_adata, copy)

    report = Report(
        cleaned=result,
        issues=issues,
        changes=changes,
        profile_name=prof.name,
        validation_level=validation,
        adata=repaired_adata,
    )

    # Return (AnnData, Report) tuple when input was AnnData — standard convention
    if repaired_adata is not None:
        return repaired_adata, report
    return report


def preflight(
    data: ObsData,
    task: str | TaskProfile,
    *,
    profile: str | Profile = "single_cell_human",
) -> Report:
    """Check whether a dataset is ready for a downstream analysis task.

    Runs task-specific preflight checks (required columns, recommended columns,
    data quality) and returns a Report with the issues found.

    Parameters
    ----------
    data : AnnData or DataFrame
        The dataset to check.
    task : str or TaskProfile
        Name of a built-in task ('clustering', 'differential_expression',
        'integration', 'cell_type_annotation') or a custom TaskProfile.
    profile : str or Profile
        Metadata profile for context (currently unused by preflight checks
        but accepted for API consistency).

    Returns
    -------
    Report
        A report where ``issues`` contains preflight findings and ``cleaned``
        is the unmodified input DataFrame.
    """
    obs, source_adata = _extract_obs(data)
    tp = resolve_task(task)
    issues = run_preflight(obs, tp, adata=source_adata)
    return Report(
        cleaned=obs,
        issues=issues,
        changes=[],
        profile_name=f"preflight:{tp.name}",
        validation_level="preflight",
        adata=source_adata,
    )


# ── Backward-compatible aliases ──────────────────────────────────────────────


def clean_obs(
    data: ObsData,
    profile: str | Profile = "single_cell_human",
    *,
    column_map: dict[str, str] | None = None,
    value_maps: dict[str, dict[str, str]] | None = None,
    validation: str = "standard",
    copy: bool = True,
) -> Report:
    """Clean and validate obs metadata.  Alias for :func:`repair`.

    Always returns a Report (not a tuple), for backward compatibility.
    """
    result = repair(
        data,
        profile=profile,
        column_map=column_map,
        value_maps=value_maps,
        validation=validation,
        copy=copy,
    )
    if isinstance(result, tuple):
        return result[1]  # Return the Report, not the AnnData
    return result


def validate_obs(
    data: ObsData,
    profile: str | Profile = "single_cell_human",
    *,
    level: str = "standard",
) -> Report:
    """Validate obs metadata without modifying it.  Alias for :func:`validate`."""
    return validate(data, profile=profile, level=level)
