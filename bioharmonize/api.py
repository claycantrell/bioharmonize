from __future__ import annotations

import re

import pandas as pd

from .changes import Change
from .issues import Issue
from .preflight import TaskProfile, resolve_task, run_preflight
from .profiles import Profile, resolve_profile
from .report import Report
from .validators import run_validation


def _normalize_col_name(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def clean_obs(
    df: pd.DataFrame,
    profile: str | Profile = "single_cell_human",
    *,
    column_map: dict[str, str] | None = None,
    value_maps: dict[str, dict[str, str]] | None = None,
    validation: str = "standard",
    copy: bool = True,
) -> Report:
    prof = resolve_profile(profile)
    result = df.copy() if copy else df
    changes: list[Change] = []
    issues: list[Issue] = []

    # -- step 1: build rename map --
    rename_map: dict[str, str] = {}

    # Build a lookup from normalized alias -> canonical
    alias_lookup: dict[str, str] = {
        _normalize_col_name(k): v for k, v in prof.column_aliases.items()
    }

    # Explicit user mappings first (highest priority)
    if column_map:
        for src, dst in column_map.items():
            if src in result.columns:
                rename_map[src] = dst

    # Built-in aliases for unmatched columns
    canonical_names = set(prof.canonical_columns)
    already_targeted = set(rename_map.values()) | (set(result.columns) & canonical_names)

    for col in result.columns:
        if col in rename_map:
            continue
        normed = _normalize_col_name(col)
        # Already a canonical name (exact match)
        if normed in canonical_names and normed == col:
            continue
        # Normalized form matches a canonical name (e.g. "cell type" -> "cell_type")
        if normed in canonical_names and normed != col:
            target = normed
        else:
            # Check alias
            target = alias_lookup.get(normed)
        if target is None:
            # Also try exact match on original name
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

    # Apply renames
    if rename_map:
        result = result.rename(columns=rename_map)
        for src, dst in rename_map.items():
            changes.append(Change(kind="rename_column", column=dst, before=src, after=dst))

    # -- step 2: normalize values --
    all_normalizers = dict(prof.value_normalizers)
    if value_maps:
        from .normalizers import make_map_normalizer

        for col_name, vmap in value_maps.items():
            all_normalizers[col_name] = make_map_normalizer(vmap)

    for col_name, normalizer in all_normalizers.items():
        if col_name not in result.columns:
            continue
        # Skip duplicated columns (already flagged in validation)
        if result.columns.duplicated().any() and isinstance(result[col_name], pd.DataFrame):
            continue
        series = result[col_name]
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
        result[col_name] = new_values

    # -- step 3: coerce dtypes --
    for col_name, spec in prof.canonical_columns.items():
        if col_name not in result.columns:
            continue
        # Skip duplicated columns (already flagged in validation)
        if result.columns.duplicated().any() and isinstance(result[col_name], pd.DataFrame):
            continue
        series = result[col_name]
        old_dtype = str(series.dtype)

        if spec.dtype in ("string", "categorical") and not pd.api.types.is_string_dtype(series):
            if pd.api.types.is_numeric_dtype(series):
                # Convert numeric to string, dropping .0 for ints
                result[col_name] = series.apply(
                    lambda x: str(int(x)) if pd.notna(x) and float(x) == int(x) else (str(x) if pd.notna(x) else x)
                )
                changes.append(
                    Change(kind="coerce_dtype", column=col_name, before=old_dtype, after="string")
                )
        elif spec.dtype == "integer" and not pd.api.types.is_integer_dtype(series):
            try:
                result[col_name] = pd.to_numeric(series, errors="raise").astype("Int64")
                changes.append(
                    Change(kind="coerce_dtype", column=col_name, before=old_dtype, after="Int64")
                )
            except (ValueError, TypeError):
                pass
        elif spec.dtype == "float" and not pd.api.types.is_float_dtype(series):
            try:
                result[col_name] = pd.to_numeric(series, errors="raise").astype("Float64")
                changes.append(
                    Change(kind="coerce_dtype", column=col_name, before=old_dtype, after="Float64")
                )
            except (ValueError, TypeError):
                pass

    # -- step 4: validate --
    validation_issues = run_validation(result, prof, level=validation)
    issues.extend(validation_issues)

    return Report(
        cleaned=result,
        issues=issues,
        changes=changes,
        profile_name=prof.name,
        validation_level=validation,
    )


def validate_obs(
    df: pd.DataFrame,
    profile: str | Profile = "single_cell_human",
    *,
    level: str = "standard",
) -> Report:
    prof = resolve_profile(profile)
    cleaned = df.copy()
    issues = run_validation(cleaned, prof, level=level)
    return Report(
        cleaned=cleaned,
        issues=issues,
        changes=[],
        profile_name=prof.name,
        validation_level=level,
    )


def preflight(
    df: pd.DataFrame,
    task: str | TaskProfile,
) -> Report:
    """Check whether a DataFrame is ready for a downstream analysis task.

    Runs task-specific preflight checks (required columns, recommended columns,
    data quality) and returns a Report with the issues found.

    Parameters
    ----------
    df : pd.DataFrame
        The obs DataFrame to check.
    task : str or TaskProfile
        Name of a built-in task ('clustering', 'differential_expression',
        'integration', 'cell_type_annotation') or a custom TaskProfile.

    Returns
    -------
    Report
        A report where ``issues`` contains preflight findings and ``cleaned``
        is the unmodified input DataFrame.
    """
    tp = resolve_task(task)
    issues = run_preflight(df, tp)
    return Report(
        cleaned=df,
        issues=issues,
        changes=[],
        profile_name=f"preflight:{tp.name}",
        validation_level="preflight",
    )
