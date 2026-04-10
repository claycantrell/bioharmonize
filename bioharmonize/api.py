from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Union

import pandas as pd

from .changes import Change
from .issues import Issue
from .profiles import Profile, resolve_profile
from .report import Report
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
) -> Report:
    """Rename columns, normalise values, coerce dtypes, then validate.

    When *data* is an ``AnnData``, the returned ``Report.adata`` holds the
    repaired ``AnnData`` object.
    """
    prof = resolve_profile(profile)
    obs, source_adata = _extract_obs(data)
    result = obs.copy() if copy else obs
    changes: list[Change] = []
    issues: list[Issue] = []

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

    return Report(
        cleaned=result,
        issues=issues,
        changes=changes,
        profile_name=prof.name,
        validation_level=validation,
        adata=repaired_adata,
    )


def preflight(
    data: ObsData,
    *,
    profile: str | Profile = "single_cell_human",
    column_map: dict[str, str] | None = None,
    value_maps: dict[str, dict[str, str]] | None = None,
    validation: str = "standard",
) -> Report:
    """Dry-run of :func:`repair` — shows planned changes without modifying data.

    Returns a Report whose ``cleaned`` holds the *repaired* preview (always a
    copy) and whose ``changes`` / ``issues`` match what ``repair`` would produce.
    The original *data* is never modified.
    """
    return repair(
        data,
        profile=profile,
        column_map=column_map,
        value_maps=value_maps,
        validation=validation,
        copy=True,
    )


# ── Backward-compatible aliases ──────────────────────────────────────────────


def clean_obs(
    df: pd.DataFrame,
    profile: str | Profile = "single_cell_human",
    *,
    column_map: dict[str, str] | None = None,
    value_maps: dict[str, dict[str, str]] | None = None,
    validation: str = "standard",
    copy: bool = True,
) -> Report:
    """Clean and validate obs metadata.  Alias for :func:`repair`."""
    return repair(
        df,
        profile=profile,
        column_map=column_map,
        value_maps=value_maps,
        validation=validation,
        copy=copy,
    )


def validate_obs(
    df: pd.DataFrame,
    profile: str | Profile = "single_cell_human",
    *,
    level: str = "standard",
) -> Report:
    """Validate obs metadata without modifying it.  Alias for :func:`validate`."""
    return validate(df, profile=profile, level=level)
