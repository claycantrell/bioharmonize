from __future__ import annotations

import re

import pandas as pd

from .issues import Issue
from .profiles import Profile


def _normalize_col_name(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _find_near_misses(columns: list[str], canonical: set[str], aliases: set[str]) -> list[Issue]:
    issues: list[Issue] = []
    known = canonical | aliases
    for col in columns:
        normed = _normalize_col_name(col)
        if normed in known:
            continue
        for canon in canonical:
            if _is_near_miss(normed, canon):
                issues.append(
                    Issue(
                        severity="warning",
                        code="SUSPICIOUS_COLUMN",
                        column=col,
                        message=f"Column {col!r} looks similar to canonical column {canon!r}",
                        suggestion=f"Did you mean {canon!r}? Add it to column_map if so.",
                    )
                )
                break
    return issues


def _is_near_miss(a: str, b: str) -> bool:
    if a == b:
        return False
    # Simple edit-distance-1 check
    if abs(len(a) - len(b)) > 2:
        return False
    # Check if one is a substring of the other
    if a in b or b in a:
        return True
    # Check single character difference
    if len(a) == len(b):
        diffs = sum(1 for ca, cb in zip(a, b) if ca != cb)
        return diffs == 1
    return False


def run_validation(
    df: pd.DataFrame,
    profile: Profile,
    *,
    level: str = "standard",
) -> list[Issue]:
    issues: list[Issue] = []

    # -- minimal checks --
    if df.columns.duplicated().any():
        dupes = list(df.columns[df.columns.duplicated()])
        issues.append(
            Issue(
                severity="error",
                code="DUPLICATE_COLUMNS",
                column=None,
                message=f"Duplicate column names: {dupes}",
            )
        )

    if len(df) == 0:
        issues.append(
            Issue(
                severity="warning",
                code="EMPTY_DATAFRAME",
                column=None,
                message="DataFrame has no rows",
            )
        )

    if df.index.duplicated().any():
        n_dupes = int(df.index.duplicated().sum())
        issues.append(
            Issue(
                severity="warning",
                code="DUPLICATE_INDEX",
                column=None,
                message=f"Index has {n_dupes} duplicate value(s)",
            )
        )

    # dtype checks for canonical columns
    for col_name, spec in profile.canonical_columns.items():
        if col_name not in df.columns:
            continue
        series = df[col_name]
        if spec.dtype in ("string", "categorical") and pd.api.types.is_numeric_dtype(series):
            issues.append(
                Issue(
                    severity="warning",
                    code="UNEXPECTED_DTYPE",
                    column=col_name,
                    message=f"Column {col_name!r} expected {spec.dtype} but got numeric dtype",
                    suggestion=f"Convert to string: df['{col_name}'] = df['{col_name}'].astype(str)",
                )
            )
        elif spec.dtype in ("integer", "float") and pd.api.types.is_string_dtype(series):
            issues.append(
                Issue(
                    severity="warning",
                    code="UNEXPECTED_DTYPE",
                    column=col_name,
                    message=f"Column {col_name!r} expected {spec.dtype} but got string dtype",
                )
            )

    if level == "minimal":
        return issues

    # -- standard checks --
    required = profile.required_columns(level)
    for col_name in required:
        if col_name not in df.columns:
            issues.append(
                Issue(
                    severity="error",
                    code="MISSING_REQUIRED_COLUMN",
                    column=col_name,
                    message=f"Required column {col_name!r} is missing",
                )
            )

    # near-miss column detection
    canonical_names = set(profile.canonical_columns)
    alias_names = set(profile.column_aliases)
    normed_current = {_normalize_col_name(c) for c in df.columns}
    # Only flag near misses for columns not already matched
    unmatched = [
        c
        for c in df.columns
        if _normalize_col_name(c) not in canonical_names
        and _normalize_col_name(c) not in alias_names
    ]
    issues.extend(_find_near_misses(unmatched, canonical_names, alias_names))

    # mixed vocabulary warnings for normalized columns
    for col_name, normalizer in profile.value_normalizers.items():
        if col_name not in df.columns:
            continue
        series = df[col_name].dropna()
        if series.empty:
            continue
        unique_vals = series.unique()
        normalized_set: set[str | None] = set()
        raw_set: set[str] = set()
        for v in unique_vals:
            s = str(v)
            n = normalizer(s)
            if n is not None:
                normalized_set.add(n)
            else:
                raw_set.add(s)
        if raw_set and normalized_set:
            issues.append(
                Issue(
                    severity="warning",
                    code="MIXED_VOCABULARY",
                    column=col_name,
                    message=(
                        f"Column {col_name!r} has a mix of recognized values "
                        f"({sorted(normalized_set)}) and unrecognized values ({sorted(raw_set)})"
                    ),
                )
            )

    if level != "strict":
        return issues

    # -- strict checks --
    for col_name, spec in profile.canonical_columns.items():
        if col_name not in df.columns:
            continue
        if not spec.nullable:
            n_null = int(df[col_name].isna().sum())
            if n_null > 0:
                issues.append(
                    Issue(
                        severity="error",
                        code="NULL_IN_NON_NULLABLE",
                        column=col_name,
                        message=f"Column {col_name!r} has {n_null} null value(s) but is non-nullable",
                        row_count=n_null,
                    )
                )
        if spec.allowed_values is not None:
            series = df[col_name].dropna()
            if not series.empty:
                bad = set(series.unique()) - set(spec.allowed_values)
                if bad:
                    issues.append(
                        Issue(
                            severity="error",
                            code="INVALID_VALUE",
                            column=col_name,
                            message=(
                                f"Column {col_name!r} has values not in allowed set: {sorted(bad)}"
                            ),
                            suggestion=f"Allowed values: {list(spec.allowed_values)}",
                            row_count=int(series.isin(bad).sum()),
                        )
                    )

    # flag unknown values in normalized columns
    for col_name, normalizer in profile.value_normalizers.items():
        if col_name not in df.columns:
            continue
        series = df[col_name].dropna()
        if series.empty:
            continue
        unmapped = []
        for v in series.unique():
            if normalizer(str(v)) is None:
                unmapped.append(str(v))
        if unmapped:
            issues.append(
                Issue(
                    severity="warning",
                    code="UNMAPPED_VALUES",
                    column=col_name,
                    message=f"Column {col_name!r} has values with no normalization mapping: {sorted(unmapped)}",
                    suggestion="Add these to value_maps if they should be normalized.",
                )
            )

    return issues
