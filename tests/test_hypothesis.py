"""Hypothesis property-based tests for bioharmonize.

Generates random metadata tables with ugly capitalization, whitespace,
synonym variants, duplicates, and mixed nulls — then asserts:
  1. No crashes (all API functions return without exception)
  2. Valid report structure (Report has expected fields/types)
  3. Row count preserved (repair never drops or adds rows)
  4. No silent column drops (all input columns survive repair)
"""

from __future__ import annotations

import string

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import bioharmonize as bh

# ---------------------------------------------------------------------------
# Strategies for generating ugly metadata
# ---------------------------------------------------------------------------

# Canonical column names the profile knows about
CANONICAL_COLS = [
    "cell_type", "condition", "sample_id", "donor_id",
    "batch_id", "tissue", "sex", "assay", "species", "disease", "platform",
]

# Known aliases that map to canonical names
ALIAS_COLS = {
    "celltype": "cell_type",
    "cell_type_annotation": "cell_type",
    "annotation": "cell_type",
    "disease_status": "condition",
    "status": "condition",
    "patient": "donor_id",
    "donor": "donor_id",
    "sample": "sample_id",
    "batch": "batch_id",
    "organ": "tissue",
    "gender": "sex",
}

# Values with known normalizations
SEX_VARIANTS = ["m", "f", "male", "female", "unknown", "M", "F", "MALE", "FEMALE"]
CONDITION_VARIANTS = ["ctrl", "control", "healthy", "case", "disease", "CTRL", "Control"]
ASSAY_VARIANTS = ["10x", "10x genomics", "smart-seq2", "smartseq2", "10x 3'", "10x 5'"]

# Arbitrary values for columns without normalizers
FREETEXT_VALUES = ["alpha", "beta", "gamma", "test_1", "sample_A", "lung", "brain", "kidney"]


def _mangle_case(s: str) -> st.SearchStrategy[str]:
    """Strategy that mangles capitalization of a string."""
    return st.sampled_from([
        s,
        s.upper(),
        s.lower(),
        s.title(),
        s.swapcase(),
        s[0].upper() + s[1:] if len(s) > 1 else s.upper(),
    ])


# Strategy: a column name that is either canonical, alias, or random noise
_canonical_col = st.sampled_from(CANONICAL_COLS)
_alias_col = st.sampled_from(list(ALIAS_COLS.keys()))
_noise_col = st.from_regex(r"[a-z][a-z0-9_]{2,15}", fullmatch=True).filter(
    lambda s: s not in CANONICAL_COLS and s not in ALIAS_COLS
)

_ugly_col_name = st.one_of(
    # Canonical with ugly case/whitespace
    _canonical_col.flatmap(lambda c: st.sampled_from([
        c.upper(),
        c.title(),
        f"  {c}  ",
        f" {c.upper()} ",
        c.replace("_", " "),
        c.replace("_", "-"),
        c.replace("_", "  "),
    ])),
    # Alias with ugly case/whitespace
    _alias_col.flatmap(lambda c: st.sampled_from([
        c,
        c.upper(),
        c.title(),
        f"  {c}",
        f"{c}  ",
    ])),
    # Random noise column
    _noise_col,
)


def _cell_value_for_col(col_name: str) -> st.SearchStrategy:
    """Return a strategy that produces values appropriate to a column name."""
    normed = col_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normed in ("sex", "gender"):
        return st.one_of(
            st.sampled_from(SEX_VARIANTS),
            st.none(),
        )
    if normed in ("condition", "status", "disease_status"):
        return st.one_of(
            st.sampled_from(CONDITION_VARIANTS),
            st.none(),
        )
    if normed == "assay":
        return st.one_of(
            st.sampled_from(ASSAY_VARIANTS),
            st.none(),
        )
    # Generic: mix of freetext, None, and ugly whitespace strings
    return st.one_of(
        st.sampled_from(FREETEXT_VALUES),
        st.none(),
        st.text(alphabet=string.ascii_letters + string.digits + " _", min_size=1, max_size=20),
    )


@st.composite
def ugly_metadata_table(draw, min_rows=1, max_rows=30, min_cols=1, max_cols=12):
    """Generate a DataFrame with ugly column names, mixed values, and nulls."""
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate column names — allow duplicates (that's part of "ugly")
    col_names = draw(st.lists(_ugly_col_name, min_size=n_cols, max_size=n_cols))

    # Build column data
    data = {}
    for i, col_name in enumerate(col_names):
        # Use a unique key for dict (pandas handles duplicate column names)
        values = draw(st.lists(
            _cell_value_for_col(col_name),
            min_size=n_rows,
            max_size=n_rows,
        ))
        data[f"__col_{i}"] = values

    df = pd.DataFrame(data)
    # Now set actual column names (may have duplicates)
    df.columns = col_names
    # Generate cell-style index
    df.index = pd.Index([f"cell_{i}" for i in range(n_rows)])
    return df


@st.composite
def ugly_metadata_table_no_dup_cols(draw, min_rows=1, max_rows=30, min_cols=1, max_cols=10):
    """Generate a DataFrame with ugly column names but NO duplicate columns."""
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    col_names = draw(st.lists(
        _ugly_col_name, min_size=n_cols, max_size=n_cols, unique=True,
    ))

    data = {}
    for col_name in col_names:
        values = draw(st.lists(
            _cell_value_for_col(col_name),
            min_size=n_rows,
            max_size=n_rows,
        ))
        data[col_name] = values

    df = pd.DataFrame(data)
    df.index = pd.Index([f"cell_{i}" for i in range(n_rows)])
    return df


# Common settings: suppress slow-test health check since we're doing real work
COMMON_SETTINGS = dict(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)


# ---------------------------------------------------------------------------
# 1. No crashes — all three API functions handle ugly input gracefully
# ---------------------------------------------------------------------------


class TestNoCrashes:
    """All API entry points must return without exception on any metadata table."""

    @given(df=ugly_metadata_table())
    @settings(**COMMON_SETTINGS)
    def test_repair_never_crashes(self, df: pd.DataFrame):
        bh.repair(df)

    @given(df=ugly_metadata_table())
    @settings(**COMMON_SETTINGS)
    def test_validate_never_crashes(self, df: pd.DataFrame):
        bh.validate(df)

    @given(df=ugly_metadata_table())
    @settings(**COMMON_SETTINGS)
    def test_inspect_never_crashes(self, df: pd.DataFrame):
        bh.inspect(df)

    @given(df=ugly_metadata_table())
    @settings(**COMMON_SETTINGS)
    def test_validate_all_levels_never_crash(self, df: pd.DataFrame):
        for level in ("minimal", "standard", "strict"):
            bh.validate(df, level=level)

    @given(df=ugly_metadata_table(min_rows=0, max_rows=0, min_cols=0, max_cols=8))
    @settings(**COMMON_SETTINGS)
    def test_empty_rows_never_crash(self, df: pd.DataFrame):
        bh.repair(df)
        bh.validate(df)
        bh.inspect(df)

    @given(df=ugly_metadata_table(min_cols=0, max_cols=0))
    @settings(**COMMON_SETTINGS)
    def test_empty_cols_never_crash(self, df: pd.DataFrame):
        bh.repair(df)
        bh.validate(df)
        bh.inspect(df)


# ---------------------------------------------------------------------------
# 2. Valid report structure
# ---------------------------------------------------------------------------


class TestValidReportStructure:
    """Reports must have the expected types and structure."""

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_repair_report_structure(self, df: pd.DataFrame):
        report = bh.repair(df)
        assert isinstance(report, bh.Report)
        assert isinstance(report.cleaned, pd.DataFrame)
        assert isinstance(report.issues, list)
        assert isinstance(report.changes, list)
        assert isinstance(report.profile_name, str)
        assert report.profile_name != ""

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_validate_report_structure(self, df: pd.DataFrame):
        report = bh.validate(df)
        assert isinstance(report, bh.Report)
        assert isinstance(report.cleaned, pd.DataFrame)
        assert isinstance(report.issues, list)
        assert isinstance(report.changes, list)

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_inspect_report_structure(self, df: pd.DataFrame):
        report = bh.inspect(df)
        assert isinstance(report, bh.Report)
        assert isinstance(report.cleaned, pd.DataFrame)
        assert isinstance(report.issues, list)
        assert isinstance(report.changes, list)

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_issues_have_valid_severity(self, df: pd.DataFrame):
        report = bh.repair(df)
        valid_severities = {"info", "warning", "error"}
        for issue in report.issues:
            assert isinstance(issue, bh.Issue)
            assert issue.severity in valid_severities, (
                f"Invalid severity {issue.severity!r}"
            )
            assert isinstance(issue.code, str)
            assert isinstance(issue.message, str)

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_changes_have_valid_kinds(self, df: pd.DataFrame):
        report = bh.repair(df)
        valid_kinds = {
            "rename_column", "normalize_value", "coerce_dtype",
            "drop_column", "fill_missing", "deduplicate_index",
            "strip_index", "generate_index",
        }
        for change in report.changes:
            assert isinstance(change, bh.Change)
            assert change.kind in valid_kinds, (
                f"Invalid change kind {change.kind!r}"
            )

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_report_summary_does_not_crash(self, df: pd.DataFrame):
        report = bh.repair(df)
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_report_frames_do_not_crash(self, df: pd.DataFrame):
        report = bh.repair(df)
        issues_df = report.issues_frame()
        changes_df = report.changes_frame()
        assert isinstance(issues_df, pd.DataFrame)
        assert isinstance(changes_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# 3. Row count preserved — repair never drops or adds rows
# ---------------------------------------------------------------------------


class TestRowCountPreserved:
    """The number of rows must be identical before and after repair."""

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_repair_preserves_row_count(self, df: pd.DataFrame):
        report = bh.repair(df)
        assert len(report.cleaned) == len(df), (
            f"Row count changed: {len(df)} -> {len(report.cleaned)}"
        )

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_inspect_preserves_row_count(self, df: pd.DataFrame):
        report = bh.inspect(df)
        assert len(report.cleaned) == len(df)

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_validate_preserves_row_count(self, df: pd.DataFrame):
        report = bh.validate(df)
        assert len(report.cleaned) == len(df)

    @given(df=ugly_metadata_table_no_dup_cols(min_rows=1, max_rows=50))
    @settings(**COMMON_SETTINGS)
    def test_repair_row_count_with_all_nulls_column(self, df: pd.DataFrame):
        """A column that is entirely null should not cause row loss."""
        df = df.copy()
        df["all_null_col"] = None
        report = bh.repair(df)
        assert len(report.cleaned) == len(df)

    @given(df=ugly_metadata_table_no_dup_cols(min_rows=1, max_rows=50))
    @settings(**COMMON_SETTINGS)
    def test_repair_row_count_with_mixed_nulls(self, df: pd.DataFrame):
        """Sparse nulls scattered across columns should not cause row loss."""
        report = bh.repair(df)
        assert len(report.cleaned) == len(df)


# ---------------------------------------------------------------------------
# 4. No silent column drops — all input columns survive repair
# ---------------------------------------------------------------------------


class TestNoSilentColumnDrops:
    """Repair must never silently drop columns. Columns may be renamed but
    the total count must be >= input count (renames are 1:1, and nothing
    gets deleted without the user asking for it)."""

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_repair_column_count_preserved(self, df: pd.DataFrame):
        report = bh.repair(df)
        assert len(report.cleaned.columns) >= len(df.columns), (
            f"Columns dropped: input had {len(df.columns)}, "
            f"output has {len(report.cleaned.columns)}. "
            f"Input cols: {list(df.columns)}, Output cols: {list(report.cleaned.columns)}"
        )

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_unrenamed_columns_survive(self, df: pd.DataFrame):
        """Columns that are not renamed must still be present in output."""
        report = bh.repair(df)
        renamed_sources = {c.before for c in report.changes if c.kind == "rename_column"}
        renamed_targets = {c.after for c in report.changes if c.kind == "rename_column"}
        for col in df.columns:
            if col not in renamed_sources:
                assert col in report.cleaned.columns, (
                    f"Column {col!r} was neither renamed nor preserved in output"
                )

    @given(df=ugly_metadata_table_no_dup_cols())
    @settings(**COMMON_SETTINGS)
    def test_renamed_targets_present(self, df: pd.DataFrame):
        """Every rename target must appear in the output columns."""
        report = bh.repair(df)
        for change in report.changes:
            if change.kind == "rename_column":
                assert change.after in report.cleaned.columns, (
                    f"Rename target {change.after!r} missing from output"
                )

    @given(df=ugly_metadata_table_no_dup_cols(min_cols=3, max_cols=10))
    @settings(**COMMON_SETTINGS)
    def test_noise_columns_not_dropped(self, df: pd.DataFrame):
        """Columns unknown to the profile should pass through untouched."""
        report = bh.repair(df)
        renamed_sources = {c.before for c in report.changes if c.kind == "rename_column"}
        for col in df.columns:
            if col not in renamed_sources:
                assert col in report.cleaned.columns, (
                    f"Unknown column {col!r} was silently dropped"
                )


# ---------------------------------------------------------------------------
# 5. Bonus: duplicate column names don't crash
# ---------------------------------------------------------------------------


class TestDuplicateColumnNamesNoCrash:
    """DataFrames with duplicate column names must not crash the API."""

    @given(df=ugly_metadata_table(min_cols=2, max_cols=6))
    @settings(**COMMON_SETTINGS)
    def test_repair_with_possible_duplicates(self, df: pd.DataFrame):
        # ugly_metadata_table may produce duplicate column names
        bh.repair(df)

    @given(df=ugly_metadata_table(min_cols=2, max_cols=6))
    @settings(**COMMON_SETTINGS)
    def test_validate_with_possible_duplicates(self, df: pd.DataFrame):
        bh.validate(df)

    @given(df=ugly_metadata_table(min_cols=2, max_cols=6))
    @settings(**COMMON_SETTINGS)
    def test_inspect_with_possible_duplicates(self, df: pd.DataFrame):
        bh.inspect(df)
