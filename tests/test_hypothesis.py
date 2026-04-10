"""Hypothesis property-based tests for bioharmonize.

Generates random metadata tables with ugly capitalization, whitespace, synonym
variants, duplicates, and mixed nulls — then asserts the pipeline never crashes,
always returns a valid Report structure, preserves row count, and never silently
drops columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import bioharmonize as bh
from bioharmonize.changes import Change
from bioharmonize.issues import Issue
from bioharmonize.report import Report

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Canonical column names that bioharmonize knows about.
CANONICAL_COLUMNS = [
    "cell_type", "condition", "sample_id", "donor_id",
    "batch_id", "tissue", "sex", "assay", "species", "disease", "platform",
]

# Alias names that map to canonical names.
ALIAS_COLUMNS = [
    "celltype", "cell_type_annotation", "annotation",
    "disease_status", "status", "patient", "donor", "sample",
    "batch", "organ", "gender",
]

# Known synonym values that the normalizers handle.
SEX_SYNONYMS = ["m", "male", "f", "female", "unknown"]
CONDITION_SYNONYMS = ["ctrl", "control", "healthy", "case", "disease"]
ASSAY_SYNONYMS = ["10x", "10x genomics", "10x 3'", "10x 5'", "smart-seq2", "smartseq2"]

ALL_KNOWN_COLUMNS = CANONICAL_COLUMNS + ALIAS_COLUMNS


def _mangle_case(s: str, draw: st.DrawFn) -> str:
    """Randomly mangle the casing of a string."""
    style = draw(st.sampled_from(["lower", "upper", "title", "random", "original"]))
    if style == "lower":
        return s.lower()
    if style == "upper":
        return s.upper()
    if style == "title":
        return s.title()
    if style == "random":
        return "".join(
            draw(st.sampled_from([c.lower(), c.upper()])) for c in s
        )
    return s


def _add_whitespace(s: str, draw: st.DrawFn) -> str:
    """Randomly add leading/trailing whitespace to a string."""
    pad = draw(st.sampled_from(["", " ", "  ", "\t", " \t"]))
    side = draw(st.sampled_from(["left", "right", "both", "none"]))
    if side == "left":
        return pad + s
    if side == "right":
        return s + pad
    if side == "both":
        return pad + s + pad
    return s


@st.composite
def ugly_column_name(draw: st.DrawFn) -> str:
    """Generate a column name: either a known name with ugly formatting,
    or a completely random column name."""
    choice = draw(st.sampled_from(["known", "random"]))
    if choice == "known":
        base = draw(st.sampled_from(ALL_KNOWN_COLUMNS))
        name = _mangle_case(base, draw)
        name = _add_whitespace(name, draw)
        # Occasionally replace underscores with hyphens or spaces.
        if draw(st.booleans()):
            replacement = draw(st.sampled_from(["-", " ", "  "]))
            name = name.replace("_", replacement)
        return name
    else:
        return draw(st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_- "),
            min_size=1, max_size=20,
        ))


@st.composite
def cell_value(draw: st.DrawFn) -> str | float | None:
    """Generate a cell value: known synonym, ugly variant, random string, or null."""
    choice = draw(st.sampled_from(["synonym", "ugly_synonym", "random", "null", "numeric"]))
    if choice == "synonym":
        return draw(st.sampled_from(
            SEX_SYNONYMS + CONDITION_SYNONYMS + ASSAY_SYNONYMS
            + ["T cell", "B cell", "monocyte", "brain", "liver", "human"]
        ))
    if choice == "ugly_synonym":
        base = draw(st.sampled_from(SEX_SYNONYMS + CONDITION_SYNONYMS + ASSAY_SYNONYMS))
        return _add_whitespace(_mangle_case(base, draw), draw)
    if choice == "random":
        return draw(st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_- "),
            min_size=0, max_size=30,
        ))
    if choice == "numeric":
        return draw(st.one_of(
            st.integers(min_value=-100, max_value=100).map(float),
            st.floats(min_value=-1e6, max_value=1e6),
            st.just(float("nan")),
        ))
    # null
    return draw(st.sampled_from([None, np.nan, float("nan")]))


@st.composite
def metadata_column(draw: st.DrawFn, n_rows: int) -> list:
    """Generate a full column of values for a metadata table."""
    # Decide column strategy: all-null, mostly-null, mixed, uniform
    strategy = draw(st.sampled_from(["mixed", "mostly_null", "all_null", "uniform"]))
    if strategy == "all_null":
        return [None] * n_rows
    if strategy == "uniform":
        val = draw(cell_value())
        return [val] * n_rows
    if strategy == "mostly_null":
        null_rate = draw(st.floats(min_value=0.7, max_value=0.99))
        return [
            None if draw(st.floats(min_value=0, max_value=1)) < null_rate
            else draw(cell_value())
            for _ in range(n_rows)
        ]
    # mixed: arbitrary values
    return [draw(cell_value()) for _ in range(n_rows)]


@st.composite
def metadata_table(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a random metadata table with ugly columns, values, and structure."""
    n_rows = draw(st.integers(min_value=0, max_value=50))
    n_cols = draw(st.integers(min_value=0, max_value=15))

    columns: dict[str, list] = {}
    col_names_seen: list[str] = []

    for _ in range(n_cols):
        name = draw(ugly_column_name())
        # Allow duplicate column names sometimes.
        if draw(st.integers(min_value=0, max_value=9)) == 0 and col_names_seen:
            name = draw(st.sampled_from(col_names_seen))
        col_names_seen.append(name)
        columns[name] = draw(metadata_column(n_rows=n_rows))

    # Build DataFrame; handle duplicate column names via from_dict workaround.
    if not columns:
        if n_rows == 0:
            return pd.DataFrame()
        return pd.DataFrame(index=[f"cell_{i}" for i in range(n_rows)])

    # If there are duplicate column names, build via concat.
    if len(set(col_names_seen)) != len(col_names_seen):
        series_list = []
        for name in col_names_seen:
            series_list.append(pd.Series(columns[name], name=name))
        df = pd.concat(series_list, axis=1)
    else:
        df = pd.DataFrame(columns)

    # Set index to cell IDs or leave as integer (another ugly edge case).
    if n_rows > 0:
        idx_style = draw(st.sampled_from(["cell_ids", "integer", "duplicates", "whitespace"]))
        if idx_style == "cell_ids":
            df.index = pd.Index([f"cell_{i}" for i in range(n_rows)])
        elif idx_style == "integer":
            pass  # Leave as default RangeIndex
        elif idx_style == "duplicates":
            base_ids = [f"cell_{i % max(1, n_rows // 2)}" for i in range(n_rows)]
            df.index = pd.Index(base_ids)
        elif idx_style == "whitespace":
            df.index = pd.Index([f"  cell_{i} " for i in range(n_rows)])

    return df


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestRepairNeverCrashes:
    """repair() must never raise on arbitrary metadata tables."""

    @given(df=metadata_table())
    @settings(
        max_examples=200,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_repair_does_not_crash(self, df: pd.DataFrame):
        report = bh.repair(df)
        # repair returns Report for DataFrame input
        assert isinstance(report, Report)

    @given(df=metadata_table())
    @settings(
        max_examples=200,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_inspect_does_not_crash(self, df: pd.DataFrame):
        report = bh.inspect(df)
        assert isinstance(report, Report)

    @given(df=metadata_table())
    @settings(
        max_examples=200,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_validate_does_not_crash(self, df: pd.DataFrame):
        report = bh.validate(df)
        assert isinstance(report, Report)

    @given(
        df=metadata_table(),
        level=st.sampled_from(["minimal", "standard", "strict"]),
    )
    @settings(
        max_examples=200,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_validate_all_levels_no_crash(self, df: pd.DataFrame, level: str):
        report = bh.validate(df, level=level)
        assert isinstance(report, Report)


class TestReportStructure:
    """Reports must always have valid structure."""

    @given(df=metadata_table())
    @settings(
        max_examples=200,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_report_has_valid_fields(self, df: pd.DataFrame):
        report = bh.repair(df)
        # cleaned is always a DataFrame
        assert isinstance(report.cleaned, pd.DataFrame)
        # issues is always a list of Issue
        assert isinstance(report.issues, list)
        for issue in report.issues:
            assert isinstance(issue, Issue)
            assert issue.severity in ("info", "warning", "error")
            assert isinstance(issue.code, str) and len(issue.code) > 0
            assert isinstance(issue.message, str) and len(issue.message) > 0
        # changes is always a list of Change
        assert isinstance(report.changes, list)
        for change in report.changes:
            assert isinstance(change, Change)
            assert change.kind in (
                "rename_column", "normalize_value", "coerce_dtype",
                "drop_column", "fill_missing", "deduplicate_index",
                "strip_index", "generate_index",
            )
            assert isinstance(change.column, str) and len(change.column) > 0
        # profile_name and validation_level are non-empty strings
        assert isinstance(report.profile_name, str) and len(report.profile_name) > 0
        assert isinstance(report.validation_level, str) and len(report.validation_level) > 0

    @given(df=metadata_table())
    @settings(
        max_examples=100,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_summary_does_not_crash(self, df: pd.DataFrame):
        report = bh.repair(df)
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    @given(df=metadata_table())
    @settings(
        max_examples=100,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_issues_frame_does_not_crash(self, df: pd.DataFrame):
        report = bh.repair(df)
        issues_df = report.issues_frame()
        assert isinstance(issues_df, pd.DataFrame)
        if report.issues:
            assert len(issues_df) == len(report.issues)

    @given(df=metadata_table())
    @settings(
        max_examples=100,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_changes_frame_does_not_crash(self, df: pd.DataFrame):
        report = bh.repair(df)
        changes_df = report.changes_frame()
        assert isinstance(changes_df, pd.DataFrame)
        if report.changes:
            assert len(changes_df) == len(report.changes)


class TestRowCountPreserved:
    """The pipeline must never add or remove rows."""

    @given(df=metadata_table())
    @settings(
        max_examples=200,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_repair_preserves_row_count(self, df: pd.DataFrame):
        n_rows_before = len(df)
        report = bh.repair(df)
        assert len(report.cleaned) == n_rows_before

    @given(df=metadata_table())
    @settings(
        max_examples=200,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_inspect_preserves_row_count(self, df: pd.DataFrame):
        n_rows_before = len(df)
        report = bh.inspect(df)
        assert len(report.cleaned) == n_rows_before

    @given(df=metadata_table())
    @settings(
        max_examples=200,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_validate_preserves_row_count(self, df: pd.DataFrame):
        n_rows_before = len(df)
        report = bh.validate(df)
        assert len(report.cleaned) == n_rows_before


class TestNoSilentColumnDrops:
    """Columns must never be silently dropped — all input columns should appear
    in the output (possibly under a different name)."""

    @given(df=metadata_table())
    @settings(
        max_examples=200,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_repair_no_silent_column_drops(self, df: pd.DataFrame):
        input_cols = set(df.columns)
        report = bh.repair(df)
        output_cols = set(report.cleaned.columns)

        # Build the set of columns that were renamed.
        renamed_from = set()
        renamed_to = set()
        for change in report.changes:
            if change.kind == "rename_column":
                renamed_from.add(change.before)
                renamed_to.add(change.after)

        # Every input column must appear in output either under its original
        # name or as a rename target. Duplicate column names are an edge case
        # handled by the DUPLICATE_COLUMNS validation issue, so we skip
        # the check when duplicates exist.
        if not df.columns.duplicated().any():
            for col in input_cols:
                assert col in output_cols or col in renamed_from, (
                    f"Column {col!r} disappeared without a rename change record"
                )

    @given(df=metadata_table())
    @settings(
        max_examples=200,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_inspect_no_column_drops(self, df: pd.DataFrame):
        """inspect() should never modify columns at all."""
        input_cols = list(df.columns)
        report = bh.inspect(df)
        output_cols = list(report.cleaned.columns)
        assert input_cols == output_cols

    @given(df=metadata_table())
    @settings(
        max_examples=200,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_validate_no_column_drops(self, df: pd.DataFrame):
        """validate() should never modify columns at all."""
        input_cols = list(df.columns)
        report = bh.validate(df)
        output_cols = list(report.cleaned.columns)
        assert input_cols == output_cols


class TestInputNotMutated:
    """The original DataFrame must not be mutated (copy=True is default)."""

    @given(df=metadata_table())
    @settings(
        max_examples=100,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_repair_does_not_mutate_input(self, df: pd.DataFrame):
        original_cols = list(df.columns)
        original_len = len(df)
        bh.repair(df)
        assert list(df.columns) == original_cols
        assert len(df) == original_len

    @given(df=metadata_table())
    @settings(
        max_examples=100,
        deadline=5000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_inspect_does_not_mutate_input(self, df: pd.DataFrame):
        original_cols = list(df.columns)
        original_len = len(df)
        bh.inspect(df)
        assert list(df.columns) == original_cols
        assert len(df) == original_len
