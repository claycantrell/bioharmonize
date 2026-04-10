"""Tests for Report.readiness dict and its display in summary()."""

from __future__ import annotations

import pandas as pd
import pytest

import bioharmonize as bh
from bioharmonize.report import Report


def _make_df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(kwargs, index=[f"cell_{i}" for i in range(len(next(iter(kwargs.values()))))])


# ---------------------------------------------------------------------------
# Report.readiness property
# ---------------------------------------------------------------------------


class TestReadinessProperty:
    def test_returns_dict_with_all_tasks(self):
        df = _make_df(x=[1, 2])
        report = Report(cleaned=df)
        rdns = report.readiness
        assert set(rdns) == {"cell_type_annotation", "clustering", "differential_expression", "integration"}

    def test_all_ready_when_fully_equipped(self):
        """A dataset with all required columns and adequate variation should be ready for everything."""
        df = _make_df(
            condition=["ctrl", "ctrl", "disease", "disease"],
            batch_id=["b1", "b1", "b2", "b2"],
            sample_id=["s1", "s2", "s3", "s4"],
            donor_id=["d1", "d2", "d3", "d4"],
            tissue=["lung"] * 4,
            species=["human"] * 4,
            assay=["10x"] * 4,
        )
        report = Report(cleaned=df)
        rdns = report.readiness
        assert rdns["clustering"] == "ready"
        assert rdns["differential_expression"] == "ready"
        assert rdns["integration"] == "ready"
        assert rdns["cell_type_annotation"] == "ready"

    def test_not_ready_when_required_column_missing(self):
        """DE requires condition; integration requires batch_id."""
        df = _make_df(x=[1, 2])
        report = Report(cleaned=df)
        rdns = report.readiness
        assert rdns["differential_expression"] == "not_ready"
        assert rdns["integration"] == "not_ready"

    def test_warning_when_recommended_missing(self):
        """Clustering has no required cols but recommends batch_id + sample_id."""
        df = _make_df(x=[1, 2])
        report = Report(cleaned=df)
        rdns = report.readiness
        assert rdns["clustering"] == "warning"

    def test_not_ready_beats_warning(self):
        """A task with both errors and warnings should report not_ready."""
        df = _make_df(condition=["ctrl"])  # single group → error, missing sample_id → warning
        report = Report(cleaned=df)
        rdns = report.readiness
        assert rdns["differential_expression"] == "not_ready"

    def test_readiness_uses_cleaned_data(self):
        """Readiness should evaluate against the cleaned DataFrame, not the original."""
        df = _make_df(
            condition=["ctrl", "ctrl", "disease", "disease"],
            sample_id=["s1", "s2", "s3", "s4"],
            donor_id=["d1", "d2", "d3", "d4"],
            batch_id=["b1", "b1", "b2", "b2"],
            tissue=["lung"] * 4,
            species=["human"] * 4,
            assay=["10x"] * 4,
        )
        report = bh.repair(df)
        rdns = report.readiness
        # All tasks should be ready since repair doesn't remove columns
        assert rdns["differential_expression"] == "ready"

    def test_readiness_after_preflight_api(self):
        """Readiness should work on reports returned by bh.preflight()."""
        df = _make_df(
            batch_id=["b1", "b2"],
            sample_id=["s1", "s2"],
            assay=["10x", "10x"],
        )
        report = bh.preflight(df, "integration")
        rdns = report.readiness
        assert rdns["integration"] == "ready"
        # DE still not ready (no condition column)
        assert rdns["differential_expression"] == "not_ready"


# ---------------------------------------------------------------------------
# summary() includes readiness
# ---------------------------------------------------------------------------


class TestReadinessInSummary:
    def test_summary_contains_task_readiness_header(self):
        df = _make_df(x=[1, 2])
        report = Report(cleaned=df)
        text = report.summary()
        assert "task readiness:" in text

    def test_summary_lists_all_tasks(self):
        df = _make_df(x=[1, 2])
        report = Report(cleaned=df)
        text = report.summary()
        assert "clustering:" in text
        assert "differential_expression:" in text
        assert "integration:" in text
        assert "cell_type_annotation:" in text

    def test_summary_shows_correct_statuses(self):
        df = _make_df(
            condition=["ctrl", "ctrl", "disease", "disease"],
            batch_id=["b1", "b1", "b2", "b2"],
            sample_id=["s1", "s2", "s3", "s4"],
            donor_id=["d1", "d2", "d3", "d4"],
            tissue=["lung"] * 4,
            species=["human"] * 4,
            assay=["10x"] * 4,
        )
        report = Report(cleaned=df)
        text = report.summary()
        assert "clustering: ready" in text
        assert "differential_expression: ready" in text
        assert "integration: ready" in text

    def test_summary_shows_not_ready(self):
        df = _make_df(x=[1, 2])
        report = Report(cleaned=df)
        text = report.summary()
        assert "differential_expression: not_ready" in text
        assert "integration: not_ready" in text

    def test_summary_shows_warning(self):
        df = _make_df(x=[1, 2])
        report = Report(cleaned=df)
        text = report.summary()
        # clustering has no required cols → only warnings for missing recommended
        assert "clustering: warning" in text
