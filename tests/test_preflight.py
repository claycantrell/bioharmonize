"""Tests for preflight task profiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import bioharmonize as bh
from bioharmonize.preflight import (
    CELL_TYPE_ANNOTATION,
    CLUSTERING,
    DIFFERENTIAL_EXPRESSION,
    INTEGRATION,
    TaskProfile,
    list_tasks,
    resolve_task,
    run_preflight,
)

try:
    from click.testing import CliRunner

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False


# ---------------------------------------------------------------------------
# Task profile registry
# ---------------------------------------------------------------------------


class TestTaskRegistry:
    def test_list_tasks_returns_all_four(self):
        tasks = list_tasks()
        assert tasks == [
            "cell_type_annotation",
            "clustering",
            "differential_expression",
            "integration",
        ]

    def test_resolve_task_by_name(self):
        tp = resolve_task("clustering")
        assert tp.name == "clustering"

    def test_resolve_task_passthrough(self):
        tp = resolve_task(CLUSTERING)
        assert tp is CLUSTERING

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown task profile"):
            resolve_task("nonexistent")

    def test_task_profiles_are_frozen(self):
        with pytest.raises(AttributeError):
            CLUSTERING.name = "something_else"


# ---------------------------------------------------------------------------
# Public API: bh.preflight()
# ---------------------------------------------------------------------------


class TestPreflightAPI:
    def test_returns_report(self):
        df = pd.DataFrame({"condition": ["ctrl", "disease"], "sample_id": ["s1", "s2"]})
        report = bh.preflight(df, "differential_expression")
        assert isinstance(report, bh.Report)
        assert report.profile_name == "preflight:differential_expression"
        assert report.validation_level == "preflight"

    def test_no_changes_in_report(self):
        df = pd.DataFrame({"batch_id": ["b1", "b2"]})
        report = bh.preflight(df, "integration")
        assert report.changes == []

    def test_cleaned_is_original_df(self):
        df = pd.DataFrame({"x": [1, 2]})
        report = bh.preflight(df, "clustering")
        assert report.cleaned is df


# ---------------------------------------------------------------------------
# Clustering preflight
# ---------------------------------------------------------------------------


class TestClusteringPreflight:
    def test_pass_with_batch_and_sample(self):
        df = pd.DataFrame({
            "batch_id": ["b1", "b1", "b2", "b2"],
            "sample_id": ["s1", "s1", "s2", "s2"],
        })
        issues = run_preflight(df, "clustering")
        assert len(issues) == 0

    def test_warns_missing_batch(self):
        df = pd.DataFrame({"sample_id": ["s1", "s2"]})
        issues = run_preflight(df, "clustering")
        codes = [(i.code, i.column) for i in issues]
        assert ("PREFLIGHT_MISSING_RECOMMENDED", "batch_id") in codes

    def test_warns_missing_sample(self):
        df = pd.DataFrame({"batch_id": ["b1", "b2"]})
        issues = run_preflight(df, "clustering")
        codes = [(i.code, i.column) for i in issues]
        assert ("PREFLIGHT_MISSING_RECOMMENDED", "sample_id") in codes

    def test_single_batch_error(self):
        df = pd.DataFrame({"batch_id": ["b1", "b1", "b1"]})
        issues = run_preflight(df, "clustering")
        codes = [i.code for i in issues]
        assert "SINGLE_BATCH" in codes

    def test_no_required_columns(self):
        """Clustering has no required columns — should only get warnings."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        issues = run_preflight(df, "clustering")
        for issue in issues:
            assert issue.severity in ("warning", "info")


# ---------------------------------------------------------------------------
# Differential expression preflight
# ---------------------------------------------------------------------------


class TestDifferentialExpressionPreflight:
    def test_pass_with_all_columns(self):
        df = pd.DataFrame({
            "condition": ["ctrl", "ctrl", "disease", "disease"],
            "sample_id": ["s1", "s2", "s3", "s4"],
            "donor_id": ["d1", "d2", "d3", "d4"],
        })
        issues = run_preflight(df, "differential_expression")
        assert len(issues) == 0

    def test_error_missing_condition(self):
        df = pd.DataFrame({"sample_id": ["s1", "s2"]})
        issues = run_preflight(df, "differential_expression")
        errors = [i for i in issues if i.severity == "error"]
        assert any(i.code == "PREFLIGHT_MISSING_REQUIRED" and i.column == "condition" for i in errors)

    def test_error_single_group(self):
        df = pd.DataFrame({"condition": ["ctrl", "ctrl", "ctrl"]})
        issues = run_preflight(df, "differential_expression")
        codes = [i.code for i in issues]
        assert "INSUFFICIENT_GROUPS" in codes

    def test_warn_low_replicates(self):
        df = pd.DataFrame({
            "condition": ["ctrl", "disease"],
            "sample_id": ["s1", "s2"],
        })
        issues = run_preflight(df, "differential_expression")
        codes = [i.code for i in issues]
        assert "LOW_REPLICATES" in codes

    def test_no_low_replicate_warn_when_adequate(self):
        df = pd.DataFrame({
            "condition": ["ctrl", "ctrl", "disease", "disease"],
            "sample_id": ["s1", "s2", "s3", "s4"],
        })
        issues = run_preflight(df, "differential_expression")
        codes = [i.code for i in issues]
        assert "LOW_REPLICATES" not in codes

    def test_warns_missing_sample_and_donor(self):
        df = pd.DataFrame({"condition": ["ctrl", "disease"]})
        issues = run_preflight(df, "differential_expression")
        cols_warned = [i.column for i in issues if i.code == "PREFLIGHT_MISSING_RECOMMENDED"]
        assert "sample_id" in cols_warned
        assert "donor_id" in cols_warned


# ---------------------------------------------------------------------------
# Integration preflight
# ---------------------------------------------------------------------------


class TestIntegrationPreflight:
    def test_pass_with_multiple_batches(self):
        df = pd.DataFrame({
            "batch_id": ["b1", "b1", "b2", "b2"],
            "sample_id": ["s1", "s1", "s2", "s2"],
            "assay": ["10x", "10x", "10x", "10x"],
        })
        issues = run_preflight(df, "integration")
        assert len(issues) == 0

    def test_error_missing_batch(self):
        df = pd.DataFrame({"sample_id": ["s1", "s2"]})
        issues = run_preflight(df, "integration")
        errors = [i for i in issues if i.severity == "error"]
        assert any(i.code == "PREFLIGHT_MISSING_REQUIRED" and i.column == "batch_id" for i in errors)

    def test_error_single_batch(self):
        df = pd.DataFrame({"batch_id": ["b1", "b1", "b1"]})
        issues = run_preflight(df, "integration")
        codes = [i.code for i in issues]
        assert "SINGLE_BATCH" in codes

    def test_warns_missing_sample_and_assay(self):
        df = pd.DataFrame({"batch_id": ["b1", "b2"]})
        issues = run_preflight(df, "integration")
        cols_warned = [i.column for i in issues if i.code == "PREFLIGHT_MISSING_RECOMMENDED"]
        assert "sample_id" in cols_warned
        assert "assay" in cols_warned


# ---------------------------------------------------------------------------
# Cell type annotation preflight
# ---------------------------------------------------------------------------


class TestCellTypeAnnotationPreflight:
    def test_pass_with_recommended_columns(self):
        df = pd.DataFrame({
            "tissue": ["lung", "lung"],
            "species": ["human", "human"],
        })
        issues = run_preflight(df, "cell_type_annotation")
        assert len(issues) == 0

    def test_info_existing_cell_type(self):
        df = pd.DataFrame({
            "cell_type": ["T cell", "B cell"],
            "tissue": ["lung", "lung"],
            "species": ["human", "human"],
        })
        issues = run_preflight(df, "cell_type_annotation")
        codes = [i.code for i in issues]
        assert "EXISTING_ANNOTATION" in codes
        info_issues = [i for i in issues if i.code == "EXISTING_ANNOTATION"]
        assert info_issues[0].severity == "info"

    def test_warns_missing_tissue_and_species(self):
        df = pd.DataFrame({"x": [1, 2]})
        issues = run_preflight(df, "cell_type_annotation")
        cols_warned = [i.column for i in issues if i.code == "PREFLIGHT_MISSING_RECOMMENDED"]
        assert "tissue" in cols_warned
        assert "species" in cols_warned

    def test_no_required_columns(self):
        """Annotation has no required columns — should only get warnings/info."""
        df = pd.DataFrame({"x": [1, 2]})
        issues = run_preflight(df, "cell_type_annotation")
        for issue in issues:
            assert issue.severity in ("warning", "info")


# ---------------------------------------------------------------------------
# Custom TaskProfile
# ---------------------------------------------------------------------------


class TestCustomTaskProfile:
    def test_custom_profile(self):
        custom = TaskProfile(
            name="my_task",
            description="A custom task",
            required_columns=("gene_score",),
            recommended_columns=("batch_id",),
        )
        df = pd.DataFrame({"x": [1]})
        report = bh.preflight(df, custom)
        assert report.profile_name == "preflight:my_task"
        errors = [i for i in report.issues if i.severity == "error"]
        assert any(i.column == "gene_score" for i in errors)


# ---------------------------------------------------------------------------
# Integration with real GEO data
# ---------------------------------------------------------------------------


FIXTURES = Path(__file__).parent / "fixtures" / "real_data"
GEO_DATASETS = {
    "lung_cancer": FIXTURES / "geo_lung_cancer_obs.csv",
    "kidney_myeloid": FIXTURES / "geo_kidney_myeloid_obs.csv",
    "crc_leukocyte": FIXTURES / "geo_crc_leukocyte_obs.csv",
    "hsc_facs": FIXTURES / "geo_hsc_facs_obs.csv",
}


@pytest.fixture
def cleaned_lung_cancer():
    from bioharmonize.io import read_obs

    df = read_obs(GEO_DATASETS["lung_cancer"])
    return bh.clean_obs(df, profile="single_cell_human").cleaned


@pytest.fixture
def cleaned_kidney_myeloid():
    from bioharmonize.io import read_obs

    df = read_obs(GEO_DATASETS["kidney_myeloid"])
    return bh.clean_obs(df, profile="single_cell_human").cleaned


class TestPreflightOnRealData:
    def test_lung_cancer_clustering(self, cleaned_lung_cancer):
        """Lung cancer has sample_id but no batch_id after clean."""
        report = bh.preflight(cleaned_lung_cancer, "clustering")
        codes = [(i.code, i.column) for i in report.issues]
        assert ("PREFLIGHT_MISSING_RECOMMENDED", "batch_id") in codes

    def test_lung_cancer_de_missing_condition(self, cleaned_lung_cancer):
        """Lung cancer lacks condition — DE preflight should error."""
        report = bh.preflight(cleaned_lung_cancer, "differential_expression")
        errors = [i for i in report.issues if i.severity == "error"]
        assert any(i.column == "condition" for i in errors)

    def test_kidney_myeloid_integration_missing_batch(self, cleaned_kidney_myeloid):
        """Kidney myeloid has batch_id after clean. Check it passes required."""
        report = bh.preflight(cleaned_kidney_myeloid, "integration")
        missing_required = [
            i for i in report.issues
            if i.code == "PREFLIGHT_MISSING_REQUIRED" and i.column == "batch_id"
        ]
        assert len(missing_required) == 0


# ---------------------------------------------------------------------------
# CLI preflight command
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CLICK, reason="click not installed")
class TestCLIPreflight:
    def test_preflight_command_runs(self):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        csv_path = GEO_DATASETS["lung_cancer"]
        result = runner.invoke(_build_cli(), ["preflight", str(csv_path), "clustering"])
        assert "preflight:clustering" in result.output

    def test_preflight_exits_1_on_errors(self):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        csv_path = GEO_DATASETS["lung_cancer"]
        result = runner.invoke(_build_cli(), ["preflight", str(csv_path), "differential_expression"])
        # lung_cancer lacks condition → should get PREFLIGHT_MISSING_REQUIRED error → exit 1
        assert result.exit_code == 1

    def test_preflight_no_clean_flag(self):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        csv_path = GEO_DATASETS["lung_cancer"]
        result = runner.invoke(_build_cli(), ["preflight", str(csv_path), "clustering", "--no-clean"])
        assert "preflight:clustering" in result.output
