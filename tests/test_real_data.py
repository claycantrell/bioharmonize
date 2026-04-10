"""Integration tests using real-world-style datasets from tests/fixtures/real_data/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

try:
    from click.testing import CliRunner

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

import bioharmonize as bh
from bioharmonize.io import read_obs

FIXTURES = Path(__file__).parent / "fixtures" / "real_data"
DATASETS = sorted(FIXTURES.glob("*.csv"))
VALIDATION_LEVELS = ["minimal", "standard", "strict"]


@pytest.fixture(params=DATASETS, ids=lambda p: p.stem)
def dataset(request) -> Path:
    return request.param


@pytest.fixture(params=VALIDATION_LEVELS)
def level(request) -> str:
    return request.param


# ---------------------------------------------------------------------------
# 1. clean_obs runs without errors on every dataset x level combination
# ---------------------------------------------------------------------------
class TestCleanObsOnRealData:
    def test_clean_obs_succeeds(self, dataset, level):
        df = read_obs(dataset)
        report = bh.clean_obs(df, profile="single_cell_human", validation=level)
        assert isinstance(report.cleaned, pd.DataFrame)
        assert report.cleaned.shape[0] == df.shape[0]

    def test_clean_obs_returns_report(self, dataset, level):
        df = read_obs(dataset)
        report = bh.clean_obs(df, profile="single_cell_human", validation=level)
        assert report.profile_name == "single_cell_human"
        assert report.validation_level == level


# ---------------------------------------------------------------------------
# 2. Column renames actually happened
# ---------------------------------------------------------------------------
EXPECTED_RENAMES = {
    "geo_messy": {
        "celltype": "cell_type",
        "gender": "sex",
        "donor": "donor_id",
        "sample": "sample_id",
        "status": "condition",
        "organ": "tissue",
        "batch": "batch_id",
    },
    "minimal_sparse": {
        "cell_type_annotation": "cell_type",
    },
}


class TestColumnRenames:
    @pytest.mark.parametrize("stem,renames", EXPECTED_RENAMES.items())
    def test_aliases_resolved(self, stem, renames):
        df = read_obs(FIXTURES / f"{stem}.csv")
        report = bh.clean_obs(df, profile="single_cell_human")
        for original, canonical in renames.items():
            assert canonical in report.cleaned.columns, (
                f"{original} should have been renamed to {canonical}"
            )
            assert original not in report.cleaned.columns

    @pytest.mark.parametrize("stem,renames", EXPECTED_RENAMES.items())
    def test_rename_changes_recorded(self, stem, renames):
        df = read_obs(FIXTURES / f"{stem}.csv")
        report = bh.clean_obs(df, profile="single_cell_human")
        rename_changes = {c.before: c.after for c in report.changes if c.kind == "rename_column"}
        for original, canonical in renames.items():
            assert rename_changes.get(original) == canonical

    def test_cellxgene_no_renames_needed(self):
        df = read_obs(FIXTURES / "cellxgene_clean.csv")
        report = bh.clean_obs(df, profile="single_cell_human")
        rename_changes = [c for c in report.changes if c.kind == "rename_column"]
        assert len(rename_changes) == 0


# ---------------------------------------------------------------------------
# 3. Value normalization (sex, condition, assay)
# ---------------------------------------------------------------------------
class TestValueNormalization:
    def test_sex_normalized_geo_messy(self):
        df = read_obs(FIXTURES / "geo_messy.csv")
        report = bh.clean_obs(df, profile="single_cell_human")
        sex_vals = set(report.cleaned["sex"].dropna().unique())
        assert sex_vals <= {"male", "female", "unknown"}

    def test_sex_normalized_minimal(self):
        df = read_obs(FIXTURES / "minimal_sparse.csv")
        report = bh.clean_obs(df, profile="single_cell_human")
        sex_vals = set(report.cleaned["sex"].dropna().unique())
        assert sex_vals <= {"male", "female", "unknown"}

    def test_condition_normalized_geo_messy(self):
        df = read_obs(FIXTURES / "geo_messy.csv")
        report = bh.clean_obs(df, profile="single_cell_human")
        cond_vals = set(report.cleaned["condition"].dropna().unique())
        assert cond_vals <= {"control", "disease"}

    def test_condition_unchanged_cellxgene(self):
        df = read_obs(FIXTURES / "cellxgene_clean.csv")
        report = bh.clean_obs(df, profile="single_cell_human")
        assert list(report.cleaned["condition"]) == ["control", "control", "disease", "disease", "control"]

    def test_assay_normalized_geo_messy(self):
        df = read_obs(FIXTURES / "geo_messy.csv")
        report = bh.clean_obs(df, profile="single_cell_human")
        assay_vals = set(report.cleaned["assay"].dropna().unique())
        assert assay_vals <= {"10x", "smart-seq2"}

    def test_normalization_changes_recorded(self):
        df = read_obs(FIXTURES / "geo_messy.csv")
        report = bh.clean_obs(df, profile="single_cell_human")
        norm_changes = [c for c in report.changes if c.kind == "normalize_value"]
        assert len(norm_changes) > 0
        columns_normalized = {c.column for c in norm_changes}
        assert "sex" in columns_normalized
        assert "condition" in columns_normalized
        assert "assay" in columns_normalized


# ---------------------------------------------------------------------------
# 4. Report summary is meaningful
# ---------------------------------------------------------------------------
class TestReportSummary:
    def test_summary_contains_profile(self, dataset, level):
        df = read_obs(dataset)
        report = bh.clean_obs(df, profile="single_cell_human", validation=level)
        summary = report.summary()
        assert "single_cell_human" in summary
        assert level in summary

    def test_summary_contains_shape(self, dataset):
        df = read_obs(dataset)
        report = bh.clean_obs(df, profile="single_cell_human")
        summary = report.summary()
        assert f"{df.shape[0]} rows" in summary

    def test_summary_contains_change_count(self, dataset):
        df = read_obs(dataset)
        report = bh.clean_obs(df, profile="single_cell_human")
        summary = report.summary()
        assert f"changes: {len(report.changes)}" in summary

    def test_summary_contains_issue_counts(self, dataset, level):
        df = read_obs(dataset)
        report = bh.clean_obs(df, profile="single_cell_human", validation=level)
        summary = report.summary()
        assert "error(s)" in summary
        assert "warning(s)" in summary


# ---------------------------------------------------------------------------
# 5. CLI clean and validate commands against real files
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_CLICK, reason="click not installed")
class TestCLIWithRealData:
    def test_clean_command(self, dataset, tmp_path):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out_dir = tmp_path / "output"
        result = runner.invoke(
            _build_cli(),
            ["clean", str(dataset), "-o", str(out_dir), "--profile", "single_cell_human"],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert (out_dir / "cleaned.csv").exists()
        assert (out_dir / "issues.csv").exists()
        assert (out_dir / "summary.txt").exists()

    @pytest.mark.parametrize("val_level", VALIDATION_LEVELS)
    def test_clean_command_with_levels(self, dataset, val_level, tmp_path):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out_dir = tmp_path / "output"
        result = runner.invoke(
            _build_cli(),
            ["clean", str(dataset), "-o", str(out_dir), "--validation", val_level],
        )
        assert result.exit_code == 0, f"CLI failed at {val_level}: {result.output}"
        summary_text = (out_dir / "summary.txt").read_text()
        assert val_level in summary_text

    def test_validate_command(self, dataset):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(
            _build_cli(),
            ["validate", str(dataset), "--profile", "single_cell_human"],
        )
        assert result.exit_code == 0, f"CLI validate failed: {result.output}"
        assert "single_cell_human" in result.output

    @pytest.mark.parametrize("val_level", VALIDATION_LEVELS)
    def test_validate_command_with_levels(self, dataset, val_level):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(
            _build_cli(),
            ["validate", str(dataset), "--level", val_level],
        )
        # validate doesn't clean/rename columns, so strict mode may exit 1
        # when required columns are missing (uses original column names)
        assert result.exit_code in (0, 1), f"CLI validate crashed at {val_level}: {result.output}"
        assert val_level in result.output
