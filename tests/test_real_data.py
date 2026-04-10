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

# The 4 real GEO datasets used for targeted assertions
GEO_DATASETS = {
    "lung_cancer": FIXTURES / "geo_lung_cancer_obs.csv",
    "kidney_myeloid": FIXTURES / "geo_kidney_myeloid_obs.csv",
    "crc_leukocyte": FIXTURES / "geo_crc_leukocyte_obs.csv",
    "hsc_facs": FIXTURES / "geo_hsc_facs_obs.csv",
}


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


# ---------------------------------------------------------------------------
# 6. Targeted assertions for 4 GEO datasets
# ---------------------------------------------------------------------------


class TestGeoLungCancerRenames:
    """geo_lung_cancer_obs: clean dataset with near-standard column names."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        df = read_obs(GEO_DATASETS["lung_cancer"])
        self.report = bh.clean_obs(df, profile="single_cell_human")
        self.cleaned = self.report.cleaned

    def test_cell_type_renamed(self):
        assert "cell_type" in self.cleaned.columns
        assert "Cell_type" not in self.cleaned.columns

    def test_sample_renamed_to_sample_id(self):
        assert "sample_id" in self.cleaned.columns
        assert "Sample" not in self.cleaned.columns

    def test_sample_origin_not_renamed(self):
        """Sample_Origin has no alias — it should pass through unchanged."""
        assert "Sample_Origin" in self.cleaned.columns

    def test_cell_subtype_not_renamed(self):
        assert "Cell_subtype" in self.cleaned.columns

    def test_barcode_not_renamed(self):
        assert "Barcode" in self.cleaned.columns

    def test_cell_type_values_preserved(self):
        """Cell type values are full readable names, not codes."""
        vals = set(self.cleaned["cell_type"].unique())
        assert "B lymphocytes" in vals
        assert "Myeloid cells" in vals

    def test_no_normalizations_applied(self):
        """No normalizable columns (sex/condition/assay) exist, so zero normalizations."""
        norm_changes = [c for c in self.report.changes if c.kind == "normalize_value"]
        assert len(norm_changes) == 0


class TestGeoKidneyMyeloidRenames:
    """geo_kidney_myeloid_obs: moderately messy with non-standard column names."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        df = read_obs(GEO_DATASETS["kidney_myeloid"])
        self.report = bh.clean_obs(df, profile="single_cell_human")
        self.cleaned = self.report.cleaned

    def test_patient_renamed_to_donor_id(self):
        assert "donor_id" in self.cleaned.columns
        assert "patient" not in self.cleaned.columns

    def test_batch_renamed_to_batch_id(self):
        assert "batch_id" in self.cleaned.columns
        assert "batch" not in self.cleaned.columns

    def test_tissue_already_canonical(self):
        """tissue is already the canonical name — should stay, not be renamed."""
        assert "tissue" in self.cleaned.columns
        rename_targets = {c.before for c in self.report.changes if c.kind == "rename_column"}
        assert "tissue" not in rename_targets

    def test_cancer_not_renamed(self):
        """'cancer' has no alias to 'disease' — should pass through."""
        assert "cancer" in self.cleaned.columns

    def test_tech_not_renamed(self):
        """'tech' has no alias to 'assay' — should pass through."""
        assert "tech" in self.cleaned.columns

    def test_major_cluster_not_renamed(self):
        """MajorCluster has no alias to cell_type — should pass through."""
        assert "MajorCluster" in self.cleaned.columns

    def test_tissue_values_are_abbreviations(self):
        """Tissue values are single-letter abbreviations, passed through as-is."""
        vals = set(self.cleaned["tissue"].unique())
        assert vals <= {"N", "T"}

    def test_donor_id_values(self):
        vals = set(self.cleaned["donor_id"].unique())
        assert "P20181217" in vals


class TestGeoCrcLeukocyteRenames:
    """geo_crc_leukocyte_obs: messy dataset with 23 columns, mostly QC metrics."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        df = read_obs(GEO_DATASETS["crc_leukocyte"])
        self.report = bh.clean_obs(df, profile="single_cell_human")
        self.cleaned = self.report.cleaned

    def test_sample_renamed_to_sample_id(self):
        assert "sample_id" in self.cleaned.columns
        assert "Sample" not in self.cleaned.columns

    def test_tissue_renamed_case_only(self):
        """Tissue → tissue (case normalization to canonical name)."""
        assert "tissue" in self.cleaned.columns
        assert "Tissue" not in self.cleaned.columns

    def test_platform_renamed_case_only(self):
        """Platform → platform (case normalization to canonical name)."""
        assert "platform" in self.cleaned.columns
        assert "Platform" not in self.cleaned.columns

    def test_global_cluster_not_renamed(self):
        """Global_Cluster has no alias to cell_type — stays as-is."""
        assert "Global_Cluster" in self.cleaned.columns
        assert "cell_type" not in self.cleaned.columns

    def test_sub_cluster_not_renamed(self):
        assert "Sub_Cluster" in self.cleaned.columns

    def test_qc_columns_survive(self):
        """QC metric columns should pass through untouched."""
        for col in ["raw.nUMI", "raw.nGene", "filter.nUMI", "filter.nGene"]:
            assert col in self.cleaned.columns

    def test_tissue_values_single_letter(self):
        vals = set(self.cleaned["tissue"].unique())
        assert vals == {"N", "P", "T"}

    def test_no_normalizations_applied(self):
        norm_changes = [c for c in self.report.changes if c.kind == "normalize_value"]
        assert len(norm_changes) == 0


class TestGeoHscFacsRenames:
    """geo_hsc_facs_obs: very messy FACS data with no cell_type column at all."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        df = read_obs(GEO_DATASETS["hsc_facs"])
        self.report = bh.clean_obs(df, profile="single_cell_human")
        self.cleaned = self.report.cleaned

    def test_donor_renamed_to_donor_id(self):
        assert "donor_id" in self.cleaned.columns
        assert "Donor" not in self.cleaned.columns

    def test_tissue_renamed_case_only(self):
        assert "tissue" in self.cleaned.columns
        assert "Tissue" not in self.cleaned.columns

    def test_no_cell_type_column(self):
        """No column maps to cell_type — it should remain absent."""
        assert "cell_type" not in self.cleaned.columns

    def test_facs_markers_survive(self):
        """FACS intensity columns should pass through untouched."""
        for marker in ["CD34", "CD38", "CD45RA", "CD49f", "CD90"]:
            assert marker in self.cleaned.columns

    def test_cycling_not_renamed(self):
        assert "Cycling" in self.cleaned.columns

    def test_tissue_values_full_text(self):
        """Unlike other GEO datasets, tissue values are full text here."""
        vals = set(self.cleaned["tissue"].unique())
        assert "Bone Marrow" in vals
        assert "Mobilized Peripheral Blood" in vals

    def test_donor_id_values(self):
        vals = set(self.cleaned["donor_id"].unique())
        assert "BM1" in vals
        assert "mPB-3" in vals

    def test_no_normalizations_applied(self):
        norm_changes = [c for c in self.report.changes if c.kind == "normalize_value"]
        assert len(norm_changes) == 0


# ---------------------------------------------------------------------------
# 7. Strict validation errors for GEO datasets
# ---------------------------------------------------------------------------


class TestStrictValidationErrors:
    """Verify that strict validation flags the right missing required columns."""

    def _error_codes(self, stem):
        df = read_obs(GEO_DATASETS[stem])
        report = bh.clean_obs(df, profile="single_cell_human", validation="strict")
        return {
            (i.code, i.column)
            for i in report.issues
            if i.severity == "error"
        }

    def test_lung_cancer_strict_errors(self):
        """Lung cancer has cell_type and sample_id but lacks donor_id, condition, sex."""
        errors = self._error_codes("lung_cancer")
        # Should NOT have errors for cell_type or sample_id (they exist after rename)
        assert ("MISSING_REQUIRED_COLUMN", "cell_type") not in errors
        assert ("MISSING_REQUIRED_COLUMN", "sample_id") not in errors
        # Should have errors for these missing columns
        assert ("MISSING_REQUIRED_COLUMN", "donor_id") in errors
        assert ("MISSING_REQUIRED_COLUMN", "condition") in errors
        assert ("MISSING_REQUIRED_COLUMN", "sex") in errors

    def test_kidney_myeloid_strict_errors(self):
        """Kidney has donor_id (from patient) but lacks cell_type, sample_id, condition, sex."""
        errors = self._error_codes("kidney_myeloid")
        assert ("MISSING_REQUIRED_COLUMN", "donor_id") not in errors
        assert ("MISSING_REQUIRED_COLUMN", "cell_type") in errors
        assert ("MISSING_REQUIRED_COLUMN", "sample_id") in errors
        assert ("MISSING_REQUIRED_COLUMN", "condition") in errors
        assert ("MISSING_REQUIRED_COLUMN", "sex") in errors

    def test_crc_leukocyte_strict_errors(self):
        """CRC has sample_id (from Sample) but lacks cell_type, donor_id, condition, sex."""
        errors = self._error_codes("crc_leukocyte")
        assert ("MISSING_REQUIRED_COLUMN", "sample_id") not in errors
        assert ("MISSING_REQUIRED_COLUMN", "cell_type") in errors
        assert ("MISSING_REQUIRED_COLUMN", "donor_id") in errors
        assert ("MISSING_REQUIRED_COLUMN", "condition") in errors
        assert ("MISSING_REQUIRED_COLUMN", "sex") in errors

    def test_hsc_facs_strict_errors(self):
        """HSC FACS has donor_id (from Donor) but lacks cell_type, sample_id, condition, sex."""
        errors = self._error_codes("hsc_facs")
        assert ("MISSING_REQUIRED_COLUMN", "donor_id") not in errors
        assert ("MISSING_REQUIRED_COLUMN", "cell_type") in errors
        assert ("MISSING_REQUIRED_COLUMN", "sample_id") in errors
        assert ("MISSING_REQUIRED_COLUMN", "condition") in errors
        assert ("MISSING_REQUIRED_COLUMN", "sex") in errors

    def test_standard_fewer_errors_than_strict(self):
        """Standard level requires only cell_type and sample_id, so fewer errors."""
        for stem in GEO_DATASETS:
            df = read_obs(GEO_DATASETS[stem])
            std = bh.clean_obs(df, profile="single_cell_human", validation="standard")
            strict = bh.clean_obs(df, profile="single_cell_human", validation="strict")
            std_errors = [i for i in std.issues if i.severity == "error"]
            strict_errors = [i for i in strict.issues if i.severity == "error"]
            assert len(std_errors) <= len(strict_errors), (
                f"{stem}: standard ({len(std_errors)}) should have <= errors than strict ({len(strict_errors)})"
            )

    def test_minimal_no_missing_column_errors(self):
        """Minimal validation does not check for required columns."""
        for stem in GEO_DATASETS:
            df = read_obs(GEO_DATASETS[stem])
            report = bh.clean_obs(df, profile="single_cell_human", validation="minimal")
            missing_errors = [
                i for i in report.issues
                if i.severity == "error" and i.code == "MISSING_REQUIRED_COLUMN"
            ]
            assert len(missing_errors) == 0, f"{stem}: minimal should have no missing-column errors"
