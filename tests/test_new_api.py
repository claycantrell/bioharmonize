"""Tests for the new top-level API: inspect, validate, repair, preflight."""

import pandas as pd
import pytest

import bioharmonize as bh


def _make_df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(kwargs, index=[f"cell_{i}" for i in range(len(next(iter(kwargs.values()))))])


# ---------------------------------------------------------------------------
# inspect()
# ---------------------------------------------------------------------------

class TestInspect:
    def test_returns_report(self):
        df = _make_df(celltype=["T cell"], sex=["m"])
        report = bh.inspect(df)
        assert isinstance(report, bh.Report)

    def test_does_not_modify_input(self):
        df = _make_df(celltype=["T cell"], sex=["m"])
        bh.inspect(df)
        assert "celltype" in df.columns
        assert df["sex"].iloc[0] == "m"

    def test_cleaned_is_copy_of_original(self):
        df = _make_df(celltype=["T cell"], sex=["m"])
        report = bh.inspect(df)
        assert "celltype" in report.cleaned.columns
        assert report.cleaned["sex"].iloc[0] == "m"

    def test_reports_planned_renames(self):
        df = _make_df(celltype=["T cell"], gender=["m"])
        report = bh.inspect(df)
        rename_changes = {c.before: c.after for c in report.changes if c.kind == "rename_column"}
        assert rename_changes.get("celltype") == "cell_type"
        assert rename_changes.get("gender") == "sex"

    def test_reports_planned_normalizations(self):
        df = _make_df(sex=["m", "f"])
        report = bh.inspect(df)
        norm_changes = [c for c in report.changes if c.kind == "normalize_value"]
        befores = {c.before for c in norm_changes}
        assert "m" in befores
        assert "f" in befores

    def test_reports_normalization_after_rename(self):
        df = _make_df(gender=["m"])
        report = bh.inspect(df)
        norm_changes = [c for c in report.changes if c.kind == "normalize_value"]
        assert len(norm_changes) == 1
        assert norm_changes[0].column == "sex"

    def test_runs_minimal_validation(self):
        df = pd.DataFrame({"cell_type": ["T cell"]}, index=["c0", "c0"])
        report = bh.inspect(df)
        codes = [i.code for i in report.issues]
        assert "DUPLICATE_INDEX" in codes

    def test_validation_level_is_minimal(self):
        df = _make_df(cell_type=["T cell"])
        report = bh.inspect(df)
        assert report.validation_level == "minimal"

    def test_conflict_reported(self):
        df = pd.DataFrame(
            {"cell_type": ["T cell"], "celltype": ["B cell"]}, index=["c0"]
        )
        report = bh.inspect(df)
        conflict = [i for i in report.issues if i.code == "COLUMN_CONFLICT"]
        assert len(conflict) == 1


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

class TestValidate:
    def test_returns_report(self):
        df = _make_df(cell_type=["T cell"])
        report = bh.validate(df)
        assert isinstance(report, bh.Report)

    def test_does_not_modify_input(self):
        df = _make_df(celltype=["T cell"], sex=["m"])
        bh.validate(df)
        assert "celltype" in df.columns
        assert df["sex"].iloc[0] == "m"

    def test_no_changes_recorded(self):
        df = _make_df(celltype=["T cell"], sex=["m"])
        report = bh.validate(df)
        assert len(report.changes) == 0

    def test_missing_required_at_standard(self):
        df = _make_df(batch_id=["B1"])
        report = bh.validate(df, level="standard")
        missing = [i for i in report.issues if i.code == "MISSING_REQUIRED_COLUMN"]
        assert len(missing) >= 1

    def test_strict_validation(self):
        df = _make_df(cell_type=["T"], sample_id=["S1"], donor_id=["D1"],
                       condition=["control"], sex=["nonbinary"])
        report = bh.validate(df, level="strict")
        invalid = [i for i in report.issues if i.code == "INVALID_VALUE"]
        assert len(invalid) >= 1

    def test_clean_data_no_errors_at_strict(self):
        df = _make_df(cell_type=["T"], sample_id=["S1"], donor_id=["D1"],
                       condition=["control"], sex=["male"])
        report = bh.validate(df, level="strict")
        errors = [i for i in report.issues if i.severity == "error"]
        assert len(errors) == 0

    def test_is_alias_for_validate_obs(self):
        df = _make_df(cell_type=["T cell"])
        r1 = bh.validate(df, level="standard")
        r2 = bh.validate_obs(df, level="standard")
        assert len(r1.issues) == len(r2.issues)
        assert r1.profile_name == r2.profile_name


# ---------------------------------------------------------------------------
# repair()
# ---------------------------------------------------------------------------

class TestRepair:
    def test_returns_report(self):
        df = _make_df(celltype=["T cell"])
        report = bh.repair(df)
        assert isinstance(report, bh.Report)

    def test_renames_columns(self):
        df = _make_df(celltype=["T cell"], donor=["D1"])
        report = bh.repair(df)
        assert "cell_type" in report.cleaned.columns
        assert "donor_id" in report.cleaned.columns

    def test_normalizes_values(self):
        df = _make_df(sex=["m", "f", "male"])
        report = bh.repair(df)
        assert list(report.cleaned["sex"]) == ["male", "female", "male"]

    def test_default_copies_input(self):
        df = _make_df(sex=["m"])
        bh.repair(df)
        assert df["sex"].iloc[0] == "m"

    def test_copy_false_modifies_input(self):
        df = _make_df(sex=["m"])
        bh.repair(df, copy=False)
        assert df["sex"].iloc[0] == "male"

    def test_records_changes(self):
        df = _make_df(gender=["m"])
        report = bh.repair(df)
        assert len(report.changes) >= 2  # rename + normalize

    def test_is_alias_for_clean_obs(self):
        df = _make_df(celltype=["T cell"], sex=["m"])
        r1 = bh.repair(df)
        r2 = bh.clean_obs(df)
        assert list(r1.cleaned.columns) == list(r2.cleaned.columns)
        assert len(r1.changes) == len(r2.changes)

    def test_adata_field_none_for_dataframe(self):
        df = _make_df(cell_type=["T cell"])
        report = bh.repair(df)
        assert report.adata is None


# ---------------------------------------------------------------------------
# preflight()
# ---------------------------------------------------------------------------

class TestPreflight:
    def test_returns_report(self):
        df = _make_df(celltype=["T cell"])
        report = bh.preflight(df)
        assert isinstance(report, bh.Report)

    def test_does_not_modify_input(self):
        df = _make_df(sex=["m"])
        bh.preflight(df)
        assert df["sex"].iloc[0] == "m"

    def test_cleaned_shows_repaired_preview(self):
        df = _make_df(celltype=["T cell"], sex=["m"])
        report = bh.preflight(df)
        assert "cell_type" in report.cleaned.columns
        assert report.cleaned["sex"].iloc[0] == "male"

    def test_same_changes_as_repair(self):
        df = _make_df(gender=["m", "f"], donor=["D1", "D2"])
        pf = bh.preflight(df)
        rp = bh.repair(df)
        assert len(pf.changes) == len(rp.changes)
        pf_kinds = [(c.kind, c.column, c.before, c.after) for c in pf.changes]
        rp_kinds = [(c.kind, c.column, c.before, c.after) for c in rp.changes]
        assert pf_kinds == rp_kinds

    def test_same_issues_as_repair(self):
        df = _make_df(gender=["m", "f"])
        pf = bh.preflight(df)
        rp = bh.repair(df)
        pf_codes = sorted(i.code for i in pf.issues)
        rp_codes = sorted(i.code for i in rp.issues)
        assert pf_codes == rp_codes


# ---------------------------------------------------------------------------
# AnnData-first support
# ---------------------------------------------------------------------------

try:
    import anndata

    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestAnnDataInput:
    @pytest.fixture()
    def adata(self):
        import anndata
        import numpy as np

        obs = pd.DataFrame(
            {
                "celltype": ["T cell", "B cell", "NK cell"],
                "gender": ["m", "f", "male"],
                "sample": ["S1", "S1", "S2"],
            },
            index=["AACG_0", "AACG_1", "AACG_2"],
        )
        X = np.zeros((3, 10))
        return anndata.AnnData(X=X, obs=obs)

    def test_inspect_accepts_anndata(self, adata):
        report = bh.inspect(adata)
        assert isinstance(report, bh.Report)
        assert report.adata is adata
        rename_changes = {c.before: c.after for c in report.changes if c.kind == "rename_column"}
        assert rename_changes.get("celltype") == "cell_type"

    def test_validate_accepts_anndata(self, adata):
        report = bh.validate(adata)
        assert isinstance(report, bh.Report)
        assert report.adata is adata
        assert len(report.changes) == 0

    def test_repair_accepts_anndata(self, adata):
        report = bh.repair(adata)
        assert isinstance(report, bh.Report)
        # cleaned is the repaired obs DataFrame
        assert "cell_type" in report.cleaned.columns
        assert report.cleaned["sex"].iloc[0] == "male"
        # adata field has the repaired AnnData
        assert report.adata is not None
        assert report.adata is not adata  # copy=True by default
        assert "cell_type" in report.adata.obs.columns

    def test_repair_does_not_modify_original_adata(self, adata):
        bh.repair(adata)
        assert "celltype" in adata.obs.columns
        assert adata.obs["gender"].iloc[0] == "m"

    def test_repair_copy_false_modifies_adata(self, adata):
        report = bh.repair(adata, copy=False)
        assert "sex" in adata.obs.columns
        assert adata.obs["sex"].iloc[0] == "male"
        assert report.adata is adata

    def test_preflight_accepts_anndata(self, adata):
        report = bh.preflight(adata)
        assert isinstance(report, bh.Report)
        assert "cell_type" in report.cleaned.columns
        # Original adata not modified
        assert "celltype" in adata.obs.columns

    def test_repair_preserves_x_matrix(self, adata):
        import numpy as np

        report = bh.repair(adata)
        assert report.adata.X.shape == (3, 10)
        assert np.all(report.adata.X == 0)

    def test_invalid_input_raises(self):
        with pytest.raises(TypeError, match="Expected AnnData or DataFrame"):
            bh.repair("not_a_dataframe")


# ---------------------------------------------------------------------------
# CLI commands for new API
# ---------------------------------------------------------------------------

try:
    from click.testing import CliRunner

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False


@pytest.mark.skipif(not HAS_CLICK, reason="click not installed")
class TestNewCLICommands:
    @pytest.fixture()
    def sample_csv(self, tmp_path):
        df = pd.DataFrame(
            {"celltype": ["T cell", "B cell"], "sex": ["m", "f"], "sample_id": ["S1", "S2"]},
            index=["cell_0", "cell_1"],
        )
        path = tmp_path / "study.csv"
        df.to_csv(path)
        return path

    def test_inspect_command(self, sample_csv):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(_build_cli(), ["inspect", str(sample_csv)])
        assert result.exit_code == 0
        assert "single_cell_human" in result.output

    def test_repair_command(self, sample_csv, tmp_path):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out_dir = tmp_path / "output"
        result = runner.invoke(
            _build_cli(), ["repair", str(sample_csv), "-o", str(out_dir)]
        )
        assert result.exit_code == 0
        assert (out_dir / "cleaned.csv").exists()
        assert (out_dir / "summary.txt").exists()

    def test_preflight_command(self, sample_csv):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(_build_cli(), ["preflight", str(sample_csv)])
        assert result.exit_code == 0
        assert "single_cell_human" in result.output
        assert "changes" in result.output
