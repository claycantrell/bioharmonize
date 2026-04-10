"""Tests for h5ad support in the bioharmonize CLI commands."""

import pytest

try:
    from click.testing import CliRunner

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

try:
    import anndata
    import numpy as np

    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

import pandas as pd

pytestmark = [
    pytest.mark.skipif(not HAS_CLICK, reason="click not installed"),
    pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed"),
]


@pytest.fixture
def sample_h5ad(tmp_path):
    """Create a minimal .h5ad file with metadata."""
    obs = pd.DataFrame(
        {"celltype": ["T cell", "B cell"], "sex": ["m", "f"], "sample_id": ["S1", "S2"]},
        index=["cell_0", "cell_1"],
    )
    ad = anndata.AnnData(
        X=np.zeros((2, 3)),
        obs=obs,
    )
    path = tmp_path / "study.h5ad"
    ad.write_h5ad(path)
    return path


class TestInspectH5ad:
    def test_inspect_h5ad(self, sample_h5ad):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(_build_cli(), ["inspect", str(sample_h5ad)])
        assert result.exit_code == 0, result.output
        assert "single_cell_human" in result.output

    def test_inspect_h5ad_shows_info_issues(self, sample_h5ad):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(_build_cli(), ["inspect", str(sample_h5ad)])
        assert result.exit_code == 0
        # AnnData-specific info diagnostics should be counted
        assert "info(s)" in result.output


class TestRepairH5ad:
    def test_repair_h5ad(self, sample_h5ad, tmp_path):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out_dir = tmp_path / "output"
        result = runner.invoke(
            _build_cli(), ["repair", str(sample_h5ad), "-o", str(out_dir)]
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "cleaned.csv").exists()
        assert (out_dir / "repaired.h5ad").exists()

    def test_repair_h5ad_output_is_valid(self, sample_h5ad, tmp_path):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out_dir = tmp_path / "output"
        result = runner.invoke(
            _build_cli(), ["repair", str(sample_h5ad), "-o", str(out_dir)]
        )
        assert result.exit_code == 0
        repaired = anndata.read_h5ad(out_dir / "repaired.h5ad")
        # celltype should have been renamed to cell_type
        assert "cell_type" in repaired.obs.columns


class TestValidateH5ad:
    def test_validate_h5ad(self, sample_h5ad):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(_build_cli(), ["validate", str(sample_h5ad)])
        assert result.exit_code == 0, result.output
        assert "single_cell_human" in result.output

    def test_validate_h5ad_strict(self, sample_h5ad):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(
            _build_cli(), ["validate", str(sample_h5ad), "--level", "strict"]
        )
        # Strict should flag missing columns — exit 1
        assert result.exit_code == 1


class TestPreflightH5ad:
    def test_preflight_h5ad(self, sample_h5ad):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(
            _build_cli(), ["preflight", str(sample_h5ad), "clustering"]
        )
        # preflight runs and produces output (may exit 0 or 1 depending on checks)
        assert "preflight:clustering" in result.output


class TestCleanH5ad:
    def test_clean_h5ad(self, sample_h5ad, tmp_path):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out_dir = tmp_path / "output"
        result = runner.invoke(
            _build_cli(), ["clean", str(sample_h5ad), "-o", str(out_dir)]
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "cleaned.csv").exists()
        assert (out_dir / "cleaned.h5ad").exists()

    def test_clean_h5ad_output_valid(self, sample_h5ad, tmp_path):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out_dir = tmp_path / "output"
        result = runner.invoke(
            _build_cli(), ["clean", str(sample_h5ad), "-o", str(out_dir)]
        )
        assert result.exit_code == 0
        cleaned = anndata.read_h5ad(out_dir / "cleaned.h5ad")
        assert "cell_type" in cleaned.obs.columns


class TestReadDataIO:
    def test_read_data_h5ad(self, sample_h5ad):
        from bioharmonize.io import read_data

        result = read_data(sample_h5ad)
        assert isinstance(result, anndata.AnnData)
        assert result.n_obs == 2

    def test_read_data_csv(self, tmp_path):
        from bioharmonize.io import read_data

        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2]}, index=["r1", "r2"]).to_csv(csv_path)
        result = read_data(csv_path)
        assert isinstance(result, pd.DataFrame)
