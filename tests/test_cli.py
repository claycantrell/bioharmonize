import pytest

try:
    from click.testing import CliRunner

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

import pandas as pd

pytestmark = pytest.mark.skipif(not HAS_CLICK, reason="click not installed")


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame(
        {"celltype": ["T cell", "B cell"], "sex": ["m", "f"], "sample_id": ["S1", "S2"]},
        index=["cell_0", "cell_1"],
    )
    path = tmp_path / "study.csv"
    df.to_csv(path)
    return path


class TestCleanCommand:
    def test_produces_output_files(self, sample_csv, tmp_path):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out_dir = tmp_path / "output"
        result = runner.invoke(_build_cli(), ["clean", str(sample_csv), "-o", str(out_dir)])
        assert result.exit_code == 0
        assert (out_dir / "cleaned.csv").exists()
        assert (out_dir / "issues.csv").exists()
        assert (out_dir / "summary.txt").exists()


class TestValidateCommand:
    def test_runs_validation(self, sample_csv):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(_build_cli(), ["validate", str(sample_csv)])
        assert result.exit_code == 0
        assert "single_cell_human" in result.output
