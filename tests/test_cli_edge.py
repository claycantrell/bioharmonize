"""Edge-case tests for the bioharmonize CLI (clean / validate commands)."""

import pytest

try:
    from click.testing import CliRunner

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

import pandas as pd

pytestmark = pytest.mark.skipif(not HAS_CLICK, reason="click not installed")


# ---------------------------------------------------------------------------
# Fixtures: edge-case files
# ---------------------------------------------------------------------------


@pytest.fixture
def bom_csv(tmp_path):
    """UTF-8 BOM-encoded CSV."""
    path = tmp_path / "bom.csv"
    content = ",cell_type,sex,sample_id\ncell_0,T cell,male,S1\ncell_1,B cell,female,S2\n"
    path.write_bytes(content.encode("utf-8-sig"))
    return path


@pytest.fixture
def semicolon_csv(tmp_path):
    """Semicolon-delimited .csv — wrong delimiter for read_obs."""
    path = tmp_path / "semi.csv"
    path.write_text(
        ";cell_type;sex;sample_id\ncell_0;T cell;male;S1\ncell_1;B cell;female;S2\n"
    )
    return path


@pytest.fixture
def empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    return path


@pytest.fixture
def header_only_csv(tmp_path):
    path = tmp_path / "header_only.csv"
    path.write_text(",cell_type,sex,sample_id\n")
    return path


@pytest.fixture
def unicode_csv(tmp_path):
    path = tmp_path / "unicode.csv"
    path.write_text(
        ",cell_type,sex,sample_id\ncell_0,T cell,male,S\u00fc1\ncell_1,B cell,female,S\u00fc2\n"
    )
    return path


@pytest.fixture
def single_row_csv(tmp_path):
    path = tmp_path / "single.csv"
    path.write_text(",cell_type,sex,sample_id\ncell_0,T cell,male,S1\n")
    return path


@pytest.fixture
def valid_tsv(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text(
        "\tcell_type\tsex\tsample_id\ncell_0\tT cell\tmale\tS1\ncell_1\tB cell\tfemale\tS2\n"
    )
    return path


# ---------------------------------------------------------------------------
# CLI clean: edge-case inputs
# ---------------------------------------------------------------------------


class TestCleanEdgeCases:
    def test_clean_bom_csv(self, bom_csv, tmp_path):
        """clean command handles BOM-encoded CSV without error."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out = tmp_path / "output"
        result = runner.invoke(_build_cli(), ["clean", str(bom_csv), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert (out / "cleaned.csv").exists()
        # Verify no BOM artifacts in the cleaned output
        cleaned = pd.read_csv(out / "cleaned.csv", index_col=0)
        for col in cleaned.columns:
            assert "\ufeff" not in col

    def test_clean_empty_csv_fails_gracefully(self, empty_csv, tmp_path):
        """clean command on an empty file exits non-zero or shows error."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out = tmp_path / "output"
        result = runner.invoke(_build_cli(), ["clean", str(empty_csv), "-o", str(out)])
        assert result.exit_code != 0

    def test_clean_header_only_csv(self, header_only_csv, tmp_path):
        """clean command on a header-only CSV should succeed with 0 rows."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out = tmp_path / "output"
        result = runner.invoke(
            _build_cli(), ["clean", str(header_only_csv), "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        cleaned = pd.read_csv(out / "cleaned.csv", index_col=0)
        assert len(cleaned) == 0

    def test_clean_unicode_csv(self, unicode_csv, tmp_path):
        """clean command preserves unicode characters in values."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out = tmp_path / "output"
        result = runner.invoke(
            _build_cli(), ["clean", str(unicode_csv), "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        cleaned = pd.read_csv(out / "cleaned.csv", index_col=0)
        assert "S\u00fc1" in cleaned["sample_id"].values

    def test_clean_single_row(self, single_row_csv, tmp_path):
        """clean command works on a single-row CSV."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out = tmp_path / "output"
        result = runner.invoke(
            _build_cli(), ["clean", str(single_row_csv), "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        cleaned = pd.read_csv(out / "cleaned.csv", index_col=0)
        assert len(cleaned) == 1

    def test_clean_tsv(self, valid_tsv, tmp_path):
        """clean command handles .tsv files."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out = tmp_path / "output"
        result = runner.invoke(_build_cli(), ["clean", str(valid_tsv), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert (out / "cleaned.csv").exists()

    def test_clean_output_files_complete(self, single_row_csv, tmp_path):
        """All four output files are created by clean."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        out = tmp_path / "output"
        result = runner.invoke(
            _build_cli(), ["clean", str(single_row_csv), "-o", str(out)]
        )
        assert result.exit_code == 0
        assert (out / "cleaned.csv").exists()
        assert (out / "issues.csv").exists()
        assert (out / "changes.csv").exists()
        assert (out / "summary.txt").exists()


# ---------------------------------------------------------------------------
# CLI validate: edge-case inputs
# ---------------------------------------------------------------------------


class TestValidateEdgeCases:
    def test_validate_bom_csv(self, bom_csv):
        """validate command handles BOM-encoded CSV."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(_build_cli(), ["validate", str(bom_csv)])
        assert result.exit_code == 0, result.output
        assert "single_cell_human" in result.output

    def test_validate_empty_csv_fails(self, empty_csv):
        """validate command on an empty file exits non-zero."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(_build_cli(), ["validate", str(empty_csv)])
        assert result.exit_code != 0

    def test_validate_header_only(self, header_only_csv):
        """validate command on header-only CSV reports missing data issues."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(_build_cli(), ["validate", str(header_only_csv)])
        assert result.exit_code == 0, result.output

    def test_validate_unicode(self, unicode_csv):
        """validate command works with unicode values."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(_build_cli(), ["validate", str(unicode_csv)])
        assert result.exit_code == 0, result.output

    def test_validate_tsv(self, valid_tsv):
        """validate command reads .tsv files correctly."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(_build_cli(), ["validate", str(valid_tsv)])
        assert result.exit_code == 0, result.output
        assert "single_cell_human" in result.output

    def test_validate_strict_with_missing_columns(self, single_row_csv):
        """Strict validation exits 1 when required columns are missing."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(
            _build_cli(), ["validate", str(single_row_csv), "--level", "strict"]
        )
        # single_row_csv has cell_type, sex, sample_id — strict also needs donor_id, condition
        assert result.exit_code == 1

    def test_validate_minimal_always_passes(self, single_row_csv):
        """Minimal validation should not exit non-zero for missing columns."""
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        result = runner.invoke(
            _build_cli(), ["validate", str(single_row_csv), "--level", "minimal"]
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# CLI clean → Report.save round-trip via CLI
# ---------------------------------------------------------------------------


class TestCLIRoundTrip:
    def test_clean_output_is_valid_csv(self, tmp_path):
        """cleaned.csv from CLI can be re-read by read_obs."""
        from bioharmonize.cli import _build_cli
        from bioharmonize.io import read_obs

        # Create input
        src = tmp_path / "input.csv"
        df = pd.DataFrame(
            {"celltype": ["T cell", "B cell"], "sex": ["m", "f"], "sample_id": ["S1", "S2"]},
            index=["c0", "c1"],
        )
        df.to_csv(src)

        # Run clean
        runner = CliRunner()
        out = tmp_path / "output"
        result = runner.invoke(_build_cli(), ["clean", str(src), "-o", str(out)])
        assert result.exit_code == 0

        # Re-read the output
        reloaded = read_obs(out / "cleaned.csv")
        assert "cell_type" in reloaded.columns
        assert list(reloaded["sex"]) == ["male", "female"]

    def test_clean_then_validate_output(self, tmp_path):
        """Running validate on the cleaned output produces no errors."""
        from bioharmonize.cli import _build_cli

        # Create input with aliases
        src = tmp_path / "input.csv"
        df = pd.DataFrame(
            {
                "celltype": ["T cell", "B cell"],
                "gender": ["male", "female"],
                "sample_id": ["S1", "S2"],
            },
            index=["c0", "c1"],
        )
        df.to_csv(src)

        runner = CliRunner()
        out = tmp_path / "output"

        # Clean
        result = runner.invoke(_build_cli(), ["clean", str(src), "-o", str(out)])
        assert result.exit_code == 0

        # Validate the cleaned output
        cleaned_path = out / "cleaned.csv"
        result = runner.invoke(_build_cli(), ["validate", str(cleaned_path)])
        assert result.exit_code == 0
