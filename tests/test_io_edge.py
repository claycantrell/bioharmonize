"""Edge-case tests for bioharmonize.io.read_obs and Report.save round-trip."""

import pandas as pd
import pytest

from bioharmonize import Report, clean_obs, read_obs
from bioharmonize.changes import Change
from bioharmonize.issues import Issue


# ---------------------------------------------------------------------------
# read_obs: malformed CSV inputs
# ---------------------------------------------------------------------------


class TestWrongDelimiter:
    """read_obs called on .csv files that use non-comma delimiters."""

    def test_semicolon_delimited_csv_misread(self, tmp_path):
        """A semicolon-delimited .csv is silently misread — all content becomes index."""
        path = tmp_path / "semi.csv"
        path.write_text(
            "idx;cell_type;sex\ncell_0;T cell;male\ncell_1;B cell;female\n"
        )
        df = read_obs(path)
        # With index_col=0, entire rows become the index, no real columns
        assert len(df.columns) == 0

    def test_pipe_delimited_csv_misread(self, tmp_path):
        path = tmp_path / "pipe.csv"
        path.write_text(
            "idx|cell_type|sex\ncell_0|T cell|male\ncell_1|B cell|female\n"
        )
        df = read_obs(path)
        assert len(df.columns) == 0

    def test_tab_in_csv_extension_misread(self, tmp_path):
        """A .csv file with tab delimiters is misread (tabs belong in .tsv)."""
        path = tmp_path / "tabby.csv"
        path.write_text(
            "idx\tcell_type\tsex\ncell_0\tT cell\tmale\ncell_1\tB cell\tfemale\n"
        )
        df = read_obs(path)
        # Tabs treated as part of values when sep=","
        assert len(df.columns) == 0


class TestMissingHeaders:
    """read_obs on files where the first row is data, not a header."""

    def test_headerless_csv_uses_first_row_as_header(self, tmp_path):
        """Without headers, pandas promotes the first data row to column names."""
        path = tmp_path / "no_header.csv"
        path.write_text("cell_0,T cell,male\ncell_1,B cell,female\n")
        df = read_obs(path)
        # First data row becomes column names — the "T cell" column name is wrong
        assert "T cell" in df.columns

    def test_headerless_tsv_uses_first_row_as_header(self, tmp_path):
        path = tmp_path / "no_header.tsv"
        path.write_text("cell_0\tT cell\tmale\ncell_1\tB cell\tfemale\n")
        df = read_obs(path)
        assert "T cell" in df.columns


class TestBOMEncoded:
    """read_obs on UTF-8 BOM-encoded files."""

    def test_bom_csv_first_column_readable(self, tmp_path):
        """UTF-8 BOM should not corrupt the first column name."""
        path = tmp_path / "bom.csv"
        content = "\ufeff,cell_type,sex\ncell_0,T cell,male\ncell_1,B cell,female\n"
        path.write_bytes(content.encode("utf-8-sig"))
        df = read_obs(path)
        # The index name or first column must not have BOM artifacts
        assert df.index.name is None or "\ufeff" not in str(df.index.name)
        for col in df.columns:
            assert "\ufeff" not in col

    def test_bom_tsv_first_column_readable(self, tmp_path):
        path = tmp_path / "bom.tsv"
        content = "\ufeff\tcell_type\tsex\ncell_0\tT cell\tmale\ncell_1\tB cell\tfemale\n"
        path.write_bytes(content.encode("utf-8-sig"))
        df = read_obs(path)
        for col in df.columns:
            assert "\ufeff" not in col


class TestEmptyAndMinimalFiles:
    """read_obs on empty or nearly-empty files."""

    def test_empty_csv_raises(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(pd.errors.EmptyDataError):
            read_obs(path)

    def test_header_only_csv_returns_empty_df(self, tmp_path):
        path = tmp_path / "header_only.csv"
        path.write_text(",cell_type,sex\n")
        df = read_obs(path)
        assert len(df) == 0
        assert list(df.columns) == ["cell_type", "sex"]

    def test_single_row_csv(self, tmp_path):
        path = tmp_path / "single.csv"
        path.write_text(",cell_type,sex\ncell_0,T cell,male\n")
        df = read_obs(path)
        assert len(df) == 1
        assert df.loc["cell_0", "cell_type"] == "T cell"

    def test_empty_tsv_raises(self, tmp_path):
        path = tmp_path / "empty.tsv"
        path.write_text("")
        with pytest.raises(pd.errors.EmptyDataError):
            read_obs(path)


class TestUnsupportedFormats:
    """read_obs rejects unknown file extensions."""

    def test_json_raises_value_error(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('{"cell_type": ["T cell"]}')
        with pytest.raises(ValueError, match="Unsupported"):
            read_obs(path)

    def test_xlsx_raises_value_error(self, tmp_path):
        path = tmp_path / "data.xlsx"
        path.write_bytes(b"fake xlsx content")
        with pytest.raises(ValueError, match="Unsupported"):
            read_obs(path)

    def test_no_extension_raises_value_error(self, tmp_path):
        path = tmp_path / "data"
        path.write_text(",cell_type\ncell_0,T cell\n")
        with pytest.raises(ValueError, match="Unsupported"):
            read_obs(path)


class TestTSVReading:
    """read_obs with valid .tsv and .txt files."""

    def test_tsv_reads_correctly(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text("\tcell_type\tsex\ncell_0\tT cell\tmale\n")
        df = read_obs(path)
        assert list(df.columns) == ["cell_type", "sex"]
        assert df.loc["cell_0", "cell_type"] == "T cell"

    def test_txt_reads_as_tsv(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("\tcell_type\tsex\ncell_0\tT cell\tmale\n")
        df = read_obs(path)
        assert list(df.columns) == ["cell_type", "sex"]


class TestSpecialCharacters:
    """read_obs with values containing special characters."""

    def test_quoted_commas_in_csv(self, tmp_path):
        path = tmp_path / "quoted.csv"
        path.write_text(',cell_type,notes\ncell_0,"T cell, activated","note, 1"\n')
        df = read_obs(path)
        assert df.loc["cell_0", "cell_type"] == "T cell, activated"

    def test_unicode_values(self, tmp_path):
        path = tmp_path / "unicode.csv"
        path.write_text(",cell_type,species\ncell_0,Zelle,Mus m\u00fcsculus\n")
        df = read_obs(path)
        assert df.loc["cell_0", "species"] == "Mus m\u00fcsculus"

    def test_newlines_in_quoted_fields(self, tmp_path):
        path = tmp_path / "newline.csv"
        path.write_text(',cell_type,notes\ncell_0,T cell,"line1\nline2"\n')
        df = read_obs(path)
        assert "line1\nline2" in df.loc["cell_0", "notes"]


# ---------------------------------------------------------------------------
# Report.save round-trip
# ---------------------------------------------------------------------------


class TestReportSaveRoundTrip:
    """Save a Report, read back each output file, verify fidelity."""

    @pytest.fixture
    def sample_report(self):
        df = pd.DataFrame(
            {
                "cell_type": ["T cell", "B cell", "NK cell"],
                "sex": ["male", "female", "male"],
                "sample_id": ["S1", "S2", "S3"],
            },
            index=["cell_0", "cell_1", "cell_2"],
        )
        issues = [
            Issue(
                severity="warning",
                code="SUSPICIOUS_COLUMN",
                column="celltyp",
                message="Near-match for canonical column 'cell_type'",
                suggestion="Rename to 'cell_type'",
                row_count=None,
            ),
            Issue(
                severity="error",
                code="MISSING_REQUIRED",
                column="donor_id",
                message="Required column 'donor_id' is missing",
                suggestion="Add a 'donor_id' column",
                row_count=None,
            ),
        ]
        changes = [
            Change(
                kind="rename_column",
                column="cell_type",
                before="celltype",
                after="cell_type",
                count=None,
            ),
            Change(
                kind="normalize_value",
                column="sex",
                before="m",
                after="male",
                count=2,
            ),
        ]
        return Report(
            cleaned=df,
            issues=issues,
            changes=changes,
            profile_name="single_cell_human",
            validation_level="standard",
        )

    def test_round_trip_cleaned_csv(self, sample_report, tmp_path):
        out = tmp_path / "report_out"
        sample_report.save(out)

        reloaded = pd.read_csv(out / "cleaned.csv", index_col=0)
        pd.testing.assert_frame_equal(reloaded, sample_report.cleaned)

    def test_round_trip_issues_csv(self, sample_report, tmp_path):
        out = tmp_path / "report_out"
        sample_report.save(out)

        issues_df = pd.read_csv(out / "issues.csv")
        assert len(issues_df) == 2
        assert set(issues_df["severity"]) == {"warning", "error"}
        assert set(issues_df["code"]) == {"SUSPICIOUS_COLUMN", "MISSING_REQUIRED"}

    def test_round_trip_changes_csv(self, sample_report, tmp_path):
        out = tmp_path / "report_out"
        sample_report.save(out)

        changes_df = pd.read_csv(out / "changes.csv")
        assert len(changes_df) == 2
        assert set(changes_df["kind"]) == {"rename_column", "normalize_value"}

    def test_round_trip_summary_txt(self, sample_report, tmp_path):
        out = tmp_path / "report_out"
        sample_report.save(out)

        summary = (out / "summary.txt").read_text()
        assert "single_cell_human" in summary
        assert "standard" in summary
        assert "3 rows" in summary
        assert "rename" in summary.lower()

    def test_save_creates_directory(self, sample_report, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        sample_report.save(deep)
        assert (deep / "cleaned.csv").exists()

    def test_empty_report_round_trip(self, tmp_path):
        """Report with no issues or changes still saves valid CSV files."""
        df = pd.DataFrame({"x": [1]}, index=["r0"])
        report = Report(cleaned=df)
        out = tmp_path / "empty_report"
        report.save(out)

        issues_df = pd.read_csv(out / "issues.csv")
        assert len(issues_df) == 0
        assert "severity" in issues_df.columns

        changes_df = pd.read_csv(out / "changes.csv")
        assert len(changes_df) == 0
        assert "kind" in changes_df.columns

    def test_clean_obs_save_round_trip(self, tmp_path):
        """End-to-end: clean_obs -> save -> read back cleaned.csv matches."""
        df = pd.DataFrame(
            {"celltype": ["T cell", "B cell"], "sex": ["m", "f"]},
            index=["c0", "c1"],
        )
        report = clean_obs(df)
        out = tmp_path / "e2e"
        report.save(out)

        reloaded = pd.read_csv(out / "cleaned.csv", index_col=0)
        # Column should have been renamed
        assert "cell_type" in reloaded.columns
        # Values should have been normalized
        assert list(reloaded["sex"]) == ["male", "female"]
