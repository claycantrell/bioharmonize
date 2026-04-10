import pandas as pd
import pytest

import bioharmonize as bh


def _make_df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(kwargs, index=[f"cell_{i}" for i in range(len(next(iter(kwargs.values()))))])


class TestMinimalValidation:
    def test_duplicate_index_flagged(self):
        df = pd.DataFrame({"cell_type": ["T", "B"]}, index=["cell_0", "cell_0"])
        report = bh.validate_obs(df, level="minimal")
        codes = [i.code for i in report.issues]
        assert "DUPLICATE_INDEX" in codes

    def test_empty_dataframe_flagged(self):
        df = pd.DataFrame({"cell_type": pd.Series([], dtype=str)})
        report = bh.validate_obs(df, level="minimal")
        codes = [i.code for i in report.issues]
        assert "EMPTY_DATAFRAME" in codes

    def test_clean_df_no_issues(self):
        df = _make_df(cell_type=["T cell", "B cell"])
        report = bh.validate_obs(df, level="minimal")
        errors = [i for i in report.issues if i.severity == "error"]
        assert len(errors) == 0


class TestStandardValidation:
    def test_missing_required_column_flagged(self):
        df = _make_df(batch_id=["B1", "B2"])
        report = bh.validate_obs(df, level="standard")
        missing = [i for i in report.issues if i.code == "MISSING_REQUIRED_COLUMN"]
        missing_cols = [i.column for i in missing]
        assert "cell_type" in missing_cols
        assert "sample_id" in missing_cols

    def test_suspicious_column_flagged(self):
        df = _make_df(cell_types=["T cell"])  # near-miss for "cell_type"
        report = bh.validate_obs(df, level="standard")
        suspicious = [i for i in report.issues if i.code == "SUSPICIOUS_COLUMN"]
        assert len(suspicious) >= 1


class TestStrictValidation:
    def test_invalid_values_flagged(self):
        df = _make_df(
            cell_type=["T cell"],
            sample_id=["S1"],
            donor_id=["D1"],
            condition=["control"],
            sex=["nonbinary"],  # not in allowed_values for sex
        )
        report = bh.validate_obs(df, level="strict")
        invalid = [i for i in report.issues if i.code == "INVALID_VALUE"]
        assert len(invalid) >= 1
        assert invalid[0].column == "sex"

    def test_all_required_present_no_missing_error(self):
        df = _make_df(
            cell_type=["T cell"],
            sample_id=["S1"],
            donor_id=["D1"],
            condition=["control"],
            sex=["male"],
        )
        report = bh.validate_obs(df, level="strict")
        missing = [i for i in report.issues if i.code == "MISSING_REQUIRED_COLUMN"]
        assert len(missing) == 0


class TestValidateDoesNotMutate:
    def test_no_changes_recorded(self):
        df = _make_df(celltype=["T cell"], sex=["m"])
        report = bh.validate_obs(df, level="standard")
        assert len(report.changes) == 0
        # Original aliases not renamed
        assert "celltype" in report.cleaned.columns
