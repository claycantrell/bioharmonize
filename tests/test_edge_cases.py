"""Edge-case battle tests for bioharmonize.

Covers: empty DataFrames, all-null columns, unicode column names, wide tables,
conflicting alias chains, mixed-type columns, duplicate column names,
single-row DataFrames.
"""

import numpy as np
import pandas as pd
import pytest

import bioharmonize as bh


# ---------------------------------------------------------------------------
# Empty DataFrames
# ---------------------------------------------------------------------------

class TestEmptyDataFrame:
    def test_clean_obs_empty_rows(self):
        df = pd.DataFrame({"cell_type": pd.Series([], dtype=str), "sex": pd.Series([], dtype=str)})
        report = bh.clean_obs(df)
        assert report.cleaned.shape[0] == 0
        assert "cell_type" in report.cleaned.columns
        codes = [i.code for i in report.issues]
        assert "EMPTY_DATAFRAME" in codes

    def test_clean_obs_empty_columns(self):
        df = pd.DataFrame(index=["cell_0", "cell_1"])
        report = bh.clean_obs(df)
        assert report.cleaned.shape == (2, 0)

    def test_validate_obs_empty(self):
        df = pd.DataFrame({"sex": pd.Series([], dtype=str)})
        report = bh.validate_obs(df, level="strict")
        codes = [i.code for i in report.issues]
        assert "EMPTY_DATAFRAME" in codes

    def test_report_summary_on_empty(self):
        df = pd.DataFrame({"cell_type": pd.Series([], dtype=str)})
        report = bh.clean_obs(df)
        summary = report.summary()
        assert "0 rows" in summary


# ---------------------------------------------------------------------------
# All-null columns
# ---------------------------------------------------------------------------

class TestAllNullColumns:
    def test_all_null_canonical_column(self):
        df = pd.DataFrame({"sex": [None, None, None]}, index=["c0", "c1", "c2"])
        report = bh.clean_obs(df)
        # Should not crash; column preserved with nulls
        assert report.cleaned["sex"].isna().all()

    def test_all_null_alias_column(self):
        df = pd.DataFrame({"gender": [None, None]}, index=["c0", "c1"])
        report = bh.clean_obs(df)
        # Should rename gender -> sex even if all null
        assert "sex" in report.cleaned.columns

    def test_all_null_no_normalization_changes(self):
        df = pd.DataFrame({"sex": [None, None]}, index=["c0", "c1"])
        report = bh.clean_obs(df)
        norm_changes = [c for c in report.changes if c.kind == "normalize_value"]
        assert len(norm_changes) == 0

    def test_strict_validation_nulls_in_non_nullable(self):
        """All-null column flagged under strict validation if spec says non-nullable."""
        prof = bh.profile("single_cell_human")
        # Default profile columns are nullable, but verify no crash
        df = pd.DataFrame(
            {
                "cell_type": [None, None],
                "sample_id": [None, None],
                "donor_id": [None, None],
                "condition": [None, None],
                "sex": [None, None],
            },
            index=["c0", "c1"],
        )
        report = bh.validate_obs(df, level="strict")
        # Should complete without error
        assert isinstance(report, bh.Report)


# ---------------------------------------------------------------------------
# Unicode column names
# ---------------------------------------------------------------------------

class TestUnicodeColumnNames:
    def test_unicode_column_preserved(self):
        df = pd.DataFrame({"细胞类型": ["T cell", "B cell"]}, index=["c0", "c1"])
        report = bh.clean_obs(df)
        assert "细胞类型" in report.cleaned.columns

    def test_unicode_column_with_alias_map(self):
        df = pd.DataFrame({"🧬type": ["T cell"]}, index=["c0"])
        report = bh.clean_obs(df, column_map={"🧬type": "cell_type"})
        assert "cell_type" in report.cleaned.columns

    def test_unicode_values_preserved(self):
        df = pd.DataFrame({"cell_type": ["T細胞", "B細胞"]}, index=["c0", "c1"])
        report = bh.clean_obs(df)
        assert list(report.cleaned["cell_type"]) == ["T細胞", "B細胞"]

    def test_accented_column_name(self):
        df = pd.DataFrame({"résumé": ["a", "b"], "cell_type": ["T", "B"]}, index=["c0", "c1"])
        report = bh.clean_obs(df)
        assert "résumé" in report.cleaned.columns


# ---------------------------------------------------------------------------
# Wide tables
# ---------------------------------------------------------------------------

class TestWideTables:
    def test_wide_table_100_columns(self):
        data = {f"col_{i}": ["val"] for i in range(100)}
        data["cell_type"] = ["T cell"]
        data["sex"] = ["m"]
        df = pd.DataFrame(data, index=["c0"])
        report = bh.clean_obs(df)
        assert report.cleaned.shape[1] == 102
        assert report.cleaned["sex"].iloc[0] == "male"

    def test_wide_table_with_many_aliases(self):
        data = {
            "celltype": ["T cell"],
            "gender": ["m"],
            "donor": ["D1"],
            "sample": ["S1"],
            "batch": ["B1"],
            "organ": ["lung"],
            "status": ["ctrl"],
        }
        # Add 50 extra columns
        for i in range(50):
            data[f"extra_{i}"] = [f"v{i}"]
        df = pd.DataFrame(data, index=["c0"])
        report = bh.clean_obs(df)
        # All aliases renamed
        assert "cell_type" in report.cleaned.columns
        assert "sex" in report.cleaned.columns
        assert "donor_id" in report.cleaned.columns
        assert "sample_id" in report.cleaned.columns
        assert "batch_id" in report.cleaned.columns
        assert "tissue" in report.cleaned.columns
        assert "condition" in report.cleaned.columns

    def test_summary_on_wide_table(self):
        data = {f"col_{i}": ["x"] for i in range(200)}
        df = pd.DataFrame(data, index=["c0"])
        report = bh.clean_obs(df)
        summary = report.summary()
        assert "200 columns" in summary


# ---------------------------------------------------------------------------
# Conflicting alias chains
# ---------------------------------------------------------------------------

class TestConflictingAliasChains:
    def test_two_aliases_same_target(self):
        """Two columns that both alias to the same canonical name."""
        df = pd.DataFrame(
            {"celltype": ["T cell"], "annotation": ["B cell"]},
            index=["c0"],
        )
        report = bh.clean_obs(df)
        conflict_issues = [i for i in report.issues if i.code == "COLUMN_CONFLICT"]
        # One of them should map, the other should produce a conflict
        assert len(conflict_issues) >= 1

    def test_canonical_plus_alias(self):
        """Canonical column present alongside its alias."""
        df = pd.DataFrame(
            {"cell_type": ["T cell"], "celltype": ["B cell"]},
            index=["c0"],
        )
        report = bh.clean_obs(df)
        conflict_issues = [i for i in report.issues if i.code == "COLUMN_CONFLICT"]
        assert len(conflict_issues) == 1
        # Canonical stays, alias not renamed
        assert "cell_type" in report.cleaned.columns

    def test_explicit_map_avoids_conflict(self):
        """User column_map can redirect an alias to avoid conflict."""
        df = pd.DataFrame(
            {"cell_type": ["T cell"], "celltype": ["B cell"]},
            index=["c0"],
        )
        report = bh.clean_obs(df, column_map={"celltype": "secondary_type"})
        assert "secondary_type" in report.cleaned.columns
        assert "cell_type" in report.cleaned.columns
        conflict_issues = [i for i in report.issues if i.code == "COLUMN_CONFLICT"]
        assert len(conflict_issues) == 0

    def test_triple_alias_conflict(self):
        """Three columns that all alias to cell_type."""
        df = pd.DataFrame(
            {
                "celltype": ["T cell"],
                "annotation": ["B cell"],
                "cell_type_annotation": ["NK cell"],
            },
            index=["c0"],
        )
        report = bh.clean_obs(df)
        conflict_issues = [i for i in report.issues if i.code == "COLUMN_CONFLICT"]
        # At least 2 conflicts (one maps, the other two conflict)
        assert len(conflict_issues) >= 2


# ---------------------------------------------------------------------------
# Mixed-type columns
# ---------------------------------------------------------------------------

class TestMixedTypeColumns:
    def test_numeric_strings_mixed(self):
        """Column with both numeric and string values."""
        df = pd.DataFrame(
            {"cell_type": ["T cell", 42, "B cell"]},
            index=["c0", "c1", "c2"],
        )
        report = bh.clean_obs(df)
        # Should not crash
        assert report.cleaned.shape[0] == 3

    def test_bool_in_string_column(self):
        df = pd.DataFrame(
            {"cell_type": [True, False, "T cell"]},
            index=["c0", "c1", "c2"],
        )
        report = bh.clean_obs(df)
        assert report.cleaned.shape[0] == 3

    def test_float_nan_mixed_with_strings(self):
        df = pd.DataFrame(
            {"sex": ["m", np.nan, "f", np.nan]},
            index=["c0", "c1", "c2", "c3"],
        )
        report = bh.clean_obs(df)
        assert report.cleaned["sex"].iloc[0] == "male"
        assert pd.isna(report.cleaned["sex"].iloc[1])
        assert report.cleaned["sex"].iloc[2] == "female"

    def test_integer_column_with_nan(self):
        """Integer-ish column that has NaN (pandas promotes to float)."""
        df = pd.DataFrame(
            {"donor_id": [1, 2, np.nan, 4]},
            index=["c0", "c1", "c2", "c3"],
        )
        report = bh.clean_obs(df)
        # donor_id spec is "string", so numeric -> string coercion should kick in
        # NaN should remain NaN
        assert pd.isna(report.cleaned["donor_id"].iloc[2])

    def test_all_numeric_in_string_column(self):
        """All-numeric values in a column with string spec."""
        df = pd.DataFrame(
            {"sample_id": [100, 200, 300]},
            index=["c0", "c1", "c2"],
        )
        report = bh.clean_obs(df)
        # Should be coerced to string without .0
        assert report.cleaned["sample_id"].iloc[0] == "100"


# ---------------------------------------------------------------------------
# Duplicate column names
# ---------------------------------------------------------------------------

class TestDuplicateColumnNames:
    def test_duplicate_columns_flagged(self):
        df = pd.DataFrame([[1, 2]], columns=["cell_type", "cell_type"], index=["c0"])
        report = bh.clean_obs(df)
        codes = [i.code for i in report.issues]
        assert "DUPLICATE_COLUMNS" in codes

    def test_duplicate_non_canonical_columns(self):
        df = pd.DataFrame([[1, 2]], columns=["foo", "foo"], index=["c0"])
        report = bh.clean_obs(df)
        codes = [i.code for i in report.issues]
        assert "DUPLICATE_COLUMNS" in codes

    def test_validate_obs_catches_duplicates(self):
        df = pd.DataFrame([[1, 2]], columns=["sex", "sex"], index=["c0"])
        report = bh.validate_obs(df, level="minimal")
        codes = [i.code for i in report.issues]
        assert "DUPLICATE_COLUMNS" in codes

    def test_duplicate_alias_columns(self):
        """Two columns with the same alias name."""
        df = pd.DataFrame([[1, 2]], columns=["donor", "donor"], index=["c0"])
        report = bh.clean_obs(df)
        codes = [i.code for i in report.issues]
        assert "DUPLICATE_COLUMNS" in codes


# ---------------------------------------------------------------------------
# Single-row DataFrames
# ---------------------------------------------------------------------------

class TestSingleRowDataFrame:
    def test_single_row_clean(self):
        df = pd.DataFrame(
            {"celltype": ["T cell"], "gender": ["m"], "donor": ["D1"]},
            index=["cell_0"],
        )
        report = bh.clean_obs(df)
        assert report.cleaned.shape[0] == 1
        assert report.cleaned["cell_type"].iloc[0] == "T cell"
        assert report.cleaned["sex"].iloc[0] == "male"
        assert report.cleaned["donor_id"].iloc[0] == "D1"

    def test_single_row_validate(self):
        df = pd.DataFrame({"cell_type": ["T cell"], "sample_id": ["S1"]}, index=["cell_0"])
        report = bh.validate_obs(df, level="standard")
        missing = [i for i in report.issues if i.code == "MISSING_REQUIRED_COLUMN"]
        assert len(missing) == 0

    def test_single_row_all_null(self):
        df = pd.DataFrame({"sex": [None], "cell_type": [None]}, index=["cell_0"])
        report = bh.clean_obs(df)
        assert report.cleaned.shape[0] == 1
        assert pd.isna(report.cleaned["sex"].iloc[0])

    def test_single_row_strict_validation(self):
        df = pd.DataFrame(
            {
                "cell_type": ["T cell"],
                "sample_id": ["S1"],
                "donor_id": ["D1"],
                "condition": ["control"],
                "sex": ["male"],
            },
            index=["cell_0"],
        )
        report = bh.validate_obs(df, level="strict")
        errors = [i for i in report.issues if i.severity == "error"]
        assert len(errors) == 0

    def test_single_row_report_save(self, tmp_path):
        df = pd.DataFrame({"sex": ["m"]}, index=["cell_0"])
        report = bh.clean_obs(df)
        report.save(tmp_path / "out")
        assert (tmp_path / "out" / "cleaned.csv").exists()


# ---------------------------------------------------------------------------
# Miscellaneous edge combos
# ---------------------------------------------------------------------------

class TestEdgeCombinations:
    def test_empty_string_values(self):
        df = pd.DataFrame({"sex": ["", "m", ""]}, index=["c0", "c1", "c2"])
        report = bh.clean_obs(df)
        # Empty string should not crash normalizer
        assert report.cleaned["sex"].iloc[1] == "male"

    def test_whitespace_only_values(self):
        df = pd.DataFrame({"sex": ["  ", "m"]}, index=["c0", "c1"])
        report = bh.clean_obs(df)
        assert report.cleaned["sex"].iloc[1] == "male"

    def test_column_name_with_spaces(self):
        df = pd.DataFrame({"cell type": ["T cell"]}, index=["c0"])
        report = bh.clean_obs(df)
        # "cell type" normalizes to "cell_type" which is a canonical name
        assert "cell_type" in report.cleaned.columns

    def test_column_name_with_hyphens(self):
        df = pd.DataFrame({"cell-type": ["T cell"]}, index=["c0"])
        report = bh.clean_obs(df)
        # "cell-type" normalizes to "cell_type"
        assert "cell_type" in report.cleaned.columns

    def test_column_name_with_mixed_case(self):
        df = pd.DataFrame({"CellType": ["T cell"]}, index=["c0"])
        report = bh.clean_obs(df)
        # "CellType" normalizes to "celltype" which is an alias for cell_type
        assert "cell_type" in report.cleaned.columns

    def test_issues_frame_on_complex_report(self):
        df = pd.DataFrame(
            {"cell_type": ["T cell"], "celltype": ["B cell"], "sex": ["invalid"]},
            index=["c0"],
        )
        report = bh.clean_obs(df, validation="strict")
        frame = report.issues_frame()
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) > 0

    def test_changes_frame_on_no_changes(self):
        df = pd.DataFrame({"cell_type": ["T cell"]}, index=["c0"])
        report = bh.clean_obs(df, validation="minimal")
        frame = report.changes_frame()
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == 0
