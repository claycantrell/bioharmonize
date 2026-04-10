import pandas as pd
import pytest

import bioharmonize as bh


def _make_df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(kwargs, index=[f"cell_{i}" for i in range(len(next(iter(kwargs.values()))))])


class TestColumnRenaming:
    def test_renames_known_aliases(self):
        df = _make_df(celltype=["T cell", "B cell"], donor=["D1", "D2"])
        report = bh.clean_obs(df)
        assert "cell_type" in report.cleaned.columns
        assert "donor_id" in report.cleaned.columns
        assert "celltype" not in report.cleaned.columns

    def test_preserves_unknown_columns(self):
        df = _make_df(cell_type=["T cell"], my_custom_col=["foo"])
        report = bh.clean_obs(df)
        assert "my_custom_col" in report.cleaned.columns

    def test_explicit_map_overrides_builtin(self):
        df = _make_df(annotation=["T cell", "B cell"])
        # By default, "annotation" maps to "cell_type"
        report = bh.clean_obs(df, column_map={"annotation": "my_label"})
        assert "my_label" in report.cleaned.columns
        assert "cell_type" not in report.cleaned.columns

    def test_records_rename_changes(self):
        df = _make_df(gender=["m", "f"])
        report = bh.clean_obs(df)
        rename_changes = [c for c in report.changes if c.kind == "rename_column"]
        assert len(rename_changes) == 1
        assert rename_changes[0].before == "gender"
        assert rename_changes[0].after == "sex"

    def test_conflicting_mappings_produce_issue(self):
        # Both "cell_type" and "celltype" would map to "cell_type"
        df = pd.DataFrame(
            {"cell_type": ["T cell"], "celltype": ["B cell"]},
            index=["cell_0"],
        )
        report = bh.clean_obs(df)
        conflict_issues = [i for i in report.issues if i.code == "COLUMN_CONFLICT"]
        assert len(conflict_issues) == 1

    def test_canonical_column_not_renamed(self):
        df = _make_df(cell_type=["T cell"], sex=["male"])
        report = bh.clean_obs(df)
        rename_changes = [c for c in report.changes if c.kind == "rename_column"]
        assert len(rename_changes) == 0


class TestValueNormalization:
    def test_normalizes_sex_values(self):
        df = _make_df(sex=["m", "f", "male", "female", "unknown"])
        report = bh.clean_obs(df)
        assert list(report.cleaned["sex"]) == ["male", "female", "male", "female", "unknown"]

    def test_normalizes_condition_values(self):
        df = _make_df(condition=["ctrl", "disease", "healthy"])
        report = bh.clean_obs(df)
        assert list(report.cleaned["condition"]) == ["control", "disease", "control"]

    def test_unmapped_values_unchanged(self):
        df = _make_df(sex=["m", "intersex"])
        report = bh.clean_obs(df)
        assert report.cleaned["sex"].iloc[0] == "male"
        assert report.cleaned["sex"].iloc[1] == "intersex"

    def test_records_normalization_changes(self):
        df = _make_df(sex=["m", "m", "female"])
        report = bh.clean_obs(df)
        norm_changes = [c for c in report.changes if c.kind == "normalize_value"]
        m_change = [c for c in norm_changes if c.before == "m"]
        assert len(m_change) == 1
        assert m_change[0].after == "male"
        assert m_change[0].count == 2

    def test_custom_value_map(self):
        df = _make_df(cell_type=["TC", "BC"])
        report = bh.clean_obs(
            df, value_maps={"cell_type": {"TC": "T cell", "BC": "B cell"}}
        )
        assert list(report.cleaned["cell_type"]) == ["T cell", "B cell"]

    def test_null_values_preserved(self):
        df = _make_df(sex=["m", None, "f"])
        report = bh.clean_obs(df)
        assert pd.isna(report.cleaned["sex"].iloc[1])

    def test_assay_normalization(self):
        df = _make_df(assay=["10x genomics", "smartseq2", "smart-seq2"])
        report = bh.clean_obs(df)
        assert list(report.cleaned["assay"]) == ["10x", "smart-seq2", "smart-seq2"]


class TestCopyBehavior:
    def test_default_copies_input(self):
        df = _make_df(sex=["m"])
        report = bh.clean_obs(df)
        assert df["sex"].iloc[0] == "m"  # original unchanged
        assert report.cleaned["sex"].iloc[0] == "male"

    def test_copy_false_modifies_input(self):
        df = _make_df(sex=["m"])
        report = bh.clean_obs(df, copy=False)
        assert df["sex"].iloc[0] == "male"  # original modified


class TestReport:
    def test_summary_includes_key_info(self):
        df = _make_df(celltype=["T cell"], sex=["m"])
        report = bh.clean_obs(df)
        summary = report.summary()
        assert "single_cell_human" in summary
        assert "changes" in summary

    def test_issues_frame(self):
        df = _make_df(cell_type=["T cell"])
        report = bh.clean_obs(df, validation="standard")
        frame = report.issues_frame()
        assert isinstance(frame, pd.DataFrame)
        assert "severity" in frame.columns

    def test_changes_frame(self):
        df = _make_df(gender=["m"])
        report = bh.clean_obs(df)
        frame = report.changes_frame()
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) > 0

    def test_save(self, tmp_path):
        df = _make_df(sex=["m"])
        report = bh.clean_obs(df)
        report.save(tmp_path / "output")
        assert (tmp_path / "output" / "cleaned.csv").exists()
        assert (tmp_path / "output" / "issues.csv").exists()
        assert (tmp_path / "output" / "changes.csv").exists()
        assert (tmp_path / "output" / "summary.txt").exists()
