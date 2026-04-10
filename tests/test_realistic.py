"""Battle tests: realistic messy metadata scenarios.

These tests simulate real-world metadata as it arrives from CELLxGENE portals,
collaborator spreadsheets, and minimal internal pipelines.  Each scenario
exercises column aliasing, value normalisation, and audit-trail accuracy
end-to-end.
"""

import pandas as pd
import pytest

import bioharmonize as bh


# ---------------------------------------------------------------------------
# Scenario 1 — CELLxGENE-style obs
#   Mixed-case column names, extra (non-canonical) columns, assay aliases.
# ---------------------------------------------------------------------------

class TestCELLxGENEStyleObs:
    @pytest.fixture()
    def obs(self) -> pd.DataFrame:
        # Realistic CELLxGENE-like: mixed-case aliases, extra portal columns
        return pd.DataFrame(
            {
                "CellType": ["T cell", "B cell", "NK cell", "Monocyte"],
                "Donor": ["D001", "D001", "D002", "D002"],
                "Sample": ["S1", "S1", "S2", "S2"],
                "gender": ["M", "F", "male", "female"],
                "status": ["control", "ctrl", "disease", "healthy"],
                "assay": ["10x Genomics", "10x Genomics", "smartseq2", "Smart-Seq2"],
                "tissue": ["lung", "lung", "blood", "blood"],
                # extra columns that should pass through untouched
                "n_genes": [2000, 1500, 1800, 2200],
                "suspension_type": ["cell", "cell", "nucleus", "nucleus"],
                "is_primary_data": [True, True, False, True],
            },
            index=[f"AACG_{i}" for i in range(4)],
        )

    def test_column_renaming(self, obs):
        report = bh.clean_obs(obs)
        cleaned = report.cleaned

        # Aliases should resolve to canonical names
        assert "cell_type" in cleaned.columns
        assert "donor_id" in cleaned.columns
        assert "sample_id" in cleaned.columns
        assert "sex" in cleaned.columns       # gender -> sex
        assert "condition" in cleaned.columns  # status -> condition

        # Original aliases should be gone
        for alias in ("CellType", "Donor", "Sample", "gender", "status"):
            assert alias not in cleaned.columns

    def test_extra_columns_preserved(self, obs):
        report = bh.clean_obs(obs)
        cleaned = report.cleaned

        for col in ("n_genes", "suspension_type", "is_primary_data"):
            assert col in cleaned.columns

        # Extra column values should be untouched
        assert list(cleaned["n_genes"]) == [2000, 1500, 1800, 2200]

    def test_sex_normalisation(self, obs):
        report = bh.clean_obs(obs)
        assert list(report.cleaned["sex"]) == ["male", "female", "male", "female"]

    def test_condition_normalisation(self, obs):
        report = bh.clean_obs(obs)
        assert list(report.cleaned["condition"]) == [
            "control",
            "control",
            "disease",
            "control",
        ]

    def test_assay_normalisation(self, obs):
        report = bh.clean_obs(obs)
        assert list(report.cleaned["assay"]) == [
            "10x",
            "10x",
            "smart-seq2",
            "smart-seq2",
        ]

    def test_audit_trail_renames(self, obs):
        report = bh.clean_obs(obs)
        renames = {
            c.before: c.after
            for c in report.changes
            if c.kind == "rename_column"
        }
        assert renames.get("CellType") == "cell_type"
        assert renames.get("Donor") == "donor_id"
        assert renames.get("Sample") == "sample_id"
        assert renames.get("gender") == "sex"
        assert renames.get("status") == "condition"

    def test_audit_trail_value_changes(self, obs):
        report = bh.clean_obs(obs)
        norm = [c for c in report.changes if c.kind == "normalize_value"]
        norm_by_col = {}
        for c in norm:
            norm_by_col.setdefault(c.column, []).append(c)

        # Sex: "M" -> "male", "F" -> "female"
        sex_befores = {c.before for c in norm_by_col.get("sex", [])}
        assert "M" in sex_befores
        assert "F" in sex_befores

        # Condition: "ctrl" -> "control", "healthy" -> "control"
        cond_befores = {c.before for c in norm_by_col.get("condition", [])}
        assert "ctrl" in cond_befores
        assert "healthy" in cond_befores

    def test_change_counts_accurate(self, obs):
        report = bh.clean_obs(obs)
        sex_norm = [
            c
            for c in report.changes
            if c.kind == "normalize_value" and c.column == "sex"
        ]
        total_sex_affected = sum(c.count for c in sex_norm)
        # "M" and "F" are normalised (2 values); "male" and "female" are
        # already the canonical form so the normaliser returns them unchanged.
        assert total_sex_affected == 2

    def test_canonical_columns_not_renamed(self, obs):
        """'assay' and 'tissue' are already canonical (lowercase) — no rename."""
        report = bh.clean_obs(obs)
        renames = {c.before for c in report.changes if c.kind == "rename_column"}
        assert "assay" not in renames
        assert "tissue" not in renames


# ---------------------------------------------------------------------------
# Scenario 2 — Collaborator CSV with typos and inconsistent values
#   gender column (alias for sex), disease_status (alias for condition),
#   misspellings in free-text fields, mixed whitespace.
# ---------------------------------------------------------------------------

class TestCollaboratorCSV:
    @pytest.fixture()
    def obs(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "  gender ": ["male", "  F ", "Male", "unknown", "female"],
                "disease_status": ["Healthy", "ctrl", " Case", "Disease", "CTRL"],
                "cell_type_annotation": [
                    "CD4+ T cell",
                    "B cell",
                    "NK cell",
                    "Monocyte",
                    "Dendritic cell",
                ],
                "patient": ["P01", "P02", "P01", "P03", "P02"],
                "batch": ["run_1", "run_1", "run_2", "run_2", "run_3"],
            },
            index=[f"barcode_{i}" for i in range(5)],
        )

    def test_whitespace_column_rename(self, obs):
        """Leading/trailing whitespace in column names should not block aliasing."""
        report = bh.clean_obs(obs)
        cleaned = report.cleaned
        # "  gender " normalizes to "gender" which is aliased to "sex"
        assert "sex" in cleaned.columns
        assert "  gender " not in cleaned.columns

    def test_all_aliases_resolve(self, obs):
        report = bh.clean_obs(obs)
        cleaned = report.cleaned
        assert "condition" in cleaned.columns  # disease_status -> condition
        assert "cell_type" in cleaned.columns  # cell_type_annotation -> cell_type
        assert "donor_id" in cleaned.columns  # patient -> donor_id
        assert "batch_id" in cleaned.columns  # batch -> batch_id

    def test_sex_with_whitespace_values(self, obs):
        report = bh.clean_obs(obs)
        expected = ["male", "female", "male", "unknown", "female"]
        assert list(report.cleaned["sex"]) == expected

    def test_condition_case_insensitive(self, obs):
        """Normaliser is case-insensitive: Healthy/CTRL/Case should all resolve."""
        report = bh.clean_obs(obs)
        expected = ["control", "control", "disease", "disease", "control"]
        assert list(report.cleaned["condition"]) == expected

    def test_cell_type_values_unchanged(self, obs):
        """cell_type has no built-in normaliser — values should pass through."""
        report = bh.clean_obs(obs)
        assert report.cleaned["cell_type"].iloc[0] == "CD4+ T cell"

    def test_audit_trail_completeness(self, obs):
        report = bh.clean_obs(obs)

        # Every rename and value change should be recorded
        rename_targets = {c.after for c in report.changes if c.kind == "rename_column"}
        assert "sex" in rename_targets
        assert "condition" in rename_targets
        assert "cell_type" in rename_targets
        assert "donor_id" in rename_targets
        assert "batch_id" in rename_targets

    def test_row_count_preserved(self, obs):
        report = bh.clean_obs(obs)
        assert len(report.cleaned) == 5

    def test_index_preserved(self, obs):
        report = bh.clean_obs(obs)
        assert list(report.cleaned.index) == [f"barcode_{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Scenario 3 — Minimal obs (barcodes + one annotation)
#   Extremely sparse metadata — only barcodes as index and a single column.
#   Validation should still produce a coherent report.
# ---------------------------------------------------------------------------

class TestMinimalObs:
    @pytest.fixture()
    def obs(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"celltype": ["T cell", "B cell", "NK cell"]},
            index=["AAACCTG-1", "AAACCTG-2", "AAACCTG-3"],
        )

    def test_single_alias_resolved(self, obs):
        report = bh.clean_obs(obs)
        assert "cell_type" in report.cleaned.columns
        assert "celltype" not in report.cleaned.columns

    def test_no_other_changes(self, obs):
        report = bh.clean_obs(obs)
        # Only change should be the column rename
        assert all(c.kind == "rename_column" for c in report.changes)
        assert len(report.changes) == 1

    def test_validation_flags_missing_sample_id(self, obs):
        """Standard validation requires sample_id — should flag it as missing."""
        report = bh.clean_obs(obs, validation="standard")
        missing = [i for i in report.issues if i.code == "MISSING_REQUIRED_COLUMN"]
        missing_cols = {i.column for i in missing}
        assert "sample_id" in missing_cols

    def test_minimal_validation_no_required_errors(self, obs):
        """Minimal validation should NOT flag missing columns."""
        report = bh.clean_obs(obs, validation="minimal")
        missing = [i for i in report.issues if i.code == "MISSING_REQUIRED_COLUMN"]
        assert len(missing) == 0

    def test_values_untouched(self, obs):
        report = bh.clean_obs(obs)
        assert list(report.cleaned["cell_type"]) == ["T cell", "B cell", "NK cell"]

    def test_index_intact(self, obs):
        report = bh.clean_obs(obs)
        assert list(report.cleaned.index) == [
            "AAACCTG-1",
            "AAACCTG-2",
            "AAACCTG-3",
        ]

    def test_report_summary_coherent(self, obs):
        report = bh.clean_obs(obs)
        summary = report.summary()
        assert "single_cell_human" in summary
        assert "3 rows" in summary
        assert "changes: 1" in summary

    def test_save_roundtrip(self, obs, tmp_path):
        report = bh.clean_obs(obs)
        report.save(tmp_path / "out")
        # All four artefacts should exist
        assert (tmp_path / "out" / "cleaned.csv").exists()
        assert (tmp_path / "out" / "changes.csv").exists()
        assert (tmp_path / "out" / "issues.csv").exists()
        assert (tmp_path / "out" / "summary.txt").exists()

        # Roundtrip the changes CSV and verify it matches
        changes_df = pd.read_csv(tmp_path / "out" / "changes.csv")
        assert len(changes_df) == 1
        assert changes_df.iloc[0]["kind"] == "rename_column"


# ---------------------------------------------------------------------------
# Scenario 4 — Mixed NaN / None with normalizable values
#   Ensures nulls are preserved through column rename + value normalisation,
#   and that audit trail counts exclude null rows.
# ---------------------------------------------------------------------------

class TestNullHandling:
    @pytest.fixture()
    def obs(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gender": ["m", None, "F", pd.NA, "male"],
                "disease_status": [None, "ctrl", "disease", "Healthy", None],
                "sample": ["S1", "S1", "S2", "S2", "S3"],
            },
            index=[f"cell_{i}" for i in range(5)],
        )

    def test_nulls_preserved_after_rename(self, obs):
        report = bh.clean_obs(obs)
        sex = report.cleaned["sex"]
        assert pd.isna(sex.iloc[1])
        assert pd.isna(sex.iloc[3])

    def test_non_null_values_normalised(self, obs):
        report = bh.clean_obs(obs)
        sex = report.cleaned["sex"]
        assert sex.iloc[0] == "male"
        assert sex.iloc[2] == "female"
        assert sex.iloc[4] == "male"

    def test_condition_nulls_preserved(self, obs):
        report = bh.clean_obs(obs)
        cond = report.cleaned["condition"]
        assert pd.isna(cond.iloc[0])
        assert pd.isna(cond.iloc[4])
        assert cond.iloc[1] == "control"
        assert cond.iloc[2] == "disease"
        assert cond.iloc[3] == "control"

    def test_audit_counts_exclude_nulls(self, obs):
        report = bh.clean_obs(obs)
        sex_norms = [
            c
            for c in report.changes
            if c.kind == "normalize_value" and c.column == "sex"
        ]
        total = sum(c.count for c in sex_norms)
        # "m" -> "male" (1 row) and "F" -> "female" (1 row) are changes.
        # "male" -> normaliser returns "male" (no-op, not recorded).
        # Nulls are skipped entirely.
        assert total == 2
