"""Metamorphic invariant tests for bioharmonize.

These tests verify structural invariants that must hold under input
transformations, rather than checking specific outputs for specific inputs.

Invariants tested:
1. Row reorder preserves issue counts
2. Column reorder preserves semantics
3. Repair is idempotent
4. Irrelevant columns don't change readiness
5. Alias rename round-trips preserve output
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

import bioharmonize as bh


def _make_df(**kwargs) -> pd.DataFrame:
    n = len(next(iter(kwargs.values())))
    return pd.DataFrame(kwargs, index=[f"cell_{i}" for i in range(n)])


def _issue_counter(issues: list[bh.Issue]) -> Counter:
    """Count issues by (severity, code) for order-independent comparison."""
    return Counter((i.severity, i.code) for i in issues)


# ---------------------------------------------------------------------------
# 1. Row reorder preserves issue counts
# ---------------------------------------------------------------------------


class TestRowReorderPreservesIssueCounts:
    """Shuffling rows must not change the set of issues found."""

    def _assert_same_issues(self, df_original: pd.DataFrame, df_shuffled: pd.DataFrame):
        r_orig = bh.validate(df_original, level="strict")
        r_shuf = bh.validate(df_shuffled, level="strict")
        assert _issue_counter(r_orig.issues) == _issue_counter(r_shuf.issues)

    def test_simple_shuffle(self):
        df = _make_df(
            cell_type=["T cell", "B cell", "NK cell", "Monocyte"],
            sex=["male", "female", "unknown", "male"],
            sample_id=["S1", "S2", "S3", "S4"],
        )
        shuffled = df.sample(frac=1, random_state=42)
        self._assert_same_issues(df, shuffled)

    def test_shuffle_with_mixed_vocabulary(self):
        df = _make_df(
            cell_type=["T cell", "B cell", "NK cell"],
            sex=["m", "female", "alien"],
            sample_id=["S1", "S2", "S3"],
            condition=["ctrl", "disease", "remission"],
        )
        shuffled = df.sample(frac=1, random_state=99)
        self._assert_same_issues(df, shuffled)

    def test_shuffle_with_nulls(self):
        df = pd.DataFrame(
            {
                "cell_type": ["T cell", None, "NK cell", "B cell"],
                "sex": ["male", "female", None, "male"],
                "sample_id": ["S1", "S2", "S3", "S4"],
                "donor_id": ["D1", "D2", "D3", "D4"],
                "condition": ["control", "disease", "control", "disease"],
            },
            index=[f"cell_{i}" for i in range(4)],
        )
        shuffled = df.sample(frac=1, random_state=7)
        self._assert_same_issues(df, shuffled)

    def test_repair_issue_counts_stable_under_shuffle(self):
        df = _make_df(
            celltype=["T cell", "B cell", "NK cell"],
            gender=["m", "f", "male"],
            sample=["S1", "S2", "S3"],
        )
        shuffled = df.sample(frac=1, random_state=13)
        r_orig = bh.repair(df)
        r_shuf = bh.repair(shuffled)
        assert _issue_counter(r_orig.issues) == _issue_counter(r_shuf.issues)

    def test_reverse_order(self):
        df = _make_df(
            cell_type=["T cell", "B cell", "NK cell", "Monocyte", "DC"],
            sex=["male", "female", "unknown", "male", "female"],
            sample_id=["S1", "S2", "S3", "S4", "S5"],
        )
        reversed_df = df.iloc[::-1]
        self._assert_same_issues(df, reversed_df)


# ---------------------------------------------------------------------------
# 2. Column reorder preserves semantics
# ---------------------------------------------------------------------------


class TestColumnReorderPreservesSemantics:
    """Reordering columns must not change repair/validate results."""

    def test_validate_same_issues(self):
        df = _make_df(
            cell_type=["T cell", "B cell"],
            sex=["male", "female"],
            sample_id=["S1", "S2"],
            batch_id=["B1", "B2"],
            condition=["control", "disease"],
        )
        reordered = df[["condition", "batch_id", "sex", "sample_id", "cell_type"]]
        r_orig = bh.validate(df, level="strict")
        r_reord = bh.validate(reordered, level="strict")
        assert _issue_counter(r_orig.issues) == _issue_counter(r_reord.issues)

    def test_repair_same_cleaned_columns(self):
        df = _make_df(
            celltype=["T cell"],
            gender=["m"],
            sample=["S1"],
            donor=["D1"],
        )
        reordered = df[["donor", "sample", "gender", "celltype"]]
        r_orig = bh.repair(df)
        r_reord = bh.repair(reordered)
        assert set(r_orig.cleaned.columns) == set(r_reord.cleaned.columns)

    def test_repair_same_values_after_reorder(self):
        df = _make_df(
            celltype=["T cell", "B cell"],
            gender=["m", "f"],
            sample=["S1", "S2"],
        )
        reordered = df[["sample", "gender", "celltype"]]
        r_orig = bh.repair(df)
        r_reord = bh.repair(reordered)
        # Same columns present
        assert set(r_orig.cleaned.columns) == set(r_reord.cleaned.columns)
        # Same values in each column (align by shared column set)
        for col in r_orig.cleaned.columns:
            orig_vals = sorted(r_orig.cleaned[col].tolist())
            reord_vals = sorted(r_reord.cleaned[col].tolist())
            assert orig_vals == reord_vals, f"Mismatch in column {col!r}"

    def test_repair_same_change_kinds(self):
        df = _make_df(
            celltype=["T cell"],
            gender=["m"],
            status=["ctrl"],
        )
        reordered = df[["status", "gender", "celltype"]]
        r_orig = bh.repair(df)
        r_reord = bh.repair(reordered)
        orig_kinds = Counter(c.kind for c in r_orig.changes)
        reord_kinds = Counter(c.kind for c in r_reord.changes)
        assert orig_kinds == reord_kinds


# ---------------------------------------------------------------------------
# 3. Repair is idempotent
# ---------------------------------------------------------------------------


class TestRepairIdempotent:
    """Applying repair twice must produce the same result as applying it once."""

    def _repair_once_and_twice(self, df: pd.DataFrame):
        r1 = bh.repair(df)
        r2 = bh.repair(r1.cleaned)
        return r1, r2

    def test_cleaned_data_unchanged(self):
        df = _make_df(
            celltype=["T cell", "B cell"],
            gender=["m", "f"],
            sample=["S1", "S2"],
            status=["ctrl", "case"],
        )
        r1, r2 = self._repair_once_and_twice(df)
        pd.testing.assert_frame_equal(r1.cleaned, r2.cleaned)

    def test_no_changes_on_second_pass(self):
        df = _make_df(
            celltype=["T cell", "B cell"],
            gender=["m", "f"],
            sample=["S1", "S2"],
        )
        r1, r2 = self._repair_once_and_twice(df)
        assert len(r1.changes) > 0, "First repair should make changes"
        assert len(r2.changes) == 0, "Second repair should make no changes"

    def test_issues_stable_on_second_pass(self):
        df = _make_df(
            celltype=["T cell"],
            gender=["m"],
            sample=["S1"],
            donor=["D1"],
            status=["ctrl"],
        )
        r1, r2 = self._repair_once_and_twice(df)
        assert _issue_counter(r1.issues) == _issue_counter(r2.issues)

    def test_idempotent_with_already_clean_data(self):
        df = _make_df(
            cell_type=["T cell"],
            sex=["male"],
            sample_id=["S1"],
            donor_id=["D1"],
            condition=["control"],
        )
        r1, r2 = self._repair_once_and_twice(df)
        pd.testing.assert_frame_equal(r1.cleaned, r2.cleaned)
        assert len(r1.changes) == 0
        assert len(r2.changes) == 0

    def test_idempotent_with_value_normalization(self):
        df = _make_df(
            sex=["m", "f", "male", "female", "unknown"],
            condition=["ctrl", "case", "control", "disease", "healthy"],
            assay=["10x genomics", "smart-seq2", "10x", "smartseq2", "10x 3'"],
        )
        r1, r2 = self._repair_once_and_twice(df)
        pd.testing.assert_frame_equal(r1.cleaned, r2.cleaned)
        assert len(r2.changes) == 0

    def test_idempotent_with_index_repair(self):
        df = pd.DataFrame(
            {"cell_type": ["T cell", "B cell", "NK cell"]},
            index=[0, 1, 2],
        )
        r1 = bh.repair(df)
        r2 = bh.repair(r1.cleaned)
        pd.testing.assert_frame_equal(r1.cleaned, r2.cleaned)


# ---------------------------------------------------------------------------
# 4. Irrelevant columns don't change readiness
# ---------------------------------------------------------------------------


class TestIrrelevantColumnsDontChangeReadiness:
    """Adding columns unknown to the profile must not affect task readiness."""

    def _readiness(self, df: pd.DataFrame) -> dict[str, str]:
        report = bh.repair(df)
        return report.readiness

    def test_extra_columns_dont_affect_readiness(self):
        base = _make_df(
            cell_type=["T cell", "B cell"],
            condition=["control", "disease"],
            sample_id=["S1", "S2"],
            donor_id=["D1", "D2"],
            batch_id=["B1", "B2"],
        )
        extended = base.copy()
        extended["favorite_color"] = ["blue", "red"]
        extended["random_score"] = [0.42, 0.99]
        extended["lab_notebook_page"] = [12, 34]
        assert self._readiness(base) == self._readiness(extended)

    def test_many_irrelevant_columns(self):
        base = _make_df(
            cell_type=["T cell"],
            condition=["control"],
            sample_id=["S1"],
            batch_id=["B1"],
        )
        extended = base.copy()
        for i in range(20):
            extended[f"noise_col_{i}"] = [f"val_{i}"]
        assert self._readiness(base) == self._readiness(extended)

    def test_irrelevant_columns_dont_create_issues(self):
        base = _make_df(
            cell_type=["T cell", "B cell"],
            sex=["male", "female"],
            sample_id=["S1", "S2"],
        )
        extended = base.copy()
        extended["experiment_notes"] = ["ran ok", "needs redo"]
        r_base = bh.validate(base, level="standard")
        r_ext = bh.validate(extended, level="standard")
        # Filter to non-info issues — irrelevant columns should not add errors/warnings
        base_significant = [(i.severity, i.code) for i in r_base.issues if i.severity != "info"]
        ext_significant = [(i.severity, i.code) for i in r_ext.issues if i.severity != "info"]
        assert Counter(base_significant) == Counter(ext_significant)

    def test_preflight_unaffected_by_extra_columns(self):
        base = _make_df(
            condition=["control", "disease"],
            sample_id=["S1", "S2"],
            donor_id=["D1", "D2"],
        )
        extended = base.copy()
        extended["instrument_serial"] = ["ABC123", "DEF456"]
        r_base = bh.preflight(base, "differential_expression")
        r_ext = bh.preflight(extended, "differential_expression")
        assert _issue_counter(r_base.issues) == _issue_counter(r_ext.issues)


# ---------------------------------------------------------------------------
# 5. Alias rename round-trips preserve output
# ---------------------------------------------------------------------------


class TestAliasRenameRoundTrips:
    """Pre-renaming alias columns to canonical names before repair must
    produce the same cleaned output as letting repair do the rename."""

    def test_manual_rename_matches_auto_repair(self):
        df = _make_df(
            celltype=["T cell", "B cell"],
            gender=["m", "f"],
            sample=["S1", "S2"],
        )
        # Let repair auto-rename
        r_auto = bh.repair(df)

        # Manually rename then repair
        manual = df.rename(columns={
            "celltype": "cell_type",
            "gender": "sex",
            "sample": "sample_id",
        })
        r_manual = bh.repair(manual)

        pd.testing.assert_frame_equal(
            r_auto.cleaned.sort_index(axis=1),
            r_manual.cleaned.sort_index(axis=1),
        )

    def test_partial_pre_rename(self):
        df = _make_df(
            celltype=["T cell"],
            gender=["m"],
            donor=["D1"],
        )
        # Pre-rename only one alias
        partial = df.rename(columns={"celltype": "cell_type"})
        r_orig = bh.repair(df)
        r_partial = bh.repair(partial)
        pd.testing.assert_frame_equal(
            r_orig.cleaned.sort_index(axis=1),
            r_partial.cleaned.sort_index(axis=1),
        )

    def test_column_map_equivalent_to_alias(self):
        """Using explicit column_map should produce the same result as alias resolution."""
        df = _make_df(
            celltype=["T cell"],
            gender=["m"],
        )
        r_alias = bh.repair(df)
        r_explicit = bh.repair(df, column_map={"celltype": "cell_type", "gender": "sex"})
        pd.testing.assert_frame_equal(
            r_alias.cleaned.sort_index(axis=1),
            r_explicit.cleaned.sort_index(axis=1),
        )

    def test_all_aliases_round_trip(self):
        """Every known alias should produce identical output to using canonical names."""
        from bioharmonize.aliases import SINGLE_CELL_HUMAN_COLUMN_ALIASES

        # Group aliases by target canonical name, pick one alias per target
        alias_to_canonical = {}
        for alias, canonical in SINGLE_CELL_HUMAN_COLUMN_ALIASES.items():
            if canonical not in alias_to_canonical:
                alias_to_canonical[canonical] = alias

        for canonical, alias in alias_to_canonical.items():
            # DataFrame with the alias column
            df_alias = pd.DataFrame(
                {alias: ["test_value"]},
                index=["cell_0"],
            )
            # DataFrame with the canonical column
            df_canonical = pd.DataFrame(
                {canonical: ["test_value"]},
                index=["cell_0"],
            )
            r_alias = bh.repair(df_alias)
            r_canonical = bh.repair(df_canonical)

            assert canonical in r_alias.cleaned.columns, (
                f"Alias {alias!r} should resolve to {canonical!r}"
            )
            assert (
                r_alias.cleaned[canonical].iloc[0]
                == r_canonical.cleaned[canonical].iloc[0]
            ), f"Value mismatch for alias {alias!r} -> {canonical!r}"

    def test_rename_then_repair_values_match(self):
        """After manual alias rename, value normalization should still work."""
        df = _make_df(
            gender=["m", "f", "male"],
            status=["ctrl", "case", "healthy"],
        )
        r_auto = bh.repair(df)

        manual = df.rename(columns={"gender": "sex", "status": "condition"})
        r_manual = bh.repair(manual)

        # Both should have normalized values
        assert list(r_auto.cleaned["sex"]) == list(r_manual.cleaned["sex"])
        assert list(r_auto.cleaned["condition"]) == list(r_manual.cleaned["condition"])
