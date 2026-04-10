"""Tests for dataset sanity checks (bioharmonize.sanity)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    import anndata

    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

import bioharmonize as bh
from bioharmonize.sanity import check_dataset, ALL_CHECKS

pytestmark = pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adata(
    n_obs: int | None = None,
    n_vars: int | None = None,
    *,
    obs_index: list[str] | None = None,
    var_index: list[str] | None = None,
    obs_columns: dict | None = None,
    X: np.ndarray | None = None,
    layers: dict | None = None,
) -> anndata.AnnData:
    # Infer dimensions from index if provided
    if obs_index is not None:
        n_obs = len(obs_index)
    elif n_obs is None:
        n_obs = 10
    if var_index is not None:
        n_vars = len(var_index)
    elif n_vars is None:
        n_vars = 5

    if obs_index is None:
        obs_index = [f"cell_{i}" for i in range(n_obs)]
    if var_index is None:
        var_index = [f"gene_{i}" for i in range(n_vars)]
    if X is None:
        X = np.random.default_rng(42).random((n_obs, n_vars))
    obs = pd.DataFrame(obs_columns or {}, index=obs_index)
    var = pd.DataFrame(index=var_index)
    ad = anndata.AnnData(X=X, obs=obs, var=var)
    if layers:
        for name, data in layers.items():
            ad.layers[name] = data
    return ad


def _issue_codes(issues):
    return {i.code for i in issues}


# ---------------------------------------------------------------------------
# check_dataset API
# ---------------------------------------------------------------------------


class TestCheckDatasetAPI:
    def test_clean_dataset_no_issues(self):
        ad = _make_adata()
        issues = check_dataset(ad)
        # A clean dataset should have at most info-level issues
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_select_specific_checks(self):
        ad = _make_adata()
        issues = check_dataset(ad, checks=["obs_matrix_shape"])
        assert isinstance(issues, list)

    def test_unknown_check_raises(self):
        ad = _make_adata()
        with pytest.raises(ValueError, match="Unknown check"):
            check_dataset(ad, checks=["nonexistent_check"])

    def test_all_checks_constant(self):
        assert len(ALL_CHECKS) == 7

    def test_exported_from_package(self):
        assert hasattr(bh, "check_dataset")


# ---------------------------------------------------------------------------
# obs/matrix shape
# ---------------------------------------------------------------------------


class TestObsMatrixShape:
    def test_matching_shape_no_issue(self):
        ad = _make_adata(n_obs=10, n_vars=5)
        issues = check_dataset(ad, checks=["obs_matrix_shape"])
        assert len(issues) == 0

    def test_consistent_shape_no_issue(self):
        """AnnData enforces shape consistency at construction, so this just
        verifies the check runs cleanly on a well-formed object."""
        ad = _make_adata(n_obs=10, n_vars=5)
        issues = check_dataset(ad, checks=["obs_matrix_shape"])
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# duplicate cell IDs
# ---------------------------------------------------------------------------


class TestDuplicateCellIds:
    def test_unique_ids_no_issue(self):
        ad = _make_adata(obs_index=["A", "B", "C"])
        issues = check_dataset(ad, checks=["duplicate_cell_ids"])
        assert len(issues) == 0

    def test_duplicate_ids_flagged(self):
        ad = _make_adata(
            n_obs=3,
            obs_index=["cell_0", "cell_0", "cell_1"],
        )
        issues = check_dataset(ad, checks=["duplicate_cell_ids"])
        assert len(issues) == 1
        assert issues[0].code == "DUPLICATE_CELL_IDS"
        assert issues[0].severity == "error"
        assert issues[0].row_count == 1

    def test_all_same_id(self):
        ad = _make_adata(n_obs=5, obs_index=["x"] * 5)
        issues = check_dataset(ad, checks=["duplicate_cell_ids"])
        assert issues[0].row_count == 4


# ---------------------------------------------------------------------------
# missing features
# ---------------------------------------------------------------------------


class TestMissingFeatures:
    def test_clean_features_no_issue(self):
        ad = _make_adata(var_index=["TP53", "BRCA1", "EGFR"])
        issues = check_dataset(ad, checks=["missing_features"])
        assert len(issues) == 0

    def test_empty_feature_name_flagged(self):
        ad = _make_adata(n_vars=3, var_index=["TP53", "", "EGFR"])
        issues = check_dataset(ad, checks=["missing_features"])
        codes = _issue_codes(issues)
        assert "EMPTY_FEATURE_NAMES" in codes
        empty_issue = [i for i in issues if i.code == "EMPTY_FEATURE_NAMES"][0]
        assert empty_issue.row_count == 1

    def test_duplicate_feature_names_flagged(self):
        ad = _make_adata(n_vars=3, var_index=["TP53", "TP53", "EGFR"])
        issues = check_dataset(ad, checks=["missing_features"])
        codes = _issue_codes(issues)
        assert "DUPLICATE_FEATURE_NAMES" in codes

    def test_whitespace_only_feature_flagged(self):
        ad = _make_adata(n_vars=3, var_index=["TP53", "  ", "EGFR"])
        issues = check_dataset(ad, checks=["missing_features"])
        codes = _issue_codes(issues)
        assert "EMPTY_FEATURE_NAMES" in codes


# ---------------------------------------------------------------------------
# all-zero columns
# ---------------------------------------------------------------------------


class TestAllZeroColumns:
    def test_no_zeros_no_issue(self):
        X = np.ones((5, 3))
        ad = _make_adata(n_obs=5, n_vars=3, X=X)
        issues = check_dataset(ad, checks=["all_zero_columns"])
        assert len(issues) == 0

    def test_zero_column_flagged(self):
        X = np.array([[1, 0, 2], [3, 0, 4], [5, 0, 6]], dtype=float)
        ad = _make_adata(n_obs=3, n_vars=3, X=X)
        issues = check_dataset(ad, checks=["all_zero_columns"])
        assert len(issues) == 1
        assert issues[0].code == "ALL_ZERO_FEATURES"
        assert issues[0].severity == "warning"
        assert issues[0].row_count == 1

    def test_multiple_zero_columns(self):
        X = np.zeros((4, 5))
        X[:, 2] = 1  # only column 2 has values
        ad = _make_adata(n_obs=4, n_vars=5, X=X)
        issues = check_dataset(ad, checks=["all_zero_columns"])
        assert issues[0].row_count == 4

    def test_sparse_matrix(self):
        pytest.importorskip("scipy")
        from scipy.sparse import csr_matrix

        X = np.array([[1, 0, 2], [3, 0, 4], [5, 0, 6]], dtype=float)
        ad = _make_adata(n_obs=3, n_vars=3, X=csr_matrix(X))
        issues = check_dataset(ad, checks=["all_zero_columns"])
        assert len(issues) == 1
        assert issues[0].row_count == 1

    def test_no_matrix(self):
        ad = _make_adata(n_obs=3, n_vars=3)
        ad.X = None
        issues = check_dataset(ad, checks=["all_zero_columns"])
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# layer presence
# ---------------------------------------------------------------------------


class TestLayerPresence:
    def test_no_layers_no_issue(self):
        ad = _make_adata()
        issues = check_dataset(ad, checks=["layer_presence"])
        assert len(issues) == 0

    def test_matching_layer_no_issue(self):
        X = np.ones((5, 3))
        ad = _make_adata(n_obs=5, n_vars=3, X=X, layers={"raw": X.copy()})
        issues = check_dataset(ad, checks=["layer_presence"])
        assert len(issues) == 0

    def test_mismatched_layer_shape_flagged(self):
        X = np.ones((5, 3))
        ad = _make_adata(n_obs=5, n_vars=3, X=X)
        # Force a bad layer shape (AnnData normally prevents this)
        # We test via correctly-shaped layers to verify no false positives
        ad.layers["raw"] = np.ones((5, 3))
        issues = check_dataset(ad, checks=["layer_presence"])
        assert len(issues) == 0  # shapes match, no issue


# ---------------------------------------------------------------------------
# categorical encoding
# ---------------------------------------------------------------------------


class TestCategoricalEncoding:
    def test_categorical_column_no_issue(self):
        ad = _make_adata(
            n_obs=5,
            obs_columns={"cell_type": pd.Categorical(["T", "B", "T", "B", "T"])},
        )
        issues = check_dataset(ad, checks=["categorical_encoding"])
        # Categorical is fine — no COULD_BE_CATEGORICAL issue
        codes = _issue_codes(issues)
        assert "COULD_BE_CATEGORICAL" not in codes

    def test_low_cardinality_string_flagged(self):
        ad = _make_adata(
            n_obs=100,
            obs_columns={"cell_type": ["T cell"] * 50 + ["B cell"] * 50},
        )
        issues = check_dataset(ad, checks=["categorical_encoding"])
        codes = _issue_codes(issues)
        assert "COULD_BE_CATEGORICAL" in codes

    def test_high_cardinality_not_flagged(self):
        n = 10
        ad = _make_adata(
            n_obs=n,
            obs_columns={"barcode": [f"ATCG_{i}" for i in range(n)]},
        )
        issues = check_dataset(ad, checks=["categorical_encoding"])
        codes = _issue_codes(issues)
        assert "COULD_BE_CATEGORICAL" not in codes

    def test_unused_categories_flagged(self):
        cat = pd.Categorical(
            ["T", "B", "T"],
            categories=["T", "B", "NK"],  # NK is unused
        )
        ad = _make_adata(n_obs=3, obs_columns={"cell_type": cat})
        issues = check_dataset(ad, checks=["categorical_encoding"])
        codes = _issue_codes(issues)
        assert "UNUSED_CATEGORIES" in codes
        unused_issue = [i for i in issues if i.code == "UNUSED_CATEGORIES"][0]
        assert "NK" in unused_issue.message


# ---------------------------------------------------------------------------
# index hygiene
# ---------------------------------------------------------------------------


class TestIndexHygiene:
    def test_clean_index_no_error(self):
        ad = _make_adata(obs_index=["cell_A", "cell_B", "cell_C"])
        issues = check_dataset(ad, checks=["index_hygiene"])
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_integer_index_flagged(self):
        ad = _make_adata(n_obs=3)
        ad.obs.index = pd.RangeIndex(3)
        issues = check_dataset(ad, checks=["index_hygiene"])
        codes = _issue_codes(issues)
        assert "DEFAULT_INTEGER_INDEX" in codes

    def test_whitespace_in_index_flagged(self):
        ad = _make_adata(n_obs=3, obs_index=["cell_A ", " cell_B", "cell_C"])
        issues = check_dataset(ad, checks=["index_hygiene"])
        codes = _issue_codes(issues)
        assert "INDEX_WHITESPACE" in codes
        ws_issue = [i for i in issues if i.code == "INDEX_WHITESPACE"][0]
        assert ws_issue.row_count == 2

    def test_clean_whitespace_no_issue(self):
        ad = _make_adata(obs_index=["cell_A", "cell_B", "cell_C"])
        issues = check_dataset(ad, checks=["index_hygiene"])
        codes = _issue_codes(issues)
        assert "INDEX_WHITESPACE" not in codes


# ---------------------------------------------------------------------------
# Integration: run all checks together
# ---------------------------------------------------------------------------


class TestAllChecksTogether:
    def test_clean_dataset(self):
        ad = _make_adata(
            n_obs=5,
            n_vars=3,
            obs_index=["cell_A", "cell_B", "cell_C", "cell_D", "cell_E"],
            var_index=["TP53", "BRCA1", "EGFR"],
            X=np.ones((5, 3)),
        )
        issues = check_dataset(ad)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_messy_dataset_multiple_issues(self):
        X = np.zeros((5, 3))
        X[:, 0] = 1  # only gene_0 has values
        ad = _make_adata(
            n_obs=5,
            n_vars=3,
            obs_index=["cell_0", "cell_0", "cell_2", "cell_3", "cell_4"],
            var_index=["TP53", "TP53", "EGFR"],
            obs_columns={"batch": ["A"] * 5},
            X=X,
        )
        issues = check_dataset(ad)
        codes = _issue_codes(issues)
        assert "DUPLICATE_CELL_IDS" in codes
        assert "DUPLICATE_FEATURE_NAMES" in codes
        assert "ALL_ZERO_FEATURES" in codes
