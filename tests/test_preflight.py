"""Tests for preflight task profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bioharmonize as bh
from bioharmonize.preflight import (
    CELL_TYPE_ANNOTATION,
    CLUSTERING,
    DIFFERENTIAL_EXPRESSION,
    INTEGRATION,
    TaskProfile,
    _check_cells_per_batch,
    _check_cells_per_condition,
    _check_counts_layer_exists,
    _check_min_cells,
    _check_x_exists,
    _check_x_is_normalized,
    _check_x_is_raw_counts,
    _check_x_sparsity,
    _x_is_likely_raw_counts,
    list_tasks,
    resolve_task,
    run_preflight,
)

try:
    import anndata

    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

try:
    from scipy import sparse

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from click.testing import CliRunner

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False


# ---------------------------------------------------------------------------
# Task profile registry
# ---------------------------------------------------------------------------


class TestTaskRegistry:
    def test_list_tasks_returns_all_four(self):
        tasks = list_tasks()
        assert tasks == [
            "cell_type_annotation",
            "clustering",
            "differential_expression",
            "integration",
        ]

    def test_resolve_task_by_name(self):
        tp = resolve_task("clustering")
        assert tp.name == "clustering"

    def test_resolve_task_passthrough(self):
        tp = resolve_task(CLUSTERING)
        assert tp is CLUSTERING

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown task profile"):
            resolve_task("nonexistent")

    def test_task_profiles_are_frozen(self):
        with pytest.raises(AttributeError):
            CLUSTERING.name = "something_else"


# ---------------------------------------------------------------------------
# Public API: bh.preflight()
# ---------------------------------------------------------------------------


class TestPreflightAPI:
    def test_returns_report(self):
        df = pd.DataFrame({"condition": ["ctrl", "disease"], "sample_id": ["s1", "s2"]})
        report = bh.preflight(df, "differential_expression")
        assert isinstance(report, bh.Report)
        assert report.profile_name == "preflight:differential_expression"
        assert report.validation_level == "preflight"

    def test_no_changes_in_report(self):
        df = pd.DataFrame({"batch_id": ["b1", "b2"]})
        report = bh.preflight(df, "integration")
        assert report.changes == []

    def test_cleaned_is_original_df(self):
        df = pd.DataFrame({"x": [1, 2]})
        report = bh.preflight(df, "clustering")
        assert report.cleaned is df


# ---------------------------------------------------------------------------
# Clustering preflight
# ---------------------------------------------------------------------------


class TestClusteringPreflight:
    def test_pass_with_batch_and_sample(self):
        df = pd.DataFrame({
            "batch_id": ["b1", "b1", "b2", "b2"],
            "sample_id": ["s1", "s1", "s2", "s2"],
        })
        issues = run_preflight(df, "clustering")
        assert len(issues) == 0

    def test_warns_missing_batch(self):
        df = pd.DataFrame({"sample_id": ["s1", "s2"]})
        issues = run_preflight(df, "clustering")
        codes = [(i.code, i.column) for i in issues]
        assert ("PREFLIGHT_MISSING_RECOMMENDED", "batch_id") in codes

    def test_warns_missing_sample(self):
        df = pd.DataFrame({"batch_id": ["b1", "b2"]})
        issues = run_preflight(df, "clustering")
        codes = [(i.code, i.column) for i in issues]
        assert ("PREFLIGHT_MISSING_RECOMMENDED", "sample_id") in codes

    def test_single_batch_error(self):
        df = pd.DataFrame({"batch_id": ["b1", "b1", "b1"]})
        issues = run_preflight(df, "clustering")
        codes = [i.code for i in issues]
        assert "SINGLE_BATCH" in codes

    def test_no_required_columns(self):
        """Clustering has no required columns — should only get warnings."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        issues = run_preflight(df, "clustering")
        for issue in issues:
            assert issue.severity in ("warning", "info")


# ---------------------------------------------------------------------------
# Differential expression preflight
# ---------------------------------------------------------------------------


class TestDifferentialExpressionPreflight:
    def test_pass_with_all_columns(self):
        df = pd.DataFrame({
            "condition": ["ctrl", "ctrl", "disease", "disease"],
            "sample_id": ["s1", "s2", "s3", "s4"],
            "donor_id": ["d1", "d2", "d3", "d4"],
        })
        issues = run_preflight(df, "differential_expression")
        assert len(issues) == 0

    def test_error_missing_condition(self):
        df = pd.DataFrame({"sample_id": ["s1", "s2"]})
        issues = run_preflight(df, "differential_expression")
        errors = [i for i in issues if i.severity == "error"]
        assert any(i.code == "PREFLIGHT_MISSING_REQUIRED" and i.column == "condition" for i in errors)

    def test_error_single_group(self):
        df = pd.DataFrame({"condition": ["ctrl", "ctrl", "ctrl"]})
        issues = run_preflight(df, "differential_expression")
        codes = [i.code for i in issues]
        assert "INSUFFICIENT_GROUPS" in codes

    def test_warn_low_replicates(self):
        df = pd.DataFrame({
            "condition": ["ctrl", "disease"],
            "sample_id": ["s1", "s2"],
        })
        issues = run_preflight(df, "differential_expression")
        codes = [i.code for i in issues]
        assert "LOW_REPLICATES" in codes

    def test_no_low_replicate_warn_when_adequate(self):
        df = pd.DataFrame({
            "condition": ["ctrl", "ctrl", "disease", "disease"],
            "sample_id": ["s1", "s2", "s3", "s4"],
        })
        issues = run_preflight(df, "differential_expression")
        codes = [i.code for i in issues]
        assert "LOW_REPLICATES" not in codes

    def test_warns_missing_sample_and_donor(self):
        df = pd.DataFrame({"condition": ["ctrl", "disease"]})
        issues = run_preflight(df, "differential_expression")
        cols_warned = [i.column for i in issues if i.code == "PREFLIGHT_MISSING_RECOMMENDED"]
        assert "sample_id" in cols_warned
        assert "donor_id" in cols_warned


# ---------------------------------------------------------------------------
# Integration preflight
# ---------------------------------------------------------------------------


class TestIntegrationPreflight:
    def test_pass_with_multiple_batches(self):
        df = pd.DataFrame({
            "batch_id": ["b1", "b1", "b2", "b2"],
            "sample_id": ["s1", "s1", "s2", "s2"],
            "assay": ["10x", "10x", "10x", "10x"],
        })
        issues = run_preflight(df, "integration")
        assert len(issues) == 0

    def test_error_missing_batch(self):
        df = pd.DataFrame({"sample_id": ["s1", "s2"]})
        issues = run_preflight(df, "integration")
        errors = [i for i in issues if i.severity == "error"]
        assert any(i.code == "PREFLIGHT_MISSING_REQUIRED" and i.column == "batch_id" for i in errors)

    def test_error_single_batch(self):
        df = pd.DataFrame({"batch_id": ["b1", "b1", "b1"]})
        issues = run_preflight(df, "integration")
        codes = [i.code for i in issues]
        assert "SINGLE_BATCH" in codes

    def test_warns_missing_sample_and_assay(self):
        df = pd.DataFrame({"batch_id": ["b1", "b2"]})
        issues = run_preflight(df, "integration")
        cols_warned = [i.column for i in issues if i.code == "PREFLIGHT_MISSING_RECOMMENDED"]
        assert "sample_id" in cols_warned
        assert "assay" in cols_warned


# ---------------------------------------------------------------------------
# Cell type annotation preflight
# ---------------------------------------------------------------------------


class TestCellTypeAnnotationPreflight:
    def test_pass_with_recommended_columns(self):
        df = pd.DataFrame({
            "tissue": ["lung", "lung"],
            "species": ["human", "human"],
        })
        issues = run_preflight(df, "cell_type_annotation")
        assert len(issues) == 0

    def test_info_existing_cell_type(self):
        df = pd.DataFrame({
            "cell_type": ["T cell", "B cell"],
            "tissue": ["lung", "lung"],
            "species": ["human", "human"],
        })
        issues = run_preflight(df, "cell_type_annotation")
        codes = [i.code for i in issues]
        assert "EXISTING_ANNOTATION" in codes
        info_issues = [i for i in issues if i.code == "EXISTING_ANNOTATION"]
        assert info_issues[0].severity == "info"

    def test_warns_missing_tissue_and_species(self):
        df = pd.DataFrame({"x": [1, 2]})
        issues = run_preflight(df, "cell_type_annotation")
        cols_warned = [i.column for i in issues if i.code == "PREFLIGHT_MISSING_RECOMMENDED"]
        assert "tissue" in cols_warned
        assert "species" in cols_warned

    def test_no_required_columns(self):
        """Annotation has no required columns — should only get warnings/info."""
        df = pd.DataFrame({"x": [1, 2]})
        issues = run_preflight(df, "cell_type_annotation")
        for issue in issues:
            assert issue.severity in ("warning", "info")


# ---------------------------------------------------------------------------
# Custom TaskProfile
# ---------------------------------------------------------------------------


class TestCustomTaskProfile:
    def test_custom_profile(self):
        custom = TaskProfile(
            name="my_task",
            description="A custom task",
            required_columns=("gene_score",),
            recommended_columns=("batch_id",),
        )
        df = pd.DataFrame({"x": [1]})
        report = bh.preflight(df, custom)
        assert report.profile_name == "preflight:my_task"
        errors = [i for i in report.issues if i.severity == "error"]
        assert any(i.column == "gene_score" for i in errors)


# ---------------------------------------------------------------------------
# Integration with real GEO data
# ---------------------------------------------------------------------------


FIXTURES = Path(__file__).parent / "fixtures" / "real_data"
GEO_DATASETS = {
    "lung_cancer": FIXTURES / "geo_lung_cancer_obs.csv",
    "kidney_myeloid": FIXTURES / "geo_kidney_myeloid_obs.csv",
    "crc_leukocyte": FIXTURES / "geo_crc_leukocyte_obs.csv",
    "hsc_facs": FIXTURES / "geo_hsc_facs_obs.csv",
}


@pytest.fixture
def cleaned_lung_cancer():
    from bioharmonize.io import read_obs

    df = read_obs(GEO_DATASETS["lung_cancer"])
    return bh.clean_obs(df, profile="single_cell_human").cleaned


@pytest.fixture
def cleaned_kidney_myeloid():
    from bioharmonize.io import read_obs

    df = read_obs(GEO_DATASETS["kidney_myeloid"])
    return bh.clean_obs(df, profile="single_cell_human").cleaned


class TestPreflightOnRealData:
    def test_lung_cancer_clustering(self, cleaned_lung_cancer):
        """Lung cancer has sample_id but no batch_id after clean."""
        report = bh.preflight(cleaned_lung_cancer, "clustering")
        codes = [(i.code, i.column) for i in report.issues]
        assert ("PREFLIGHT_MISSING_RECOMMENDED", "batch_id") in codes

    def test_lung_cancer_de_missing_condition(self, cleaned_lung_cancer):
        """Lung cancer lacks condition — DE preflight should error."""
        report = bh.preflight(cleaned_lung_cancer, "differential_expression")
        errors = [i for i in report.issues if i.severity == "error"]
        assert any(i.column == "condition" for i in errors)

    def test_kidney_myeloid_integration_missing_batch(self, cleaned_kidney_myeloid):
        """Kidney myeloid has batch_id after clean. Check it passes required."""
        report = bh.preflight(cleaned_kidney_myeloid, "integration")
        missing_required = [
            i for i in report.issues
            if i.code == "PREFLIGHT_MISSING_REQUIRED" and i.column == "batch_id"
        ]
        assert len(missing_required) == 0


# ---------------------------------------------------------------------------
# AnnData-aware preflight checks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestRawCountsHeuristic:
    def test_integer_matrix_is_raw(self):
        X = np.array([[1, 0, 5], [0, 3, 2]])
        assert _x_is_likely_raw_counts(X) is True

    def test_float_matrix_is_not_raw(self):
        X = np.array([[1.5, 0.0, 2.3], [0.1, 3.7, 0.0]])
        assert _x_is_likely_raw_counts(X) is False

    def test_negative_values_not_raw(self):
        X = np.array([[-1, 0, 5], [0, 3, 2]])
        assert _x_is_likely_raw_counts(X) is False

    def test_empty_matrix_is_raw(self):
        X = np.array([]).reshape(0, 0)
        assert _x_is_likely_raw_counts(X) is True

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_sparse_integer_is_raw(self):
        X = sparse.csr_matrix(np.array([[1, 0, 5], [0, 3, 0]]))
        assert _x_is_likely_raw_counts(X) is True

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_sparse_float_is_not_raw(self):
        X = sparse.csr_matrix(np.array([[1.5, 0, 2.3], [0.1, 0, 0]]))
        assert _x_is_likely_raw_counts(X) is False


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestCheckXExists:
    def test_x_none(self):
        adata = anndata.AnnData(obs=pd.DataFrame(index=["c1"]))
        adata.X = None
        issues = _check_x_exists(adata)
        assert any(i.code == "MISSING_X_MATRIX" for i in issues)

    def test_x_empty(self):
        adata = anndata.AnnData(X=np.array([]).reshape(0, 5))
        issues = _check_x_exists(adata)
        assert any(i.code == "EMPTY_X_MATRIX" for i in issues)

    def test_x_present(self):
        adata = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))
        issues = _check_x_exists(adata)
        assert len(issues) == 0


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestCheckXNormalized:
    def test_raw_counts_warns(self):
        adata = anndata.AnnData(X=np.array([[1, 0, 5], [0, 3, 2]]))
        issues = _check_x_is_normalized(adata)
        assert any(i.code == "X_APPEARS_RAW" for i in issues)

    def test_normalized_no_warning(self):
        adata = anndata.AnnData(X=np.array([[1.5, 0.1], [0.3, 2.7]]))
        issues = _check_x_is_normalized(adata)
        assert len(issues) == 0

    def test_x_none_no_issue(self):
        adata = anndata.AnnData(obs=pd.DataFrame(index=["c1"]))
        adata.X = None
        issues = _check_x_is_normalized(adata)
        assert len(issues) == 0


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestCheckXRawCounts:
    def test_normalized_warns_no_counts_layer(self):
        adata = anndata.AnnData(X=np.array([[1.5, 0.1], [0.3, 2.7]]))
        issues = _check_x_is_raw_counts(adata)
        assert any(i.code == "X_NOT_RAW_COUNTS" for i in issues)

    def test_normalized_with_counts_layer_info(self):
        adata = anndata.AnnData(X=np.array([[1.5, 0.1], [0.3, 2.7]]))
        adata.layers["counts"] = np.array([[2, 0], [1, 5]])
        issues = _check_x_is_raw_counts(adata)
        assert any(i.code == "X_NOT_COUNTS_BUT_LAYER_EXISTS" for i in issues)
        assert all(i.severity == "info" for i in issues)

    def test_raw_counts_no_warning(self):
        adata = anndata.AnnData(X=np.array([[1, 0, 5], [0, 3, 2]]))
        issues = _check_x_is_raw_counts(adata)
        assert len(issues) == 0


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestCheckCountsLayerExists:
    def test_no_counts_layer_and_normalized_x(self):
        adata = anndata.AnnData(X=np.array([[1.5, 0.1], [0.3, 2.7]]))
        issues = _check_counts_layer_exists(adata)
        assert any(i.code == "NO_COUNTS_LAYER" for i in issues)

    def test_counts_layer_present(self):
        adata = anndata.AnnData(X=np.array([[1.5, 0.1], [0.3, 2.7]]))
        adata.layers["counts"] = np.array([[2, 0], [1, 5]])
        issues = _check_counts_layer_exists(adata)
        assert len(issues) == 0

    def test_x_is_raw_counts(self):
        adata = anndata.AnnData(X=np.array([[1, 0], [0, 5]]))
        issues = _check_counts_layer_exists(adata)
        assert len(issues) == 0


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestCheckXSparsity:
    def test_dense_matrix_reports_sparsity(self):
        adata = anndata.AnnData(X=np.array([[1, 0, 0], [0, 0, 2]]))
        issues = _check_x_sparsity(adata)
        assert len(issues) == 1
        assert issues[0].code == "X_SPARSITY"
        assert "dense" in issues[0].message

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_sparse_matrix_reports_sparsity(self):
        X = sparse.csr_matrix(np.array([[1, 0, 0], [0, 0, 2]]))
        adata = anndata.AnnData(X=X)
        issues = _check_x_sparsity(adata)
        assert len(issues) == 1
        assert issues[0].code == "X_SPARSITY"
        assert "sparse" in issues[0].message

    def test_x_none_no_issue(self):
        adata = anndata.AnnData(obs=pd.DataFrame(index=["c1"]))
        adata.X = None
        issues = _check_x_sparsity(adata)
        assert len(issues) == 0


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestCheckMinCells:
    def test_few_cells_warns(self):
        adata = anndata.AnnData(X=np.ones((10, 5)))
        issues = _check_min_cells(adata)
        assert any(i.code == "LOW_CELL_COUNT" for i in issues)

    def test_enough_cells_no_warning(self):
        adata = anndata.AnnData(X=np.ones((100, 5)))
        issues = _check_min_cells(adata)
        assert len(issues) == 0


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestCheckCellsPerCondition:
    def test_low_cells_per_condition(self):
        obs = pd.DataFrame(
            {"condition": ["ctrl", "ctrl", "disease"]},
            index=["c1", "c2", "c3"],
        )
        adata = anndata.AnnData(X=np.ones((3, 5)), obs=obs)
        issues = _check_cells_per_condition(adata)
        assert any(i.code == "LOW_CELLS_PER_CONDITION" for i in issues)

    def test_adequate_cells_per_condition(self):
        obs = pd.DataFrame(
            {"condition": ["ctrl"] * 5 + ["disease"] * 5},
            index=[f"c{i}" for i in range(10)],
        )
        adata = anndata.AnnData(X=np.ones((10, 5)), obs=obs)
        issues = _check_cells_per_condition(adata)
        assert len(issues) == 0

    def test_no_condition_column(self):
        adata = anndata.AnnData(X=np.ones((5, 5)))
        issues = _check_cells_per_condition(adata)
        assert len(issues) == 0


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestCheckCellsPerBatch:
    def test_low_cells_per_batch(self):
        obs = pd.DataFrame(
            {"batch_id": ["b1"] * 3 + ["b2"] * 3},
            index=[f"c{i}" for i in range(6)],
        )
        adata = anndata.AnnData(X=np.ones((6, 5)), obs=obs)
        issues = _check_cells_per_batch(adata)
        assert any(i.code == "LOW_CELLS_PER_BATCH" for i in issues)

    def test_adequate_cells_per_batch(self):
        obs = pd.DataFrame(
            {"batch_id": ["b1"] * 20 + ["b2"] * 20},
            index=[f"c{i}" for i in range(40)],
        )
        adata = anndata.AnnData(X=np.ones((40, 5)), obs=obs)
        issues = _check_cells_per_batch(adata)
        assert len(issues) == 0

    def test_no_batch_column(self):
        adata = anndata.AnnData(X=np.ones((5, 5)))
        issues = _check_cells_per_batch(adata)
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# AnnData-aware preflight via public API (bh.preflight)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestPreflightWithAnnData:
    def test_clustering_with_raw_counts_warns(self):
        """Clustering on raw counts should warn about normalization."""
        obs = pd.DataFrame(
            {"batch_id": ["b1"] * 50 + ["b2"] * 50},
            index=[f"c{i}" for i in range(100)],
        )
        adata = anndata.AnnData(X=np.random.randint(0, 100, (100, 50)), obs=obs)
        report = bh.preflight(adata, "clustering")
        codes = [i.code for i in report.issues]
        assert "X_APPEARS_RAW" in codes

    def test_clustering_with_normalized_data_no_raw_warn(self):
        """Clustering on normalized data should not warn about raw counts."""
        obs = pd.DataFrame(
            {"batch_id": ["b1"] * 50 + ["b2"] * 50},
            index=[f"c{i}" for i in range(100)],
        )
        adata = anndata.AnnData(X=np.random.randn(100, 50).astype(np.float32), obs=obs)
        report = bh.preflight(adata, "clustering")
        codes = [i.code for i in report.issues]
        assert "X_APPEARS_RAW" not in codes

    def test_de_with_normalized_x_warns(self):
        """DE on normalized X without counts layer should warn."""
        obs = pd.DataFrame(
            {
                "condition": ["ctrl"] * 5 + ["disease"] * 5,
                "sample_id": [f"s{i}" for i in range(10)],
                "donor_id": [f"d{i}" for i in range(10)],
            },
            index=[f"c{i}" for i in range(10)],
        )
        adata = anndata.AnnData(X=np.random.randn(10, 20).astype(np.float32), obs=obs)
        report = bh.preflight(adata, "differential_expression")
        codes = [i.code for i in report.issues]
        assert "X_NOT_RAW_COUNTS" in codes

    def test_de_with_raw_counts_no_warn(self):
        """DE on raw counts should not warn about normalization."""
        obs = pd.DataFrame(
            {
                "condition": ["ctrl"] * 5 + ["disease"] * 5,
                "sample_id": [f"s{i}" for i in range(10)],
                "donor_id": [f"d{i}" for i in range(10)],
            },
            index=[f"c{i}" for i in range(10)],
        )
        adata = anndata.AnnData(X=np.random.randint(0, 100, (10, 20)), obs=obs)
        report = bh.preflight(adata, "differential_expression")
        codes = [i.code for i in report.issues]
        assert "X_NOT_RAW_COUNTS" not in codes

    def test_integration_with_small_batches_warns(self):
        """Integration with very few cells per batch should warn."""
        obs = pd.DataFrame(
            {"batch_id": ["b1"] * 3 + ["b2"] * 3, "sample_id": ["s1"] * 3 + ["s2"] * 3},
            index=[f"c{i}" for i in range(6)],
        )
        adata = anndata.AnnData(X=np.ones((6, 5)), obs=obs)
        report = bh.preflight(adata, "integration")
        codes = [i.code for i in report.issues]
        assert "LOW_CELLS_PER_BATCH" in codes

    def test_dataframe_skips_adata_checks(self):
        """When passing a DataFrame (not AnnData), adata_checks are skipped."""
        df = pd.DataFrame({
            "batch_id": ["b1", "b1", "b2", "b2"],
            "sample_id": ["s1", "s1", "s2", "s2"],
        })
        report = bh.preflight(df, "clustering")
        codes = [i.code for i in report.issues]
        # No adata-level codes should appear
        assert "MISSING_X_MATRIX" not in codes
        assert "X_APPEARS_RAW" not in codes
        assert "X_SPARSITY" not in codes

    def test_missing_x_matrix_errors(self):
        """AnnData with no X matrix should error."""
        adata = anndata.AnnData(obs=pd.DataFrame({"batch_id": ["b1", "b2"]}, index=["c1", "c2"]))
        adata.X = None
        report = bh.preflight(adata, "clustering")
        codes = [i.code for i in report.issues]
        assert "MISSING_X_MATRIX" in codes

    def test_sparsity_reported(self):
        """Sparsity info should be reported for AnnData."""
        obs = pd.DataFrame(
            {"batch_id": ["b1"] * 50 + ["b2"] * 50},
            index=[f"c{i}" for i in range(100)],
        )
        adata = anndata.AnnData(X=np.random.randn(100, 50).astype(np.float32), obs=obs)
        report = bh.preflight(adata, "clustering")
        codes = [i.code for i in report.issues]
        assert "X_SPARSITY" in codes

    def test_custom_profile_no_adata_checks(self):
        """Custom TaskProfile without adata_checks should work fine with AnnData."""
        custom = TaskProfile(
            name="custom",
            description="A custom task",
            required_columns=(),
            recommended_columns=(),
        )
        adata = anndata.AnnData(X=np.ones((5, 3)))
        report = bh.preflight(adata, custom)
        # No adata-level issues since no adata_checks defined
        adata_codes = {"MISSING_X_MATRIX", "X_APPEARS_RAW", "X_SPARSITY", "LOW_CELL_COUNT"}
        for issue in report.issues:
            assert issue.code not in adata_codes


# ---------------------------------------------------------------------------
# CLI preflight command
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CLICK, reason="click not installed")
class TestCLIPreflight:
    def test_preflight_command_runs(self):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        csv_path = GEO_DATASETS["lung_cancer"]
        result = runner.invoke(_build_cli(), ["preflight", str(csv_path), "clustering"])
        assert "preflight:clustering" in result.output

    def test_preflight_exits_1_on_errors(self):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        csv_path = GEO_DATASETS["lung_cancer"]
        result = runner.invoke(_build_cli(), ["preflight", str(csv_path), "differential_expression"])
        # lung_cancer lacks condition → should get PREFLIGHT_MISSING_REQUIRED error → exit 1
        assert result.exit_code == 1

    def test_preflight_no_clean_flag(self):
        from bioharmonize.cli import _build_cli

        runner = CliRunner()
        csv_path = GEO_DATASETS["lung_cancer"]
        result = runner.invoke(_build_cli(), ["preflight", str(csv_path), "clustering", "--no-clean"])
        assert "preflight:clustering" in result.output
