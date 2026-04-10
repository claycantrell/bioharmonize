"""Tests for structural repair features: index deduplication, index hygiene, h5ad I/O."""

import numpy as np
import pandas as pd
import pytest

import bioharmonize as bh

try:
    import anndata

    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False


# ---------------------------------------------------------------------------
# repair() — deduplicate cell IDs
# ---------------------------------------------------------------------------


class TestRepairDeduplicateCellIds:
    def test_deduplicates_index(self):
        df = pd.DataFrame(
            {"cell_type": ["T cell", "B cell", "NK cell"]},
            index=["AACG", "AACG", "TTGG"],
        )
        report = bh.repair(df)
        assert report.cleaned.index.is_unique

    def test_dedup_appends_suffixes(self):
        df = pd.DataFrame(
            {"cell_type": ["T cell", "B cell", "NK cell"]},
            index=["AACG", "AACG", "TTGG"],
        )
        report = bh.repair(df)
        assert "AACG" in report.cleaned.index
        assert "AACG-1" in report.cleaned.index
        assert "TTGG" in report.cleaned.index

    def test_dedup_triple_duplicates(self):
        df = pd.DataFrame(
            {"cell_type": ["T", "B", "NK"]},
            index=["X", "X", "X"],
        )
        report = bh.repair(df)
        assert list(report.cleaned.index) == ["X", "X-1", "X-2"]

    def test_dedup_records_change(self):
        df = pd.DataFrame(
            {"cell_type": ["T", "B"]},
            index=["A", "A"],
        )
        report = bh.repair(df)
        dedup_changes = [c for c in report.changes if c.kind == "deduplicate_index"]
        assert len(dedup_changes) == 1
        assert dedup_changes[0].count == 1

    def test_no_dedup_when_unique(self):
        df = pd.DataFrame(
            {"cell_type": ["T", "B"]},
            index=["c0", "c1"],
        )
        report = bh.repair(df)
        dedup_changes = [c for c in report.changes if c.kind == "deduplicate_index"]
        assert len(dedup_changes) == 0

    def test_does_not_modify_original(self):
        df = pd.DataFrame(
            {"cell_type": ["T", "B"]},
            index=["A", "A"],
        )
        bh.repair(df)
        assert list(df.index) == ["A", "A"]


# ---------------------------------------------------------------------------
# repair() — fix index hygiene (whitespace, integer index)
# ---------------------------------------------------------------------------


class TestRepairIndexHygiene:
    def test_strips_index_whitespace(self):
        df = pd.DataFrame(
            {"cell_type": ["T", "B"]},
            index=["  AACG  ", "TTGG\t"],
        )
        report = bh.repair(df)
        assert list(report.cleaned.index) == ["AACG", "TTGG"]

    def test_strip_records_change(self):
        df = pd.DataFrame(
            {"cell_type": ["T"]},
            index=[" AACG "],
        )
        report = bh.repair(df)
        strip_changes = [c for c in report.changes if c.kind == "strip_index"]
        assert len(strip_changes) == 1
        assert strip_changes[0].count == 1

    def test_no_strip_when_clean(self):
        df = pd.DataFrame(
            {"cell_type": ["T"]},
            index=["AACG"],
        )
        report = bh.repair(df)
        strip_changes = [c for c in report.changes if c.kind == "strip_index"]
        assert len(strip_changes) == 0

    def test_replaces_integer_index(self):
        df = pd.DataFrame({"cell_type": ["T", "B", "NK"]})
        # Default RangeIndex (0, 1, 2)
        report = bh.repair(df)
        assert list(report.cleaned.index) == ["cell_0", "cell_1", "cell_2"]

    def test_integer_index_records_change(self):
        df = pd.DataFrame({"cell_type": ["T"]})
        report = bh.repair(df)
        gen_changes = [c for c in report.changes if c.kind == "generate_index"]
        assert len(gen_changes) == 1
        assert gen_changes[0].count == 1

    def test_no_generate_when_string_index(self):
        df = pd.DataFrame({"cell_type": ["T"]}, index=["c0"])
        report = bh.repair(df)
        gen_changes = [c for c in report.changes if c.kind == "generate_index"]
        assert len(gen_changes) == 0

    def test_whitespace_then_dedup(self):
        """Whitespace stripping may create duplicates — both should be repaired."""
        df = pd.DataFrame(
            {"cell_type": ["T", "B"]},
            index=["AACG", " AACG"],
        )
        report = bh.repair(df)
        assert report.cleaned.index.is_unique
        strip_changes = [c for c in report.changes if c.kind == "strip_index"]
        dedup_changes = [c for c in report.changes if c.kind == "deduplicate_index"]
        assert len(strip_changes) == 1
        assert len(dedup_changes) == 1


# ---------------------------------------------------------------------------
# repair() — structural repair with AnnData
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestRepairStructuralAnnData:
    def test_dedup_anndata_obs_index(self):
        ad = anndata.AnnData(
            X=np.zeros((3, 5)),
            obs=pd.DataFrame(
                {"cell_type": ["T", "B", "NK"]},
                index=["A", "A", "B"],
            ),
        )
        repaired_ad, report = bh.repair(ad)
        assert repaired_ad.obs.index.is_unique
        assert report.cleaned.index.is_unique

    def test_strip_whitespace_anndata(self):
        ad = anndata.AnnData(
            X=np.zeros((2, 3)),
            obs=pd.DataFrame(
                {"cell_type": ["T", "B"]},
                index=[" c0 ", "c1"],
            ),
        )
        repaired_ad, report = bh.repair(ad)
        assert "c0" in repaired_ad.obs.index
        assert " c0 " not in repaired_ad.obs.index

    def test_integer_index_anndata(self):
        ad = anndata.AnnData(
            X=np.zeros((2, 3)),
            obs=pd.DataFrame({"cell_type": ["T", "B"]}),
        )
        repaired_ad, report = bh.repair(ad)
        assert list(repaired_ad.obs.index) == ["cell_0", "cell_1"]

    def test_preserves_x_matrix_after_structural_repair(self):
        X = np.ones((3, 5))
        ad = anndata.AnnData(
            X=X.copy(),
            obs=pd.DataFrame(
                {"cell_type": ["T", "B", "NK"]},
                index=["A", "A", "B"],
            ),
        )
        repaired_ad, report = bh.repair(ad)
        assert repaired_ad.X.shape == (3, 5)
        assert np.all(repaired_ad.X == 1)


# ---------------------------------------------------------------------------
# io.py — read_h5ad and read_obs with .h5ad
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")
class TestReadH5ad:
    def test_read_h5ad_returns_anndata(self, tmp_path):
        ad = anndata.AnnData(
            X=np.zeros((3, 5)),
            obs=pd.DataFrame(
                {"cell_type": ["T", "B", "NK"]},
                index=["c0", "c1", "c2"],
            ),
            var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        )
        path = tmp_path / "test.h5ad"
        ad.write_h5ad(path)

        result = bh.read_h5ad(path)
        assert isinstance(result, anndata.AnnData)
        assert result.n_obs == 3
        assert result.n_vars == 5

    def test_read_obs_h5ad_returns_dataframe(self, tmp_path):
        ad = anndata.AnnData(
            X=np.zeros((3, 5)),
            obs=pd.DataFrame(
                {"cell_type": ["T", "B", "NK"], "sex": ["male", "female", "male"]},
                index=["c0", "c1", "c2"],
            ),
            var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        )
        path = tmp_path / "test.h5ad"
        ad.write_h5ad(path)

        result = bh.read_obs(path)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["cell_type", "sex"]
        assert len(result) == 3

    def test_read_obs_h5ad_preserves_index(self, tmp_path):
        ad = anndata.AnnData(
            X=np.zeros((2, 3)),
            obs=pd.DataFrame(
                {"cell_type": ["T", "B"]},
                index=["AACG_1", "TTGG_2"],
            ),
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
        )
        path = tmp_path / "test.h5ad"
        ad.write_h5ad(path)

        result = bh.read_obs(path)
        assert list(result.index) == ["AACG_1", "TTGG_2"]

    def test_read_h5ad_with_h5_extension(self, tmp_path):
        ad = anndata.AnnData(
            X=np.zeros((2, 3)),
            obs=pd.DataFrame({"cell_type": ["T", "B"]}, index=["c0", "c1"]),
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
        )
        path = tmp_path / "test.h5ad"
        ad.write_h5ad(path)
        # Rename to .h5
        h5_path = tmp_path / "test.h5"
        path.rename(h5_path)

        result = bh.read_obs(h5_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


@pytest.mark.skipif(HAS_ANNDATA, reason="only runs when anndata is NOT installed")
class TestReadH5adNoAnndata:
    def test_read_h5ad_raises_import_error(self, tmp_path):
        path = tmp_path / "test.h5ad"
        path.touch()
        with pytest.raises(ImportError, match="anndata is required"):
            bh.read_h5ad(path)
