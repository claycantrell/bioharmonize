import pandas as pd
import pytest

try:
    import anndata

    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

import bioharmonize as bh

pytestmark = pytest.mark.skipif(not HAS_ANNDATA, reason="anndata not installed")


def _make_adata(**obs_kwargs) -> "anndata.AnnData":
    import anndata
    import numpy as np

    n = len(next(iter(obs_kwargs.values())))
    obs = pd.DataFrame(obs_kwargs, index=[f"cell_{i}" for i in range(n)])
    X = np.zeros((n, 5))
    return anndata.AnnData(X=X, obs=obs)


class TestPatchAnnData:
    def test_copy_mode_returns_tuple(self):
        ad = _make_adata(gender=["m", "f"], celltype=["T cell", "B cell"])
        result = bh.patch_anndata(ad)
        assert isinstance(result, tuple)
        new_ad, report = result
        assert "sex" in new_ad.obs.columns
        assert "cell_type" in new_ad.obs.columns
        # Original unchanged
        assert "gender" in ad.obs.columns

    def test_inplace_mode_returns_report(self):
        ad = _make_adata(gender=["m", "f"])
        result = bh.patch_anndata(ad, inplace=True)
        assert isinstance(result, bh.Report)
        assert "sex" in ad.obs.columns

    def test_preserves_shape(self):
        ad = _make_adata(sex=["m", "f", "unknown"])
        new_ad, report = bh.patch_anndata(ad)
        assert new_ad.shape == ad.shape
        assert list(new_ad.obs.index) == list(ad.obs.index)
