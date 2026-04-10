from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .api import repair
from .profiles import Profile
from .report import Report

if TYPE_CHECKING:
    import anndata


def patch_anndata(
    ad: anndata.AnnData,
    profile: str | Profile = "single_cell_human",
    *,
    inplace: bool = False,
    **clean_kwargs: Any,
) -> anndata.AnnData | Report:
    try:
        import anndata as _anndata
    except ImportError:
        raise ImportError(
            "anndata is required for patch_anndata. "
            "Install it with: pip install bioharmonize[anndata]"
        ) from None

    report = repair(ad.obs, profile=profile, copy=True, **clean_kwargs)

    if inplace:
        ad.obs = report.cleaned
        return report
    else:
        new_ad = ad.copy()
        new_ad.obs = report.cleaned
        return new_ad, report
