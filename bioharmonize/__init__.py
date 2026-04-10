"""bioharmonize - Normalize and validate biological study metadata."""

from .anndata import patch_anndata
from .api import clean_obs, inspect, preflight, repair, validate, validate_obs
from .changes import Change
from .io import read_obs
from .issues import Issue
from .profiles import Profile, resolve_profile as profile
from .report import Report
from .sanity import check_dataset

__version__ = "0.1.0"

__all__ = [
    # Primary API (AnnData-first)
    "inspect",
    "validate",
    "repair",
    "preflight",
    # Backward-compatible aliases
    "clean_obs",
    "validate_obs",
    "check_dataset",
    # AnnData helper
    "patch_anndata",
    # Utilities
    "read_obs",
    "profile",
    # Types
    "Profile",
    "Report",
    "Issue",
    "Change",
]
