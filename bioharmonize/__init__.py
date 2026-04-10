"""bioharmonize - Normalize and validate biological study metadata."""

from .anndata import patch_anndata
from .api import clean_obs, inspect, preflight, repair, validate, validate_obs
from .changes import Change
from .io import read_data, read_h5ad, read_obs
from .issues import Issue
from .preflight import TaskProfile, list_tasks, resolve_task
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
    "read_data",
    "read_obs",
    "read_h5ad",
    "profile",
    # Types
    "Profile",
    "TaskProfile",
    "list_tasks",
    "resolve_task",
    "Report",
    "Issue",
    "Change",
]
