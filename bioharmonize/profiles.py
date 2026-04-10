from __future__ import annotations

from dataclasses import dataclass, field

from .aliases import SINGLE_CELL_HUMAN_COLUMN_ALIASES
from .normalizers import SINGLE_CELL_HUMAN_NORMALIZERS, ValueNormalizer
from .specs import ColumnSpec


@dataclass(frozen=True)
class Profile:
    name: str
    canonical_columns: dict[str, ColumnSpec]
    column_aliases: dict[str, str]
    value_normalizers: dict[str, ValueNormalizer]
    required_columns_minimal: tuple[str, ...] = ()
    required_columns_standard: tuple[str, ...] = ()
    required_columns_strict: tuple[str, ...] = ()

    def required_columns(self, level: str) -> tuple[str, ...]:
        if level == "strict":
            return self.required_columns_strict
        if level == "standard":
            return self.required_columns_standard
        return self.required_columns_minimal


SINGLE_CELL_HUMAN = Profile(
    name="single_cell_human",
    canonical_columns={
        "cell_type": ColumnSpec(
            name="cell_type",
            dtype="string",
            description="Cell type annotation",
        ),
        "condition": ColumnSpec(
            name="condition",
            dtype="string",
            description="Experimental condition (e.g. control, disease)",
        ),
        "sample_id": ColumnSpec(
            name="sample_id",
            dtype="string",
            description="Sample identifier",
        ),
        "donor_id": ColumnSpec(
            name="donor_id",
            dtype="string",
            description="Donor/patient identifier",
        ),
        "batch_id": ColumnSpec(
            name="batch_id",
            dtype="string",
            description="Batch identifier",
        ),
        "tissue": ColumnSpec(
            name="tissue",
            dtype="string",
            description="Tissue or organ of origin",
        ),
        "sex": ColumnSpec(
            name="sex",
            dtype="string",
            description="Biological sex",
            allowed_values=("male", "female", "unknown"),
        ),
        "assay": ColumnSpec(
            name="assay",
            dtype="string",
            description="Sequencing assay type",
        ),
        "species": ColumnSpec(
            name="species",
            dtype="string",
            description="Species",
        ),
        "disease": ColumnSpec(
            name="disease",
            dtype="string",
            description="Disease name",
        ),
        "platform": ColumnSpec(
            name="platform",
            dtype="string",
            description="Sequencing platform",
        ),
    },
    column_aliases=SINGLE_CELL_HUMAN_COLUMN_ALIASES,
    value_normalizers=SINGLE_CELL_HUMAN_NORMALIZERS,
    required_columns_minimal=(),
    required_columns_standard=("cell_type", "sample_id"),
    required_columns_strict=("cell_type", "sample_id", "donor_id", "condition", "sex"),
)

_BUILTIN_PROFILES: dict[str, Profile] = {
    "single_cell_human": SINGLE_CELL_HUMAN,
}


def resolve_profile(profile: str | Profile) -> Profile:
    if isinstance(profile, Profile):
        return profile
    if profile in _BUILTIN_PROFILES:
        return _BUILTIN_PROFILES[profile]
    raise ValueError(f"Unknown profile: {profile!r}. Available: {list(_BUILTIN_PROFILES)}")
