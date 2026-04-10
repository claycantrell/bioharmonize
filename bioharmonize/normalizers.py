from __future__ import annotations

from typing import Callable

ValueNormalizer = Callable[[str], str | None]


def make_map_normalizer(mapping: dict[str, str]) -> ValueNormalizer:
    lowered = {k.lower().strip(): v for k, v in mapping.items()}

    def normalize(value: str) -> str | None:
        return lowered.get(value.lower().strip())

    return normalize


SEX_MAP: dict[str, str] = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
    "unknown": "unknown",
}

CONDITION_MAP: dict[str, str] = {
    "ctrl": "control",
    "control": "control",
    "healthy": "control",
    "case": "disease",
    "disease": "disease",
}

ASSAY_MAP: dict[str, str] = {
    "10x": "10x",
    "10x genomics": "10x",
    "10x 3'": "10x",
    "10x 5'": "10x",
    "smart-seq2": "smart-seq2",
    "smartseq2": "smart-seq2",
}

SINGLE_CELL_HUMAN_NORMALIZERS: dict[str, ValueNormalizer] = {
    "sex": make_map_normalizer(SEX_MAP),
    "condition": make_map_normalizer(CONDITION_MAP),
    "assay": make_map_normalizer(ASSAY_MAP),
}
