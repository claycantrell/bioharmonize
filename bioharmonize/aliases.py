from __future__ import annotations

SINGLE_CELL_HUMAN_COLUMN_ALIASES: dict[str, str] = {
    "celltype": "cell_type",
    "cell_type_annotation": "cell_type",
    "annotation": "cell_type",
    "disease_status": "condition",
    "status": "condition",
    "patient": "donor_id",
    "donor": "donor_id",
    "sample": "sample_id",
    "batch": "batch_id",
    "organ": "tissue",
    "gender": "sex",
}
