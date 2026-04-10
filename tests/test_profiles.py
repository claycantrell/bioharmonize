import pytest

import bioharmonize as bh


class TestProfileLookup:
    def test_returns_single_cell_human(self):
        prof = bh.profile("single_cell_human")
        assert prof.name == "single_cell_human"
        assert "cell_type" in prof.canonical_columns
        assert "sex" in prof.value_normalizers

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            bh.profile("nonexistent")

    def test_profile_is_frozen(self):
        prof = bh.profile("single_cell_human")
        with pytest.raises(AttributeError):
            prof.name = "something_else"

    def test_required_columns_by_level(self):
        prof = bh.profile("single_cell_human")
        assert len(prof.required_columns("minimal")) == 0
        assert "cell_type" in prof.required_columns("standard")
        assert "donor_id" in prof.required_columns("strict")
