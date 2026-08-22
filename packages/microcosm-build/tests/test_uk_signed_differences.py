"""The committed signed-differences register and its loader.

The register is what makes "anything differing that is not signed is a defect"
enforceable, so the loader has to be strict about the vocabulary and the
scoping, and the committed file has to stay internally coherent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcosm.build.uk_runtime.signed_differences import (
    SIGNED_DIFFERENCE_CLASSES,
    SIGNED_DIFFERENCE_EXPECTATIONS,
    SIGNED_DIFFERENCE_SURFACES,
    UKSignedDifference,
    UKSignedDifferenceRegister,
    load_uk_spine_swap_signed_differences,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _valid_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": "example-difference",
        "class": "mechanism_change",
        "scope": {
            "surface": "nonzero_shares",
            "columns": ["some_column"],
            "entities": ["household"],
        },
        "expectation": "column_differs",
        "magnitude_evidence": "A disclosure-safe statement of the magnitude.",
        "evidence": "experiments/686-uk-spine-swap-receipts.md#r0",
        "adjudicator": "juaristi22",
        "adjudicated_on": "2026-08-22",
    }
    entry.update(overrides)
    return entry


def _write(tmp_path: Path, payload: object) -> str:
    path = tmp_path / "register.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestCommittedRegister:
    def test_committed_register_loads(self) -> None:
        register = load_uk_spine_swap_signed_differences()
        assert register.schema_version == 1
        assert register.differences
        assert register.scope_note

    def test_committed_entries_are_precisely_scoped(self) -> None:
        # A surface-wide entry (empty columns) signs every column on that
        # surface. That is a real capability for entity_counts, but on a
        # column surface it would sign away defects wholesale.
        register = load_uk_spine_swap_signed_differences()
        for difference in register.differences:
            if difference.surface in {"nonzero_shares", "weighted_totals"}:
                assert difference.columns, (
                    f"{difference.id} signs a column surface without naming "
                    "columns; it would absorb unrelated divergences."
                )

    def test_committed_evidence_anchors_point_at_a_real_file(self) -> None:
        register = load_uk_spine_swap_signed_differences()
        for difference in register.differences:
            relative = difference.evidence.split("#", 1)[0]
            assert (REPO_ROOT / relative).is_file(), (
                f"{difference.id} cites missing evidence file {relative}"
            )

    def test_no_committed_entry_expires(self) -> None:
        payload = json.loads(
            (
                REPO_ROOT
                / "packages/microcosm-build/src/microcosm/build/uk"
                / "spine_swap_signed_differences.json"
            ).read_text(encoding="utf-8")
        )
        for entry in payload["differences"]:
            assert "expires_on" not in entry


class TestLookup:
    def test_matching_finds_the_signing_entry(self) -> None:
        register = load_uk_spine_swap_signed_differences()
        found = register.matching(
            surface="nonzero_shares", column="water_and_sewerage_charges"
        )
        assert found is not None
        assert found.id == "scottish-water-incumbent-nan-zeroing"

    def test_a_column_signed_on_one_surface_is_not_signed_on_another(self) -> None:
        # The Scottish level change moves weighted totals but deliberately not
        # the nonzero share, and the share entry is a different adjudication.
        register = load_uk_spine_swap_signed_differences()
        weighted = register.matching(
            surface="weighted_totals", column="water_and_sewerage_charges"
        )
        assert weighted is not None
        assert weighted.id == "scottish-water-sewerage-successor-level"
        assert register.matching(surface="entity_counts", column="household") is None

    def test_unsigned_column_returns_none(self) -> None:
        register = load_uk_spine_swap_signed_differences()
        assert (
            register.matching(surface="nonzero_shares", column="employment_income")
            is None
        )

    def test_surface_wide_entry_covers_any_column(self) -> None:
        entry = UKSignedDifference(
            id="counts",
            difference_class="rng_stream",
            surface="entity_counts",
            expectation="count_differs",
            columns=(),
            entities=("person",),
            magnitude_evidence="evidence",
            evidence="experiments/686-uk-spine-swap-receipts.md",
            adjudicator="juaristi22",
            adjudicated_on="2026-08-22",
        )
        assert entry.covers(surface="entity_counts", column="person")
        assert entry.covers(surface="entity_counts", column="benunit")
        assert not entry.covers(surface="nonzero_shares", column="person")


class TestValidation:
    def test_duplicate_ids_are_refused(self, tmp_path: Path) -> None:
        payload = {
            "schema_version": 1,
            "scope_note": "note",
            "differences": [_valid_entry(), _valid_entry()],
        }
        with pytest.raises(ValueError, match="unique"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_expiry_is_refused(self, tmp_path: Path) -> None:
        payload = {
            "schema_version": 1,
            "scope_note": "note",
            "differences": [_valid_entry(expires_on="2027-01-01")],
        }
        with pytest.raises(ValueError, match="do not expire"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("class", "not_a_class", "must be one of"),
            ("expectation", "not_an_expectation", "must be one of"),
            ("id", "Not Kebab Case", "kebab-case"),
            ("adjudicated_on", "22-08-2026", "ISO date"),
            ("magnitude_evidence", "", "non-empty string"),
            ("adjudicator", "", "non-empty string"),
            ("evidence", "", "non-empty string"),
        ],
    )
    def test_field_validation(
        self, tmp_path: Path, field: str, value: str, match: str
    ) -> None:
        payload = {
            "schema_version": 1,
            "scope_note": "note",
            "differences": [_valid_entry(**{field: value})],
        }
        with pytest.raises(ValueError, match=match):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_unknown_surface_is_refused(self, tmp_path: Path) -> None:
        payload = {
            "schema_version": 1,
            "scope_note": "note",
            "differences": [
                _valid_entry(scope={"surface": "made_up", "columns": ["c"]})
            ],
        }
        with pytest.raises(ValueError, match="must be one of"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_unsupported_schema_version_is_refused(self, tmp_path: Path) -> None:
        payload = {"schema_version": 2, "scope_note": "note", "differences": []}
        with pytest.raises(ValueError, match="schema_version"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_missing_scope_is_refused(self, tmp_path: Path) -> None:
        entry = _valid_entry()
        del entry["scope"]
        payload = {"schema_version": 1, "scope_note": "note", "differences": [entry]}
        with pytest.raises(ValueError, match="scope"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_columns_must_be_a_list_of_strings(self, tmp_path: Path) -> None:
        payload = {
            "schema_version": 1,
            "scope_note": "note",
            "differences": [
                _valid_entry(scope={"surface": "nonzero_shares", "columns": "col"})
            ],
        }
        with pytest.raises(ValueError, match="list of strings"):
            load_uk_spine_swap_signed_differences(_write(tmp_path, payload))

    def test_vocabularies_are_disjoint_and_populated(self) -> None:
        assert SIGNED_DIFFERENCE_CLASSES
        assert SIGNED_DIFFERENCE_SURFACES
        assert SIGNED_DIFFERENCE_EXPECTATIONS
        assert not SIGNED_DIFFERENCE_CLASSES & SIGNED_DIFFERENCE_SURFACES

    def test_register_rejects_duplicate_ids_at_construction(self) -> None:
        entry = UKSignedDifference(
            id="same",
            difference_class="vintage",
            surface="nonzero_shares",
            expectation="column_differs",
            columns=("a",),
            entities=("household",),
            magnitude_evidence="evidence",
            evidence="experiments/686-uk-spine-swap-receipts.md",
            adjudicator="juaristi22",
            adjudicated_on="2026-08-22",
        )
        with pytest.raises(ValueError, match="unique"):
            UKSignedDifferenceRegister(differences=(entry, entry), scope_note="note")
