"""Evidence and schema contracts for the live Section 199A SSTB crosswalk."""

from __future__ import annotations

import copy
import json
import re
from importlib.resources import files

import pytest

from populace.build.us_runtime.qbi_simulation import (
    QBI_SIMULATION_V2,
    load_qbi_simulation_assumptions,
    load_sstb_crosswalk,
    parse_sstb_crosswalk,
)

_HOSPITAL_AND_FACILITY_CODES = {"8191", "8192", "8270", "8290"}


def _payload() -> dict[str, object]:
    resource = files("populace.build.us").joinpath("sstb_crosswalk_v1.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _expected_probability(entry: dict[str, object], *, family: str) -> float:
    if entry["classification"] == "clear_sstb":
        return 1.0
    code = str(entry["census_code"])
    rationale = str(entry["rationale"]).lower()
    if "lean non-sstb" in rationale:
        return 0.10
    if family == "industry" and code in _HOSPITAL_AND_FACILITY_CODES:
        return 0.25
    if family == "occupation" and 3300 <= int(code) <= 3630:
        return 0.20
    return 0.30


def test_live_crosswalk_preserves_both_maps_meta_and_probability_tiers() -> None:
    payload = _payload()
    crosswalk = load_sstb_crosswalk("sstb_crosswalk_v1.json")
    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_V2)

    assert payload["status"] == "live"
    assert crosswalk.status == "live"
    assert len(crosswalk.industry_entries) == 27
    assert len(crosswalk.occupation_entries) == 101
    assert assumptions.sstb_classification.crosswalk_resource == (
        "sstb_crosswalk_v1.json"
    )
    assert assumptions.sstb_classification.occupation_column == "PEIOOCC"
    assert assumptions.sstb_classification.industry_column is None
    assert any(
        "reputation_or_skill" in note
        and "income, not an industry or occupation" in note
        for note in payload["meta"]["wiring_notes"]
    )
    assert "26 CFR 1.199A-5(b)" in payload["meta"]["legal_basis"]

    expected_counts = {
        "industry": {0.10: 3, 0.25: 4, 0.30: 7, 1.0: 13},
        "occupation": {0.10: 4, 0.20: 20, 0.30: 24, 1.0: 53},
    }
    for family, resource_key in (
        ("industry", "industry_2017"),
        ("occupation", "occupation_2018"),
    ):
        counts: dict[float, int] = {}
        for raw_entry in payload[resource_key]:
            entry = dict(raw_entry)
            probability = float(entry["probability"])
            assert probability == _expected_probability(entry, family=family)
            counts[probability] = counts.get(probability, 0) + 1
            if entry["classification"] == "ambiguous":
                assert entry["provisional"] is True
                assert "26 CFR 1.199A-5(c)(1)" in entry["basis"]
                assert "less than 10% SSTB receipts" in entry["basis"]
            else:
                assert "provisional" not in entry
                assert "basis" not in entry
        assert counts == expected_counts[family]

    occupation_probabilities = crosswalk.mapping_for("occupation")
    assert occupation_probabilities[3601] == 0.10
    assert occupation_probabilities[3602] == 0.10
    assert occupation_probabilities[3300] == 0.20


def test_live_crosswalk_occupation_codes_use_2018_four_digit_format() -> None:
    occupations = _payload()["occupation_2018"]
    codes = [entry["census_code"] for entry in occupations]

    assert len(codes) == len(set(codes)) == 101
    assert all(re.fullmatch(r"[0-9]{4}", code) for code in codes)
    assert {"0120", "0500", "0800"} <= set(codes)


def test_explicit_non_sstb_documentation_is_zero_probability() -> None:
    payload = _payload()

    for key, expected_count in (
        ("industry_explicit_nonsstb_neighbors", 10),
        ("occupation_explicit_nonsstb_notes", 11),
    ):
        entries = payload[key]
        assert len(entries) == expected_count
        assert all(entry["probability"] == 0.0 for entry in entries)


def test_live_crosswalk_rejects_an_ambiguous_prior_without_basis() -> None:
    payload = copy.deepcopy(_payload())
    ambiguous = next(
        entry
        for entry in payload["occupation_2018"]
        if entry["classification"] == "ambiguous"
    )
    del ambiguous["basis"]

    with pytest.raises(ValueError, match="keys must match"):
        parse_sstb_crosswalk(payload)
