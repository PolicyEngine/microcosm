"""Every UK target held at an earlier period carries a declared reason (#1123)."""

from __future__ import annotations

from datetime import date

import pytest

from microcosm.build.uk_runtime.uprating_holds import (
    assert_uk_uprating_holds_declared,
    is_uprating_hold,
    load_uk_uprating_holds,
    uk_hold_is_signed,
)
from microcosm.calibrate import TargetRegistry, TargetSpec


def _spec(target_id: str, period: str, **metadata) -> TargetSpec:
    return TargetSpec(
        name=f"{target_id}@2025",
        entity="household",
        measure=target_id,
        value=1.0,
        period=2025,
        family="test",
        source="test",
        metadata={
            "contract_target_id": target_id,
            "ledger_fact_period": period,
            **metadata,
        },
    )


def _holds(*entries) -> dict:
    return {"schema_version": 1, "holds": list(entries)}


def _hold(target_id: str, kind: str = "in_year_snapshot", **fields) -> dict:
    return {
        "scope": "national",
        "kind": kind,
        "target_ids": [target_id],
        "reason": "test",
        "tracking": "test",
        "approved_by": "reviewer",
        "approved_on": "2026-10-01",
        "status": "approved",
        "expires_on": "2027-04-01",
        **fields,
    }


@pytest.mark.parametrize(
    ("period", "metadata", "held"),
    [
        ("2024", {}, True),
        ("2025-05", {}, True),
        ("2025", {}, False),
        ("2023", {"uprating_factor": "1.1"}, False),
        ("2025-12", {"fact_aggregation": "time_mean"}, False),
        ("2025-11", {"ledger_value_operation": "monthly_window_average"}, False),
        ("2025-12", {"ledger_value_operation": "calendar_year_average"}, False),
        ("2024", {"ledger_value_operation": "rolled_forward_by_ratio"}, False),
        # An applier that says it moved nothing leaves the row a hold.
        ("2021", {"uprating_factor": "1", "uprating_hold": "census_share_held"}, True),
        ("2025-11", {"ledger_value_operation": "latest_plateau"}, True),
        ("2025-03", {"period_match_policy": "source_window"}, False),
    ],
)
def test_a_hold_is_an_older_fact_no_index_moved(period, metadata, held):
    assert is_uprating_hold(_spec("t", period, **metadata), 2025) is held


def test_an_undeclared_hold_is_refused():
    registry = TargetRegistry([_spec("t", "2024")], country="uk")
    with pytest.raises(ValueError, match=r"undeclared \['t'\]"):
        assert_uk_uprating_holds_declared(
            registry,
            calibration_period=2025,
            evaluated_on=date(2026, 10, 7),
            scope="national",
            holds=_holds(),
        )


def test_a_declared_hold_is_receipted_by_kind():
    registry = TargetRegistry([_spec("t", "2025-10")], country="uk")
    receipt = assert_uk_uprating_holds_declared(
        registry,
        calibration_period=2025,
        evaluated_on=date(2026, 10, 7),
        scope="national",
        holds=_holds(_hold("t")),
    )
    assert receipt["by_kind"] == {"in_year_snapshot": ["t"]}


def test_an_expired_hold_is_refused():
    registry = TargetRegistry([_spec("t", "2024")], country="uk")
    with pytest.raises(ValueError, match=r"expired \['t'\]"):
        assert_uk_uprating_holds_declared(
            registry,
            calibration_period=2025,
            evaluated_on=date(2026, 10, 7),
            scope="national",
            holds=_holds(_hold("t", expires_on="2026-01-01")),
        )


def test_a_declaration_for_a_target_no_longer_held_is_stale():
    registry = TargetRegistry([_spec("t", "2025")], country="uk")
    with pytest.raises(ValueError, match=r"stale declarations \['t'\]"):
        assert_uk_uprating_holds_declared(
            registry,
            calibration_period=2025,
            evaluated_on=date(2026, 10, 7),
            scope="national",
            holds=_holds(_hold("t")),
        )


def _periodless(**metadata) -> TargetSpec:
    spec = _spec("t", "2025", **metadata)
    return TargetSpec(
        name=spec.name,
        entity=spec.entity,
        measure=spec.measure,
        value=spec.value,
        period=spec.period,
        family=spec.family,
        source=spec.source,
        metadata={
            key: value
            for key, value in spec.metadata.items()
            if key != "ledger_fact_period"
        },
    )


@pytest.mark.parametrize("spec", [_periodless(), _spec("t", "not-a-period")])
def test_a_row_with_no_readable_period_is_refused(spec):
    registry = TargetRegistry([spec], country="uk")
    assert not is_uprating_hold(spec, 2025)
    with pytest.raises(ValueError, match="no readable fact period"):
        assert_uk_uprating_holds_declared(
            registry,
            calibration_period=2025,
            evaluated_on=date(2026, 10, 7),
            scope="national",
            holds=_holds(),
        )


def test_a_moved_row_needs_no_readable_period():
    registry = TargetRegistry([_periodless(uprating_factor="1.1")], country="uk")
    receipt = assert_uk_uprating_holds_declared(
        registry,
        calibration_period=2025,
        evaluated_on=date(2026, 10, 7),
        scope="national",
        holds=_holds(),
    )
    assert receipt["held_targets"] == 0


@pytest.mark.parametrize(
    "unsigned",
    [
        {"approved_by": None},
        {"approved_on": None},
        {"approved_by": " "},
        {"status": "proposed"},
    ],
)
def test_a_hold_nobody_signed_is_not_in_force(unsigned):
    registry = TargetRegistry([_spec("t", "2025-10")], country="uk")
    with pytest.raises(ValueError, match=r"unsigned \['t'\]"):
        assert_uk_uprating_holds_declared(
            registry,
            calibration_period=2025,
            evaluated_on=date(2026, 10, 7),
            scope="national",
            holds=_holds(_hold("t", **unsigned)),
        )


def test_every_committed_hold_is_signed():
    # Fails until María signs each entry in uk/uprating_holds.json: a proposed
    # hold is not in force, so the compile refuses its rows and this branch
    # cannot be merged around the open decision (microcosm#1198 review).
    unsigned = [
        target_id
        for entry in load_uk_uprating_holds()["holds"]
        if not uk_hold_is_signed(entry)
        for target_id in entry["target_ids"]
    ]
    assert unsigned == []


def test_the_committed_declarations_are_well_formed():
    payload = load_uk_uprating_holds()
    seen: set[str] = set()
    for entry in payload["holds"]:
        assert entry["scope"] in {"national", "local"}
        assert entry["reason"].strip()
        assert entry["tracking"].strip()
        date.fromisoformat(entry["expires_on"])
        for target_id in entry["target_ids"]:
            assert target_id not in seen
            seen.add(target_id)
