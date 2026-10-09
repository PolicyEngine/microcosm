"""Every UK target held at an earlier period carries a declared reason (#1123)."""

from __future__ import annotations

from datetime import date

import pytest

from microcosm.build.uk_runtime.uprating_holds import (
    assert_uk_uprating_holds_declared,
    is_uprating_hold,
    load_uk_uprating_holds,
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
        **fields,
    }


_NO_EXCEPTIONS = {"schema_version": 1, "entries": []}


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
            exceptions=_NO_EXCEPTIONS,
        )


def test_a_declared_hold_is_receipted_by_kind():
    registry = TargetRegistry([_spec("t", "2025-10")], country="uk")
    receipt = assert_uk_uprating_holds_declared(
        registry,
        calibration_period=2025,
        evaluated_on=date(2026, 10, 7),
        scope="national",
        holds=_holds(_hold("t")),
        exceptions=_NO_EXCEPTIONS,
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
            exceptions=_NO_EXCEPTIONS,
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
            exceptions=_NO_EXCEPTIONS,
        )


def test_a_tolerated_hold_must_still_be_held():
    registry = TargetRegistry([_spec("t", "2025")], country="uk")
    exceptions = {
        "schema_version": 1,
        "entries": [
            {
                "kind": "undeclared_hold",
                "scope": "national",
                "target_id": "t",
                "closed_by": "test",
            }
        ],
    }
    with pytest.raises(ValueError, match=r"stale doctrine exceptions \['t'\]"):
        assert_uk_uprating_holds_declared(
            registry,
            calibration_period=2025,
            evaluated_on=date(2026, 10, 7),
            scope="national",
            holds=_holds(),
            exceptions=exceptions,
        )


def test_the_committed_declarations_are_well_formed():
    payload = load_uk_uprating_holds()
    seen: set[str] = set()
    for entry in payload["holds"]:
        assert entry["scope"] in {"national", "local"}
        assert entry["reason"].strip()
        for target_id in entry["target_ids"]:
            assert target_id not in seen
            seen.add(target_id)
