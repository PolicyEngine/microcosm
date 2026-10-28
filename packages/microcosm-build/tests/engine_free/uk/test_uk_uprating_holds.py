"""Every UK target held at an earlier period carries a declared reason (#1123)."""

from __future__ import annotations

import json
import warnings
from datetime import date, timedelta
from importlib import resources as importlib_resources

import pytest

from microcosm.build.ledger_targets import ROLLED_FORWARD_BY_RATIO
from microcosm.build.uk_runtime.uprating_holds import (
    _WINDOW_VALUE_OPERATIONS,
    UK_HOLD_EXPIRY_NOTICE_DAYS,
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


def test_a_hold_signed_after_the_evaluation_date_is_not_yet_in_force():
    registry = TargetRegistry([_spec("t", "2024")], country="uk")
    with pytest.raises(ValueError, match=r"signed after the evaluation date.*\['t'\]"):
        assert_uk_uprating_holds_declared(
            registry,
            calibration_period=2025,
            evaluated_on=date(2026, 9, 30),
            scope="national",
            holds=_holds(_hold("t", approved_on="2026-10-01")),
        )
    # In force on the day it is signed.
    assert_uk_uprating_holds_declared(
        registry,
        calibration_period=2025,
        evaluated_on=date(2026, 10, 1),
        scope="national",
        holds=_holds(_hold("t", approved_on="2026-10-01")),
    )


def test_the_compile_warns_when_a_hold_is_about_to_lapse():
    registry = TargetRegistry([_spec("t", "2024")], country="uk")
    expires = date(2026, 12, 11)

    def run(evaluated_on: date) -> dict:
        return assert_uk_uprating_holds_declared(
            registry,
            calibration_period=2025,
            evaluated_on=evaluated_on,
            scope="national",
            holds=_holds(_hold("t", expires_on=expires.isoformat())),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        receipt = run(expires - timedelta(days=UK_HOLD_EXPIRY_NOTICE_DAYS + 1))
    assert receipt["earliest_expiry"] == "2026-12-11"
    with pytest.warns(UserWarning, match="expire 2026-12-11, within 14 days"):
        run(expires - timedelta(days=UK_HOLD_EXPIRY_NOTICE_DAYS))
    with pytest.warns(UserWarning, match="expire 2026-12-11"):
        run(expires)


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
        # Only ``approved`` puts a hold in force, whoever signed the entry.
        {"status": "rejected"},
        {"status": "withdrawn"},
        {"status": None},
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


@pytest.mark.parametrize("missing", ["reason", "tracking", "expires_on"])
def test_a_register_passed_in_is_validated_like_the_committed_one(missing):
    registry = TargetRegistry([_spec("t", "2025-10")], country="uk")
    entry = _hold("t")
    del entry[missing]
    with pytest.raises(ValueError, match=f"has no {missing}"):
        assert_uk_uprating_holds_declared(
            registry,
            calibration_period=2025,
            evaluated_on=date(2026, 10, 7),
            scope="national",
            holds=_holds(entry),
        )


def test_every_committed_hold_is_signed():
    # An unsigned hold is not in force, so the compile refuses its rows: an
    # entry added without a signature fails here before it fails a build.
    unsigned = [
        target_id
        for entry in load_uk_uprating_holds()["holds"]
        if not uk_hold_is_signed(entry)
        for target_id in entry["target_ids"]
    ]
    assert unsigned == []


def test_no_committed_hold_is_about_to_lapse():
    # Feed-backed compiles are skipped in CI, so nothing there would show that
    # the holds are about to lapse: this fails UK_HOLD_EXPIRY_NOTICE_DAYS days
    # before the earliest expiry, while the builds still pass. It is a clock
    # on purpose. Clear it by binding a newer fact or an index, or by a new
    # review and signature in uk/uprating_holds.json.
    earliest = min(
        date.fromisoformat(entry["expires_on"])
        for entry in load_uk_uprating_holds()["holds"]
    )
    days_left = (earliest - date.today()).days
    assert days_left > UK_HOLD_EXPIRY_NOTICE_DAYS, (
        f"UK uprating holds expire on {earliest.isoformat()}, in {days_left} "
        "day(s); every UK build refuses them from the next day."
    )


def test_the_committed_declarations_are_well_formed():
    payload = load_uk_uprating_holds()
    seen: set[str] = set()
    for entry in payload["holds"]:
        assert entry["scope"] in {"national", "local"}
        assert entry["reason"].strip()
        assert entry["tracking"].strip()
        # A proposed entry has no approval date yet; it is not in force, and
        # test_every_committed_hold_is_signed names it.
        if uk_hold_is_signed(entry):
            assert date.fromisoformat(entry["approved_on"]) < date.fromisoformat(
                entry["expires_on"]
            )
        for target_id in entry["target_ids"]:
            assert target_id not in seen
            seen.add(target_id)


def test_every_reference_measured_at_an_earlier_year_is_declared():
    """A national reference whose executable binding measures an earlier year
    than the reference's period, with no index (a null one applies none) and
    no value operation that covers the period, binds a fact of that earlier
    year as published. The compile reads that fact's period and refuses the
    row without a declaration, so a missing one fails here, before a build.

    The compile decides from each spec's resolved fact period
    (``is_uprating_hold``); this reads only what the committed files declare,
    so holds that only a fact's period reveals are checked at compile time.
    """

    resources = importlib_resources.files("microcosm.build.uk")
    references = json.loads(resources.joinpath("target_references.json").read_text())[
        "target_references"
    ]
    contract = {
        target["target_id"]: target
        for target in json.loads(
            resources.joinpath("uk_population_targets.json").read_text()
        )["targets"]
    }
    declared = {
        target_id
        for entry in load_uk_uprating_holds()["holds"]
        if entry.get("scope", "national") == "national"
        for target_id in entry["target_ids"]
    }
    held = set()
    for reference in references:
        metadata = reference["metadata"]
        target = contract.get(metadata["contract_target_id"], {})
        binding = (target.get("bindings") or {}).get("policyengine") or {}
        measured = binding.get("measurement_period")
        if measured is None or int(measured) >= int(reference["period"]):
            continue
        if (
            reference.get("uprating_index")
            or metadata.get("uprating_index")
            or target.get("uprating_index")
        ):
            continue
        if reference.get("value_operation") in {
            *_WINDOW_VALUE_OPERATIONS,
            ROLLED_FORWARD_BY_RATIO,
        }:
            continue
        held.add(metadata["contract_target_id"])
    assert held, "the committed bindings measure no earlier year"
    assert sorted(held - declared) == []
