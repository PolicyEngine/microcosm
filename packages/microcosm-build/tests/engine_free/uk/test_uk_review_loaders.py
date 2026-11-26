"""The UK signed-entry loaders on the shared review core, against the old code.

microcosm#1759 step D6 moved the date and window checks of the calibration
measure exclusions (``uk_runtime.measure_simulation``) and of the support
floor (``uk_runtime.support_floor``) onto ``microcosm.build.declared_registers``
(``canonical_date``, ``ReviewWindow``, ``stale_entries``). Each UK function
keeps its texts and its policy. ``test_support/microcosm_build/
uk_review_loaders_reference.py`` holds the old functions verbatim. On the
packaged file, on changed copies and on dates around each window edge, the
two give the same value, the same warnings, or the same refusal.
"""

from __future__ import annotations

import copy
import json
import warnings
from datetime import date, timedelta
from importlib.resources import files

import pytest

from microcosm.build.uk_runtime import measure_simulation, support_floor
from microcosm.calibrate import TargetRegistry, TargetSpec
from test_support.microcosm_build import uk_review_loaders_reference as reference

_PAYLOAD = json.loads(
    files("microcosm.build.uk")
    .joinpath("calibration_measure_exclusions.json")
    .read_text(encoding="utf-8")
)


def _outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = call()
        except Exception as error:  # noqa: BLE001 - the refusal is the outcome
            value = ("refused", isinstance(error, ValueError), str(error))
    return value, [(w.category.__name__, str(w.message)) for w in caught]


def _entry_changes():
    first = 0
    return {
        "packaged": lambda payload: None,
        "compact approval": lambda p: p["exclusions"][first].update(
            approved_on=p["exclusions"][first]["approved_on"].replace("-", "")
        ),
        "week-date expiry": lambda p: p["exclusions"][first].update(
            expires_on="2027-W05-3"
        ),
        "impossible day": lambda p: p["exclusions"][first].update(
            approved_on="2026-02-30"
        ),
        "expiry before approval": lambda p: p["exclusions"][first].update(
            approved_on="2027-01-02", expires_on="2027-01-01"
        ),
        "expiry on approval": lambda p: p["exclusions"][first].update(
            approved_on="2027-01-01", expires_on="2027-01-01"
        ),
        "padded date": lambda p: p["exclusions"][first].update(
            expires_on=" 2027-01-01"
        ),
        "empty tracking": lambda p: p["exclusions"][first].update(tracking=" "),
        "missing reason": lambda p: p["exclusions"][first].pop("reason"),
        "unknown field": lambda p: p["exclusions"][first].update(extra="x"),
        "duplicate name": lambda p: p["exclusions"].append(
            copy.deepcopy(p["exclusions"][first])
        ),
        "entry not an object": lambda p: p["exclusions"].append("x"),
        "bad schema": lambda p: p.update(schema_version=3),
        "unknown top key": lambda p: p.update(extra=1),
    }


@pytest.mark.parametrize("change", sorted(_entry_changes()))
def test_the_exclusion_loader_matches_the_replaced_code(change, tmp_path) -> None:
    payload = copy.deepcopy(_PAYLOAD)
    _entry_changes()[change](payload)
    path = tmp_path / "exclusions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    old = _outcome(lambda: reference.load_uk_calibration_measure_exclusions(path))
    new = _outcome(
        lambda: measure_simulation.load_uk_calibration_measure_exclusions(path)
    )
    assert new == old


def _registry(names) -> TargetRegistry:
    return TargetRegistry(
        [
            TargetSpec(
                name=name,
                entity="household",
                measure="household_weight",
                value=1.0,
                period=2025,
                source="test",
                family="test_family",
            )
            for name in names
        ],
        country="uk",
    )


def _edge_dates(entry) -> list[date]:
    approved = date.fromisoformat(entry["approved_on"])
    expires = date.fromisoformat(entry["expires_on"])
    return [
        approved - timedelta(days=1),
        approved,
        expires - timedelta(days=8),
        expires - timedelta(days=7),
        expires,
        expires + timedelta(days=1),
    ]


_EXCLUSIONS = reference.load_uk_calibration_measure_exclusions()


@pytest.mark.parametrize("index", range(0, len(_EXCLUSIONS), 9))
def test_the_exclusion_window_matches_the_replaced_code(index) -> None:
    entry = _EXCLUSIONS[index]
    registry = _registry([entry["name"], "other_target@2025"])
    for on in _edge_dates(entry):
        old = _outcome(
            lambda on=on: _applied(
                reference.apply_uk_calibration_measure_exclusions(
                    registry, (entry,), now=on
                )
            )
        )
        new = _outcome(
            lambda on=on: _applied(
                measure_simulation.apply_uk_calibration_measure_exclusions(
                    registry, (entry,), now=on
                )
            )
        )
        assert new == old


def _applied(result):
    registry, receipt = result
    return registry.version, [spec.name for spec in registry.specs], receipt


def test_every_exclusion_on_one_date_and_a_stale_entry() -> None:
    on = date(2026, 11, 26)
    names = [entry["name"] for entry in _EXCLUSIONS]
    for registry in (_registry(names), _registry(names[1:]), _registry([])):
        old = _outcome(
            lambda registry=registry: _applied(
                reference.apply_uk_calibration_measure_exclusions(
                    registry, _EXCLUSIONS, now=on
                )
            )
        )
        new = _outcome(
            lambda registry=registry: _applied(
                measure_simulation.apply_uk_calibration_measure_exclusions(
                    registry, _EXCLUSIONS, now=on
                )
            )
        )
        assert new == old


def _floor_changes():
    return {
        "packaged": lambda block: None,
        "compact approval": lambda b: b.update(
            approved_on=b["approved_on"].replace("-", "")
        ),
        "impossible expiry": lambda b: b.update(expires_on="2027-02-30"),
        "expiry on approval": lambda b: b.update(expires_on=b["approved_on"]),
        "date as a number": lambda b: b.update(approved_on=20261126),
        "padded date": lambda b: b.update(expires_on=b["expires_on"] + " "),
    }


@pytest.mark.parametrize("change", sorted(_floor_changes()))
def test_the_support_floor_reader_matches_the_replaced_code(change) -> None:
    block = copy.deepcopy(_PAYLOAD["support_floor"])
    _floor_changes()[change](block)
    old = _outcome(lambda: reference.uk_support_floor_from_payload(block))
    new = _outcome(lambda: support_floor.uk_support_floor_from_payload(block))
    assert new == old


def test_the_support_floor_window_matches_the_replaced_code() -> None:
    floor = support_floor.uk_support_floor_from_payload(_PAYLOAD["support_floor"])
    edges = _edge_dates(
        {
            "approved_on": floor.approved_on.isoformat(),
            "expires_on": floor.expires_on.isoformat(),
        }
    )
    for on in edges:
        old = _outcome(lambda on=on: reference.assert_in_force(floor, on))
        new = _outcome(lambda on=on: floor.assert_in_force(on))
        assert new == old
