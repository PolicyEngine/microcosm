"""The local compile-parity receipt: its gate and its signed texts (microcosm#1300)."""

from __future__ import annotations

import copy
import json
from importlib import resources as importlib_resources

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import EvidenceContext
from microcosm.build.uk_runtime.battery_bindings import (
    _LEDGER_COMPILE_PARITY_UNSUPPORTED_ARTIFACTS,
    UK_GATE_REGISTRY,
)
from microcosm.build.uk_runtime.ledger_targets import (
    load_uk_local_target_reference_membership,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from tools import build_uk_ledger_compile_parity_signed_differences as parity_tool

_LOCAL_PARITY_GATE_ID = "uk_ledger_compile_parity_local_incumbent_2025"


def _local_parity_gate():
    (entry,) = [
        entry
        for entry in load_country_spec("uk").gates.gates
        if entry.id == _LOCAL_PARITY_GATE_ID
    ]
    return entry


@pytest.fixture(scope="module")
def committed_local_compile() -> TargetRegistry:
    """The local compile on the pinned feed, as the committed surfaces record it.

    The local reference generator writes a reference for every cell that
    compiles on the pin and records the cell's compiled value, after its
    declared uprating, in the membership; a cell that does not compile is
    signed there instead. So the compile is the references' cells at the
    membership's values, with no reference unsupported, and the gate replays
    without the feed.
    """

    membership = load_uk_local_target_reference_membership()
    active = {
        candidate["name"]: candidate
        for target in membership["targets"].values()
        for level in target["geography_levels"].values()
        for candidate in level["candidates"]
        if candidate["status"] == "active"
    }
    names = [
        reference.name for reference in load_country_spec("uk").local_target_references
    ]
    assert sorted(names) == sorted(active), (
        "The local references and the membership's active cells differ: "
        "regenerate them together (docs/uk-chronicle-feed-repin.md)."
    )
    return TargetRegistry(
        [
            TargetSpec(
                name=name,
                entity="target",
                measure=name,
                value=float(active[name]["resolved_value"]),
                period=active[name]["resolved_period"],
                source="committed_local_membership",
                family="ledger_compile_parity",
            )
            for name in names
        ],
        country="uk",
    )


def _replay(registry: TargetRegistry, **parameters):
    entry = _local_parity_gate()
    declared = {**entry.parameters, **parameters}
    period = declared["target_period"]
    artifact = declared["registry_artifact"]
    context = EvidenceContext(
        artifacts={
            artifact: {period: registry},
            _LEDGER_COMPILE_PARITY_UNSUPPORTED_ARTIFACTS[artifact]: {period: ()},
        }
    )
    return UK_GATE_REGISTRY[entry.gate].evaluate(context, declared)


@pytest.fixture(scope="module")
def committed_replay(committed_local_compile):
    return _replay(committed_local_compile)


def test_the_local_compile_parity_gate_passes_on_the_committed_membership(
    committed_replay,
):
    """The release-blocking preflight gate, replayed on the committed surfaces.

    Every full or dense build and the release-cut certifier run this gate on
    the live compile, so a receipt the last regeneration left behind blocks
    them all (microcosm#1302 review). The replay catches it in CI.
    """

    failures = committed_replay.failures
    assert not failures, (len(failures), failures[:10])


def test_the_replay_refuses_a_signed_value_that_moved(
    committed_local_compile, committed_replay
):
    signed = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(_local_parity_gate().parameters["signed_differences_resource"])
        .read_text(encoding="utf-8")
    )["differences"]
    clean = committed_replay
    failing = " ".join(clean.failures)
    position, row = next(
        (position, row)
        for position, row in enumerate(signed)
        if row["kind"] == "calibration_drift" and f"{row['name']}[" not in failing
    )
    moved = [*signed]
    moved[position] = {**row, "ledger_value": row["ledger_value"] + 1.0}
    corrupted = _replay(
        committed_local_compile,
        signed_differences_resource=None,
        signed_differences=moved,
    )
    # The moved value is the one new failure, named with the value it signed.
    (failure,) = set(corrupted.failures) - set(clean.failures)
    assert failure.startswith(
        f"signed ledger parity difference {row['name']}[{row['period']}] changed "
        f"ledger_value: signed={row['ledger_value'] + 1.0!r}, live="
    )


_UC_TOTALS = "dwp.uc.households_by_area"
_UC_CHILD_BANDS = tuple(
    f"dwp.uc.households_by_area_children_{band}" for band in ("0", "1", "2", "3plus")
)


def _row(name: str, kind: str, fixture: float | None, ours: float | None) -> dict:
    row = {"kind": kind, "name": name, "period": 2025, "reason": "generic"}
    if fixture is not None:
        row["fixture_value"] = fixture
    if ours is not None:
        row["ledger_value"] = ours
    return row


def _report() -> dict:
    # Synthetic values. Twenty-one area totals whose fixture runs from 0.900
    # to 1.000 of ours in steps of 0.005, so the 5th and 95th percentiles are
    # 0.905 and 0.995; twenty-one child-band cells, spread over the four
    # bands, from 0.80 to 1.20 in steps of 0.02 (0.820 and 1.180).
    totals = [
        _row(f"{_UC_TOTALS}@E1400{1000 + i}", "calibration_drift", 1000.0 - 5 * i, 1e3)
        for i in range(21)
    ]
    children = [
        _row(
            f"{_UC_CHILD_BANDS[i % 4]}@E1400{1000 + i}",
            "calibration_drift",
            200.0 * (0.80 + 0.02 * i),
            200.0,
        )
        for i in range(21)
    ]
    return {
        "differences": [
            *totals,
            *children,
            _row(f"{_UC_TOTALS}@N05000001", "fixture_only", 10.0, None),
            _row("ons.tenure.private_rent@E14001000", "calibration_drift", 9.0, 10.0),
        ]
    }


def test_the_uc_drift_rationales_quote_the_spread_of_the_rows_they_cover():
    report = _report()
    parity_tool._add_local_signed_rationale_notes(report)
    reasons = {row["name"]: row["reason"] for row in report["differences"]}
    totals = {
        reason for name, reason in reasons.items() if name.startswith(f"{_UC_TOTALS}@E")
    }
    children = {
        reason for name, reason in reasons.items() if name.startswith(_UC_CHILD_BANDS)
    }
    # One text per rationale, each quoting its own rows.
    (totals_reason,) = totals
    (children_reason,) = children
    assert "(P5-P95 0.905-0.995 across the 21 rows)" in totals_reason
    assert "(P5-P95 fixture/ours 0.820-1.180 across the 21 rows)" in children_reason
    # Both describe the paid-claims series the area cells bind, not the
    # uniform rescale of all claims they replaced (microcosm#1302 re-review).
    for reason in (totals_reason, children_reason):
        assert "Payment Indicator 'Yes'" in reason
        assert "0.8925" not in reason and "982" not in reason
    assert "each area's own paid claims" in totals_reason
    # The other rows keep their own rationales.
    assert (
        reasons[f"{_UC_TOTALS}@N05000001"] == parity_tool._UC_NI_FIXTURE_ONLY_RATIONALE
    )
    assert reasons["ons.tenure.private_rent@E14001000"] == (
        parity_tool._TENURE_DRIFT_RATIONALE
    )


def test_the_receipt_tool_streams_every_row_of_the_feed(tmp_path):
    feed = tmp_path / "consumer_facts.jsonl"
    feed.write_bytes(b'{"a": 1}\n\n{"b": "x"}\r\n   \n{"c": [1, 2]}')
    assert parity_tool._load_jsonl(feed) == [{"a": 1}, {"b": "x"}, {"c": [1, 2]}]


def test_a_receipt_without_uc_drift_rows_formats_no_uc_rationale():
    report = {
        "differences": [_row(f"{_UC_TOTALS}@N05000001", "fixture_only", 10.0, None)]
    }
    metric_by_target_id = parity_tool._local_metric_by_target_id()
    assert parity_tool._uc_drift_rationales(report, metric_by_target_id) == {}
    parity_tool._add_local_signed_rationale_notes(report)
    (row,) = report["differences"]
    assert row["reason"] == parity_tool._UC_NI_FIXTURE_ONLY_RATIONALE


@pytest.mark.parametrize(
    "spec", parity_tool.RECEIPTS, ids=lambda spec: f"{spec.surface}_receipt"
)
def test_each_committed_parity_receipt_gives_the_tool_s_reasons(spec):
    """A reason changed in the tool fails here until its receipt is
    regenerated: the national replay needs the feed, and nothing else in CI
    compared the local receipt's reasons with the tool (the fourth review of
    microcosm#1422)."""
    committed = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(spec.output_resource)
        .read_text(encoding="utf-8")
    )
    replayed = copy.deepcopy(committed)
    parity_tool._add_signed_rationale_notes(
        replayed, fixture_resource=spec.fixture_resource
    )
    if spec.surface == "local":
        parity_tool._add_local_signed_rationale_notes(replayed)
    stale = [
        row["name"]
        for row, expected in zip(
            committed["differences"], replayed["differences"], strict=True
        )
        if row.get("reason") != expected.get("reason")
    ]
    assert stale == []
