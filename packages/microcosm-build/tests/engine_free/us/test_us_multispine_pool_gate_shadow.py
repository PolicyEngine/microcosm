"""The pool builder's stacked path runs the gate battery in shadow (#1585 item 11).

``tools/build_us_multispine_pool.py`` arms the ``us_pool`` scope around the
stacked path and finishes it after every file the build writes, beside the
pool's own files. These tests drive that seam with stand-in gates; no pool is
built.
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from microcosm.build import FitWeightRecord, GateResult
from microcosm.build.gate_battery import EvidenceContext
from microcosm.build.us_runtime.battery_bindings import us_gate_specs
from microcosm.build.us_runtime.gate_shadow import (
    GateBatteryShadowMismatchError,
    shadow_file_paths,
    shadow_gate,
    shadow_not_run,
)
from test_support.microcosm_build.us_gate_shadow import load_us_gate_tool


@pytest.fixture
def pool():
    return load_us_gate_tool("us_pool")


def _namespace(pool, reference: str):
    namespace, _, name = reference.partition(":")
    target = (
        vars(pool) if namespace == "tool" else vars(importlib.import_module(namespace))
    )
    return target, name


def _stand_in_gates(monkeypatch, pool, *, failing=()):
    for spec in us_gate_specs("us_pool"):
        target, name = _namespace(pool, spec.gate)

        def gate(*args, _legacy=spec.legacy_name, _failed=spec.check in failing, **kw):
            if _failed:
                return GateResult(
                    name=_legacy, passed=False, failures=(f"{_legacy}: stand-in.",)
                )
            return GateResult(name=_legacy, passed=True)

        monkeypatch.setitem(target, name, gate)


def _cold_legacy_rule_run(pool, *, legacy_terminal=None):
    """What a cold stacked build on the legacy location rule observes."""

    tool = vars(pool)
    household, weights, frame = object(), object(), object()
    shadow_gate(
        "us_pool_puma_ladder",
        lambda: tool["us_puma_ladder_gate"](household, weights, assign_tract=False),
        EvidenceContext(
            artifacts={"household": household, "household_weights": weights}
        ),
    )
    shadow_not_run("us_pool_block_location", "the legacy location rule draws no block")
    runtime = vars(
        importlib.import_module("microcosm.build.us_runtime.multispine_pool")
    )
    shadow_gate(
        "us_pool_hours_worked",
        lambda: runtime["us_hours_worked_signal_gate"](frame),
        EvidenceContext(frame=frame),
    )
    records = [FitWeightRecord("us_puf_tax_detail_support", "design")]
    shadow_gate(
        "us_pool_weights_audit",
        lambda: tool["weights_audit_gate"](records),
        EvidenceContext(artifacts={"fit_weight_records": records}),
    )
    terminal = EvidenceContext(frame=frame, artifacts={"tail_manifest": {}})
    shadow_gate(
        "us_pool_stacked_completeness",
        lambda: tool["stacked_completeness_gate"](frame, tail_manifest={}),
        terminal,
    )
    shadow_gate(
        "us_pool_by_origin_battery",
        legacy_terminal or (lambda: tool["by_origin_battery"](frame, tail_manifest={})),
        terminal,
    )
    return 0


def test_the_stacked_path_finishes_its_shadow_beside_the_pool(
    tmp_path, monkeypatch, pool
) -> None:
    args = SimpleNamespace(out=tmp_path / "pools" / "stacked.h5")
    _stand_in_gates(monkeypatch, pool)
    with pool._armed_pool_gate_shadow(args):
        assert _cold_legacy_rule_run(pool) == 0
    directory = tmp_path / "pools" / "stacked.gate_battery_shadow"
    assert pool._pool_gate_shadow_directory(args) == directory.resolve()
    report, comparison = shadow_file_paths(directory, "us_pool")
    payload = json.loads(comparison.read_text())
    assert payload["run_completed"] is True
    assert payload["verdict_difference"] is False
    assert payload["phases_recorded"] == ["assembled", "transferred", "terminal"]
    assert json.loads(report.read_text())["shippable"] is False


def test_a_terminal_difference_fails_the_build_after_its_files(
    tmp_path, monkeypatch, pool
) -> None:
    args = SimpleNamespace(out=tmp_path / "stacked.h5")
    _stand_in_gates(monkeypatch, pool, failing={"by_origin_battery"})
    passing = GateResult(name="us_by_origin_battery", passed=True)
    with pytest.raises(GateBatteryShadowMismatchError, match="us_pool_by_origin"):
        with pool._armed_pool_gate_shadow(args):
            _cold_legacy_rule_run(pool, legacy_terminal=lambda: passing)


def test_a_marked_terminal_failure_is_a_match(tmp_path, monkeypatch, pool) -> None:
    # The terminal pair marks today: the build publishes a gate_failed pool and
    # returns 1. The battery fails it too, which is the same verdict.
    args = SimpleNamespace(out=tmp_path / "stacked.h5")
    _stand_in_gates(monkeypatch, pool, failing={"stacked_completeness"})
    with pool._armed_pool_gate_shadow(args):
        _cold_legacy_rule_run(pool)
    _, comparison = shadow_file_paths(pool._pool_gate_shadow_directory(args), "us_pool")
    payload = json.loads(comparison.read_text())
    assert payload["verdict_difference"] is False
    (completeness,) = [
        row for row in payload["entries"] if row["id"] == "us_pool_stacked_completeness"
    ]
    assert completeness["legacy"]["status"] == "failed"
    assert completeness["battery"]["status"] == "failed"
