"""US gate scopes and bindings over today's gates (microcosm#1585 item 11).

PR 2 of 4. ``us/gates.json`` declares every US gate the four build tools
evaluate in shadow; these tests hold the spec to today's code. Each scope is
a closed-world share of the spec with pinned digests; every declared
threshold equals the constant or keyword default today's gate uses; every
declared entry is wired at a site in its tool, calling the gate its binding
calls; and a fault injected into any gate fails it identically in today's
code and in the battery, in the entry's own phase.
"""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import replace

import pytest

from microcosm.build import FitWeightRecord
from microcosm.build.country_spec import load_gates_manifest
from microcosm.build.gate_battery import DEFAULT_REGISTRY, EvidenceContext, GateStatus
from microcosm.build.gates import UNWEIGHTED_KIND, GateResult
from microcosm.build.us_runtime.battery_bindings import (
    FRAME,
    USCheckBinding,
    declared_value,
    resolve_reference,
    us_gate_registry,
    us_gate_specs,
)
from microcosm.build.us_runtime.gate_scopes import (
    US_GATE_SCOPES,
    check_us_gate_scopes,
    us_gate_scope_digests,
    us_scope_entry_ids,
    us_scoped_gate_manifest,
)
from microcosm.build.us_runtime.gate_shadow import (
    GateBatteryShadow,
    LegacyVerdict,
    active_gate_shadow,
    shadow_file_paths,
    shadow_gate,
    shadow_value,
)
from test_support.microcosm_build.us_gate_shadow import (
    load_us_gate_tool,
    shadow_wiring,
)

#: The scopes PR 2 has declared so far, each wired in its tool.
WIRED_SCOPES = ("us_base", "us_pool", "us_national", "us_local")

#: Each scope's pinned digests. They move only when the scope's entries,
#: parameters, phases or the manifest's policy line do. The policy line's
#: shadow sentence moved every scope's manifest digest and fingerprint
#: (review of #1635, round 3).
SCOPE_PINS: dict[str, dict[str, str]] = {
    "us_base": {
        "gates_manifest_sha256": (
            "edaac2f8318d1abb694d6e88f31e236fb13b5cce74ca8067947541d3a3509463"
        ),
        "policy_sha256": (
            "1a86c63cc20a6259c3349cc9dd831f0571d8c7e82e1ba77f06e4c4a102274e22"
        ),
        "spec_fingerprint": (
            "66cf423d4cf2e9685d2cb58aca3e4a3b8dfad4327a25d9889c0426230957b1b4"
        ),
    },
    "us_pool": {
        "gates_manifest_sha256": (
            "f5e76c30e9f958690777777f2f40dd48bfc4cb804fac62f1c81518006fc3006e"
        ),
        "policy_sha256": (
            "98ff80d5bbe6f7a11f3ee3f63cae5f953fc6607559de22e2f552fe3e1f06d872"
        ),
        "spec_fingerprint": (
            "395968f4f14021d59ca902ecb2a3561da78c617231d12dd4966ae83ef1226215"
        ),
    },
    "us_national": {
        # Moved by the SSI take-up and release-gate entries that pin the
        # thresholds they compare against (review of #1635, round 2), and by
        # the QRF tail gate's notes, which now say it grades base weights.
        "gates_manifest_sha256": (
            "717f1daa916ec31879f1cfddba32738a98e5055ee6ad5b8451cad277ddbd055c"
        ),
        "policy_sha256": (
            "8be8670ce4f58b10c34d58b994645885b0d8758a4996c4f77797ddfbb698d8bb"
        ),
        "spec_fingerprint": (
            "417365f8ee874eebb3ce242fe3c5b95bca0f049b3ac5f2f9aa13e1e8330deb35"
        ),
    },
    "us_local": {
        # Moved by the entries that pin the thresholds they compare against
        # (review of #1635, rounds 1 and 2).
        "gates_manifest_sha256": (
            "e436bc1fc2ce141270e49684e43e2546881ba93969f42e0ae7092857d509ed0f"
        ),
        "policy_sha256": (
            "8c0a58ddee2fd707ba2e45d285c197b689c11723281e7f925965ff33d65ace21"
        ),
        "spec_fingerprint": (
            "6b329c7dd72713c127dcd917e62aaa875e6b87f2b072c76db3724e8dace0bc08"
        ),
    },
}

#: The arguments each tool's parser needs to report its defaults.
MINIMAL_ARGV = {
    "us_national": ["--ledger-facts", "facts.jsonl", "--out", "out"],
    "us_local": [
        "--staging-h5",
        "staging.h5",
        "--checkpoint-dir",
        "checkpoints",
        "--stage",
        "qa",
        "--out-h5",
        "out.h5",
    ],
}


def _entries(scope: str):
    return us_scoped_gate_manifest(scope).gates


def _registry(scope: str):
    return us_gate_registry(scope, tool=vars(load_us_gate_tool(scope)))


def _check_binding(scope: str) -> USCheckBinding:
    binding = _registry(scope)["stage_health"]
    assert isinstance(binding, USCheckBinding)
    return binding


def _check_entries(scope: str):
    return [entry for entry in _entries(scope) if entry.gate == "stage_health"]


def _same(left: object, right: object) -> bool:
    return json.dumps(declared_value(left), sort_keys=True) == json.dumps(
        declared_value(right), sort_keys=True
    )


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------


def test_the_scopes_partition_the_spec() -> None:
    gates = load_gates_manifest("us")
    check_us_gate_scopes(gates)
    owned = [
        entry_id for scope in US_GATE_SCOPES for entry_id in us_scope_entry_ids(scope)
    ]
    assert sorted(owned) == sorted(entry.id for entry in gates.gates)
    assert len(owned) == len(set(owned))


def test_a_manifest_the_scopes_do_not_partition_is_refused() -> None:
    gates = load_gates_manifest("us")
    first = gates.gates[0]

    def with_entries(*entries):
        return replace(gates, gates=(*entries, *gates.gates[1:]))

    with pytest.raises(
        RuntimeError, match="owned by no scope \\['us_elsewhere_gate'\\]"
    ):
        check_us_gate_scopes(with_entries(replace(first, id="us_elsewhere_gate")))
    with pytest.raises(RuntimeError, match="does not evaluate"):
        check_us_gate_scopes(with_entries(replace(first, phase="simulated")))
    with pytest.raises(ValueError, match="unknown US gate scope"):
        us_scoped_gate_manifest("us_elsewhere")


@pytest.mark.parametrize("scope", WIRED_SCOPES)
def test_each_scope_has_its_own_pinned_digests(scope) -> None:
    digests = us_gate_scope_digests(scope, _registry(scope))
    assert digests == SCOPE_PINS[scope]
    others = {
        other: us_gate_scope_digests(other, _registry(other))
        for other in WIRED_SCOPES
        if other != scope
    }
    assert all(
        pins["gates_manifest_sha256"] != digests["gates_manifest_sha256"]
        for pins in others.values()
    )


# ---------------------------------------------------------------------------
# Thresholds: the spec carries today's values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", WIRED_SCOPES)
def test_every_pinned_threshold_equals_todays_constant(scope) -> None:
    binding = _check_binding(scope)
    for entry in _check_entries(scope):
        check = entry.parameters["check"]
        today = binding.pinned_values(check)
        declared = {key: entry.parameters[key] for key in today}
        assert _same(declared, today), (entry.id, declared, today)


@pytest.mark.parametrize("scope", WIRED_SCOPES)
def test_every_routed_threshold_equals_todays_default(scope) -> None:
    binding = _check_binding(scope)
    for entry in _check_entries(scope):
        spec = binding.checks[entry.parameters["check"]]
        if not spec.parameters:
            continue
        gate = resolve_reference(spec.gate, tool=binding.tool)
        defaults = inspect.signature(gate).parameters
        parsed = (
            binding.tool["_parse_args"](MINIMAL_ARGV[scope]) if spec.flags else None
        )
        for key in spec.parameters:
            today = (
                getattr(parsed, spec.flags[key])
                if key in spec.flags
                else defaults[key].default
            )
            assert _same(entry.parameters[key], today), (entry.id, key)


@pytest.mark.parametrize("scope", WIRED_SCOPES)
def test_entries_declare_exactly_their_gates_thresholds(scope) -> None:
    binding = _check_binding(scope)
    used = set()
    for entry in _check_entries(scope):
        check = entry.parameters["check"]
        assert check in binding.checks, entry.id
        used.add(check)
        declared = set(entry.parameters) - {"check"}
        assert declared == binding.checks[check].parameter_keys, entry.id
    # No spec is dead weight in its scope.
    assert used == set(binding.checks)


# ---------------------------------------------------------------------------
# Wiring: every entry has a site, calling the gate its binding calls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", WIRED_SCOPES)
def test_every_entry_is_wired_at_a_site_calling_its_gate(scope) -> None:
    binding = _check_binding(scope)
    declared = {entry.id: entry for entry in _entries(scope)}
    sites = shadow_wiring(scope)
    unknown = sorted({entry_id for entry_id, _, _ in sites} - set(declared))
    assert not unknown, unknown
    assert {entry_id for entry_id, _, _ in sites} == set(declared)
    tool = binding.tool
    for entry_id, reference, where in sites:
        entry = declared[entry_id]
        if reference is None or entry.gate != "stage_health":
            continue
        spec = binding.checks[entry.parameters["check"]]
        name = reference.removeprefix("tool:")
        if reference.startswith("tool:") and name not in tool:
            # The site imports the gate inside its function, so the binding
            # names the gate's own module instead.
            assert spec.gate.endswith(f":{name}"), (entry_id, where, spec.gate)
            continue
        assert spec.gate == reference, (entry_id, where, reference, spec.gate)


# ---------------------------------------------------------------------------
# Fault injection: one failure per gate, identical in both, in its phase
# ---------------------------------------------------------------------------


def _shadow(tmp_path, scope: str) -> GateBatteryShadow:
    report, comparison = shadow_file_paths(tmp_path, scope)
    return GateBatteryShadow(
        us_scoped_gate_manifest(scope),
        scope=scope,
        release_id=f"{scope}-fault-injection",
        report_path=report,
        comparison_path=comparison,
        registry=_registry(scope),
    )


def _evidence(spec) -> EvidenceContext:
    """Evidence of the right shape; the injected gate ignores its values."""

    return EvidenceContext(
        frame=object() if spec.needs_frame else None,
        artifacts={key: object() for key in spec.artifact_keys},
    )


def _fault_cases(scope: str):
    return [
        pytest.param(scope, entry.id, id=entry.id) for entry in _check_entries(scope)
    ]


@pytest.mark.parametrize(
    ("scope", "entry_id"),
    [case for scope in WIRED_SCOPES for case in _fault_cases(scope)],
)
@pytest.mark.parametrize("fault", ["fails", "crashes"])
def test_an_injected_fault_fails_both_verdicts_alike(
    tmp_path, monkeypatch, scope, entry_id, fault
) -> None:
    tool = vars(load_us_gate_tool(scope))
    shadow = _shadow(tmp_path, scope)
    entry = next(entry for entry in shadow.gates.gates if entry.id == entry_id)
    spec = {spec.check: spec for spec in us_gate_specs(scope)}[
        entry.parameters["check"]
    ]
    namespace, _, gate_name = spec.gate.partition(":")
    target = tool if namespace == "tool" else vars(importlib.import_module(namespace))
    lines = (f"{entry_id}: injected failure.",)

    failing = GateResult(name=spec.legacy_name, passed=False, failures=lines)

    def injected(*args, **kwargs):
        if fault == "crashes":
            raise KeyError(f"{entry_id}: injected crash")
        if spec.refusals:
            raise spec.refusals[0](lines[0])
        return {
            "result": failing,
            "failures": list(lines),
            "failures_and_details": (list(lines), {}),
            "result_and_details": (failing, {}),
            "failure_or_none": lines[0],
            "mapping": {"passed": False, "failures": list(lines)},
        }[spec.returns]

    monkeypatch.setitem(target, gate_name, injected)
    evidence = _evidence(spec)

    def today():
        return target[gate_name](
            *(
                evidence.frame if source == FRAME else evidence.artifacts[source]
                for source in spec.positional
            ),
        )

    # Today's verdict as each kind of site reads it.
    verdicts = {
        "result": LegacyVerdict.from_result,
        "receipt": lambda value: LegacyVerdict.from_failures(()),
        "failures": LegacyVerdict.from_failures,
        "failures_and_details": lambda value: LegacyVerdict.from_failures(value[0]),
        "result_and_details": lambda value: LegacyVerdict.from_result(value[0]),
        "failure_or_none": lambda value: LegacyVerdict.from_failures(
            [] if value is None else [value]
        ),
        "mapping": lambda value: LegacyVerdict.from_failures(
            () if value["passed"] else value["failures"]
        ),
    }
    expected = (KeyError,) if fault == "crashes" else spec.refusals
    with active_gate_shadow(shadow):
        if expected:
            with pytest.raises(expected):
                shadow_value(
                    entry_id,
                    today,
                    evidence,
                    verdict=verdicts[spec.returns],
                    refusals=spec.refusals,
                )
        else:
            shadow_value(entry_id, today, evidence, verdict=verdicts[spec.returns])
    outcome = shadow.outcome(entry_id)
    assert outcome is not None and outcome.status is GateStatus.FAILED
    assert outcome.entry.phase == entry.phase
    if fault == "fails":
        assert outcome.result.failures == lines
    else:
        assert "evaluation_error" in outcome.result.details
    assert shadow.differences() == ()


@pytest.mark.parametrize("scope", ["us_base", "us_pool"])
def test_the_shared_weights_audit_fails_both_verdicts_alike(tmp_path, scope) -> None:
    shadow = _shadow(tmp_path, scope)
    tool = vars(load_us_gate_tool(scope))
    records = [FitWeightRecord("us_puf_tax_detail_support", UNWEIGHTED_KIND)]
    with active_gate_shadow(shadow):
        result = shadow_gate(
            f"{scope}_weights_audit",
            lambda: tool["weights_audit_gate"](records),
            EvidenceContext(artifacts={"fit_weight_records": records}),
        )
    assert not result.passed
    outcome = shadow.outcome(f"{scope}_weights_audit")
    assert outcome.status is GateStatus.FAILED
    assert outcome.result.failures == result.failures
    assert shadow.differences() == ()


@pytest.mark.parametrize("scope", ["us_base", "us_pool"])
def test_the_audit_site_calls_the_shared_bindings_own_gate(scope) -> None:
    # The one binding not looked up in the tool: PR 1's shared weights_audit
    # holds the gate function itself, so the tool must call that same object.
    tool = load_us_gate_tool(scope)
    assert tool.weights_audit_gate is DEFAULT_REGISTRY["weights_audit"].gate


def test_a_declared_threshold_off_todays_constant_is_a_difference(tmp_path) -> None:
    # The pin is live: a spec edit that today's constant does not follow is
    # caught at the next evaluation, as a verdict difference.
    scoped = us_scoped_gate_manifest("us_base")
    altered = replace(
        scoped,
        gates=tuple(
            replace(
                entry,
                parameters={**entry.parameters, "pregnant_share_band": [0.0, 1.0]},
            )
            if entry.id == "us_base_pregnancy"
            else entry
            for entry in scoped.gates
        ),
    )
    report, comparison = shadow_file_paths(tmp_path, "us_base")
    shadow = GateBatteryShadow(
        altered,
        scope="us_base",
        release_id="us-base-altered",
        report_path=report,
        comparison_path=comparison,
        registry=_registry("us_base"),
    )
    passing = GateResult(name="pregnancy_signal", passed=True)
    with active_gate_shadow(shadow):
        shadow_gate(
            "us_base_pregnancy",
            lambda: passing,
            EvidenceContext(frame=object()),
        )
    (difference,) = shadow.differences()
    assert "differs from today's" in difference


def test_a_blocking_district_collapse_fails_both_verdicts_on_one_line(
    tmp_path,
) -> None:
    # The district ESS gate fails a blocking collapse on its counts, with no
    # failure line, and a result that fails must name why. Both verdicts read
    # the line district_ess_collapse_failures makes from the counts; before,
    # the battery recorded an evaluation error and today's verdict a pass.
    tool = vars(load_us_gate_tool("us_local"))
    shadow = _shadow(tmp_path, "us_local")
    origin = {
        "design": {"effective_sample_size_by_district": {"0601": 100.0, "0602": 80.0}},
        "calibrated": {
            "effective_sample_size_by_district": {"0601": 20.0, "0602": 60.0}
        },
    }
    with active_gate_shadow(shadow):
        entry = shadow_value(
            "us_local_district_ess_collapse",
            lambda: tool["district_ess_collapse_gate"](
                origin, relative_floor=0.25, absolute_floor=15.0, blocking=True
            ),
            EvidenceContext(
                artifacts={"weight_origin": origin, "district_ess_blocking": True}
            ),
            verdict=tool["_legacy_district_ess_verdict"],
        )
    # Today's entry is written as before: failing, with its counts only.
    assert (entry["passed"], entry["failures"], entry["n_collapsed"]) == (
        False,
        [],
        1,
    )
    line = (
        "blocking district ESS gate: congressional districts with a calibrated "
        "Kish ESS 1 of 2 below 0.25 x their design-weight Kish ESS; 0 of 2 "
        "below 15"
    )
    outcome = shadow.outcome("us_local_district_ess_collapse")
    assert outcome is not None and outcome.status is GateStatus.FAILED
    assert outcome.result.failures == (line,)
    assert tool["_legacy_district_ess_verdict"](entry).failures == (line,)
    assert shadow.differences() == ()


@pytest.mark.parametrize(
    "origin",
    [
        pytest.param({}, id="no-per-district-ess"),
        pytest.param(
            {
                "design": {"effective_sample_size_by_district": {"0601": 100.0}},
                "calibrated": {"effective_sample_size_by_district": {"0602": 60.0}},
            },
            id="different-districts",
        ),
    ],
)
def test_a_report_only_district_entry_it_cannot_evaluate_passes_both_verdicts(
    tmp_path, origin
) -> None:
    # Report-only, the district ESS gate passes an entry it cannot evaluate
    # and keeps the reason as a failure line. A result that passes must name
    # nothing, so the binding reads district_ess_collapse_failures, which
    # gives no line for a passing entry, as today's verdict does; before,
    # the binding kept the entry's line and GateResult refused it, which the
    # shadow recorded as a difference (review of #1635).
    tool = vars(load_us_gate_tool("us_local"))
    shadow = _shadow(tmp_path, "us_local")
    with active_gate_shadow(shadow):
        entry = shadow_value(
            "us_local_district_ess_collapse",
            lambda: tool["district_ess_collapse_gate"](
                origin, relative_floor=0.25, absolute_floor=15.0, blocking=False
            ),
            EvidenceContext(
                artifacts={"weight_origin": origin, "district_ess_blocking": False}
            ),
            verdict=tool["_legacy_district_ess_verdict"],
        )
    # Today's entry is written as before: passing, with its reason.
    assert entry["passed"] is True and entry["criteria_met"] is None
    assert len(entry["failures"]) == 1
    outcome = shadow.outcome("us_local_district_ess_collapse")
    assert outcome is not None and outcome.status is GateStatus.PASSED
    assert outcome.result.failures == ()
    assert tool["_legacy_district_ess_verdict"](entry).failures == ()
    # Differences include the shadow's own errors, such as a refused result.
    assert shadow.differences() == ()


def test_only_a_mapping_gate_takes_a_failure_line_maker() -> None:
    (spec,) = [
        spec
        for spec in us_gate_specs("us_local")
        if spec.check == "district_ess_collapse"
    ]
    assert spec.failure_lines in spec.references
    with pytest.raises(ValueError, match="failure_lines reads a mapping"):
        replace(spec, returns="failures")
