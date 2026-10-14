"""``gates.battery@1``: one gate-battery phase as a graph GATE node.

Invariants (Hypothesis property tests):

- F1: a phase report resolves to exactly one of the five graph outcomes,
  with the documented precedence (fail > unreached > evidence_absent >
  all-not-applicable > pass), checked against an independent oracle;
- enforcement follows the battery's own blocking rule for every posture:
  ``artifact_permitted`` holds exactly when no entry blocks, the phase was
  reached and no upstream phase blocked, checked against an independent
  oracle over criticality, status, ``population_fact_check``,
  ``evidence_absent_blocks``, release-candidate and synthetic-smoke postures.

Example tests:

- F4: a kernel exception becomes the node outcome ``fail`` and its artifact
  consumers become ``unreached``; an exception inside one binding fails that
  gate closed without masking the rest of the batch;
- a report is self-consistent: an inconsistent outcome, permission or row
  status is refused on decode;
- phase order: every earlier phase that can block must be wired as upstream,
  under the identical manifest document, resource hash and posture, and a
  blocked upstream makes this phase ``unreached`` (declared
  ``not_applicable`` entries keep that status);
- a binding's fields, function defaults and the first-party modules its code
  reaches enter the implementation hash; a binding outside the describable
  vocabulary is refused when the kernel is built, and one mutated afterwards
  refuses to run;
- gate details must be plain data, and common Python repr addresses in failure
  text and details are blanked; bindings control their text and item order;
- a gate manifest edit re-keys the gate and its descendants only.

The bindings are test-only stand-ins (``test_support``); the neutral
bindings for a real country are a separate work package.
"""

from __future__ import annotations

import enum
import functools
import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.country_spec import GateSelectionSpec
from microcosm.build.gate_battery import (
    DEFAULT_REGISTRY,
    FunctionBinding,
    GateOutcome,
    GatePhaseReport,
    GateStatus,
)
from microcosm.build.gates import GateResult
from microcosm.build.transport.artifact_types import GATE_REPORT_TYPE
from microcosm.build.transport.gate_kernels import (
    GateBatteryKernel,
    decode_gate_report,
    phase_enforcement,
    phase_outcome,
    register_gate_kernel,
)
from microcosm.graph import (
    GATE_OUTCOMES,
    ArtifactOutput,
    ArtifactValue,
    ContentStore,
    Kernel,
    KernelContext,
    KernelRegistry,
    KernelRole,
    Node,
    NumericScope,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.transport_graph import (
    COUNTRY,
    DEFAULT_CONFIG,
    TOY_BINDINGS,
    canonical_text,
    descendants,
    run_through,
    support_gate,
    toy_gates,
    toy_graph,
    toy_registry,
    with_config,
    write_toy_sources,
)

GATE = "toy.gates.terminal"
_KEY = "c" * 64


# ---------------------------------------------------------------------------
# F1 and enforcement: pure functions against independent oracles
# ---------------------------------------------------------------------------


def _outcome(
    index: int,
    criticality: str,
    status: GateStatus,
    *,
    fact_check: bool = False,
    absent_blocks: bool = False,
) -> GateOutcome:
    entry = GateSelectionSpec(
        id=f"g{index}",
        gate="support",
        phase="terminal",
        criticality=criticality,
        population_fact_check=fact_check,
        evidence_absent_blocks=absent_blocks,
    )
    if status in (GateStatus.PASSED, GateStatus.FAILED):
        passed = status is GateStatus.PASSED
        result = GateResult("support", passed, failures=() if passed else ("no",))
        return GateOutcome(entry=entry, status=status, result=result)
    return GateOutcome(entry=entry, status=status, reason="declared")


def _outcome_oracle(pairs) -> str:
    blocking = {
        status for criticality, status, *_ in pairs if criticality == "release_blocking"
    }
    every = {status for _, status, *_ in pairs}
    if GateStatus.FAILED in blocking:
        return "fail"
    if GateStatus.UNREACHED in every:
        return "unreached"
    if GateStatus.EVIDENCE_ABSENT in blocking:
        return "evidence_absent"
    if every <= {GateStatus.NOT_APPLICABLE}:
        return "not_applicable"
    return "pass"


def _blocks_oracle(entry, *, release_candidate: bool, synthetic_smoke: bool) -> bool:
    criticality, status, fact_check, absent_blocks = entry
    if criticality != "release_blocking":
        return False
    if status is GateStatus.FAILED:
        return not (synthetic_smoke and fact_check)
    if status is GateStatus.EVIDENCE_ABSENT:
        return release_candidate or absent_blocks
    return False


_ENTRIES = st.lists(
    st.tuples(
        st.sampled_from(("release_blocking", "diagnostic")),
        st.sampled_from(tuple(GateStatus)),
        st.booleans(),
        st.booleans(),
    ),
    max_size=8,
)


def _report(entries) -> GatePhaseReport:
    return GatePhaseReport(
        phase="terminal",
        outcomes=tuple(
            _outcome(index, criticality, status, fact_check=fact, absent_blocks=absent)
            for index, (criticality, status, fact, absent) in enumerate(entries)
        ),
    )


@settings(max_examples=300, deadline=None)
@given(_ENTRIES)
def test_property_phase_outcome_is_exactly_one_of_five(entries) -> None:
    outcome = phase_outcome(_report(entries))

    assert outcome in GATE_OUTCOMES
    assert outcome == _outcome_oracle(entries)
    if outcome == "pass":
        assert not any(
            criticality == "release_blocking"
            and status in (GateStatus.FAILED, GateStatus.EVIDENCE_ABSENT)
            for criticality, status, *_ in entries
        )


@settings(max_examples=400, deadline=None)
@given(
    _ENTRIES,
    st.sampled_from(((False, False), (True, False), (False, True))),
    st.booleans(),
)
def test_property_enforcement_follows_the_battery_blocking_rule(
    entries, posture, upstream_blocked
) -> None:
    release_candidate, synthetic_smoke = posture
    decision = phase_enforcement(
        _report(entries),
        release_candidate=release_candidate,
        synthetic_smoke=synthetic_smoke,
        upstream_blocked=upstream_blocked,
    )
    expected_blocking = tuple(
        f"g{index}"
        for index, entry in enumerate(entries)
        if _blocks_oracle(
            entry, release_candidate=release_candidate, synthetic_smoke=synthetic_smoke
        )
    )

    assert decision.outcome == _outcome_oracle(entries)
    assert decision.blocking == expected_blocking
    assert decision.artifact_permitted is (
        not expected_blocking
        and decision.outcome != "unreached"
        and not upstream_blocked
    )
    if decision.outcome == "fail" and not synthetic_smoke:
        assert decision.artifact_permitted is False


def test_enforcement_refuses_a_smoke_release_candidate() -> None:
    with pytest.raises(ValueError, match="cannot be a release candidate"):
        phase_enforcement(
            _report([]),
            release_candidate=True,
            synthetic_smoke=True,
            upstream_blocked=False,
        )


# ---------------------------------------------------------------------------
# The kernel contract
# ---------------------------------------------------------------------------


def _other_support_gate(*, frame, min_persons):  # pragma: no cover - not evaluated
    return GateResult("support", True)


def _exploding_fit_gate(*, diagnostics, max_abs_relative_error):
    raise RuntimeError("binding bug")


def _support_with_object_details(*, frame, min_persons):
    return GateResult("support", True, details={"checked": _Floor(min_persons)})


def _support_with_address_details(*, frame, min_persons):
    return GateResult("support", True, details={f"key {object()}": f"value {object()}"})


def _support_with_repr_failure(*, frame, min_persons):
    return GateResult("support", False, failures=(f"refused {object()} at the floor",))


def _support_with_code_repr(*, frame, min_persons):
    text = repr(support_gate.__code__)
    return GateResult("support", False, failures=(text,), details={"code": text})


def _support_with_positional_default(frame, min_persons=1):
    return GateResult("support", True)


class _Mode(enum.Enum):
    STRICT = "strict"
    LENIENT = "lenient"


@dataclass(frozen=True)
class _Floor:
    minimum: int


@dataclass(frozen=True)
class _CustomBinding:
    """A non-FunctionBinding binding: every field is behaviour."""

    name: str
    threshold_key: str
    parameter_keys: frozenset[str] = frozenset({"min_persons"})
    mode: _Mode = _Mode.STRICT
    blob: bytes = b""
    floor: _Floor | None = None

    def required_artifacts(self, parameters):
        return frozenset()

    def requires_frame(self, parameters):
        return True

    def evaluate(self, context, parameters):
        return support_gate(
            frame=context.frame, min_persons=parameters[self.threshold_key]
        )

    def evidence_payload(self, context, parameters):
        return None


def _with_support(binding) -> dict:
    return {**TOY_BINDINGS, "support": binding}


def _hash_with_gate(gate) -> str:
    binding = replace(TOY_BINDINGS["support"], gate=gate)
    return GateBatteryKernel(_with_support(binding)).implementation_hash()


def test_kernel_is_a_gate_and_binds_its_registry_into_its_hash() -> None:
    kernel = GateBatteryKernel(TOY_BINDINGS)
    assert isinstance(kernel, Kernel)
    assert kernel.ref == "gates.battery@1"
    assert kernel.capabilities.role is KernelRole.GATE
    assert (
        kernel.implementation_hash()
        == GateBatteryKernel(TOY_BINDINGS).implementation_hash()
    )
    assert GateBatteryKernel(DEFAULT_REGISTRY).implementation_hash() != (
        kernel.implementation_hash()
    )
    widened = replace(
        TOY_BINDINGS["support"], parameter_keys=frozenset({"min_persons", "extra"})
    )
    assert GateBatteryKernel(_with_support(widened)).implementation_hash() != (
        kernel.implementation_hash()
    )
    assert _hash_with_gate(_other_support_gate) != kernel.implementation_hash()
    with pytest.raises(ValueError, match="must carry that name"):
        GateBatteryKernel({"support": TOY_BINDINGS["weight_ratio"]})


def test_binding_fields_defaults_and_reached_modules_enter_the_hash() -> None:
    one = _CustomBinding("support", "min_persons")
    base = GateBatteryKernel(_with_support(one)).implementation_hash()
    for change in (
        {"threshold_key": "other"},
        {"mode": _Mode.LENIENT},
        {"blob": b"b"},
        {"floor": _Floor(2)},
    ):
        changed = GateBatteryKernel(_with_support(replace(one, **change)))
        assert changed.implementation_hash() != base, change

    defaults = _other_support_gate.__kwdefaults__
    before = _hash_with_gate(_other_support_gate)
    try:
        _other_support_gate.__kwdefaults__ = {"min_persons": 7}
        assert _hash_with_gate(_other_support_gate) != before
    finally:
        _other_support_gate.__kwdefaults__ = defaults

    # A gate's helpers in other first-party modules are bound too.
    reached = set(GateBatteryKernel(DEFAULT_REGISTRY)._identity[1])
    assert {"microcosm.build.gates", "microcosm.build.ledger_targets"} <= reached


def test_bindings_outside_the_vocabulary_are_refused_at_construction() -> None:
    with pytest.raises(
        ValueError, match="closes over|decorated|top level|module-level"
    ):
        _hash_with_gate(functools.wraps(support_gate)(lambda **kw: None))
    with pytest.raises(ValueError, match="outside the describable vocabulary"):
        _hash_with_gate(functools.partial(support_gate, min_persons=3))


def test_positional_function_defaults_enter_the_binding_hash() -> None:
    defaults = _support_with_positional_default.__defaults__
    before = _hash_with_gate(_support_with_positional_default)
    try:
        _support_with_positional_default.__defaults__ = (7,)
        assert _hash_with_gate(_support_with_positional_default) != before
    finally:
        _support_with_positional_default.__defaults__ = defaults


def test_a_binding_mutated_after_construction_refuses_to_run() -> None:
    arguments = {"diagnostics": "diagnostics"}
    binding = replace(TOY_BINDINGS["per_family_fit"], artifact_arguments=arguments)
    kernel = GateBatteryKernel({**TOY_BINDINGS, "per_family_fit": binding})
    kernel.run(_context(_node()))
    arguments["diagnostics"] = "other_alias"
    with pytest.raises(ValueError, match="changed after the kernel was built"):
        kernel.run(_context(_node()))


def test_registration_is_idempotent_for_the_same_binding_objects() -> None:
    registry = KernelRegistry()
    first = register_gate_kernel(registry, TOY_BINDINGS)
    assert register_gate_kernel(registry, dict(TOY_BINDINGS)) is first
    rebuilt = {**TOY_BINDINGS, "support": replace(TOY_BINDINGS["support"])}
    with pytest.raises(ValueError, match="already registered"):
        register_gate_kernel(registry, rebuilt)


def test_gate_details_must_be_plain_data_and_failure_text_loses_addresses() -> None:
    from test_support.microcosm_build.transport_graph import ENTITIES

    node = _node(entities=ENTITIES, weight_entity="household")
    objects = replace(TOY_BINDINGS["support"], gate=_support_with_object_details)
    gates = toy_gates(
        extra=(
            {
                "id": "toy_floor",
                "gate": "support",
                "phase": "terminal",
                "criticality": "diagnostic",
                "parameters": {"min_persons": 1},
            },
        )
    )
    node = _node(gates, entities=ENTITIES, weight_entity="household")
    with pytest.raises(ValueError, match="not plain data"):
        GateBatteryKernel(_with_support(objects)).run(_frame_context(node))

    text = replace(TOY_BINDINGS["support"], gate=_support_with_repr_failure)
    first = GateBatteryKernel(_with_support(text)).run(_frame_context(node))
    second = GateBatteryKernel(_with_support(text)).run(_frame_context(node))
    assert first.artifacts["gate_report"] == second.artifacts["gate_report"]
    assert b"object at 0x>" in first.artifacts["gate_report"]
    decode_gate_report(first.artifacts["gate_report"])

    keyed = replace(TOY_BINDINGS["support"], gate=_support_with_address_details)
    one = GateBatteryKernel(_with_support(keyed)).run(_frame_context(node))
    two = GateBatteryKernel(_with_support(keyed)).run(_frame_context(node))
    assert one.artifacts["gate_report"] == two.artifacts["gate_report"]
    assert b"key <object object at 0x>" in one.artifacts["gate_report"]
    assert b"value <object object at 0x>" in one.artifacts["gate_report"]

    code = replace(TOY_BINDINGS["support"], gate=_support_with_code_repr)
    payload = (
        GateBatteryKernel(_with_support(code))
        .run(_frame_context(node))
        .artifacts["gate_report"]
    )
    outcome = next(
        item
        for item in decode_gate_report(payload).report.outcomes
        if item.entry.id == "toy_floor"
    )
    assert " at 0x," in outcome.result.failures[0]
    assert " at 0x," in outcome.result.details["code"]


def _node(gates=None, *, phase="terminal", **params) -> Node:
    return Node(
        "gate",
        "gates.battery@1",
        population="terminal",
        params={
            "country": COUNTRY,
            "gates": canonical_text(toy_gates() if gates is None else gates),
            "gates_sha256": "2" * 64,
            "phase": phase,
            "release_candidate": False,
            **params,
        },
        artifact_outputs=(ArtifactOutput("gate_report", GATE_REPORT_TYPE),),
    )


def _context(node: Node, artifacts=None) -> KernelContext:
    return KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series([], dtype=object, name="stratum"),
        params=node.params,
        rng=np.random.default_rng(0),
        artifacts={} if artifacts is None else artifacts,
    )


def _report_value(payload: bytes) -> ArtifactValue:
    return ArtifactValue(
        payload=payload,
        type=GATE_REPORT_TYPE,
        key=_KEY,
        producer_key=_KEY,
        numerics=NumericScope(),
    )


def test_frameless_phase_names_its_missing_evidence() -> None:
    result = GateBatteryKernel(TOY_BINDINGS).run(_context(_node()))
    report = decode_gate_report(result.artifacts["gate_report"])

    assert result.receipt["outcome"] == "evidence_absent"
    assert result.receipt["evidence"]["statuses"] == {
        "toy_weight_ratio": "evidence_absent",
        "toy_fit": "evidence_absent",
        "toy_nonnegative": "evidence_absent",
    }
    # A development posture records the gap and still permits the artifact.
    assert report.artifact_permitted is True
    reasons = {o.entry.id: o.reason for o in report.report.outcomes}
    assert reasons["toy_weight_ratio"] == "missing evidence: problem, solution"
    assert reasons["toy_nonnegative"] == "missing evidence: frame"

    release = GateBatteryKernel(TOY_BINDINGS).run(
        _context(_node(release_candidate=True))
    )
    assert decode_gate_report(release.artifacts["gate_report"]).artifact_permitted is (
        False
    )


def test_configuration_errors_raise_rather_than_return_a_verdict() -> None:
    kernel = GateBatteryKernel(TOY_BINDINGS)
    typo = toy_gates()
    typo["gates"][0]["parameters"] = {"max_ratoi": 2.0}
    with pytest.raises(ValueError, match="does not route"):
        kernel.run(_context(_node(typo)))
    with pytest.raises(ValueError, match="not in the declared order"):
        kernel.run(_context(_node(phase="simulated")))
    with pytest.raises(ValueError, match="unknown parameter"):
        kernel.run(_context(_node(threshold=1.0)))
    with pytest.raises(ValueError, match="declares country"):
        kernel.run(_context(_node(country="yy")))
    with pytest.raises(ValueError, match="typed outputs"):
        kernel.run(_context(replace(_node(), artifact_outputs=())))
    with pytest.raises(
        ValueError,
        match=r"^gates\.battery@1: a synthetic smoke build cannot be a release "
        r"candidate\.$",
    ):
        kernel.run(_context(_node(release_candidate=True, synthetic_smoke=True)))
    # The posture is refused before the manifest is even consulted.
    with pytest.raises(ValueError, match="cannot be a release candidate"):
        kernel.run(
            _context(
                _node(phase="simulated", release_candidate=True, synthetic_smoke=True)
            )
        )
    with pytest.raises(ValueError, match="come together"):
        kernel.run(_context(_node(weight_entity="household")))


def test_decode_recomputes_and_refuses_a_forged_report() -> None:
    dev = json.loads(
        GateBatteryKernel(TOY_BINDINGS).run(_context(_node())).artifacts["gate_report"]
    )
    release = json.loads(
        GateBatteryKernel(TOY_BINDINGS)
        .run(_context(_node(release_candidate=True)))
        .artifacts["gate_report"]
    )

    forged_outcome = json.loads(json.dumps(dev))
    forged_outcome["outcome"] = "pass"
    forged_permit = json.loads(json.dumps(release))
    forged_permit["enforcement"]["artifact_permitted"] = True
    forged_status = json.loads(json.dumps(release))
    forged_status["report"]["outcomes"][0]["status"] = "not_applicable"
    forged_status["report"]["outcomes"][0]["reason"] = "forged"
    forged_posture = json.loads(json.dumps(release))
    forged_posture["enforcement"]["release_candidate"] = False
    smoke_candidate = json.loads(json.dumps(release))
    smoke_candidate["enforcement"]["synthetic_smoke"] = True
    for forged, match in (
        (forged_outcome, "differs from its replayed outcomes"),
        (forged_permit, "differs from its replayed outcomes"),
        (forged_status, "different gate manifest|differ from its manifest|differs"),
        (forged_posture, "differs from its replayed outcomes"),
        (smoke_candidate, "cannot be a release candidate"),
    ):
        with pytest.raises(ValueError, match=match):
            decode_gate_report(canonical_json(forged))
    with pytest.raises(ValueError, match="canonical JSON"):
        decode_gate_report(json.dumps(dev).encode())


def _forged_rows(report: dict, change) -> bytes:
    forged = json.loads(json.dumps(report))
    change(forged["report"]["outcomes"])
    return canonical_json(forged)


def test_decode_refuses_rows_the_battery_could_not_have_written(tmp_path) -> None:
    sources = write_toy_sources(tmp_path / "src")
    manifest, store = run_through(tmp_path, sources, DEFAULT_CONFIG, GATE)
    report = json.loads(
        store.load_bytes(manifest.nodes[GATE].opaque_artifacts["gate_report"])
    )
    decode_gate_report(canonical_json(report))

    def extra_key(rows):
        rows[0]["extra"] = 1

    def failures_as_text(rows):
        rows[0]["failures"] = "none"

    def unblanked_address(rows):
        rows[0]["status"] = "failed"
        rows[0]["failures"] = ["<object object at 0x10f3a2b50>"]

    def bad_evidence(rows):
        rows[0]["evidence_sha256"] = "nothex"

    def reason_on_passed(rows):
        rows[0]["reason"] = "invented"

    def renamed_result(rows):
        rows[0]["result_name"] = "zzz"

    for change in (
        extra_key,
        failures_as_text,
        unblanked_address,
        bad_evidence,
        reason_on_passed,
        renamed_result,
    ):
        with pytest.raises(ValueError):
            decode_gate_report(_forged_rows(report, change))


def test_decode_refuses_rows_that_contradict_the_manifest_or_upstream(
    tmp_path,
) -> None:
    sources = write_toy_sources(tmp_path / "src")
    manifest, store = run_through(tmp_path, sources, DEFAULT_CONFIG, GATE)
    report = json.loads(
        store.load_bytes(manifest.nodes[GATE].opaque_artifacts["gate_report"])
    )

    def invented_not_applicable(rows):
        rows[0].update(
            status="not_applicable",
            reason="invented",
            failures=[],
            details={},
            result_name=None,
        )

    def unreached_without_block(rows):
        rows[0].update(
            status="unreached", reason=None, failures=[], details={}, result_name=None
        )

    for change in (invented_not_applicable, unreached_without_block):
        with pytest.raises(ValueError):
            decode_gate_report(_forged_rows(report, change))

    for key, value in (("schema_version", True), ("country", None)):
        forged = json.loads(json.dumps(report))
        forged[key] = value
        with pytest.raises(ValueError):
            decode_gate_report(canonical_json(forged))


def test_decode_refuses_inconsistent_upstream_entries() -> None:
    gates = _preflight_gates(min_persons=1)
    passed = _preflight_report(gates)
    terminal = json.loads(
        GateBatteryKernel(TOY_BINDINGS)
        .run(
            _context(
                _node(gates, upstream=("preflight",)),
                {"preflight": _report_value(passed)},
            )
        )
        .artifacts["gate_report"]
    )
    decode_gate_report(canonical_json(terminal))

    missing_alias = json.loads(json.dumps(terminal))
    missing_alias["evidence"] = []
    impossible = json.loads(json.dumps(terminal))
    impossible["upstream"]["preflight"]["outcome"] = "unreached"
    dropped = json.loads(json.dumps(terminal))
    dropped["upstream"] = {}
    pass_blocked = json.loads(json.dumps(terminal))
    pass_blocked["upstream"]["preflight"]["artifact_permitted"] = False
    pass_blocked["enforcement"]["upstream_blocked"] = ["preflight"]
    for forged, match in (
        (missing_alias, "upstream entries"),
        (impossible, "upstream entries"),
        (dropped, "one upstream report per earlier phase"),
        (pass_blocked, "upstream entries|reached exactly"),
    ):
        with pytest.raises(ValueError, match=match):
            decode_gate_report(canonical_json(forged))


def test_decode_refuses_unreached_rows_without_blocking_upstream() -> None:
    report = json.loads(
        GateBatteryKernel(TOY_BINDINGS).run(_context(_node())).artifacts["gate_report"]
    )
    for row in report["report"]["outcomes"]:
        row.update(status="unreached", reason=None)
    # Keep the derived verdict and enforcement consistent with the forged rows.
    report["outcome"] = "unreached"
    report["enforcement"].update(blocking=[], artifact_permitted=False)
    assert report["upstream"] == {}
    assert report["enforcement"]["upstream_blocked"] == []

    with pytest.raises(ValueError, match="reached exactly when no upstream phase"):
        decode_gate_report(canonical_json(report))


def test_decode_refuses_a_passing_upstream_phase_that_blocks_the_artifact() -> None:
    gates = _preflight_gates(min_persons=1000)
    blocked = _preflight_report(gates)
    report = json.loads(
        GateBatteryKernel(TOY_BINDINGS)
        .run(
            _context(
                _node(gates, upstream=("preflight",)),
                {"preflight": _report_value(blocked)},
            )
        )
        .artifacts["gate_report"]
    )
    decode_gate_report(canonical_json(report))
    assert report["enforcement"]["upstream_blocked"] == ["preflight"]
    report["upstream"]["preflight"]["outcome"] = "pass"

    with pytest.raises(ValueError, match="upstream entries are malformed"):
        decode_gate_report(canonical_json(report))


# ---------------------------------------------------------------------------
# Phase order
# ---------------------------------------------------------------------------


def _preflight_gates(*, min_persons: int, extra=()) -> dict:
    return toy_gates(
        extra=(
            {
                "id": "toy_support_floor",
                "gate": "support",
                "phase": "preflight",
                "criticality": "release_blocking",
                "parameters": {"min_persons": min_persons},
            },
            *extra,
        )
    )


def _frame_context(node: Node, artifacts=None) -> KernelContext:
    from test_support.microcosm_build.transport_graph import ENTITIES, toy_frame

    frame = toy_frame()
    return KernelContext(
        node=node,
        tables={entity: frame.table(entity) for entity in ENTITIES},
        weights={"household": frame.weights_for("household")},
        strata=frame.strata,
        params=node.params,
        rng=np.random.default_rng(0),
        artifacts={} if artifacts is None else artifacts,
    )


def _preflight_report(gates, **params) -> bytes:
    from test_support.microcosm_build.transport_graph import ENTITIES

    node = _node(
        gates, phase="preflight", entities=ENTITIES, weight_entity="household", **params
    )
    return (
        GateBatteryKernel(TOY_BINDINGS)
        .run(_frame_context(node))
        .artifacts["gate_report"]
    )


def test_upstream_reports_must_cover_earlier_phases_under_one_manifest_and_posture() -> (
    None
):
    kernel = GateBatteryKernel(TOY_BINDINGS)
    gates = _preflight_gates(min_persons=1)
    passed = _preflight_report(gates)

    # An applicable earlier phase that is not wired is refused.
    with pytest.raises(ValueError, match="each earlier phase"):
        kernel.run(_context(_node(gates)))
    # Wired, it evaluates.
    ok = kernel.run(
        _context(
            _node(gates, upstream=("preflight",)),
            {"preflight": _report_value(passed)},
        )
    )
    assert decode_gate_report(ok.artifacts["gate_report"]).phase == "terminal"
    # A report from another manifest is refused.
    other = _preflight_report(_preflight_gates(min_persons=2))
    with pytest.raises(ValueError, match="different gate manifest"):
        kernel.run(
            _context(
                _node(gates, upstream=("preflight",)),
                {"preflight": _report_value(other)},
            )
        )
    # A report enforced under another posture is refused.
    with pytest.raises(ValueError, match="different posture"):
        kernel.run(
            _context(
                _node(gates, upstream=("preflight",), release_candidate=True),
                {"preflight": _report_value(passed)},
            )
        )
    # A later phase is not an upstream.
    terminal = kernel.run(
        _context(
            _node(gates, upstream=("preflight",)),
            {"preflight": _report_value(passed)},
        )
    ).artifacts["gate_report"]
    with pytest.raises(ValueError, match="does not precede"):
        kernel.run(
            _context(
                _node(gates, phase="preflight", upstream=("later",)),
                {"later": _report_value(terminal)},
            )
        )
    with pytest.raises(
        ValueError, match="declared gates.phase-report|must be a declared"
    ):
        kernel.run(_context(_node(gates, upstream=("missing",))))


def test_upstream_must_share_the_manifest_document_not_just_its_policy_hash() -> None:
    """``population_fact_check`` is outside the battery's policy hash."""

    lenient = _preflight_gates(min_persons=1000)
    strict = json.loads(json.dumps(lenient))
    lenient["gates"][-1]["population_fact_check"] = True
    smoke = _preflight_report(lenient, synthetic_smoke=True)
    assert decode_gate_report(smoke).artifact_permitted is True
    with pytest.raises(ValueError, match="different gate manifest"):
        GateBatteryKernel(TOY_BINDINGS).run(
            _context(
                _node(strict, upstream=("preflight",), synthetic_smoke=True),
                {"preflight": _report_value(smoke)},
            )
        )
    with pytest.raises(ValueError, match="different gate manifest"):
        GateBatteryKernel(TOY_BINDINGS).run(
            _context(
                _node(
                    lenient,
                    upstream=("preflight",),
                    synthetic_smoke=True,
                    gates_sha256="9" * 64,
                ),
                {"preflight": _report_value(smoke)},
            )
        )


def test_an_earlier_phase_that_cannot_block_need_not_be_wired() -> None:
    diagnostic_only = toy_gates(
        extra=(
            {
                "id": "toy_support_note",
                "gate": "support",
                "phase": "preflight",
                "criticality": "diagnostic",
                "parameters": {"min_persons": 1000},
            },
        )
    )
    result = GateBatteryKernel(TOY_BINDINGS).run(_context(_node(diagnostic_only)))
    assert result.receipt["evidence"]["upstream_blocked"] == []


def test_a_blocked_upstream_leaves_not_applicable_entries_not_applicable() -> None:
    excused = {
        "id": "toy_excused",
        "gate": "macro_realism",
        "phase": "terminal",
        "criticality": "release_blocking",
        "not_applicable": "Excused for the toy.",
    }
    gates = _preflight_gates(min_persons=1000, extra=(excused,))
    blocked = _preflight_report(gates)
    assert decode_gate_report(blocked).artifact_permitted is False
    result = GateBatteryKernel(TOY_BINDINGS).run(
        _context(
            _node(gates, upstream=("preflight",)),
            {"preflight": _report_value(blocked)},
        )
    )
    statuses = result.receipt["evidence"]["statuses"]

    assert result.receipt["outcome"] == "unreached"
    assert statuses["toy_excused"] == "not_applicable"
    assert {statuses[k] for k in statuses if k != "toy_excused"} == {"unreached"}
    assert result.receipt["evidence"]["artifact_permitted"] is False
    assert result.receipt["evidence"]["upstream_blocked"] == ["preflight"]


# ---------------------------------------------------------------------------
# F1 and F4 on the toy graph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "config", "endpoint", "expected"),
    [
        ("pass", DEFAULT_CONFIG, GATE, "pass"),
        (
            "fail",
            with_config(DEFAULT_CONFIG, gates=toy_gates(max_ratio=1.0)),
            GATE,
            "fail",
        ),
        (
            "evidence_absent",
            with_config(
                DEFAULT_CONFIG,
                gates=toy_gates(
                    extra=(
                        {
                            "id": "toy_macro",
                            "gate": "macro_realism",
                            "phase": "terminal",
                            "criticality": "release_blocking",
                        },
                    )
                ),
            ),
            GATE,
            "evidence_absent",
        ),
        (
            "not_applicable",
            with_config(DEFAULT_CONFIG, preflight=True),
            "toy.gates.preflight",
            "not_applicable",
        ),
        (
            "unreached",
            with_config(
                DEFAULT_CONFIG, preflight=True, gates=_preflight_gates(min_persons=1000)
            ),
            GATE,
            "unreached",
        ),
    ],
)
def test_gate_node_yields_each_of_the_five_outcomes(
    tmp_path, case, config, endpoint, expected
) -> None:
    sources = write_toy_sources(tmp_path / "src", config)
    manifest, store = run_through(tmp_path, sources, config, endpoint)
    receipt = manifest.nodes[endpoint]
    report = decode_gate_report(
        store.load_bytes(receipt.opaque_artifacts["gate_report"])
    )

    assert receipt.receipt["outcome"] == expected
    assert report.outcome == expected
    statuses = set(receipt.receipt["evidence"]["statuses"].values())
    if case == "unreached":
        assert statuses == {"unreached"}
        assert report.artifact_permitted is False
        assert manifest.nodes["toy.gates.preflight"].receipt["outcome"] == "fail"
    if case == "fail":
        assert "toy_weight_ratio" in receipt.receipt["evidence"]["blocking"]
    if case == "not_applicable":
        assert statuses == {"not_applicable"}


def _swap_gate(graph, node):
    return replace(
        graph, nodes=tuple(node if n.id == node.id else n for n in graph.nodes)
    )


def test_a_gate_kernel_exception_fails_the_node_and_unreaches_consumers(
    tmp_path,
) -> None:
    sources = write_toy_sources(tmp_path / "src")
    graph = toy_graph(DEFAULT_CONFIG, through_prepare=True)
    gate = graph.node(GATE)
    broken = replace(gate, params={**gate.params, "gates": '{"not": "canonical" }'})
    manifest = run_graph(
        compile_graph(_swap_gate(graph, broken)),
        sources=sources.mapping(),
        store=ContentStore(tmp_path / "store"),
        kernels=toy_registry(),
    )
    failed = manifest.nodes[GATE].receipt
    assert failed["outcome"] == "fail"
    assert failed["evidence"]["exception_type"] == "ValueError"
    assert failed["execution"]["state"] == "gate_exception"
    assert manifest.nodes["toy.export.prepare"].receipt["outcome"] == "unreached"
    assert GATE in manifest.known_failures


def test_a_raising_binding_fails_closed_without_masking_the_batch(tmp_path) -> None:
    bindings = dict(TOY_BINDINGS)
    bindings["per_family_fit"] = FunctionBinding(
        name="per_family_fit",
        gate=_exploding_fit_gate,
        parameter_keys=frozenset({"max_abs_relative_error"}),
        artifact_arguments={"diagnostics": "diagnostics"},
    )
    sources = write_toy_sources(tmp_path / "src")
    manifest, store = run_through(
        tmp_path, sources, DEFAULT_CONFIG, GATE, registry=toy_registry(bindings)
    )
    report = decode_gate_report(
        store.load_bytes(manifest.nodes[GATE].opaque_artifacts["gate_report"])
    )
    rows = {row["id"]: row for row in report.document["report"]["outcomes"]}

    assert manifest.nodes[GATE].receipt["outcome"] == "fail"
    assert rows["toy_fit"]["status"] == "failed"
    assert "failed closed with RuntimeError" in rows["toy_fit"]["failures"][0]
    assert rows["toy_weight_ratio"]["status"] == "passed"
    assert rows["toy_nonnegative"]["status"] == "passed"


def test_a_synthetic_smoke_build_records_but_does_not_block_a_fact_check(
    tmp_path,
) -> None:
    fact_check = {
        "id": "toy_population_floor",
        "gate": "support",
        "phase": "terminal",
        "criticality": "release_blocking",
        "parameters": {"min_persons": 1000},
        "population_fact_check": True,
    }
    config = with_config(DEFAULT_CONFIG, gates=toy_gates(extra=(fact_check,)))
    graph = toy_graph(config, through_prepare=True)
    gate = graph.node(GATE)
    smoke = replace(gate, params={**gate.params, "synthetic_smoke": True})
    sources = write_toy_sources(tmp_path / "src", config)
    manifest = run_graph(
        compile_graph(_swap_gate(graph, smoke)),
        sources=sources.mapping(),
        store=ContentStore(tmp_path / "store"),
        kernels=toy_registry(),
    )
    receipt = manifest.nodes[GATE].receipt
    assert receipt["outcome"] == "fail"
    assert tuple(receipt["evidence"]["blocking"]) == ()
    assert receipt["evidence"]["artifact_permitted"] is True
    # The export proceeds; the failed fact check stays on the record.
    assert manifest.nodes["toy.export.prepare"].receipt.get("outcome") is None
    assert GATE in manifest.known_failures


def test_a_gate_manifest_edit_rekeys_only_the_gate_and_its_descendants(
    tmp_path,
) -> None:
    sources = write_toy_sources(tmp_path / "src")
    edited = with_config(DEFAULT_CONFIG, gates=toy_gates(max_ratio=2.5))
    before, _ = run_through(
        tmp_path / "a", sources, DEFAULT_CONFIG, "toy.export.prepare"
    )
    after, _ = run_through(tmp_path / "b", sources, edited, "toy.export.prepare")
    moved = {
        node for node in before.nodes if before.nodes[node].key != after.nodes[node].key
    }

    graph = toy_graph(DEFAULT_CONFIG, through_prepare=True)
    assert moved == descendants(graph, {GATE})
    assert "toy.calibrate" not in moved
    assert "toy.targets.problem" not in moved
