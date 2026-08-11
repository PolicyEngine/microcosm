"""Country-agnostic gate battery: phased evaluation, unconditional reporting.

microcosm#578's acceptance rule ships a candidate iff it beats the incumbent on
the frozen comparison register **and** the versioned invariant battery passes —
"machine-checked in one batched pass, and recorded in both manifests". This
module is the one executor that makes that sentence true the same way for
every country (microcosm#611): a country declares *which* gates run, *when*,
and with *what* thresholds as pure data in its ``gates.json``
(:class:`~microcosm.build.country_spec.GatesManifest`); the battery evaluates
each phase's gates in one batch, writes the full report **before** any
blocking decision, and leaves the block itself to a separate call so the
report on disk can never be a casualty of the failure it documents.

Three disciplines, held by construction rather than convention:

- **Batched within a phase, fail-closed at the phase boundary.** Every gate in
  a phase evaluates even when an earlier one fails (a raising evaluator
  becomes a failed result, never a masked one), and the build stops at the
  boundary where the failure surfaced — a gate sits where its evidence
  appears without giving up the single batched report.
- **Write-then-block.** :meth:`GateBatteryRun.run_phase` persists the full
  report — later phases as ``unreached`` placeholders — before returning;
  :meth:`GateBatteryRun.enforce` may only run afterwards. A build killed
  mid-run leaves a report saying exactly which phases ran, what failed, and
  what was never reached.
- **Every declared gate has exactly one outcome.** ``passed`` / ``failed`` /
  ``not_applicable`` (a reviewed reason declared in the spec) /
  ``evidence_absent`` (no implementation or missing evidence — a named gap,
  never a silent omission) / ``unreached`` (an earlier phase blocked first).
  Conflating ``unreached`` with ``not_applicable`` is the failure mode the
  battery exists to prevent, so the report carries ``phases_evaluated`` and
  ``blocked_at_phase`` and the attestation covers both.

The gate *comparisons* stay in :mod:`microcosm.build.gates` — pure functions
from evidence to :class:`~microcosm.build.gates.GateResult`, shared across
countries. This module adds the *binding* layer between declared gate names
and those functions: a :class:`GateBinding` extracts a gate's evidence from
the phase's :class:`EvidenceContext` and translates declared parameters into
gate keyword arguments. Countries differ in data, not code.
"""

from __future__ import annotations

import base64
import binascii
import enum
import hashlib
import hmac
import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from microcosm.build.country_spec import CountrySpec, GateSelectionSpec, GatesManifest
from microcosm.build.gates import (
    GateResult,
    input_mass_parity_gate,
    tail_concentration_gate,
    weights_audit_gate,
)
from microcosm.build.trace import compute_composition_fingerprint
from microcosm.frame import Frame

__all__ = [
    "BlockingMode",
    "DEFAULT_REGISTRY",
    "EvidenceContext",
    "FunctionBinding",
    "GATE_BATTERY_ATTESTATION_SCHEMA_VERSION",
    "GATE_BATTERY_PRODUCER",
    "GATE_BATTERY_SCHEMA_VERSION",
    "GATE_BATTERY_SIGNATURE_ALGORITHM",
    "GateBatteryBlockedError",
    "GateBatteryRun",
    "GateBinding",
    "GateOutcome",
    "GatePhaseReport",
    "GateStatus",
    "evaluate_phase",
    "gate_signing_key_env",
]

#: Continues the ``terminal_gates.json`` numbering: schema 3 is the UK-only
#: report the June release shipped (attestation 5); schema 4 is the shared
#: battery shape with phases and the five-state outcome taxonomy.
GATE_BATTERY_SCHEMA_VERSION = 4
GATE_BATTERY_ATTESTATION_SCHEMA_VERSION = 6
GATE_BATTERY_SIGNATURE_ALGORITHM = "hmac-sha256"
GATE_BATTERY_PRODUCER = "microcosm.build.gate_battery"


def gate_signing_key_env(country: str) -> str:
    """Environment variable carrying the country's report-signing key.

    The UK convention (``POPULACE_UK_TERMINAL_GATE_SIGNING_KEY``),
    generalized: the same name pattern per country, each a base64-encoded
    32-byte key.
    """

    if not country or not country.isascii() or not country.isalpha():
        raise ValueError(f"country must be an ASCII country code, got {country!r}.")
    return f"MICROCOSM_{country.upper()}_TERMINAL_GATE_SIGNING_KEY"


# ---------------------------------------------------------------------------
# Canonical JSON and signing (the terminal_gates.py conventions, shared)
# ---------------------------------------------------------------------------


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _signing_key(country: str) -> bytes:
    """Read the country's 256-bit report-signing key from the environment."""

    env = gate_signing_key_env(country)
    encoded = os.environ.get(env)
    if not encoded:
        raise RuntimeError(
            f"{env} must contain a base64-encoded 32-byte key before "
            "attesting a gate battery report."
        )
    try:
        key = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"{env} must be valid base64.") from exc
    if len(key) != 32:
        raise RuntimeError(f"{env} must decode to exactly 32 bytes.")
    return key


def _json_safe(value: object) -> object:
    """Coerce a gate detail payload into strict-JSON territory.

    Gate details are written by many hands; the report writer must never
    crash on one of them (write-then-block would die with it). Numpy scalars
    collapse to their Python values, non-finite floats and unknown objects
    are preserved as readable strings rather than dropped.
    """

    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return repr(value)
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = (
            list(value)
            if not isinstance(value, (set, frozenset))
            else sorted(value, key=repr)
        )
        return [_json_safe(child) for child in items]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _gates_manifest_payload(gates: GatesManifest) -> dict[str, object]:
    """Project the exact evaluated manifest into canonical JSON data."""

    return {
        "country": gates.country,
        "version": gates.version,
        "policy": gates.policy,
        "phases": list(gates.phases),
        "gates": [
            {
                "id": entry.id,
                "gate": entry.gate,
                "phase": entry.phase,
                "criticality": entry.criticality,
                "parameters": _json_safe(entry.parameters),
                "not_applicable": entry.not_applicable,
                "notes": entry.notes,
            }
            for entry in gates.gates
        ],
    }


# ---------------------------------------------------------------------------
# Evidence and bindings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceContext:
    """Everything one phase can offer its gates.

    Attributes:
        frame: The phase's carrier, when the phase has one (every stage
            boundary does once a build carries
            :class:`~microcosm.frame.Frame`; preflight phases may run
            frameless).
        artifacts: Named runtime evidence the build tool supplies — objects
            not derivable from the frame, e.g. ``"fit_weight_records"``,
            ``"reference_input_mass_totals"``, a parity snapshot, a coverage
            engine adapter. A gate whose declared artifacts are absent
            reports ``evidence_absent`` with the missing keys named.
        spec: The loaded country spec, when available, so bindings can
            derive surfaces from declared manifests (e.g. the QRF output
            surface from ``fit_weighted_qrf_stage*`` operations) instead of
            hand lists.
    """

    frame: Frame | None = None
    artifacts: Mapping[str, object] = field(default_factory=dict)
    spec: CountrySpec | None = None


class GateBinding(Protocol):
    """Binds an allowlisted gate name to evidence extraction and evaluation.

    The gate *comparison* lives in :mod:`microcosm.build.gates`; the binding
    is the country-agnostic adapter that knows what evidence the comparison
    needs, where in the :class:`EvidenceContext` it lives, and how declared
    spec parameters map onto gate keyword arguments.

    ``parameter_keys`` is the binding's declared parameter vocabulary: the
    complete set of spec ``parameters`` keys it can route into the gate.
    :func:`validate_gate_parameters` refuses any declared key outside it
    before a single gate runs, so a typo'd or speculative parameter cannot
    sit inside ``policy_sha256`` while governing nothing.
    """

    name: str
    parameter_keys: frozenset[str]

    def required_artifacts(self, parameters: Mapping[str, Any]) -> frozenset[str]:
        """Artifact keys the gate needs, checked before evaluation."""
        ...

    def requires_frame(self, parameters: Mapping[str, Any]) -> bool:
        """Whether the gate needs the phase's frame."""
        ...

    def evaluate(
        self, context: EvidenceContext, parameters: Mapping[str, Any]
    ) -> GateResult:
        """Run the gate over the context; returns its verdict."""
        ...

    def evidence_payload(
        self, context: EvidenceContext, parameters: Mapping[str, Any]
    ) -> object | None:
        """Canonicalizable payload for the entry's ``evidence_sha256`` line,
        or ``None`` when the evidence is fully spec-declared."""
        ...


@dataclass(frozen=True)
class FunctionBinding:
    """A :class:`GateBinding` over a pure gate function.

    Declared spec ``parameters`` pass through as keyword arguments verbatim
    (an unknown parameter raises inside the evaluator and fails the gate
    closed); ``artifact_arguments`` routes named context artifacts into the
    remaining keyword arguments; ``frame_argument`` routes the phase frame.

    Attributes:
        name: The allowlisted gate name.
        gate: The pure gate function returning a
            :class:`~microcosm.build.gates.GateResult` whose ``name`` matches.
        parameter_keys: The spec parameter keys this binding routes — the
            gate function's declarable keyword surface. Fail-closed empty by
            default: a binding that declares nothing accepts no parameters.
        artifact_arguments: Gate keyword argument -> context artifact key.
        frame_argument: Gate keyword argument receiving the phase frame, or
            ``None`` for frameless gates.
        evidence: Optional hook producing the entry's evidence payload from
            ``(context, parameters)``; ``None`` records no evidence hash.
    """

    name: str
    gate: Callable[..., GateResult]
    parameter_keys: frozenset[str] = frozenset()
    artifact_arguments: Mapping[str, str] = field(default_factory=dict)
    frame_argument: str | None = None
    evidence: Callable[[EvidenceContext, Mapping[str, Any]], object] | None = None

    def required_artifacts(self, parameters: Mapping[str, Any]) -> frozenset[str]:
        return frozenset(self.artifact_arguments.values())

    def requires_frame(self, parameters: Mapping[str, Any]) -> bool:
        return self.frame_argument is not None

    def evaluate(
        self, context: EvidenceContext, parameters: Mapping[str, Any]
    ) -> GateResult:
        kwargs: dict[str, Any] = dict(parameters)
        for argument, artifact_key in self.artifact_arguments.items():
            kwargs[argument] = context.artifacts[artifact_key]
        if self.frame_argument is not None:
            kwargs[self.frame_argument] = context.frame
        return self.gate(**kwargs)

    def evidence_payload(
        self, context: EvidenceContext, parameters: Mapping[str, Any]
    ) -> object | None:
        if self.evidence is None:
            return None
        return self.evidence(context, parameters)


def _input_mass_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    return {
        "candidate_totals": _json_safe(
            context.artifacts["candidate_input_mass_totals"]
        ),
        "reference_totals": _json_safe(
            context.artifacts["reference_input_mass_totals"]
        ),
    }


#: Bindings for gates whose evidence already travels as plain data. The
#: registry grows as consumers migrate (microcosm#611 increments 2-3); a
#: declared gate with no binding is a named ``evidence_absent`` gap in the
#: report, never a crash, so an incomplete registry cannot manufacture a
#: pass.
DEFAULT_REGISTRY: Mapping[str, GateBinding] = {
    "weights_audit": FunctionBinding(
        name="weights_audit",
        gate=weights_audit_gate,
        parameter_keys=frozenset({"allowed_unweighted"}),
        artifact_arguments={"fit_records": "fit_weight_records"},
    ),
    "input_mass_parity": FunctionBinding(
        name="input_mass_parity",
        gate=input_mass_parity_gate,
        parameter_keys=frozenset(
            {
                "candidate_name",
                "reference_name",
                "relative_tolerance",
                "minimum_reference_total",
                "reviewed_exclusions",
            }
        ),
        artifact_arguments={
            "candidate_totals": "candidate_input_mass_totals",
            "reference_totals": "reference_input_mass_totals",
        },
        evidence=_input_mass_evidence,
    ),
    "tail_concentration": FunctionBinding(
        name="tail_concentration",
        gate=tail_concentration_gate,
        parameter_keys=frozenset(
            {
                "top_k",
                "max_top_share",
                "min_nonzero_records",
                "reviewed_exclusions",
            }
        ),
        artifact_arguments={
            "column_values": "tail_concentration_values",
            "column_weights": "tail_concentration_weights",
        },
    ),
}


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


class GateStatus(enum.Enum):
    """Exactly one per declared gate entry, in every report."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    EVIDENCE_ABSENT = "evidence_absent"
    UNREACHED = "unreached"


#: Statuses a release-blocking entry may hold on a shippable report.
_SHIPPABLE_STATUSES = frozenset({GateStatus.PASSED, GateStatus.NOT_APPLICABLE})


@dataclass(frozen=True)
class GateOutcome:
    """One declared entry's resolved status.

    Attributes:
        entry: The spec entry this outcome resolves.
        status: The resolved :class:`GateStatus`.
        result: The gate's verdict — present iff the gate actually
            evaluated (``passed`` / ``failed``).
        reason: Why the gate did not evaluate — required for
            ``not_applicable`` (the reviewed spec reason) and
            ``evidence_absent`` (the named gap).
        evidence_sha256: Canonical hash of the evidence the gate consumed,
            when its binding supplies a payload.
    """

    entry: GateSelectionSpec
    status: GateStatus
    result: GateResult | None = None
    reason: str | None = None
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        evaluated = self.status in (GateStatus.PASSED, GateStatus.FAILED)
        if evaluated:
            if self.result is None:
                raise ValueError(
                    f"gate {self.entry.id!r}: status {self.status.value!r} "
                    "requires a GateResult."
                )
            if self.result.passed is not (self.status is GateStatus.PASSED):
                raise ValueError(
                    f"gate {self.entry.id!r}: status {self.status.value!r} "
                    "contradicts the gate result."
                )
        else:
            if self.result is not None:
                raise ValueError(
                    f"gate {self.entry.id!r}: status {self.status.value!r} "
                    "cannot carry a GateResult."
                )
            if (
                self.status in (GateStatus.NOT_APPLICABLE, GateStatus.EVIDENCE_ABSENT)
                and not self.reason
            ):
                raise ValueError(
                    f"gate {self.entry.id!r}: status {self.status.value!r} "
                    "requires a reason."
                )

    def to_payload(self) -> dict[str, object]:
        """The entry's JSON form in the battery report."""

        return {
            "gate": self.entry.gate,
            "phase": self.entry.phase,
            "criticality": self.entry.criticality,
            "status": self.status.value,
            "failures": (
                [str(line) for line in self.result.failures]
                if self.result is not None
                else []
            ),
            "details": (
                _json_safe(dict(self.result.details)) if self.result is not None else {}
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GatePhaseReport:
    """One phase's batched outcomes."""

    phase: str
    outcomes: tuple[GateOutcome, ...]

    def blocking_outcomes(self, *, release_candidate: bool) -> tuple[GateOutcome, ...]:
        """Outcomes that stop the build at this phase boundary.

        A failed release-blocking gate always blocks. A release-blocking
        gate with absent evidence blocks release candidates only: a dev
        build without, say, an incumbent parity snapshot gets an honest
        non-shippable report instead of a crash, while a release build
        cannot excuse missing evidence — a missing frozen reference is not
        a passing gate. Diagnostic entries never block.
        """

        blocking = []
        for outcome in self.outcomes:
            if outcome.entry.criticality != "release_blocking":
                continue
            if outcome.status is GateStatus.FAILED:
                blocking.append(outcome)
            elif outcome.status is GateStatus.EVIDENCE_ABSENT and release_candidate:
                blocking.append(outcome)
        return tuple(blocking)

    @property
    def failures(self) -> tuple[str, ...]:
        """Every failure line across the phase, entry-prefixed."""

        return tuple(
            f"[{outcome.entry.id}] {line}"
            for outcome in self.outcomes
            if outcome.result is not None
            for line in outcome.result.failures
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_gate(name: str, evaluator: Callable[[], GateResult]) -> GateResult:
    """Run one evaluator, failing closed on any misbehaviour.

    The one fail-closed wrapper for every battery, shared with the legacy UK
    terminal report: a raising evaluator becomes a failed result (the batch
    must keep evaluating — a crash that masked the remaining gates would
    hide exactly the failures the battery exists to surface), and a result
    under the wrong name fails rather than letting one gate impersonate
    another.
    """

    try:
        result = evaluator()
    except Exception as exc:  # noqa: BLE001 - the batch must keep evaluating
        return GateResult(
            name=name,
            passed=False,
            failures=(
                f"Gate evaluation failed closed with {type(exc).__name__}: {exc}",
            ),
            details={
                "evaluation_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            },
        )
    if not isinstance(result, GateResult):
        return GateResult(
            name=name,
            passed=False,
            failures=(
                "Gate evaluation failed closed because the evaluator did not "
                "return GateResult.",
            ),
            details={"returned_type": type(result).__name__},
        )
    if result.name != name:
        return GateResult(
            name=name,
            passed=False,
            failures=(
                f"Gate evaluator returned name {result.name!r}, expected {name!r}.",
            ),
            details={"returned_gate": result.name},
        )
    return result


def _resolve_entry(
    entry: GateSelectionSpec,
    context: EvidenceContext,
    registry: Mapping[str, GateBinding],
) -> GateOutcome:
    if entry.not_applicable is not None:
        return GateOutcome(
            entry=entry,
            status=GateStatus.NOT_APPLICABLE,
            reason=entry.not_applicable,
        )
    binding = registry.get(entry.gate)
    if binding is None:
        return GateOutcome(
            entry=entry,
            status=GateStatus.EVIDENCE_ABSENT,
            reason=f"no implementation registered for gate {entry.gate!r}",
        )
    missing = sorted(
        binding.required_artifacts(entry.parameters) - set(context.artifacts)
    )
    if binding.requires_frame(entry.parameters) and context.frame is None:
        missing.insert(0, "frame")
    if missing:
        return GateOutcome(
            entry=entry,
            status=GateStatus.EVIDENCE_ABSENT,
            reason=f"missing evidence: {', '.join(missing)}",
        )
    result = _evaluate_gate(
        entry.gate, lambda: binding.evaluate(context, entry.parameters)
    )
    evidence_sha256: str | None = None
    try:
        payload = binding.evidence_payload(context, entry.parameters)
        if payload is not None:
            evidence_sha256 = _canonical_sha256(payload)
    except Exception as exc:  # noqa: BLE001 - unattestable evidence fails closed
        if result.passed:
            result = GateResult(
                name=entry.gate,
                passed=False,
                failures=(
                    "Gate passed but its evidence could not be attested: "
                    f"{type(exc).__name__}: {exc}",
                ),
                details={
                    "evidence_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                },
            )
    return GateOutcome(
        entry=entry,
        status=GateStatus.PASSED if result.passed else GateStatus.FAILED,
        result=result,
        evidence_sha256=evidence_sha256,
    )


def validate_gate_parameters(
    gates: GatesManifest,
    registry: Mapping[str, GateBinding],
) -> None:
    """Refuse declared parameters outside their binding's vocabulary.

    The entry-level schema check (``GateSelectionSpec.from_mapping``) closes
    the world one level up; this closes it inside ``parameters``, where the
    same argument applies: a key the binding cannot route would ship inside
    ``policy_sha256`` while governing nothing — declared intent with no
    effect on the verdict. Runs over the whole manifest so a typo in any
    phase surfaces before the first gate does.

    Entries whose gate has no binding are skipped: they resolve to
    ``evidence_absent`` and run nothing, so their parameters govern nothing
    *visibly*. A binding that does not declare ``parameter_keys`` accepts no
    parameters — fail closed, never fail open.

    Raises:
        ValueError: Naming the first offending entry and its unknown keys.
    """

    for entry in gates.gates:
        binding = registry.get(entry.gate)
        if binding is None:
            continue
        allowed = getattr(binding, "parameter_keys", frozenset())
        unknown = sorted(set(entry.parameters) - set(allowed))
        if unknown:
            raise ValueError(
                f"gate entry {entry.id!r} declares parameters {unknown} that "
                f"the {entry.gate!r} binding does not route; vocabulary: "
                f"{sorted(allowed)}. A silently ignored parameter would sit "
                "inside policy_sha256 while governing nothing."
            )


def evaluate_phase(
    gates: GatesManifest,
    phase: str,
    context: EvidenceContext,
    *,
    registry: Mapping[str, GateBinding] = DEFAULT_REGISTRY,
) -> GatePhaseReport:
    """Evaluate every gate the manifest binds to ``phase``, in one batch.

    Never raises on a gate failure and never stops early: every entry in the
    phase resolves to exactly one outcome, misbehaving evaluators fail
    closed, and blocking is the caller's decision — downstream of
    persistence, via :class:`GateBatteryRun`.

    Raises:
        ValueError: If ``phase`` is not in the manifest's declared order, or
            if any entry declares parameters outside its binding's
            vocabulary (:func:`validate_gate_parameters`) — configuration
            errors, not gate verdicts.
    """

    if phase not in gates.phases:
        raise ValueError(
            f"phase {phase!r} is not in the declared order {list(gates.phases)}."
        )
    validate_gate_parameters(gates, registry)
    outcomes = tuple(
        _resolve_entry(entry, context, registry)
        for entry in gates.gates
        if entry.phase == phase
    )
    return GatePhaseReport(phase=phase, outcomes=outcomes)


# ---------------------------------------------------------------------------
# The battery run
# ---------------------------------------------------------------------------


class BlockingMode(enum.Enum):
    """What a blocked phase does to the build's artifact."""

    #: Publications: nothing is written downstream of a block (the UK and US
    #: national releases — the H5 write sits after ``enforce``).
    BLOCKS_ARTIFACT = "blocks_artifact"
    #: Intermediates: the artifact is written anyway and the manifest records
    #: the verdict (the US pool builder — a pool with a failed agreement gate
    #: is still worth keeping, it just is not ready).
    MARKS_ARTIFACT = "marks_artifact"


class GateBatteryBlockedError(RuntimeError):
    """A phase's release-blocking gates stopped the build.

    Raised by :meth:`GateBatteryRun.enforce` only after the full report —
    including the block itself — is on disk.
    """

    def __init__(self, phase: str, failures: Sequence[str], report_path: Path) -> None:
        self.phase = phase
        self.failures = tuple(failures)
        self.report_path = report_path
        lines = "\n".join(f"  - {line}" for line in self.failures)
        super().__init__(
            f"Gate battery blocked at phase {phase!r} (report: {report_path}):\n{lines}"
        )


class GateBatteryRun:
    """One build's battery: the declared gates, their outcomes, the report.

    Phases run in the manifest's declared order, each through
    :meth:`run_phase` (which persists the full report before returning) and
    then :meth:`enforce` (which blocks — or marks — strictly after
    persistence). Once a phase blocks, no later phase may run: the remaining
    entries stay ``unreached`` in the report, which is the truth.

    Args:
        gates: The country's gate manifest.
        release_id: Identity of the build the report attests.
        report_path: Where the report JSON lives; written atomically at
            every phase boundary.
        release_candidate: Whether this build may ship. Release candidates
            cannot excuse absent evidence; dev builds record it and
            continue.
        registry: Gate bindings; defaults to :data:`DEFAULT_REGISTRY`.
        release_evidence: Digests of release inputs the gates themselves do
            not consume but the release contract links (for the UK, the
            calibration-diagnostics digest). Carried in the report and the
            signed attestation, so the linkage survives the build that
            attested it.
    """

    def __init__(
        self,
        gates: GatesManifest,
        *,
        release_id: str,
        report_path: Path | str,
        release_candidate: bool,
        registry: Mapping[str, GateBinding] = DEFAULT_REGISTRY,
        release_evidence: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(release_id, str) or not release_id.strip():
            raise ValueError("release_id must be a non-empty string.")
        validate_gate_parameters(gates, registry)
        evidence = dict(release_evidence or {})
        for key, value in evidence.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("release_evidence keys must be non-empty strings.")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"release_evidence[{key!r}] must be a non-empty string digest."
                )
        self._gates = gates
        self._release_id = release_id
        self._report_path = Path(report_path)
        self._release_candidate = bool(release_candidate)
        self._registry = dict(registry)
        self._release_evidence = dict(sorted(evidence.items()))
        self._gates_manifest_sha256 = _canonical_sha256(
            _gates_manifest_payload(self._gates)
        )
        self._spec_fingerprint = compute_composition_fingerprint(
            (self._gates_manifest_sha256,)
        )
        self._phase_reports: dict[str, GatePhaseReport] = {}
        self._blocked_at_phase: str | None = None

    @property
    def report_path(self) -> Path:
        return self._report_path

    @property
    def blocked_at_phase(self) -> str | None:
        return self._blocked_at_phase

    @property
    def phases_evaluated(self) -> tuple[str, ...]:
        return tuple(self._phase_reports)

    def phase_report(self, phase: str) -> GatePhaseReport:
        """The evaluated report for ``phase``; refuses a phase that has not run."""

        try:
            return self._phase_reports[phase]
        except KeyError:
            raise ValueError(
                f"phase {phase!r} has not run; evaluated: {list(self._phase_reports)}."
            ) from None

    def _next_phase(self) -> str | None:
        for phase in self._gates.phases:
            if phase not in self._phase_reports:
                return phase
        return None

    def run_phase(self, phase: str, context: EvidenceContext) -> GatePhaseReport:
        """Evaluate one phase and persist the full report before returning.

        Raises:
            ValueError: If a phase runs out of declared order, re-runs, or
                runs after a block — each would make the persisted
                ``unreached`` placeholders a lie.
        """

        if self._blocked_at_phase is not None:
            raise ValueError(
                f"battery blocked at phase {self._blocked_at_phase!r}; no "
                "later phase may run."
            )
        expected = self._next_phase()
        if expected is None:
            raise ValueError("every declared phase has already run.")
        if phase != expected:
            raise ValueError(
                f"phase {phase!r} is out of order; expected {expected!r} "
                f"(declared order {list(self._gates.phases)})."
            )
        report = evaluate_phase(self._gates, phase, context, registry=self._registry)
        self._phase_reports[phase] = report
        self._write_report()
        return report

    def enforce(self, phase: str, *, mode: BlockingMode) -> bool:
        """Apply the phase's blocking verdict, strictly after persistence.

        Returns:
            Whether the phase blocked. Under
            :attr:`BlockingMode.BLOCKS_ARTIFACT` a block raises instead —
            after the report, block included, is re-persisted.

        Raises:
            GateBatteryBlockedError: A blocking outcome under
                ``BLOCKS_ARTIFACT``.
            ValueError: If the phase has not run, or is not the most recent
                phase to have run.
        """

        evaluated = list(self._phase_reports)
        if not evaluated or evaluated[-1] != phase:
            raise ValueError(
                f"enforce({phase!r}) must follow run_phase({phase!r}); "
                f"evaluated so far: {evaluated}."
            )
        report = self._phase_reports[phase]
        blocking = report.blocking_outcomes(release_candidate=self._release_candidate)
        if not blocking:
            return False
        self._blocked_at_phase = phase
        self._write_report()
        if mode is BlockingMode.BLOCKS_ARTIFACT:
            failures = list(report.failures)
            for outcome in blocking:
                if outcome.status is GateStatus.EVIDENCE_ABSENT:
                    failures.append(f"[{outcome.entry.id}] {outcome.reason}")
            raise GateBatteryBlockedError(phase, failures, self._report_path)
        return True

    # -- report assembly ----------------------------------------------------

    def _outcomes_by_entry(self) -> Iterator[GateOutcome]:
        evaluated: dict[str, GateOutcome] = {
            outcome.entry.id: outcome
            for report in self._phase_reports.values()
            for outcome in report.outcomes
        }
        for entry in self._gates.gates:
            outcome = evaluated.get(entry.id)
            if outcome is not None:
                yield outcome
            elif entry.not_applicable is not None:
                # Declared non-applicability is evaluation-independent: the
                # entry would not have run in any phase, reached or not.
                yield GateOutcome(
                    entry=entry,
                    status=GateStatus.NOT_APPLICABLE,
                    reason=entry.not_applicable,
                )
            else:
                yield GateOutcome(entry=entry, status=GateStatus.UNREACHED)

    def _policy_sha256(self) -> str:
        return _canonical_sha256(
            [
                {
                    "id": entry.id,
                    "gate": entry.gate,
                    "phase": entry.phase,
                    "criticality": entry.criticality,
                    "parameters": _json_safe(dict(entry.parameters)),
                    "not_applicable": entry.not_applicable,
                }
                for entry in sorted(self._gates.gates, key=lambda e: e.id)
            ]
        )

    def report_payload(self) -> dict[str, object]:
        """The full schema-4 report, attested when a signing key is present."""

        outcomes = list(self._outcomes_by_entry())
        gates_payload = {outcome.entry.id: outcome.to_payload() for outcome in outcomes}
        evidence_sha256 = {
            outcome.entry.id: outcome.evidence_sha256
            for outcome in outcomes
            if outcome.evidence_sha256 is not None
        }
        policy_sha256 = self._policy_sha256()
        signing_error: RuntimeError | None = None
        try:
            signing_key: bytes | None = _signing_key(self._gates.country)
        except RuntimeError as exc:
            signing_key = None
            signing_error = exc
        blocking_ok = all(
            outcome.status in _SHIPPABLE_STATUSES
            for outcome in outcomes
            if outcome.entry.criticality == "release_blocking"
        )
        shippable = (
            self._release_candidate
            and self._blocked_at_phase is None
            and blocking_ok
            and signing_key is not None
            and self._gates_manifest_sha256 is not None
        )
        attestation: dict[str, object] = {
            "schema_version": GATE_BATTERY_ATTESTATION_SCHEMA_VERSION,
            "producer": GATE_BATTERY_PRODUCER,
            "country": self._gates.country,
            "release_id": self._release_id,
            "release_candidate": self._release_candidate,
            "spec_fingerprint": self._spec_fingerprint,
            "gates_manifest_sha256": self._gates_manifest_sha256,
            "policy_sha256": policy_sha256,
            "phases": list(self._gates.phases),
            "phases_evaluated": list(self._phase_reports),
            "blocked_at_phase": self._blocked_at_phase,
            "release_evidence": dict(self._release_evidence),
            "evidence_sha256": dict(evidence_sha256),
            "gate_outcomes_sha256": _canonical_sha256(gates_payload),
            "signature_algorithm": GATE_BATTERY_SIGNATURE_ALGORITHM,
            "signing_key_sha256": (
                hashlib.sha256(signing_key).hexdigest()
                if signing_key is not None
                else None
            ),
            "signature": None,
        }
        if signing_error is not None:
            attestation["signing_error"] = str(signing_error)
        payload: dict[str, object] = {
            "schema_version": GATE_BATTERY_SCHEMA_VERSION,
            "country": self._gates.country,
            "release_id": self._release_id,
            "release_candidate": self._release_candidate,
            "spec_fingerprint": self._spec_fingerprint,
            "gates_manifest_sha256": self._gates_manifest_sha256,
            "phases": list(self._gates.phases),
            "phases_evaluated": list(self._phase_reports),
            "blocked_at_phase": self._blocked_at_phase,
            "shippable": shippable,
            "gates": gates_payload,
            "policy_sha256": policy_sha256,
            "release_evidence": dict(self._release_evidence),
            "evidence_sha256": dict(evidence_sha256),
            "attestation": attestation,
        }
        if signing_key is not None:
            # The UK signing dance, shared: sign the canonical report with a
            # null signature slot, then fill the slot — a verifier re-nulls
            # it and recomputes. The signature must be valid over FAILED
            # reports too: a signed failure is evidence, an unsigned one is
            # a hole.
            attestation["signature"] = hmac.new(
                signing_key, _canonical_json_bytes(payload), hashlib.sha256
            ).hexdigest()
        return payload

    def _write_report(self) -> None:
        payload = self.report_payload()
        path = self._report_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
