"""``gates.battery@1``: one gate-battery phase as a graph GATE node.

The kernel is a thin wrapper over the shared battery
(:func:`microcosm.build.gate_battery.evaluate_phase`). A node declares one
phase of any country's ``gates.json``; the kernel evaluates that phase's
gates in one batch against the evidence the node declares, and returns:

- the phase report as a ``microcosm.gates.phase-report`` artifact: the
  battery's own portable payload
  (:func:`~microcosm.build.gate_battery.gate_phase_report_payload`), the gate
  manifest it was evaluated against, and the enforcement decision for the
  node's posture; and
- exactly one graph outcome (``microcosm.graph.GATE_OUTCOMES``) in the
  receipt, from :func:`phase_outcome`.

The report is self-consistent and replayable: :func:`decode_gate_report`
replays its outcomes against the manifest document it carries and recomputes
the outcome and the enforcement, so an inconsistent permission is refused. A
report cannot prove which manifest it was *meant* to use; consumers
(:func:`require_terminal_gate_reports`) therefore require every report they
read to carry the same manifest document, resource hash and posture, and the
composer must wire ``gate_report`` edges from ``gates.battery@1`` nodes only.

Evidence. Every declared artifact input is decoded by its nominal type
(:data:`EVIDENCE_DECODERS`) and offered to the battery under its local alias,
so a binding's ``artifact_arguments`` name aliases the composer declares. A
type with no decoder is offered as its verified bytes. When the node declares
``entities`` (with ``weight_entity``), the population slices it reads are
rebuilt as the phase's ``frame``. No spec object is offered: thresholds reach
a gate only through the ``gates.json`` document the node carries as a
parameter.

Bindings. The kernel takes its binding registry as a constructor argument and
binds each binding's behaviour into its implementation hash
(:mod:`~microcosm.build.transport.binding_identity`): every binding must be a
frozen dataclass built from a closed vocabulary (plain data, enums, classes
and undecorated top-level functions without closures), and the hash covers
its fields, its functions' defaults and the source of every first-party module
its code can import, read from the import statements in the source. A binding
outside that vocabulary is refused when the kernel is built. A binding, or a
module in its source closure, that changes afterwards (or can no longer be
read or parsed) makes the kernel raise
:class:`~microcosm.graph.KernelIdentityChangedError`: from
``implementation_hash``, and from ``run``, which checks before it evaluates,
before it returns, and when evaluation raises. The executor never files that
error as a ``fail`` verdict (graph amendment 29), so a change made while a
build runs leaves no record for the refused gate node under the unchanged
binding's key. Data a binding's modules read from disk is not bound, so a binding
takes its values through the manifest's parameters or evidence artifacts. A
declared gate with no binding still resolves to a named ``evidence_absent``
gap, exactly as the battery does.

Determinism. Gate details must be plain data (the battery would otherwise keep
an object's ``repr``). Common Python repr addresses (`` at 0x...>`` or
`` at 0x...,``) inside failure text and details are blanked. Other text and
detail ordering are the binding's: failure lines, details and exception text
must not depend on unordered set iteration, or the report's bytes would vary
between runs of one computation.

Phase order. The battery runs phases in their declared order and none after a
block (``GateBatteryRun.run_phase``). In the graph, a node names the reports of
earlier phases as ``upstream`` artifact inputs: every earlier phase that
declares an applicable release-blocking entry (the only kind that can block)
must be named, no phase twice, each evaluated under this node's posture and
the identical manifest document and resource hash. If any of them does not
permit the artifact, this phase does not evaluate: its entries are
``unreached``, except those declared ``not_applicable``, which keep that status
as the battery's own report does (``GateBatteryRun.outcomes``).

Report signing (``gate_battery.py`` attestation) needs a key and a release
id; both belong to the outer command that materializes a release, never to a
cached kernel.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

import microcosm.build.country_spec as country_spec_module
import microcosm.build.gate_battery as gate_battery_module
import microcosm.build.gates as gates_module
import microcosm.calibrate.artifacts as calibrate_artifacts_module
import microcosm.frame.bundle as frame_bundle_module
import microcosm.frame.schema as frame_schema_module
from microcosm.build.country_spec import GatesManifest
from microcosm.build.gate_battery import (
    EvidenceContext,
    GateBinding,
    GateOutcome,
    GatePhaseReport,
    GateStatus,
    evaluate_phase,
    gate_phase_report_from_payload,
    gate_phase_report_payload,
    validate_gate_parameters,
)
from microcosm.calibrate.artifacts import (
    PROBLEM_TYPE,
    SOLUTION_TYPE,
    decode_problem,
    decode_solution,
)
from microcosm.graph import (
    GATE_OUTCOMES,
    ArtifactType,
    ArtifactValue,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelIdentityChangedError,
    KernelRegistry,
    KernelResult,
    KernelRole,
    Numeric,
    SeedSource,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from . import artifact_types, binding_identity, graph_inputs, target_kernels
from .artifact_types import (
    COMPARISON_TYPE,
    DIAGNOSTICS_TYPE,
    EXPORT_DESCRIPTOR_TYPE,
    EXPORT_READBACK_TYPE,
    GATE_REPORT_TYPE,
    TARGET_SURFACE_TYPE,
)
from .graph_inputs import (
    bool_param,
    canonical_document_param,
    context_frame,
    entities_param,
    optional_bool_param,
    require_outputs,
    require_params,
    sha256_param,
    sha256_text,
    string_param,
)
from .target_kernels import decode_target_surface

__all__ = [
    "EVIDENCE_DECODERS",
    "GateBatteryKernel",
    "GateReport",
    "PhaseEnforcement",
    "decode_gate_report",
    "phase_enforcement",
    "phase_outcome",
    "register_gate_kernel",
    "require_terminal_gate_reports",
]

_REF = "gates.battery@1"
_REPORT_KIND = "transport_gate_phase_report"
#: The address in common Python reprs (objects, functions and code objects):
#: the hex after `` at `` and before a closing ``>`` or a comma.
_ADDRESS = re.compile(r"(?<= at )0x[0-9a-fA-F]+(?=[,>])")


def phase_outcome(report: GatePhaseReport) -> str:
    """The one graph outcome a phase report resolves to.

    Precedence, first match wins:

    1. ``fail``: a release-blocking entry failed.
    2. ``unreached``: an entry was not reached (an earlier phase blocked).
    3. ``evidence_absent``: a release-blocking entry lacked its evidence or
       implementation.
    4. ``not_applicable``: the phase has no entries, or every entry carries a
       reviewed ``not_applicable`` reason.
    5. ``pass``: otherwise. A failed or evidence-absent *diagnostic* entry is
       recorded in the report but does not fail the phase, matching the
       battery's blocking rule (``GatePhaseReport.blocking_outcomes``).
    """

    outcomes = report.outcomes
    blocking = [o for o in outcomes if o.entry.criticality == "release_blocking"]
    if any(o.status is GateStatus.FAILED for o in blocking):
        return "fail"
    if any(o.status is GateStatus.UNREACHED for o in outcomes):
        return "unreached"
    if any(o.status is GateStatus.EVIDENCE_ABSENT for o in blocking):
        return "evidence_absent"
    if all(o.status is GateStatus.NOT_APPLICABLE for o in outcomes):
        return "not_applicable"
    return "pass"


@dataclass(frozen=True)
class PhaseEnforcement:
    """What a phase report decides for one posture.

    Attributes:
        outcome: The graph outcome (:func:`phase_outcome`).
        blocking: Ids of the entries that block under the posture.
        artifact_permitted: Whether a downstream artifact may be produced:
            nothing blocks, the phase was reached, and no upstream phase
            blocked.
    """

    outcome: str
    blocking: tuple[str, ...]
    artifact_permitted: bool


def phase_enforcement(
    report: GatePhaseReport,
    *,
    release_candidate: bool,
    synthetic_smoke: bool,
    upstream_blocked: bool,
) -> PhaseEnforcement:
    """Apply the battery's blocking rule to one phase report.

    ``blocking`` is :meth:`GatePhaseReport.blocking_outcomes` for the
    posture. Outside a synthetic smoke build a ``fail`` outcome always has a
    blocking entry; inside one, a failed ``population_fact_check`` entry is
    recorded without blocking, as the battery does.
    """

    if release_candidate and synthetic_smoke:
        raise ValueError("A synthetic smoke build cannot be a release candidate.")
    outcome = phase_outcome(report)
    blocking = tuple(
        item.entry.id
        for item in report.blocking_outcomes(
            release_candidate=release_candidate, synthetic_smoke=synthetic_smoke
        )
    )
    permitted = not blocking and outcome != "unreached" and not upstream_blocked
    return PhaseEnforcement(outcome, blocking, permitted)


@dataclass(frozen=True)
class GateReport:
    """A decoded, replayed ``microcosm.gates.phase-report`` artifact."""

    country: str
    phase: str
    outcome: str
    artifact_permitted: bool
    release_candidate: bool
    synthetic_smoke: bool
    gates_sha256: str
    gates: GatesManifest
    report: GatePhaseReport
    document: Mapping[str, object]


_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "country",
        "phase",
        "gates",
        "gates_sha256",
        "outcome",
        "report",
        "enforcement",
        "upstream",
        "evidence",
    }
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def decode_gate_report(payload: bytes) -> GateReport:
    """Replay a phase report and recompute everything it decides.

    The stored outcomes are replayed against the manifest the report carries
    (:func:`~microcosm.build.gate_battery.gate_phase_report_from_payload`
    refuses a manifest whose policy hash differs, or rows that do not cover
    the phase) and must re-serialize to themselves; each row must be one the
    battery could have produced; the upstream entries must be consistent and
    name one earlier report per phase that can block; and the outcome and the
    enforcement are recomputed for the recorded posture and compared with
    what the report claims. Any malformed report raises ``ValueError``.
    """

    if type(payload) is not bytes:
        raise TypeError("A gate report is immutable bytes.")
    try:
        return _decode_gate_report(payload)
    except (AttributeError, KeyError, RecursionError, TypeError) as error:
        raise ValueError(
            f"Malformed gate report ({type(error).__name__}: {error})."
        ) from error


_ROW_KEYS = frozenset(
    {
        "id",
        "gate",
        "phase",
        "criticality",
        "status",
        "failures",
        "details",
        "reason",
        "result_name",
        "evidence_sha256",
    }
)


def _check_rows(rows: list) -> None:
    """Each row must be one the battery could have produced."""

    for row in rows:
        evaluated = row["status"] in ("passed", "failed")
        evidence = row["evidence_sha256"]
        if (
            set(row) != _ROW_KEYS
            or not isinstance(row["failures"], list)
            or not isinstance(row["details"], dict)
            or (evidence is not None and not _is_sha256(evidence))
            or (
                evaluated
                and (row["reason"] is not None or row["result_name"] != row["gate"])
            )
            or (
                not evaluated
                and (
                    row["failures"]
                    or row["details"]
                    or row["result_name"] is not None
                    or evidence is not None
                )
            )
        ):
            raise ValueError(
                f"Gate report row {row.get('id')!r} is not one the battery produces."
            )


def _decode_gate_report(payload: bytes) -> GateReport:
    document = json.loads(payload)
    if not isinstance(document, dict) or canonical_json(document) != payload:
        raise ValueError("A gate report must be canonical JSON.")
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or document.get("kind") != _REPORT_KIND
        or set(document) != _REPORT_KEYS
        or not isinstance(document["country"], str)
        or not isinstance(document["phase"], str)
    ):
        raise ValueError("Unsupported gate report artifact.")
    enforcement = document["enforcement"]
    upstream = document["upstream"]
    if (
        not isinstance(enforcement, dict)
        or not isinstance(upstream, dict)
        or set(enforcement)
        != {
            "release_candidate",
            "synthetic_smoke",
            "blocking",
            "artifact_permitted",
            "upstream_blocked",
        }
        or not isinstance(enforcement["release_candidate"], bool)
        or not isinstance(enforcement["synthetic_smoke"], bool)
        or not isinstance(enforcement["upstream_blocked"], list)
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("artifact_permitted"), bool)
            for item in upstream.values()
        )
        or not isinstance(document["gates"], dict)
        or not _is_sha256(document["gates_sha256"])
    ):
        raise ValueError("Gate report enforcement is malformed.")
    if not isinstance(document["report"], dict) or not isinstance(
        enforcement["blocking"], list
    ):
        raise ValueError("Gate report enforcement is malformed.")
    gates = GatesManifest.from_mapping(document["gates"], country=document["country"])
    if not isinstance(document["report"].get("outcomes"), list):
        raise ValueError("Gate report rows are malformed.")
    _check_rows(document["report"]["outcomes"])
    report = gate_phase_report_from_payload(document["report"], gates=gates)
    if report.phase != document["phase"] or document["report"] != _without_addresses(
        gate_phase_report_payload(report, gates=gates)
    ):
        raise ValueError("Gate report rows differ from their replayed outcomes.")
    evidence = document["evidence"]
    if (
        not isinstance(evidence, list)
        or any(not isinstance(alias, str) for alias in evidence)
        or evidence != sorted(set(evidence))
    ):
        raise ValueError("Gate report evidence aliases are malformed.")
    blocked_upstream_aliases = list(enforcement["upstream_blocked"])
    for outcome in report.outcomes:
        declared = outcome.entry.not_applicable
        if (outcome.status is GateStatus.NOT_APPLICABLE) != (declared is not None) or (
            declared is not None and outcome.reason != declared
        ):
            raise ValueError(
                f"Gate report row {outcome.entry.id!r} differs from its declared "
                "applicability."
            )
        if (outcome.status is GateStatus.UNREACHED) != bool(
            blocked_upstream_aliases
        ) and (declared is None):
            raise ValueError(
                f"Gate report row {outcome.entry.id!r} is reached exactly when no "
                "upstream phase blocked."
            )
    upstream_phases = []
    smoke = enforcement["synthetic_smoke"]
    candidate = enforcement["release_candidate"]
    for alias, item in upstream.items():
        if (
            alias not in evidence
            or set(item) != {"phase", "outcome", "artifact_permitted"}
            or item["outcome"] not in GATE_OUTCOMES
            or item["phase"] not in gates.phases
            or gates.phases.index(item["phase"]) >= gates.phases.index(report.phase)
            or (item["artifact_permitted"] and item["outcome"] == "unreached")
            or (item["artifact_permitted"] and item["outcome"] == "fail" and not smoke)
            or (
                item["artifact_permitted"]
                and item["outcome"] == "evidence_absent"
                and candidate
            )
            or (not item["artifact_permitted"] and item["outcome"] == "pass")
        ):
            raise ValueError("Gate report upstream entries are malformed.")
        upstream_phases.append(item["phase"])
    if len(set(upstream_phases)) != len(upstream_phases) or not (
        _required_upstream_phases(gates, report.phase) <= set(upstream_phases)
    ):
        raise ValueError(
            "Gate report does not name one upstream report per earlier phase "
            "that can block."
        )
    recomputed = phase_enforcement(
        report,
        release_candidate=enforcement["release_candidate"],
        synthetic_smoke=enforcement["synthetic_smoke"],
        upstream_blocked=bool(enforcement["upstream_blocked"]),
    )
    blocked_upstream = sorted(
        alias for alias, item in upstream.items() if not item["artifact_permitted"]
    )
    if (
        document["outcome"] != recomputed.outcome
        or document["outcome"] not in GATE_OUTCOMES
        or list(enforcement["blocking"]) != list(recomputed.blocking)
        or enforcement["artifact_permitted"] is not recomputed.artifact_permitted
        or list(enforcement["upstream_blocked"]) != blocked_upstream
    ):
        raise ValueError(
            "Gate report outcome or enforcement differs from its replayed outcomes."
        )
    return GateReport(
        country=gates.country,
        phase=report.phase,
        outcome=recomputed.outcome,
        artifact_permitted=recomputed.artifact_permitted,
        release_candidate=enforcement["release_candidate"],
        synthetic_smoke=enforcement["synthetic_smoke"],
        gates_sha256=document["gates_sha256"],
        gates=gates,
        report=report,
        document=MappingProxyType(document),
    )


def _plain(value: object) -> bool:
    """Data ``gate_battery._json_safe`` renders the same way in every run."""

    if value is None or isinstance(value, bool | int | float | str | np.generic):
        return True
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str | int | float | bool) and _plain(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return all(_plain(item) for item in value)
    if isinstance(value, set | frozenset):
        return all(
            item is None or isinstance(item, bool | int | float | str | np.generic)
            for item in value
        )
    return False


def _require_plain_details(report: GatePhaseReport) -> None:
    """Refuse gate details the report could not render deterministically.

    The battery keeps an unknown detail object as its ``repr``, which may
    carry a memory address or a hash-seed-dependent set order; a kernel that
    declares bitwise output must not write such bytes.
    """

    for outcome in report.outcomes:
        if outcome.result is not None and not _plain(dict(outcome.result.details)):
            raise ValueError(
                f"{_REF} gate {outcome.entry.id!r} reported details that are not "
                "plain data; a binding must report numbers, strings, booleans, "
                "and lists or string-keyed mappings of them."
            )


def _without_addresses(payload: dict[str, object]) -> dict[str, object]:
    """Blank memory addresses inside failure text and string details."""

    def clean(value: object) -> object:
        if isinstance(value, str):
            return _ADDRESS.sub("0x", value)
        if isinstance(value, dict):
            return {clean(key): clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return {
        **payload,
        "outcomes": [
            {
                **row,
                "failures": clean(row["failures"]),
                "details": clean(row["details"]),
            }
            for row in payload["outcomes"]
        ],
    }


def require_terminal_gate_reports(
    context: KernelContext, ref: str
) -> dict[str, GateReport]:
    """The gate reports a terminal node reads, checked as one battery's.

    Every ``microcosm.gates.phase-report`` input is decoded and replayed. At
    least one is required; all must carry the same country, manifest
    document, manifest resource hash and posture; and the last phase that can
    block must be among them, because its permission already folds in every
    earlier blocking phase. A phase that cannot block needs no report.
    """

    reports = {
        alias: decode_gate_report(value.payload)
        for alias, value in sorted(context.artifacts.items())
        if value.type == GATE_REPORT_TYPE
    }
    if not reports:
        raise ValueError(
            f"{ref} needs at least one {GATE_REPORT_TYPE.name} input: an export "
            "no gate has evaluated is refused."
        )
    identities = {
        (
            report.country,
            canonical_json(dict(report.document["gates"])),
            report.gates_sha256,
            report.release_candidate,
            report.synthetic_smoke,
        )
        for report in reports.values()
    }
    if len(identities) != 1:
        raise ValueError(
            f"{ref} gate reports come from different manifests or postures; a "
            "terminal node reads one battery's reports."
        )
    gates = next(iter(reports.values())).gates
    blocking = _blocking_phases(gates)
    if blocking and blocking[-1] not in {report.phase for report in reports.values()}:
        raise ValueError(
            f"{ref} needs the report of the last phase that can block, "
            f"{blocking[-1]!r}."
        )
    return reports


def _json_artifact(payload: bytes) -> object:
    return json.loads(payload)


#: Artifact type -> decoder producing the evidence object a binding receives.
#: A type absent here is offered to the battery as its verified bytes.
EVIDENCE_DECODERS: Mapping[ArtifactType, Callable[[bytes], object]] = MappingProxyType(
    {
        PROBLEM_TYPE: decode_problem,
        SOLUTION_TYPE: decode_solution,
        TARGET_SURFACE_TYPE: decode_target_surface,
        COMPARISON_TYPE: _json_artifact,
        DIAGNOSTICS_TYPE: _json_artifact,
        GATE_REPORT_TYPE: decode_gate_report,
        EXPORT_DESCRIPTOR_TYPE: _json_artifact,
        EXPORT_READBACK_TYPE: _json_artifact,
    }
)


def _decoded(value: ArtifactValue) -> object:
    decoder = EVIDENCE_DECODERS.get(value.type)
    return value.payload if decoder is None else decoder(value.payload)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


def _unreached_report(gates: GatesManifest, phase: str, reason: str) -> GatePhaseReport:
    outcomes = []
    for entry in gates.gates:
        if entry.phase != phase:
            continue
        if entry.not_applicable is not None:
            outcomes.append(
                GateOutcome(
                    entry=entry,
                    status=GateStatus.NOT_APPLICABLE,
                    reason=entry.not_applicable,
                )
            )
        else:
            outcomes.append(
                GateOutcome(entry=entry, status=GateStatus.UNREACHED, reason=reason)
            )
    return GatePhaseReport(phase=phase, outcomes=tuple(outcomes))


def _blocking_phases(gates: GatesManifest) -> list[str]:
    """Phases, in order, that declare an applicable release-blocking entry."""

    phases = {
        entry.phase
        for entry in gates.gates
        if entry.criticality == "release_blocking" and entry.not_applicable is None
    }
    return [phase for phase in gates.phases if phase in phases]


def _required_upstream_phases(gates: GatesManifest, phase: str) -> set[str]:
    """Earlier phases that can block: they must be wired as upstream."""

    earlier = set(gates.phases[: gates.phases.index(phase)])
    return {item for item in _blocking_phases(gates) if item in earlier}


class GateBatteryKernel(KernelBase):
    """Evaluate one declared ``gates.json`` phase with a given binding registry.

    Args:
        registry: Gate name to binding. Pass the country-neutral bindings
            merged with :data:`~microcosm.build.gate_battery.DEFAULT_REGISTRY`;
            ``DEFAULT_REGISTRY`` alone binds only four gates.
    """

    ref = _REF
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        role=KernelRole.GATE,
        dependencies=("numpy", "pandas"),
    )
    _required = frozenset(
        {"country", "gates", "gates_sha256", "phase", "release_candidate"}
    )
    _optional = frozenset({"entities", "weight_entity", "synthetic_smoke", "upstream"})

    def __init__(self, registry: Mapping[str, GateBinding]) -> None:
        if not isinstance(registry, Mapping):
            raise TypeError(f"{_REF} needs a Mapping of gate name to binding.")
        for name, binding in registry.items():
            if not isinstance(name, str) or getattr(binding, "name", None) != name:
                raise ValueError(
                    f"{_REF} binding registered as {name!r} must carry that name."
                )
        self.registry: Mapping[str, GateBinding] = MappingProxyType(dict(registry))
        # Refuse an undescribable binding now, not at the first key derivation,
        # and fix the identity the kernel's key will carry.
        self._identity = self._binding_identity()
        self.implementation_hash()

    def _binding_identity(self) -> tuple[dict[str, object], dict[str, str]]:
        descriptions: dict[str, object] = {}
        sources: dict[str, str] = {}
        for name in sorted(self.registry):
            description, digests = binding_identity.describe(
                self.registry[name], path=f"{_REF} binding {name!r}"
            )
            descriptions[name] = description
            sources.update(digests)
        return descriptions, dict(sorted(sources.items()))

    def _require_identity(self, cause: Exception | None = None) -> None:
        """Refuse once a binding or its source closure moved since construction.

        Raised during key derivation, this refuses the run before anything is
        written; raised from ``run``, the executor refuses the node instead of
        filing an outcome under the key of the unchanged registry (graph
        amendment 29). A closure file that can no longer be read or parsed is
        a change too. ``cause`` is the exception evaluation raised, if any.
        """

        try:
            moved = self._binding_identity() != self._identity
        except Exception as error:
            raise KernelIdentityChangedError(
                f"{_REF} binding registry, or a module in its source closure, "
                "changed after the kernel was built and can no longer be "
                f"described ({type(error).__name__}: {error}); its node keys "
                "would no longer describe it."
            ) from error
        if moved:
            raise KernelIdentityChangedError(
                f"{_REF} binding registry, or a module in its source closure, "
                "changed after the kernel was built; its node keys would no "
                "longer describe it."
            ) from cause

    def implementation_hash(self) -> str:
        self._require_identity()
        descriptions, sources = self._identity
        code = source_hash(
            sys.modules[__name__],
            graph_inputs,
            artifact_types,
            binding_identity,
            target_kernels,
            gate_battery_module,
            gates_module,
            country_spec_module,
            calibrate_artifacts_module,
            frame_bundle_module,
            frame_schema_module,
            dependencies=self.capabilities.dependencies,
        )
        return hashlib.sha256(
            canonical_json(
                {"code": code, "binding_sources": sources, "bindings": descriptions}
            )
        ).hexdigest()

    def _upstream(
        self,
        context: KernelContext,
        gates: GatesManifest,
        phase: str,
        *,
        document: Mapping[str, object],
        gates_sha256: str,
        release_candidate: bool,
        synthetic_smoke: bool,
    ) -> dict[str, GateReport]:
        aliases = context.params.get("upstream", ())
        if (
            not isinstance(aliases, tuple)
            or any(not isinstance(alias, str) for alias in aliases)
            or len(set(aliases)) != len(aliases)
        ):
            raise TypeError(
                f"{_REF} parameter 'upstream' must be a tuple of distinct aliases."
            )
        upstream: dict[str, GateReport] = {}
        for alias in aliases:
            value = context.artifacts.get(alias)
            if value is None or value.type != GATE_REPORT_TYPE:
                raise ValueError(
                    f"{_REF} upstream {alias!r} must be a declared "
                    f"{GATE_REPORT_TYPE.name} artifact input."
                )
            report = decode_gate_report(value.payload)
            # The identical manifest: the battery's policy hash leaves some
            # entry fields out (``population_fact_check``), so compare the
            # documents themselves, then replay against this node's manifest.
            if (
                canonical_json(dict(report.document["gates"]))
                != canonical_json(document)
                or report.gates_sha256 != gates_sha256
            ):
                raise ValueError(
                    f"{_REF} upstream {alias!r} was evaluated against a different "
                    "gate manifest."
                )
            gate_phase_report_from_payload(report.document["report"], gates=gates)
            if gates.phases.index(report.phase) >= gates.phases.index(phase):
                raise ValueError(
                    f"{_REF} upstream {alias!r} is phase {report.phase!r}, which "
                    f"does not precede {phase!r}."
                )
            if (report.release_candidate, report.synthetic_smoke) != (
                release_candidate,
                synthetic_smoke,
            ):
                raise ValueError(
                    f"{_REF} upstream {alias!r} was enforced under a different "
                    "posture (release_candidate, synthetic_smoke)."
                )
            upstream[alias] = report
        phases = [report.phase for report in upstream.values()]
        required = _required_upstream_phases(gates, phase)
        if len(set(phases)) != len(phases) or not required <= set(phases):
            raise ValueError(
                f"{_REF} phase {phase!r} must name one upstream report for each "
                f"earlier phase that can block {sorted(required)}, and "
                f"at most one per phase; got {sorted(phases)}."
            )
        return upstream

    def run(self, context: KernelContext) -> KernelResult:
        require_params(context, _REF, required=self._required, optional=self._optional)
        require_outputs(context, _REF)
        # The identity is checked on both sides of evaluation, and again when
        # evaluation raises: an outcome or failure produced while it moved is
        # not one the node key names, so it must never be filed as a verdict.
        self._require_identity()
        try:
            result = self._evaluate(context)
        except Exception as error:
            self._require_identity(cause=error)
            raise
        self._require_identity()
        return result

    def _evaluate(self, context: KernelContext) -> KernelResult:
        country = string_param(context, _REF, "country")
        phase = string_param(context, _REF, "phase")
        gates_sha256 = sha256_param(context, _REF, "gates_sha256")
        release_candidate = bool_param(context, _REF, "release_candidate")
        synthetic_smoke = optional_bool_param(context, _REF, "synthetic_smoke")
        if release_candidate and synthetic_smoke:
            raise ValueError(
                f"{_REF}: a synthetic smoke build cannot be a release candidate."
            )
        if ("entities" in context.params) != ("weight_entity" in context.params):
            raise ValueError(
                f"{_REF} parameters 'entities' and 'weight_entity' come together: "
                "they rebuild the phase frame."
            )
        document = canonical_document_param(context, _REF, "gates")
        gates = GatesManifest.from_mapping(document, country=country)
        if phase not in gates.phases:
            raise ValueError(
                f"{_REF} phase {phase!r} is not in the declared order "
                f"{list(gates.phases)}."
            )
        # A configuration error in any phase is the node's failure, not a
        # verdict: the executor records it as ``fail`` (charter F4).
        validate_gate_parameters(gates, self.registry)
        upstream = self._upstream(
            context,
            gates,
            phase,
            document=document,
            gates_sha256=gates_sha256,
            release_candidate=release_candidate,
            synthetic_smoke=synthetic_smoke,
        )
        upstream_blocked = sorted(
            alias for alias, report in upstream.items() if not report.artifact_permitted
        )
        if upstream_blocked:
            report = _unreached_report(
                gates,
                phase,
                "an earlier phase blocked: "
                + ", ".join(f"{a} ({upstream[a].phase})" for a in upstream_blocked),
            )
        else:
            frame = None
            if "entities" in context.params:
                frame = context_frame(
                    context,
                    _REF,
                    entities=entities_param(context, _REF, "entities"),
                    weight_entity=string_param(context, _REF, "weight_entity"),
                )
            evidence = EvidenceContext(
                frame=frame,
                artifacts={
                    alias: _decoded(value) for alias, value in context.artifacts.items()
                },
            )
            report = evaluate_phase(gates, phase, evidence, registry=self.registry)
        _require_plain_details(report)
        enforcement = phase_enforcement(
            report,
            release_candidate=release_candidate,
            synthetic_smoke=synthetic_smoke,
            upstream_blocked=bool(upstream_blocked),
        )
        payload = canonical_json(
            {
                "schema_version": 1,
                "kind": _REPORT_KIND,
                "country": gates.country,
                "phase": phase,
                "gates": document,
                "gates_sha256": gates_sha256,
                "outcome": enforcement.outcome,
                "report": _without_addresses(
                    gate_phase_report_payload(report, gates=gates)
                ),
                "enforcement": {
                    "release_candidate": release_candidate,
                    "synthetic_smoke": synthetic_smoke,
                    "blocking": list(enforcement.blocking),
                    "artifact_permitted": enforcement.artifact_permitted,
                    "upstream_blocked": upstream_blocked,
                },
                "upstream": {
                    alias: {
                        "phase": item.phase,
                        "outcome": item.outcome,
                        "artifact_permitted": item.artifact_permitted,
                    }
                    for alias, item in sorted(upstream.items())
                },
                "evidence": sorted(context.artifacts),
            }
        )
        # The artifact must replay to the decision it records.
        decode_gate_report(payload)
        return KernelResult(
            artifacts={"gate_report": payload},
            receipt={
                "outcome": enforcement.outcome,
                "evidence": {
                    "phase": phase,
                    "statuses": {o.entry.id: o.status.value for o in report.outcomes},
                    "blocking": list(enforcement.blocking),
                    "artifact_permitted": enforcement.artifact_permitted,
                    "upstream_blocked": upstream_blocked,
                    "gate_report_sha256": sha256_text(payload),
                },
            },
        )


def register_gate_kernel(
    registry: KernelRegistry, bindings: Mapping[str, GateBinding]
) -> GateBatteryKernel:
    """Register one ``gates.battery@1`` kernel bound to ``bindings``.

    Registering the same binding objects again returns the kernel already
    registered; any other bindings under the same ref are refused by the
    registry, as for any kernel.
    """

    if _REF in registry.refs():
        existing = registry.get(_REF)
        if (
            isinstance(existing, GateBatteryKernel)
            and existing.registry.keys() == bindings.keys()
            and all(existing.registry[name] is bindings[name] for name in bindings)
        ):
            return existing
    kernel = GateBatteryKernel(bindings)
    registry.register(kernel)
    return kernel
