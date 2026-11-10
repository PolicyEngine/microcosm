"""UK spine evidence and gate bindings for the shared graph/store contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    KernelRole,
    Node,
    Numeric,
    SeedSource,
    compile_graph,
    source_hash,
)

from .. import gate_battery
from ..country_spec import CountrySpec, GatesManifest
from ..gate_battery import (
    BlockingMode,
    EvidenceContext,
    GateBatteryRun,
    evaluate_phase,
    gate_phase_report_from_payload,
    gate_phase_report_payload,
)
from ..stage_evidence import (
    STAGE_EVIDENCE_TYPE,
    decode_stage_evidence,
    encode_stage_evidence,
)
from . import battery_bindings, stage_health
from .cgt_asset_type import CGT_ASSET_TYPE_DOMAIN
from .frs_relationships import CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS
from .gate_scopes import UK_SPINE_GATE_SCOPE, uk_scoped_gate_manifest
from .graph import UK_SPINE_ASSEMBLED_FINAL_STAGE

#: The spine checkpoint's build state the terminal gates read as the spine
#: frame: household weight kind, period and mass log (microcosm#1115).
SPINE_BUILD_STATE_TYPE = ArtifactType("microcosm.uk.spine-build-state", 1)

SPINE_GATE_REPORT_TYPE = ArtifactType("microcosm.gate-phase-report", 1)


class SpineGateBlockedError(ValueError):
    """A stored spine gate report refused the stage that depends on it.

    Raised by :func:`require_uk_spine_gate_admission` with the report it
    decoded from the gate node's persisted artifact (the executor loaded
    those bytes from the store by their verified key). The executor wraps
    the refusal in ``NodeRejected`` with this as its cause, so the driver
    can find the report on the failure path and materialise it for the
    operator before it re-raises: the graph holds the evidence, the
    sidecar is how the operator reads it.
    """

    def __init__(
        self,
        report: gate_battery.GatePhaseReport,
        blocking: Sequence[gate_battery.GateOutcome],
    ) -> None:
        self.report = report
        self.phase = report.phase
        self.blocking_gate_ids = tuple(outcome.entry.id for outcome in blocking)
        super().__init__(
            f"Stored {report.phase} spine gates block downstream execution: "
            + ", ".join(outcome.entry.id for outcome in blocking)
        )


def require_uk_spine_gate_admission(
    context: KernelContext, *, alias: str = "spine_gate"
) -> None:
    """Enforce a declared, already persisted spine phase before doing more work."""
    artifact = context.artifacts.get(alias)
    if artifact is None:
        if any(edge.name == alias for edge in context.node.artifact_inputs):
            raise ValueError("The declared spine gate admission artifact is absent.")
        return
    from ..country_spec import load_country_spec

    report = gate_phase_report_from_payload(
        json.loads(artifact.payload),
        gates=uk_spine_gate_manifest(load_country_spec("uk")),
    )
    if report.phase != context.params["spine_gate_phase"]:
        raise ValueError("Spine admission report belongs to a different phase.")
    blocking = report.blocking_outcomes(
        release_candidate=bool(context.params["spine_gate_release_candidate"]),
        synthetic_smoke=bool(context.params["spine_gate_synthetic_smoke"]),
    )
    if blocking:
        raise SpineGateBlockedError(report, blocking)


def blocked_spine_gate_report(
    error: BaseException,
) -> gate_battery.GatePhaseReport | None:
    """The stored phase report behind a graph failure, when a gate refused it.

    Walks the cause/context chain the executor builds around a kernel
    refusal; ``None`` when the run failed for any other reason.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, SpineGateBlockedError):
            return current.report
        current = current.__cause__ or current.__context__
    return None


def materialize_blocked_spine_gate_report(
    error: BaseException, *, battery: GateBatteryRun
) -> None:
    """Apply the write-before-block policy to a run ``run_graph`` refused.

    The success path restores every stored phase through
    :func:`materialize_spine_gate_reports`; a run the assembled gate blocks
    never returns a manifest, so the report reaches the operator from the
    refusal itself. Recording it through ``battery`` writes the sidecar with
    ``blocked_at_phase`` set and the later phase ``unreached``, then raises
    :class:`~microcosm.build.gate_battery.GateBatteryBlockedError` naming the
    report path, exactly as the in-process battery did. A failure that is not
    a gate refusal, or a battery that already blocked, leaves nothing to do.
    """
    report = blocked_spine_gate_report(error)
    if report is None or battery.blocked_at_phase is not None:
        return
    battery.record_phase(report)
    battery.enforce(report.phase, mode=BlockingMode.BLOCKS_ARTIFACT)


def uk_spine_gate_manifest(spec: CountrySpec) -> GatesManifest | None:
    if getattr(spec, "gates", None) is None:
        return None
    return uk_scoped_gate_manifest(
        UK_SPINE_GATE_SCOPE,
        phases=("assembled", "transferred"),
        policy_suffix="spine_build_scope",
        source=spec.gates,
    )


def uk_spine_gate_artifacts(engine: object) -> dict[str, object]:
    """Evidence artifacts every UK gate phase needs beside stage evidence.

    The rules engine, plus frame-only enum domains: #791 made
    ``ons_household_type`` a frame column rather than an engine variable, so
    its ``enum_domain`` gate takes the declared domain as an artifact, and
    #725 made ``capital_gains_asset_type`` a frame column whose domain the
    asset-type stage declares. One definition serves the spine-phase gate
    kernel, the full-build final gates and the tests that reproduce either
    report.
    """

    return {
        "rules_engine": engine,
        "ons_household_type_enum_domain": CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS,
        "capital_gains_asset_type_enum_domain": CGT_ASSET_TYPE_DOMAIN,
    }


def load_spine_stage_artifacts(
    manifest, store, *, stage_names: Sequence[str]
) -> dict[str, dict[str, object]]:
    """Read every requested stage from verified bytes, with no live transforms."""
    result = {}
    for stage in stage_names:
        node = "create_uk_frs" if stage == "frs_spine" else stage
        receipt = manifest.nodes[node]
        try:
            key = receipt.opaque_artifacts["stage_evidence"]
        except KeyError as error:
            raise ValueError(
                f"Spine stage {stage!r} has no stored evidence artifact."
            ) from error
        result[stage] = decode_stage_evidence(store.load_bytes(key), stage=stage)
    return result


def spine_sidecar_evidence(
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Project portable stage contracts onto the maintained sidecar schema."""
    return {
        "stage_evidence": {
            stage: document["evidence"]
            for stage, document in artifacts.items()
            if document["evidence"] is not None
        },
        "fit_weight_records": {
            stage: document["fit_weight_records"]
            for stage, document in artifacts.items()
            if "fit_weight_records" in document
        },
        "sampling": artifacts["frs_spine"]["sampling"],
    }


def add_uk_spine_gate_nodes(
    graph: Graph,
    *,
    spec: CountrySpec,
    engine_identity: str,
    release_candidate: bool = False,
    synthetic_smoke: bool = False,
) -> Graph:
    """Attach gates to exact assembled/transferred versions and stored evidence.

    The assembled gate reads the version frozen *by* the BRMA checkpoint,
    before subsequent wealth rewrites. Signing and output paths stay external.
    ``synthetic_smoke`` mirrors the driver's ``GateBatteryRun`` posture: a
    data-only fixture is not the population its facts describe, so
    population-fact gates never block such a run.
    """
    from .gate_phase_codes import UK_SPINE_GATE_PHASE_CODE
    from .graph import uk_spine_endpoint

    gates = uk_spine_gate_manifest(spec)
    if gates is None:
        return graph
    if not engine_identity:
        raise ValueError("Spine gates require a declared rules-engine identity.")
    if synthetic_smoke and release_candidate:
        raise ValueError("a synthetic smoke build cannot be a release candidate.")
    if any(node.kernel == UKSpineGateKernel.ref for node in graph.nodes):
        raise ValueError("Spine gate nodes are already registered.")
    compiled = compile_graph(graph)
    endpoint = uk_spine_endpoint(graph)
    stages = endpoint.stage_names
    if UK_SPINE_ASSEMBLED_FINAL_STAGE not in stages:
        raise ValueError(
            "Spine gates require the assembled "
            f"{UK_SPINE_ASSEMBLED_FINAL_STAGE} boundary."
        )
    end = stages.index(UK_SPINE_ASSEMBLED_FINAL_STAGE) + 1
    checkpoints = (
        (
            "assembled",
            compiled.versions[UK_SPINE_ASSEMBLED_FINAL_STAGE],
            graph.node(f"{UK_SPINE_ASSEMBLED_FINAL_STAGE}.checkpoint").inputs,
            stages[:end],
        ),
        ("transferred", endpoint.population, endpoint.inputs, stages),
    )
    nodes = list(graph.nodes)
    for phase, population, inputs, phase_stages in checkpoints:
        if phase == "transferred" and end == len(stages):
            continue
        code = UK_SPINE_GATE_PHASE_CODE.get(phase)
        nodes.append(
            Node(
                id=f"spine.gates.{phase}",
                kernel=UKSpineGateKernel.ref,
                inputs=inputs,
                population=population,
                artifact_inputs=tuple(
                    ArtifactInput(
                        stage,
                        "create_uk_frs" if stage == "frs_spine" else stage,
                        "stage_evidence",
                        STAGE_EVIDENCE_TYPE,
                    )
                    for stage in phase_stages
                )
                + (
                    (
                        ArtifactInput(
                            "previous_gate",
                            "spine.gates.assembled",
                            "gate_report",
                            SPINE_GATE_REPORT_TYPE,
                        ),
                    )
                    if phase == "transferred"
                    else ()
                ),
                artifact_outputs=(
                    ArtifactOutput("gate_report", SPINE_GATE_REPORT_TYPE),
                ),
                params={
                    "phase": phase,
                    "time_period": "2024",
                    "stage_names": phase_stages,
                    **({f"{phase}_code": code()} if code is not None else {}),
                    "gate_manifest": json.dumps(
                        gate_battery._gates_manifest_payload(gates), sort_keys=True
                    ),
                    "engine_identity": engine_identity,
                    "release_candidate": release_candidate,
                    "synthetic_smoke": synthetic_smoke,
                },
                description=f"Evaluate the {phase} spine battery against its exact population and stage evidence.",
            )
        )
    if end < len(stages):
        # GATE receipts preserve failures but do not themselves stop kernels.
        # Make the original assembled admission boundary an explicit dependency
        # of the first later model, after the report has reached the store.
        next_stage = stages[end]
        nodes = [
            replace(
                node,
                artifact_inputs=(
                    *node.artifact_inputs,
                    ArtifactInput(
                        "spine_gate",
                        "spine.gates.assembled",
                        "gate_report",
                        SPINE_GATE_REPORT_TYPE,
                    ),
                ),
                params={
                    **node.params,
                    "spine_gate_phase": "assembled",
                    "spine_gate_release_candidate": release_candidate,
                    "spine_gate_synthetic_smoke": synthetic_smoke,
                },
            )
            if node.id == next_stage
            else node
            for node in nodes
        ]
    return replace(graph, nodes=tuple(nodes))


class UKSpineGateKernel(KernelBase):
    """Evaluate one spine gate phase from stored stage evidence.

    Node parameters: ``phase``, ``stage_names`` (the evidence artifacts to
    decode), ``gate_manifest`` and ``engine_identity`` (which must match the
    bound manifest and engine), ``release_candidate`` and ``synthetic_smoke``.
    The last two are the battery's blocking posture, carried on the node so
    the persisted verdict binds the policy it was judged under:
    ``synthetic_smoke`` mirrors ``GateBatteryRun(synthetic_smoke=...)`` — on a
    data-only fixture the population-fact gates are recorded but never block.
    Each phase also carries ``<phase>_code``, the hash of the code only its
    own gates compute through
    (:data:`.gate_phase_codes.UK_SPINE_GATE_PHASE_CODE`), so the assembled
    gate's key, on which every later spine stage's chains, carries none of the
    transferred phase's (the re-review of microcosm#1420, E; microcosm#931
    ruling 6). An edit to a module only the transferred phase computes through
    still re-keys the assembled gate where a stage before it hashes the
    module, as for ``stochastic_assignment`` and ``take_up_contract``. The
    kernel refuses a stale or missing ``<phase>_code`` before it evaluates
    anything, as the full gates do
    (:func:`.gate_phase_codes.require_current_phase_code`; microcosm#931,
    ruled 9 Nov (late)), so a spine gate node whose ``<phase>_code`` is stale
    or missing is refused whenever it runs, and no spine gate evaluates its
    phase under a key that names other phase code. A record stored under the
    node's key is reused without running the kernel, so the check does not
    guard a cache hit.
    """

    ref = "uk.spine-gates@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        role=KernelRole.GATE,
    )

    def __init__(self, *, gates: GatesManifest, engine: object, engine_identity: str):
        self.gates = gates
        self.engine = engine
        self.engine_identity = engine_identity

    def implementation_hash(self) -> str:
        import microcosm.calibrate.geography_constants as geography_constants
        from microcosm.frame.adapters import policyengine_uk as uk_engine_adapter
        from microcosm.graph.canonical import canonical_json

        from .. import gates, stage_evidence
        from . import (
            council_tax_constants,
            graph_kernels,
            national_frame,
            receipt_counts,
        )
        from .solve_scope import uk_solve_fingerprint

        # The spine stages after a gate wait on its verdict, so its key is
        # part of theirs. No spine gate reads a gate-only register, so it keys
        # on the solve fingerprint and a sign-off between two release builds
        # leaves an in-graph spine a cache hit (microcosm#1301). The code is
        # what both phases compute through (microcosm#931 ruling 6): this
        # kernel's run, the frame it rebuilds (graph_kernels' _minimal_frame,
        # national_frame) and the evidence it decodes (stage_evidence); the
        # battery and the gate registry (gate_battery, battery_bindings,
        # gates); the stage-health checks (stage_health, the independent
        # review of microcosm#1420, finding 5), with the receipt bands of the
        # take-up checks both phases declare (receipt_counts); the enum
        # gate's nation partition (council_tax_constants, the region tier);
        # and the engine the kernel is built with, through which each phase's
        # enum gates resolve a domain no artifact supplies (the engine
        # adapter's PolicyEngineUKEngine.enum_domain). What only one phase's
        # gates compute through is that phase's own code, a node param that
        # run checks (UK_SPINE_GATE_PHASE_CODE, in gate_phase_codes, which this
        # hash leaves out): the energy level, for one, re-keys the transferred
        # gate and not the assembled gate every later spine stage waits on,
        # since no stage before that gate hashes energy_pricing (the
        # re-review, E). A phase-only module a stage before the assembled gate
        # hashes still re-keys it by chaining
        # (gate_phase_codes.uk_spine_transferred_only_code_hash).
        return hashlib.sha256(
            canonical_json(
                {
                    "code": source_hash(
                        type(self),
                        gate_battery,
                        battery_bindings,
                        stage_health,
                        gates,
                        stage_evidence,
                        graph_kernels,
                        national_frame,
                        receipt_counts,
                        council_tax_constants,
                        geography_constants,
                        uk_engine_adapter,
                    ),
                    "country_resources": uk_solve_fingerprint(),
                }
            )
        ).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        from microcosm.frame import Frame, MassChangeRecord

        from .gate_phase_codes import require_current_phase_code
        from .graph_kernels import _minimal_frame

        phase = str(context.params["phase"])
        expected = json.dumps(
            gate_battery._gates_manifest_payload(self.gates), sort_keys=True
        )
        if (
            context.params["gate_manifest"] != expected
            or context.params["engine_identity"] != self.engine_identity
        ):
            raise ValueError(
                "Spine gate binding differs from its declared manifest or engine."
            )
        # The phase's own code keys the node through its parameter, not this
        # kernel's hash, so a node whose parameter is stale or missing is
        # refused before anything is read or evaluated (microcosm#931, ruled
        # 9 Nov (late)), as the full gates refuse theirs.
        require_current_phase_code(context.params, phase, battery="spine")
        release_candidate = bool(context.params["release_candidate"])
        synthetic_smoke = bool(context.params["synthetic_smoke"])
        previous = context.artifacts.get("previous_gate")
        if previous is not None:
            previous_report = gate_phase_report_from_payload(
                json.loads(previous.payload), gates=self.gates
            )
            if previous_report.blocking_outcomes(
                release_candidate=release_candidate,
                synthetic_smoke=synthetic_smoke,
            ):
                report = gate_battery.GatePhaseReport(
                    phase=phase,
                    outcomes=tuple(
                        gate_battery.GateOutcome(
                            entry=entry,
                            status=gate_battery.GateStatus.UNREACHED,
                            reason="The assembled spine gate blocked this phase.",
                        )
                        for entry in self.gates.gates
                        if entry.phase == phase
                    ),
                )
                return KernelResult(
                    artifacts={
                        "gate_report": encode_stage_evidence(
                            gate_phase_report_payload(report, gates=self.gates)
                        )
                    },
                    receipt={"outcome": "unreached", "phase": report.phase},
                )
        artifacts = {
            stage: decode_stage_evidence(context.artifacts[stage].payload, stage=stage)
            for stage in context.params["stage_names"]
        }
        evidence = spine_sidecar_evidence(artifacts)
        minimal = _minimal_frame(context)
        root_context = artifacts["frs_spine"]["frame_context"]
        mass_records = [
            MassChangeRecord(**row)
            for document in artifacts.values()
            for row in document["frame_mass_log_append"]
        ]
        frame = Frame(
            {entity: minimal.table(entity) for entity in minimal.entities},
            minimal.schema,
            {
                entity: minimal.weights_for(entity)
                for entity in minimal.weighted_entities
            },
            minimal.strata,
            mass_log=tuple(mass_records),
            metadata=root_context["metadata"],
        )
        report = evaluate_phase(
            self.gates,
            phase,
            EvidenceContext(
                frame=frame,
                artifacts={**evidence, **uk_spine_gate_artifacts(self.engine)},
            ),
            registry=battery_bindings.UK_GATE_REGISTRY,
        )
        blocked = report.blocking_outcomes(
            release_candidate=release_candidate,
            synthetic_smoke=synthetic_smoke,
        )
        return KernelResult(
            artifacts={
                "gate_report": encode_stage_evidence(
                    gate_phase_report_payload(report, gates=self.gates)
                )
            },
            receipt={"outcome": "fail" if blocked else "pass", "phase": report.phase},
        )


def register_spine_gate_kernel(
    registry: KernelRegistry,
    *,
    spec: CountrySpec,
    engine: object,
    engine_identity: str,
) -> None:
    gates = uk_spine_gate_manifest(spec)
    if gates is not None:
        registry.register(
            UKSpineGateKernel(
                gates=gates, engine=engine, engine_identity=engine_identity
            )
        )


def materialize_spine_gate_reports(
    manifest, store, *, battery: GateBatteryRun, gates: GatesManifest
) -> None:
    """Restore cached verdicts, then use the original write-before-block policy."""
    for phase in gates.phases:
        node = f"spine.gates.{phase}"
        if node not in manifest.nodes:
            continue
        key = manifest.nodes[node].opaque_artifacts.get("gate_report")
        if key is None:
            raise ValueError(f"Spine gate {phase!r} produced no persisted report.")
        report = gate_phase_report_from_payload(
            json.loads(store.load_bytes(key)), gates=gates
        )
        battery.record_phase(report)
        battery.enforce(phase, mode=BlockingMode.BLOCKS_ARTIFACT)
