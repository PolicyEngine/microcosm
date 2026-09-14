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
from . import battery_bindings
from .calibration_run import UK_SPINE_GATE_SCOPE, uk_scoped_gate_manifest
from .frs_relationships import CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS

SPINE_GATE_REPORT_TYPE = ArtifactType("microcosm.gate-phase-report", 1)


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
        release_candidate=bool(context.params["spine_gate_release_candidate"])
    )
    if blocking:
        raise ValueError(
            f"Stored {report.phase} spine gates block downstream execution: "
            + ", ".join(outcome.entry.id for outcome in blocking)
        )


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
    its ``enum_domain`` gate takes the declared domain as an artifact. One
    definition serves the spine-phase gate kernel, the full-build final gates
    and the tests that reproduce either report.
    """

    return {
        "rules_engine": engine,
        "ons_household_type_enum_domain": CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS,
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
) -> Graph:
    """Attach gates to exact assembled/transferred versions and stored evidence.

    The assembled gate reads the version frozen *by* the BRMA checkpoint,
    before subsequent wealth rewrites. Signing and output paths stay external.
    """
    from .graph import uk_spine_endpoint

    gates = uk_spine_gate_manifest(spec)
    if gates is None:
        return graph
    if not engine_identity:
        raise ValueError("Spine gates require a declared rules-engine identity.")
    if any(node.kernel == UKSpineGateKernel.ref for node in graph.nodes):
        raise ValueError("Spine gate nodes are already registered.")
    compiled = compile_graph(graph)
    endpoint = uk_spine_endpoint(graph)
    stages = endpoint.stage_names
    if "frs_brma" not in stages:
        raise ValueError("Spine gates require the assembled frs_brma boundary.")
    end = stages.index("frs_brma") + 1
    checkpoints = (
        (
            "assembled",
            compiled.versions["frs_brma"],
            graph.node("frs_brma.checkpoint").inputs,
            stages[:end],
        ),
        ("transferred", endpoint.population, endpoint.inputs, stages),
    )
    nodes = list(graph.nodes)
    for phase, population, inputs, phase_stages in checkpoints:
        if phase == "transferred" and end == len(stages):
            continue
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
                    "gate_manifest": json.dumps(
                        gate_battery._gates_manifest_payload(gates), sort_keys=True
                    ),
                    "engine_identity": engine_identity,
                    "release_candidate": release_candidate,
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
                },
            )
            if node.id == next_stage
            else node
            for node in nodes
        ]
    return replace(graph, nodes=tuple(nodes))


class UKSpineGateKernel(KernelBase):
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
        from microcosm.graph.canonical import canonical_json

        from ..country_spec import load_country_spec

        return hashlib.sha256(
            canonical_json(
                {
                    "code": source_hash(type(self), gate_battery, battery_bindings),
                    "country_resources": load_country_spec("uk").fingerprint,
                }
            )
        ).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        from microcosm.frame import Frame, MassChangeRecord

        from .graph_kernels import _minimal_frame

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
        previous = context.artifacts.get("previous_gate")
        if previous is not None:
            previous_report = gate_phase_report_from_payload(
                json.loads(previous.payload), gates=self.gates
            )
            if previous_report.blocking_outcomes(
                release_candidate=bool(context.params["release_candidate"])
            ):
                report = gate_battery.GatePhaseReport(
                    phase=str(context.params["phase"]),
                    outcomes=tuple(
                        gate_battery.GateOutcome(
                            entry=entry,
                            status=gate_battery.GateStatus.UNREACHED,
                            reason="The assembled spine gate blocked this phase.",
                        )
                        for entry in self.gates.gates
                        if entry.phase == context.params["phase"]
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
            str(context.params["phase"]),
            EvidenceContext(
                frame=frame,
                artifacts={**evidence, **uk_spine_gate_artifacts(self.engine)},
            ),
            registry=battery_bindings.UK_GATE_REGISTRY,
        )
        blocked = report.blocking_outcomes(
            release_candidate=bool(context.params["release_candidate"])
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
