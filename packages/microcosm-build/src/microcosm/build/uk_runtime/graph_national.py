"""The UK national release role on the shared executable graph.

The certified national line (microcosm#823) used to run through the
calibration seam library, an in-process pipeline the graph driver dispatched
to before preparing any graph. This module composes the same computation as
graph nodes over the bound spine checkpoint, so the national and the dense
line share one executor, one content store and one evidence discipline:

* ``uk.full.national_targets`` compiles the national register from the
  pinned Chronicle artifact (measure exclusions, band edges, the optional
  frozen scoring register); no ladder and no local surface.
* ``uk.full.national_problem`` resolves the engine measures on the bound
  frame, materialises the register and encodes one ordered problem whose
  bindings carry the national doctrine's loss weights, mass reason and
  bounds.
* ``uk.full.dense`` / ``uk.full.calibrated`` are the existing calibration
  nodes; the dense node's admission is the bound spine's provenance
  artifact rather than a source preflight battery, because the national
  line runs no pre-solve battery (its source and reference gates belong to
  the release-cut certification producer).
* ``uk.full.gates.calibrated`` evaluates the calibration-seam gate scope on
  the calibrated population and stores the calibration manifest block and
  the per-target diagnostic rows beside the phase report.

The driver (:mod:`.full_build_cli`) materialises the seam-shaped files from
those artifacts: the diagnostics through the seam writer, the battery
replayed from the stored phase report and signed, the H5, the build record,
the registries and the manifest (:mod:`.national_role`).
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import (
    EvidenceContext,
    _gates_manifest_payload,
    _json_safe,
    evaluate_phase,
    gate_phase_report_payload,
)
from microcosm.build.target_materialization import assert_calibration_input_finite
from microcosm.calibrate import TargetRegistry, build_constraint_matrix
from microcosm.calibrate.artifacts import (
    PROBLEM_TYPE,
    RESULT_TYPE,
    SOLUTION_TYPE,
    decode_calibration_result,
    decode_problem,
    decode_solution,
    encode_problem,
)
from microcosm.frame import Frame, WeightKind
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
    SourceRef,
    compile_graph,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from . import (
    calibration_run,
    full_measure,
    full_targets,
    ledger_targets,
    national_calibration,
    national_doctrine,
)
from .battery_bindings import UK_GATE_REGISTRY
from .calibration_run import (
    UK_CALIBRATION_GATE_SCOPE,
    UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS,
    uk_scoped_gate_manifest,
)
from .cgt_projection import UK_CGT_PROJECTION_ARTIFACT_KEY
from .content_identity import uk_frame_content_identity
from .diagnostics import uk_target_geography_levels
from .full_targets import CHRONICLE_SOURCE_CODEC, load_uk_national_target_inputs
from .graph_build import SPINE_PROVENANCE_TYPE
from .graph_calibration import (
    UKCalibrationNodes,
    UKGraphCalibrationConfig,
    uk_calibration_nodes,
)
from .graph_population import context_frame, population_columns, population_slices
from .graph_targets import (
    register_uk_source_codecs,
    registry_from_payload,
    registry_payload,
    target_geography,
)
from .local_rowwise import UKRowwiseNationalRows
from .measure_simulation import UKMeasureResolver
from .national_calibration import (
    CalibrationFrameAdapter,
    national_calibration_mass_reason,
)
from .national_doctrine import UKNationalSolveDoctrine, uk_national_target_loss_weights
from .national_frame import load_uk_national_frame
from .rowwise_posture import UK_ROWWISE_NATIONAL_POSTURE, UKRowwisePosture

NATIONAL_TARGET_TYPE = ArtifactType("microcosm.uk.national-target-registry", 1)
NATIONAL_GATE_REPORT_TYPE = ArtifactType("microcosm.uk.national-gate-report", 1)
NATIONAL_EVIDENCE_TYPE = ArtifactType("microcosm.uk.national-calibration-evidence", 1)
NATIONAL_READBACK_TYPE = ArtifactType("microcosm.uk.national-export-readback", 1)
NATIONAL_MATERIALIZATION = "uk_national_measure"

#: The doctrine fields every national evidence block records, in the seam's
#: order (``calibration_run._doctrine_payload``); the problem node binds the
#: solve-shaping ones so the ordered problem carries its own doctrine.
_DOCTRINE_FIELDS = (
    "epochs",
    "learning_rate",
    "max_weight_ratio",
    "seed",
    "target_loss_cap",
    "scale_rule",
    "target_weight_rule",
    "mass_rule",
    "l0_lambda",
)


def national_doctrine_payload(doctrine: UKNationalSolveDoctrine) -> dict[str, object]:
    """The seam's nine-field doctrine block."""

    return {field: getattr(doctrine, field) for field in _DOCTRINE_FIELDS}


def national_doctrine_from_payload(
    payload: Mapping[str, object],
) -> UKNationalSolveDoctrine:
    return UKNationalSolveDoctrine(
        **{field: payload[field] for field in _DOCTRINE_FIELDS}
    )


# ---------------------------------------------------------------------------
# Configuration and composition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UKNationalBuildConfig:
    """The national posture's declared build: one doctrine, one register."""

    calibration_year: int
    time_period: str
    source_year: int
    doctrine: UKNationalSolveDoctrine
    doctrine_overrides: Mapping[str, Mapping[str, object]]
    allow_unpinned_feed: bool = False

    def __post_init__(self) -> None:
        if type(self.calibration_year) is not int or self.calibration_year <= 0:
            raise ValueError("calibration_year must be a positive integer.")
        if not str(self.time_period).strip():
            raise ValueError("time_period must be non-empty.")
        if type(self.source_year) is not int or self.source_year <= 0:
            raise ValueError("source_year must be a positive integer.")
        if not isinstance(self.doctrine, UKNationalSolveDoctrine):
            raise TypeError("doctrine must be a UKNationalSolveDoctrine.")
        object.__setattr__(
            self,
            "doctrine_overrides",
            {str(k): dict(v) for k, v in dict(self.doctrine_overrides).items()},
        )

    @property
    def calibration(self) -> UKGraphCalibrationConfig:
        """The solve settings the shared calibration nodes bind."""

        return UKGraphCalibrationConfig(
            epochs=self.doctrine.epochs,
            learning_rate=self.doctrine.learning_rate,
            seed=self.doctrine.seed,
            target_weight_rule=self.doctrine.target_weight_rule,
        )

    def payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "doctrine": national_doctrine_payload(self.doctrine),
            "doctrine_overrides": dict(self.doctrine_overrides),
        }


@dataclass(frozen=True)
class UKNationalGraph:
    graph: Graph
    calibration: UKCalibrationNodes
    config: UKNationalBuildConfig
    posture: UKRowwisePosture

    @property
    def population(self) -> str:
        return self.calibration.population

    def operation_inventory(self) -> dict:
        compiled = compile_graph(self.graph)
        return {
            "schema": "microcosm.uk.national-build-operations.v1",
            "release_role": self.posture.role,
            "configuration": self.config.payload(),
            "nodes": [
                {
                    "id": node_id,
                    "kernel": self.graph.node(node_id).kernel,
                    "description": self.graph.node(node_id).description,
                    "population": compiled.versions[node_id],
                    "dependencies": list(compiled.predecessors[node_id]),
                    "artifacts": [
                        o.name for o in self.graph.node(node_id).artifact_outputs
                    ],
                }
                for node_id in compiled.order
            ],
        }


NATIONAL_TARGETS_NODE = "uk.full.national_targets"
NATIONAL_PROBLEM_NODE = "uk.full.national_problem"
NATIONAL_GATES_NODE = "uk.full.gates.calibrated"
NATIONAL_READBACK_NODE = "uk.full.national.readback"


def uk_national_gate_manifest(posture: UKRowwisePosture = UK_ROWWISE_NATIONAL_POSTURE):
    """The calibration-seam scope, filtered from the declared spec."""

    return uk_scoped_gate_manifest(
        tuple(posture.gate_scope),
        phases=("terminal",),
        policy_suffix=posture.gate_policy_suffix,
    )


def uk_national_graph(
    config: UKNationalBuildConfig,
    *,
    spine: Graph,
    spine_population: str,
    optional_target_sources: tuple[str, ...] = (),
    review_date: str | None = None,
    posture: UKRowwisePosture = UK_ROWWISE_NATIONAL_POSTURE,
    ledger_pins: Mapping[str, str | None] | None = None,
) -> UKNationalGraph:
    """Append the national line to a bound spine checkpoint graph.

    ``spine`` must end in a node that emits the bound spine's provenance
    artifact (:class:`~.graph_build.UKBoundSpineKernel`): the national line
    solves the checkpoint as bound, never a sampled or cloned pool.
    """

    if posture.role != "national":
        raise ValueError("uk_national_graph composes the national posture only.")
    if config.doctrine.target_weight_rule != posture.doctrine.target_weight_rule:
        # A receipted override changes the rule; the posture still names the
        # doctrine the run deviated from, which the manifest records.
        pass
    holder = spine.node(spine_population)
    if not any(
        output.name == "spine_provenance" and output.type == SPINE_PROVENANCE_TYPE
        for output in holder.artifact_outputs
    ):
        raise ValueError(
            "The national line requires a bound spine checkpoint population "
            "that emits its spine_provenance artifact."
        )
    review_date = date.today().isoformat() if review_date is None else review_date
    date.fromisoformat(review_date)
    cells = population_columns(spine, spine_population)
    slices = population_slices(cells)
    contract_identity = load_country_spec("uk").fingerprint
    compile_sources = ("uk_ledger_facts", *optional_target_sources)
    registry_input = ArtifactInput(
        "registry", NATIONAL_TARGETS_NODE, "registry", NATIONAL_TARGET_TYPE
    )
    provenance_input = ArtifactInput(
        "spine_provenance", spine_population, "spine_provenance", SPINE_PROVENANCE_TYPE
    )
    doctrine = config.doctrine
    nodes: list[Node] = [
        Node(
            NATIONAL_TARGETS_NODE,
            UKNationalTargetKernel.ref,
            population=spine_population,
            sources=compile_sources,
            params={
                "calibration_year": int(config.calibration_year),
                "review_date": review_date,
                "country_contract_sha256": contract_identity,
                "allow_unpinned_feed": bool(config.allow_unpinned_feed),
                # The request's Ledger pins: the artifact must measure them,
                # as the seam's loader required (the source identity binds
                # the bytes; the pins bind what the operator asked for).
                "ledger_facts_sha256": (ledger_pins or {}).get("facts_sha256"),
                "ledger_manifest_sha256": (ledger_pins or {}).get("manifest_sha256"),
            },
            artifact_outputs=(ArtifactOutput("registry", NATIONAL_TARGET_TYPE),),
            description="Compile the pinned national register, its exclusions and band edges.",
        ),
        Node(
            NATIONAL_PROBLEM_NODE,
            UKNationalProblemKernel.ref,
            population=spine_population,
            inputs=slices,
            params={
                "time_period": str(config.time_period),
                "target_weight_rule": doctrine.target_weight_rule,
                "target_loss_cap": float(doctrine.target_loss_cap),
                "max_weight_ratio": doctrine.max_weight_ratio,
                "l0_lambda": float(doctrine.l0_lambda),
                "mass_rule": doctrine.mass_rule,
                "scale_rule": doctrine.scale_rule,
            },
            artifact_inputs=(registry_input,),
            artifact_outputs=(ArtifactOutput("problem", PROBLEM_TYPE),),
            description="Resolve engine measures on the bound spine and build the one ordered national problem.",
        ),
    ]
    calibration = uk_calibration_nodes(
        base=spine_population,
        columns=cells,
        problem_producer=NATIONAL_PROBLEM_NODE,
        config=config.calibration,
    )
    for node in calibration.nodes:
        if node.id == calibration.dense_producer:
            node = replace(
                node,
                params={**node.params, "admission": "bound_spine"},
                artifact_inputs=(*node.artifact_inputs, provenance_input),
                description="Solve the national doctrine on the bound spine checkpoint.",
            )
        nodes.append(node)
    gates = uk_national_gate_manifest(posture)
    nodes.append(
        Node(
            NATIONAL_GATES_NODE,
            UKNationalGateKernel.ref,
            population=calibration.population,
            inputs=population_slices(population_columns(spine, spine_population)),
            params={
                "time_period": str(config.time_period),
                "review_date": review_date,
                "gate_posture": posture.gate_posture,
                "gate_policy_suffix": posture.gate_policy_suffix,
                "gate_manifest": canonical_json(
                    _gates_manifest_payload(gates)
                ).decode(),
                "doctrine": canonical_json(
                    national_doctrine_payload(doctrine)
                ).decode(),
            },
            artifact_inputs=(
                registry_input,
                ArtifactInput(
                    "problem", NATIONAL_PROBLEM_NODE, "problem", PROBLEM_TYPE
                ),
                ArtifactInput(
                    "result", calibration.result_producer, "result", RESULT_TYPE
                ),
                ArtifactInput(
                    "solution", calibration.solution_producer, "solution", SOLUTION_TYPE
                ),
            ),
            artifact_outputs=(
                ArtifactOutput("gate_report", NATIONAL_GATE_REPORT_TYPE),
                ArtifactOutput("calibration_evidence", NATIONAL_EVIDENCE_TYPE),
            ),
            description="Evaluate the calibration-seam gate scope on the calibrated national population.",
        )
    )
    existing = {source.name for source in spine.sources}
    sources = tuple(
        SourceRef(
            name,
            CHRONICLE_SOURCE_CODEC if name == "uk_ledger_facts" else "raw-bytes-v1",
        )
        for name in compile_sources
        if name not in existing
    )
    graph = replace(
        spine, nodes=(*spine.nodes, *nodes), sources=(*spine.sources, *sources)
    )
    compile_graph(graph)
    return UKNationalGraph(graph, calibration, config, posture)


def add_uk_national_readback(
    graph: Graph, *, population: str, dataset_source: str = "exported_dataset"
) -> Graph:
    """Continue the graph over the written H5: a readback gate on its content."""

    node = Node(
        NATIONAL_READBACK_NODE,
        UKNationalReadbackKernel.ref,
        population=population,
        inputs=population_slices(population_columns(graph, population)),
        sources=(dataset_source,),
        params={"dataset_source": dataset_source},
        artifact_outputs=(ArtifactOutput("readback", NATIONAL_READBACK_TYPE),),
        description="Read the written national H5 back and compare its content identity with the calibrated population.",
    )
    return replace(
        graph,
        nodes=(*graph.nodes, node),
        sources=(
            *graph.sources,
            SourceRef(
                dataset_source,
                "uk-spine-h5-v1",
                "Materialized UK national H5; graph-owned readback.",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


class _NationalKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, dependencies=("policyengine-uk",)
    )

    def implementation_hash(self) -> str:
        implementation = source_hash(
            type(self),
            sys.modules[__name__],
            full_measure,
            full_targets,
            ledger_targets,
            national_calibration,
            national_doctrine,
        )
        return hashlib.sha256(
            canonical_json(
                {
                    "implementation": implementation,
                    "country_spec": load_country_spec("uk").fingerprint,
                }
            )
        ).hexdigest()


class UKNationalTargetKernel(_NationalKernel):
    ref = "uk.full.national_targets@1"

    def run(self, context: KernelContext) -> KernelResult:
        inputs = load_uk_national_target_inputs(
            context.sources["uk_ledger_facts"],
            measure_exclusions=context.sources.get("uk_measure_exclusions"),
            register_json=context.sources.get("uk_frozen_register"),
            calibration_year=int(context.params["calibration_year"]),
            exclusions_evaluated_on=date.fromisoformat(
                str(context.params["review_date"])
            ),
            allow_unpinned_feed=bool(context.params["allow_unpinned_feed"]),
            expected_facts_sha256=context.params.get("ledger_facts_sha256"),
            expected_manifest_sha256=context.params.get("ledger_manifest_sha256"),
        )
        payload = {
            "schema_version": 1,
            "kind": "uk_national_target_registry",
            "calibration_year": int(inputs["calibration_year"]),
            "national_registry": registry_payload(inputs["national_registry"]),
            "band_edge_registry": registry_payload(inputs["band_edge_registry"]),
            "measure_exclusions": inputs["measure_exclusions"],
            "register_completeness": inputs["register_completeness"],
            "ledger_provenance": inputs["ledger_provenance"],
            "chronicle_feed_pin": inputs["chronicle_feed_pin"],
            "chronicle_provenance": inputs["chronicle_provenance"],
            "allow_unpinned_feed": bool(context.params["allow_unpinned_feed"]),
        }
        return KernelResult(artifacts={"registry": canonical_json(payload)})


def _registry_artifact(context: KernelContext) -> dict[str, Any]:
    payload = json.loads(context.artifacts["registry"].payload)
    if (
        payload.get("kind") != "uk_national_target_registry"
        or payload.get("schema_version") != 1
    ):
        raise ValueError("Unsupported UK national target registry artifact.")
    return payload


def materialize_uk_national_rows(
    frame: Frame,
    registry: TargetRegistry,
    *,
    period: int,
    band_edge_registry: TargetRegistry,
    resolver_factory=UKMeasureResolver,
    scratch_dir: Path | None = None,
) -> tuple[Frame, Any, UKRowwiseNationalRows, dict[str, Any]]:
    """Resolve, materialise and return the national rows over a prepared frame.

    With a resolver factory the engine route of
    :func:`~.full_measure.resolve_uk_full_measures` runs (one block, no local
    grains). Without one (``None``) the register materialises from the
    frame's own columns, the seam's ``measure_resolver=None`` route: the
    hermetic tests' path, and a data-only environment's.
    """

    if resolver_factory is None:
        adapter = CalibrationFrameAdapter(frame)
        materialized = ledger_targets.materialize_uk_ledger_targets(
            adapter, registry, period=period, band_edge_registry=band_edge_registry
        )
        if materialized.skipped:
            raise RuntimeError(
                "UK national calibration could not materialize every activated "
                f"target reference: skipped={[s.__dict__ for s in materialized.skipped]}."
            )
        rows = UKRowwiseNationalRows(
            targets=registry.to_target_set(),
            registry=registry,
            families=tuple(sorted({spec.family for spec in registry.specs})),
        )
        evidence = {
            "mode": "frame_only",
            "national_inputs": 0,
            "target_materialization": materialized.report(),
        }
        return adapter.prepared_frame(), adapter.restore, rows, evidence
    if scratch_dir is None:
        raise ValueError("engine measure resolution needs a scratch directory.")
    prepared, restore, rows, _metrics, evidence = full_measure.resolve_uk_full_measures(
        frame,
        registry,
        period=period,
        scratch_dir=scratch_dir,
        band_edge_registry=band_edge_registry,
        resolver_factory=resolver_factory,
        blocks=1,
        local_grains=(),
    )
    return prepared, restore, rows, dict(evidence)


class UKNationalProblemKernel(_NationalKernel):
    """One ordered national problem with the doctrine bound into it.

    ``resolver_factory`` is instance state like the gate kernels' coverage
    engine: the engine identity the run stands on is provenance recorded by
    the driver, and the dependency declaration on the capabilities binds the
    installed engine version into the node key.
    """

    ref = "uk.full.national_problem@1"

    def __init__(self, *, resolver_factory=UKMeasureResolver) -> None:
        self.resolver_factory = resolver_factory

    def run(self, context: KernelContext) -> KernelResult:
        frame = context_frame(context)
        registry_document = _registry_artifact(context)
        with tempfile.TemporaryDirectory(
            prefix="microcosm-uk-national-measures-"
        ) as scratch:
            payload = encode_uk_national_problem(
                frame,
                national_registry=registry_from_payload(
                    registry_document["national_registry"]
                ),
                band_edge_registry=registry_from_payload(
                    registry_document["band_edge_registry"]
                ),
                period=int(registry_document["calibration_year"]),
                measure_exclusions=registry_document["measure_exclusions"],
                doctrine={
                    "target_weight_rule": str(context.params["target_weight_rule"]),
                    "target_loss_cap": float(context.params["target_loss_cap"]),
                    "max_weight_ratio": context.params["max_weight_ratio"],
                    "l0_lambda": float(context.params["l0_lambda"]),
                    "mass_rule": str(context.params["mass_rule"]),
                    "scale_rule": str(context.params["scale_rule"]),
                },
                resolver_factory=self.resolver_factory,
                scratch_dir=Path(scratch),
            )
        return KernelResult(artifacts={"problem": payload})


def encode_uk_national_problem(
    frame: Frame,
    *,
    national_registry: TargetRegistry,
    band_edge_registry: TargetRegistry,
    period: int,
    doctrine: Mapping[str, object],
    measure_exclusions: Mapping[str, Mapping[str, str]] | None = None,
    resolver_factory=UKMeasureResolver,
    scratch_dir: Path | None = None,
) -> bytes:
    """One ordered national problem, the doctrine bound into it.

    The seam stage's route up to the solve: refuse non-finite inputs,
    resolve the engine measures (or materialise from the frame's own
    columns with ``resolver_factory=None``), materialise the register on
    the adapter, compile the constraint matrix row for row, restore the
    pristine tables, and encode the problem with the national doctrine's
    ``family_equal`` loss weights, mass reason and bounds in its bindings,
    which the shared dense solve node reads.
    """

    assert_calibration_input_finite(frame)
    if not len(national_registry.specs):
        raise ValueError("The national register selects no target.")
    prepared, restore, rows, evidence = materialize_uk_national_rows(
        frame,
        national_registry,
        period=period,
        band_edge_registry=band_edge_registry,
        resolver_factory=resolver_factory,
        scratch_dir=scratch_dir,
    )
    problem = build_constraint_matrix(prepared, rows.targets, "household")
    if problem.skipped:
        failures = "; ".join(
            f"{item.target.key}: {item.reason}" for item in problem.skipped
        )
        raise ValueError("National constraints failed to compile: " + failures)
    clean = restore(prepared)
    for entity in frame.entities:
        pd.testing.assert_frame_equal(clean.table(entity), frame.table(entity))
    if len(problem.targets) != len(national_registry.specs):
        raise ValueError(
            "UK national calibration matrix did not contain every activated "
            f"reference: declared={len(national_registry.specs)}, "
            f"rows={len(problem.targets)}."
        )
    families = [spec.family for spec in national_registry.specs]
    rule = str(doctrine["target_weight_rule"])
    target_loss_weights = uk_national_target_loss_weights(families, rule=rule)
    mass_reason = national_calibration_mass_reason(families)
    ids = frame.table("household")["household_id"].tolist()
    levels = national_target_geography_levels(national_registry)
    target_metadata = [
        {
            **spec.metadata,
            "family": spec.family,
            "source": spec.source,
            "geography_level": levels[spec.to_target().row_name],
            "materialization": NATIONAL_MATERIALIZATION,
        }
        for spec in national_registry.specs
    ]
    return encode_problem(
        problem,
        entity_ids=ids,
        target_metadata=target_metadata,
        bindings={
            "release_role": "national",
            "mass_reason": mass_reason,
            "mass_rule": str(doctrine["mass_rule"]),
            "max_weight_ratio": doctrine["max_weight_ratio"],
            "target_loss_cap": float(doctrine["target_loss_cap"]),
            "target_loss_weights": (
                np.ones(problem.n_targets, dtype=np.float64)
                if target_loss_weights is None
                else target_loss_weights
            ).tolist(),
            "target_weight_rule": rule,
            "scale_rule": str(doctrine["scale_rule"]),
            "l0_lambda": float(doctrine["l0_lambda"]),
            "bound_families": [
                f"national/{family}" for family in sorted(set(families))
            ],
            "measure_resolution": _json_safe(evidence),
            "measure_exclusions": dict(measure_exclusions or {}),
            "register_sha256": national_registry.version,
            "band_edge_register_sha256": band_edge_registry.version,
            "calibration_year": int(period),
            "activated_reference_count": len(national_registry.specs),
            "resolved_reference_count": len(national_registry.specs),
        },
    )


def _axis(frame: Frame) -> list:
    return frame.table("household")["household_id"].tolist()


def national_target_geography_levels(registry: TargetRegistry) -> dict[str, str]:
    """Each national row's geography level, by row name.

    A compiled national spec that declares its level (``geography_level`` or
    ``ledger_geography_level``, as the Chronicle compiler writes) resolves
    through the shared rule; one that declares none, as the fixture registers
    do, resolves through the population-target contract, the seam
    diagnostics' rule (:func:`~.diagnostics.uk_target_geography_levels`).
    """

    declared = {
        spec.to_target().row_name: target_geography(spec)
        for spec in registry.specs
        if spec.metadata.get("geography_level") is not None
        or spec.metadata.get("ledger_geography_level") is not None
    }
    if len(declared) == len(registry.specs):
        return declared
    contract = uk_target_geography_levels(
        TargetRegistry(
            [
                spec
                for spec in registry.specs
                if spec.to_target().row_name not in declared
            ],
            country="uk",
        )
    )
    return {**contract, **declared}


def national_calibration_manifest(
    result,
    frame: Frame,
    *,
    problem,
    doctrine: Mapping[str, object],
) -> dict[str, object]:
    """The seam's ``calibration`` evidence block, from the graph's artifacts.

    ``result`` is the decoded dense result rebound to the calibrated
    population ``frame``; ``problem`` its ordered problem, whose bindings
    carry the materialisation evidence and the activation counts.
    """

    from microcosm.calibrate import effective_sample_size

    bindings = problem.bindings
    weights = frame.weights_for("household")
    if weights.kind is not WeightKind.CALIBRATED:
        raise RuntimeError(
            "UK national calibration returned household weights whose kind is "
            f"{weights.kind.value!r}, not 'calibrated'."
        )
    if not frame.mass_log:
        raise RuntimeError("UK national calibration must append one mass record.")
    record = frame.mass_log[-1]
    if record.entity != "household" or "calibration" not in record.reason:
        raise RuntimeError(
            "UK national calibration latest mass record is not the calibration "
            f"record: entity={record.entity!r}, reason={record.reason!r}."
        )
    before_count = len(frame.mass_log) - 1
    ratios = np.asarray(result.weights, dtype=np.float64) / np.asarray(
        result.initial_weights, dtype=np.float64
    )
    old_total = float(record.old_total)
    new_total = float(record.new_total)
    before_kind = problem.problem.initial_weights.kind
    measure_resolution = dict(bindings.get("measure_resolution", {}))
    materialization = measure_resolution.pop("target_materialization", None)
    manifest: dict[str, object] = {
        "activated_reference_count": int(bindings["activated_reference_count"]),
        "resolved_reference_count": int(bindings["resolved_reference_count"]),
        "matrix_target_count": len(result.problem.names),
        "loss": float(result.final_loss),
        "effective_sample_size": effective_sample_size(result.weights),
        "max_weight_ratio": float(ratios.max()),
        "max_weight_ratio_bound": doctrine["max_weight_ratio"],
        "target_materialization": materialization,
        "weights": {
            "household_weight_kind": weights.kind.value,
            "household_weight_kind_chain": [
                {"stage": "staging", "kind": before_kind.value},
                {"stage": "national_calibration", "kind": weights.kind.value},
            ],
            "mass_log_records_before_calibration": before_count,
            "mass_log_records": len(frame.mass_log),
            "calibration_mass_change": {
                "entity": str(record.entity),
                "old_total": old_total,
                "new_total": new_total,
                "relative_shift": (new_total - old_total) / old_total,
                "declared_factor": record.declared_factor,
                "reason": str(record.reason),
            },
        },
        "solve": {
            "n_targets": len(result.problem.names),
            "n_households": len(frame.table("household")),
            "initial_loss": float(result.initial_loss),
            "final_loss": float(result.final_loss),
            "n_nonzero": int(np.count_nonzero(result.weights)),
        },
        "parameters": {"doctrine": dict(doctrine)},
    }
    if measure_resolution.get("mode") != "frame_only":
        manifest["measure_resolution"] = measure_resolution
    return manifest


def decode_national_result(context: KernelContext, frame: Frame):
    """The dense result and its ordered problem, rebound to ``frame``."""

    problem = decode_problem(context.artifacts["problem"].payload)
    solution = decode_solution(
        context.artifacts["solution"].payload,
        problem_sha256=problem.sha256,
        entity_ids=_axis(frame),
    )
    initial_frame = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {"household": problem.problem.initial_weights},
        frame.strata,
        metadata=frame.metadata,
    )
    result = decode_calibration_result(
        context.artifacts["result"].payload, frame=initial_frame, problem=problem
    )
    if not np.array_equal(result.weights, solution.weights):
        raise ValueError(
            "Final diagnostics result differs from the installed solution."
        )
    if not np.array_equal(frame.weights_for("household").values, solution.weights):
        raise ValueError("Gate population does not carry the bound solution weights.")
    return replace(result, frame=frame), problem


class UKNationalGateKernel(_NationalKernel):
    ref = "uk.full.national-gates@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        role=KernelRole.GATE,
        dependencies=("policyengine-uk",),
    )

    def implementation_hash(self) -> str:
        from .. import gate_battery
        from . import battery_bindings, diagnostics, weighted_integrity

        return hashlib.sha256(
            canonical_json(
                {
                    "code": source_hash(
                        type(self),
                        sys.modules[__name__],
                        gate_battery,
                        battery_bindings,
                        calibration_run,
                        weighted_integrity,
                        diagnostics,
                    ),
                    "country_resources": load_country_spec("uk").fingerprint,
                }
            )
        ).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        frame = context_frame(context)
        registry_document = _registry_artifact(context)
        national = registry_from_payload(registry_document["national_registry"])
        result, problem = decode_national_result(context, frame)
        doctrine = json.loads(str(context.params["doctrine"]))
        gates = uk_scoped_gate_manifest(
            tuple(UK_CALIBRATION_GATE_SCOPE),
            phases=("terminal",),
            policy_suffix=str(context.params["gate_policy_suffix"]),
        )
        if (
            canonical_json(_gates_manifest_payload(gates)).decode()
            != context.params["gate_manifest"]
        ):
            raise ValueError(
                "UK national gate manifest differs from its declared binding."
            )
        manifest_block = national_calibration_manifest(
            result, frame, problem=problem, doctrine=doctrine
        )
        diagnostics_rows = [
            {
                "name": row.name,
                "estimate": row.final_estimate,
                "target": row.target,
                "relative_error": row.relative_error,
            }
            for row in result.diagnostics
        ]
        admin_totals, admin_receipt = calibration_run.uk_aggregate_admin_totals(
            frame, gates
        )
        projection = calibration_run.uk_cgt_projection_artifact(frame, gates)
        artifacts = {
            "national_calibration": manifest_block,
            "parity_evidence": SimpleNamespace(
                target_relative_errors={
                    str(row["name"]): float(row["relative_error"])
                    for row in diagnostics_rows
                }
            ),
            "aggregate_admin": admin_totals,
            UK_CGT_PROJECTION_ARTIFACT_KEY: projection,
            "exclusions_evaluated_on": date.fromisoformat(
                str(context.params["review_date"])
            ),
        }
        report = evaluate_phase(
            gates,
            "terminal",
            EvidenceContext(frame=frame, artifacts=artifacts),
            registry=UK_GATE_REGISTRY,
        )
        blocking = report.blocking_outcomes(release_candidate=False)
        payload = {
            "schema_version": 1,
            "kind": "uk_national_gate_report",
            "posture": str(context.params["gate_posture"]),
            "policy_suffix": str(context.params["gate_policy_suffix"]),
            "scope": list(UK_CALIBRATION_GATE_SCOPE),
            "scope_exclusions": dict(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS),
            "report": gate_phase_report_payload(report, gates=gates),
            "aggregate_admin_measurement": _json_safe(admin_receipt),
            "blocking": [outcome.entry.id for outcome in blocking],
            "artifact_permitted": not blocking,
            "artifacts": {name: value.key for name, value in context.artifacts.items()},
        }
        evidence = {
            "schema_version": 1,
            "kind": "uk_national_calibration_evidence",
            "calibration": _json_safe(manifest_block),
            "diagnostics": _json_safe(diagnostics_rows),
            "target_geography_levels": uk_target_geography_levels(national),
            "register": calibration_run._register_census(
                national, registry_document["measure_exclusions"]
            ),
        }
        return KernelResult(
            artifacts={
                "gate_report": canonical_json(_json_safe(payload)),
                "calibration_evidence": canonical_json(evidence),
            },
            receipt={"outcome": "pass" if not blocking else "fail"},
        )


class UKNationalReadbackKernel(KernelBase):
    ref = "uk.full.national-readback@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC, role=KernelRole.GATE)

    def implementation_hash(self) -> str:
        from . import content_identity, national_frame

        return source_hash(type(self), content_identity, national_frame)

    def run(self, context: KernelContext) -> KernelResult:
        from ..artifact_files import file_artifact

        frame = context_frame(context)
        path = Path(context.sources[str(context.params["dataset_source"])])
        dataset = file_artifact(path)
        written, _provenance = load_uk_national_frame(path)
        failures = []
        expected = uk_frame_content_identity(frame)
        actual = uk_frame_content_identity(written)
        if actual != expected:
            failures.append(
                "Written national H5 content identity differs from the calibrated population."
            )
        if written.weights_for("household").kind is not WeightKind.CALIBRATED:
            failures.append("Written national H5 household weights are not calibrated.")
        if written.mass_log != frame.mass_log:
            failures.append(
                "Written national H5 mass log differs from the calibrated population."
            )
        if file_artifact(path) != dataset:
            raise ValueError("UK national H5 changed during graph readback validation.")
        report = {
            "schema_version": 1,
            "kind": "uk_national_export_readback",
            "passed": not failures,
            "failures": failures,
            "dataset": dataset,
            "content_identity": actual,
            "expected_content_identity": expected,
        }
        return KernelResult(
            artifacts={"readback": canonical_json(report)},
            receipt={"outcome": "pass" if not failures else "fail"},
        )


#: "The module's engine resolver at registration time": the default the
#: driver registers with, resolved when the registry is built so a hermetic
#: suite that stands the resolver in (or out, ``None``) on this module is
#: honoured, as the seam's tests stood theirs in on the role module.
ENGINE_RESOLVER = object()


def register_uk_national_kernels(
    registry: KernelRegistry, *, resolver_factory=ENGINE_RESOLVER
) -> KernelRegistry:
    """Register the national kernels beside the shared ones the graph binds."""

    if resolver_factory is ENGINE_RESOLVER:
        resolver_factory = UKMeasureResolver
    register_uk_source_codecs()
    for kernel in (
        UKNationalTargetKernel(),
        UKNationalProblemKernel(resolver_factory=resolver_factory),
        UKNationalGateKernel(),
        UKNationalReadbackKernel(),
    ):
        try:
            registry.get(kernel.ref)
        except KeyError:
            registry.register(kernel)
    return registry


# ---------------------------------------------------------------------------
# Driver-side projections: the seam-shaped files from the stored artifacts
# ---------------------------------------------------------------------------


def graph_payload(manifest, store, node: str, artifact: str) -> bytes:
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[artifact])


def national_result_from_manifest(manifest, store, *, frame: Frame):
    """The dense result and its ordered problem, rebound to the calibrated frame."""

    problem = decode_problem(
        graph_payload(manifest, store, NATIONAL_PROBLEM_NODE, "problem")
    )
    dense = manifest.nodes["uk.full.dense"].opaque_artifacts
    solution = decode_solution(
        store.load_bytes(dense["solution"]),
        problem_sha256=problem.sha256,
        entity_ids=_axis(frame),
    )
    initial_frame = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {"household": problem.problem.initial_weights},
        frame.strata,
        metadata=frame.metadata,
    )
    result = decode_calibration_result(
        store.load_bytes(dense["result"]), frame=initial_frame, problem=problem
    )
    if not np.array_equal(result.weights, solution.weights):
        raise ValueError("Stored national result differs from the installed solution.")
    if not np.array_equal(frame.weights_for("household").values, solution.weights):
        raise ValueError(
            "Calibrated national population does not carry the bound solution weights."
        )
    return replace(result, frame=frame), problem


def replay_uk_national_gate_battery(
    report_document: Mapping[str, Any],
    *,
    report_path: Path,
    release_id: str,
    diagnostics_sha256: str,
    posture: UKRowwisePosture = UK_ROWWISE_NATIONAL_POSTURE,
) -> dict[str, object]:
    """Persist, enforce and sign the seam battery from the graph's phase report.

    The graph evaluated the gates once (``uk.full.gates.calibrated``); the
    battery is replayed from that stored phase report through the same
    write-then-block boundary the seam used (``GateBatteryRun.record_phase``,
    microcosm#901), the diagnostics digest measured from the written file
    rides its attestation, and the scoped trio is grafted and signed exactly
    as the seam signed it. A blocking outcome raises
    :class:`~microcosm.build.gate_battery.GateBatteryBlockedError` after the
    report, block included, is on disk.
    """

    from microcosm.build.gate_battery import (
        BlockingMode,
        GateBatteryRun,
        gate_phase_report_from_payload,
    )

    if (
        report_document.get("kind") != "uk_national_gate_report"
        or report_document.get("schema_version") != 1
    ):
        raise ValueError("Unsupported UK national gate report artifact.")
    gates = uk_scoped_gate_manifest(
        tuple(posture.gate_scope),
        phases=("terminal",),
        policy_suffix=posture.gate_policy_suffix,
    )
    if report_document.get("policy_suffix") != posture.gate_policy_suffix:
        raise ValueError("UK national gate report was evaluated under another scope.")
    report = gate_phase_report_from_payload(report_document["report"], gates=gates)
    battery = GateBatteryRun(
        gates,
        release_id=release_id,
        report_path=report_path,
        # The seam never runs release-candidate posture: its scoped battery
        # covers the seven entries and must never sign a shippability claim;
        # shippability comes only from the release-cut certification.
        release_candidate=False,
        registry=UK_GATE_REGISTRY,
        release_evidence={"calibration_diagnostics_sha256": diagnostics_sha256},
    )
    battery.record_phase(report)
    battery.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
    payload = battery.report_payload()
    calibration_run.finalize_uk_scoped_gate_report(
        payload,
        posture=str(report_document["posture"]),
        scope_exclusions=dict(report_document["scope_exclusions"]),
        aggregate_admin_measurement=report_document["aggregate_admin_measurement"],
    )
    calibration_run._write_json(report_path, payload)
    return payload


def national_run_config(
    *,
    posture: UKRowwisePosture,
    config: UKNationalBuildConfig,
    registry_document: Mapping[str, Any],
    driver_parameters: Mapping[str, Any],
) -> dict[str, object]:
    """The seam's ``run_config``: the attempt identity every record carries."""

    return {
        "pipeline": posture.pipeline,
        "release_id": posture.release_id,
        "register_sha256": registry_document["national_registry"]
        and registry_from_payload(registry_document["national_registry"]).version,
        "calibration_year": int(config.calibration_year),
        "doctrine": national_doctrine_payload(config.doctrine),
        "doctrine_overrides": dict(config.doctrine_overrides),
        "ledger": dict(registry_document["ledger_provenance"]),
        "release_role": posture.role,
        "allow_unpinned_feed": bool(config.allow_unpinned_feed),
        "chronicle_feed_pin": dict(registry_document["chronicle_feed_pin"]),
        "rowwise_driver_parameters": dict(driver_parameters),
        "band_edge_register_sha256": registry_from_payload(
            registry_document["band_edge_registry"]
        ).version,
    }


def national_build_record(
    *,
    posture: UKRowwisePosture,
    build_id: str,
    run_config: Mapping[str, object],
    source_pins: Mapping[str, Mapping[str, object]],
    input_posture: Mapping[str, object],
    spine_provenance: Mapping[str, object],
    register: Mapping[str, object],
    calibration: Mapping[str, object],
    gate_report: Mapping[str, object],
    artifacts: Mapping[str, Mapping[str, object]],
    staging_delivery: Mapping[str, object] | None,
    graph: Mapping[str, object],
) -> dict[str, object]:
    """The seam-shaped ``build_record.json`` the release-cut certifier reads."""

    from microcosm.build.logbook_adoption import role_pins_digest
    from microcosm.build.staging_v2 import validate_staging_delivery

    record: dict[str, object] = {
        "schema_version": 1,
        "pipeline": posture.pipeline,
        "build_id": build_id,
        "run_config": dict(run_config),
        "source_pins": {k: dict(v) for k, v in source_pins.items()},
        "role_pins_digest": role_pins_digest(source_pins),
        "input_posture": dict(input_posture),
        "spine_provenance": dict(spine_provenance),
        "register": dict(register),
        "calibration": dict(calibration),
        "gate_summary": calibration_run._gate_summary(gate_report),
        # No shippability claim lives here: the calibration-scoped battery
        # covers seven of the declared gate entries. The release verdict is
        # the release-cut certification's, produced over this record.
        "certification": {
            "expected_artifact": str(
                Path(str(artifacts["staging_h5"]["path"])).with_suffix(
                    ".release_certification.json"
                )
            ),
            "producer": "tools/certify_uk_release_cut.py",
        },
        "artifacts": {k: dict(v) for k, v in artifacts.items()},
        "graph": dict(graph),
    }
    if staging_delivery is not None:
        record["staging_delivery"] = validate_staging_delivery(staging_delivery)
    return record
