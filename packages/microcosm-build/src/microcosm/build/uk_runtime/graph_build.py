"""Compose the canonical UK full build on the shared executable graph.

The same graph handles all targets (the default), explicit target filters,
dense exports and exact-count exports. A bound spine checkpoint resumes this
composition; an arbitrary historical uk-data H5 is not a build source. The
full build composes its targets, measures and problem on the spine's own
version, and its pool is that version whole: the spine's geographic clones,
areas and sampling are the build's (microcosm#931).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import pandas as pd

from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import SOURCE_CODECS

from . import calibration_run, national_frame, release_certification
from .frs_release import load_uk_frs_release
from .graph import uk_spine_endpoint, uk_spine_graph
from .graph_calibration import (
    UKCalibrationNodes,
    UKGraphCalibrationConfig,
    register_uk_calibration_kernels,
    uk_calibration_nodes,
)
from .graph_evidence import SPINE_BUILD_STATE_TYPE
from .graph_frames import normalize_create_frame
from .graph_kernels import UKClaimKernel, UKIdentityKernel
from .graph_population import (
    UKPoolCheckpointKernel,
    population_columns,
    population_slices,
    register_uk_population_kernels,
)
from .graph_targets import append_uk_target_nodes, register_uk_target_kernels

SPINE_PROVENANCE_TYPE = ArtifactType("microcosm.uk.bound-spine-provenance", 1)


@dataclass(frozen=True)
class UKFullBuildConfig:
    """Two independent controls, geography target scope and export k, over the
    bound spine.

    The spine binds the rest (microcosm#931): ``n_clones`` is its geographic
    clone count K and ``source_sample_fraction`` its sampling rung, both read
    from the spine the build binds, never chosen here. The full build samples,
    expands and places nothing of its own.
    """

    calibration_year: int
    time_period: str = field(default_factory=lambda: load_uk_frs_release().time_period)
    source_year: int = field(default_factory=lambda: load_uk_frs_release().survey_year)
    geography_levels: tuple[str, ...] | None = None
    target_families: tuple[str, ...] | None = None
    n_clones: int = 1
    source_sample_fraction: float = 1.0
    seed: int = 42
    engine_blocks: int = 1
    calibration: UKGraphCalibrationConfig = UKGraphCalibrationConfig()

    def __post_init__(self) -> None:
        from microcosm.build.frame_sampling import validate_sample_fraction

        validate_sample_fraction(self.source_sample_fraction, label="UK source spine")
        if type(self.n_clones) is not int or self.n_clones < 1:
            raise ValueError(
                "The spine's geographic clone count K must be a positive integer."
            )
        if self.engine_blocks not in {1, self.n_clones}:
            raise ValueError(
                "Engine blocks must be one or equal the spine's geographic clone count K."
            )
        if self.geography_levels is not None:
            if not self.geography_levels or set(self.geography_levels) - {
                "country",
                "region",
                "constituency",
                "la",
                "brma",
            }:
                raise ValueError(
                    "Use explicit supported geography levels or omit the selector for all."
                )
            if len(set(self.geography_levels)) != len(self.geography_levels):
                raise ValueError("Geographic target levels must not repeat.")
        if self.target_families is not None and (
            not self.target_families
            or any(not isinstance(f, str) or not f for f in self.target_families)
            or len(set(self.target_families)) != len(self.target_families)
        ):
            raise ValueError(
                "Use explicit non-repeating target families or omit the selector for all."
            )
        if self.seed != self.calibration.seed:
            raise ValueError(
                "Pool and dense solve share the existing build seed; selection_seed is separate."
            )

    @property
    def effective_sample_fraction(self) -> float:
        """The spine's sampling rung: the dense line samples nothing more."""
        return self.source_sample_fraction


@dataclass(frozen=True)
class UKFullGraph:
    graph: Graph
    calibration: UKCalibrationNodes
    config: UKFullBuildConfig

    @property
    def population(self) -> str:
        return self.calibration.population

    def operation_inventory(self) -> dict:
        compiled = compile_graph(self.graph)
        return {
            "schema": "microcosm.uk.full-build-operations.v1",
            "default_scope": "all_geographies",
            "configuration": asdict(self.config),
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


def _declared_spine_clone_count(graph: Graph) -> int | None:
    """The clone count K a composed spine's ``geographic_expand`` node declares.

    ``None`` where the graph composes no such node, as a bound checkpoint does:
    its K is read from its frame and build sidecar where it is bound.
    """

    from .geographic_expand import GEOGRAPHIC_EXPAND_STAGE_NAME

    for node in graph.nodes:
        if node.id == GEOGRAPHIC_EXPAND_STAGE_NAME:
            return int(node.params["geographic_clones"])
    return None


def uk_full_graph(
    config: UKFullBuildConfig,
    *,
    spine: Graph | None = None,
    spine_population: str | None = None,
    optional_target_sources: tuple[str, ...] = (),
    checkpoint_identity: dict | None = None,
    review_date: str | None = None,
) -> UKFullGraph:
    """Append the full build to the existing source-owned UK spine graph.

    The targets, measures and problem run on the spine's own version, and the
    pool checkpoint keeps every row of it: the dense line takes the spine's
    geographic clones, areas and sampling as they stand (microcosm#931). A
    composed spine whose ``geographic_expand`` node declares another clone
    count than the config's ``n_clones`` is refused, naming both K, and the
    measures refuse a frame that does not hold ``n_clones`` clones, whatever
    their engine blocks.
    """

    initial = uk_spine_graph(source_mode="split") if spine is None else spine
    endpoint = (
        uk_spine_endpoint(initial).population
        if spine_population is None
        else spine_population
    )
    declared = _declared_spine_clone_count(initial)
    if declared is not None and declared != config.n_clones:
        raise ValueError(
            f"The full build is configured for K = {config.n_clones} geographic "
            f"clones, but the spine it composes makes K = {declared}; the "
            "dense line takes K from its spine (microcosm#931)."
        )
    graph = append_uk_target_nodes(
        initial,
        population=endpoint,
        calibration_year=config.calibration_year,
        time_period=config.time_period,
        geography_levels=config.geography_levels,
        target_families=config.target_families,
        engine_blocks=config.engine_blocks,
        geographic_clones=config.n_clones,
        sample_fraction=config.effective_sample_fraction,
        target_weight_rule=config.calibration.target_weight_rule,
        optional_sources=optional_target_sources,
        review_date=review_date,
    )
    cells = population_columns(graph, endpoint)
    # This structural checkpoint depends on every operation on the spine's
    # version, the evidence-only gates and the contribution problem among
    # them. It binds the complete incoming ledger before downstream nodes
    # reconstruct any Frame slices, and keeps every row.
    graph = replace(
        graph,
        nodes=(
            *graph.nodes,
            Node(
                "uk.full.pool",
                UKPoolCheckpointKernel.ref,
                structural=StructuralDelta.FILTER,
                base=endpoint,
                inputs=population_slices(cells),
                description="Checkpoint the bound spine and the selected-target problem.",
            ),
        ),
    )
    calibration = uk_calibration_nodes(
        base="uk.full.pool",
        columns=cells,
        problem_producer="uk.full.problem",
        config=config.calibration,
        checkpoint_identity=checkpoint_identity,
    )
    checkpoint_sources = (
        ()
        if checkpoint_identity is None
        else (
            SourceRef("uk_size_checkpoint_manifest", "raw-bytes-v1"),
            SourceRef("uk_size_checkpoint_arrays", "raw-bytes-v1"),
        )
    )
    graph = replace(
        graph,
        nodes=(*graph.nodes, *calibration.nodes),
        sources=(*graph.sources, *checkpoint_sources),
    )
    compile_graph(graph)
    return UKFullGraph(graph, calibration, config)


def _load_spine_h5(path: Path) -> Frame:
    frame = national_frame.load_uk_national_frame(path)[0]
    national_frame.require_uk_producer_isa_income_contract(frame, label=str(path))
    return frame


SOURCE_CODECS.register("uk-spine-h5-v1", _load_spine_h5)


class UKBoundSpineKernel(KernelBase):
    ref = "uk.full.bound_spine@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def implementation_hash(self) -> str:
        from microcosm.frame import materialize as frame_materialize
        from microcosm.frame.adapters import policyengine_uk as uk_engine_adapter

        from .. import gate_battery, gates, trace
        from . import (
            content_identity,
            gate_scopes,
            geographic_clones,
            graph_frames,
            hmrc_uprating,
        )

        # The node also computes through the H5 it loads (frame_materialize,
        # the engine adapter's contracts), the checkpoint's content identity
        # and pinned engine (content_identity, hmrc_uprating), the frame's
        # geographic clone count it holds the sidecar's to (geographic_clones),
        # the gate digests it derives through a battery run (gate_battery,
        # gates, trace) over the spine gate scope and its scoped manifest
        # (gate_scopes), the create-frame normalisation it shares with the
        # spine's root (graph_frames, not the spine's kernel module, so an
        # edit there does not re-key the national line) and the canonical
        # encoding of its artifacts (microcosm#931).
        return source_hash(
            type(self),
            national_frame,
            calibration_run,
            gate_scopes,
            geographic_clones,
            release_certification,
            frame_materialize,
            uk_engine_adapter,
            content_identity,
            hmrc_uprating,
            gate_battery,
            gates,
            trace,
            graph_frames,
            canonical_json,
        )

    def run(self, context: KernelContext) -> KernelResult:
        if (
            json.loads(context.params["spine_gate_digests"])
            != calibration_run.uk_spine_checkpoint_gate_digests()
        ):
            raise ValueError(
                "Bound spine gate declarations differ from the compiled checkpoint request."
            )
        frame = _load_spine_h5(context.sources["uk_spine"])
        sidecar_path = context.sources["uk_spine_evidence"]
        sidecar = calibration_run.load_bound_spine_checkpoint(
            sidecar_path, frame, gate_report_path=context.sources["uk_spine_gates"]
        )
        provenance = calibration_run.strict_spine_provenance_from_sidecar(
            sidecar_path, sidecar, gate_report_path=context.sources["uk_spine_gates"]
        )
        created = normalize_create_frame(frame, context)
        return KernelResult(
            frame=created,
            artifacts={
                "spine_provenance": canonical_json(provenance),
                "spine_build_state": canonical_json(spine_build_state(created)),
            },
        )


def spine_build_state(frame: Frame) -> dict[str, object]:
    """The checkpoint's build state the terminal gates read as the spine frame.

    The input-coverage gate's family build-state half checks each stage
    family's typed importance weights, period and mass receipt on the spine
    the stages produced (the certifier's ``--spine-h5``); a graph build has
    no second population to hand a kernel, so the checkpoint publishes the
    three things that half reads.
    """

    return {
        "schema": "microcosm.uk.spine-build-state.v1",
        "household_weight_kind": frame.weights_for("household").kind.value,
        "time_period": str(frame.metadata["time_period"]),
        "mass_log": [asdict(record) for record in frame.mass_log],
    }


def bound_spine_graph(frame: Frame) -> Graph:
    """Declare a checkpoint schema; the CREATE kernel verifies bound evidence."""

    structural = {
        "person_id",
        "person_household_id",
        "person_benunit_id",
        "household_id",
        "benunit_id",
    }

    def token(dtype):
        if isinstance(dtype, pd.StringDtype) or dtype.kind in "OUS":
            return "string"
        return str(dtype)

    outputs = tuple(
        Owned(entity, str(column), token(frame.table(entity)[column].dtype))
        for entity in frame.entities
        for column in frame.table(entity).columns
        if column not in structural
    )
    return Graph(
        "uk",
        (
            SourceRef(
                "uk_spine",
                "uk-spine-h5-v1",
                "Bound canonical Microcosm spine checkpoint.",
            ),
            SourceRef(
                "uk_spine_evidence",
                "raw-bytes-v1",
                "Exact source lineage, graph and gate evidence for the checkpoint.",
            ),
            SourceRef(
                "uk_spine_gates",
                "raw-bytes-v1",
                "Exact gate report bound by the canonical spine sidecar.",
            ),
        ),
        (
            Node(
                "uk.full.spine_checkpoint",
                UKBoundSpineKernel.ref,
                structural=StructuralDelta.CREATE,
                sources=("uk_spine", "uk_spine_evidence", "uk_spine_gates"),
                params={
                    "spine_gate_digests": canonical_json(
                        calibration_run.uk_spine_checkpoint_gate_digests()
                    ).decode(),
                    # The checkpoint keeps the H5's own period: the CREATE
                    # normaliser would otherwise default it to the FRS
                    # vintage's, and a bound frame from another period (the
                    # seam fixtures, a historical checkpoint) would be
                    # re-dated on admission (found by the national line's
                    # projection fence, which declares the frame's period).
                    "time_period": str(frame.metadata["time_period"]),
                },
                outputs=outputs,
                artifact_outputs=(
                    ArtifactOutput("spine_provenance", SPINE_PROVENANCE_TYPE),
                    ArtifactOutput("spine_build_state", SPINE_BUILD_STATE_TYPE),
                ),
                description="Resume a canonical spine with its bound lineage and source evidence.",
            ),
        ),
    )


def register_uk_full_kernels(
    registry: KernelRegistry, *, progress_callback=None
) -> KernelRegistry:
    """Extend the existing UK source/stage registry with the full build.

    ``progress_callback`` is an operational observer of the solves' epochs
    (staging telemetry, stderr progress): the dense solve and, with a
    ``dataset_households`` request, the size search and refit, whose events
    carry their ``phase``; it never enters a node key.
    """

    # Source graph registries already contain these primitive kernels.
    for kernel in (UKBoundSpineKernel(), UKIdentityKernel(), UKClaimKernel()):
        try:
            registry.get(kernel.ref)
        except KeyError:
            registry.register(kernel)
    register_uk_population_kernels(registry)
    register_uk_target_kernels(registry)
    register_uk_calibration_kernels(registry, progress_callback=progress_callback)
    return registry
