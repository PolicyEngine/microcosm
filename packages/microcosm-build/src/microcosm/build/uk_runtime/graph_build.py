"""Compose the canonical UK full build on the shared executable graph.

The same graph handles all targets (the default), explicit target filters,
dense exports and exact-count exports. A bound spine checkpoint resumes this
composition; an arbitrary historical uk-data H5 is not a build source.
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
from .graph_kernels import UKClaimKernel, UKIdentityKernel, _normalize_create_frame
from .graph_population import (
    append_uk_population_nodes,
    population_columns,
    population_slices,
    register_uk_population_kernels,
)
from .graph_targets import append_uk_target_nodes, register_uk_target_kernels
from .local_doctrine import UK_LOCAL_CLONE_COUNT
from .national_sampling import UK_SAMPLE_SEED_DEFAULT

SPINE_PROVENANCE_TYPE = ArtifactType("microcosm.uk.bound-spine-provenance", 1)


@dataclass(frozen=True)
class UKFullBuildConfig:
    """Three independent controls: geography target scope, pool K, export k."""

    calibration_year: int
    time_period: str = field(default_factory=lambda: load_uk_frs_release().time_period)
    source_year: int = field(default_factory=lambda: load_uk_frs_release().survey_year)
    geography_levels: tuple[str, ...] | None = None
    n_clones: int = UK_LOCAL_CLONE_COUNT
    sample_fraction: float = 1.0
    source_sample_fraction: float = 1.0
    sample_seed: int = UK_SAMPLE_SEED_DEFAULT
    seed: int = 42
    engine_blocks: int = 1
    constituency_vintage: str = "2024_pcon"
    source_lineage_modulus: int | None = None
    calibration: UKGraphCalibrationConfig = UKGraphCalibrationConfig()

    def __post_init__(self) -> None:
        from .national_sampling import validate_sample_fraction

        validate_sample_fraction(self.sample_fraction, label="UK full pool")
        validate_sample_fraction(self.source_sample_fraction, label="UK source spine")
        if self.sample_fraction != 1.0 and self.source_sample_fraction != 1.0:
            raise ValueError(
                "A sampled spine cannot be sampled a second time in the full build."
            )
        if type(self.n_clones) is not int or self.n_clones < 1:
            raise ValueError("Geographic pool K must be a positive integer.")
        if self.engine_blocks not in {1, self.n_clones}:
            raise ValueError(
                "Engine blocks must be one or equal the geographic pool K."
            )
        if self.geography_levels is not None:
            if not self.geography_levels or set(self.geography_levels) - {
                "country",
                "region",
                "constituency",
                "la",
            }:
                raise ValueError(
                    "Use explicit supported geography levels or omit the selector for all."
                )
            if len(set(self.geography_levels)) != len(self.geography_levels):
                raise ValueError("Geographic target levels must not repeat.")
        if self.seed != self.calibration.seed:
            raise ValueError(
                "Pool and dense solve share the existing build seed; selection_seed is separate."
            )

    @property
    def effective_sample_fraction(self) -> float:
        return self.sample_fraction * self.source_sample_fraction


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


def uk_full_graph(
    config: UKFullBuildConfig,
    *,
    spine: Graph | None = None,
    spine_population: str | None = None,
    spine_weight_kind: str = "importance",
    optional_target_sources: tuple[str, ...] = (),
    checkpoint_identity: dict | None = None,
    review_date: str | None = None,
) -> UKFullGraph:
    """Append the full build to the existing source-owned UK spine graph."""

    initial = uk_spine_graph(source_mode="split") if spine is None else spine
    endpoint = (
        uk_spine_endpoint(initial).population
        if spine_population is None
        else spine_population
    )
    graph = append_uk_population_nodes(
        initial,
        population=endpoint,
        time_period=config.time_period,
        weight_kind=spine_weight_kind,
        sample_fraction=config.sample_fraction,
        sample_seed=config.sample_seed,
        n_clones=config.n_clones,
        seed=config.seed,
        source_year=config.source_year,
        constituency_vintage=config.constituency_vintage,
        source_lineage_modulus=config.source_lineage_modulus,
    )
    graph = append_uk_target_nodes(
        graph,
        calibration_year=config.calibration_year,
        time_period=config.time_period,
        geography_levels=config.geography_levels,
        engine_blocks=config.engine_blocks,
        sample_fraction=config.effective_sample_fraction,
        target_weight_rule=config.calibration.target_weight_rule,
        optional_sources=optional_target_sources,
        review_date=review_date,
    )
    cells = population_columns(graph, "uk.full.expand")
    # This structural checkpoint depends on every pool operation, including
    # evidence-only gates and the contribution problem. It binds the complete
    # incoming ledger before downstream nodes reconstruct any Frame slices.
    graph = replace(
        graph,
        nodes=(
            *graph.nodes,
            Node(
                "uk.full.pool",
                UKIdentityKernel.ref,
                structural=StructuralDelta.FILTER,
                base="uk.full.expand",
                inputs=population_slices(cells),
                description="Checkpoint the complete geographic pool and selected-target problem.",
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
    return national_frame.load_uk_national_frame(path)[0]


SOURCE_CODECS.register("uk-spine-h5-v1", _load_spine_h5)


class UKBoundSpineKernel(KernelBase):
    ref = "uk.full.bound_spine@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self), national_frame, calibration_run, release_certification
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
        return KernelResult(
            frame=_normalize_create_frame(frame, context),
            artifacts={"spine_provenance": canonical_json(provenance)},
        )


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
                    ).decode()
                },
                outputs=outputs,
                artifact_outputs=(
                    ArtifactOutput("spine_provenance", SPINE_PROVENANCE_TYPE),
                ),
                description="Resume a canonical spine with its bound lineage and source evidence.",
            ),
        ),
    )


def register_uk_full_kernels(registry: KernelRegistry) -> KernelRegistry:
    """Extend the existing UK source/stage registry with the full build."""

    # Source graph registries already contain these primitive kernels.
    for kernel in (UKBoundSpineKernel(), UKIdentityKernel(), UKClaimKernel()):
        try:
            registry.get(kernel.ref)
        except KeyError:
            registry.register(kernel)
    register_uk_population_kernels(registry)
    register_uk_target_kernels(registry)
    register_uk_calibration_kernels(registry)
    return registry
