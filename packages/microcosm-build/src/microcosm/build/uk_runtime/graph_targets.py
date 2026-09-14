"""One full UK target surface: source compilation, selection and contributions."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.build import cross_grain
from microcosm.build.country_spec import load_country_spec
from microcosm.calibrate import CalibrationHierarchy, TargetRegistry, TargetSpec
from microcosm.calibrate.artifacts import (
    PROBLEM_TYPE,
    _pack,
    _unpack,
    decode_problem,
    encode_problem,
)
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.calibrate.target_selection import select_targets
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
    Node,
    SourceRef,
    source_hash,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import SOURCE_CODECS

from . import full_measure, full_problem, ladder_targets, ledger_targets, local_doctrine
from .full_measure import resolve_uk_full_measures
from .full_problem import build_uk_full_local_problem
from .full_targets import CHRONICLE_SOURCE_CODEC, load_chronicle_source_bytes
from .geography_ladder import load_uk_oa_ladder
from .graph_population import (
    GEOGRAPHY_GATE_TYPE,
    context_frame,
    population_columns,
    population_slices,
)
from .ladder_targets import ladder_vs_chronicle_household_dispersion
from .ledger_targets import uk_census_household_uprating, uk_ledger_households_total
from .local_rowwise import UKRowwiseNationalRows, prepare_uk_full_solve

TARGET_SURFACE_TYPE = ArtifactType("microcosm.uk.full-target-surface", 1)
TARGET_SELECTION_TYPE = ArtifactType("microcosm.uk.full-target-selection", 1)
MEASURE_TYPE = ArtifactType("microcosm.uk.full-measured-contributions", 1)

SOURCE_CODECS.register_bytes(CHRONICLE_SOURCE_CODEC, load_chronicle_source_bytes)


def registry_payload(registry: TargetRegistry) -> dict:
    return {"country": "uk", "specs": [asdict(spec) for spec in registry.specs]}


def registry_from_payload(payload: Mapping) -> TargetRegistry:
    if payload.get("country") != "uk" or not isinstance(payload.get("specs"), list):
        raise ValueError("Invalid UK target registry artifact.")
    # Decode the schema-8 hierarchy the same way TargetRegistry.from_json does
    # (#855): asdict() flattens CalibrationHierarchy to a mapping.
    return TargetRegistry(
        [
            TargetSpec(
                **{
                    **spec,
                    "hierarchy": (
                        CalibrationHierarchy.from_dict(spec["hierarchy"])
                        if spec.get("hierarchy") is not None
                        else None
                    ),
                }
            )
            for spec in payload["specs"]
        ],
        country="uk",
    )


def target_geography(spec: TargetSpec) -> str:
    """Resolve explicit metadata without interpreting legacy registry buckets."""

    metadata = spec.metadata
    local = metadata.get("geography_level")
    ledger = metadata.get("ledger_geography_level")
    if local and ledger and local != ledger:
        raise ValueError(f"Target {spec.name!r} has contradictory geography levels.")
    level = local or ledger
    if level not in {"country", "region", "constituency", "la"}:
        raise ValueError(
            f"Target {spec.name!r} has unsupported geography level {level!r}."
        )
    return str(level)


def _surface_records(frame: pd.DataFrame) -> list[dict]:
    def native(value):
        if isinstance(value, np.generic):
            value = value.item()
        if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
            return None
        return value

    def encode(key, value):
        # #855: the local surface carries each spec's CalibrationHierarchy;
        # the artifact stores it as TargetRegistry.to_json does (asdict).
        if key == "hierarchy":
            return None if value is None else asdict(value)
        return native(value)

    # Preserve the exact binary float values; decimal-rounding table codecs
    # can silently change target values on their first graph registration.
    return [
        {key: encode(key, value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _local_specs(surface: pd.DataFrame) -> list[TargetSpec]:
    return [
        TargetSpec(
            name=str(row["target_name"]),
            entity="household",
            measure=str(row["metric"]),
            value=float(row["value"]),
            period=row.get("period", 0),
            source=str(row.get("source", "uk_rowwise_local_surface")),
            family=str(row["family"]),
            # #855: schema-8 diagnostics need the hierarchy on every
            # registry-backed target; decode it as TargetRegistry.from_json does.
            hierarchy=(
                CalibrationHierarchy.from_dict(row["hierarchy"])
                if row.get("hierarchy") is not None
                else None
            ),
            metadata={
                **{key: value for key, value in row.items() if key != "hierarchy"},
                "geography_level": str(row["area_type"]),
                "geography_id": str(row["area_code"]),
                "materialization": "uk_local_surface",
            },
        )
        for row in _surface_records(surface)
    ]


def _selected_local_surface(surface: pd.DataFrame, specs) -> pd.DataFrame:
    """Keep exact selected facts, including repeated names in different years."""
    keys = {spec.key for spec in specs}
    periods = surface["period"] if "period" in surface else [0] * len(surface)
    keep = [
        (str(name), period) in keys
        for name, period in zip(surface["target_name"], periods, strict=True)
    ]
    return surface.loc[keep].reset_index(drop=True)


class _TargetKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, dependencies=("policyengine-uk",)
    )

    def implementation_hash(self) -> str:
        from . import full_targets

        implementation = source_hash(
            type(self),
            full_measure,
            full_problem,
            full_targets,
            cross_grain,
            ladder_targets,
            ledger_targets,
            local_doctrine,
        )
        return hashlib.sha256(
            canonical_json(
                {
                    "implementation": implementation,
                    "country_spec": load_country_spec("uk").fingerprint,
                }
            )
        ).hexdigest()


class UKFullTargetCompilationKernel(_TargetKernel):
    ref = "uk.full.target_compilation@1"

    def run(self, context: KernelContext) -> KernelResult:
        from .full_targets import load_uk_full_target_inputs
        from .ledger_targets import uk_local_target_surface

        inputs = load_uk_full_target_inputs(
            context.sources["uk_ledger_facts"],
            measure_exclusions=context.sources.get("uk_measure_exclusions"),
            register_json=context.sources.get("uk_frozen_register"),
            calibration_year=int(context.params["calibration_year"]),
            exclusions_evaluated_on=date.fromisoformat(
                str(context.params["review_date"])
            ),
        )
        ladder = load_uk_oa_ladder(context.sources["uk_ladder"])
        period = int(inputs["calibration_year"])
        national = inputs["national_registry"]
        local = inputs["local_registry"]
        uprating = uk_census_household_uprating(
            local,
            uk_ledger_households_total(inputs["artifact"].facts, period=period),
            period=period,
        )
        dispersion = ladder_vs_chronicle_household_dispersion(ladder, local.specs)
        surface, reconciliation = uk_local_target_surface(
            full_problem._joint_surface_registry(local, national),
            bound_national_target_ids=full_problem._national_contract_target_ids(
                national
            ),
            period=period,
            reviewed_unbound_higher_targets=inputs["reviewed_unbound_higher_targets"],
            census_household_uprating=uprating,
        )
        full = TargetRegistry(
            [
                *_local_specs(surface),
                *[
                    replace(
                        spec,
                        metadata={
                            **spec.metadata,
                            "materialization": "uk_national_measure",
                            "geography_level": target_geography(spec),
                        },
                    )
                    for spec in national.specs
                ],
            ],
            country="uk",
        )
        with Path(context.sources["uk_ladder"]).open("rb") as stream:
            ladder_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        payload = {
            "registry": registry_payload(full),
            "local_registry": registry_payload(local),
            "national_registry": registry_payload(national),
            "band_edge_registry": registry_payload(inputs["band_edge_registry"]),
            "surface": _surface_records(surface),
            "surface_columns": surface.columns.tolist(),
            "cross_geography": reconciliation,
            "census_household_uprating": reconciliation["census_household_uprating"],
            "household_dispersion": dispersion,
            "measure_exclusions": inputs["measure_exclusions"],
            "reviewed_unbound_higher_targets": inputs[
                "reviewed_unbound_higher_targets"
            ],
            "source_validation": {
                "national_source_pin": inputs["national_source_pin"],
                "local_source_pin": inputs["local_source_pin"],
                "register_completeness": inputs["register_completeness"],
                "ledger_provenance": inputs["ledger_provenance"],
                "targets": {
                    "chronicle": inputs["ledger_provenance"],
                    "paired_ladder_sha256": ladder_sha256,
                },
            },
            "uk_ledger_compiled_registries": {
                str(period): registry_payload(registry)
                for period, registry in inputs["uk_ledger_compiled_registries"].items()
            },
            "uk_ledger_compiled_local_registries": {
                str(period): registry_payload(registry)
                for period, registry in inputs[
                    "uk_ledger_compiled_local_registries"
                ].items()
            },
            "calibration_year": period,
        }
        return KernelResult(artifacts={"surface": canonical_json(payload)})


class UKFullTargetSelectionKernel(_TargetKernel):
    ref = "uk.full.target_selection@1"

    def run(self, context: KernelContext) -> KernelResult:
        full = json.loads(context.artifacts["surface"].payload)
        registry = registry_from_payload(full["registry"])
        levels = context.params.get("geography_levels")
        selected = select_targets(
            registry, geography_levels=levels, geography_resolver=target_geography
        )
        if not len(selected.registry):
            raise ValueError("Full build target selection contains no constraints.")
        payload = {
            "registry": registry_payload(selected.registry),
            "receipt": selected.receipt,
        }
        return KernelResult(artifacts={"selection": canonical_json(payload)})


class UKFullMeasureKernel(_TargetKernel):
    ref = "uk.full.measures@1"

    def run(self, context: KernelContext) -> KernelResult:
        gate = json.loads(context.artifacts["geography_gate"].payload)
        if not gate["passed"]:
            raise ValueError(
                "Geography integrity gate failed: " + "; ".join(gate["failures"])
            )
        frame = context_frame(context)
        full = json.loads(context.artifacts["surface"].payload)
        selected = registry_from_payload(
            json.loads(context.artifacts["selection"].payload)["registry"]
        )
        national = selected.select(
            predicate=lambda spec: (
                spec.metadata["materialization"] == "uk_national_measure"
            )
        )
        grains = tuple(
            sorted(
                {
                    target_geography(spec)
                    for spec in selected
                    if spec.metadata["materialization"] == "uk_local_surface"
                }
            )
        )
        with tempfile.TemporaryDirectory(
            prefix="microcosm-uk-full-measures-"
        ) as scratch:
            prepared, restore, rows, metrics, evidence = resolve_uk_full_measures(
                frame,
                national,
                period=int(full["calibration_year"]),
                scratch_dir=Path(scratch),
                band_edge_registry=registry_from_payload(full["band_edge_registry"]),
                blocks=int(context.params["engine_blocks"]),
                local_grains=grains,
            )
            # Compile national measures while temporary columns exist. The
            # immutable pool, rather than that evaluation Frame, goes forward.
            national_problem = (
                build_constraint_matrix(prepared, rows.targets, "household")
                if len(rows.targets)
                else None
            )
            if national_problem is not None and national_problem.skipped:
                failures = "; ".join(
                    f"{item.target.key}: {item.reason}"
                    for item in national_problem.skipped
                )
                raise ValueError(
                    "Selected national constraints failed to compile: " + failures
                )
            clean = restore(prepared)
            for entity in frame.entities:
                pd.testing.assert_frame_equal(clean.table(entity), frame.table(entity))
        arrays = {
            f"metrics_{grain}": metrics[grain].to_numpy(dtype=np.float64)
            for grain in grains
        }
        if national_problem is not None:
            arrays["national_problem"] = np.frombuffer(
                encode_problem(
                    national_problem,
                    entity_ids=frame.table("household")["household_id"].tolist(),
                ),
                dtype=np.uint8,
            )
        metadata = {
            "schema": MEASURE_TYPE.name + ".v1",
            "household_ids": frame.table("household")["household_id"].tolist(),
            "grains": {grain: metrics[grain].columns.tolist() for grain in grains},
            "has_national": national_problem is not None,
            "evidence": evidence,
        }
        return KernelResult(artifacts={"measures": _pack(metadata, arrays)})


def decode_measures(payload: bytes, *, selected: TargetRegistry) -> tuple[dict, dict]:
    grains = {
        target_geography(spec)
        for spec in selected
        if spec.metadata["materialization"] == "uk_local_surface"
    }
    has_national = any(
        spec.metadata["materialization"] == "uk_national_measure" for spec in selected
    )
    members = {f"metrics_{grain}" for grain in grains}
    if has_national:
        members.add("national_problem")
    return _unpack(payload, schema=MEASURE_TYPE.name + ".v1", members=members)


@dataclass(frozen=True)
class UKFullProblemInputs:
    """The exact admitted problem inputs shared by solve and holdout branches."""

    frame: object
    local_problem: object
    national_rows: UKRowwiseNationalRows | None
    bound_families: tuple[str, ...]
    metadata: dict
    full: dict
    selection: dict
    cross: dict
    rung: object
    national: TargetRegistry
    selected: TargetRegistry


def reconstruct_uk_full_problem_inputs(context: KernelContext) -> UKFullProblemInputs:
    """Reconstruct stored contributions without evaluating engine measures again."""
    frame = context_frame(context)
    full = json.loads(context.artifacts["surface"].payload)
    selection = json.loads(context.artifacts["selection"].payload)
    selected = registry_from_payload(selection["registry"])
    national = selected.select(
        predicate=lambda spec: spec.metadata["materialization"] == "uk_national_measure"
    )
    local_specs = [
        spec
        for spec in selected
        if spec.metadata["materialization"] == "uk_local_surface"
    ]
    metadata, arrays = decode_measures(
        context.artifacts["measures"].payload, selected=selected
    )
    ids = frame.table("household")["household_id"].tolist()
    if ids != metadata["household_ids"]:
        raise ValueError("Measured contributions belong to a different household axis.")
    metrics = {
        grain: pd.DataFrame(arrays[f"metrics_{grain}"], columns=columns, index=ids)
        for grain, columns in metadata["grains"].items()
    }
    surface = pd.DataFrame(full["surface"], columns=full["surface_columns"])
    surface = _selected_local_surface(surface, local_specs)
    ladder = load_uk_oa_ladder(context.sources["uk_ladder"])
    _, local_problem, cross, bound_families, rung = build_uk_full_local_problem(
        SimpleNamespace(result=SimpleNamespace(frame=frame), ladder=ladder),
        local_registry=registry_from_payload(full["local_registry"]),
        national_registry=national,
        local_metrics=metrics,
        period=int(full["calibration_year"]),
        sample_fraction=float(context.params["sample_fraction"]),
        reviewed_unbound_higher_targets=full["reviewed_unbound_higher_targets"],
        selected_surface=surface,
        surface_receipt=full["cross_geography"],
    )
    national_rows = None
    if metadata["has_national"]:
        national_problem = decode_problem(arrays["national_problem"].tobytes())
        national_rows = UKRowwiseNationalRows(
            national_problem.to_target_set(),
            national,
            tuple(sorted({spec.family for spec in national})),
        )
    return UKFullProblemInputs(
        frame,
        local_problem,
        national_rows,
        tuple(bound_families),
        metadata,
        full,
        selection,
        cross,
        rung,
        national,
        selected,
    )


class UKFullProblemKernel(_TargetKernel):
    ref = "uk.full.problem@1"

    def run(self, context: KernelContext) -> KernelResult:
        inputs = reconstruct_uk_full_problem_inputs(context)
        frame, local_problem = inputs.frame, inputs.local_problem
        national_rows, bound_families = inputs.national_rows, inputs.bound_families
        selection, selected = inputs.selection, inputs.selected
        metadata, full = inputs.metadata, inputs.full
        cross, rung = inputs.cross, inputs.rung
        ids = frame.table("household")["household_id"].tolist()
        prepared = prepare_uk_full_solve(
            frame,
            local_problem,
            bound_families=bound_families,
            national_rows=national_rows,
            target_weight_rule=str(context.params["target_weight_rule"]),
        )
        problem = build_constraint_matrix(frame, prepared.target_set, "household")
        if problem.skipped:
            raise ValueError("Selected full-build constraints failed to compile.")
        by_key = {spec.key: spec for spec in selected}
        target_metadata = []
        for target in problem.targets:
            spec = by_key[target.key]
            target_metadata.append(
                {
                    **spec.metadata,
                    "family": spec.family,
                    "source": spec.source,
                    "geography_level": target_geography(spec),
                }
            )
        doctrine = local_doctrine.UK_LOCAL_SOLVE_DOCTRINE
        payload = encode_problem(
            problem,
            entity_ids=ids,
            target_metadata=target_metadata,
            bindings={
                "target_selection": selection["receipt"],
                "target_selection_sha256": hashlib.sha256(
                    canonical_json(selection["receipt"])
                ).hexdigest(),
                "source_surface_sha256": context.artifacts["surface"].key,
                "target_loss_weights": (
                    np.ones(problem.n_targets, dtype=np.float64)
                    if prepared.target_loss_weights is None
                    else prepared.target_loss_weights
                ).tolist(),
                "mass_reason": prepared.mass_reason,
                "target_loss_cap": doctrine.target_loss_cap,
                "max_weight_ratio": doctrine.max_weight_ratio,
                "binding_adjudications": prepared.binding_adjudications,
                "rung_surface": rung,
                "bound_families": list(bound_families),
                "measure_resolution": metadata["evidence"],
                "cross_geography": cross,
                "measure_exclusions": full["measure_exclusions"],
                "calibration_year": full["calibration_year"],
            },
        )
        return KernelResult(artifacts={"problem": payload})


def append_uk_target_nodes(
    graph: Graph,
    *,
    population: str = "uk.full.expand",
    calibration_year: int,
    time_period: str,
    geography_levels: tuple[str, ...] | None = None,
    engine_blocks: int = 1,
    sample_fraction: float = 1.0,
    target_weight_rule: str = "uniform",
    optional_sources: tuple[str, ...] = (),
    review_date: str | None = None,
) -> Graph:
    """Default all geographies. Scope is independent of K, k and solver outcomes."""

    if geography_levels is not None and not geography_levels:
        raise ValueError("An explicit geography selector must contain levels.")
    cells = population_columns(graph, population)
    slices = population_slices(cells)
    compile_sources = ("uk_ladder", "uk_ledger_facts", *optional_sources)
    common = {"time_period": time_period}
    review_date = date.today().isoformat() if review_date is None else review_date
    date.fromisoformat(review_date)
    contract_identity = load_country_spec("uk").fingerprint
    surface = ArtifactInput(
        "surface", "uk.full.target_compilation", "surface", TARGET_SURFACE_TYPE
    )
    selection = ArtifactInput(
        "selection", "uk.full.target_selection", "selection", TARGET_SELECTION_TYPE
    )
    nodes = (
        Node(
            "uk.full.target_compilation",
            UKFullTargetCompilationKernel.ref,
            population=population,
            sources=compile_sources,
            params={
                "calibration_year": calibration_year,
                "review_date": review_date,
                "country_contract_sha256": contract_identity,
            },
            artifact_outputs=(ArtifactOutput("surface", TARGET_SURFACE_TYPE),),
            description="Compile Chronicle national and local targets and validate the paired assignment ladder.",
        ),
        Node(
            "uk.full.target_selection",
            UKFullTargetSelectionKernel.ref,
            population=population,
            params={"geography_levels": geography_levels},
            artifact_inputs=(surface,),
            artifact_outputs=(ArtifactOutput("selection", TARGET_SELECTION_TYPE),),
            description="Select all geographies unless a target filter is explicitly requested.",
        ),
        Node(
            "uk.full.measures",
            UKFullMeasureKernel.ref,
            population=population,
            inputs=slices,
            params={**common, "engine_blocks": engine_blocks},
            artifact_inputs=(
                surface,
                selection,
                ArtifactInput(
                    "geography_gate",
                    "uk.full.geography_gate",
                    "gate",
                    GEOGRAPHY_GATE_TYPE,
                ),
            ),
            artifact_outputs=(ArtifactOutput("measures", MEASURE_TYPE),),
            description="Evaluate selected measures and compile temporary national contributions.",
        ),
        Node(
            "uk.full.problem",
            UKFullProblemKernel.ref,
            population=population,
            inputs=slices,
            params={
                **common,
                "sample_fraction": float(sample_fraction),
                "target_weight_rule": target_weight_rule,
            },
            sources=("uk_ladder",),
            artifact_inputs=(
                surface,
                selection,
                ArtifactInput("measures", "uk.full.measures", "measures", MEASURE_TYPE),
            ),
            artifact_outputs=(ArtifactOutput("problem", PROBLEM_TYPE),),
            description="Build the one ordered selected-target problem and admission receipts.",
        ),
    )
    existing = {source.name for source in graph.sources}
    sources = tuple(
        SourceRef(
            name,
            CHRONICLE_SOURCE_CODEC if name == "uk_ledger_facts" else "raw-bytes-v1",
        )
        for name in compile_sources
        if name not in existing
    )
    return replace(
        graph, nodes=(*graph.nodes, *nodes), sources=(*graph.sources, *sources)
    )


def register_uk_target_kernels(registry: KernelRegistry) -> None:
    for kernel in (
        UKFullTargetCompilationKernel(),
        UKFullTargetSelectionKernel(),
        UKFullMeasureKernel(),
        UKFullProblemKernel(),
    ):
        registry.register(kernel)
