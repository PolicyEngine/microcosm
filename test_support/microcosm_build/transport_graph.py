"""A toy transport-country graph for the shared calibration/terminal kernels.

Everything here is synthetic. The population is six made-up households; the
"Chronicle" consumer feed is a JSONL file of invented facts whose values are
the toy population's own weighted totals times fixed factors. Nothing here is
New Zealand data or donor microdata.

The graph mirrors the plan's calibration and terminal chain: CREATE, a
keep-all boundary FILTER (``open``), target compilation and the ordered
problem in ``open``, the ordered Adam REWEIGHT (``calibrate``), a keep-all
``terminal`` FILTER, then diagnostics, a hold-out comparison, a gate-battery
phase, export preparation, readback of the written H5 and the package
receipt. Two kernels stand in for G5a's population side: a JSON-source CREATE
and a keep-all FILTER that builds its mask from a declared person slice.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.gate_battery import DEFAULT_REGISTRY, FunctionBinding
from microcosm.build.gates import GateResult
from microcosm.build.transport.artifact_types import KERNEL_OUTPUTS
from microcosm.build.transport.gate_kernels import register_gate_kernel
from microcosm.build.transport.target_kernels import register_target_kernels
from microcosm.build.transport.terminal_kernels import (
    EXPORT_SOURCE_CODEC,
    register_terminal_kernels,
)
from microcosm.calibrate.artifacts import RESULT_TYPE, SOLUTION_TYPE
from microcosm.calibrate.ordered_kernels import CALIBRATE_ORDERED_ADAM
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    RunManifest,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    load_source_bytes,
    run_graph,
)
from microcosm.graph.canonical import canonical_json

COUNTRY = "xx"
PERIOD = 2026
SCHEMA = EntitySchema(person_entity="person", group_entities=("household", "family"))
ENTITIES = ("person", "household", "family")

#: The toy population: households with their families and persons.
TOY_POPULATION: dict[str, object] = {
    "households": [
        {"rent": 210.0, "is_renter": True, "region": "north", "weight": 100.0},
        {"rent": 0.0, "is_renter": False, "region": "north", "weight": 150.0},
        {"rent": 320.0, "is_renter": True, "region": "south", "weight": 75.0},
        {"rent": 0.0, "is_renter": False, "region": "south", "weight": 200.0},
        {"rent": 180.0, "is_renter": True, "region": "north", "weight": 125.0},
        {"rent": 260.0, "is_renter": True, "region": "south", "weight": 50.0},
    ],
    "families": [
        {"household": 0, "n_children": 0, "family_support": 0.0},
        {"household": 1, "n_children": 0, "family_support": 0.0},
        {"household": 2, "n_children": 1, "family_support": 95.0},
        {"household": 2, "n_children": 0, "family_support": 0.0},
        {"household": 3, "n_children": 0, "family_support": 0.0},
        {"household": 4, "n_children": 0, "family_support": 0.0},
        {"household": 5, "n_children": 1, "family_support": 120.0},
        {"household": 5, "n_children": 0, "family_support": 0.0},
    ],
    "persons": [
        {"family": 0, "age": 41, "employment_income": 52_000.0, "support": False},
        {"family": 0, "age": 39, "employment_income": 31_000.0, "support": False},
        {"family": 1, "age": 67, "employment_income": 0.0, "support": True},
        {"family": 2, "age": 35, "employment_income": 18_000.0, "support": True},
        {"family": 2, "age": 6, "employment_income": 0.0, "support": False},
        {"family": 3, "age": 19, "employment_income": 9_000.0, "support": True},
        {"family": 4, "age": 52, "employment_income": 74_000.0, "support": False},
        {"family": 4, "age": 50, "employment_income": 46_000.0, "support": False},
        {"family": 5, "age": 28, "employment_income": 38_000.0, "support": False},
        {"family": 6, "age": 30, "employment_income": 12_000.0, "support": True},
        {"family": 6, "age": 3, "employment_income": 0.0, "support": False},
        {"family": 7, "age": 72, "employment_income": 0.0, "support": True},
    ],
}

#: Owned columns of the CREATE node, in table order.
CREATE_COLUMNS = (
    Owned("person", "age", "int64"),
    Owned("person", "is_adult", "bool"),
    Owned("person", "employment_income", "float64"),
    Owned("person", "receives_support", "bool"),
    Owned("household", "rent", "float64"),
    Owned("household", "is_renter", "bool"),
    Owned("household", "region", "string"),
    Owned("family", "n_children", "int64"),
    Owned("family", "family_support", "float64"),
)

#: The calibration targets: (name, entity, measure, filter, factor).
#: Each fact's value is the toy population's design-weighted total of
#: ``measure`` times ``factor``, so the solve has somewhere to go.
CALIBRATION_TARGETS = (
    ("toy_adults", "person", "is_adult", None, 1.08),
    ("toy_employment_income", "person", "employment_income", None, 0.94),
    ("toy_adult_income", "person", "employment_income", "is_adult", 0.96),
    ("toy_renters", "household", "is_renter", None, 1.05),
    ("toy_family_support", "family", "family_support", None, 1.10),
    ("toy_parent_support", "family", "family_support", "n_children", 1.08),
)
#: Hold-out comparators measured by ``takeup.compare@1``.
HOLDOUT_TARGETS = (("toy_support_recipients", "person", "receives_support", None, 1.2),)


def toy_frame(
    population: Mapping[str, object] = TOY_POPULATION,
    *,
    weights: Sequence[float] | None = None,
    kind: WeightKind = WeightKind.DESIGN,
) -> Frame:
    """Build the toy population as a ``Frame``; design weights by default."""

    households = list(population["households"])
    families = list(population["families"])
    persons = list(population["persons"])
    person = pd.DataFrame(
        {
            "person_id": np.arange(len(persons), dtype=np.int64),
            "person_household_id": np.asarray(
                [families[p["family"]]["household"] for p in persons], dtype=np.int64
            ),
            "person_family_id": np.asarray(
                [p["family"] for p in persons], dtype=np.int64
            ),
            "age": np.asarray([p["age"] for p in persons], dtype=np.int64),
            "is_adult": np.asarray([p["age"] >= 18 for p in persons], dtype=bool),
            "employment_income": np.asarray(
                [p["employment_income"] for p in persons], dtype=np.float64
            ),
            "receives_support": np.asarray([p["support"] for p in persons], dtype=bool),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.arange(len(households), dtype=np.int64),
            "rent": np.asarray([h["rent"] for h in households], dtype=np.float64),
            "is_renter": np.asarray([h["is_renter"] for h in households], dtype=bool),
            "region": pd.array([h["region"] for h in households], dtype="string"),
        }
    )
    family = pd.DataFrame(
        {
            "family_id": np.arange(len(families), dtype=np.int64),
            "n_children": np.asarray(
                [f["n_children"] for f in families], dtype=np.int64
            ),
            "family_support": np.asarray(
                [f["family_support"] for f in families], dtype=np.float64
            ),
        }
    )
    regions = household["region"].astype(str).to_numpy()
    strata = pd.Series(
        regions[person["person_household_id"].to_numpy()],
        index=person.index,
        dtype=object,
        name="stratum",
    )
    if weights is None:
        weights = [h["weight"] for h in households]
    return Frame(
        {"person": person, "household": household, "family": family},
        SCHEMA,
        {"household": Weights(np.asarray(weights, dtype=np.float64), kind)},
        strata,
        metadata={"time_period": str(PERIOD)},
    )


def weighted_total(
    frame: Frame, entity: str, measure: str, filter_: str | None = None
) -> float:
    """The weighted total of one column (where ``filter_`` is nonzero).

    A plain ``numpy`` dot product over the entity's resolved weights: the
    independent reference the constraint-matrix path is compared with.
    """

    table = frame.table(entity)
    values = table[measure].to_numpy(dtype=np.float64)
    if filter_ is not None:
        values = values * (table[filter_].to_numpy(dtype=np.float64) != 0)
    weights = frame.resolve_weights(entity).values
    return float(np.dot(values, weights))


# ---------------------------------------------------------------------------
# Synthetic Chronicle facts and references
# ---------------------------------------------------------------------------


def toy_fact(name: str, value: float, *, entity: str) -> dict[str, object]:
    """One synthetic Chronicle consumer fact for the toy country."""

    return {
        "aggregate_fact_key": f"toy.aggregate_fact.v1:{name}",
        "semantic_fact_key": f"toy.semantic_fact.v1:{name}",
        "label": f"Toy fact {name}",
        "lineage": {
            "source_record_id": f"toy.record.{name}",
            "source_cell_keys": [f"toy.cell.{name}"],
            "source_row_keys": [],
        },
        "value": value,
        "period": {"type": "year", "value": PERIOD},
        "geography": {
            "level": "country",
            "id": "XX",
            "name": "Toy country",
            "vintage": "toy",
        },
        "entity": {"name": entity},
        "observed_measure": {
            "source_name": "toy_statistics",
            "source_table": "Synthetic toy table",
            "source_measure_id": name,
            "source_concept": f"toy:{name}",
            "unit": "count",
        },
        "aggregation": {"method": "sum"},
        "source": {
            "source_name": "toy_statistics",
            "source_table": "Synthetic toy table",
            "source_file": "synthetic_toy_facts.jsonl",
            "url": "https://example.invalid/toy",
            "vintage": str(PERIOD),
        },
        "dimensions": {},
        "dimension_labels": {},
        "dimension_value_labels": {},
        "universe_constraints": {"domain": "population"},
    }


def toy_facts(
    targets: Sequence[tuple[str, str, str, str | None, float]],
    frame: Frame | None = None,
) -> list[dict[str, object]]:
    """Facts for ``targets``, valued off the toy population."""

    frame = toy_frame() if frame is None else frame
    return [
        toy_fact(
            name,
            round(weighted_total(frame, entity, measure, filter_) * factor, 6),
            entity=entity,
        )
        for name, entity, measure, filter_, factor in targets
    ]


def write_facts(path: Path, facts: Sequence[Mapping[str, object]]) -> Path:
    """Write a bare consumer-facts JSONL file (one canonical row per line)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json(dict(fact)) + b"\n" for fact in facts))
    return path


def reference_row(
    name: str,
    *,
    entity: str,
    measure: str,
    filter_: str | None = None,
    activation_status: str = "active",
    category_id: str = "toy.population",
) -> dict[str, object]:
    """One reference selecting its fact by aggregate key."""

    row: dict[str, object] = {
        "name": name,
        "ledger_fact_key": f"toy.aggregate_fact.v1:{name}",
        "entity": entity,
        "measure": measure,
        "period": PERIOD,
        "family": "toy",
        "category_id": category_id,
        "metadata": {"activation_status": activation_status},
    }
    if filter_ is not None:
        row["filter"] = filter_
    return row


def reference_document(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """A ``target_references.json``-shaped document (schema 2)."""

    return {
        "country": COUNTRY,
        "schema_version": 2,
        "hierarchy": {
            "providers": {"toy_statistics": {"label": "Toy statistics office"}},
            "categories": {
                "toy.population": {
                    "provider_id": "toy_statistics",
                    "label": "Toy population",
                }
            },
        },
        "target_references": [dict(row) for row in rows],
    }


def references_for(
    targets: Sequence[tuple[str, str, str, str | None, float]],
) -> dict[str, object]:
    return reference_document(
        [
            reference_row(name, entity=entity, measure=measure, filter_=filter_)
            for name, entity, measure, filter_, _factor in targets
        ]
    )


def canonical_text(document: Mapping[str, object]) -> str:
    return canonical_json(dict(document)).decode("utf-8")


# ---------------------------------------------------------------------------
# Toy gates: test-only bindings over the decoded artifacts
# ---------------------------------------------------------------------------


def weight_ratio_gate(*, solution, problem, max_ratio: float) -> GateResult:
    """Test-only: the largest calibrated/starting weight ratio is bounded."""

    start = problem.problem.initial_weights.values
    ratio = float(np.max(solution.weights / start))
    if ratio <= max_ratio:
        return GateResult("weight_ratio", True, details={"max_ratio": ratio})
    return GateResult(
        "weight_ratio",
        False,
        failures=(f"max ratio {ratio:.4f} exceeds {max_ratio}",),
        details={"max_ratio": ratio},
    )


def per_family_fit_gate(*, diagnostics, max_abs_relative_error: float) -> GateResult:
    """Test-only: every diagnostics row is within a relative error bound."""

    worst = max(abs(row["relative_error"]) for row in diagnostics["targets"])
    if worst <= max_abs_relative_error:
        return GateResult("per_family_fit", True, details={"worst": worst})
    return GateResult(
        "per_family_fit",
        False,
        failures=(f"worst relative error {worst:.4f}",),
        details={"worst": worst},
    )


def nonnegative_columns_gate(*, frame, columns) -> GateResult:
    """Test-only: named person columns are non-negative on the frame."""

    table = frame.table("person")
    bad = [column for column in columns if (table[column] < 0).any()]
    if not bad:
        return GateResult("nonnegative_columns", True)
    return GateResult(
        "nonnegative_columns", False, failures=tuple(f"{c} < 0" for c in bad)
    )


def support_gate(*, frame, min_persons: int) -> GateResult:
    """Test-only: the frame has at least ``min_persons`` person records."""

    persons = frame.n("person")
    if persons >= min_persons:
        return GateResult("support", True, details={"persons": persons})
    return GateResult(
        "support", False, failures=(f"{persons} persons < {min_persons}",)
    )


TOY_BINDINGS = {
    **DEFAULT_REGISTRY,
    "support": FunctionBinding(
        name="support",
        gate=support_gate,
        parameter_keys=frozenset({"min_persons"}),
        frame_argument="frame",
    ),
    "weight_ratio": FunctionBinding(
        name="weight_ratio",
        gate=weight_ratio_gate,
        parameter_keys=frozenset({"max_ratio"}),
        artifact_arguments={"solution": "solution", "problem": "problem"},
    ),
    "per_family_fit": FunctionBinding(
        name="per_family_fit",
        gate=per_family_fit_gate,
        parameter_keys=frozenset({"max_abs_relative_error"}),
        artifact_arguments={"diagnostics": "diagnostics"},
    ),
    "nonnegative_columns": FunctionBinding(
        name="nonnegative_columns",
        gate=nonnegative_columns_gate,
        parameter_keys=frozenset({"columns"}),
        frame_argument="frame",
    ),
}


def toy_gates(
    *,
    max_ratio: float = 3.0,
    max_abs_relative_error: float = 0.5,
    extra: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """A ``gates.json``-shaped document with one ``terminal`` phase."""

    return {
        "country": COUNTRY,
        "version": 1,
        "policy": "Toy gates for the shared transport kernels (test only).",
        "phases": ["preflight", "terminal"],
        "gates": [
            {
                "id": "toy_weight_ratio",
                "gate": "weight_ratio",
                "phase": "terminal",
                "criticality": "release_blocking",
                "parameters": {"max_ratio": max_ratio},
            },
            {
                "id": "toy_fit",
                "gate": "per_family_fit",
                "phase": "terminal",
                "criticality": "release_blocking",
                "parameters": {"max_abs_relative_error": max_abs_relative_error},
            },
            {
                "id": "toy_nonnegative",
                "gate": "nonnegative_columns",
                "phase": "terminal",
                "criticality": "diagnostic",
                "parameters": {"columns": ["employment_income"]},
            },
            {
                "id": "toy_preflight_mass",
                "gate": "input_mass_parity",
                "phase": "preflight",
                "criticality": "diagnostic",
                "not_applicable": "The toy has no reference input mass.",
            },
            *extra,
        ],
    }


# ---------------------------------------------------------------------------
# Population stand-ins for the G5a kernels
# ---------------------------------------------------------------------------


class ToyCreate(KernelBase):
    """CREATE: the toy population from a JSON source (stands in for G5a)."""

    ref = "test.toy_create@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context: KernelContext) -> KernelResult:
        payload = load_source_bytes("raw-bytes-v1", context.sources["population"])
        return KernelResult(frame=toy_frame(json.loads(payload)))


class ToyKeepAll(KernelBase):
    """FILTER keep-all over a declared person slice (stands in for G5a)."""

    ref = "test.keep_all@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.FILTER
    )

    def run(self, context: KernelContext) -> KernelResult:
        ids = pd.Index(context.tables["person"]["person_id"].to_numpy(copy=True))
        return KernelResult(keep=pd.Series(True, index=ids, dtype="bool"))


def toy_registry(bindings: Mapping[str, object] = TOY_BINDINGS) -> KernelRegistry:
    """Every kernel the toy graph uses, with ``bindings`` on the gate."""

    registry = KernelRegistry()
    registry.register(ToyCreate())
    registry.register(ToyKeepAll())
    registry.register(CALIBRATE_ORDERED_ADAM)
    register_target_kernels(registry)
    register_terminal_kernels(registry)
    register_gate_kernel(registry, bindings)
    return registry


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def outputs(ref: str) -> tuple[ArtifactOutput, ...]:
    """The typed outputs a node running ``ref`` must declare."""

    if ref == CALIBRATE_ORDERED_ADAM.ref:
        return (
            ArtifactOutput("solution", SOLUTION_TYPE),
            ArtifactOutput("result", RESULT_TYPE),
        )
    return tuple(
        ArtifactOutput(name, kind) for name, kind in KERNEL_OUTPUTS[ref].items()
    )


def edge(alias: str, producer: Node, artifact: str | None = None) -> ArtifactInput:
    """An artifact edge typed from the producer's declaration."""

    name = alias if artifact is None else artifact
    declared = {output.name: output.type for output in producer.artifact_outputs}
    return ArtifactInput(alias, producer.id, name, declared[name])


ALL_SLICES = (
    Slice("person", ("age", "is_adult", "employment_income", "receives_support")),
    Slice("household", ("rent", "is_renter", "region")),
    Slice("family", ("n_children", "family_support")),
)


@dataclass(frozen=True)
class ToyConfig:
    """The knobs the tests turn."""

    calibration_targets: tuple = CALIBRATION_TARGETS
    holdout_targets: tuple = HOLDOUT_TARGETS
    gates: Mapping[str, object] = field(default_factory=toy_gates)
    max_weight_ratio: float | None = 3.0
    epochs: int = 40
    learning_rate: float = 0.05
    release_candidate: bool = False
    references_sha256: str = "1" * 64
    gates_sha256: str = "2" * 64
    description: str = ""
    #: Add a ``preflight`` gate phase on its own FILTER branch off ``open``
    #: and make the terminal phase name its report as upstream.
    preflight: bool = False


DEFAULT_CONFIG = ToyConfig()


def toy_graph(
    config: ToyConfig = DEFAULT_CONFIG, *, through_prepare: bool = False
) -> Graph:
    """The toy calibration/terminal graph; optionally stop at export.prepare."""

    create = Node(
        "toy.create",
        ToyCreate.ref,
        sources=("population",),
        structural=StructuralDelta.CREATE,
        outputs=CREATE_COLUMNS,
        description=config.description,
    )
    open_ = Node(
        "toy.open",
        ToyKeepAll.ref,
        structural=StructuralDelta.FILTER,
        base=create.id,
        inputs=(Slice("person", ("age",)),),
    )
    compile_ = Node(
        "toy.targets.compile",
        "targets.compile@1",
        population=open_.id,
        sources=("calibration_facts",),
        params={
            "country": COUNTRY,
            "references": canonical_text(references_for(config.calibration_targets)),
            "references_sha256": config.references_sha256,
        },
        artifact_outputs=outputs("targets.compile@1"),
        citation=config.description,
    )
    problem = Node(
        "toy.targets.problem",
        "targets.problem@1",
        population=open_.id,
        inputs=ALL_SLICES,
        params={"entities": ENTITIES, "weight_entity": "household"},
        artifact_inputs=(edge("surface", compile_),),
        artifact_outputs=outputs("targets.problem@1"),
    )
    solve_params = {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "mass": "free",
        "max_weight_ratio": config.max_weight_ratio,
    }
    if config.max_weight_ratio is not None:
        solve_params["weight_anchor"] = "design"
    calibrate = Node(
        "toy.calibrate",
        CALIBRATE_ORDERED_ADAM.ref,
        structural=StructuralDelta.REWEIGHT,
        base=open_.id,
        inputs=(Slice("household", ("rent",)),),
        params=solve_params,
        weights=WeightTransition("household", "calibrated", mass="free"),
        mass="free",
        artifact_inputs=(edge("problem", problem),),
        artifact_outputs=outputs(CALIBRATE_ORDERED_ADAM.ref),
    )
    terminal = Node(
        "toy.terminal",
        ToyKeepAll.ref,
        structural=StructuralDelta.FILTER,
        base=calibrate.id,
        inputs=(Slice("person", ("age",)),),
    )
    diagnostics = Node(
        "toy.calibration.diagnostics",
        "diagnostics.calibration@1",
        population=terminal.id,
        inputs=(Slice("household", ("rent",)),),
        params={"weight_entity": "household"},
        artifact_inputs=(
            edge("problem", problem),
            edge("solution", calibrate),
            edge("result", calibrate),
            edge("surface", compile_),
        ),
        artifact_outputs=outputs("diagnostics.calibration@1"),
    )
    compare = Node(
        "toy.validate.compare",
        "takeup.compare@1",
        population=terminal.id,
        sources=("holdout_facts",),
        inputs=ALL_SLICES,
        params={
            "country": COUNTRY,
            "references": canonical_text(references_for(config.holdout_targets)),
            "references_sha256": config.references_sha256,
            "entities": ENTITIES,
            "weight_entity": "household",
        },
        artifact_outputs=outputs("takeup.compare@1"),
    )
    gate_params = {
        "country": COUNTRY,
        "gates": canonical_text(config.gates),
        "gates_sha256": config.gates_sha256,
        "release_candidate": config.release_candidate,
        "entities": ENTITIES,
        "weight_entity": "household",
    }
    preflight_nodes: list[Node] = []
    upstream: tuple[ArtifactInput, ...] = ()
    if config.preflight:
        # Its own version, so no calibration node binds the gate manifest
        # (a member of ``open`` would be a predecessor of ``calibrate``).
        branch = Node(
            "toy.preflight",
            ToyKeepAll.ref,
            structural=StructuralDelta.FILTER,
            base=open_.id,
            inputs=(Slice("person", ("age",)),),
        )
        preflight = Node(
            "toy.gates.preflight",
            "gates.battery@1",
            population=branch.id,
            inputs=ALL_SLICES,
            params={**gate_params, "phase": "preflight"},
            artifact_outputs=outputs("gates.battery@1"),
        )
        preflight_nodes = [branch, preflight]
        upstream = (edge("preflight", preflight, "gate_report"),)
        gate_params["upstream"] = ("preflight",)
    gates = Node(
        "toy.gates.terminal",
        "gates.battery@1",
        population=terminal.id,
        inputs=ALL_SLICES,
        params={**gate_params, "phase": "terminal"},
        artifact_inputs=(
            edge("problem", problem),
            edge("solution", calibrate),
            edge("diagnostics", diagnostics),
            edge("comparison", compare),
            *upstream,
        ),
        artifact_outputs=outputs("gates.battery@1"),
    )
    prepare = Node(
        "toy.export.prepare",
        "export.prepare@1",
        population=terminal.id,
        inputs=ALL_SLICES,
        params={
            "entities": ENTITIES,
            "weight_entity": "household",
            "time_period": PERIOD,
        },
        artifact_inputs=(edge("gate_report", gates),),
        artifact_outputs=outputs("export.prepare@1"),
    )
    nodes = [
        create,
        open_,
        compile_,
        problem,
        calibrate,
        terminal,
        diagnostics,
        compare,
        *preflight_nodes,
        gates,
        prepare,
    ]
    if not through_prepare:
        readback = Node(
            "toy.export.readback",
            "export.readback@1",
            population=terminal.id,
            sources=("exported_dataset",),
            artifact_inputs=(edge("export_descriptor", prepare),),
            artifact_outputs=outputs("export.readback@1"),
        )
        package = Node(
            "toy.package",
            "transport.package@1",
            population=terminal.id,
            artifact_inputs=(
                edge("export_readback", readback),
                edge("gate_report", gates),
                edge("diagnostics", diagnostics),
                edge("comparison", compare),
                edge("surface", compile_),
            ),
            artifact_outputs=outputs("transport.package@1"),
        )
        nodes += [readback, package]
    return Graph(
        "toy",
        (
            SourceRef("population", "raw-bytes-v1"),
            SourceRef("calibration_facts", "raw-bytes-v1"),
            SourceRef("holdout_facts", "raw-bytes-v1"),
            SourceRef("exported_dataset", EXPORT_SOURCE_CODEC),
        ),
        tuple(nodes),
    )


@dataclass(frozen=True)
class ToySources:
    population: Path
    calibration_facts: Path
    holdout_facts: Path

    def mapping(self, exported: Path | None = None) -> dict[str, Path]:
        sources = {
            "population": self.population,
            "calibration_facts": self.calibration_facts,
            "holdout_facts": self.holdout_facts,
        }
        if exported is not None:
            sources["exported_dataset"] = exported
        return sources


def write_toy_sources(root: Path, config: ToyConfig = DEFAULT_CONFIG) -> ToySources:
    """Write the population and both fact feeds under ``root``."""

    root.mkdir(parents=True, exist_ok=True)
    population = root / "population.json"
    population.write_bytes(canonical_json(TOY_POPULATION))
    return ToySources(
        population=population,
        calibration_facts=write_facts(
            root / "calibration_facts.jsonl", toy_facts(config.calibration_targets)
        ),
        holdout_facts=write_facts(
            root / "holdout_facts.jsonl", toy_facts(config.holdout_targets)
        ),
    )


@dataclass(frozen=True)
class ToyRun:
    prepared: RunManifest
    manifest: RunManifest
    store: ContentStore
    exported: Path

    def artifact(self, node_id: str, name: str) -> bytes:
        return self.store.load_bytes(
            self.manifest.nodes[node_id].opaque_artifacts[name]
        )


def run_toy(
    root: Path,
    sources: ToySources,
    config: ToyConfig = DEFAULT_CONFIG,
    *,
    store: ContentStore | None = None,
    registry: KernelRegistry | None = None,
    corrupt=None,
    exported: Path | None = None,
    endpoint: str | None = None,
) -> ToyRun:
    """Run through export.prepare, write the H5 outside the graph, then finish.

    ``corrupt`` may rewrite the written file before the continuation reads
    it. ``exported`` reuses an already written file instead of writing one
    (two writes of one descriptor can differ in bytes: the HDF5 writer
    records object times). ``endpoint`` stops the continuation at that node.
    """

    from microcosm.build.transport.terminal_kernels import materialize_export

    store = ContentStore(root / "store") if store is None else store
    registry = toy_registry() if registry is None else registry
    prepared = run_graph(
        compile_graph(toy_graph(config, through_prepare=True)),
        sources=sources.mapping(),
        store=store,
        kernels=registry,
    )
    if exported is None:
        receipt = prepared.nodes["toy.export.prepare"]
        descriptor = json.loads(
            store.load_bytes(receipt.opaque_artifacts["export_descriptor"])
        )
        frame = store.load_frame(prepared.nodes["toy.terminal"].frame_key)
        exported = root / "export" / "toy_2026.h5"
        exported.parent.mkdir(parents=True, exist_ok=True)
        materialize_export(frame, descriptor, exported)
        if corrupt is not None:
            corrupt(exported)
    graph = toy_graph(config)
    if endpoint is not None:
        graph = through(graph, endpoint)
    manifest = run_graph(
        compile_graph(graph),
        sources=sources.mapping(exported),
        store=store,
        kernels=registry,
    )
    return ToyRun(prepared=prepared, manifest=manifest, store=store, exported=exported)


def with_config(config: ToyConfig, **changes) -> ToyConfig:
    return replace(config, **changes)


def through(graph: Graph, *endpoints: str) -> Graph:
    """The ancestor-closed subgraph that ends at ``endpoints``."""

    predecessors = compile_graph(graph).predecessors
    keep: set[str] = set()
    pending = list(endpoints)
    while pending:
        node_id = pending.pop()
        if node_id not in keep:
            keep.add(node_id)
            pending.extend(predecessors[node_id])
    return replace(graph, nodes=tuple(n for n in graph.nodes if n.id in keep))


def descendants(graph: Graph, roots) -> set[str]:
    """Every node downstream of (and including) ``roots``."""

    predecessors = compile_graph(graph).predecessors
    found = set(roots)
    changed = True
    while changed:
        changed = False
        for node_id, parents in predecessors.items():
            if node_id not in found and found.intersection(parents):
                found.add(node_id)
                changed = True
    return found


def run_through(
    root: Path,
    sources: ToySources,
    config: ToyConfig = DEFAULT_CONFIG,
    *endpoints: str,
    store: ContentStore | None = None,
    registry: KernelRegistry | None = None,
) -> tuple[RunManifest, ContentStore]:
    """Run the ancestor closure of ``endpoints`` (no written export)."""

    store = ContentStore(root / "store") if store is None else store
    graph = through(toy_graph(config, through_prepare=True), *endpoints)
    used = {name for node in graph.nodes for name in node.sources}
    manifest = run_graph(
        compile_graph(graph),
        sources={k: v for k, v in sources.mapping().items() if k in used},
        store=store,
        kernels=toy_registry() if registry is None else registry,
    )
    return manifest, store
