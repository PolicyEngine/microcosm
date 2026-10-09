"""Authenticated transport bindings with three engine-free adapters in one run."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import microcosm.build.transport.registry as registry_module
from microcosm.build.transport.codecs import (
    DONOR_SOURCE_CODEC,
    FACTS_SOURCE_CODEC,
    RULESPEC_SOURCE_CODEC,
)
from microcosm.build.transport.gate_bindings import TRANSPORT_GATE_REGISTRY
from microcosm.build.transport.registry import build_transport_registry
from microcosm.build.transport.terminal_kernels import EXPORT_SOURCE_CODEC
from microcosm.frame import (
    EntitySchema,
    ExportContract,
    Frame,
    VariableMetadata,
    WeightKind,
    Weights,
)
from microcosm.frame.adapters.axiom import NZ_SCHEMA, AxiomEngine, AxiomPeriod
from microcosm.frame.kernels import SimulateRulesKernel
from microcosm.frame.rules_kernels import SimulateRulesByRefKernel
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SeedSource,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.codecs import load_raw_bytes
from test_support.microcosm_build.transport_population import UNIT_RULE


class _ToyEngine:
    """Pure adapter whose formula and configuration have explicit identities."""

    def __init__(self, root: Path, binding: Mapping, *, offset: float = 0.0):
        self.rulespec_root = root
        self.binding = binding
        self.offset = offset
        self.relation_checks = []

    def cache_identity(self) -> Mapping:
        return {"offset": self.offset}

    def assert_no_relations(self, entity: str) -> None:
        self.relation_checks.append(entity)
        if self._program()["relations"]:
            raise NotImplementedError(f"{self.binding['id']} declares relations")

    def _program(self) -> Mapping:
        return json.loads(
            (self.rulespec_root / self.binding["rulespec_path"]).read_text()
        )

    def variable_metadata(self, name: str) -> VariableMetadata:
        if name not in self.binding["variables"]:
            raise ValueError(f"Unknown toy output {name!r}")
        return VariableMetadata(name, self.binding["entity"], "float", "year")

    def variables(self) -> Sequence[str]:
        return (self._program()["input"],)

    def entity_schema(self) -> EntitySchema:
        return NZ_SCHEMA

    def materialize(
        self, bundle: Frame, variables: Sequence[str], period: int | str
    ) -> Mapping[str, np.ndarray]:
        program = self._program()
        values = bundle.table(self.binding["entity"])[program["input"]].to_numpy(
            dtype=np.float64
        )
        return {
            name: values * program["coefficient"] + self.offset for name in variables
        }

    def export_contract(self) -> ExportContract:
        return ExportContract.empty()

    def write_dataset(self, bundle: Frame, path: str | Path, period: int | str):
        raise NotImplementedError("The toy fixture does not export datasets")


def _fixture(tmp_path: Path):
    root = tmp_path / "rulespec"
    root.mkdir(exist_ok=True)
    rows = []
    for name, entity, column, coefficient in (
        ("main_benefit", "person", "age", 2.0),
        ("pension", "person", "age", 3.0),
        ("housing", "family", "cost", 4.0),
    ):
        payload = json.dumps(
            {"input": column, "coefficient": coefficient, "relations": []},
            sort_keys=True,
        ).encode()
        module = root / f"{name}.yaml"
        module.write_bytes(payload)
        rows.append(
            {
                "id": name,
                "rulespec_path": module.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "entity": entity,
                "engine_entity": entity.capitalize(),
                "period": "2026-27",
                "variables": [f"toy_{name}"],
            }
        )
    document = {
        "engine": {"commit": "a" * 40, "wheel_sha256": "b" * 64},
        "rulespec": {"commit": "c" * 40},
        "entity_names": {
            "person": "Person",
            "household": "Household",
            "family": "Family",
        },
        "periods": {
            "2026-27": {
                "kind": "tax_year",
                "start": "2026-04-01",
                "end": "2027-03-31",
            }
        },
        "bindings": rows,
    }
    engines = {row["id"]: _ToyEngine(root, row) for row in rows}
    return root, document, engines


def test_registers_available_kernels_and_codecs(tmp_path):
    root, document, engines = _fixture(tmp_path)
    result = build_transport_registry(
        document, root, unit_rule=UNIT_RULE, engines_by_binding=engines
    )
    assert isinstance(
        result.kernels.get("simulate.rules_by_ref@1"), SimulateRulesByRefKernel
    )
    assert "simulate.rules@1" not in result.kernels.refs()
    assert {
        "transport.create@1",
        "transport.boundary@1",
        "geography.assign_atomic@1",
        "targets.compile@1",
        "targets.problem@1",
        "calibrate.ordered_adam@1",
        "diagnostics.calibration@1",
        "export.prepare@1",
        "export.readback@1",
        "transport.package@1",
    } <= set(result.kernels.refs())
    gate = result.kernels.get("gates.battery@1")
    assert all(
        gate.registry[name] is binding
        for name, binding in TRANSPORT_GATE_REGISTRY.items()
    )
    for codec in (
        DONOR_SOURCE_CODEC,
        FACTS_SOURCE_CODEC,
        RULESPEC_SOURCE_CODEC,
        EXPORT_SOURCE_CODEC,
    ):
        assert callable(result.codecs.get(codec))
    assert all(
        engine.relation_checks == [engine.binding["entity"]]
        for engine in engines.values()
    )
    assert len(result.engines) == 3
    assert all(
        json.loads(ref)["engine"] == "python-rules-engine" for ref in result.engines
    )
    with pytest.raises(TypeError):
        result.engine_refs["other"] = "unused"
    with pytest.raises(TypeError):
        result.engines["other"] = engines["housing"]


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.integers(min_value=-20, max_value=20))
def test_added_binding_and_prose_leave_existing_refs_and_hashes_unchanged(
    tmp_path, offset
):
    root, document, engines = _fixture(tmp_path)
    engines["main_benefit"].offset = float(offset)
    first_document = copy.deepcopy(document)
    first_document["bindings"] = first_document["bindings"][:2]
    first = build_transport_registry(
        first_document,
        root,
        unit_rule=UNIT_RULE,
        engines_by_binding={
            name: engines[name] for name in ("main_benefit", "pension")
        },
    )
    document["description"] = "Different explanatory prose"
    document["bindings"][0]["notes"] = "This is not an engine configuration"
    document["periods"]["unused_period"] = {
        "kind": "year",
        "start": "2028-01-01",
        "end": "2028-12-31",
    }
    second = build_transport_registry(
        document, root, unit_rule=UNIT_RULE, engines_by_binding=engines
    )
    assert all(
        second.engine_refs[name] == ref for name, ref in first.engine_refs.items()
    )
    assert first.kernels.implementation_hash(
        "simulate.rules_by_ref@1"
    ) == second.kernels.implementation_hash("simulate.rules_by_ref@1")


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.integers(min_value=1, max_value=30))
def test_adapter_configuration_changes_only_its_reference(tmp_path, offset):
    root, document, engines = _fixture(tmp_path)
    first = build_transport_registry(
        document, root, unit_rule=UNIT_RULE, engines_by_binding=engines
    )
    engines["housing"].offset = float(offset)
    second = build_transport_registry(
        document, root, unit_rule=UNIT_RULE, engines_by_binding=engines
    )
    assert first.engine_refs["housing"] != second.engine_refs["housing"]
    assert first.engine_refs["main_benefit"] == second.engine_refs["main_benefit"]
    assert first.engine_refs["pension"] == second.engine_refs["pension"]
    assert first.kernels.implementation_hash(
        "simulate.rules_by_ref@1"
    ) == second.kernels.implementation_hash("simulate.rules_by_ref@1")


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.binary(min_size=1, max_size=8))
def test_rulespec_tree_bytes_change_refs_without_changing_kernel_hash(tmp_path, edit):
    root, document, engines = _fixture(tmp_path)
    shared = root / "shared-resource.txt"
    shared.write_bytes(b"before")
    first = build_transport_registry(
        document, root, unit_rule=UNIT_RULE, engines_by_binding=engines
    )
    shared.write_bytes(b"after" + edit)
    second = build_transport_registry(
        document, root, unit_rule=UNIT_RULE, engines_by_binding=engines
    )
    assert all(first.engine_refs[name] != second.engine_refs[name] for name in engines)
    assert first.kernels.implementation_hash(
        "simulate.rules_by_ref@1"
    ) == second.kernels.implementation_hash("simulate.rules_by_ref@1")


def test_native_engines_use_pinned_refs_before_relation_compilation(
    tmp_path, monkeypatch
):
    root, document, _ = _fixture(tmp_path)
    document["periods"]["unused_period"] = {
        "kind": "year",
        "start": "2028-01-01",
        "end": "2028-12-31",
    }
    checks = []

    def check(engine, entity):
        assert engine._pinned_tree_sha256 is not None
        assert engine._schema == NZ_SCHEMA
        assert engine._nesting == {"family": "household"}
        assert engine._rulespec_roots == (root.resolve(),)
        assert engine._output_dtypes == "graph"
        assert engine._periods == {
            "2026-27": AxiomPeriod("2026-04-01", "2027-03-31", "tax_year")
        }
        checks.append((engine._module.name, entity))

    def metadata(engine, name):
        row = next(row for row in document["bindings"] if name in row["variables"])
        return VariableMetadata(name, row["entity"], "float", "year")

    monkeypatch.setattr(registry_module, "assert_no_relations", check)
    monkeypatch.setattr(AxiomEngine, "variable_metadata", metadata)
    monkeypatch.setattr(AxiomEngine, "graph_dtype", lambda engine, name: "float64")
    result = build_transport_registry(document, root, unit_rule=UNIT_RULE)
    assert len(checks) == 3
    assert all(isinstance(engine, AxiomEngine) for engine in result.engines.values())
    for row in document["bindings"]:
        ref = json.loads(result.engine_refs[row["id"]])
        assert ref["engine"] == "axiom-rules-engine"
        assert ref["module_sha256"] == row["sha256"]
        assert ref["module"] == row["rulespec_path"]
        assert ref["periods"] == {row["period"]: document["periods"][row["period"]]}


@pytest.mark.parametrize(
    "problem", ["root", "module", "schema", "periods", "names", "nesting", "dtypes"]
)
def test_injected_axiom_configuration_must_match_the_binding(tmp_path, problem):
    root, document, _ = _fixture(tmp_path)
    periods = {"2026-27": AxiomPeriod("2026-04-01", "2027-03-31", "tax_year")}
    engines = {
        row["id"]: AxiomEngine(
            root / row["rulespec_path"],
            schema=NZ_SCHEMA,
            rulespec_roots=(root,),
            periods=periods,
            entity_names=document["entity_names"],
            nesting={"family": "household"},
            output_dtypes="graph",
        )
        for row in document["bindings"]
    }
    engine = engines["main_benefit"]
    if problem == "root":
        engine._rulespec_roots = (tmp_path,)
    elif problem == "module":
        engine._module = root / "pension.yaml"
    elif problem == "schema":
        engine._schema = EntitySchema(group_entities=("household",))
    elif problem == "periods":
        engine._periods = None
    elif problem == "names":
        engine._entity_names = {"person": "DifferentPerson"}
    elif problem == "nesting":
        engine._nesting = {}
    else:
        engine._output_dtypes = "native"
    with pytest.raises(ValueError, match="rules binding 'main_benefit'"):
        build_transport_registry(
            document, root, unit_rule=UNIT_RULE, engines_by_binding=engines
        )


@pytest.mark.parametrize(
    ("section", "field"),
    [("engine", "commit"), ("engine", "wheel_sha256"), ("rulespec", "commit")],
)
def test_missing_pin_names_the_activation_gap(tmp_path, section, field):
    root, document, engines = _fixture(tmp_path)
    document[section][field] = None
    with pytest.raises(ValueError, match=rf"axiom_rules_bindings\.{section}\.{field}"):
        build_transport_registry(
            document, root, unit_rule=UNIT_RULE, engines_by_binding=engines
        )


@pytest.mark.parametrize("problem", ["hash", "path", "period", "duplicate", "entity"])
def test_malformed_binding_is_refused(tmp_path, problem):
    root, document, engines = _fixture(tmp_path)
    first = document["bindings"][0]
    if problem == "hash":
        first["sha256"] = "d" * 64
    elif problem == "path":
        first["rulespec_path"] = "../outside.yaml"
    elif problem == "period":
        first["period"] = "unmapped"
    elif problem == "duplicate":
        document["bindings"].append(dict(first))
    else:
        first["engine_entity"] = "Family"
    with pytest.raises(ValueError):
        build_transport_registry(
            document, root, unit_rule=UNIT_RULE, engines_by_binding=engines
        )


@pytest.mark.parametrize("problem", ["root", "schema", "relations", "identity", "keys"])
def test_injected_adapter_contract_is_checked(tmp_path, problem):
    root, document, engines = _fixture(tmp_path)
    engine = engines["housing"]
    if problem == "root":
        engine.rulespec_root = tmp_path
    elif problem == "schema":
        engine.entity_schema = lambda: EntitySchema(group_entities=("household",))
    elif problem == "relations":
        engine.assert_no_relations = lambda entity: (_ for _ in ()).throw(
            NotImplementedError("relations")
        )
    elif problem == "identity":
        engine.cache_identity = None
    else:
        engines.pop("housing")
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        build_transport_registry(
            document, root, unit_rule=UNIT_RULE, engines_by_binding=engines
        )


@pytest.mark.parametrize("value", [float("nan"), {1: "key"}, lambda: None])
def test_adapter_identity_uses_closed_canonical_data(tmp_path, value):
    root, document, engines = _fixture(tmp_path)
    engines["housing"].cache_identity = lambda: {"configuration": value}
    with pytest.raises((TypeError, ValueError)):
        build_transport_registry(
            document, root, unit_rule=UNIT_RULE, engines_by_binding=engines
        )


def _frame(document: Mapping) -> Frame:
    return Frame(
        {
            "person": pd.DataFrame(document["person"]).astype("int64"),
            "household": pd.DataFrame(document["household"]).astype("int64"),
            "family": pd.DataFrame(document["family"]).astype(
                {"family_id": "int64", "cost": "float64"}
            ),
        },
        NZ_SCHEMA,
        {
            "household": Weights(
                np.asarray(document["weights"], dtype=np.float64), WeightKind.DESIGN
            )
        },
    )


class _Population(KernelBase):
    ref = "test.registry_population@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.CREATE,
    )

    def run(self, context: KernelContext) -> KernelResult:
        return KernelResult(
            frame=_frame(json.loads(context.sources["population"].read_text()))
        )


def _node(
    row: Mapping, reference: str, kernel: str = SimulateRulesByRefKernel.ref
) -> Node:
    inputs = (Slice("person", ("age",)),)
    if row["entity"] == "family":
        inputs += (Slice("family", ("cost",)),)
    return Node(
        f"toy.rules.{row['id']}",
        kernel,
        sources=("rulespec_nz",),
        inputs=inputs,
        outputs=tuple(
            Owned(row["entity"], name, "float64") for name in row["variables"]
        ),
        params={
            "engine_ref": reference,
            "period": row["period"],
            "variables": tuple(row["variables"]),
        },
    )


def test_three_engines_run_together_and_equal_single_engine_dispatch(tmp_path):
    root, document, engines = _fixture(tmp_path)
    result = build_transport_registry(
        document, root, unit_rule=UNIT_RULE, engines_by_binding=engines
    )
    result.codecs.register_bytes("raw-bytes-v1", load_raw_bytes)
    result.kernels.register(_Population())
    population_path = tmp_path / "population.json"
    population_path.write_text(
        json.dumps(
            {
                "person": {
                    "person_id": [1, 2, 3],
                    "person_household_id": [10, 10, 20],
                    "person_family_id": [100, 100, 200],
                    "age": [20, 5, 70],
                },
                "household": {"household_id": [10, 20]},
                "family": {"family_id": [100, 200], "cost": [12.5, 32.0]},
                "weights": [1.5, 2.5],
            }
        )
    )
    population = Node(
        "toy.create",
        _Population.ref,
        sources=("population",),
        structural=StructuralDelta.CREATE,
        outputs=(Owned("person", "age", "int64"), Owned("family", "cost", "float64")),
    )
    sources = (
        SourceRef("population", "raw-bytes-v1"),
        SourceRef("rulespec_nz", RULESPEC_SOURCE_CODEC),
    )
    nodes = tuple(
        _node(row, result.engine_refs[row["id"]]) for row in document["bindings"]
    )
    graph = Graph("toy.registry", sources, (population, *nodes))
    store = ContentStore(tmp_path / "combined", codecs=result.codecs)
    manifest = run_graph(
        compile_graph(graph),
        sources={"population": population_path, "rulespec_nz": root},
        store=store,
        kernels=result.kernels,
        resume="forbid",
    )
    assert len(manifest.nodes) == 4
    assert all(not node.hit for node in manifest.nodes.values())
    warm = run_graph(
        compile_graph(graph),
        sources={"population": population_path, "rulespec_nz": root},
        store=store,
        kernels=result.kernels,
        resume="require",
    )
    assert all(node.hit for node in warm.nodes.values())
    for row in document["bindings"]:
        ref = result.engine_refs[row["id"]]
        registry = KernelRegistry()
        registry.register(_Population())
        registry.register(SimulateRulesKernel(ref, engines[row["id"]]))
        direct_node = _node(row, ref, SimulateRulesKernel.ref)
        direct_graph = Graph("toy.registry.direct", sources, (population, direct_node))
        direct_store = ContentStore(
            tmp_path / f"direct-{row['id']}", codecs=result.codecs
        )
        direct = run_graph(
            compile_graph(direct_graph),
            sources={"population": population_path, "rulespec_nz": root},
            store=direct_store,
            kernels=registry,
            resume="forbid",
        )
        for variable in row["variables"]:
            coordinate = (row["entity"], variable)
            combined = store.load_column(
                manifest.nodes[direct_node.id].artifacts[coordinate]
            )
            single = direct_store.load_column(
                direct.nodes[direct_node.id].artifacts[coordinate]
            )
            assert combined.to_numpy().tobytes() == single.to_numpy().tobytes()
            assert combined.dtype == single.dtype == np.dtype("float64")
        assert dict(manifest.nodes[direct_node.id].receipt) == dict(
            direct.nodes[direct_node.id].receipt
        )
