"""``simulate.rules_by_ref@1``: one kernel, several bound rules engines.

The kernel must behave, node for node, exactly like ``simulate.rules@1``
bound to the engine the node names. These tests use two pure-Python engines
on the New Zealand schema, one with person outputs and one with family
outputs, and check that property three ways: directly on kernel contexts
(Hypothesis over random populations), through one ``run_graph`` that reaches
both engines, and against two single-engine ``simulate.rules@1`` graph runs.
"""

import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import microcosm.frame.bundle as frame_bundle_module
import microcosm.frame.rules as frame_rules_module
import microcosm.frame.schema as frame_schema_module
from microcosm.frame import (
    EntitySchema,
    ExportContract,
    Frame,
    VariableMetadata,
    WeightKind,
    Weights,
)
from microcosm.frame.adapters.axiom import NZ_SCHEMA, AxiomEngine
from microcosm.frame.kernels import SimulateRulesKernel
from microcosm.frame.rules_kernels import SimulateRulesByRefKernel
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    Kernel,
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
    source_hash,
)
from test_support.microcosm_frame.kernels import _context, _StubRulesEngine
from test_support.paths import paths_for

_RULESPEC_ROOT = paths_for("microcosm-frame").tests / "fixtures" / "rulespec-zz"

PERSON_REF = "toy-person-engine"
FAMILY_REF = "toy-family-engine"
PERSON_VARIABLES = ("person_net_income", "person_is_adult")
FAMILY_VARIABLES = ("family_support", "family_code")


class _PersonEngine:
    """Pure-Python engine with two person outputs."""

    def variable_metadata(self, name: str) -> VariableMetadata:
        metadata = {
            "person_net_income": VariableMetadata(
                name="person_net_income", entity="person", dtype="float", period="year"
            ),
            "person_is_adult": VariableMetadata(
                name="person_is_adult", entity="person", dtype="bool", period="year"
            ),
        }
        try:
            return metadata[name]
        except KeyError as error:
            raise ValueError(f"Unknown person variable {name!r}.") from error

    def variables(self) -> Sequence[str]:
        return ("earnings", "age")

    def entity_schema(self) -> EntitySchema:
        return NZ_SCHEMA

    def materialize(
        self, bundle: Frame, variables: Sequence[str], period: int | str
    ) -> Mapping[str, np.ndarray]:
        year = int(str(period)[:4])
        person = bundle.table("person")
        earnings = person["earnings"].to_numpy(dtype=np.float64)
        age = person["age"].to_numpy(dtype=np.int64)
        available = {
            "person_net_income": earnings * np.float64(0.8) + np.float64(year - 2025),
            "person_is_adult": age >= 18,
        }
        return {name: available[name] for name in variables}

    def export_contract(self) -> ExportContract:
        return ExportContract.empty()

    def write_dataset(self, bundle: Frame, path: str | Path, period: int | str):
        return None


class _FamilyEngine:
    """Pure-Python engine with two family outputs, one tri-state code."""

    def variable_metadata(self, name: str) -> VariableMetadata:
        metadata = {
            "family_support": VariableMetadata(
                name="family_support", entity="family", dtype="float", period="year"
            ),
            "family_code": VariableMetadata(
                name="family_code", entity="family", dtype="int", period="year"
            ),
        }
        try:
            return metadata[name]
        except KeyError as error:
            raise ValueError(f"Unknown family variable {name!r}.") from error

    def variables(self) -> Sequence[str]:
        return ("family_rent",)

    def entity_schema(self) -> EntitySchema:
        return NZ_SCHEMA

    def materialize(
        self, bundle: Frame, variables: Sequence[str], period: int | str
    ) -> Mapping[str, np.ndarray]:
        rent = bundle.table("family")["family_rent"].to_numpy(dtype=np.float64)
        available = {
            "family_support": np.maximum(
                (rent - np.float64(100.0)) * np.float64(0.7), np.float64(0.0)
            ),
            "family_code": np.sign(rent - np.float64(250.0)).astype(np.int64),
        }
        return {name: available[name] for name in variables}

    def export_contract(self) -> ExportContract:
        return ExportContract.empty()

    def write_dataset(self, bundle: Frame, path: str | Path, period: int | str):
        return None


def _person_node(kernel_ref: str, engine_ref: str = PERSON_REF) -> Node:
    return Node(
        "rules.person",
        kernel_ref,
        inputs=(Slice("person", ("earnings", "age")),),
        outputs=(
            Owned("person", "person_net_income", "float64"),
            Owned("person", "person_is_adult", "bool"),
        ),
        params={
            "engine_ref": engine_ref,
            "variables": PERSON_VARIABLES,
            "period": 2026,
        },
    )


def _family_node(kernel_ref: str, engine_ref: str = FAMILY_REF) -> Node:
    # A family-only engine node also slices one person data column: the
    # kernel context holds only declared entities, and the kernel rebuilds
    # group tables from the person memberships.
    return Node(
        "rules.family",
        kernel_ref,
        inputs=(Slice("family", ("family_rent",)), Slice("person", ("age",))),
        outputs=(
            Owned("family", "family_support", "float64"),
            Owned("family", "family_code", "int64"),
        ),
        params={
            "engine_ref": engine_ref,
            "variables": FAMILY_VARIABLES,
            "period": "2026-27",
        },
    )


def _by_ref(**extra) -> SimulateRulesByRefKernel:
    return SimulateRulesByRefKernel(
        {PERSON_REF: _PersonEngine(), FAMILY_REF: _FamilyEngine(), **extra}
    )


@st.composite
def nz_frames(draw: st.DrawFn) -> Frame:
    """A nested NZ frame with person and family inputs and household weights."""

    person_households: list[int] = []
    person_families: list[int] = []
    rents: list[float] = []
    family = 100
    household_count = draw(st.integers(min_value=1, max_value=5))
    for household in range(1, household_count + 1):
        for _ in range(draw(st.integers(min_value=1, max_value=3))):
            family += 1
            rents.append(
                draw(st.floats(min_value=0.0, max_value=900.0, allow_nan=False))
            )
            for _ in range(draw(st.integers(min_value=1, max_value=3))):
                person_households.append(household)
                person_families.append(family)
    count = len(person_households)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, count + 1, dtype=np.int64) * 7,
            "person_household_id": np.asarray(person_households, dtype=np.int64),
            "person_family_id": np.asarray(person_families, dtype=np.int64),
            "earnings": np.asarray(
                draw(
                    st.lists(
                        st.floats(min_value=0.0, max_value=2e5, allow_nan=False),
                        min_size=count,
                        max_size=count,
                    )
                ),
                dtype=np.float64,
            ),
            "age": np.asarray(
                draw(
                    st.lists(
                        st.integers(min_value=0, max_value=99),
                        min_size=count,
                        max_size=count,
                    )
                ),
                dtype=np.int64,
            ),
        }
    )
    household = pd.DataFrame(
        {"household_id": np.arange(1, household_count + 1, dtype=np.int64)}
    )
    families = pd.DataFrame(
        {
            "family_id": np.arange(101, family + 1, dtype=np.int64),
            "family_rent": np.asarray(rents, dtype=np.float64),
        }
    )
    weights = draw(
        st.lists(
            st.floats(min_value=1.0, max_value=5e3, allow_nan=False),
            min_size=household_count,
            max_size=household_count,
        )
    )
    return Frame(
        {"person": person, "household": household, "family": families},
        NZ_SCHEMA,
        {"household": Weights(np.asarray(weights), WeightKind.DESIGN)},
    )


def _assert_same_result(actual: KernelResult, expected: KernelResult) -> None:
    assert set(actual.columns) == set(expected.columns)
    for coordinate, series in expected.columns.items():
        observed = actual.columns[coordinate]
        assert observed.dtype == series.dtype
        assert observed.index.equals(series.index)
        assert observed.index.name == series.index.name
        assert observed.to_numpy().tobytes() == series.to_numpy().tobytes()
    assert dict(actual.receipt) == dict(expected.receipt)


# ----------------------------------------------------------------------
# Construction and identity
# ----------------------------------------------------------------------


class TestConstruction:
    def test_satisfies_the_kernel_protocol_with_simulate_rules_capabilities(
        self,
    ) -> None:
        kernel = _by_ref()
        assert isinstance(kernel, Kernel)
        assert kernel.ref == "simulate.rules_by_ref@1"
        assert (
            kernel.capabilities
            == SimulateRulesKernel(PERSON_REF, _PersonEngine()).capabilities
        )
        assert kernel.capabilities.determinism is Determinism.DETERMINISTIC
        assert kernel.capabilities.numeric is Numeric.BITWISE
        assert kernel.capabilities.structural is StructuralDelta.NONE
        assert kernel.engine_refs() == (FAMILY_REF, PERSON_REF)

    def test_dependencies_reach_the_capabilities(self) -> None:
        kernel = SimulateRulesByRefKernel(
            {PERSON_REF: _PersonEngine()}, dependencies=("numpy",)
        )
        assert kernel.capabilities.dependencies == ("numpy",)

    @pytest.mark.parametrize(
        ("engines", "error", "message"),
        [
            ([(PERSON_REF, _PersonEngine())], TypeError, "mapping"),
            ({}, ValueError, "at least one"),
            ({"": _PersonEngine()}, ValueError, "engine_ref"),
            ({PERSON_REF: object()}, TypeError, "RulesEngine"),
        ],
    )
    def test_invalid_bindings_are_refused(self, engines, error, message) -> None:
        with pytest.raises(error, match=message):
            SimulateRulesByRefKernel(engines)

    def test_implementation_hash_binds_both_kernels_and_each_adapter_class(
        self,
    ) -> None:
        kernel = _by_ref()
        assert kernel.implementation_hash() == source_hash(
            SimulateRulesByRefKernel,
            SimulateRulesKernel,
            _FamilyEngine,
            _PersonEngine,
            frame_bundle_module,
            frame_rules_module,
            frame_schema_module,
        )
        assert (
            kernel.implementation_hash()
            != SimulateRulesKernel(PERSON_REF, _PersonEngine()).implementation_hash()
        )

    def test_adding_a_binding_of_a_bound_class_moves_no_hash(self) -> None:
        base = _by_ref()
        extended = _by_ref(**{"another-person-engine": _PersonEngine()})
        reordered = SimulateRulesByRefKernel(
            {FAMILY_REF: _FamilyEngine(), PERSON_REF: _PersonEngine()}
        )
        assert extended.implementation_hash() == base.implementation_hash()
        assert reordered.implementation_hash() == base.implementation_hash()
        assert extended.capabilities == base.capabilities

    def test_the_hash_is_independent_of_binding_order(self) -> None:
        """Invariant: the hash never depends on the order bindings arrive in.

        Three adapter classes from three modules make the source order
        observable: all six binding orders must give the hash of the classes
        in ``(module, qualname)`` order. These three classes sort the same way
        by qualname alone, so the test pins order independence, not the
        choice of sort key.
        """

        bindings = {
            "axiom": AxiomEngine(
                _RULESPEC_ROOT / "zz/policies/tests/axiom_toy_family.yaml",
                schema=NZ_SCHEMA,
                rulespec_roots=(_RULESPEC_ROOT,),
            ),
            "stub": _StubRulesEngine(),
            "person": _PersonEngine(),
        }
        classes = sorted(
            {type(engine) for engine in bindings.values()},
            key=lambda cls: (cls.__module__, cls.__qualname__),
        )
        assert len({cls.__module__ for cls in classes}) == 3
        expected = source_hash(
            SimulateRulesByRefKernel,
            SimulateRulesKernel,
            *classes,
            frame_bundle_module,
            frame_rules_module,
            frame_schema_module,
        )
        orders = list(itertools.permutations(bindings))
        assert len(orders) == 6
        for order in orders:
            kernel = SimulateRulesByRefKernel({ref: bindings[ref] for ref in order})
            assert kernel.implementation_hash() == expected, order

    def test_binding_a_new_adapter_class_moves_the_hash(self) -> None:
        # Documented: a new class's source joins the hash.
        assert (
            _by_ref(stub=_StubRulesEngine()).implementation_hash()
            != _by_ref().implementation_hash()
        )


# ----------------------------------------------------------------------
# Routing on kernel contexts
# ----------------------------------------------------------------------


class TestRouting:
    def test_an_unknown_engine_ref_is_refused(self) -> None:
        frame = _population_frame()
        node = _person_node(SimulateRulesByRefKernel.ref, engine_ref="unbound")
        context = _context(frame, node, weighted_entities=("household",))
        with pytest.raises(
            ValueError,
            match=r"no engine bound to engine_ref sha256:[0-9a-f]{16} \('unbound'\); "
            r"bound: sha256:",
        ):
            _by_ref().run(context)

    def test_an_unknown_axiom_style_ref_is_named_by_digest_and_module(self) -> None:
        # Canonical JSON references share their leading keys, so the message
        # names a digest and the module rather than a truncated prefix.
        bound = json.dumps({"engine_commit": "a" * 40, "module": "nz/as/core.yaml"})
        unbound = json.dumps({"engine_commit": "a" * 40, "module": "nz/mb/rates.yaml"})
        frame = _population_frame()
        node = _person_node(SimulateRulesByRefKernel.ref, engine_ref=unbound)
        context = _context(frame, node, weighted_entities=("household",))
        with pytest.raises(
            ValueError,
            match=r"engine_ref sha256:[0-9a-f]{16} \(module 'nz/mb/rates\.yaml'\); "
            r"bound: sha256:[0-9a-f]{16} \(module 'nz/as/core\.yaml'\)",
        ):
            SimulateRulesByRefKernel({bound: _PersonEngine()}).run(context)

    @pytest.mark.parametrize(
        ("change", "error"),
        [
            ({"drop": "engine_ref"}, ValueError),
            ({"drop": "period"}, ValueError),
            ({"set": ("extra", 1)}, ValueError),
            ({"set": ("engine_ref", "")}, TypeError),
            ({"set": ("engine_ref", 7)}, TypeError),
            ({"set": ("variables", ["person_net_income"])}, TypeError),
            ({"set": ("period", True)}, TypeError),
        ],
    )
    def test_differential_malformed_params_fail_as_under_simulate_rules(
        self, change, error
    ) -> None:
        """A malformed node fails with the same exception type either way."""

        frame = _population_frame()
        context = _context(
            frame,
            _person_node(SimulateRulesByRefKernel.ref),
            weighted_entities=("household",),
        )
        params = dict(context.params)
        if "drop" in change:
            params.pop(change["drop"])
        else:
            key, value = change["set"]
            params[key] = value
        context = KernelContext(
            node=context.node,
            tables=context.tables,
            weights=context.weights,
            strata=context.strata,
            params=params,
            rng=context.rng,
        )
        with pytest.raises(error):
            SimulateRulesKernel(PERSON_REF, _PersonEngine()).run(context)
        with pytest.raises(error):
            _by_ref().run(context)

    def test_the_delegate_parameter_contract_still_applies(self) -> None:
        frame = _population_frame()
        node = _person_node(SimulateRulesByRefKernel.ref)
        context = _context(frame, node, weighted_entities=("household",))
        context = KernelContext(
            node=context.node,
            tables=context.tables,
            weights=context.weights,
            strata=context.strata,
            params={**context.params, "extra": 1},
            rng=context.rng,
        )
        with pytest.raises(ValueError, match="parameters must be exactly"):
            _by_ref().run(context)

    @settings(max_examples=60, deadline=None)
    @given(frame=nz_frames())
    def test_differential_each_node_equals_its_single_engine_kernel(
        self, frame
    ) -> None:
        """Two implementations of one semantics: routed and direct agree."""

        kernel = _by_ref()
        for node, single in (
            (
                _person_node(SimulateRulesByRefKernel.ref),
                SimulateRulesKernel(PERSON_REF, _PersonEngine()),
            ),
            (
                _family_node(SimulateRulesByRefKernel.ref),
                SimulateRulesKernel(FAMILY_REF, _FamilyEngine()),
            ),
        ):
            context = _context(
                frame, node, weighted_entities=("person", "household", "family")
            )
            _assert_same_result(kernel.run(context), single.run(context))


# ----------------------------------------------------------------------
# The graph
# ----------------------------------------------------------------------


class _JsonPopulation(KernelBase):
    """CREATE kernel loading an NZ population from a JSON source."""

    ref = "test.json_population@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.CREATE,
    )

    def run(self, context: KernelContext) -> KernelResult:
        document = json.loads(context.sources["population"].read_text())
        frame = _frame_from_document(document)
        return KernelResult(frame=frame, receipt={"persons": frame.n("person")})


def _frame_from_document(document: Mapping[str, object]) -> Frame:
    person = pd.DataFrame(document["person"]).astype(
        {
            "person_id": "int64",
            "person_household_id": "int64",
            "person_family_id": "int64",
            "earnings": "float64",
            "age": "int64",
        }
    )
    household = pd.DataFrame(document["household"]).astype({"household_id": "int64"})
    family = pd.DataFrame(document["family"]).astype(
        {"family_id": "int64", "family_rent": "float64"}
    )
    return Frame(
        {"person": person, "household": household, "family": family},
        NZ_SCHEMA,
        {
            "household": Weights(
                np.asarray(document["household_weights"], dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _population_frame() -> Frame:
    return _frame_from_document(_POPULATION)


_POPULATION = {
    "person": {
        "person_id": [1, 2, 3, 4, 5, 6],
        "person_household_id": [1, 1, 2, 2, 2, 3],
        "person_family_id": [10, 10, 20, 21, 21, 30],
        "earnings": [52_000.0, 0.0, 31_500.0, 18_250.0, 0.0, 74_000.0],
        "age": [41, 9, 67, 35, 4, 23],
    },
    "household": {"household_id": [1, 2, 3]},
    "family": {
        "family_id": [10, 20, 21, 30],
        "family_rent": [420.0, 0.0, 250.0, 180.0],
    },
    "household_weights": [310.0, 145.5, 890.25],
}

_POPULATION_NODE = Node(
    "population",
    _JsonPopulation.ref,
    outputs=(
        Owned("person", "earnings", "float64"),
        Owned("person", "age", "int64"),
        Owned("family", "family_rent", "float64"),
    ),
    structural=StructuralDelta.CREATE,
    sources=("population",),
)


def _run(root: Path, nodes: tuple[Node, ...], kernels: tuple[object, ...]):
    source = root / "population.json"
    root.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(_POPULATION))
    registry = KernelRegistry()
    registry.register(_JsonPopulation())
    for kernel in kernels:
        registry.register(kernel)
    graph = Graph(
        "rules-by-ref",
        (SourceRef("population", "raw-bytes-v1"),),
        (_POPULATION_NODE, *nodes),
    )
    store = ContentStore(root / "store")
    manifest = run_graph(
        compile_graph(graph),
        sources={"population": source},
        store=store,
        kernels=registry,
        resume="forbid",
        decisions=(),
    )
    return manifest, store


def _load(manifest, store, node_id: str, entity: str, column: str) -> pd.Series:
    return store.load_column(manifest.nodes[node_id].artifacts[(entity, column)])


class TestTwoEngineGraph:
    def test_two_engines_compile_and_run_in_one_run_graph(self, tmp_path) -> None:
        kernel = _by_ref()
        manifest, store = _run(
            tmp_path / "by-ref",
            (
                _person_node(kernel.ref),
                _family_node(kernel.ref),
            ),
            (kernel,),
        )
        np.testing.assert_allclose(
            _load(manifest, store, "rules.person", "person", "person_net_income"),
            [41_601.0, 1.0, 25_201.0, 14_601.0, 1.0, 59_201.0],
        )
        assert _load(
            manifest, store, "rules.person", "person", "person_is_adult"
        ).tolist() == [True, False, True, True, False, True]
        np.testing.assert_allclose(
            _load(manifest, store, "rules.family", "family", "family_support"),
            [224.0, 0.0, 105.0, 56.0],
        )
        codes = _load(manifest, store, "rules.family", "family", "family_code")
        assert codes.dtype == np.dtype(np.int64)
        assert codes.tolist() == [1, -1, 0, -1]

    def test_differential_outputs_equal_two_single_engine_runs(self, tmp_path) -> None:
        kernel = _by_ref()
        manifest, store = _run(
            tmp_path / "by-ref",
            (_person_node(kernel.ref), _family_node(kernel.ref)),
            (kernel,),
        )
        for node_id, node, single, variables in (
            (
                "rules.person",
                _person_node(SimulateRulesKernel.ref),
                SimulateRulesKernel(PERSON_REF, _PersonEngine()),
                (("person", name) for name in PERSON_VARIABLES),
            ),
            (
                "rules.family",
                _family_node(SimulateRulesKernel.ref),
                SimulateRulesKernel(FAMILY_REF, _FamilyEngine()),
                (("family", name) for name in FAMILY_VARIABLES),
            ),
        ):
            single_manifest, single_store = _run(
                tmp_path / f"single-{node_id}", (node,), (single,)
            )
            assert (
                single_manifest.nodes[node_id].receipt
                == manifest.nodes[node_id].receipt
            )
            for entity, column in variables:
                expected = _load(single_manifest, single_store, node_id, entity, column)
                actual = _load(manifest, store, node_id, entity, column)
                pd.testing.assert_series_equal(actual, expected)
                assert actual.to_numpy().tobytes() == expected.to_numpy().tobytes()

    def test_adding_a_binding_re_keys_no_existing_node(self, tmp_path) -> None:
        kernel = _by_ref()
        manifest, _ = _run(
            tmp_path / "two",
            (_person_node(kernel.ref), _family_node(kernel.ref)),
            (kernel,),
        )
        extended = _by_ref(**{"third-engine": _PersonEngine()})
        extended_manifest, _ = _run(
            tmp_path / "three",
            (_person_node(extended.ref), _family_node(extended.ref)),
            (extended,),
        )
        for node_id in ("population", "rules.person", "rules.family"):
            assert extended_manifest.nodes[node_id].key == manifest.nodes[node_id].key

    def test_each_node_key_binds_its_own_engine_ref(self, tmp_path) -> None:
        kernel = _by_ref(**{"renamed-person-engine": _PersonEngine()})
        manifest, _ = _run(tmp_path / "base", (_person_node(kernel.ref),), (kernel,))
        renamed, _ = _run(
            tmp_path / "renamed",
            (_person_node(kernel.ref, engine_ref="renamed-person-engine"),),
            (kernel,),
        )
        assert renamed.nodes["rules.person"].key != manifest.nodes["rules.person"].key
        assert renamed.nodes["population"].key == manifest.nodes["population"].key

    def test_a_node_naming_an_unbound_engine_rejects_the_run(self, tmp_path) -> None:
        kernel = _by_ref()
        with pytest.raises(Exception, match="no engine bound"):
            _run(
                tmp_path / "unbound",
                (_person_node(kernel.ref, engine_ref="unbound"),),
                (kernel,),
            )
