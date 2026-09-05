"""The PUF growth contract running as real Microcosm graph nodes.

The population is three invented returns written to a temporary ``csv-tables``
source; the factor levels are invented and the table says so. Nothing here
reads a PUF, downloads anything, or invokes a country-model calculation.
Importing the package can initialize the ABI when US engine extras are installed.
"""

import ast
import inspect
import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf_growth_graph
from microcosm.build.us_runtime.puf_growth import (
    PROVENANCE_RECID_COLUMN,
    PROVENANCE_SOURCE_AGI_COLUMN,
    CompiledPufGrowth,
    FieldRole,
    GrowthFactorTable,
    GrowthKind,
    GrowthRecipe,
    GrowthRule,
    PufGrowthRefusalError,
    SignBranch,
    apply_puf_design_weight_growth,
    apply_puf_growth,
    compile_puf_growth,
)
from microcosm.build.us_runtime.puf_growth_graph import (
    PUF_GROWTH_BOUNDARY_NODE,
    PUF_GROWTH_DESIGN_WEIGHT_NODE,
    PUF_GROWTH_MONEY_NODE,
    PufGrowthBoundaryKernel,
    PufGrowthDesignWeightKernel,
    PufGrowthMoneyKernel,
    register_us_puf_growth_kernels,
    us_puf_growth_nodes,
)
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
    source_hash,
)
from microcosm.graph.codecs import load_csv_tables
from microcosm.graph.errors import NodeRejectedError, StoreMissError

SOURCE_YEAR = 2021
TARGET_YEAR = 2024
ENTITY = "tax_unit"

_SERIES = {
    "fixture.agi_per_return": {
        "kind": "real_per_return_aging",
        "levels": {"2021": "1.0", "2024": "1.25"},
    },
    "fixture.business_losses_per_return": {
        "kind": "real_per_return_aging",
        "levels": {"2021": "1.0", "2024": "0.8"},
    },
    "fixture.business_profits_per_return": {
        "kind": "real_per_return_aging",
        "levels": {"2021": "1.0", "2024": "1.5"},
    },
    "fixture.price_index": {
        "kind": "nominal_price_restatement",
        "levels": {"2021": "2.0", "2024": "3.0"},
    },
    "fixture.returns": {
        "kind": "design_weight_growth",
        "levels": {"2021": "100.0", "2024": "110.0"},
    },
}

_SOURCE_COLUMNS = ("RECID", "MARS", "XTOT", "S006", "E00100", "E00200", "E00900")

_CREATE_OUTPUTS = (
    Owned("person", "is_primary_filer", "bool"),
    Owned(ENTITY, "RECID", "int64"),
    Owned(ENTITY, "MARS", "int64"),
    Owned(ENTITY, "XTOT", "int64"),
    Owned(ENTITY, "S006", "float64"),
    Owned(ENTITY, "E00100", "float64"),
    Owned(ENTITY, "E00200", "float64"),
    Owned(ENTITY, "E00900", "float64"),
)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    ).encode("ascii")


def factor_table(series=None):
    return GrowthFactorTable.from_bytes(
        canonical(
            {
                "authority": "invented_fixture_nonauthority",
                "provenance": {"note": "invented fixture levels; no IRS or SOI value"},
                "schema_version": 1,
                "series": _SERIES if series is None else series,
                "table_id": "puf_growth_fixture_v1",
                "years": [SOURCE_YEAR, TARGET_YEAR],
            }
        )
    )


def growth_recipe():
    def money(
        field, series, branch=SignBranch.ANY, kind=GrowthKind.REAL_PER_RETURN_AGING
    ):
        return GrowthRule(
            field, FieldRole.MONEY, SOURCE_YEAR, TARGET_YEAR, kind, branch, series
        )

    return GrowthRecipe(
        "puf_growth_fixture_recipe_v1",
        (
            GrowthRule(
                "RECID", FieldRole.IDENTIFIER, SOURCE_YEAR, SOURCE_YEAR, GrowthKind.NONE
            ),
            GrowthRule(
                "MARS", FieldRole.CODE, SOURCE_YEAR, SOURCE_YEAR, GrowthKind.NONE
            ),
            GrowthRule(
                "XTOT", FieldRole.COUNT, SOURCE_YEAR, SOURCE_YEAR, GrowthKind.NONE
            ),
            GrowthRule(
                "S006",
                FieldRole.DESIGN_WEIGHT,
                SOURCE_YEAR,
                TARGET_YEAR,
                GrowthKind.DESIGN_WEIGHT_GROWTH,
                factor_series="fixture.returns",
            ),
            money("E00100", "fixture.agi_per_return"),
            money(
                "E00200",
                "fixture.price_index",
                kind=GrowthKind.NOMINAL_PRICE_RESTATEMENT,
            ),
            money("E00900", "fixture.business_profits_per_return", SignBranch.POSITIVE),
            money("E00900", "fixture.business_losses_per_return", SignBranch.NEGATIVE),
        ),
    )


def raw_tax_units():
    return pd.DataFrame(
        {
            "RECID": np.array([11, 22, 33], dtype="int64"),
            "MARS": np.array([1, 2, 4], dtype="int64"),
            "XTOT": np.array([1, 3, 2], dtype="int64"),
            "S006": np.array([100.0, 200.0, 300.0]),
            "E00100": np.array([1000.0, -500.0, 0.0]),
            "E00200": np.array([100.0, 0.0, -25.0]),
            "E00900": np.array([500.0, -200.0, 0.0]),
        }
    )


def write_source(directory: Path, *, recid_dtype: str = "int64") -> Path:
    """One invented ``csv-tables`` source: three returns, one filer each."""
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_tax_unit_id": [10, 20, 30],
            "is_primary_filer": [True, True, True],
        }
    ).to_csv(directory / "person.csv", index=False)
    tax_units = raw_tax_units()
    tax_units.insert(0, "tax_unit_id", [10, 20, 30])
    tax_units.to_csv(directory / "tax_unit.csv", index=False)
    pd.DataFrame(
        {"tax_unit_id": [10, 20, 30], "design_weight": [1.0, 2.0, 3.0]}
    ).to_csv(directory / "weights.csv", index=False)
    (directory / "schema.json").write_text(
        json.dumps(
            {
                "person_entity": "person",
                "group_entities": [ENTITY],
                "tables": {"person": "person.csv", ENTITY: "tax_unit.csv"},
                "dtypes": {
                    "person_id": "int64",
                    "person_tax_unit_id": "int64",
                    "is_primary_filer": "bool",
                    "tax_unit_id": "int64",
                    "RECID": recid_dtype,
                    "MARS": "int64",
                    "XTOT": "int64",
                    "S006": "float64",
                    "E00100": "float64",
                    "E00200": "float64",
                    "E00900": "float64",
                    "design_weight": "float64",
                },
                "weights": {
                    "entity": ENTITY,
                    "column": "design_weight",
                    "kind": "design",
                },
                "weights_table": "weights.csv",
            }
        )
    )
    return directory


class FixtureSourceKernel(KernelBase):
    """CREATE: the invented raw-source tables, through the shared codec."""

    ref = "us.puf.growth.fixture_source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        numeric=Numeric.BITWISE,
    )

    def run(self, context):
        frame = load_csv_tables(context.sources["puf_raw"])
        return KernelResult(frame=frame, receipt={"returns": frame.n(ENTITY)})


def build(contract, *, recid_dtype: str = "int64"):
    graph = Graph(
        country="us",
        sources=(SourceRef("puf_raw", "csv-tables"),),
        nodes=(
            Node(
                "source",
                FixtureSourceKernel.ref,
                sources=("puf_raw",),
                structural=StructuralDelta.CREATE,
                outputs=tuple(
                    Owned(o.entity, o.column, recid_dtype) if o.column == "RECID" else o
                    for o in _CREATE_OUTPUTS
                ),
            ),
            *us_puf_growth_nodes(
                contract,
                base="source",
                entity=ENTITY,
                person_boundary_columns=("is_primary_filer",),
            ),
        ),
    )
    registry = KernelRegistry()
    registry.register(FixtureSourceKernel())
    register_us_puf_growth_kernels(registry)
    return compile_graph(graph), registry


def expected_frame():
    """The parity oracle: the same contract applied by two direct calls."""
    contract = compile_puf_growth(growth_recipe(), factor_table())
    raw = raw_tax_units()
    grown = apply_puf_growth(raw.drop(columns=["S006"]), contract).table
    grown["S006"] = apply_puf_design_weight_growth(raw.S006, contract)
    return grown


@pytest.fixture
def prepared(tmp_path):
    contract = compile_puf_growth(growth_recipe(), factor_table())
    compiled_graph, registry = build(contract)
    return (
        compiled_graph,
        {"puf_raw": write_source(tmp_path / "source")},
        ContentStore(tmp_path / "store"),
        registry,
    )


def test_actual_graph_cold_warm_required_match_the_direct_transform(prepared, tmp_path):
    compiled_graph, sources, store, registry = prepared
    expected = expected_frame()
    artifacts = None
    for resume in ("forbid", "auto", "require"):
        manifest = run_graph(
            compiled_graph,
            sources=sources,
            store=store,
            kernels=registry,
            resume=resume,
        )
        frame = manifest.population(PUF_GROWTH_BOUNDARY_NODE)
        table = frame.table(ENTITY)
        for column in expected.columns:
            assert (
                table[column].to_numpy().tobytes()
                == expected[column].to_numpy().tobytes()
            ), column
        # Identifiers and the entity axis are the graph's, not the kernel's.
        assert table.tax_unit_id.tolist() == [10, 20, 30]
        # This stage grows the raw design-weight column only; the frame's typed
        # design weights are still exactly what the source loaded.
        assert frame.weights_for(ENTITY).values.tolist() == [1.0, 2.0, 3.0]
        current = {
            node_id: dict(manifest.nodes[node_id].artifacts)
            for node_id in compiled_graph.order
        }
        if artifacts is None:
            artifacts = current
            assert set(manifest.nodes) == set(compiled_graph.order)
        else:
            assert current == artifacts
        if resume != "forbid":
            assert all(node.hit for node in manifest.nodes.values())

    # A fresh store recomputes everything and lands on the same artifacts.
    fresh = run_graph(
        compiled_graph,
        sources=sources,
        store=ContentStore(tmp_path / "fresh"),
        kernels=registry,
        resume="auto",
    )
    assert not any(node.hit for node in fresh.nodes.values())
    assert {
        node_id: dict(fresh.nodes[node_id].artifacts)
        for node_id in compiled_graph.order
    } == artifacts
    fresh_table = fresh.population(PUF_GROWTH_BOUNDARY_NODE).table(ENTITY)
    for column in expected.columns:
        assert (
            fresh_table[column].to_numpy().tobytes()
            == expected[column].to_numpy().tobytes()
        )


def test_public_compiled_constructor_and_graph_use_the_same_table(tmp_path):
    compiled = compile_puf_growth(growth_recipe(), factor_table())
    public = CompiledPufGrowth(
        compiled.recipe, compiled.factors, compiled.resolved, compiled.identity
    )
    compiled_graph, registry = build(public)
    sources = {"puf_raw": write_source(tmp_path / "source")}
    expected = apply_puf_growth(raw_tax_units().drop(columns="S006"), public).table
    manifest = run_graph(
        compiled_graph,
        sources=sources,
        store=ContentStore(tmp_path / "store"),
        kernels=registry,
        resume="forbid",
    )
    actual = manifest.population(PUF_GROWTH_BOUNDARY_NODE).table(ENTITY)
    for column in expected.columns:
        assert (
            actual[column].to_numpy().tobytes() == expected[column].to_numpy().tobytes()
        )
    # A constructor with new valid levels and a recomputed identity cannot
    # create a second direct result under the same attached factor table.
    changed = tuple(
        replace(factor, target_level="7.0", ratio_bits=np.float64(7).tobytes().hex())
        if factor.field == "E00100"
        else factor
        for factor in public.resolved
    )
    identity = json.loads(public.identity)
    identity["resolved"] = [factor.as_document() for factor in changed]
    with pytest.raises(PufGrowthRefusalError, match="RESOLVED_FACTOR_BINDING"):
        CompiledPufGrowth(public.recipe, public.factors, changed, canonical(identity))


@pytest.mark.parametrize("dependency", ["numpy", "pandas"])
def test_dependency_version_change_refuses_old_warm_cache(
    prepared, monkeypatch, dependency
):
    from microcosm.graph import kernel as kernel_module

    compiled_graph, sources, store, registry = prepared
    first = run_graph(
        compiled_graph, sources=sources, store=store, kernels=registry, resume="forbid"
    )
    warm = run_graph(
        compiled_graph, sources=sources, store=store, kernels=registry, resume="require"
    )
    assert all(node.hit for node in warm.nodes.values())
    classes = (
        PufGrowthBoundaryKernel,
        PufGrowthMoneyKernel,
        PufGrowthDesignWeightKernel,
    )
    original_hashes = {cls: cls().implementation_hash() for cls in classes}
    version = kernel_module.importlib_metadata.version
    monkeypatch.setattr(
        kernel_module.importlib_metadata,
        "version",
        lambda name: (
            version(name) + ".changed" if name == dependency else version(name)
        ),
    )
    assert all(cls().implementation_hash() != original_hashes[cls] for cls in classes)
    with pytest.raises(StoreMissError):
        run_graph(
            compiled_graph,
            sources=sources,
            store=store,
            kernels=registry,
            resume="require",
        )
    recomputed = run_graph(
        compiled_graph, sources=sources, store=store, kernels=registry, resume="auto"
    )
    assert recomputed.nodes["source"].hit
    assert all(
        not recomputed.nodes[name].hit
        for name in (
            PUF_GROWTH_BOUNDARY_NODE,
            PUF_GROWTH_MONEY_NODE,
            PUF_GROWTH_DESIGN_WEIGHT_NODE,
        )
    )
    before = first.population(PUF_GROWTH_BOUNDARY_NODE).table(ENTITY)
    after = recomputed.population(PUF_GROWTH_BOUNDARY_NODE).table(ENTITY)
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_undeclared_raw_money_still_passes_through_pending_source_inventory(
    tmp_path, monkeypatch
):
    """Document the admitted limitation; this stage is not a full source classifier."""
    source = write_source(tmp_path / "source")
    table = pd.read_csv(source / "tax_unit.csv")
    table["E77777"] = [91.0, 92.0, 93.0]
    table.to_csv(source / "tax_unit.csv", index=False)
    schema = json.loads((source / "schema.json").read_text())
    schema["dtypes"]["E77777"] = "float64"
    (source / "schema.json").write_text(json.dumps(schema))
    monkeypatch.setitem(
        globals(),
        "_CREATE_OUTPUTS",
        (*_CREATE_OUTPUTS, Owned(ENTITY, "E77777", "float64")),
    )
    graph, registry = build(compile_puf_growth(growth_recipe(), factor_table()))
    result = run_graph(
        graph,
        sources={"puf_raw": source},
        store=ContentStore(tmp_path / "store"),
        kernels=registry,
        resume="forbid",
    )
    assert result.population(PUF_GROWTH_BOUNDARY_NODE).table(
        ENTITY
    ).E77777.tolist() == [91.0, 92.0, 93.0]


def test_the_graph_carries_provenance_and_refuses_release_eligibility(prepared):
    compiled_graph, sources, store, registry = prepared
    manifest = run_graph(
        compiled_graph, sources=sources, store=store, kernels=registry, resume="auto"
    )
    table = manifest.population(PUF_GROWTH_BOUNDARY_NODE).table(ENTITY)
    raw = raw_tax_units()
    assert table[PROVENANCE_RECID_COLUMN].tolist() == raw.RECID.tolist()
    assert table[PROVENANCE_RECID_COLUMN].dtype == np.dtype("int64")
    assert table[PROVENANCE_SOURCE_AGI_COLUMN].tolist() == raw.E00100.tolist()
    assert table.E00100.tolist() != raw.E00100.tolist()

    boundary = manifest.node(PUF_GROWTH_BOUNDARY_NODE).receipt["puf_growth_contract"]
    assert boundary["factor_authority"] == "invented_fixture_nonauthority"
    assert boundary["release_eligible"] is False
    assert boundary["target_year"] == TARGET_YEAR
    assert (
        manifest.node(PUF_GROWTH_MONEY_NODE).receipt["puf_growth"][
            "design_weight_in_money_view"
        ]
        is False
    )


def test_weight_growth_is_owned_by_its_own_node(prepared):
    compiled_graph, sources, store, registry = prepared
    weight_node = compiled_graph.graph.node(PUF_GROWTH_DESIGN_WEIGHT_NODE)
    assert [(o.entity, o.column) for o in weight_node.outputs] == [(ENTITY, "S006")]
    money_node = compiled_graph.graph.node(PUF_GROWTH_MONEY_NODE)
    assert "S006" not in {o.column for o in money_node.outputs}
    # The money node must not even read the weight, or the executor would make
    # the weight node its predecessor and couple their cache keys.
    assert "S006" not in {
        column for slice_ in money_node.inputs for column in slice_.columns
    }
    manifest = run_graph(
        compiled_graph, sources=sources, store=store, kernels=registry, resume="auto"
    )
    receipt = manifest.node(PUF_GROWTH_DESIGN_WEIGHT_NODE).receipt[
        "puf_design_weight_growth"
    ]
    assert receipt["field"] == "S006"
    assert receipt["series"] == "fixture.returns"


def test_a_changed_factor_level_moves_the_node_key_and_is_not_reused(
    prepared, tmp_path
):
    compiled_graph, sources, store, registry = prepared
    first = run_graph(
        compiled_graph, sources=sources, store=store, kernels=registry, resume="auto"
    )
    moved = dict(_SERIES)
    moved["fixture.price_index"] = {
        "kind": "nominal_price_restatement",
        "levels": {"2021": "2.0", "2024": "3.5"},
    }
    other_graph, other_registry = build(
        compile_puf_growth(growth_recipe(), factor_table(moved))
    )
    second = run_graph(
        other_graph,
        sources=sources,
        store=store,
        kernels=other_registry,
        resume="auto",
    )
    assert (
        second.node(PUF_GROWTH_MONEY_NODE).key != first.node(PUF_GROWTH_MONEY_NODE).key
    )
    assert not second.node(PUF_GROWTH_MONEY_NODE).hit
    table = second.population(PUF_GROWTH_BOUNDARY_NODE).table(ENTITY)
    assert table.E00200.tolist() == [100.0 * 1.75, 0.0, -25.0 * 1.75]


def test_the_node_factory_refuses_a_contract_without_a_design_weight():
    rules = tuple(rule for rule in growth_recipe().rules if rule.field != "S006")
    contract = compile_puf_growth(
        GrowthRecipe("puf_growth_fixture_recipe_v1", rules), factor_table()
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        us_puf_growth_nodes(
            contract,
            base="source",
            entity=ENTITY,
            person_boundary_columns=("is_primary_filer",),
        )
    assert error.value.reason == "NO_DESIGN_WEIGHT_RULE"


def test_a_kernel_refuses_an_unexpected_parameter_set():
    with pytest.raises(PufGrowthRefusalError) as error:
        PufGrowthBoundaryKernel.contract({"entity": ENTITY})
    assert error.value.reason == "NODE_PARAMETER_CONTRACT"


def test_a_nonint64_record_identifier_is_refused_by_the_money_node(tmp_path):
    """The provenance key must stay an exact integer identifier, not a variant."""
    contract = compile_puf_growth(growth_recipe(), factor_table())
    compiled_graph, registry = build(contract, recid_dtype="Int64")
    sources = {"puf_raw": write_source(tmp_path / "source", recid_dtype="Int64")}
    with pytest.raises(NodeRejectedError) as error:
        run_graph(
            compiled_graph,
            sources=sources,
            store=ContentStore(tmp_path / "store"),
            kernels=registry,
            resume="auto",
        )
    assert "PROVENANCE_RECID_DTYPE" in str(error.value)


def test_the_graph_module_reads_no_file_and_imports_no_country_engine():
    """Same lane-independent form as the pure module's guard."""
    tree = ast.parse(inspect.getsource(puf_growth_graph))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported
    assert not any(name.lower().startswith("policyengine") for name in imported)
    source = inspect.getsource(puf_growth_graph)
    for token in ("open(", "read_csv", "h5py", "urlopen", "requests", "Path("):
        assert token not in source


def test_the_node_factory_refuses_a_bad_contract_or_no_boundary_column():
    contract = compile_puf_growth(growth_recipe(), factor_table())
    with pytest.raises(PufGrowthRefusalError) as error:
        us_puf_growth_nodes(
            contract, base="source", entity=ENTITY, person_boundary_columns=()
        )
    assert error.value.reason == "PERSON_BOUNDARY_COLUMNS"

    with pytest.raises(PufGrowthRefusalError) as error:
        us_puf_growth_nodes(
            "not a contract",
            base="source",
            entity=ENTITY,
            person_boundary_columns=("is_primary_filer",),
        )
    assert error.value.reason == "COMPILED_TYPE"


def test_the_kernel_hash_attests_the_pure_transform_and_its_dependencies():
    """A behaviour-only edit to puf_growth.py must move the node key."""
    modules = puf_growth_graph._IMPLEMENTATION_MODULES
    assert "microcosm.build.us_runtime.puf_growth" in modules
    assert "microcosm.build.us_runtime.puf_growth_graph" in modules
    kernel = PufGrowthMoneyKernel()
    assert kernel.implementation_hash() == source_hash(
        *(import_module(name) for name in modules),
        dependencies=kernel.capabilities.dependencies,
    )
    assert (
        PufGrowthBoundaryKernel().implementation_hash() == kernel.implementation_hash()
    )


def test_the_kernel_helpers_refuse_a_context_missing_what_they_declared():
    context = SimpleNamespace(tables={ENTITY: raw_tax_units()})
    with pytest.raises(PufGrowthRefusalError) as error:
        puf_growth_graph._identifier_index(context, ENTITY)
    assert error.value.reason == "ENTITY_ID_COLUMN"

    with pytest.raises(PufGrowthRefusalError) as error:
        puf_growth_graph._declared_source(context, ENTITY, ("RECID", "E99999"))
    assert error.value.reason == "MISSING_DECLARED_FIELD"
    assert error.value.field == "E99999"
