"""The PUF growth contract running as real Microcosm graph nodes.

The population is three invented returns written to a temporary ``csv-tables``
source; the factor levels are invented and the table says so. Nothing here
reads a PUF, downloads anything, or imports a country engine.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.puf_growth import (
    PROVENANCE_RECID_COLUMN,
    PROVENANCE_SOURCE_AGI_COLUMN,
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
)
from microcosm.graph.codecs import load_csv_tables

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


def write_source(directory: Path) -> Path:
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
                    "RECID": "int64",
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


def build(contract):
    graph = Graph(
        country="us",
        sources=(SourceRef("puf_raw", "csv-tables"),),
        nodes=(
            Node(
                "source",
                FixtureSourceKernel.ref,
                sources=("puf_raw",),
                structural=StructuralDelta.CREATE,
                outputs=_CREATE_OUTPUTS,
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
    """The parity oracle: the same contract applied by a direct call."""
    contract = compile_puf_growth(growth_recipe(), factor_table())
    grown = apply_puf_growth(raw_tax_units(), contract).table
    grown["S006"] = apply_puf_design_weight_growth(raw_tax_units().S006, contract)
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
            "design_weight_grown_here"
        ]
        is False
    )


def test_weight_growth_is_owned_by_its_own_node(prepared):
    compiled_graph, sources, store, registry = prepared
    weight_node = compiled_graph.graph.node(PUF_GROWTH_DESIGN_WEIGHT_NODE)
    assert [(o.entity, o.column) for o in weight_node.outputs] == [(ENTITY, "S006")]
    money_node = compiled_graph.graph.node(PUF_GROWTH_MONEY_NODE)
    assert "S006" not in {o.column for o in money_node.outputs}
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
