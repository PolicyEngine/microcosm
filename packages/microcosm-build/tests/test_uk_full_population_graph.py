"""Full-build population graph: legacy ladder draw and identity-keyed assignment."""

import hashlib

import numpy as np
import pandas as pd
import pytest
from test_uk_ladder_rowwise_clone import _seam_frame
from test_uk_ladder_rowwise_clone import toy_ladder as toy_ladder
from uk_atomic_support_fixtures import toy_support_sources, write_toy_supports

from microcosm.build import atomic_geography as geo
from microcosm.build.uk_runtime.atomic_area_support import (
    IDENTITY_COLUMN,
    UK_NATIVE_ALIAS_COLUMNS,
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.geography_ladder import UK_GEOGRAPHY_LADDER_COLUMNS
from microcosm.build.uk_runtime.graph_kernels import UKClaimKernel
from microcosm.build.uk_runtime.graph_population import (
    append_uk_population_nodes,
    register_uk_population_kernels,
)
from microcosm.build.uk_runtime.rowwise_dataset import (
    clone_uk_dataset_with_ladder_geography,
    expand_uk_geographic_pool,
    ladder_clone_index_column,
)
from microcosm.frame import Frame
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)

SOURCE_VINTAGE = "toy_2023_24"


def source_frame():
    original = _seam_frame()
    tables = {e: original.table(e).copy() for e in original.entities}
    tables["person"]["age"] = 40
    tables["benunit"]["would_claim_uc"] = True
    household = tables["household"]
    household["region"] = household["region"].astype("string")
    # Explicit spine lineage: every toy household is its own FRS original.
    household["source_household_id"] = household["household_id"].astype("int64")
    household["household_support_channel"] = pd.array(
        ["frs"] * len(household), dtype="string"
    )
    household["household_support_clone_index"] = np.zeros(len(household), dtype="int64")
    household["household_is_spi_synthetic"] = False
    household["household_is_capital_gains_clone"] = False
    household["household_is_cgt_band_donor"] = False
    return Frame(
        tables,
        original.schema,
        {"household": original.weights_for("household")},
        original.strata,
        mass_log=original.mass_log,
        metadata=original.metadata,
    )


class Source(KernelBase):
    ref = "uk.test.full-source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def implementation_hash(self):
        return hashlib.sha256(self.ref.encode()).hexdigest()

    def run(self, context):
        return KernelResult(frame=source_frame())


def source_node(frame):
    return Node(
        "source",
        Source.ref,
        structural=StructuralDelta.CREATE,
        sources=("fixture",),
        outputs=tuple(
            Owned(
                entity,
                col,
                "string" if table[col].dtype.kind in "OUS" else str(table[col].dtype),
            )
            for entity in frame.entities
            for table in [frame.table(entity)]
            for col in table.columns
            if col
            not in {
                "person_id",
                "person_household_id",
                "person_benunit_id",
                "household_id",
                "benunit_id",
            }
        ),
    )


def graph_and_registry(k, *, geography_assignment="legacy", definition=None, seed=7):
    graph = append_uk_population_nodes(
        Graph(
            "uk",
            (SourceRef("fixture", "raw-bytes-v1"),),
            (source_node(source_frame()),),
        ),
        population="source",
        time_period="2023",
        weight_kind="importance",
        n_clones=k,
        seed=seed,
        source_year=2023,
        geography_assignment=geography_assignment,
        atomic_geography_definition=definition,
        source_vintage=SOURCE_VINTAGE if geography_assignment == "atomic" else None,
    )
    registry = KernelRegistry()
    registry.register(Source())
    registry.register(UKClaimKernel())
    register_uk_population_kernels(registry)
    return graph, registry


@pytest.mark.parametrize("k", [1, 2, 5])
def test_legacy_population_graph_preserves_rows_weights_geography_and_replays(
    k, toy_ladder, tmp_path
):
    ladder, path = toy_ladder
    expected = clone_uk_dataset_with_ladder_geography(
        source_frame(),
        ladder,
        n_clones=k,
        seed=7,
        source_year=2023,
        expected_constituency_vintage="2024_pcon",
    ).frame
    graph, registry = graph_and_registry(k, geography_assignment="legacy")
    store = ContentStore(tmp_path / "store")
    compiled = compile_graph(graph)
    assert "uk.full.locations" in compiled.predecessors["uk.full.geography_mapping"]
    assert "uk.full.identity" not in compiled.order
    first = run_graph(
        compiled,
        sources={"uk_ladder": path, "fixture": path},
        store=store,
        kernels=registry,
    )
    actual = first.population("uk.full.expand")
    for entity in expected.entities:
        pd.testing.assert_frame_equal(
            actual.table(entity)[expected.table(entity).columns],
            expected.table(entity),
            check_dtype=False,
        )
    np.testing.assert_array_equal(
        actual.weights_for("household").values, expected.weights_for("household").values
    )
    assert actual.mass_log == expected.mass_log
    # A new registry cannot obtain results from mutable objects of the cold run.
    _, fresh_registry = graph_and_registry(k, geography_assignment="legacy")
    replay = run_graph(
        compiled,
        sources={"uk_ladder": path, "fixture": path},
        store=store,
        kernels=fresh_registry,
        resume="require",
    )
    assert replay.population("uk.full.expand").mass_log == actual.mass_log


def test_k_changes_expansion_but_never_sampling():
    first, _ = graph_and_registry(1)
    second, _ = graph_and_registry(3)
    assert first.node("uk.full.sample") == second.node("uk.full.sample")
    assert first.node("uk.full.normalize") == second.node("uk.full.normalize")
    assert first.node("uk.full.expand").params["n_clones"] == 1
    assert second.node("uk.full.expand").params["n_clones"] == 3


def _atomic_oracle(k, payloads, definition):
    """The same pool, keyed and assigned in process through the shared operators."""
    frame = source_frame()
    household = frame.table("household").copy()
    household["household_weight"] = frame.weights_for("household").values
    pool = expand_uk_geographic_pool(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        n_clones=k,
        source_year=2023,
        time_period="2023",
        household_weight_kind=frame.weights_for("household").kind,
        mass_log=frame.mass_log,
    ).frame.table("household")
    clone = ladder_clone_index_column("household")
    keys = [
        household_draw_key(
            source="frs",
            source_vintage=SOURCE_VINTAGE,
            source_household_id=int(source_id),
            clone_path=(("geographic_support", int(index)),) if index else (),
        )
        for source_id, index in zip(
            pool["source_household_id"], pool[clone], strict=True
        )
    ]
    table = pool[["household_id", "region"]].copy()
    table[IDENTITY_COLUMN] = pd.array(keys, dtype="string")
    supports = {s: geo.decode_atomic_support(p) for s, p in payloads.items()}
    assigned = pd.concat(
        [table, geo.assign_atomic(table, definition, supports)], axis=1
    )
    derived = geo.derive_geography(assigned, definition, supports)
    return pd.concat([assigned, derived], axis=1).set_index("household_id")


@pytest.mark.parametrize("k", [1, 2, 5])
def test_atomic_population_graph_keys_assigns_derives_gates_and_replays(
    k, toy_ladder, tmp_path
):
    _, ladder_path = toy_ladder
    payloads, paths = write_toy_supports(tmp_path / "supports")
    definition = uk_atomic_assignment_definition(payloads, seed=7)
    graph, registry = graph_and_registry(
        k, geography_assignment="atomic", definition=definition
    )
    compiled = compile_graph(graph)
    for absent in ("uk.full.locations", "uk.full.geography_mapping"):
        assert absent not in compiled.order
    gate = compiled.predecessors["uk.full.geography_gate"]
    assert {"uk.full.identity", "uk.full.geography.derive"} <= set(gate)
    assert "uk.full.identity" in compiled.predecessors["uk.full.geography.assign"]
    assert "uk.full.geography.gate" in compiled.order
    sources = {
        "uk_ladder": ladder_path,
        "fixture": ladder_path,
        **toy_support_sources(paths),
    }
    store = ContentStore(tmp_path / "store")
    first = run_graph(compiled, sources=sources, store=store, kernels=registry)
    household = first.population("uk.full.expand").table("household")
    expected = _atomic_oracle(k, payloads, definition)
    assert len(household) == 4 * k
    actual = household.set_index("household_id")
    for column in (
        IDENTITY_COLUMN,
        "atomic_area_code",
        "atomic_area_system",
        "atomic_area_basis",
        *UK_GEOGRAPHY_LADDER_COLUMNS,
        *UK_NATIVE_ALIAS_COLUMNS,
    ):
        pd.testing.assert_series_equal(
            actual[column].astype("string"),
            expected.loc[actual.index, column].astype("string"),
            check_names=False,
        )
    assert set(actual["atomic_area_basis"]) == {"assigned"}
    assert actual["region_code"].tolist() == [
        {
            "LONDON": "E12000007",
            "WALES": "W99999999",
            "SCOTLAND": "S99999999",
            "NORTHERN_IRELAND": "N99999999",
        }[r]
        for r in actual["region"]
    ]
    assert first.nodes["uk.full.geography_gate"].receipt["outcome"] == "pass"
    assert first.nodes["uk.full.geography.gate"].receipt["outcome"] == "pass"
    assert first.nodes["uk.full.identity"].receipt["source_vintage"] == SOURCE_VINTAGE
    _, fresh = graph_and_registry(
        k, geography_assignment="atomic", definition=definition
    )
    replay = run_graph(
        compiled, sources=sources, store=store, kernels=fresh, resume="require"
    )
    assert all(receipt.hit for receipt in replay.nodes.values())


def test_atomic_keys_and_draws_are_stable_when_k_grows(toy_ladder, tmp_path):
    _, ladder_path = toy_ladder
    payloads, paths = write_toy_supports(tmp_path / "supports")
    definition = uk_atomic_assignment_definition(payloads, seed=7)
    sources = {
        "uk_ladder": ladder_path,
        "fixture": ladder_path,
        **toy_support_sources(paths),
    }
    by_key = {}
    for k in (2, 5):
        graph, registry = graph_and_registry(
            k, geography_assignment="atomic", definition=definition
        )
        run = run_graph(
            compile_graph(graph),
            sources=sources,
            store=ContentStore(tmp_path / f"store-{k}"),
            kernels=registry,
        )
        household = run.population("uk.full.expand").table("household")
        by_key[k] = dict(
            zip(household[IDENTITY_COLUMN], household["atomic_area_code"], strict=True)
        )
    assert set(by_key[2]) < set(by_key[5])
    assert all(by_key[5][key] == area for key, area in by_key[2].items())


def test_atomic_declaration_refuses_seed_or_identity_drift(tmp_path):
    payloads, _ = write_toy_supports(tmp_path / "supports")
    with pytest.raises(ValueError, match="seed differs"):
        graph_and_registry(
            1,
            geography_assignment="atomic",
            definition=uk_atomic_assignment_definition(payloads, seed=8),
        )
    with pytest.raises(ValueError, match="requires the UK assignment definition"):
        graph_and_registry(1, geography_assignment="atomic")
    with pytest.raises(ValueError, match="takes no atomic definition"):
        graph_and_registry(
            1,
            geography_assignment="legacy",
            definition=uk_atomic_assignment_definition(payloads, seed=7),
        )
    with pytest.raises(ValueError, match="one of"):
        graph_and_registry(1, geography_assignment="random")
