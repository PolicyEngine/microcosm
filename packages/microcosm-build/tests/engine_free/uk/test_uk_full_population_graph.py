"""Full-build population graph: legacy ladder draw and identity-keyed assignment."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build import atomic_geography as geo
from microcosm.build.uk_runtime.atomic_area_support import (
    IDENTITY_COLUMN,
    UK_NATIVE_ALIAS_COLUMNS,
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.geography_ladder import UK_GEOGRAPHY_LADDER_COLUMNS
from microcosm.build.uk_runtime.local_authority_input import (
    resolve_local_authority_engine_keys,
)
from microcosm.build.uk_runtime.rowwise_dataset import (
    clone_uk_dataset_with_ladder_geography,
    expand_uk_geographic_pool,
    ladder_clone_index_column,
)
from microcosm.graph import ContentStore, compile_graph, run_graph
from test_support.microcosm_build.uk_atomic_support_fixtures import (
    toy_support_sources,
    write_toy_supports,
)
from test_support.microcosm_build.uk_full_population_graph import (
    SOURCE_VINTAGE,
    expected_clone_brmas,
    graph_and_registry,
    source_frame,
)
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)


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
    _assert_brma_nodes(graph, compiled)
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
        columns = expected.table(entity).columns
        if entity == "household":
            columns = columns.drop("brma")
        pd.testing.assert_frame_equal(
            actual.table(entity)[columns],
            expected.table(entity)[columns],
            check_dtype=False,
        )
    np.testing.assert_array_equal(
        actual.table("household")["brma"],
        expected_clone_brmas(actual.table("household"), k),
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
    assert all(receipt.hit for receipt in replay.nodes.values())
    pd.testing.assert_frame_equal(
        replay.population("uk.full.expand").table("household"),
        actual.table("household"),
    )


def _assert_brma_nodes(graph, compiled):
    table = graph.node("uk.full.brma_table")
    rewrite = graph.node("uk.full.brma")
    assert table.population == "uk.full.normalize"
    assert "uk.full.normalize" in compiled.predecessors[table.id]
    assert rewrite.population == "uk.full.expand"
    assert {table.id, "uk.full.expand"} <= set(compiled.predecessors[rewrite.id])
    assert [(o.entity, o.column, o.rewrite) for o in rewrite.outputs] == [
        ("household", "brma", True)
    ]


def test_k_changes_expansion_but_never_sampling():
    first, _ = graph_and_registry(1)
    second, _ = graph_and_registry(3)
    assert first.node("uk.full.sample") == second.node("uk.full.sample")
    assert first.node("uk.full.normalize") == second.node("uk.full.normalize")
    assert first.node("uk.full.brma_table") == second.node("uk.full.brma_table")
    assert first.node("uk.full.expand").params["n_clones"] == 1
    assert second.node("uk.full.expand").params["n_clones"] == 3


def _expanded_oracle(k):
    frame = source_frame()
    household = frame.table("household").copy()
    household["household_weight"] = frame.weights_for("household").values
    return expand_uk_geographic_pool(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        n_clones=k,
        source_year=2023,
        time_period="2023",
        household_weight_kind=frame.weights_for("household").kind,
        mass_log=frame.mass_log,
    ).frame


def _atomic_oracle(k, payloads, definition):
    """The same pool, keyed and assigned in process through the shared operators."""
    pool = _expanded_oracle(k).table("household")
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
    # The engine input resolved UK-side after the shared derive (microcosm#953).
    derived["local_authority"] = resolve_local_authority_engine_keys(
        derived["local_authority_code"]
    )
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
    _assert_brma_nodes(graph, compiled)
    for absent in ("uk.full.locations", "uk.full.geography_mapping"):
        assert absent not in compiled.order
    gate = compiled.predecessors["uk.full.geography_gate"]
    assert {
        "uk.full.identity",
        "uk.full.geography.derive",
        "uk.full.geography.local_authority",
    } <= set(gate)
    assert (
        "uk.full.geography.derive"
        in compiled.predecessors["uk.full.geography.local_authority"]
    )
    assert "uk.full.identity" in compiled.predecessors["uk.full.geography.assign"]
    assert "uk.full.geography.gate" in compiled.order
    sources = {
        "uk_ladder": ladder_path,
        "fixture": ladder_path,
        **toy_support_sources(paths),
    }
    store = ContentStore(tmp_path / "store")
    first = run_graph(compiled, sources=sources, store=store, kernels=registry)
    frame = first.population("uk.full.expand")
    household = frame.table("household")
    expected = _atomic_oracle(k, payloads, definition)
    assert len(household) == 4 * k
    expanded = _expanded_oracle(k)
    for entity in expanded.entities:
        columns = expanded.table(entity).columns
        if entity == "household":
            columns = columns.drop("brma")
        pd.testing.assert_frame_equal(
            frame.table(entity)[columns],
            expanded.table(entity)[columns],
            check_dtype=False,
        )
    np.testing.assert_array_equal(
        frame.weights_for("household").values,
        expanded.weights_for("household").values,
    )
    assert frame.mass_log == expanded.mass_log
    np.testing.assert_array_equal(household["brma"], expected_clone_brmas(household, k))
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
    pd.testing.assert_frame_equal(
        replay.population("uk.full.expand").table("household"), household
    )


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
