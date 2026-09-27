"""Invented two-arm population through the actual geography/clone executor."""

import json

import numpy as np
import pandas as pd

from microcosm.build import atomic_geography as geo
from microcosm.build.graph_atomic_geography import register_atomic_geography_kernels
from microcosm.build.us_runtime import atomic_block_support as us
from microcosm.build.us_runtime.graph_atomic_survey_clone import (
    ASSIGNMENT_IDENTITY,
    atomic_survey_clone_nodes,
)
from microcosm.build.us_runtime.graph_combined_clone import (
    register_us_combined_survey_clone_kernels,
)
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.population import dtype_for_token


def invented_population(*, last_id=42):
    arms = {}
    string = dtype_for_token("string")
    for channel, state, puma in (("asec", "01", None), ("acs", "02", "0200002")):
        person = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "age": [40, 10, 50],
                **{
                    f"person_{entity}_id": [11, 11, last_id]
                    for entity in US_SCHEMA.group_entities
                },
            }
        )
        tables = {
            "person": person,
            **{
                entity: pd.DataFrame({f"{entity}_id": [11, last_id]})
                for entity in US_SCHEMA.group_entities
            },
        }
        tables["household"][ASSIGNMENT_IDENTITY[0]] = pd.Series(
            [json.dumps([channel, 2024, 2024, str(i)]) for i in (11, last_id)],
            dtype=string,
        )
        tables["household"]["observed_state"] = pd.Series([state] * 2, dtype=string)
        tables["household"]["observed_puma"] = pd.Series([puma] * 2, dtype=string)
        arms[channel] = Frame(
            tables,
            US_SCHEMA,
            {"household": Weights(np.array([3.0, 7.0]), WeightKind.DESIGN)},
            pd.Series([channel] * 3, dtype=string),
        )
    frame = assemble_spines(arms, household_mass_shares={"asec": 0.6, "acs": 0.4})
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for table in tables.values():
        for name in table:
            if not pd.api.types.is_numeric_dtype(table[name]):
                table[name] = table[name].astype(string)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
        frame.strata.astype(string),
        metadata=frame.metadata,
    )


class InventedHost(KernelBase):
    ref = "invented.us_atomic_host@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        assert context.sources["host"].read_bytes() == b"invented literal host v1"
        return KernelResult(frame=invented_population())


def test_block_assignment_follows_complete_clone_and_survives_required_replay(tmp_path):
    source_ids = {k: "invented-" + k for k in ("population", "district", "puma")}
    blocks = [int(x) for x in ("010010201001000", "010010201001001", "020130001001000")]
    payload = us.assemble_atomic_block_support(
        block_population=dict(zip(blocks, (2, 6, 1), strict=True)),
        cd_by_block=dict(zip(blocks, (101, 102, 200), strict=True)),
        puma_by_tract={blocks[0] // 10000: 100001, blocks[2] // 10000: 200002},
        source_ids=source_ids,
    )
    spec = us.assignment_definition(
        identity=ASSIGNMENT_IDENTITY,
        state_column="observed_state",
        puma_column="observed_puma",
        source_ids=source_ids,
        seed=17,
    )
    original = invented_population()
    columns = frame_column_declarations(original)
    stages = atomic_survey_clone_nodes(spec, columns, base="host")
    graph = compile_graph(
        Graph(
            "us",
            (
                SourceRef("host", "raw-bytes-v1"),
                SourceRef(us.SOURCE, "raw-bytes-v1"),
            ),
            (
                Node(
                    "host",
                    InventedHost.ref,
                    structural=StructuralDelta.CREATE,
                    sources=("host",),
                    outputs=columns,
                ),
                *stages,
            ),
        )
    )
    clone_id = "combined_survey_puf_support_clone"
    assert graph.order.index(clone_id) < graph.order.index(clone_id + ".owned")
    assert clone_id in graph.predecessors["geography.assign"]
    assert clone_id + ".owned" in graph.predecessors["geography.assign"]
    assert "geography.support.0" in graph.predecessors["geography.assign"]
    assert "geography.clone_gate" not in graph.order
    registry = KernelRegistry()
    registry.register(InventedHost())
    register_atomic_geography_kernels(registry)
    register_us_combined_survey_clone_kernels(registry)
    host_path, support_path = tmp_path / "host.bin", tmp_path / "support.npz"
    host_path.write_bytes(b"invented literal host v1")
    support_path.write_bytes(payload)
    sources = {"host": host_path, us.SOURCE: support_path}
    store = ContentStore(tmp_path / "store")
    cold = run_graph(graph, sources=sources, store=store, kernels=registry)
    warm = run_graph(
        graph, sources=sources, store=store, kernels=registry, resume="require"
    )
    for manifest in (cold, warm):
        before, after = manifest.population("host"), manifest.population(clone_id)
        hh = after.table("household")
        assert len(hh) == 8 and len(after.person) == 12
        native = hh[hh.household_support_clone_index.eq(0)]
        copies = hh[hh.household_support_clone_index.eq(1)]
        geo_columns = [
            *spec["outputs"].values(),
            *(x["output"] for x in spec["systems"][0]["layers"]),
        ]
        assert not set(geo_columns) & set(before.table("household"))
        for column in (ASSIGNMENT_IDENTITY[0], "observed_state", "observed_puma"):
            pd.testing.assert_series_equal(
                native[column].reset_index(drop=True),
                copies[column].reset_index(drop=True),
            )
        assert not hh.duplicated(list(ASSIGNMENT_IDENTITY)).any()
        expected = geo.assign_atomic(
            hh.drop(columns=geo_columns),
            spec,
            {us.SYSTEM: geo.decode_atomic_support(payload)},
        )
        for column in spec["outputs"].values():
            assert hh[column].tolist() == expected[column].tolist()
        for entity in original.entities:
            assert len(after.table(entity)) == 2 * len(original.table(entity))
        np.testing.assert_array_equal(
            after.weights_for("household").values,
            np.tile(before.weights_for("household").values / 2, 2),
        )
        assert manifest.nodes["geography.gate"].receipt["outcome"] == "pass"
        geo.validate_geography(
            hh.iloc[[6, 1]], spec, {us.SYSTEM: geo.decode_atomic_support(payload)}
        )
    for node in stages:
        assert cold.nodes[node.id].key == warm.nodes[node.id].key
