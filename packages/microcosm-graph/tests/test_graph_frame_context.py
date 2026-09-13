"""Amendment 26: the context carries the version's metadata, mass log, order.

The executor projects each table in *declaration* order, so a kernel that
reconstructs its population version's layout cannot do it from
``KernelContext.tables`` alone, and it cannot see the version's metadata or
its incoming ``Frame`` mass log at all. These properties are about the three
fields that close that gap, and about the one thing they must never do:
name a column the node was not given.

Everything here runs real shared graph operations over the invented toy
country. No country model, engine, or build artifact is involved.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from microcosm.frame import Frame, MassChangeRecord
from microcosm.graph import (
    ContentStore,
    Graph,
    KernelContext,
    KernelResult,
    Node,
    Owned,
    Slice,
    compile_graph,
    run_graph,
)

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

SOURCE_REF = "source.metadata@1"
PROBE_REF = "probe.frame_context@1"

#: Metadata the CREATE kernel puts on the version, including a nested value
#: so the frozen projection is exercised rather than a flat string map.
VERSION_METADATA = {"time_period": "2024", "vintage": ("frs", "was")}


class MetadataSource(toy.ToyKernel):
    """CREATE: the toy population, carrying declared version metadata."""

    def compute(self, context: KernelContext) -> KernelResult:
        frame = toy.read_toy_frame(context.sources["survey"])
        tables = {entity: frame.table(entity) for entity in frame.entities}
        tables.update({link: frame.link(link) for link in frame.links})
        return KernelResult(
            frame=Frame(
                tables,
                frame.schema,
                {
                    entity: frame.weights_for(entity)
                    for entity in frame.weighted_entities
                },
                frame.strata,
                metadata=VERSION_METADATA,
            ),
            receipt={"persons": frame.n("person")},
        )


class FrameContextProbe(toy.ToyKernel):
    """Owns one column and records the frame view it was handed.

    With ``mass_reason`` set it also appends one ``Frame`` mass record
    asserting the household total is unchanged, which is what the legacy
    ``Frame`` contract calls an explicit conservation check.
    """

    def __init__(self, ref: str, capabilities, *, variant: str = "base") -> None:
        super().__init__(ref, capabilities, variant=variant)
        self.seen: list[dict[str, object]] = []

    def compute(self, context: KernelContext) -> KernelResult:
        self.seen.append(
            {
                "node": context.node.id,
                "column_order": {
                    entity: tuple(columns)
                    for entity, columns in context.frame_column_order.items()
                },
                "projected": {
                    entity: tuple(table.columns)
                    for entity, table in context.tables.items()
                },
                "metadata": dict(context.frame_metadata),
                "mass_log": tuple(context.frame_mass_log),
            }
        )
        ids = pd.Index(context.tables["person"]["person_id"], name="person_id")
        result_columns = {
            ("person", str(context.params["target"])): pd.Series(
                1.0, index=ids, dtype="float64"
            )
        }
        reason = context.params.get("mass_reason")
        if reason is None:
            return KernelResult(columns=result_columns)
        total = float(context.weights["household"].values.sum())
        return KernelResult(
            columns=result_columns,
            receipt={
                "frame_mass_log_append": [
                    {
                        "entity": "household",
                        "old_total": total,
                        "new_total": total,
                        "declared_factor": None,
                        "reason": str(reason),
                    }
                ]
            },
        )


def build_registry() -> tuple[object, FrameContextProbe]:
    """The toy registry with the metadata source and the probe registered."""
    registry = toy.toy_registry()
    registry.register(MetadataSource(SOURCE_REF, toy._CREATE))
    probe = FrameContextProbe(PROBE_REF, toy._DETERMINISTIC)
    registry.register(probe)
    return registry, probe


CREATE = dataclasses.replace(toy.CREATE, kernel=SOURCE_REF)


def probe_node(
    node_id: str,
    *,
    columns: tuple[str, ...],
    target: str,
    mass_reason: str | None = None,
) -> Node:
    """A probe reading ``columns`` of the person entity, in that order.

    A probe that states a mass record also reads the household entity,
    because the record is stated against the household total and a node
    only receives weights for the entities it projects.
    """
    inputs = (Slice("person", columns),)
    if mass_reason is not None:
        inputs = (*inputs, Slice("household", ("household_size",)))
    return Node(
        node_id,
        PROBE_REF,
        inputs=inputs,
        outputs=(Owned("person", target, "float64"),),
        params={"target": target, "mass_reason": mass_reason},
        population="survey",
    )


#: ``income`` before ``age`` is the reverse of the source table's own order,
#: so the projection and the version's layout genuinely disagree.
FIRST = probe_node(
    "probe_first",
    columns=("income", "age"),
    target="probe_a",
    mass_reason="households are unchanged by a derivation",
)
SECOND = probe_node("probe_second", columns=("probe_a",), target="probe_b")


def probe_graph(*extra: Node) -> Graph:
    return Graph("toy", (toy.SOURCE,), (CREATE, FIRST, SECOND, *extra))


def run_probe(
    root: Path,
    graph: Graph | None = None,
    *,
    registry=None,
    probe=None,
    store: ContentStore | None = None,
    sources=None,
    resume: str = "auto",
    observer=None,
):
    """Run ``graph`` and return ``(manifest, probe, sources, store)``."""
    if registry is None:
        registry, probe = build_registry()
    if sources is None:
        sources = {"survey": toy.copy_source(root / "source")}
    if store is None:
        store = ContentStore(root / "store")
    manifest = run_graph(
        compile_graph(graph or probe_graph()),
        sources=dict(sources),
        store=store,
        kernels=registry,
        resume=resume,
        _population_observer=observer,
    )
    return manifest, probe, sources, store


def observation(probe: FrameContextProbe, node_id: str) -> dict[str, object]:
    (found,) = [item for item in probe.seen if item["node"] == node_id]
    return found


# ----------------------------------------------------------------------
# The field contract
# ----------------------------------------------------------------------


def test_the_three_fields_ride_before_the_amendment_13_17_pair() -> None:
    """Amendment 17's "numerics rides at the end" stays literally true."""
    fields = [f.name for f in dataclasses.fields(KernelContext)]
    assert fields[-2:] == ["tolerances", "numerics"]
    assert fields[fields.index("artifacts") + 1 : fields.index("tolerances")] == [
        "frame_metadata",
        "frame_mass_log",
        "frame_column_order",
    ]


def bare_context(**kwargs) -> KernelContext:
    return KernelContext(
        node=Node("n", PROBE_REF),
        tables={"person": pd.DataFrame({"person_id": [1, 2], "age": [30, 40]})},
        weights={},
        strata=pd.Series(dtype="string"),
        params={},
        rng=None,
        **kwargs,
    )


def test_the_fields_default_to_an_empty_view() -> None:
    context = bare_context()
    assert dict(context.frame_metadata) == {}
    assert context.frame_mass_log == ()
    assert dict(context.frame_column_order) == {}


def test_column_order_refuses_an_undeclared_column() -> None:
    """The one thing an order must never do is name a column not given.

    A name is itself information about the version, so an order that
    mentions an unprojected column is refused rather than trimmed.
    """
    with pytest.raises(TypeError, match="exactly the projected columns"):
        bare_context(frame_column_order={"person": ("person_id", "age", "income")})


def test_column_order_refuses_a_partial_or_repeated_order() -> None:
    with pytest.raises(TypeError, match="exactly the projected columns"):
        bare_context(frame_column_order={"person": ("person_id",)})
    with pytest.raises(TypeError, match="exactly the projected columns"):
        bare_context(frame_column_order={"person": ("person_id", "person_id", "age")})
    with pytest.raises(TypeError, match="exactly the projected columns"):
        bare_context(frame_column_order={"person": ["person_id", "age"]})


def test_column_order_refuses_an_unprojected_entity() -> None:
    with pytest.raises(TypeError, match="exactly the projected columns"):
        bare_context(frame_column_order={"household": ("household_id",)})


def test_mass_log_refuses_anything_but_mass_records() -> None:
    with pytest.raises(TypeError, match="tuple of MassChangeRecord"):
        bare_context(frame_mass_log=({"entity": "household"},))
    with pytest.raises(TypeError, match="tuple of MassChangeRecord"):
        bare_context(
            frame_mass_log=[
                MassChangeRecord(
                    entity="household",
                    old_total=1.0,
                    new_total=1.0,
                    declared_factor=None,
                    reason="why",
                )
            ]
        )


def test_metadata_is_a_read_only_view() -> None:
    source = {"time_period": "2024"}
    context = bare_context(frame_metadata=source)
    assert dict(context.frame_metadata) == source
    with pytest.raises(TypeError):
        context.frame_metadata["time_period"] = "2025"  # type: ignore[index]
    source["time_period"] = "2025"
    assert context.frame_metadata["time_period"] == "2024"
    with pytest.raises(TypeError, match="non-empty strings"):
        bare_context(frame_metadata={"": "no"})


# ----------------------------------------------------------------------
# What the executor actually supplies
# ----------------------------------------------------------------------


def test_column_order_is_the_versions_own_order_not_the_projection(
    tmp_path: Path,
) -> None:
    """The real property: two different orders over the same column set."""
    _, probe, _, _ = run_probe(tmp_path / "run")
    seen = observation(probe, "probe_first")
    order = seen["column_order"]["person"]
    projected = seen["projected"]["person"]
    assert set(order) == set(projected)
    assert order != projected
    assert projected.index("income") < projected.index("age")
    assert order.index("age") < order.index("income")


def test_column_order_names_no_column_the_node_did_not_read(tmp_path: Path) -> None:
    """``receives_x`` exists on the version and is in neither view."""
    _, probe, _, _ = run_probe(tmp_path / "run")
    for key in ("column_order", "projected"):
        person = observation(probe, "probe_first")[key]["person"]
        assert "receives_x" not in person
        assert "is_adult" not in person
    assert set(observation(probe, "probe_second")["column_order"]["person"]) == {
        "person_id",
        "person_household_id",
        "person_release_id",
        "probe_a",
    }


def test_version_metadata_reaches_every_node_of_the_version(tmp_path: Path) -> None:
    _, probe, _, _ = run_probe(tmp_path / "run")
    for node_id in ("probe_first", "probe_second"):
        metadata = observation(probe, node_id)["metadata"]
        assert metadata["time_period"] == "2024"
        assert tuple(metadata["vintage"]) == ("frs", "was")


def test_mass_log_is_the_incoming_log(tmp_path: Path) -> None:
    """The appending node sees the log before its own record; its successor after."""
    _, probe, _, _ = run_probe(tmp_path / "run")
    assert observation(probe, "probe_first")["mass_log"] == ()
    after = observation(probe, "probe_second")["mass_log"]
    assert len(after) == 1
    assert after[0].entity == "household"
    assert after[0].reason == "households are unchanged by a derivation"
    assert after[0].old_total == after[0].new_total


# ----------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------


def test_required_replay_is_a_full_hit(tmp_path: Path) -> None:
    """The three fields are additive: no key moves, everything replays."""
    cold_manifest, _, sources, store = run_probe(tmp_path / "run")
    assert not any(receipt.hit for receipt in cold_manifest.nodes.values())

    registry, probe = build_registry()
    warm = run_graph(
        compile_graph(probe_graph()),
        sources=dict(sources),
        store=ContentStore(tmp_path / "run" / "store"),
        kernels=registry,
        resume="require",
    )
    assert all(receipt.hit for receipt in warm.nodes.values())
    assert probe.seen == []
    assert {n: r.key for n, r in warm.nodes.items()} == {
        n: r.key for n, r in cold_manifest.nodes.items()
    }


def test_a_new_node_over_restored_populations_sees_the_same_frame(
    tmp_path: Path,
) -> None:
    """The fields are rebuilt from a restored Frame, not only a computed one.

    The store round trip is where metadata and the ``Frame`` mass log could
    quietly disappear, so the property is stated against a node that runs
    cold on top of cache hits.
    """
    _, cold_probe, sources, _ = run_probe(tmp_path / "run")
    expected = observation(cold_probe, "probe_second")

    third = probe_node("probe_third", columns=("probe_a",), target="probe_c")
    registry, probe = build_registry()
    warm = run_graph(
        compile_graph(probe_graph(third)),
        sources=dict(sources),
        store=ContentStore(tmp_path / "run" / "store"),
        kernels=registry,
        resume="auto",
    )
    assert {n for n, r in warm.nodes.items() if not r.hit} == {"probe_third"}
    seen = observation(probe, "probe_third")
    assert seen["metadata"] == expected["metadata"]
    assert seen["mass_log"] == expected["mass_log"]
    assert seen["column_order"] == expected["column_order"]


def test_a_retained_mutating_observer_changes_nothing(tmp_path: Path) -> None:
    """Amendment 24 still holds across the new fields.

    The observer keeps every snapshot and rewrites its tables and its
    metadata view after the callback returns; what later nodes read through
    the frame fields, and the run's identity, are unchanged.
    """
    plain_manifest, plain_probe, sources, _ = run_probe(tmp_path / "plain")
    retained: list[object] = []

    def observe(node_id: str, population) -> None:
        retained.append(population)
        for entity in population.frame.entities:
            table = population.frame.table(entity)
            for column in table.columns:
                table.loc[:, column] = table[column].iloc[0]

    observed_manifest, observed_probe, _, _ = run_probe(
        tmp_path / "observed", sources=sources, observer=observe
    )
    assert len(retained) == len(observed_manifest.nodes)
    assert {n: r.key for n, r in observed_manifest.nodes.items()} == {
        n: r.key for n, r in plain_manifest.nodes.items()
    }
    for node_id in ("probe_first", "probe_second"):
        assert observation(observed_probe, node_id) == observation(plain_probe, node_id)
