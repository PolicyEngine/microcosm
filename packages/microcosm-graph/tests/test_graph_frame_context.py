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
from types import MappingProxyType

import pandas as pd
import pytest

from microcosm.frame import Frame, MassChangeRecord, WeightKind, Weights
from microcosm.graph import (
    ContentStore,
    Graph,
    KernelContext,
    KernelResult,
    Node,
    NodeRejectedError,
    Owned,
    Slice,
    StructuralDelta,
    WeightTransition,
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
REWEIGHT_PROBE_REF = "probe.frame_context_reweight@1"
ISOLATION_REF = "probe.frame_context_isolation@1"

#: What a tampering probe or observer writes, so an assertion can tell
#: "the rewrite never happened" from "the rewrite never escaped".
TAMPERED = "tampered"

#: Metadata the CREATE kernel puts on the version, including a nested
#: mapping and a nested sequence, so the frozen projection is exercised
#: rather than a flat string map -- and so a property can rewrite a value
#: one level below the mapping the context hands out.
VERSION_METADATA = {
    "time_period": "2024",
    "vintage": ("frs", "was"),
    "provenance": {"source": "frs", "year": 2024},
}


class MetadataSource(toy.ToyKernel):
    """CREATE: the toy population, carrying declared version metadata."""

    def compute(self, context: KernelContext) -> KernelResult:
        frame = toy.read_toy_frame(context.sources["survey"])
        tables = {entity: frame.table(entity) for entity in frame.entities}
        tables.update({link: frame.link(link) for link in frame.links})
        mass_log = ()
        reason = context.params.get("source_mass_reason")
        if reason is not None:
            total = float(frame.weights_for("household").values.sum())
            mass_log = (MassChangeRecord("household", total, total, None, reason),)
        return KernelResult(
            frame=Frame(
                tables,
                frame.schema,
                {
                    entity: frame.weights_for(entity)
                    for entity in frame.weighted_entities
                },
                frame.strata,
                mass_log=mass_log,
                metadata=VERSION_METADATA,
            ),
            receipt={"persons": frame.n("person")},
        )


def observed(context: KernelContext) -> dict[str, object]:
    """Everything a property asserts about one node's frame view."""
    return {
        "node": context.node.id,
        "column_order": {
            entity: tuple(columns)
            for entity, columns in context.frame_column_order.items()
        },
        "projected": {
            entity: tuple(table.columns) for entity, table in context.tables.items()
        },
        "metadata": dict(context.frame_metadata),
        "mass_log": tuple(context.frame_mass_log),
    }


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
        self.seen.append(observed(context))
        ids = pd.Index(context.tables["person"]["person_id"], name="person_id")
        # The output *is* the mass-log view, so the stored bytes of a node
        # differ whenever what it was shown differs. A key that survives an
        # unrelated sibling therefore has to have been shown the same log.
        result_columns = {
            ("person", str(context.params["target"])): pd.Series(
                float(len(context.frame_mass_log)), index=ids, dtype="float64"
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


class ReweightProbe(toy.ToyKernel):
    """REWEIGHT: records the frame view it was handed, then scales weights.

    A structural node, unlike an ordinary one, has a key that binds its
    base *and* every ordinary member of that version, so it is the node
    that may be shown the version's cumulative mass log.
    """

    def __init__(self, ref: str, capabilities, *, variant: str = "base") -> None:
        super().__init__(ref, capabilities, variant=variant)
        self.seen: list[dict[str, object]] = []

    def compute(self, context: KernelContext) -> KernelResult:
        self.seen.append(observed(context))
        before = context.weights["household"].values
        after = before * 2.0
        return KernelResult(
            weights=Weights(values=after, kind=WeightKind.IMPORTANCE),
            receipt={"mass": toy._mass_record(context, before, after, "free")},
        )


def rewrite(context: KernelContext, field: str) -> None:
    """Rewrite one of the three frame fields of ``context``, in place.

    This is the adversary the amendment defends against, so it uses the
    capability a kernel actually has rather than a public setter: a frozen
    dataclass yields to ``object.__setattr__``, and the nested metadata
    leaves are frozen dataclasses. The nested field is found through
    ``dataclasses.fields`` rather than named, so the property survives a
    rename inside ``microcosm.frame``.
    """

    if field == "metadata":
        nested = context.frame_metadata["provenance"]
        object.__setattr__(
            nested, dataclasses.fields(nested)[0].name, (("source", TAMPERED),)
        )
    elif field == "mass_record":
        object.__setattr__(context.frame_mass_log[0], "reason", TAMPERED)
    elif field == "column_order":
        object.__setattr__(
            context,
            "frame_column_order",
            MappingProxyType(
                {
                    entity: tuple(reversed(columns))
                    for entity, columns in context.frame_column_order.items()
                }
            ),
        )
    elif field in ("column_order_collapse", "column_order_regroup"):
        (first_entity, first_columns), (second_entity, second_columns) = (
            context.frame_column_order.items()
        )
        if field == "column_order_collapse":
            changed = {first_entity: (*first_columns, second_entity, *second_columns)}
        else:
            # Keep two entries and the same flattened string sequence, but
            # reinterpret the first entity's last column as the second entity.
            changed = {
                first_entity: first_columns[:-1],
                first_columns[-1]: (second_entity, *second_columns),
            }
        assert changed != dict(context.frame_column_order)
        assert [
            value for entity, columns in changed.items() for value in (entity, *columns)
        ] == [
            value
            for entity, columns in context.frame_column_order.items()
            for value in (entity, *columns)
        ]
        object.__setattr__(context, "frame_column_order", MappingProxyType(changed))
    else:  # pragma: no cover - guards the fixture itself
        raise AssertionError(f"unknown frame field {field!r}")


class IsolationProbe(toy.ToyKernel):
    """Retains the frame view it is handed, and rewrites views on request.

    ``retain`` keeps this node's own context. ``tamper`` rewrites every
    context retained so far, which happens *after* those nodes' mutation
    checks have already passed -- at that point only detachment keeps the
    live version intact. ``tamper_self`` rewrites one field of this node's
    own context instead, which the before/after comparison has to refuse.
    """

    def __init__(self, ref: str, capabilities, *, variant: str = "base") -> None:
        super().__init__(ref, capabilities, variant=variant)
        self.retained: list[KernelContext] = []
        self.seen: list[dict[str, object]] = []

    def compute(self, context: KernelContext) -> KernelResult:
        if context.params.get("tamper"):
            for held in self.retained:
                rewrite(held, "metadata")
                rewrite(held, "mass_record")
        if context.params.get("retain"):
            self.retained.append(context)
        self.seen.append(observed(context))
        ids = pd.Index(context.tables["person"]["person_id"], name="person_id")
        result = KernelResult(
            columns={
                ("person", str(context.params["target"])): pd.Series(
                    1.0, index=ids, dtype="float64"
                )
            }
        )
        tamper_self = context.params.get("tamper_self")
        if tamper_self is not None:
            rewrite(context, str(tamper_self))
        return result


def build_registry() -> tuple[object, FrameContextProbe]:
    """The toy registry with the metadata source and both probes registered.

    The structural probe is reachable as ``registry.get(REWEIGHT_PROBE_REF)``
    for the properties that are about a boundary rather than a member.
    """
    registry = toy.toy_registry()
    registry.register(MetadataSource(SOURCE_REF, toy._CREATE))
    probe = FrameContextProbe(PROBE_REF, toy._DETERMINISTIC)
    registry.register(probe)
    registry.register(ReweightProbe(REWEIGHT_PROBE_REF, toy._REWEIGHT))
    registry.register(IsolationProbe(ISOLATION_REF, toy._DETERMINISTIC))
    return registry, probe


CREATE = dataclasses.replace(toy.CREATE, kernel=SOURCE_REF)


def probe_node(
    node_id: str,
    *,
    columns: tuple[str, ...],
    target: str,
    mass_reason: str | None = None,
    population: str = "survey",
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
        population=population,
    )


def reweight_probe_node(node_id: str = "boundary", *, base: str = "survey") -> Node:
    """A structural boundary that observes the log it is handed."""
    return Node(
        node_id,
        REWEIGHT_PROBE_REF,
        structural=StructuralDelta.REWEIGHT,
        base=base,
        inputs=(Slice("person", ("age",)), Slice("household", ("household_size",))),
        weights=WeightTransition("household", "importance", mass="free"),
        mass="free",
        description="design -> importance, observing the frame view",
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


# ----------------------------------------------------------------------
# The mass log is the one the node's key binds
# ----------------------------------------------------------------------


#: One ordinary member that appends a ``Frame`` mass record, and one that
#: reads a different column of the same version and declares nothing of the
#: appender's. ``compile_graph`` orders equal-depth nodes by id, so
#: ``appender`` runs first and its record is in the version's cumulative log
#: by the time ``unrelated`` is projected.
APPEND_REASON = "households are unchanged by a derivation"


def sibling_graph(*, appender: bool = True, reason: str = APPEND_REASON) -> Graph:
    """``unrelated`` beside an optional, unread same-version mass appender."""
    unrelated = probe_node("unrelated", columns=("income",), target="unrelated_a")
    if not appender:
        return Graph("toy", (toy.SOURCE,), (CREATE, unrelated))
    return Graph(
        "toy",
        (toy.SOURCE,),
        (
            CREATE,
            probe_node(
                "appender",
                columns=("age",),
                target="appender_a",
                mass_reason=reason,
            ),
            unrelated,
        ),
    )


def boundary_graph() -> Graph:
    """An appender, a structural boundary over it, and a member of the new version."""
    return Graph(
        "toy",
        (toy.SOURCE,),
        (
            CREATE,
            probe_node(
                "appender",
                columns=("age",),
                target="appender_a",
                mass_reason=APPEND_REASON,
            ),
            reweight_probe_node(),
            probe_node(
                "after_boundary",
                columns=("age",),
                target="after_a",
                population="boundary",
            ),
        ),
    )


def boundary_graph_with_reason(reason: str) -> Graph:
    """``boundary_graph`` with the appender stating a different purpose."""
    return Graph(
        "toy",
        (toy.SOURCE,),
        tuple(
            probe_node(
                "appender",
                columns=("age",),
                target="appender_a",
                mass_reason=reason,
            )
            if node.id == "appender"
            else node
            for node in boundary_graph().nodes
        ),
    )


def unrelated_bytes(store: ContentStore, manifest) -> tuple[bytes, bytes]:
    """The stored bytes of ``unrelated``'s output column."""
    key = manifest.nodes["unrelated"].artifacts[("person", "unrelated_a")]
    return toy.artifact_bytes(store, key)


def only_record(view: object) -> MassChangeRecord:
    """The single mass record in an observed view, asserted to be alone."""
    log = tuple(view)  # type: ignore[call-overload]
    assert len(log) == 1
    return log[0]


def test_an_ordinary_node_sees_its_versions_boundary_log(tmp_path: Path) -> None:
    """Not the cumulative log, and specifically not a sibling's record.

    ``appender`` runs first and appends one record to the ``survey``
    version. ``unrelated`` declares no column of the appender's, so its key
    binds ``survey``'s boundary and nothing else; showing it the record
    would be showing it an input its key does not bind.
    """
    _, probe, _, _ = run_probe(tmp_path / "run", graph=sibling_graph())
    assert observation(probe, "appender")["mass_log"] == ()
    assert observation(probe, "unrelated")["mass_log"] == ()


def test_the_appenders_own_record_reaches_the_version_it_leaves(
    tmp_path: Path,
) -> None:
    """The record is applied — it is the *view* that is bounded, not the log."""
    manifest, _, _, _ = run_probe(tmp_path / "run", graph=sibling_graph())
    record = only_record(manifest.population("survey").mass_log)
    assert record.entity == "household"
    assert record.reason == APPEND_REASON
    assert record.old_total == record.new_total


def test_a_keyed_structural_predecessor_does_see_the_cumulative_log(
    tmp_path: Path,
) -> None:
    """A structural node's key binds its base *and* that version's members.

    That is the difference the projection turns on: the boundary may be
    shown what an ordinary member may not, because re-parameterising the
    appender moves the boundary's key and so cannot be replayed onto it.
    """
    registry, probe = build_registry()
    manifest, _, sources, _ = run_probe(
        tmp_path / "run", graph=boundary_graph(), registry=registry, probe=probe
    )
    boundary = registry.get(REWEIGHT_PROBE_REF)
    record = only_record(observation(boundary, "boundary")["mass_log"])
    assert record.reason == APPEND_REASON

    other_registry, _ = build_registry()
    moved = run_graph(
        compile_graph(boundary_graph_with_reason("a different stated reason")),
        sources=dict(sources),
        store=ContentStore(tmp_path / "moved" / "store"),
        kernels=other_registry,
    )
    assert moved.nodes["boundary"].key != manifest.nodes["boundary"].key
    assert moved.nodes["after_boundary"].key != manifest.nodes["after_boundary"].key


def test_a_member_of_the_next_version_sees_the_carried_boundary_log(
    tmp_path: Path,
) -> None:
    """The boundary carries the record forward, so its own members do see it."""
    registry, probe = build_registry()
    run_probe(tmp_path / "run", graph=boundary_graph(), registry=registry, probe=probe)
    record = only_record(observation(probe, "after_boundary")["mass_log"])
    assert record.reason == APPEND_REASON


def test_a_restored_boundary_log_is_the_one_a_cold_member_sees(
    tmp_path: Path,
) -> None:
    """The boundary survives the store round trip, not only the computed frame."""
    registry, probe = build_registry()
    _, _, sources, _ = run_probe(
        tmp_path / "run", graph=boundary_graph(), registry=registry, probe=probe
    )
    expected = observation(probe, "after_boundary")

    later = probe_node(
        "later", columns=("age",), target="later_a", population="boundary"
    )
    warm_registry, warm_probe = build_registry()
    warm = run_graph(
        compile_graph(
            Graph("toy", (toy.SOURCE,), (*boundary_graph().nodes, later)),
        ),
        sources=dict(sources),
        store=ContentStore(tmp_path / "run" / "store"),
        kernels=warm_registry,
    )
    assert {n for n, r in warm.nodes.items() if not r.hit} == {"later"}
    assert observation(warm_probe, "later")["mass_log"] == expected["mass_log"]


def test_an_unrelated_appender_changes_neither_the_key_nor_the_view(
    tmp_path: Path,
) -> None:
    """Same key, same truth: cold with the sibling equals cold without it.

    This is the property the boundary rule exists for. ``unrelated``'s
    output column *is* its mass-log view, so equal stored bytes mean it was
    shown the same log in both graphs — and its key is the same, so the
    store would serve either result for the other.
    """
    with_manifest, with_probe, sources, with_store = run_probe(
        tmp_path / "with", graph=sibling_graph()
    )
    without_manifest, without_probe, _, without_store = run_probe(
        tmp_path / "without", graph=sibling_graph(appender=False), sources=sources
    )
    assert (
        with_manifest.nodes["unrelated"].key == without_manifest.nodes["unrelated"].key
    )
    assert observation(with_probe, "unrelated")["mass_log"] == ()
    assert observation(without_probe, "unrelated")["mass_log"] == ()
    assert unrelated_bytes(with_store, with_manifest) == unrelated_bytes(
        without_store, without_manifest
    )


def test_required_replay_agrees_when_the_sibling_is_added(tmp_path: Path) -> None:
    """A store written with the appender serves the graph without it."""
    _, _, sources, _ = run_probe(tmp_path / "run", graph=sibling_graph())
    registry, probe = build_registry()
    warm = run_graph(
        compile_graph(sibling_graph(appender=False)),
        sources=dict(sources),
        store=ContentStore(tmp_path / "run" / "store"),
        kernels=registry,
        resume="require",
    )
    assert all(receipt.hit for receipt in warm.nodes.values())
    assert probe.seen == []


def test_required_replay_agrees_when_the_sibling_is_reparameterized(
    tmp_path: Path,
) -> None:
    """Re-stating the appender's reason moves its key, and no other node's."""
    first_manifest, _, sources, _ = run_probe(tmp_path / "run", graph=sibling_graph())
    registry, probe = build_registry()
    second = run_graph(
        compile_graph(sibling_graph(reason="a different stated reason")),
        sources=dict(sources),
        store=ContentStore(tmp_path / "run" / "store"),
        kernels=registry,
    )
    assert second.nodes["appender"].key != first_manifest.nodes["appender"].key
    assert second.nodes["unrelated"].key == first_manifest.nodes["unrelated"].key
    assert {n for n, r in second.nodes.items() if not r.hit} == {"appender"}
    assert [item["node"] for item in probe.seen] == ["appender"]


# ----------------------------------------------------------------------
# Isolation: the fields are detached, and rewriting one is refused
# ----------------------------------------------------------------------


def isolation_node(
    node_id: str,
    *,
    target: str,
    retain: bool = False,
    tamper: bool = False,
    tamper_self: str | None = None,
) -> Node:
    """One ordinary member of the ``boundary`` version, with a tamper role."""
    return Node(
        node_id,
        ISOLATION_REF,
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", target, "float64"),),
        params={
            "target": target,
            "retain": retain,
            "tamper": tamper,
            "tamper_self": tamper_self,
        },
        population="boundary",
    )


def isolation_graph(*nodes: Node) -> Graph:
    """``boundary_graph``'s appender and boundary, then the given members.

    The members sit in the ``boundary`` version because that is the version
    whose boundary log is non-empty: a property about rewriting a mass
    record needs a node that was actually handed one.
    """
    return Graph(
        "toy",
        (toy.SOURCE,),
        (
            CREATE,
            probe_node(
                "appender",
                columns=("age",),
                target="appender_a",
                mass_reason=APPEND_REASON,
            ),
            reweight_probe_node(),
            *nodes,
        ),
    )


def run_isolation(root: Path, *nodes: Node):
    """Run ``isolation_graph`` and return ``(manifest, isolation probe)``."""
    registry, probe = build_registry()
    manifest, _, _, _ = run_probe(
        root, graph=isolation_graph(*nodes), registry=registry, probe=probe
    )
    return manifest, registry.get(ISOLATION_REF)


def test_a_retained_context_cannot_rewrite_the_live_version(tmp_path: Path) -> None:
    """Detachment, stated where the mutation check cannot reach.

    ``iso_b_tamper`` rewrites the nested metadata and the mass record of a
    context ``iso_a_retain`` was handed, after that node finished and its
    own before/after comparison passed. The rewrite lands -- the assertions
    on the retained view prove the property is not vacuous -- and reaches
    neither the node that runs next nor the version itself.
    """
    manifest, isolation = run_isolation(
        tmp_path / "run",
        isolation_node("iso_a_retain", target="iso_a", retain=True),
        isolation_node("iso_b_tamper", target="iso_b", tamper=True),
        isolation_node("iso_c_reader", target="iso_c"),
    )
    (held,) = isolation.retained
    assert held.frame_mass_log[0].reason == TAMPERED
    assert held.frame_metadata["provenance"]["source"] == TAMPERED

    reader = observation(isolation, "iso_c_reader")
    assert only_record(reader["mass_log"]).reason == APPEND_REASON
    assert reader["metadata"]["provenance"]["source"] == "frs"
    for version in ("survey", "boundary"):
        assert manifest.population(version).metadata["provenance"]["source"] == "frs"
    assert only_record(manifest.population("boundary").mass_log).reason == APPEND_REASON


def test_rewriting_nested_metadata_is_refused(tmp_path: Path) -> None:
    """A leaf one level below the mapping the context hands out."""
    with pytest.raises(NodeRejectedError, match="mutated its input context"):
        run_isolation(
            tmp_path / "run",
            isolation_node("iso_tamper", target="iso_t", tamper_self="metadata"),
        )


def test_rewriting_a_mass_record_is_refused(tmp_path: Path) -> None:
    """The record is the node's input, not a description of one."""
    with pytest.raises(NodeRejectedError, match="mutated its input context"):
        run_isolation(
            tmp_path / "run",
            isolation_node("iso_tamper", target="iso_t", tamper_self="mass_record"),
        )


def test_rewriting_the_projected_column_order_is_refused(tmp_path: Path) -> None:
    """Reversing an order is a different claim about the version's layout."""
    with pytest.raises(NodeRejectedError, match="mutated its input context"):
        run_isolation(
            tmp_path / "run",
            isolation_node("iso_tamper", target="iso_t", tamper_self="column_order"),
        )


def test_collapsing_entity_column_groups_is_refused(tmp_path: Path) -> None:
    node = isolation_node(
        "iso_collapse", target="iso_t", tamper_self="column_order_collapse"
    )
    node = dataclasses.replace(
        node, inputs=(*node.inputs, Slice("household", ("household_id",)))
    )
    with pytest.raises(NodeRejectedError, match="mutated its input context"):
        run_isolation(tmp_path / "run", node)


def test_moving_a_column_to_an_entity_boundary_is_refused(tmp_path: Path) -> None:
    node = isolation_node(
        "iso_regroup", target="iso_t", tamper_self="column_order_regroup"
    )
    node = dataclasses.replace(
        node, inputs=(*node.inputs, Slice("household", ("household_id",)))
    )
    with pytest.raises(NodeRejectedError, match="mutated its input context"):
        run_isolation(tmp_path / "run", node)


def test_an_untouched_frame_view_still_passes_the_mutation_check(
    tmp_path: Path,
) -> None:
    """The digest additions do not make an ordinary node look mutated."""
    manifest, isolation = run_isolation(
        tmp_path / "run", isolation_node("iso_quiet", target="iso_q")
    )
    assert [item["node"] for item in isolation.seen] == ["iso_quiet"]
    assert manifest.nodes["iso_quiet"].hit is False


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


def test_a_cached_create_preserves_its_nonempty_boundary_log(tmp_path: Path) -> None:
    reason = "source weights reconciled before graph admission"
    source = dataclasses.replace(CREATE, params={"source_mass_reason": reason})
    first = probe_node("first", columns=("age",), target="first_count")
    graph = Graph("toy", (toy.SOURCE,), (source, first))
    cold, probe, sources, store = run_probe(tmp_path / "run", graph=graph)
    expected = observation(probe, "first")["mass_log"]
    assert len(expected) == 1 and expected[0].reason == reason

    registry, next_probe = build_registry()
    second = probe_node("second", columns=("age",), target="second_count")
    warm = run_graph(
        compile_graph(Graph("toy", (toy.SOURCE,), (source, second))),
        sources=dict(sources),
        store=store,
        kernels=registry,
    )
    assert warm.nodes[source.id].hit is True
    assert warm.nodes["second"].hit is False
    assert warm.nodes[source.id].key == cold.nodes[source.id].key
    assert observation(next_probe, "second")["mass_log"] == expected
    assert warm.population(source.id).mass_log == expected


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

    The observer keeps every snapshot and rewrites its tables, its nested
    metadata *and* its mass records after the callback returns. Table
    mutation alone would not touch the amendment-26 fields at all, so it is
    the metadata and mass-record rewrites that make this property about
    them; the run is over ``boundary_graph`` because that is the graph
    whose snapshots carry a mass record to rewrite.
    """
    plain_manifest, plain_probe, sources, _ = run_probe(
        tmp_path / "plain", graph=boundary_graph()
    )
    retained: list[object] = []

    def observe(node_id: str, population) -> None:
        retained.append(population)
        for entity in population.frame.entities:
            table = population.frame.table(entity)
            for column in table.columns:
                table.loc[:, column] = table[column].iloc[0]
        for record in population.frame.mass_log:
            object.__setattr__(record, "reason", TAMPERED)
        nested = population.frame.metadata["provenance"]
        object.__setattr__(
            nested, dataclasses.fields(nested)[0].name, (("source", TAMPERED),)
        )

    observed_manifest, observed_probe, _, _ = run_probe(
        tmp_path / "observed",
        graph=boundary_graph(),
        sources=sources,
        observer=observe,
    )
    assert len(retained) == len(observed_manifest.nodes)
    assert {n: r.key for n, r in observed_manifest.nodes.items()} == {
        n: r.key for n, r in plain_manifest.nodes.items()
    }
    for node_id in ("appender", "after_boundary"):
        assert observation(observed_probe, node_id) == observation(plain_probe, node_id)

    # The rewrites landed on the snapshots, and on nothing else.
    rewritten = [record for snapshot in retained for record in snapshot.frame.mass_log]
    assert rewritten and all(record.reason == TAMPERED for record in rewritten)
    assert all(
        snapshot.frame.metadata["provenance"]["source"] == TAMPERED
        for snapshot in retained
    )
    version = observed_manifest.population("boundary")
    assert version.metadata["provenance"]["source"] == "frs"
    assert only_record(version.mass_log).reason == APPEND_REASON
