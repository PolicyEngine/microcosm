"""Amendment 25: a same-kind weight update is declarable and axis-bound.

``WeightTransition`` only moves a kind forward, so a stage that recomputes
weights it already holds — a sampling normalization, a re-solve of an
existing calibration — had no declaration at all. These properties are
about the one that does: the kind cannot move, mass cannot be free, and
positional replacement values are refused unless the kernel binds the
ordered entity axis they were computed against.

Everything here runs real shared graph operations over the invented toy
country. No country model, engine, or build artifact is involved.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from microcosm.frame import WeightKind, Weights
from microcosm.graph import (
    WEIGHT_UPDATE_AXIS_SCHEMA,
    WEIGHT_UPDATE_MASS_POLICIES,
    ContentStore,
    Graph,
    GraphError,
    KernelResult,
    Node,
    NodeRejectedError,
    Slice,
    StructuralDelta,
    WeightTransition,
    WeightUpdate,
    graph_from_json,
    graph_to_json,
    weight_update_receipt,
)
from microcosm.graph.canonical import normative

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

UPDATE_REF = "reweight.same_kind@1"


class ScaleSameKind(toy.ToyKernel):
    """A REWEIGHT that replaces weight values and keeps their kind.

    ``axis`` selects which ordered axis the kernel claims its positional
    values belong to: the incumbent one, the incumbent one reversed, or
    none at all. ``returns_kind`` lets a property separate "the incumbent
    disagrees with the declaration" from "the kernel disagrees with it".
    """

    def compute(self, context):
        entity = str(context.params["entity"])
        incumbent = context.weights[entity]
        after = incumbent.values * float(context.params["factor"])
        returns = context.params.get("returns_kind")
        kind = incumbent.kind if returns is None else WeightKind(str(returns))
        ids = context.tables[entity][toy.id_column(entity)].tolist()
        receipt: dict[str, object] = {
            "mass": toy._mass_record(
                context, incumbent.values, after, str(context.params["policy"])
            )
        }
        axis = str(context.params.get("axis", "incumbent"))
        if axis == "incumbent":
            receipt["weight_update"] = weight_update_receipt(ids)
        elif axis == "reversed":
            receipt["weight_update"] = weight_update_receipt(list(reversed(ids)))
        elif axis == "short":
            receipt["weight_update"] = weight_update_receipt(ids[:-1])
        elif axis != "absent":  # pragma: no cover - guards the fixture itself
            raise AssertionError(f"unknown axis fixture {axis!r}")
        return KernelResult(weights=Weights(values=after, kind=kind), receipt=receipt)


def registry_with_update():
    """The toy registry plus the same-kind update kernel."""
    registry = toy.toy_registry()
    registry.register(ScaleSameKind(UPDATE_REF, toy._REWEIGHT))
    return registry


def update_node(
    *,
    kind: str = "design",
    reason: str = "reinstall normalized source mass",
    mass: str = "declared",
    factor: float = 2.0,
    axis: str = "incumbent",
    returns_kind: str | None = None,
) -> Node:
    """One same-kind update of the toy household weights, on ``survey``."""
    return Node(
        "update",
        UPDATE_REF,
        structural=StructuralDelta.REWEIGHT,
        base="survey",
        inputs=(Slice("person", ("age",)), Slice("household", ("household_size",))),
        params={
            "entity": "household",
            "factor": factor,
            "policy": mass,
            "axis": axis,
            "returns_kind": returns_kind,
        },
        weights=WeightUpdate("household", kind, reason, mass=mass),
        mass=mass,
        description="same-kind weight update",
    )


def update_graph(**kwargs) -> Graph:
    return Graph("toy", (toy.SOURCE,), (toy.CREATE, update_node(**kwargs)))


def run_update(root: Path, **kwargs) -> object:
    return toy.run_toy(update_graph(**kwargs), root, registry=registry_with_update())


# ----------------------------------------------------------------------
# The declaration
# ----------------------------------------------------------------------


def test_weight_update_declares_a_non_empty_reason() -> None:
    """The stated purpose is required, and it is normative."""
    with pytest.raises(GraphError, match="WeightUpdate.reason"):
        WeightUpdate("household", "design", "")
    one = WeightUpdate("household", "design", "normalize sampled mass")
    other = WeightUpdate("household", "design", "re-solve the calibration")
    assert normative(one) != normative(other)


def test_weight_update_refuses_free_mass() -> None:
    """An update that neither moves kind nor bounds mass records nothing."""
    assert WEIGHT_UPDATE_MASS_POLICIES == frozenset({"conserve", "declared"})
    for mass in sorted(WEIGHT_UPDATE_MASS_POLICIES):
        assert WeightUpdate("household", "design", "why", mass=mass).mass == mass
    with pytest.raises(GraphError, match="free"):
        WeightUpdate("household", "design", "why", mass="free")


def test_weight_update_refuses_an_unknown_kind() -> None:
    with pytest.raises(GraphError, match="WeightUpdate.kind"):
        WeightUpdate("household", "provisional", "why")


def test_weight_update_is_never_a_transition() -> None:
    """Disjoint field sets, so the two can never canonicalize alike.

    ``to_kind`` is a property on the update, which is what keeps it out of
    the normative projection.
    """
    update = WeightUpdate("household", "calibrated", "re-solve", mass="declared")
    transition = WeightTransition("household", "calibrated", mass="declared")
    assert update.to_kind == "calibrated"
    assert set(normative(update)) == {"entity", "kind", "reason", "mass"}
    assert set(normative(transition)) == {"entity", "to_kind", "mass"}
    assert normative(update) != normative(transition)


def test_reweight_node_accepts_either_declaration() -> None:
    """A REWEIGHT node still needs one of them, and rejects a look-alike."""
    with pytest.raises(GraphError, match="REWEIGHT node declares"):
        Node("n", UPDATE_REF, structural=StructuralDelta.REWEIGHT, base="survey")
    with pytest.raises(GraphError, match="look-alike"):
        Node(
            "n",
            UPDATE_REF,
            structural=StructuralDelta.REWEIGHT,
            base="survey",
            weights={"entity": "household", "kind": "design", "reason": "no"},
            mass="declared",
        )


def test_node_mass_must_agree_with_the_update() -> None:
    with pytest.raises(GraphError, match="disagrees"):
        Node(
            "n",
            UPDATE_REF,
            structural=StructuralDelta.REWEIGHT,
            base="survey",
            weights=WeightUpdate("household", "design", "why", mass="declared"),
            mass="conserve",
        )


# ----------------------------------------------------------------------
# JSON round trips
# ----------------------------------------------------------------------


def test_weight_update_round_trips_as_itself() -> None:
    """Not as a transition, which its ``to_kind`` property would allow."""
    graph = update_graph()
    restored = graph_from_json(graph_to_json(graph))
    assert restored == graph
    weights = restored.node("update").weights
    assert isinstance(weights, WeightUpdate)
    assert (weights.entity, weights.kind, weights.mass) == (
        "household",
        "design",
        "declared",
    )
    assert weights.reason == "reinstall normalized source mass"


def test_transition_payload_is_unchanged_by_the_amendment() -> None:
    """Every declaration serialized before amendment 25 restores unchanged."""
    graph = Graph("toy", (toy.SOURCE,), (toy.CREATE, toy.POOL))
    text = graph_to_json(graph)
    assert '"weights":{"entity":"household","mass":"free","to_kind":"importance"}' in (
        text
    )
    restored = graph_from_json(text)
    assert restored == graph
    assert isinstance(restored.node("pool").weights, WeightTransition)


def test_round_trip_refuses_a_mixed_weights_payload() -> None:
    """``kind`` selects the update arm; its fields are then exact."""
    text = graph_to_json(update_graph())
    mixed = text.replace(',"reason":"reinstall normalized source mass"', "")
    with pytest.raises(ValueError, match="weights"):
        graph_from_json(mixed)
    both = text.replace('"kind":"design"', '"kind":"design","to_kind":"design"')
    with pytest.raises(ValueError, match="weights"):
        graph_from_json(both)


def test_axis_receipt_is_a_tagged_digest() -> None:
    """The binding answers one question and does not carry the axis."""
    receipt = weight_update_receipt([3, 1, 2])
    assert set(receipt) == {"schema", "count", "entity_ids_sha256"}
    assert receipt["schema"] == WEIGHT_UPDATE_AXIS_SCHEMA
    assert receipt["count"] == 3
    assert receipt != weight_update_receipt([1, 2, 3])
    assert receipt == weight_update_receipt([3, 1, 2])
    with pytest.raises(ValueError, match="unique"):
        weight_update_receipt([1, 1])
    with pytest.raises(ValueError, match="integers or non-empty strings"):
        weight_update_receipt([True])
    with pytest.raises(ValueError, match="integers or non-empty strings"):
        weight_update_receipt([""])


# ----------------------------------------------------------------------
# The shared graph operation
# ----------------------------------------------------------------------


def test_same_kind_update_replaces_values_and_keeps_the_kind(tmp_path: Path) -> None:
    """The real property: new numbers, same kind, mass declared."""
    run = run_update(tmp_path / "run")
    before = run.manifest.population("survey").weights_for("household")
    after = run.manifest.population("update").weights_for("household")
    assert after.kind is WeightKind.DESIGN is before.kind
    assert list(after.values) == [value * 2.0 for value in before.values]


def test_conserving_update_is_accepted(tmp_path: Path) -> None:
    run = run_update(tmp_path / "run", mass="conserve", factor=1.0)
    weights = run.manifest.population("update").weights_for("household")
    assert weights.kind is WeightKind.DESIGN


def test_update_refuses_a_kind_that_moved(tmp_path: Path) -> None:
    """Declaring a different kind than the incumbent is a transition."""
    with pytest.raises(NodeRejectedError, match="a change of kind is a"):
        run_update(tmp_path / "run", kind="importance", returns_kind="importance")


def test_update_refuses_weights_of_another_kind(tmp_path: Path) -> None:
    """The kernel must also return the kind the node declared."""
    with pytest.raises(NodeRejectedError, match="the kernel returned"):
        run_update(tmp_path / "run", returns_kind="importance")


def test_update_requires_its_ordered_axis(tmp_path: Path) -> None:
    with pytest.raises(NodeRejectedError, match="unverifiable"):
        run_update(tmp_path / "run", axis="absent")


def test_update_refuses_a_foreign_axis(tmp_path: Path) -> None:
    """Same ids, other order: the values would land on the wrong rows."""
    with pytest.raises(NodeRejectedError, match="different .household. axis"):
        run_update(tmp_path / "run", axis="reversed")


def test_update_refuses_a_short_axis(tmp_path: Path) -> None:
    with pytest.raises(NodeRejectedError, match="different .household. axis"):
        run_update(tmp_path / "run", axis="short")


# ----------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------


def test_cold_then_required_replay_revalidates_the_axis(tmp_path: Path) -> None:
    """A hit re-applies the REWEIGHT, so the axis is checked again.

    The cached result is reconstructed with its stored weights and receipt
    and passed back through the same application, which is why replay
    needs no parallel rule. ``resume="require"`` proves the second run read
    the store rather than recomputing.
    """
    cold = run_update(tmp_path / "run")
    assert cold.misses() == set(cold.compiled.order)

    registry = registry_with_update()
    warm = toy.run_toy(
        update_graph(),
        tmp_path / "run",
        sources=cold.sources,
        registry=registry,
        store=ContentStore(tmp_path / "run" / "store"),
        resume="require",
    )
    assert warm.misses() == set()
    assert toy.total_calls(registry) == 0
    assert warm.keys() == cold.keys()
    replayed = warm.manifest.population("update").weights_for("household")
    original = cold.manifest.population("update").weights_for("household")
    assert replayed.kind is original.kind
    assert list(replayed.values) == list(original.values)


def test_reason_is_part_of_the_node_identity(tmp_path: Path) -> None:
    """Two updates that state different purposes are different nodes."""
    first = run_update(tmp_path / "first")
    second = toy.run_toy(
        update_graph(reason="re-solve the calibration"),
        tmp_path / "second",
        registry=registry_with_update(),
    )
    assert first.keys()["survey"] == second.keys()["survey"]
    assert first.keys()["update"] != second.keys()["update"]
