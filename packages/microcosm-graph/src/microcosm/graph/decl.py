"""Declarations: the only authority for what a build computes.

Pure, frozen data. No callables, no logic. Everything the executor, the
store, the identity regime, and the documentation know about a build is
derived from one :class:`Graph`. Two kinds of fields exist: normative fields
enter node keys; descriptive fields (:data:`DESCRIPTIVE_FIELDS`) never do.

The population model behind the declarations:

- A ``CREATE`` node turns declared sources into a population version (a
  ``Frame``) and declares every column it loads, so ownership is total from
  the first node. Every other node lives in exactly one population version,
  named by :attr:`Node.population`.
- A structural node (``FILTER``, ``EXPAND``, ``REWEIGHT``) transforms a
  base version into a new version and carries every column of its base;
  it therefore depends on every node of its base version, and a node in
  the new version that reads a carried column depends on the structural
  node, whose artifact is what it actually reads.
- A non-structural node reads declared slices of its version and owns the
  cells it declares. Its predecessors are exactly the owners of the columns
  it reads. Chained imputation is expressed by listing an earlier target's
  column as an input; there is no other ordering mechanism.

This file is a frozen interface (see ``docs/graph-acceptance.md``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import StrEnum
from types import MappingProxyType

__all__ = [
    "DESCRIPTIVE_FIELDS",
    "DTYPES",
    "GATE_OUTCOMES",
    "MASS_POLICIES",
    "ROWS_ALL",
    "WEIGHT_KINDS",
    "CompiledGraph",
    "Graph",
    "GraphError",
    "Node",
    "Owned",
    "Ownership",
    "Param",
    "Slice",
    "SourceRef",
    "StructuralDelta",
    "WeightTransition",
    "compile_graph",
]

type Param = bool | int | float | str | None | tuple["Param", ...]

#: Fields that never enter a node key. Everything else on a declaration is
#: normative.
DESCRIPTIVE_FIELDS = frozenset({"description", "citation"})

#: Closed set of dtype tokens a node may own. The executor maps them to
#: pandas dtypes; ``boolean`` and ``Int64`` are the nullable kinds.
DTYPES = frozenset(
    {"bool", "boolean", "int32", "int64", "Int64", "float32", "float64", "string"}
)

#: Row scope meaning "every row of the entity".
ROWS_ALL = "all"

#: Legal weight kinds, in transition order.
WEIGHT_KINDS = ("design", "importance", "calibrated")

#: Mass policies a weight transition or structural node may declare.
MASS_POLICIES = frozenset({"conserve", "free", "declared"})

#: The closed set of gate outcomes (charter F4). ``unreached`` is also the
#: outcome of a release whose required human decisions are absent.
GATE_OUTCOMES = ("pass", "fail", "evidence_absent", "not_applicable", "unreached")


class GraphError(ValueError):
    """A declaration violates a compile-time invariant."""


class Ownership(StrEnum):
    """What a node promises about the cells it owns."""

    PRODUCED = "produced"
    ABSENT = "absent"


class StructuralDelta(StrEnum):
    """How a node changes the row set of its population version."""

    NONE = "none"
    CREATE = "create"
    FILTER = "filter"
    EXPAND = "expand"
    REWEIGHT = "reweight"


def _check_param(name: str, value: object) -> None:
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GraphError(f"Parameter {name!r} is not finite: {value!r}.")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _check_param(f"{name}[{index}]", item)
        return
    raise GraphError(
        f"Parameter {name!r} has type {type(value).__name__}; parameters are "
        "bool, int, float, str, None, or tuples of those."
    )


def _nonempty(label: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise GraphError(f"{label} must be a non-empty string, got {value!r}.")


@dataclass(frozen=True)
class SourceRef:
    """A named external input, identified by content at run time.

    Attributes:
        name: The name nodes refer to.
        codec: How bytes become a table (a codec registered with the store).
        description: Descriptive; never hashed.
    """

    name: str
    codec: str
    description: str = ""

    def __post_init__(self) -> None:
        _nonempty("SourceRef.name", self.name)
        _nonempty("SourceRef.codec", self.codec)


@dataclass(frozen=True)
class Slice:
    """What a node reads: columns of one entity, optionally under a row mask.

    Attributes:
        entity: The entity table.
        columns: The columns the kernel receives. Nothing else is visible.
        rows: :data:`ROWS_ALL`, or the name of a boolean column of the same
            entity that must also appear in the node's inputs.
    """

    entity: str
    columns: tuple[str, ...]
    rows: str = ROWS_ALL

    def __post_init__(self) -> None:
        _nonempty("Slice.entity", self.entity)
        if not self.columns:
            raise GraphError(f"Slice on {self.entity!r} declares no columns.")
        if len(set(self.columns)) != len(self.columns):
            raise GraphError(f"Slice on {self.entity!r} repeats a column.")
        for column in self.columns:
            _nonempty("Slice.columns[]", column)
        _nonempty("Slice.rows", self.rows)


@dataclass(frozen=True)
class Owned:
    """What a node writes: one column of one entity, at declared positions.

    Attributes:
        entity: The entity table.
        column: The column this node owns.
        dtype: One of :data:`DTYPES`; the executor enforces it exactly.
        rows: :data:`ROWS_ALL`, or a boolean input column naming the owned
            positions. Positions outside the mask are never written.
        ownership: ``PRODUCED`` writes values; ``ABSENT`` asserts null.
    """

    entity: str
    column: str
    dtype: str
    rows: str = ROWS_ALL
    ownership: Ownership = Ownership.PRODUCED

    def __post_init__(self) -> None:
        _nonempty("Owned.entity", self.entity)
        _nonempty("Owned.column", self.column)
        if self.dtype not in DTYPES:
            raise GraphError(
                f"Owned {self.entity}.{self.column}: dtype {self.dtype!r} is not "
                f"one of {sorted(DTYPES)}."
            )
        _nonempty("Owned.rows", self.rows)
        if not isinstance(self.ownership, Ownership):
            raise GraphError("Owned.ownership must be an Ownership value.")


@dataclass(frozen=True)
class WeightTransition:
    """A declared change of weight kind on one entity.

    Attributes:
        entity: The entity whose explicit weights change.
        to_kind: The resulting kind; must be the next kind after the
            current one in :data:`WEIGHT_KINDS`.
        mass: ``conserve`` (total and per-stratum mass unchanged), ``free``
            (any mass, recorded), or ``declared`` (the kernel's receipt
            states the target mass and the executor checks it).
    """

    entity: str
    to_kind: str
    mass: str = "conserve"

    def __post_init__(self) -> None:
        _nonempty("WeightTransition.entity", self.entity)
        if self.to_kind not in WEIGHT_KINDS:
            raise GraphError(
                f"WeightTransition.to_kind {self.to_kind!r} is not one of "
                f"{WEIGHT_KINDS}."
            )
        if self.mass not in MASS_POLICIES:
            raise GraphError(
                f"WeightTransition.mass {self.mass!r} is not one of "
                f"{sorted(MASS_POLICIES)}."
            )


@dataclass(frozen=True)
class Node:
    """One unit of computation and cell ownership.

    Attributes:
        id: Unique within the graph.
        kernel: Kernel reference, e.g. ``"fit.qrf@1"``.
        inputs: Slices the kernel receives. Their owners are this node's
            predecessors.
        outputs: Cells this node owns. A ``CREATE`` node declares every
            column it loads; other structural nodes declare none (they
            carry their base's columns).
        params: Normative parameters; pure data (see :data:`Param`).
        population: Id of the structural node whose row set this node lives
            in. May be omitted only when the graph has exactly one
            structural node.
        structural: The row-set change this node makes; ``NONE`` for
            ordinary nodes.
        base: For a structural node other than ``CREATE``: the population
            version it transforms.
        sources: Names of :class:`SourceRef` entries this node reads.
        weights: A declared weight-kind transition, if any.
        mass: Mass policy for structural nodes that change rows or weights.
        description: Descriptive; never hashed.
        citation: Descriptive; never hashed.
    """

    id: str
    kernel: str
    inputs: tuple[Slice, ...] = ()
    outputs: tuple[Owned, ...] = ()
    params: Mapping[str, Param] = field(default_factory=dict)
    population: str | None = None
    structural: StructuralDelta = StructuralDelta.NONE
    base: str | None = None
    sources: tuple[str, ...] = ()
    weights: WeightTransition | None = None
    mass: str = "conserve"
    description: str = ""
    citation: str = ""

    def __post_init__(self) -> None:
        _nonempty("Node.id", self.id)
        _nonempty("Node.kernel", self.kernel)
        if not isinstance(self.structural, StructuralDelta):
            raise GraphError(f"Node {self.id!r}: structural must be a StructuralDelta.")
        if self.mass not in MASS_POLICIES:
            raise GraphError(f"Node {self.id!r}: mass {self.mass!r} is not legal.")
        for name in sorted(self.params):
            _nonempty("Node.params key", name)
            _check_param(name, self.params[name])
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
        if len({(o.entity, o.column) for o in self.outputs}) != len(self.outputs):
            raise GraphError(f"Node {self.id!r} declares the same owned cell twice.")
        if len(set(self.sources)) != len(self.sources):
            raise GraphError(f"Node {self.id!r} repeats a source.")
        if self.structural is StructuralDelta.CREATE:
            if self.base is not None or self.population is not None:
                raise GraphError(
                    f"Node {self.id!r}: a CREATE node has no base or population."
                )
            if not self.sources:
                raise GraphError(f"Node {self.id!r}: a CREATE node needs sources.")
            if self.inputs:
                raise GraphError(f"Node {self.id!r}: a CREATE node has no inputs.")
            if not self.outputs:
                raise GraphError(
                    f"Node {self.id!r}: a CREATE node must declare every column "
                    "it loads, so ownership is total from the first node."
                )
        elif self.structural is not StructuralDelta.NONE:
            if self.base is None:
                raise GraphError(
                    f"Node {self.id!r}: a {self.structural.value} node needs a base."
                )
            if self.population is not None:
                raise GraphError(
                    f"Node {self.id!r}: a structural node declares base, not "
                    "population."
                )
            if self.outputs:
                raise GraphError(
                    f"Node {self.id!r}: a {self.structural.value} node carries its "
                    "base's columns and owns none of its own."
                )
        elif self.base is not None:
            raise GraphError(f"Node {self.id!r}: only structural nodes have a base.")
        declared_inputs = {(s.entity, c) for s in self.inputs for c in s.columns}
        for s in self.inputs:
            if s.rows != ROWS_ALL and (s.entity, s.rows) not in declared_inputs:
                raise GraphError(
                    f"Node {self.id!r}: row mask {s.rows!r} on {s.entity!r} must be "
                    "one of the node's input columns."
                )
        for o in self.outputs:
            if o.rows != ROWS_ALL and (o.entity, o.rows) not in declared_inputs:
                raise GraphError(
                    f"Node {self.id!r}: owned-row mask {o.rows!r} on {o.entity!r} "
                    "must be one of the node's input columns."
                )
            if (o.entity, o.column) in declared_inputs:
                raise GraphError(
                    f"Node {self.id!r} both reads and owns {o.entity}.{o.column}; "
                    "a node cannot own a column it consumes."
                )

    def normative(self) -> dict[str, object]:
        """The projection that enters the node key (descriptive fields dropped)."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in DESCRIPTIVE_FIELDS
        }


@dataclass(frozen=True)
class Graph:
    """A complete build declaration for one country.

    Attributes:
        country: Descriptive label carried into manifests; never hashed
            into node keys (a node's identity is its computation).
        sources: External inputs by name.
        nodes: Every node. Declaration order carries no meaning.
    """

    country: str
    sources: tuple[SourceRef, ...]
    nodes: tuple[Node, ...]

    def __post_init__(self) -> None:
        _nonempty("Graph.country", self.country)
        if len({s.name for s in self.sources}) != len(self.sources):
            raise GraphError("Graph repeats a source name.")
        if len({n.id for n in self.nodes}) != len(self.nodes):
            raise GraphError("Graph repeats a node id.")

    def node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise GraphError(f"Unknown node {node_id!r}.")


@dataclass(frozen=True)
class CompiledGraph:
    """A validated graph with its derived structure.

    Attributes:
        graph: The declaration.
        order: A canonical topological order (by depth, then id), so two
            declarations of the same nodes compile to the same order.
        owners: ``(version, entity, column)`` to the node that produces that
            column in that version: a ``CREATE`` node for the columns it
            loads, otherwise the owning node.
        predecessors: Node id to the sorted ids it depends on.
        versions: Node id to the population version it lives in (for a
            structural node, the version it creates: its own id).
    """

    graph: Graph
    order: tuple[str, ...]
    owners: Mapping[tuple[str, str, str], str]
    predecessors: Mapping[str, tuple[str, ...]]
    versions: Mapping[str, str]


def compile_graph(graph: Graph) -> CompiledGraph:
    """Validate a graph and derive ownership, predecessors, and order.

    Raises:
        GraphError: A cell with two owners or none (ownership is total and
            exclusive), an unknown source or population, a structural node
            whose base is not structural, a cycle, or a graph with several
            structural nodes and a node that omits ``population``.
    """

    by_id = {node.id: node for node in graph.nodes}
    source_names = {s.name for s in graph.sources}
    structural = [n for n in graph.nodes if n.structural is not StructuralDelta.NONE]
    if not any(n.structural is StructuralDelta.CREATE for n in structural):
        raise GraphError("A graph needs at least one CREATE node.")
    default_version = structural[0].id if len(structural) == 1 else None

    versions: dict[str, str] = {}
    for node in graph.nodes:
        for name in node.sources:
            if name not in source_names:
                raise GraphError(f"Node {node.id!r} reads unknown source {name!r}.")
        if node.structural is not StructuralDelta.NONE:
            if node.base is not None:
                base = by_id.get(node.base)
                if base is None or base.structural is StructuralDelta.NONE:
                    raise GraphError(
                        f"Node {node.id!r}: base {node.base!r} is not a structural "
                        "node."
                    )
            versions[node.id] = node.id
            continue
        version = node.population or default_version
        if version is None:
            raise GraphError(
                f"Node {node.id!r} omits population, and the graph has "
                f"{len(structural)} structural nodes."
            )
        holder = by_id.get(version)
        if holder is None or holder.structural is StructuralDelta.NONE:
            raise GraphError(
                f"Node {node.id!r}: population {version!r} is not a structural node."
            )
        versions[node.id] = version

    owners: dict[tuple[str, str, str], str] = {}
    for node in graph.nodes:
        for owned in node.outputs:
            key = (versions[node.id], owned.entity, owned.column)
            if key in owners:
                raise GraphError(
                    f"{owned.entity}.{owned.column} in version {key[0]!r} is owned "
                    f"by both {owners[key]!r} and {node.id!r}."
                )
            owners[key] = node.id

    def defined(version: str, entity: str, column: str) -> bool:
        """Whether a column exists in ``version``, walking base chains."""
        while True:
            if (version, entity, column) in owners:
                return True
            holder = by_id[version]
            if holder.structural is StructuralDelta.CREATE:
                return False
            version = holder.base  # type: ignore[assignment]

    def reader_of(node_id: str, version: str, entity: str, column: str) -> str:
        """The node whose artifact a reader in ``version`` receives."""
        if not defined(version, entity, column):
            raise GraphError(
                f"Node {node_id!r} reads {entity}.{column}, which no node owns in "
                f"version {version!r} or its bases."
            )
        owner = owners.get((version, entity, column))
        return owner if owner is not None else version

    members: dict[str, set[str]] = {}
    for node_id, version in versions.items():
        if by_id[node_id].structural is StructuralDelta.NONE:
            members.setdefault(version, set()).add(node_id)

    predecessors: dict[str, set[str]] = {n.id: set() for n in graph.nodes}
    for node in graph.nodes:
        if node.structural is StructuralDelta.CREATE:
            continue
        if node.structural is not StructuralDelta.NONE:
            base = node.base
            assert base is not None
            predecessors[node.id].add(base)
            predecessors[node.id].update(members.get(base, ()))
            for s in node.inputs:
                for column in s.columns:
                    predecessors[node.id].add(
                        reader_of(node.id, base, s.entity, column)
                    )
            continue
        version = versions[node.id]
        predecessors[node.id].add(version)
        for s in node.inputs:
            for column in s.columns:
                source = reader_of(node.id, version, s.entity, column)
                if source == node.id:
                    raise GraphError(f"Node {node.id!r} depends on itself.")
                predecessors[node.id].add(source)

    depth: dict[str, int] = {}

    def depth_of(node_id: str, trail: tuple[str, ...]) -> int:
        if node_id in depth:
            return depth[node_id]
        if node_id in trail:
            cycle = " -> ".join((*trail[trail.index(node_id) :], node_id))
            raise GraphError(f"Cycle: {cycle}.")
        preds = predecessors[node_id]
        value = (
            0 if not preds else 1 + max(depth_of(p, (*trail, node_id)) for p in preds)
        )
        depth[node_id] = value
        return value

    for node in graph.nodes:
        depth_of(node.id, ())
    order = tuple(sorted(by_id, key=lambda i: (depth[i], i)))
    return CompiledGraph(
        graph=graph,
        order=order,
        owners=MappingProxyType(owners),
        predecessors=MappingProxyType(
            {i: tuple(sorted(p)) for i, p in predecessors.items()}
        ),
        versions=MappingProxyType(versions),
    )
