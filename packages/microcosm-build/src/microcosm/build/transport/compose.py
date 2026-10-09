"""Compose the transport skeleton graph from a country's spec data.

The country's ``transport_graph.json`` (a ``legacy_json`` spec resource)
declares the skeleton: its sources, its nodes, the execution endpoints and
the outputs a later package still owes. This module turns that declaration
into a :class:`~microcosm.graph.Graph`; it holds no country value.

Node parameters are pure data. A parameter that is a JSON object must be one
explicit selection:

- ``{"resource": name, "path": [...], "encoding": e}`` reads the value at
  ``path`` (a list of object keys) inside the named spec resource. ``e`` is
  ``"value"`` (the JSON value, lists as tuples), ``"json"`` (an object as
  canonical JSON text) or ``"sha256"`` (the SHA-256 of the whole resource's
  bytes, so the node's key moves with that one resource only).
- ``{"prepared": "engine_refs"}`` is the canonical JSON of the authenticated
  binding-id -> engine reference map that preparation built.

So a node binds exactly the resources it selects. The graph resource's own
digest and the country spec's fingerprint enter no node: editing one
resource re-keys only the nodes that select it and their descendants.

A rules node names its binding (``rules_binding``) instead of an engine
reference. Composition supplies the authenticated ``engine_ref`` and the
binding's period, and every rules node runs through the one
``simulate.rules_by_ref@1`` router.

``TransportGraphConfig.extensions`` is the extension point for the package
that adds the entitlement chain. Each extension is a factory called with
the spec, the config and the composed skeleton; it returns new nodes only.
Composition refuses an extension whose nodes change any skeleton node's
compiled predecessors (for example a new member of a calibration base
version), so an extension can never re-key the skeleton.

Composition reads no donor, no facts and no engine, and writes nothing.
Unresolved country evidence is refused up front with every missing item
named (:func:`validate_transport_activation`); no proposed value is chosen.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from microcosm.build.country_spec import load_country_spec
from microcosm.build.graph_atomic_geography import (
    ATOMIC_GEOGRAPHY_VALIDATION_TYPE,
    ATOMIC_SUPPORT_TYPE,
)
from microcosm.calibrate.ordered_kernels import OUTPUT_TYPES
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Graph,
    Node,
    Owned,
    Ownership,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
)
from microcosm.graph.canonical import canonical_json

from .artifact_types import KERNEL_OUTPUTS

__all__ = [
    "GRAPH_RESOURCE",
    "RULES_ROUTER",
    "SKELETON_KERNELS",
    "TransportGraphConfig",
    "TransportGraphExtension",
    "compose_transport_graph",
    "CreateDeclaration",
    "load_transport_spec",
    "skeleton_create_declaration",
    "transport_endpoints",
    "transport_graph_document",
    "transport_pending_outputs",
    "transport_resource",
    "validate_transport_activation",
]

#: The spec resource that declares the skeleton.
GRAPH_RESOURCE = "transport_graph"
#: The single kernel every rules node runs through (one kernel per ref).
RULES_ROUTER = "simulate.rules_by_ref@1"
#: The kernels a skeleton declaration may use. A kernel outside this set
#: (for example an entitlement bridge) enters only through an extension.
SKELETON_KERNELS = frozenset(
    {
        "transport.create@1",
        "transport.boundary@1",
        "transport.currency@1",
        "transport.quantile_map@1",
        "transport.unit_attributes@1",
        "concepts.encode@1",
        "concepts.encode_groups@1",
        RULES_ROUTER,
        "takeup.assign@1",
        "targets.compile@1",
        "targets.problem@1",
        "calibrate.ordered_adam@1",
        "geography.support_from_facts@1",
        "geography.assign_atomic@1",
        "geography.derive@1",
        "geography.gate@1",
        "diagnostics.calibration@1",
        "gates.battery@1",
        "export.prepare@1",
        "export.readback@1",
        "transport.package@1",
    }
)
_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "country",
        "description",
        "sources",
        "nodes",
        "endpoints",
        "pending_outputs",
        "mass_partition",
    }
)
_NODE_FIELDS = frozenset(
    {
        "id",
        "kernel",
        "inputs",
        "outputs",
        "params",
        "population",
        "structural",
        "base",
        "sources",
        "weights",
        "mass",
        "description",
        "citation",
        "artifact_inputs",
        "artifact_outputs",
        "rules_binding",
    }
)
_ENCODINGS = frozenset({"value", "json", "sha256"})
_PREPARED = frozenset({"engine_refs"})
#: Typed outputs each skeleton kernel must declare, when the declaration
#: leaves them implicit.
_OUTPUT_TYPES: Mapping[str, Mapping[str, ArtifactType]] = MappingProxyType(
    {
        **KERNEL_OUTPUTS,
        "calibrate.ordered_adam@1": OUTPUT_TYPES,
        "geography.support_from_facts@1": {"support": ATOMIC_SUPPORT_TYPE},
        "geography.gate@1": {"validation": ATOMIC_GEOGRAPHY_VALIDATION_TYPE},
    }
)


class TransportGraphExtension(Protocol):
    """A factory of new nodes appended to the composed skeleton.

    It receives the spec, the config and the skeleton, and returns a tuple
    of new :class:`Node` objects. Ordinary extension nodes belong in
    populations the extension creates (a FILTER may branch from any
    skeleton population). Composition compares every skeleton node's
    compiled predecessors with and without the extension and refuses any
    change, so a member added to a calibration base version is refused.
    """

    def __call__(
        self,
        spec: Mapping[str, object],
        config: TransportGraphConfig,
        skeleton: Graph,
    ) -> tuple[Node, ...]: ...


@dataclass(frozen=True)
class TransportGraphConfig:
    """What preparation supplies; every country value stays in the spec.

    Attributes:
        engine_refs: Binding id -> authenticated engine reference, from
            :func:`microcosm.build.transport.registry.build_transport_registry`.
        create_outputs: CREATE's declared column inventory, prepared from
            the pinned donor (every loaded column must be declared).
        extensions: Node factories appended after the skeleton, in order.
    """

    engine_refs: Mapping[str, str]
    create_outputs: tuple[Owned, ...]
    extensions: tuple[TransportGraphExtension, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.engine_refs, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in self.engine_refs.items()
        ):
            raise ValueError("engine_refs must map binding ids to authenticated refs.")
        if (
            not isinstance(self.create_outputs, tuple)
            or not self.create_outputs
            or any(not isinstance(item, Owned) for item in self.create_outputs)
        ):
            raise ValueError("create_outputs must declare CREATE's prepared inventory.")
        if not isinstance(self.extensions, tuple) or any(
            not callable(extension) for extension in self.extensions
        ):
            raise TypeError("extensions must be a tuple of node factories.")
        object.__setattr__(
            self, "engine_refs", MappingProxyType(dict(self.engine_refs))
        )


def _mapping(value: object, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object.")
    return value


def _sequence(value: object, label: str) -> Sequence:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a list.")
    return value


def _stem(name: str) -> str:
    return name.removesuffix(".json")


def _resources(spec: Mapping[str, object]) -> Mapping:
    return _mapping(_mapping(spec, "transport spec").get("resources"), "resources")


def _has_resource(spec: Mapping[str, object], name: str) -> bool:
    documents = _resources(spec)
    stem = _stem(name)
    return stem in documents or f"{stem}.json" in documents


def transport_resource(spec: Mapping[str, object], name: str) -> Mapping:
    """One JSON resource of a normalized transport spec.

    A spec is ``{"country", "resources", "resource_hashes"?}``. A resource
    key may be the JSON filename or its stem. ``resource_hashes`` keeps each
    resource's exact file digest (:func:`load_transport_spec` supplies it);
    without it a resource's digest is that of its canonical JSON.
    """

    documents = _resources(spec)
    stem = _stem(name)
    value = documents.get(stem, documents.get(f"{stem}.json"))
    if value is None:
        raise ValueError(f"Transport spec is missing resource {stem}.json.")
    return _mapping(value, f"{stem}.json")


def _resource_digest(spec: Mapping[str, object], name: str) -> str:
    hashes = _mapping(spec.get("resource_hashes", {}), "resource_hashes")
    stem = _stem(name)
    digest = hashes.get(stem, hashes.get(f"{stem}.json"))
    if digest is None:
        return hashlib.sha256(
            canonical_json(transport_resource(spec, name))
        ).hexdigest()
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or not set(digest) <= set("0123456789abcdef")
    ):
        raise ValueError(f"Resource hash for {name!r} must be lowercase SHA-256.")
    return digest


def load_transport_spec(source: str | Path) -> dict[str, object]:
    """Load a country spec's JSON resources and their exact byte digests.

    A string names a packaged country; a :class:`~pathlib.Path` is a local
    spec directory. The country spec loader validates the package first.
    Only local files are read.
    """

    country = load_country_spec(source)
    root = (
        Path(source)
        if isinstance(source, Path)
        else files("microcosm.build").joinpath(source)
    )
    resources = {}
    hashes = {}
    for row in country.resource_rows:
        if row.kind != "legacy_json":
            continue
        payload = root.joinpath(row.path).read_bytes()
        name = _stem(row.path)
        resources[name] = _mapping(json.loads(payload), row.path)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return {
        "country": country.country,
        "resources": resources,
        "resource_hashes": hashes,
    }


def _selections(value: object) -> Iterator[Mapping]:
    """Every explicit selection inside a declared parameter value."""

    if isinstance(value, Mapping):
        yield value
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _selections(item)


def _selected_resources(document: Mapping) -> set[str]:
    names = set()
    for row in document.get("nodes", ()):
        if not isinstance(row, Mapping):
            continue
        params = row.get("params", {})
        if not isinstance(params, Mapping):
            continue
        for value in params.values():
            for selection in _selections(value):
                if isinstance(selection.get("resource"), str):
                    names.add(_stem(selection["resource"]))
    return names


def validate_transport_activation(spec: Mapping[str, object]) -> None:
    """Refuse unresolved country evidence, naming every missing item at once.

    The items are the skeleton declaration and every resource it selects,
    the dependent-child age limit of the benefit-unit rule (``max_age``, or
    the proposal's ``age_limit_years``), any reference not marked active,
    an executable receipt contract (``receipt_contract.json``), the engine
    commit and wheel pins, and a scenario knob still awaiting evidence
    (``rent_stock_factor``). Nothing here picks a value for a missing item.
    """

    missing = []
    for name in ("benefit_unit_rule", "axiom_rules_bindings", "scenarios"):
        if not _has_resource(spec, name):
            missing.append(f"{name}.json")
    if _has_resource(spec, "benefit_unit_rule"):
        unit = transport_resource(spec, "benefit_unit_rule")
        child = _mapping(
            unit.get("dependent_child", {}), "benefit_unit_rule.dependent_child"
        )
        age_key = "max_age" if "max_age" in child else "age_limit_years"
        if child.get(age_key) is None:
            missing.append(f"benefit_unit_rule.dependent_child.{age_key}")
    for resource in ("precal_references", "target_references", "holdout_references"):
        if not _has_resource(spec, resource):
            missing.append(f"{resource}.json")
            continue
        rows = transport_resource(spec, resource).get("target_references")
        for row in _sequence(rows, f"{resource}.target_references"):
            reference = _mapping(row, f"{resource} reference")
            metadata = _mapping(reference.get("metadata", {}), f"{resource}.metadata")
            if metadata.get("activation_status", "active") != "active":
                missing.append(
                    f"placeholder reference {resource}.{reference.get('name', '?')}"
                )
    receipt = (
        transport_resource(spec, "receipt_contract")
        if _has_resource(spec, "receipt_contract")
        else None
    )
    contracts = None if receipt is None else receipt.get("receipts")
    if not isinstance(contracts, Mapping):
        missing.append(
            "receipt_contract.json (the input closure's receipt_requirements "
            "are prose, not an executable contract)"
        )
    elif any(
        not isinstance(row.get("programs"), list) or not row["programs"]
        for row in ((contracts,) if "programs" in contracts else contracts.values())
        if isinstance(row, Mapping)
    ):
        missing.append("receipt_contract.json receipts.programs")
    if _has_resource(spec, "axiom_rules_bindings"):
        bindings = transport_resource(spec, "axiom_rules_bindings")
        pins = _mapping(bindings.get("engine", {}), "axiom_rules_bindings.engine")
        for name in ("commit", "wheel_sha256"):
            if pins.get(name) is None:
                missing.append(f"axiom_rules_bindings.engine.{name}")
    if _has_resource(spec, "scenarios"):
        rows = transport_resource(spec, "scenarios").get("scenarios", ())
        for row in _sequence(rows, "scenarios.scenarios"):
            scenario = _mapping(row, "scenario")
            knobs = _mapping(scenario.get("knobs", {}), "scenario.knobs")
            if "rent_stock_factor" in knobs and knobs["rent_stock_factor"] is None:
                missing.append(f"scenario {scenario.get('id', '?')} rent_stock_factor")
    if not _has_resource(spec, GRAPH_RESOURCE):
        missing.append(f"{GRAPH_RESOURCE}.json (the skeleton declaration)")
    else:
        selected = _selected_resources(transport_resource(spec, GRAPH_RESOURCE))
        missing.extend(
            f"{name}.json (selected by {GRAPH_RESOURCE}.json)"
            for name in sorted(selected)
            if not _has_resource(spec, name)
        )
    if missing:
        raise ValueError("Transport spec is not activated: " + "; ".join(missing) + ".")


def transport_graph_document(spec: Mapping[str, object]) -> Mapping:
    """The skeleton declaration, checked for its schema and country."""

    document = transport_resource(spec, GRAPH_RESOURCE)
    unknown = set(document) - _DOCUMENT_FIELDS
    if unknown:
        raise ValueError(f"{GRAPH_RESOURCE}.json has unknown fields {sorted(unknown)}.")
    if document.get("schema_version") != 1 or document.get("country") != spec.get(
        "country"
    ):
        raise ValueError(
            f"{GRAPH_RESOURCE}.json must declare schema_version 1 and the spec country."
        )
    return document


def transport_endpoints(spec: Mapping[str, object]) -> dict[str, object]:
    """The ordered checkpoints: geography, calibration, then the terminals."""

    values = _mapping(
        transport_graph_document(spec).get("endpoints"), "transport endpoints"
    )
    if set(values) != {"geography", "calibration", "terminal"}:
        raise ValueError(
            "Transport endpoints must be exactly geography, calibration and terminal."
        )
    for name in ("geography", "calibration"):
        if not isinstance(values[name], str) or not values[name]:
            raise ValueError(f"Transport endpoint {name!r} must name one node.")
    terminals = tuple(_sequence(values["terminal"], "terminal endpoints"))
    if not terminals or any(
        not isinstance(name, str) or not name for name in terminals
    ):
        raise ValueError("Terminal endpoints must be a nonempty list of node ids.")
    return {
        "geography": values["geography"],
        "calibration": values["calibration"],
        "terminal": terminals,
    }


def transport_pending_outputs(spec: Mapping[str, object]) -> tuple[str, ...]:
    """Outputs of the full build that this skeleton does not yet produce."""

    rows = transport_graph_document(spec).get("pending_outputs", ())
    values = tuple(_sequence(rows, "pending_outputs"))
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("pending_outputs must be nonempty strings.")
    return values


def _param(
    spec: Mapping[str, object],
    value: object,
    engine_refs: Mapping[str, str] | None,
) -> object:
    if isinstance(value, list | tuple):
        return tuple(_param(spec, item, engine_refs) for item in value)
    if not isinstance(value, Mapping):
        return value
    if set(value) == {"prepared"}:
        if value["prepared"] not in _PREPARED:
            raise ValueError(f"Unknown prepared parameter {value['prepared']!r}.")
        if engine_refs is None:
            raise ValueError("Prepared engine references are not available here.")
        return canonical_json(
            {"engine_refs": dict(sorted(engine_refs.items()))}
        ).decode("utf-8")
    if set(value) - {"resource", "path", "encoding"} or "resource" not in value:
        raise ValueError(
            "An object parameter must be a resource or prepared selection."
        )
    name = value["resource"]
    encoding = value.get("encoding", "value")
    if encoding not in _ENCODINGS:
        raise ValueError(f"Unknown resource encoding {encoding!r}.")
    document: object = transport_resource(spec, name)
    path = tuple(_sequence(value.get("path", ()), "resource selection path"))
    if encoding == "sha256":
        if path:
            raise ValueError("A sha256 selection names a whole resource (no path).")
        return _resource_digest(spec, name)
    for component in path:
        if not isinstance(document, Mapping) or component not in document:
            raise ValueError(f"Resource {name!r} has no selected path {list(path)!r}.")
        document = document[component]
    if document is None:
        raise ValueError(f"Resource {name!r} path {list(path)!r} is not activated.")
    if encoding == "json":
        if not isinstance(document, Mapping):
            raise ValueError(f"Resource {name!r} json selection must be an object.")
        return canonical_json(document).decode("utf-8")
    if isinstance(document, Mapping):
        raise ValueError(
            f"Resource {name!r} value selection is an object; use json encoding."
        )
    return _param(spec, document, engine_refs)


def _artifact_type(value: object) -> ArtifactType:
    document = _mapping(value, "artifact type")
    if set(document) != {"name", "schema_version"}:
        raise ValueError("An artifact type declares exactly name and schema_version.")
    return ArtifactType(document["name"], document["schema_version"])


def _artifact_outputs(row: Mapping) -> tuple[ArtifactOutput, ...]:
    if "artifact_outputs" not in row:
        return tuple(
            ArtifactOutput(name, kind)
            for name, kind in _OUTPUT_TYPES.get(row["kernel"], {}).items()
        )
    return tuple(
        ArtifactOutput(value["name"], _artifact_type(value["type"]))
        for value in _sequence(row["artifact_outputs"], "artifact outputs")
    )


def _rules_params(
    spec: Mapping[str, object],
    row: Mapping,
    params: dict,
    config: TransportGraphConfig,
) -> dict:
    binding_id = row["rules_binding"]
    if row["kernel"] != RULES_ROUTER:
        raise ValueError(f"Rules bindings run through the one {RULES_ROUTER} router.")
    rows = transport_resource(spec, "axiom_rules_bindings").get("bindings", ())
    matches = [
        binding
        for binding in _sequence(rows, "axiom_rules_bindings.bindings")
        if isinstance(binding, Mapping) and binding.get("id") == binding_id
    ]
    if len(matches) != 1 or binding_id not in config.engine_refs:
        raise ValueError(
            f"Rules binding {binding_id!r} needs one declaration and one prepared "
            "engine reference."
        )
    binding = matches[0]
    if set(params) - {"variables"}:
        raise ValueError(
            f"Rules node {row['id']!r} may set only variables; its engine_ref and "
            "period come from its binding."
        )
    declared = tuple(binding["variables"])
    variables = tuple(params.get("variables", declared))
    if (
        not variables
        or len(set(variables)) != len(variables)
        or not set(variables) <= set(declared)
    ):
        raise ValueError(
            f"Rules node {row['id']!r} variables must be distinct binding variables."
        )
    columns = {item["column"] for item in _sequence(row.get("outputs", ()), "outputs")}
    if columns != set(variables):
        raise ValueError(f"Rules node {row['id']!r} must own exactly its variables.")
    return {
        "engine_ref": config.engine_refs[binding_id],
        "period": binding["period"],
        "variables": variables,
    }


def _node(
    spec: Mapping[str, object],
    row: Mapping,
    config: TransportGraphConfig,
    types: Mapping[tuple[str, str], ArtifactType],
) -> Node:
    unknown = set(row) - _NODE_FIELDS
    if unknown:
        raise ValueError(
            f"Transport node {row.get('id')!r} has unknown fields {sorted(unknown)}."
        )
    kernel = row.get("kernel")
    if kernel not in SKELETON_KERNELS:
        raise ValueError(
            f"Kernel {kernel!r} is not a skeleton kernel; add it through an extension."
        )
    params = {
        name: _param(spec, value, config.engine_refs)
        for name, value in _mapping(row.get("params", {}), "node params").items()
    }
    if "rules_binding" in row:
        params = _rules_params(spec, row, params, config)
    elif kernel == RULES_ROUTER:
        raise ValueError("Every skeleton rules node must name its rules_binding.")
    structural = StructuralDelta(row.get("structural", "none"))
    if structural is StructuralDelta.CREATE:
        if row.get("outputs"):
            raise ValueError("CREATE's outputs come from preparation, not the spec.")
        outputs = config.create_outputs
    else:
        outputs = tuple(
            Owned(
                **{
                    **value,
                    "ownership": Ownership(value.get("ownership", "produced")),
                }
            )
            for value in _sequence(row.get("outputs", ()), "owned outputs")
        )
    inputs = tuple(
        Slice(**{**value, "columns": tuple(value["columns"])})
        for value in _sequence(row.get("inputs", ()), "input slices")
    )
    artifacts = []
    for value in _sequence(row.get("artifact_inputs", ()), "artifact inputs"):
        edge = _mapping(value, "artifact input")
        if set(edge) - {"name", "producer", "artifact"}:
            raise ValueError(
                f"Node {row['id']!r}: an artifact input declares name, producer "
                "and optionally artifact; its type is the producer's."
            )
        producer, artifact = edge["producer"], edge.get("artifact", edge["name"])
        if (producer, artifact) not in types:
            raise ValueError(
                f"Node {row['id']!r} reads undeclared artifact {producer}.{artifact}."
            )
        artifacts.append(
            ArtifactInput(edge["name"], producer, artifact, types[producer, artifact])
        )
    weights = row.get("weights")
    return Node(
        id=row["id"],
        kernel=kernel,
        inputs=inputs,
        outputs=outputs,
        params=params,
        population=row.get("population"),
        structural=structural,
        base=row.get("base"),
        sources=tuple(_sequence(row.get("sources", ()), "node sources")),
        weights=None if weights is None else WeightTransition(**weights),
        mass=row.get("mass", "conserve"),
        description=row.get("description", ""),
        citation=row.get("citation", ""),
        artifact_inputs=tuple(artifacts),
        artifact_outputs=_artifact_outputs(row),
    )


@dataclass(frozen=True)
class CreateDeclaration:
    """The declared CREATE node's id, kernel, resolved params and sources."""

    id: str
    kernel: str
    params: Mapping[str, object]
    sources: tuple[str, ...]


def skeleton_create_declaration(spec: Mapping[str, object]) -> CreateDeclaration:
    """The declared CREATE with its parameters resolved, for preparation.

    Preparation runs this kernel once to read CREATE's column inventory,
    which the composed CREATE node must then declare.
    """

    validate_transport_activation(spec)
    rows = [
        row
        for row in _sequence(
            transport_graph_document(spec).get("nodes"), "transport nodes"
        )
        if isinstance(row, Mapping) and row.get("structural") == "create"
    ]
    if len(rows) != 1 or rows[0].get("kernel") != "transport.create@1":
        raise ValueError(
            "A transport skeleton declares exactly one transport.create@1."
        )
    row = rows[0]
    return CreateDeclaration(
        id=row["id"],
        kernel=row["kernel"],
        params=MappingProxyType(
            {
                name: _param(spec, value, None)
                for name, value in _mapping(row.get("params", {}), "params").items()
            }
        ),
        sources=tuple(_sequence(row.get("sources", ()), "node sources")),
    )


def compose_transport_graph(
    spec: Mapping[str, object], config: TransportGraphConfig
) -> Graph:
    """The skeleton declared by the spec, plus any extension branches.

    The result is a pure function of the spec's resources and ``config``:
    composing twice gives equal ``graph_to_json``. Activation is checked
    first, so pending country evidence fails with its items named.
    """

    if not isinstance(config, TransportGraphConfig):
        raise TypeError("config must be a TransportGraphConfig.")
    validate_transport_activation(spec)
    document = transport_graph_document(spec)
    rows = tuple(
        _mapping(row, "transport node")
        for row in _sequence(document.get("nodes"), "transport nodes")
    )
    types = {
        (row["id"], output.name): output.type
        for row in rows
        for output in _artifact_outputs(row)
    }
    sources = tuple(
        SourceRef(**_mapping(row, "transport source"))
        for row in _sequence(document.get("sources"), "transport sources")
    )
    partition = document.get("mass_partition")
    skeleton = Graph(
        country=spec["country"],
        sources=sources,
        nodes=tuple(_node(spec, row, config, types) for row in rows),
        mass_partition=None if partition is None else tuple(partition),
    )
    compiled = compile_graph(skeleton)
    endpoints = transport_endpoints(spec)
    named = {endpoints["geography"], endpoints["calibration"], *endpoints["terminal"]}
    if not named <= set(compiled.order):
        raise ValueError("Transport endpoints must name declared skeleton nodes.")
    transport_pending_outputs(spec)
    additional: list[Node] = []
    for factory in config.extensions:
        nodes = factory(spec, config, skeleton)
        if not isinstance(nodes, tuple) or any(
            not isinstance(node, Node) for node in nodes
        ):
            raise TypeError("A transport extension must return a tuple of Nodes.")
        additional.extend(nodes)
    if not additional:
        return skeleton
    graph = Graph(
        skeleton.country,
        skeleton.sources,
        (*skeleton.nodes, *additional),
        skeleton.mass_partition,
    )
    extended = compile_graph(graph)
    for node in skeleton.nodes:
        if extended.predecessors[node.id] != compiled.predecessors[node.id]:
            raise ValueError(
                f"An extension would re-key skeleton node {node.id!r}; "
                "branch a new FILTER population instead."
            )
    return graph
