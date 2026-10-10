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
- ``{"prepared": "engine_refs"}`` is the canonical JSON of the prepared
  binding-id -> engine reference map that preparation built.

So a node binds exactly the resources it selects. The graph resource's own
digest and the country spec's fingerprint enter no node: editing one
resource re-keys only the nodes that select the changed data and their
descendants. Selector-shaped objects inside selected resource data remain
literal data; they cannot introduce hidden resource dependencies.
Objects in selected lists remain objects and are subject to the graph's
scalar/tuple parameter boundary; encode an enclosing object as JSON when
structured data must enter a node.

A rules node names its binding (``rules_binding``) instead of an engine
reference. Composition supplies the prepared ``engine_ref`` and the
binding's period, and every rules node runs through the one
``simulate.rules_by_ref@1`` router.

``TransportGraphConfig.extensions`` is the extension point for the package
that adds the entitlement chain. Each extension is a factory called with
the spec, the config and the composed skeleton; it returns new nodes.
Wrap a factory in :class:`TransportExtension` to declare additional sources
and checkpoints, run after calibration and before export. The public
:func:`transport_rules_node` supplies a rules node's engine ref and period
from its binding, including for extension nodes.
Composition refuses an extension whose nodes change any skeleton node's
compiled predecessors (for example a new member of a calibration base
version), so an extension can never re-key the skeleton.

Composition reads no donor, no facts and no engine, and writes nothing.
Activation checks selected resource paths, non-null selected roots and encoding
shapes, the receipt kernel's data-independent contract constraints, reference
activation rows, required age/pin presence and null scenario knobs. Refusals
name every enumerated gap (:func:`validate_transport_activation`). Optional
null JSON fields remain allowed; no proposed value is chosen.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
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
from .column_kernels import validate_receipt_contract

__all__ = [
    "GRAPH_RESOURCE",
    "RULES_ROUTER",
    "SKELETON_KERNELS",
    "TransportGraphConfig",
    "TransportGraphExtension",
    "TransportExtension",
    "compose_transport_graph",
    "CreateDeclaration",
    "load_transport_spec",
    "skeleton_create_declaration",
    "transport_endpoints",
    "transport_graph_document",
    "transport_pending_outputs",
    "transport_resource",
    "transport_rules_node",
    "validate_transport_activation",
    "validate_transport_calibration_ancestry",
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
class TransportExtension:
    """A node factory with additional local sources and ordered checkpoints.

    Sources must have new names, so they cannot replace a skeleton input.
    Checkpoints name nodes in the extended graph and run after calibration,
    before the skeleton H5 is written; they cannot read the exported dataset.
    The factory receives a deep copy of the caller's spec.
    """

    factory: TransportGraphExtension
    sources: tuple[SourceRef, ...] = ()
    checkpoints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError("An extension factory must be callable.")
        if not isinstance(self.sources, tuple) or any(
            not isinstance(source, SourceRef) for source in self.sources
        ):
            raise TypeError("Extension sources must be a tuple of SourceRefs.")
        if not isinstance(self.checkpoints, tuple):
            raise TypeError("Extension checkpoints must be a tuple of node ids.")
        for checkpoint in self.checkpoints:
            _safe_node_id(checkpoint)

    def __call__(
        self,
        spec: Mapping[str, object],
        config: TransportGraphConfig,
        skeleton: Graph,
    ) -> tuple[Node, ...]:
        return self.factory(spec, config, skeleton)


@dataclass(frozen=True)
class TransportGraphConfig:
    """What preparation supplies; every country value stays in the spec.

    Attributes:
        engine_refs: Binding id -> prepared engine reference, from
            :func:`microcosm.build.transport.registry.build_transport_registry`.
        create_outputs: CREATE's declared column inventory, prepared from
            the pinned donor (every loaded column must be declared).
        extensions: Node factories appended after the skeleton, in order.
            A :class:`TransportExtension` also declares sources and checkpoints.
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
            raise ValueError("engine_refs must map binding ids to prepared refs.")
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
    if not isinstance(name, str) or not name:
        raise ValueError("A resource selection must name a nonempty string.")
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


def _declared_resource_selections(document: Mapping) -> Iterator[Mapping]:
    """Resource selections in declaration parameters, including nested lists."""

    for row in document.get("nodes", ()):
        if not isinstance(row, Mapping):
            continue
        params = row.get("params", {})
        if not isinstance(params, Mapping):
            continue
        for value in params.values():
            for selection in _selections(value):
                # Any mapping naming a resource is a selection, whatever the
                # name's type: a malformed name is refused here, not later.
                if "resource" in selection:
                    yield selection


def _null_paths(value: object, path: str = "") -> Iterator[str]:
    if value is None:
        yield path
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _null_paths(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _null_paths(item, f"{path}[{index}]")


def validate_transport_activation(spec: Mapping[str, object]) -> None:
    """Refuse enumerated activation gaps before registry/CREATE/source reads.

    Resolve declaration resource selections with the same path, non-null root
    and encoding checks as parameter resolution. Optional JSON nulls remain
    allowed. Reuse the receipt kernel's data-independent contract validator.
    Also check dependent-child age and required engine pin presence, mandatory
    and selected reference activation rows, and every null scenario knob.
    Selections inside parameter lists count; selected resource data is literal.
    """

    missing = []
    selections = (
        tuple(_declared_resource_selections(transport_resource(spec, GRAPH_RESOURCE)))
        if _has_resource(spec, GRAPH_RESOURCE)
        else ()
    )
    selected = {
        selection["resource"].removesuffix(".json")
        for selection in selections
        if isinstance(selection["resource"], str) and selection["resource"]
    }
    for selection in selections:
        try:
            _resource_param(spec, selection)
        except (ValueError, TypeError) as error:
            missing.append(
                f"Resource {selection['resource']!r} path {selection.get('path', [])!r}: "
                f"{error}"
            )
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
    reference_resources = {
        "precal_references",
        "target_references",
        "holdout_references",
    }
    reference_resources.update(
        name
        for name in selected
        if _has_resource(spec, name)
        and "target_references" in transport_resource(spec, name)
    )
    for resource in sorted(reference_resources):
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
            "receipt_contract.json path ['receipts'] "
            "(the input closure's receipt_requirements "
            "are prose, not an executable contract)"
        )
    else:
        try:
            validate_receipt_contract(contracts)
        except (ValueError, TypeError) as error:
            missing.append(f"receipt_contract.json path ['receipts']: {error}")
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
            for path in _null_paths(knobs):
                missing.append(f"scenario {scenario.get('id', '?')} {path}")
    if not _has_resource(spec, GRAPH_RESOURCE):
        missing.append(f"{GRAPH_RESOURCE}.json (the skeleton declaration)")
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


def _ancestor_closure(
    predecessors: Mapping[str, Sequence[str]], roots: set[str]
) -> set[str]:
    ancestors = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in ancestors:
            continue
        if name not in predecessors:
            raise ValueError(f"Calibration endpoint {name!r} is not a declared node.")
        ancestors.add(name)
        pending.extend(predecessors[name])
    return ancestors


def _embeds_engine_ref(
    value: object,
    references: set[str],
    unit_modules: frozenset[str] = frozenset(),
) -> bool:
    if not references and not unit_modules:
        return False
    if isinstance(value, Mapping):
        if unit_modules:
            from microcosm.frame.adapters.axiom import _ENGINE_REF_SCHEMA

            from .registry import PYTHON_ENGINE_REF_FORMAT

            module = None
            if value.get("format") == _ENGINE_REF_SCHEMA:
                module = value.get("module")
            elif value.get("format") == PYTHON_ENGINE_REF_FORMAT:
                binding = value.get("binding")
                if isinstance(binding, Mapping):
                    module = binding.get("rulespec_path")
            if isinstance(module, str) and module in unit_modules:
                return True
        return any(
            _embeds_engine_ref(item, references, unit_modules)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, list | tuple):
        return any(_embeds_engine_ref(item, references, unit_modules) for item in value)
    if not isinstance(value, str):
        return False
    if any(
        reference in value or json.dumps(reference, ensure_ascii=False)[1:-1] in value
        for reference in references
    ):
        return True
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError):
        decoder = json.JSONDecoder()
        for index, character in enumerate(value):
            if character not in {"{", '"'}:
                continue
            try:
                embedded, _ = decoder.raw_decode(value, index)
            except ValueError:
                continue
            if _embeds_engine_ref(embedded, references, unit_modules):
                return True
        return False
    return decoded != value and _embeds_engine_ref(decoded, references, unit_modules)


def validate_transport_calibration_ancestry(
    spec: Mapping[str, object],
    *,
    graph: Graph | None = None,
    config: TransportGraphConfig | None = None,
) -> None:
    """Refuse hold-out or benefit-unit engine semantics upstream of calibration.

    Without a graph, check the declaration's structural-base membership and
    artifact ancestry. This permits a packaged declaration check before donor
    preparation. Canonical engine-reference documents are matched to the
    benefit-unit bindings' module paths even before references are prepared.
    A composed graph also checks its exact compiled predecessors and the
    prepared references, including references embedded in JSON text.
    Benefit-unit attributes remain allowed; excluded columns are the unit
    rules bindings' outputs and the group encoder's declared outputs.
    """
    if not _has_resource(spec, GRAPH_RESOURCE):
        return
    document = transport_graph_document(spec)
    rows = {
        row["id"]: row
        for row in (
            _mapping(item, "transport node")
            for item in _sequence(document.get("nodes"), "transport nodes")
        )
    }
    unit_entity = transport_resource(spec, "benefit_unit_rule").get("entity")
    bindings = transport_resource(spec, "axiom_rules_bindings").get("bindings", ())
    unit_bindings = {
        row["id"]: row
        for row in _sequence(bindings, "axiom_rules_bindings.bindings")
        if isinstance(row, Mapping) and row.get("entity") == unit_entity
    }
    unit_modules = frozenset(
        row["rulespec_path"]
        for row in unit_bindings.values()
        if isinstance(row.get("rulespec_path"), str)
    )
    forbidden_columns = {
        (unit_entity, variable)
        for row in unit_bindings.values()
        for variable in row.get("variables", ())
    }
    for row in rows.values():
        if row.get("kernel") == "concepts.encode_groups@1":
            forbidden_columns.update(
                (output["entity"], output["column"])
                for output in row.get("outputs", ())
                if output.get("entity") == unit_entity
            )
    references = {
        row["engine_ref"]
        for row in unit_bindings.values()
        if isinstance(row.get("engine_ref"), str) and row["engine_ref"]
    }
    if config is not None:
        references.update(
            config.engine_refs[name]
            for name in unit_bindings
            if name in config.engine_refs
        )
    endpoints = _mapping(document.get("endpoints"), "transport endpoints")
    roots = {
        row["id"]
        for row in rows.values()
        if isinstance(row.get("kernel"), str) and row["kernel"].startswith("calibrate.")
    }
    if isinstance(endpoints.get("calibration"), str):
        roots.add(endpoints["calibration"])
    predecessors = {name: set() for name in rows}
    for name, row in rows.items():
        base = row.get("base")
        if isinstance(base, str):
            predecessors[name].add(base)
            predecessors[name].update(
                item["id"] for item in rows.values() if item.get("population") == base
            )
        elif isinstance(row.get("population"), str):
            predecessors[name].add(row["population"])
        predecessors[name].update(
            edge["producer"] for edge in row.get("artifact_inputs", ())
        )
    for name in _ancestor_closure(predecessors, roots):
        row = rows[name]
        reason = None
        if row.get("rules_binding") in unit_bindings:
            reason = "benefit-unit rules binding"
        for value in _mapping(row.get("params", {}), "node params").values():
            for selection in _selections(value):
                if selection.get("prepared") == "engine_refs":
                    reason = "prepared engine_refs map"
                if isinstance(selection.get("resource"), str):
                    resource = _stem(selection["resource"])
                    if resource == "holdout_references":
                        reason = "holdout_references selection"
                    if (
                        resource == GRAPH_RESOURCE
                        and selection.get("encoding") == "sha256"
                    ):
                        reason = "graph resource's own digest"
                    if selection.get("encoding", "value") != "sha256" and _has_resource(
                        spec, resource
                    ):
                        selected = transport_resource(spec, resource)
                        for component in selection.get("path", ()):
                            if (
                                not isinstance(selected, Mapping)
                                or component not in selected
                            ):
                                selected = None
                                break
                            selected = selected[component]
                        if _embeds_engine_ref(selected, references, unit_modules):
                            reason = "selected data embedding a benefit-unit engine reference"
        columns = {
            (item["entity"], column)
            for item in row.get("inputs", ())
            for column in item["columns"]
        }
        columns.update(
            (item["entity"], item["column"]) for item in row.get("outputs", ())
        )
        if columns & forbidden_columns:
            reason = "benefit-unit engine input or output column"
        if any(
            isinstance(source, str) and "holdout" in source.lower()
            for source in row.get("sources", ())
        ):
            reason = "hold-out source"
        if _embeds_engine_ref(row.get("params", {}), references, unit_modules):
            reason = "embedded benefit-unit engine reference"
        if reason:
            raise ValueError(
                f"Transport calibration ancestry: node {name!r} carries {reason}."
            )
    if graph is None:
        return
    compiled = compile_graph(graph)
    roots.update(
        node.id for node in graph.nodes if node.kernel.startswith("calibrate.")
    )
    for node in graph.nodes:
        if node.kernel == "concepts.encode_groups@1":
            forbidden_columns.update(
                (item.entity, item.column)
                for item in node.outputs
                if item.entity == unit_entity
            )
    for name in _ancestor_closure(compiled.predecessors, roots):
        node = graph.node(name)
        columns = {
            (item.entity, column) for item in node.inputs for column in item.columns
        }
        columns.update((item.entity, item.column) for item in node.outputs)
        if (
            any("holdout" in source.lower() for source in node.sources)
            or _embeds_engine_ref(node.params, references, unit_modules)
            or columns & forbidden_columns
        ):
            raise ValueError(
                f"Transport calibration ancestry: node {name!r} carries hold-out or benefit-unit engine semantics."
            )


def transport_endpoints(
    spec: Mapping[str, object],
    *,
    extensions: tuple[TransportGraphExtension, ...] = (),
) -> dict[str, object]:
    """Geography, calibration, extension checkpoints, then export and terminals."""

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
    checkpoints = tuple(
        checkpoint
        for extension in extensions
        if isinstance(extension, TransportExtension)
        for checkpoint in extension.checkpoints
    )
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Extension checkpoints must name distinct nodes.")
    return {
        "geography": values["geography"],
        "calibration": values["calibration"],
        "terminal": terminals,
        "checkpoints": checkpoints,
    }


def transport_pending_outputs(spec: Mapping[str, object]) -> tuple[str, ...]:
    """Outputs of the full build that this skeleton does not yet produce."""

    rows = transport_graph_document(spec).get("pending_outputs", ())
    values = tuple(_sequence(rows, "pending_outputs"))
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("pending_outputs must be nonempty strings.")
    return values


def _literal_data(value: object) -> object:
    """Convert selected JSON sequences without resolving selector-shaped data."""
    if isinstance(value, list | tuple):
        return tuple(_literal_data(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _literal_data(item) for key, item in value.items()}
    return value


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
        if value["prepared"] in {"fingerprint", "spec_fingerprint"}:
            raise ValueError("The whole-spec fingerprint cannot enter node parameters.")
        if value["prepared"] not in _PREPARED:
            raise ValueError(f"Unknown prepared parameter {value['prepared']!r}.")
        if engine_refs is None:
            raise ValueError("Prepared engine references are not available here.")
        return canonical_json(
            {"engine_refs": dict(sorted(engine_refs.items()))}
        ).decode("utf-8")
    return _resource_param(spec, value)


def _resource_param(spec: Mapping[str, object], value: Mapping) -> object:
    """Resolve and validate one literal resource selection for preflight/params.

    Selected roots cannot be null; JSON roots must be objects, and value roots
    must not be objects. Optional JSON fields remain unchanged.
    """

    if set(value) - {"resource", "path", "encoding"} or "resource" not in value:
        raise ValueError(
            "An object parameter must be a resource or prepared selection."
        )
    name = value["resource"]
    encoding = value.get("encoding", "value")
    if encoding not in _ENCODINGS:
        raise ValueError(f"Unknown resource encoding {encoding!r}.")
    if _stem(name) == GRAPH_RESOURCE and encoding == "sha256":
        raise ValueError(
            "The graph resource's own digest cannot enter node parameters."
        )
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
            raise ValueError(
                f"Resource {name!r} path {list(path)!r} json selection must be an object."
            )
        return canonical_json(document).decode("utf-8")
    if isinstance(document, Mapping):
        raise ValueError(
            f"Resource {name!r} path {list(path)!r} value selection is an object; "
            "use json encoding."
        )
    return _literal_data(document)


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


def transport_rules_node(
    spec: Mapping[str, object],
    config: TransportGraphConfig,
    binding_id: str,
    node: Node,
) -> Node:
    """Bind an extension's rules node to its prepared engine ref and period.

    ``node`` supplies its inputs, owned outputs, population and sources. Its
    params may contain only ``variables`` (a subset of the binding's outputs);
    omitting them selects every binding variable. The node must use the one
    rules router and own exactly the selected variables. Callers supply no
    engine reference or period: both come from the named binding.
    """

    if not isinstance(node, Node):
        raise TypeError("A transport rules node must be a Node.")
    _safe_node_id(node.id)
    declaration = {
        "id": node.id,
        "kernel": node.kernel,
        "rules_binding": binding_id,
        "outputs": [{"column": output.column} for output in node.outputs],
    }
    return replace(
        node,
        params=_rules_params(spec, declaration, dict(node.params), config),
    )


def _check_extension_checkpoints(spec, config, graph, compiled) -> None:
    checkpoints = transport_endpoints(spec, extensions=config.extensions)["checkpoints"]
    if not set(checkpoints) <= set(compiled.order):
        raise ValueError("Extension checkpoints must name nodes in the extended graph.")
    exported = {
        name
        for node in graph.nodes
        if node.kernel == "export.readback@1"
        for name in node.sources
    }
    for name in _ancestor_closure(compiled.predecessors, set(checkpoints)):
        if exported.intersection(graph.node(name).sources):
            raise ValueError("Extension checkpoints must precede export readback.")


def _safe_node_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value == "."
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or ".." in value
    ):
        raise ValueError(
            f"Transport node id {value!r} must be a safe filename without path separators, NUL or '..'."
        )
    return value


def _node(
    spec: Mapping[str, object],
    row: Mapping,
    config: TransportGraphConfig,
    types: Mapping[tuple[str, str], ArtifactType],
) -> Node:
    _safe_node_id(row.get("id"))
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

    Preparation uses the stored CREATE column inventory when its implementation,
    parameters and source bytes match; a cold inventory runs the kernel once.
    The composed CREATE node must declare that inventory.
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
        id=_safe_node_id(row.get("id")),
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
    validate_transport_calibration_ancestry(spec, config=config)
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
    validate_transport_calibration_ancestry(spec, graph=skeleton, config=config)
    endpoints = transport_endpoints(spec)
    named = {endpoints["geography"], endpoints["calibration"], *endpoints["terminal"]}
    if not named <= set(compiled.order):
        raise ValueError("Transport endpoints must name declared skeleton nodes.")
    transport_pending_outputs(spec)
    additional: list[Node] = []
    additional_sources: list[SourceRef] = []
    source_names = {source.name for source in skeleton.sources}
    for factory in config.extensions:
        if isinstance(factory, TransportExtension):
            for source in factory.sources:
                if source.name in source_names:
                    raise ValueError(
                        f"Extension source {source.name!r} must have a new name."
                    )
                source_names.add(source.name)
                additional_sources.append(source)
        nodes = factory(json.loads(canonical_json(spec)), config, skeleton)
        if not isinstance(nodes, tuple) or any(
            not isinstance(node, Node) for node in nodes
        ):
            raise TypeError("A transport extension must return a tuple of Nodes.")
        for node in nodes:
            _safe_node_id(node.id)
        additional.extend(nodes)
    if not additional and not additional_sources:
        _check_extension_checkpoints(spec, config, skeleton, compiled)
        return skeleton
    graph = Graph(
        skeleton.country,
        (*skeleton.sources, *additional_sources),
        (*skeleton.nodes, *additional),
        skeleton.mass_partition,
    )
    extended = compile_graph(graph)
    _check_extension_checkpoints(spec, config, graph, extended)
    validate_transport_calibration_ancestry(spec, graph=graph, config=config)
    for node in skeleton.nodes:
        if extended.predecessors[node.id] != compiled.predecessors[node.id]:
            raise ValueError(
                f"An extension would re-key skeleton node {node.id!r}; "
                "branch a new FILTER population instead."
            )
    return graph
