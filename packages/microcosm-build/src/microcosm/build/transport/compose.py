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

Target compilers also bind the complete ``chronicle_feed.json`` declaration
when the spec carries it, including CREATE's inventory probe and extension
nodes. Other nodes bind exactly the resources they select. The graph resource's own
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
version). Explicit ``package_inputs`` may add extension artifacts to the final
receipt; every other skeleton predecessor set remains fixed.

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
import re
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
from microcosm.build.ledger_artifact import CHRONICLE_FEED_RESOURCE
from microcosm.calibrate.ordered_kernels import OUTPUT_TYPES
from microcosm.graph import (
    ROWS_ALL,
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    CompiledGraph,
    Graph,
    GraphError,
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
from microcosm.graph.expansion import (
    declared_expand_cells,
    materialized_expand_coordinates,
)

from .artifact_types import KERNEL_OUTPUTS
from .column_kernels import validate_receipt_contract
from .target_kernels import TARGET_COMPILATION_KERNELS

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
_PRECALIBRATION_SCHEMA_VERSION = 1
_PRECALIBRATION_PURPOSES = frozenset({"calibration_measure", "classification"})
_PRECALIBRATION_STAGE_FIELDS = frozenset(
    {"id", "rules_binding", "rulespec_path", "outputs", "citation"}
)
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
    ``package_inputs`` explicitly connect typed extension evidence to a
    skeleton package receipt. ``gate_inputs`` connect required extension
    evidence to a skeleton battery, whose verdict already governs export.
    No other skeleton node can gain predecessors.
    The factory receives a deep copy of the caller's spec.
    """

    factory: TransportGraphExtension
    sources: tuple[SourceRef, ...] = ()
    checkpoints: tuple[str, ...] = ()
    package_inputs: tuple[tuple[str, ArtifactInput], ...] = ()
    gate_inputs: tuple[tuple[str, ArtifactInput], ...] = ()

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
        if not isinstance(self.package_inputs, tuple):
            raise TypeError("Extension package inputs must be a tuple.")
        for target, edge in self.package_inputs:
            _safe_node_id(target)
            if not isinstance(edge, ArtifactInput):
                raise TypeError(
                    "Extension package inputs require typed artifact edges."
                )
        if not isinstance(self.gate_inputs, tuple):
            raise TypeError("Extension gate inputs must be a tuple.")
        for target, edge in self.gate_inputs:
            _safe_node_id(target)
            if not isinstance(edge, ArtifactInput):
                raise TypeError("Extension gate inputs require typed artifact edges.")

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


def _target_feed_params(
    spec: Mapping[str, object], kernel: str, params: Mapping[str, object]
) -> dict[str, object]:
    """Bind the declared feed to every target compiler, including preparation."""

    result = dict(params)
    if kernel in TARGET_COMPILATION_KERNELS and _has_resource(
        spec, CHRONICLE_FEED_RESOURCE
    ):
        pin = canonical_json(transport_resource(spec, CHRONICLE_FEED_RESOURCE)).decode(
            "utf-8"
        )
        if "chronicle_feed_pin" in result and result["chronicle_feed_pin"] != pin:
            raise ValueError(
                f"{kernel} Chronicle feed pin differs from {CHRONICLE_FEED_RESOURCE}."
            )
        result["chronicle_feed_pin"] = pin
    return result


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
    Entitlement selections use the factory's instantiated rows, so explicit
    scenario nodes replace their tier templates during preflight too.
    """

    missing = []
    selections = (
        tuple(_declared_resource_selections(transport_resource(spec, GRAPH_RESOURCE)))
        if _has_resource(spec, GRAPH_RESOURCE)
        else ()
    )
    selections = [(selection, False) for selection in selections]
    if _has_resource(spec, "entitlement_graph") and _has_resource(spec, "scenarios"):
        from .entitlement import _rows

        rows, _ = _rows(spec)
        selections.extend(
            (selection, True)
            for selection in _declared_resource_selections(
                {"nodes": [row for row, _ in rows]}
            )
        )
    selected = {
        selection["resource"].removesuffix(".json")
        for selection, _ in selections
        if isinstance(selection["resource"], str) and selection["resource"]
    }
    for selection, allow_json_sequences in selections:
        try:
            _resource_param(spec, selection, allow_json_sequences=allow_json_sequences)
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
    allowed_references: frozenset[str] = frozenset(),
) -> bool:
    if not references and not unit_modules:
        return False
    if isinstance(value, str) and value in allowed_references:
        return False
    if isinstance(value, Mapping):
        if (
            allowed_references
            and canonical_json(value).decode("utf-8") in allowed_references
        ):
            return False
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
            _embeds_engine_ref(item, references, unit_modules, allowed_references)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, list | tuple):
        return any(
            _embeds_engine_ref(item, references, unit_modules, allowed_references)
            for item in value
        )
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
            if _embeds_engine_ref(
                embedded, references, unit_modules, allowed_references
            ):
                return True
        return False
    return decoded != value and _embeds_engine_ref(
        decoded, references, unit_modules, allowed_references
    )


def _unit_entities(spec: Mapping[str, object]) -> tuple[object, ...]:
    """The declared benefit-unit entities (S2-d PR 1 adds every unit kind)."""
    return (transport_resource(spec, "benefit_unit_rule").get("entity"),)


def _is_unit_entity(entity: object, unit_entities: tuple[object, ...]) -> bool:
    # Compare, never hash, so a malformed entity behaves as it always did.
    return any(entity == unit for unit in unit_entities)


_INVENTORY_KEY = re.compile(r'"engine_refs"\s*:')
_JSON_ESCAPE = re.compile(r'\\(?:u[0-9A-Fa-f]{4}|["\\/bfnrt])')
_JSON_ESCAPES = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
# Decoding stops after this many layers. Each layer of JSON string encoding
# doubles a string's backslashes, so no encoded spelling comes near; only a
# crafted chain of escaped backslashes goes deeper, and it counts as carrying
# the inventory.
_INVENTORY_LAYERS = 32


def _json_unescape(match: re.Match[str]) -> str:
    escape = match.group()
    if escape[1] == "u":
        return chr(int(escape[2:], 16))
    return _JSON_ESCAPES.get(escape[1], escape[1])


def _carries_engine_ref_inventory(value: object) -> bool:
    """Whether a value carries a prepared ``engine_refs`` map, in any JSON spelling.

    A string carries it when, with its JSON escapes decoded layer by layer, it
    holds the quoted key and a colon. Decoding a JSON string literal gives the
    same text as decoding its span of the enclosing string, and a key once
    spelled out survives every later layer, so every spelling of the map, at
    any depth of string encoding, is found. Each layer is one pass over the
    string. A string still escaped after ``_INVENTORY_LAYERS`` layers counts
    as carrying the map, so a string costs at most that many passes, plus one.
    """

    if isinstance(value, Mapping):
        return "engine_refs" in value or any(
            _carries_engine_ref_inventory(item)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, list | tuple):
        return any(_carries_engine_ref_inventory(item) for item in value)
    if not isinstance(value, str):
        return False
    for _ in range(_INVENTORY_LAYERS + 1):
        if _INVENTORY_KEY.search(value):
            return True
        value, decoded = _JSON_ESCAPE.subn(_json_unescape, value)
        if not decoded:
            return False
    return True


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_list(value: object, item: object = None) -> bool:
    """A JSON list (never a string), each item passing ``item`` when given."""
    return (
        not isinstance(value, str | bytes)
        and isinstance(value, Sequence)
        and (item is None or all(item(entry) for entry in value))
    )


def _is_rows(value: object, **fields: object) -> bool:
    """A list of objects whose named fields each pass their check."""
    return _is_list(
        value,
        lambda row: isinstance(row, Mapping)
        and all(check(row.get(key)) for key, check in fields.items()),
    )


def _precalibration_params(params: object) -> bool:
    """Whether every resource or binding selection the stage checks read is well formed.

    A resource selection names text, a list path and a known encoding, as
    activation requires. A binding selection (an object with binding and
    field) names both as text; engine-reference documents carry no field.
    """
    if not isinstance(params, Mapping):
        return False
    for value in params.values():
        for selection in _precalibration_selections(value):
            encoding = selection.get("encoding", "value")
            if "resource" in selection and not (
                _is_text(selection["resource"])
                and _is_list(selection.get("path", ()))
                and isinstance(encoding, str)
                and encoding in _ENCODINGS
            ):
                return False
            if (
                "binding" in selection
                and "field" in selection
                and not (
                    _is_text(selection["binding"]) and _is_text(selection["field"])
                )
            ):
                return False
    return True


def _precalibration_bindings(spec: Mapping[str, object]) -> dict[str, Mapping]:
    """The rules bindings, refusing any field the stage checks read malformed.

    The registry validates bindings in full only when it builds engines, after
    these checks run; a string ``variables`` would otherwise be read character
    by character.
    """

    bindings: dict[str, Mapping] = {}
    for item in _sequence(
        transport_resource(spec, "axiom_rules_bindings").get("bindings"),
        "axiom_rules_bindings.bindings",
    ):
        row = _mapping(item, "rules binding")
        name = row.get("id")
        if not _is_text(name):
            raise ValueError(
                f"precalibration_stages needs text rules binding ids, not {name!r}."
            )
        if name in bindings:
            raise ValueError(
                f"precalibration_stages needs distinct rules binding ids; {name!r} "
                "repeats."
            )
        if not (
            _is_text(row.get("entity"))
            and _is_text(row.get("rulespec_path"))
            and _is_list(row.get("variables"), _is_text)
        ):
            raise ValueError(
                f"precalibration_stages needs rules binding {name!r} to name its "
                "entity, rulespec_path and variables."
            )
        bindings[name] = row
    return bindings


def _precalibration_rows(document: Mapping) -> dict[str, Mapping]:
    """The declared nodes, refusing any field the stage checks read malformed.

    The declaration pass runs before the nodes are parsed. Read character by
    character, a string column, source or path list would drop a read, a
    hold-out source or a selection, and a repeated id would hide a node.
    """

    rows: dict[str, Mapping] = {}
    for item in _sequence(document.get("nodes"), "transport nodes"):
        row = _mapping(item, "transport node")
        name = row.get("id")
        if not _is_text(name):
            raise ValueError(
                f"Transport calibration ancestry: node id {name!r} is not text."
            )
        if name in rows:
            raise ValueError(
                f"Transport calibration ancestry: node {name!r} is declared twice."
            )
        checks = {
            "kernel": _is_text(row.get("kernel")),
            # Null is the same as absent for the optional links.
            **{
                field: row.get(field) is None or _is_text(row[field])
                for field in ("population", "base", "rules_binding")
            },
            "structural": row.get("structural", StructuralDelta.NONE)
            in tuple(StructuralDelta),
            "params": _precalibration_params(row.get("params", {})),
            "inputs": _is_rows(
                row.get("inputs", ()),
                entity=_is_text,
                columns=lambda columns: _is_list(columns, _is_text),
            ),
            # An absent mask is every row.
            "outputs": _is_rows(
                row.get("outputs", ()), entity=_is_text, column=_is_text
            )
            and all(
                _is_text(output.get("rows", ROWS_ALL))
                for output in row.get("outputs", ())
            ),
            "artifact_inputs": _is_rows(
                row.get("artifact_inputs", ()), producer=_is_text
            ),
            "sources": _is_list(row.get("sources", ()), _is_text),
        }
        malformed = [field for field, valid in checks.items() if not valid]
        if malformed:
            raise ValueError(
                f"Transport calibration ancestry: node {name!r} has malformed "
                f"{', '.join(malformed)}."
            )
        rows[name] = row
    return rows


def _precalibration_stages(spec: Mapping[str, object]) -> dict[str, Mapping]:
    """Validate stage declarations before inspecting either graph."""

    if not _has_resource(spec, "precalibration_stages"):
        return {}
    resource = transport_resource(spec, "precalibration_stages")
    if set(resource) != {"schema_version", "stages"} or (
        type(resource.get("schema_version")) is not int
        or resource["schema_version"] != _PRECALIBRATION_SCHEMA_VERSION
    ):
        raise ValueError(
            "precalibration_stages must declare exactly schema_version 1 and stages."
        )
    stages = _sequence(resource["stages"], "precalibration_stages.stages")
    if not stages:
        return {}
    if any(
        not isinstance(entity, str) or not entity for entity in _unit_entities(spec)
    ):
        raise ValueError("precalibration_stages needs a named benefit-unit entity.")
    bindings = _precalibration_bindings(spec)
    holdout = (
        transport_resource(spec, "holdout_references").get("target_references", ())
        if _has_resource(spec, "holdout_references")
        else ()
    )
    holdout_columns = {(row["entity"], row["measure"]) for row in holdout}
    holdout_modules = {
        row["rulespec_path"]
        for row in bindings.values()
        if any(
            (row["entity"], variable) in holdout_columns
            for variable in row["variables"]
        )
    }
    result = {}
    identifiers = set()
    for item in stages:
        stage = _mapping(item, "precalibration_stages stage")
        if not _PRECALIBRATION_STAGE_FIELDS <= set(stage) or set(stage) - (
            _PRECALIBRATION_STAGE_FIELDS | {"derived_outputs"}
        ):
            raise ValueError(
                "precalibration_stages stage has missing or unknown fields."
            )
        try:
            identifier = _safe_node_id(stage["id"])
        except ValueError as error:
            raise ValueError(f"precalibration_stages stage id: {error}") from error
        if identifier in identifiers:
            raise ValueError("precalibration_stages stage ids must be distinct.")
        identifiers.add(identifier)
        name = stage["rules_binding"]
        if not isinstance(name, str) or name not in bindings:
            raise ValueError(f"precalibration_stages names unknown binding {name!r}.")
        if name in result:
            raise ValueError("precalibration_stages bindings must be distinct.")
        binding = bindings[name]
        if stage["rulespec_path"] != binding["rulespec_path"]:
            raise ValueError(
                "precalibration_stages rulespec_path must match its binding."
            )
        outputs = _sequence(stage["outputs"], "precalibration_stages outputs")
        derived = _sequence(
            stage.get("derived_outputs", []), "precalibration_stages derived_outputs"
        )
        # A stage whose binding variables stay internal lists only derived ones.
        if not outputs and not derived:
            raise ValueError(
                "precalibration_stages stage declares no outputs or derived outputs."
            )
        columns = set()
        for item in outputs:
            output = _mapping(item, "precalibration_stages output")
            if set(output) != {"entity", "column", "purpose"}:
                raise ValueError(
                    "precalibration_stages output has missing or unknown fields."
                )
            if (
                output["entity"] != binding["entity"]
                or output["column"] not in binding["variables"]
            ):
                raise ValueError(
                    "precalibration_stages output must name a binding variable on its entity."
                )
            if (
                not isinstance(output["purpose"], str)
                or output["purpose"] not in _PRECALIBRATION_PURPOSES
            ):
                raise ValueError("precalibration_stages output has unknown purpose.")
            column = (output["entity"], output["column"])
            if column in columns:
                raise ValueError(
                    "precalibration_stages outputs must be nonempty and distinct."
                )
            columns.add(column)
        for item in derived:
            output = _mapping(item, "precalibration_stages derived output")
            if set(output) != {"entity", "column", "purpose"}:
                raise ValueError(
                    "precalibration_stages derived output has missing or unknown fields."
                )
            if any(
                not isinstance(output[key], str) or not output[key]
                for key in ("entity", "column")
            ):
                raise ValueError(
                    "precalibration_stages derived output must name an entity and a column."
                )
            if (
                output["entity"] == binding["entity"]
                and output["column"] in binding["variables"]
            ):
                raise ValueError(
                    "precalibration_stages derived output is a binding variable; "
                    "list it in outputs."
                )
            if (
                not isinstance(output["purpose"], str)
                or output["purpose"] not in _PRECALIBRATION_PURPOSES
            ):
                raise ValueError(
                    "precalibration_stages derived output has unknown purpose."
                )
            column = (output["entity"], output["column"])
            if column in columns:
                raise ValueError(
                    "precalibration_stages outputs must be nonempty and distinct."
                )
            columns.add(column)
        if columns & holdout_columns:
            raise ValueError("precalibration_stages output is a holdout measure.")
        if stage["rulespec_path"] in holdout_modules:
            raise ValueError("precalibration_stages module produces a holdout measure.")
        result[name] = stage
    return result


def _precalibration_selections(value: object) -> Iterator[Mapping]:
    """Walk nested stage parameters without interpreting selected resource data."""

    for selection in _selections(value):
        yield selection
        for item in selection.values():
            yield from _precalibration_selections(item)


def _precalibration_extension_rows(
    spec: Mapping[str, object], rows: Mapping[str, Mapping]
) -> dict[str, Mapping]:
    """The entitlement rows the extension instantiates, by node id.

    Their parameters are resolved before the composed graph exists, so the
    composed pass reads their selections here, as it reads a skeleton row's.
    """

    if not (
        _has_resource(spec, "entitlement_graph") and _has_resource(spec, "scenarios")
    ):
        return {}
    from .entitlement import _rows

    result: dict[str, Mapping] = {}
    for row, _ in _rows(spec)[0]:
        name = row.get("id")
        if not _is_text(name):
            raise ValueError(
                f"Transport calibration ancestry: node id {name!r} is not text."
            )
        if name in rows or name in result:
            raise ValueError(
                f"Transport calibration ancestry: node {name!r} is declared twice."
            )
        if not _precalibration_params(row.get("params", {})):
            raise ValueError(
                f"Transport calibration ancestry: node {name!r} has malformed params."
            )
        result[name] = row
    return result


def _precalibration_claims(row: Mapping) -> set[tuple[str, str]]:
    """The EXPAND cells a declared node claims, whose values it receives.

    A literal list of ``entity.column`` names is read as the executor reads
    it. Any other value, such as a resource selection, claims every
    non-rewrite output: only composition can tell.
    """

    params = row.get("params", {})
    if "materialized_expand_outputs" not in params:
        return set()
    claimed = params["materialized_expand_outputs"]
    if _is_list(claimed, lambda name: isinstance(name, str) and "." in name):
        return {
            (entity, column)
            for entity, column in (name.split(".", 1) for name in claimed)
        }
    return {
        (output["entity"], output["column"])
        for output in row.get("outputs", ())
        if not output.get("rewrite")
    }


def _precalibration_cells(row: Mapping) -> set[tuple[str, str]] | None:
    """The cells a declared EXPAND may write; None when only composition can tell.

    A literal list of ``[entity, column, dtype]`` texts names them. Any other
    value, such as a resource selection, is unknown.
    """

    if row.get("structural") != StructuralDelta.EXPAND:
        return set()
    cells = row.get("params", {}).get("expand_cells")
    if _is_list(cells, lambda cell: _is_list(cell, _is_text) and len(cell) == 3):
        return {(entity, column) for entity, column, _ in cells}
    return None


def _precalibration_declared_reads(
    rows: Mapping[str, Mapping],
) -> dict[str, set[tuple[str, str, str]]]:
    """Each declared node's column reads, as (entity, column, writer).

    A read's writers are those the executor tracks, over declarations: the
    owner in the reader's population version and, while that owner keeps
    earlier values (a rewrite, a masked owner or an EXPAND claim) or the
    version owns none, the version's EXPAND when it may write the cell, then
    the writers in the base version. A structural node reads its base; a
    rewrite and an EXPAND claim read their own cells. Where the declaration
    leaves the version unknown (no population, an undeclared version or a
    cycle), every declared writer of the column counts. A writer without a
    population counts in every version, and an EXPAND whose cells only
    composition can tell writes every column.
    """

    owners: dict[tuple[object, str, str], set[str]] = {}
    writers: dict[tuple[str, str], set[str]] = {}
    keeps: set[tuple[str, str, str]] = set()
    cells = {name: _precalibration_cells(row) for name, row in rows.items()}
    unknown = {name for name, written in cells.items() if written is None}
    for name, row in rows.items():
        version = row.get("population") if row.get("base") is None else None
        claimed = _precalibration_claims(row)
        for output in row.get("outputs", ()):
            key = (output["entity"], output["column"])
            writers.setdefault(key, set()).add(name)
            owners.setdefault((version, *key), set()).add(name)
            # Rows outside a mask keep the incumbent's values, and a rewrite
            # or claim receives them.
            if (
                output.get("rewrite")
                or output.get("rows", ROWS_ALL) != ROWS_ALL
                or key in claimed
            ):
                keeps.add((name, *key))
        for key in cells[name] or ():
            writers.setdefault(key, set()).add(name)

    def resolve(version: object, key: tuple[str, str]) -> set[str]:
        everyone = writers.get(key, set()) | unknown
        if not isinstance(version, str):
            return everyone
        found = set(owners.get((None, *key), set()))
        seen = set()
        while isinstance(version, str):
            if version in seen or version not in rows:
                return everyone
            seen.add(version)
            owned = owners.get((version, *key), set())
            found |= owned
            if owned and not any((owner, *key) in keeps for owner in owned):
                return found
            if cells[version] is None or key in cells[version]:
                found.add(version)
            version = rows[version].get("base")
        return found

    reads: dict[str, set[tuple[str, str, str]]] = {}
    for name, row in rows.items():
        base = row.get("base")
        start = row.get("population") if base is None else base
        keys = (
            {
                (item["entity"], column)
                for item in row.get("inputs", ())
                for column in item["columns"]
            }
            | {
                (output["entity"], output["column"])
                for output in row.get("outputs", ())
                if output.get("rewrite")
            }
            | _precalibration_claims(row)
        )
        reads[name] = {
            (*key, writer)
            for key in keys
            for writer in resolve(start, key)
            if writer != name
        }
    return reads


def _precalibration_expansions(
    graph: Graph,
) -> tuple[
    dict[str, frozenset[tuple[str, str]]], dict[str, frozenset[tuple[str, str]]]
]:
    """Each compiled node's EXPAND claims, and the cells each EXPAND may write.

    Both are read as the executor reads them, so a malformed one refuses.
    """

    try:
        claims = {
            node.id: materialized_expand_coordinates(node) for node in graph.nodes
        }
        cells = {
            node.id: frozenset(
                (entity, column) for entity, column, _ in declared_expand_cells(node)
            )
            if node.structural is StructuralDelta.EXPAND
            else frozenset()
            for node in graph.nodes
        }
    except GraphError as error:
        raise ValueError(f"Transport calibration ancestry: {error}") from error
    return claims, cells


def _precalibration_compiled_reads(
    graph: Graph,
    compiled: CompiledGraph,
    claims: Mapping[str, frozenset[tuple[str, str]]],
    cells: Mapping[str, frozenset[tuple[str, str]]],
) -> dict[str, set[tuple[str, str, str]]]:
    """Each compiled node's column reads, as (entity, column, writer).

    A read's writers are those the executor tracks: the owner in the reader's
    version and, while that owner keeps earlier values (a rewrite, a masked
    owner or an EXPAND claim) or the version owns none, the version's EXPAND
    when it may write the cell, then the writers in the base version. A
    structural node reads its base; a rewrite and an EXPAND claim read their
    own cells.
    """

    by_id = {node.id: node for node in graph.nodes}

    def resolve(version: str, coordinate: tuple[str, str]) -> set[str]:
        found = set()
        while True:
            holder = by_id[version]
            owner = compiled.owners.get((version, *coordinate))
            if owner is not None:
                found.add(owner)
                output = next(
                    item
                    for item in by_id[owner].outputs
                    if (item.entity, item.column) == coordinate
                )
                if not (
                    output.rewrite
                    or output.rows != ROWS_ALL
                    or coordinate in claims[owner]
                ):
                    return found
            if coordinate in cells[version]:
                found.add(version)
            if holder.structural is StructuralDelta.CREATE or holder.base is None:
                return found
            version = holder.base

    reads: dict[str, set[tuple[str, str, str]]] = {}
    for node in graph.nodes:
        if node.structural is StructuralDelta.CREATE:
            reads[node.id] = set()
            continue
        start = (
            compiled.versions[node.id]
            if node.structural is StructuralDelta.NONE
            else node.base
        )
        coordinates = (
            {(item.entity, column) for item in node.inputs for column in item.columns}
            | {(item.entity, item.column) for item in node.outputs if item.rewrite}
            | claims[node.id]
        )
        reads[node.id] = {
            (*coordinate, writer)
            for coordinate in coordinates
            for writer in resolve(start, coordinate)
            if writer != node.id
        }
    return reads


def _validate_precalibration_ancestry(
    spec: Mapping[str, object],
    stages: Mapping[str, Mapping],
    graph: Graph | None,
    config: TransportGraphConfig | None,
) -> None:
    document = transport_graph_document(spec)
    rows = _precalibration_rows(document)
    bindings = _precalibration_bindings(spec)
    unit_entities = _unit_entities(spec)
    unit_bindings = {
        name: row for name, row in bindings.items() if row["entity"] in unit_entities
    }
    binding_columns = {
        name: {(row["entity"], variable) for variable in row["variables"]}
        for name, row in bindings.items()
    }
    stage_columns = {
        name: {(row["entity"], row["column"]) for row in stage["outputs"]}
        for name, stage in stages.items()
    }
    derived_columns = {
        name: {
            (row["entity"], row["column"]) for row in stage.get("derived_outputs", ())
        }
        for name, stage in stages.items()
    }
    binding_refs = {
        name: {
            reference
            for reference in (
                row.get("engine_ref"),
                None if config is None else config.engine_refs.get(name),
            )
            if isinstance(reference, str) and reference
        }
        for name, row in bindings.items()
    }
    roots = {
        name for name, row in rows.items() if row["kernel"].startswith("calibrate.")
    }
    endpoints = _mapping(document.get("endpoints"), "transport endpoints")
    # Present means a node: an explicit null would drop the only root.
    if "calibration" in endpoints:
        if not _is_text(endpoints["calibration"]):
            raise ValueError(
                "Transport calibration ancestry: the calibration endpoint must "
                "name a node."
            )
        roots.add(endpoints["calibration"])
    predecessors = {name: set() for name in rows}
    reads = _precalibration_declared_reads(rows)
    writes = {
        name: {
            (output["entity"], output["column"]) for output in row.get("outputs", ())
        }
        | (_precalibration_cells(row) or set())
        for name, row in rows.items()
    }
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
        predecessors[name].update(producer for _, _, producer in reads[name])

    declarations = rows

    def check(
        rows: Mapping[str, Mapping],
        predecessors: Mapping,
        reads: Mapping[str, set[tuple[str, str, str]]],
        writes: Mapping[str, set[tuple[str, str]]],
        *,
        composed: bool,
        derived: bool,
    ) -> None:
        node_bindings = {}
        for name, row in rows.items():
            if composed:
                node_bindings[name] = {
                    binding
                    for binding in bindings
                    if config is not None
                    and binding in config.engine_refs
                    and _embeds_engine_ref(
                        row.get("params", {}), {config.engine_refs[binding]}
                    )
                }
            else:
                node_bindings[name] = {
                    selection["binding"]
                    for value in row.get("params", {}).values()
                    for selection in _precalibration_selections(value)
                    if isinstance(selection.get("binding"), str)
                    and selection["binding"] in bindings
                }
                if row.get("rules_binding") in bindings:
                    node_bindings[name].add(row["rules_binding"])
        # Only nodes upstream of calibration are pre-calibration. A stage node
        # there brings its own ancestors into this closure, so each listed
        # stage's ancestry passes the same checks; a stage binding's nodes
        # after calibration (extension checkpoints) are outputs, not inputs.
        ancestors = _ancestor_closure(predecessors, roots)
        # A node produces what it owns and, for an EXPAND, its cells.
        produced = {
            binding: {
                column
                for name in rows
                if name in ancestors and binding in node_bindings[name]
                for column in writes[name]
            }
            for binding in stages
        }
        if derived:
            for binding, stage in stages.items():
                for entity, column in sorted(
                    derived_columns[binding] - produced[binding]
                ):
                    raise ValueError(
                        f"precalibration_stages derived output {entity}.{column} of "
                        f"{stage['id']!r} is not produced by a stage node upstream "
                        "of calibration."
                    )
        # A stage restricts only what its own upstream nodes produce: each read
        # follows its producer in the reader's version lineage, so a column of
        # the same name from another node or version is not the stage's output.
        public = {
            binding: stage_columns[binding] | derived_columns[binding]
            for binding in stages
        }
        consumed = {
            name: {
                binding
                for entity, column, producer in reads[name]
                if producer in ancestors
                for binding in node_bindings.get(producer, ())
                if binding in stages and (entity, column) not in public[binding]
            }
            for name in rows
        }
        inputs = {
            name: {
                (item["entity"], column)
                for item in row.get("inputs", ())
                for column in item["columns"]
            }
            for name, row in rows.items()
        }
        # Outputs are columns, so a stage's artifacts are never declared: only
        # the stage's own nodes may read them.
        artifact_stages = {
            name: {
                binding
                for edge in row.get("artifact_inputs", ())
                if edge["producer"] in ancestors
                for binding in node_bindings.get(edge["producer"], ())
                if binding in stages
            }
            for name, row in rows.items()
        }
        encoded = {
            (output["entity"], output["column"])
            for row in rows.values()
            if row["kernel"] == "concepts.encode_groups@1"
            for output in row.get("outputs", ())
            if output["entity"] in unit_entities
        }
        unit_stages = set(stages).intersection(unit_bindings)
        declared_columns = set().union(*(stage_columns[name] for name in unit_stages))
        unit_columns = set().union(
            *(binding_columns[binding] for binding in unit_bindings)
        )
        for name, row in rows.items():
            if name not in ancestors:
                continue
            named = node_bindings[name]
            own = named.intersection(stages)
            own_units = own.intersection(unit_bindings)
            forbidden = set(unit_bindings) - own_units
            forbidden_refs = set().union(
                *(binding_refs[binding] for binding in forbidden)
            )
            forbidden_modules = frozenset(
                bindings[binding]["rulespec_path"] for binding in forbidden
            )
            allowed_refs = frozenset().union(
                *(binding_refs[binding] for binding in own_units)
            )
            reason = None
            if named.intersection(unit_bindings) - own_units:
                reason = (
                    "mixes listed and unlisted unit bindings"
                    if own_units
                    else "not a declared pre-calibration stage"
                )
            # The prepared map carries every engine: never a stage's own.
            if _carries_engine_ref_inventory(row.get("params", {})):
                reason = "prepared engine_refs map"
            params = (
                declarations.get(name, {}).get("params", {})
                if composed
                else row.get("params", {})
            )
            selections = (
                selection
                for value in params.values()
                for selection in _precalibration_selections(value)
            )
            for selection in selections:
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
                        if _carries_engine_ref_inventory(selected):
                            reason = "prepared engine_refs map"
                        if _embeds_engine_ref(
                            selected,
                            forbidden_refs,
                            forbidden_modules,
                            allowed_refs,
                        ):
                            reason = reason or "not a declared pre-calibration stage"
            outputs = {
                (item["entity"], item["column"]) for item in row.get("outputs", ())
            }
            own_columns = set().union(
                *(binding_columns[binding] for binding in own_units)
            )
            if (
                inputs[name] & unit_columns - own_columns - declared_columns
                or outputs & unit_columns - own_columns
                or (inputs[name] & encoded and not own_units)
            ):
                reason = reason or "not a declared pre-calibration stage"
            if row["kernel"] == "concepts.encode_groups@1":
                for column in outputs & encoded:
                    consumers = [
                        consumer
                        for consumer in rows
                        if consumer in ancestors and column in inputs[consumer]
                    ]
                    if not consumers:
                        reason = reason or "not a declared pre-calibration stage"
            elif outputs & encoded:
                reason = reason or "not a declared pre-calibration stage"
            if (consumed[name] | artifact_stages[name]) - own:
                reason = "undeclared pre-calibration stage output"
            if _embeds_engine_ref(
                row.get("params", {}),
                forbidden_refs,
                forbidden_modules,
                allowed_refs,
            ):
                reason = reason or "not a declared pre-calibration stage"
            if any(
                isinstance(source, str) and "holdout" in source.lower()
                for source in row.get("sources", ())
            ):
                reason = "hold-out source"
            if reason:
                raise ValueError(
                    f"Transport calibration ancestry: node {name!r} carries {reason}."
                )

    # A derived output is checked against the fullest view: the composed graph
    # when there is one, otherwise the declaration.
    check(rows, predecessors, reads, writes, composed=False, derived=graph is None)
    if graph is None:
        return
    declarations = {**rows, **_precalibration_extension_rows(spec, rows)}
    compiled = compile_graph(graph)
    claims, cells = _precalibration_expansions(graph)
    roots.update(
        node.id for node in graph.nodes if node.kernel.startswith("calibrate.")
    )
    compiled_rows = {
        node.id: {
            "kernel": node.kernel,
            "inputs": [
                {"entity": item.entity, "columns": item.columns} for item in node.inputs
            ],
            "outputs": [
                {"entity": item.entity, "column": item.column} for item in node.outputs
            ],
            "params": node.params,
            "sources": node.sources,
            "artifact_inputs": [
                {"producer": edge.producer} for edge in node.artifact_inputs
            ],
        }
        for node in graph.nodes
    }
    check(
        compiled_rows,
        compiled.predecessors,
        _precalibration_compiled_reads(graph, compiled, claims, cells),
        {
            node.id: {(item.entity, item.column) for item in node.outputs}
            | cells[node.id]
            for node in graph.nodes
        },
        composed=True,
        derived=True,
    )


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
    Declared pre-calibration stages admit their own unit semantics, restrict
    outputs flowing onward and recursively check each stage's ancestry. With
    stages and a graph, each derived output must be produced in the composed
    graph; without a graph, in the declaration. Composition always runs the
    graph-less check first.
    """
    stages = _precalibration_stages(spec)
    if not _has_resource(spec, GRAPH_RESOURCE):
        return
    if stages:
        _validate_precalibration_ancestry(spec, stages, graph, config)
        return
    document = transport_graph_document(spec)
    rows = {
        row["id"]: row
        for row in (
            _mapping(item, "transport node")
            for item in _sequence(document.get("nodes"), "transport nodes")
        )
    }
    unit_entities = _unit_entities(spec)
    bindings = transport_resource(spec, "axiom_rules_bindings").get("bindings", ())
    unit_bindings = {
        row["id"]: row
        for row in _sequence(bindings, "axiom_rules_bindings.bindings")
        if isinstance(row, Mapping)
        and _is_unit_entity(row.get("entity"), unit_entities)
    }
    unit_modules = frozenset(
        row["rulespec_path"]
        for row in unit_bindings.values()
        if isinstance(row.get("rulespec_path"), str)
    )
    forbidden_columns = {
        (row.get("entity"), variable)
        for row in unit_bindings.values()
        for variable in row.get("variables", ())
    }
    for row in rows.values():
        if row.get("kernel") == "concepts.encode_groups@1":
            forbidden_columns.update(
                (output["entity"], output["column"])
                for output in row.get("outputs", ())
                if _is_unit_entity(output.get("entity"), unit_entities)
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
                if _is_unit_entity(item.entity, unit_entities)
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

    from .entitlement import transport_extensions

    extensions = transport_extensions(spec, extensions)
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


def _resource_param(
    spec: Mapping[str, object], value: Mapping, *, allow_json_sequences: bool = False
) -> object:
    """Resolve and validate one literal resource selection for preflight/params.

    Selected roots cannot be null; skeleton JSON roots must be objects.
    Entitlement JSON may also select lists. Value roots must not be objects.
    Optional JSON fields remain unchanged.
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
        if not isinstance(document, Mapping) and not (
            allow_json_sequences and isinstance(document, list | tuple)
        ):
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
    *,
    allowed_kernels: frozenset[str] = SKELETON_KERNELS,
    resolved_params: Mapping[str, object] | None = None,
) -> Node:
    _safe_node_id(row.get("id"))
    unknown = set(row) - _NODE_FIELDS
    if unknown:
        raise ValueError(
            f"Transport node {row.get('id')!r} has unknown fields {sorted(unknown)}."
        )
    kernel = row.get("kernel")
    if kernel not in allowed_kernels:
        raise ValueError(
            f"Kernel {kernel!r} is not a skeleton kernel; add it through an extension."
        )
    params = (
        dict(resolved_params)
        if resolved_params is not None
        else {
            name: _param(spec, value, config.engine_refs)
            for name, value in _mapping(row.get("params", {}), "node params").items()
        }
    )
    params = _target_feed_params(spec, kernel, params)
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
            _target_feed_params(
                spec,
                row["kernel"],
                {
                    name: _param(spec, value, None)
                    for name, value in _mapping(row.get("params", {}), "params").items()
                },
            )
        ),
        sources=tuple(_sequence(row.get("sources", ()), "node sources")),
    )


def compose_transport_graph(
    spec: Mapping[str, object], config: TransportGraphConfig
) -> Graph:
    """The skeleton declared by the spec, plus any extension branches.

    An optional entitlement graph resource installs its extension automatically.
    The result is a pure function of the spec's resources and ``config``:
    composing twice gives equal ``graph_to_json``. Activation is checked
    first, so pending country evidence fails with its items named.
    """

    if not isinstance(config, TransportGraphConfig):
        raise TypeError("config must be a TransportGraphConfig.")
    validate_transport_activation(spec)
    from .entitlement import transport_extensions

    config = replace(config, extensions=transport_extensions(spec, config.extensions))
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
    package_inputs: dict[str, list[ArtifactInput]] = {}
    gate_inputs: dict[str, list[ArtifactInput]] = {}
    source_names = {source.name for source in skeleton.sources}
    for factory in config.extensions:
        if isinstance(factory, TransportExtension):
            for target, edge in factory.gate_inputs:
                if (
                    target not in compiled.order
                    or skeleton.node(target).kernel != "gates.battery@1"
                ):
                    raise ValueError(
                        "Extension gate evidence may target only a skeleton battery node."
                    )
                gate_inputs.setdefault(target, []).append(edge)
            for target, edge in factory.package_inputs:
                if (
                    target not in compiled.order
                    or skeleton.node(target).kernel != "transport.package@1"
                ):
                    raise ValueError(
                        "Extension artifact edges may target only a skeleton package node."
                    )
                if any(
                    target in predecessors
                    for predecessors in compiled.predecessors.values()
                ):
                    raise ValueError(
                        "Extension package inputs require a terminal skeleton package node."
                    )
                package_inputs.setdefault(target, []).append(edge)
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
        additional.extend(
            replace(node, params=_target_feed_params(spec, node.kernel, node.params))
            for node in nodes
        )
    if (
        not additional
        and not additional_sources
        and not package_inputs
        and not gate_inputs
    ):
        _check_extension_checkpoints(spec, config, skeleton, compiled)
        return skeleton
    skeleton_nodes = []
    for node in skeleton.nodes:
        edges = (
            *package_inputs.get(node.id, ()),
            *gate_inputs.get(node.id, ()),
        )
        aliases = [edge.name for edge in (*node.artifact_inputs, *edges)]
        if len(aliases) != len(set(aliases)):
            raise ValueError(
                "Extension package artifact aliases must be new and distinct."
            )
        if any(edge.producer not in {item.id for item in additional} for edge in edges):
            raise ValueError("Extension package inputs must come from extension nodes.")
        skeleton_nodes.append(
            replace(node, artifact_inputs=(*node.artifact_inputs, *edges))
        )
    graph = Graph(
        skeleton.country,
        (*skeleton.sources, *additional_sources),
        (*skeleton_nodes, *additional),
        skeleton.mass_partition,
    )
    extended = compile_graph(graph)
    _check_extension_checkpoints(spec, config, graph, extended)
    validate_transport_calibration_ancestry(spec, graph=graph, config=config)
    for node in skeleton.nodes:
        expected = (
            set(compiled.predecessors[node.id])
            | {edge.producer for edge in package_inputs.get(node.id, ())}
            | {edge.producer for edge in gate_inputs.get(node.id, ())}
        )
        if set(extended.predecessors[node.id]) != expected:
            raise ValueError(
                f"An extension would re-key skeleton node {node.id!r}; "
                "branch a new FILTER population instead."
            )
    return graph
