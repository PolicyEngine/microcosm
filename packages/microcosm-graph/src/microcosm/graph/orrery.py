"""Adapter from compiler metadata and recorded evidence to Orrery.

Static export never reads population data. The optional execution command
validates saved manifests and store bytes; it never executes a kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .canonical import canonical_json
from .decl import CompiledGraph, Graph, compile_graph
from .errors import GraphRuntimeError
from .orrery_evidence import apply_contracts
from .presentation import presentation_id, validate_presentation
from .schema import graph_schema, validate_graph_schema
from .serialize import graph_from_json

__all__ = [
    "orrery_document",
    "orrery_document_from_schema",
    "orrery_json",
    "orrery_json_from_schema",
]

_SAFE_INTEGER = 2**53 - 1
_MAX_INPUT_BYTES = 32 * 1024 * 1024
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_ITEMS = 2_000_000
_MAX_NODES = 20_000
_MAX_EDGES = 100_000
_FIELD_KEYS = ("population", "entity", "column", "provider", "declared_in")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Orrery adapter: {message}.")


def _transport_json(value: object) -> object:
    """Detach JSON and tag numeric values JavaScript cannot preserve exactly."""

    items = 0
    charge = 0

    def walk(child: object, depth: int) -> object:
        nonlocal items, charge
        items += 1
        charge += 16
        _require(items <= _MAX_ITEMS and depth <= 64, "metadata complexity")
        kind = type(child)
        if child is None or kind is bool:
            result = child
        elif kind is str:
            _require(len(child) <= 16_384, "string length")
            charge += len(child.encode("utf-8"))
            result = child
        elif kind is int:
            _require(child.bit_length() <= 1024, "integer size")
            charge += child.bit_length()
            result = (
                {"integer_literal": str(child)} if abs(child) > _SAFE_INTEGER else child
            )
        elif kind is float:
            _require(math.isfinite(child), "finite numbers required")
            result = (
                {"float_literal": repr(child)}
                if child.is_integer() and abs(child) > _SAFE_INTEGER
                else child
            )
        elif kind is list:
            result = [walk(item, depth + 1) for item in child]
        elif kind is dict:
            _require(all(type(key) is str for key in child), "string keys required")
            result = {
                walk(key, depth + 1): walk(item, depth + 1)
                for key, item in child.items()
            }
        else:
            raise ValueError("Orrery adapter: plain JSON values required.")
        _require(charge <= _MAX_OUTPUT_BYTES, "prospective metadata size")
        return result

    return walk(value, 0)


def _object(value: object, label: str) -> dict[str, object]:
    _require(type(value) is dict, f"{label} must be an object")
    return value  # type: ignore[return-value]


def _array(value: object, label: str) -> list[object]:
    _require(type(value) is list, f"{label} must be an array")
    return value  # type: ignore[return-value]


def _text(value: object, label: str) -> str:
    _require(type(value) is str and bool(value.strip()), f"{label} must be text")
    return value  # type: ignore[return-value]


def _index(values: object, key: str, label: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for item in _array(values, label):
        record = _object(item, f"{label} item")
        identity = _text(record.get(key), f"{label} {key}")
        _require(identity not in result, f"duplicate {label} {key}")
        result[identity] = record
    return result


def _field_id(field: dict[str, object]) -> str:
    return presentation_id(
        "field", *(_text(field.get(key), f"field {key}") for key in _FIELD_KEYS)
    )


def _revision(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _bounded_json(value: object, limit: int) -> str:
    parts: list[str] = []
    size = 0
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    for part in encoder.iterencode(value):
        size += len(part.encode("utf-8"))
        _require(size <= limit, "serialized metadata size")
        parts.append(part)
    return "".join(parts)


def _owned_declarations(
    operations: dict[str, dict[str, object]],
) -> dict[tuple[str, str, str], dict[str, object]]:
    result: dict[tuple[str, str, str], dict[str, object]] = {}
    for node_id, operation in operations.items():
        for value in _array(operation.get("outputs"), f"operation {node_id} outputs"):
            owned = _object(value, f"operation {node_id} output")
            key = (
                node_id,
                _text(owned.get("entity"), "output entity"),
                _text(owned.get("column"), "output column"),
            )
            _require(key not in result, "duplicate Owned declaration")
            result[key] = {**owned, "rewrite": owned.get("rewrite", False)}
    return result


def orrery_document_from_schema(
    schema: object, *, title: str | None = None, execution: object | None = None
) -> dict[str, object]:
    """Transform compiler metadata into a complete Orrery document.

    The schema is recompiled before projection. Stable node and edge identities
    are derived from declared coordinates and relationships. Revisions are
    content digests, not runtime cache keys or verification claims.
    """

    document = validate_graph_schema(schema)
    graph = _object(document["graph"], "graph")
    compiled = _object(document["compiled"], "compiled")
    operations = _index(graph.get("nodes"), "id", "operations")
    sources = _index(graph.get("sources"), "name", "sources")
    versions = _object(compiled.get("versions"), "compiled versions")
    predecessors = _object(compiled.get("predecessors"), "compiled predecessors")
    declarations = _owned_declarations(operations)
    transport_document = _transport_json(document)

    nodes: dict[str, dict[str, object]] = {}
    edges: dict[str, dict[str, object]] = {}
    used = (
        len(_bounded_json(transport_document, _MAX_OUTPUT_BYTES).encode("utf-8"))
        + 65_536
    )

    def presentation_record(record: dict[str, object]) -> dict[str, object]:
        nonlocal used
        transported = _object(_transport_json(record), "presentation record")
        transported["revision"] = _revision(transported)
        encoded = _bounded_json(transported, _MAX_OUTPUT_BYTES - used)
        used += len(encoded.encode("utf-8")) + 1
        return transported

    def add_node(node: dict[str, object]) -> None:
        identity = _text(node.get("id"), "presentation node id")
        _require(identity not in nodes, "duplicate presentation node")
        _require(len(nodes) < _MAX_NODES, "node limit")
        nodes[identity] = presentation_record(node)

    def add_edge(
        source: str,
        target: str,
        kind: str,
        category: str,
        data: dict[str, object] | None = None,
    ) -> None:
        facts = {} if data is None else data
        identity = presentation_id(
            "edge", kind, source, target, _bounded_json(facts, _MAX_OUTPUT_BYTES)
        )
        if identity in edges:
            return
        _require(len(edges) < _MAX_EDGES, "edge limit")
        edges[identity] = presentation_record(
            {
                "id": identity,
                "source": source,
                "target": target,
                "kind": kind,
                "category": category,
                "data": facts,
            }
        )

    for node_id, operation in operations.items():
        node: dict[str, object] = {
            "id": presentation_id("operation", node_id),
            "label": node_id,
            "kind": "operation",
            "data": {
                "declaration": operation,
                "population": versions[node_id],
            },
        }
        if operation.get("description"):
            node["description"] = operation["description"]
        add_node(node)
    for source_name, source in sources.items():
        node = {
            "id": presentation_id("source", source_name),
            "label": source_name,
            "kind": "source",
            "data": {"declaration": source},
        }
        if source.get("description"):
            node["description"] = source["description"]
        add_node(node)

    fields: dict[str, dict[str, object]] = {}
    visible: dict[tuple[str, str, str], str] = {}

    def add_field(field: dict[str, object], *, visible_field: bool) -> str:
        identity = _field_id(field)
        declared_in = _text(field.get("declared_in"), "field declared_in")
        entity = _text(field.get("entity"), "field entity")
        column = _text(field.get("column"), "field column")
        declaration = declarations[(declared_in, entity, column)]
        for prop in ("dtype", "rows", "ownership", "rewrite"):
            _require(
                field.get(prop) == declaration.get(prop),
                "field declaration disagreement",
            )
        if identity not in fields:
            fields[identity] = field
            add_node(
                {
                    "id": identity,
                    "label": f"{entity}.{column}",
                    "kind": "field",
                    "data": {**field, "visible_in_schema": visible_field},
                }
            )
            add_edge(
                presentation_id(
                    "operation", _text(field.get("provider"), "field provider")
                ),
                identity,
                "provided_field",
                "provenance",
            )
        else:
            _require(fields[identity] == field, "ambiguous field value")
        return identity

    for value in _array(document["fields"], "fields"):
        field = _object(value, "field")
        coordinate = (
            _text(field.get("population"), "field population"),
            _text(field.get("entity"), "field entity"),
            _text(field.get("column"), "field column"),
        )
        _require(coordinate not in visible, "duplicate visible field")
        visible[coordinate] = add_field(field, visible_field=True)

    for value in _array(document["input_bindings"], "input bindings"):
        binding = _object(value, "input binding")
        declared_in = _text(binding.get("declared_in"), "binding declared_in")
        entity = _text(binding.get("entity"), "binding entity")
        column = _text(binding.get("column"), "binding column")
        declaration = declarations[(declared_in, entity, column)]
        field = {
            key: binding[key]
            for key in ("population", "entity", "column", "provider", "declared_in")
        }
        field.update(
            {key: declaration[key] for key in ("dtype", "rows", "ownership", "rewrite")}
        )
        field_id = add_field(field, visible_field=False)
        add_edge(
            field_id,
            presentation_id("operation", _text(binding.get("node"), "binding node")),
            "declared_read",
            "dependency",
            {"read_kind": binding["kind"], "rows": binding["rows"]},
        )

    for node_id, operation in operations.items():
        target = presentation_id("operation", node_id)
        for parent in _array(predecessors[node_id], f"predecessors for {node_id}"):
            add_edge(
                presentation_id("operation", _text(parent, "predecessor")),
                target,
                "compiled_predecessor",
                "dependency",
            )
        base = operation.get("base")
        if base is not None:
            base_id = _text(base, "structural base")
            for (population, _entity, _column), field_id in visible.items():
                if population == base_id:
                    add_edge(
                        field_id,
                        target,
                        "structural_input",
                        "provenance",
                    )
        for source in _array(operation.get("sources"), f"sources for {node_id}"):
            add_edge(
                presentation_id("source", _text(source, "source reference")),
                target,
                "declared_source",
                "provenance",
            )
        for value in _array(
            operation.get("artifact_inputs", []), f"artifact inputs for {node_id}"
        ):
            binding = _object(value, "artifact input")
            add_edge(
                presentation_id(
                    "operation",
                    _text(binding.get("producer"), "artifact producer"),
                ),
                target,
                "artifact_input",
                "dependency",
                binding,
            )

    _require(
        all(
            edge["source"] in nodes and edge["target"] in nodes
            for edge in edges.values()
        ),
        "dangling presentation edge",
    )
    country = _text(document["country"], "country")
    result: dict[str, object] = {
        "schemaVersion": "graph-explorer/v1",
        "id": presentation_id("microcosm", country),
        "title": _text(title, "title")
        if title is not None
        else f"{country.upper()} graph schema",
        "revision": _revision(document),
        "description": (
            "Compiler-derived declaration metadata; no execution, verification, "
            "or release result."
        ),
        "nodes": [nodes[key] for key in sorted(nodes)],
        "edges": [edges[key] for key in sorted(edges)],
        "metadata": {
            "adapter": "microcosm.graph.orrery.v1",
            "scope": "complete_compiler_schema",
            "truncated": False,
            "evidence_scope": (
                "The embedded graph was recompiled; no source bytes, kernel "
                "execution, or authenticity check was performed."
            ),
            "revision_scope": (
                "Content digests of declarations and presentation records; not "
                "runtime cache keys."
            ),
            "edge_scope": (
                "Declared operation reads and compiler dependencies; not "
                "individual-output formulas or observed runtime access."
            ),
            "microcosm": transport_document,
            "missing_runtime_data": [
                "entity identifiers and memberships",
                "field values",
                "source bytes and preparation internals",
                "kernel execution and implicit runtime reads",
                "runtime receipts and release decisions",
            ],
        },
    }
    apply_contracts(result, document, execution)
    _require(len(result["nodes"]) <= _MAX_NODES, "node limit")
    transported_result = _object(_transport_json(result), "Orrery document")
    for node in transported_result["nodes"]:
        node["revision"] = _revision(
            {key: value for key, value in node.items() if key != "revision"}
        )
    _bounded_json(transported_result, _MAX_OUTPUT_BYTES)
    return transported_result


def orrery_json_from_schema(
    schema: object, *, title: str | None = None, execution: object | None = None
) -> str:
    """Return deterministic UTF-8-ready JSON for Orrery."""

    return (
        _bounded_json(
            orrery_document_from_schema(schema, title=title, execution=execution),
            _MAX_OUTPUT_BYTES,
        )
        + "\n"
    )


def _compiled(value: Graph | CompiledGraph) -> CompiledGraph:
    if isinstance(value, Graph):
        return compile_graph(value)
    if isinstance(value, CompiledGraph):
        return value
    raise TypeError(
        f"orrery_document expects Graph or CompiledGraph, got {type(value).__name__}"
    )


def orrery_document(
    graph: Graph | CompiledGraph,
    *,
    title: str | None = None,
    extensions: Mapping[str, object] | None = None,
    execution: object | None = None,
) -> dict[str, object]:
    """Compile and transform a graph into a complete Orrery document."""

    return orrery_document_from_schema(
        graph_schema(_compiled(graph), extensions=extensions),
        title=title,
        execution=execution,
    )


def orrery_json(
    graph: Graph | CompiledGraph,
    *,
    title: str | None = None,
    extensions: Mapping[str, object] | None = None,
    execution: object | None = None,
) -> str:
    """Compile and return deterministic UTF-8-ready Orrery JSON."""

    return orrery_json_from_schema(
        graph_schema(_compiled(graph), extensions=extensions),
        title=title,
        execution=execution,
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _invalid_json_constant(_value: str) -> None:
    raise ValueError("Orrery adapter: finite JSON numbers required.")


def _read_json(path: Path) -> object:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode), "input must be a regular file")
        _require(info.st_size <= _MAX_INPUT_BYTES, "input byte limit")
        raw = stream.read(_MAX_INPUT_BYTES + 1)
    _require(len(raw) <= _MAX_INPUT_BYTES, "input byte limit")
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_invalid_json_constant,
    )


def _rebase_references(schema: dict, source: Path, output: Path) -> None:
    """Keep saved-schema relative links correct beside a relocated export."""
    contract = validate_presentation(schema)
    if contract is None:
        return
    references = [
        ref
        for kind in ("operations", "sources")
        for record in contract.get(kind, {}).values()
        for ref in record.get("references", [])
    ] + [
        ref
        for boundary in contract.get("scope", {}).get("boundaries", [])
        for ref in boundary.get("upstream", [])
    ]
    for ref in references:
        if "url" not in ref:
            continue
        parsed = urlsplit(ref["url"])
        if not parsed.scheme and not parsed.netloc and not parsed.path.startswith("/"):
            relative = Path(os.path.relpath(source.parent / parsed.path, output.parent))
            ref["url"] = urlunsplit(
                ("", "", relative.as_posix(), parsed.query, parsed.fragment)
            )
    schema["extensions"]["microcosm.presentation"] = contract


def main(argv: list[str] | None = None) -> int:
    """Create Orrery JSON from a graph declaration or compiler schema."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--graph", type=Path, help="canonical Graph JSON")
    inputs.add_argument("--schema", type=Path, help="microcosm.graph.schema.v1 JSON")
    parser.add_argument("--output", type=Path, required=True, help="new Orrery JSON")
    parser.add_argument("--title", help="Orrery document title")
    parser.add_argument(
        "--evidence", type=Path, help="recorded execution-input.v1 index"
    )
    parser.add_argument(
        "--store", type=Path, help="content store for evidence validation"
    )
    args = parser.parse_args(argv)
    if (args.evidence is None) != (args.store is None):
        parser.error("--evidence and --store must be supplied together")
    try:
        source = args.graph if args.graph is not None else args.schema
        value = _read_json(source)
        if args.graph is not None:
            graph = graph_from_json(canonical_json(value).decode("utf-8"))
            schema = graph_schema(_compiled(graph))
        else:
            schema = validate_graph_schema(value)
            _rebase_references(schema, source, args.output)
        execution = None
        if args.evidence is not None:
            from .evidence import collect_execution_evidence, load_run_evidence
            from .store import ContentStore

            _require(args.store.is_dir(), "evidence store must exist")
            store = ContentStore(args.store, create=False)
            execution = collect_execution_evidence(
                schema,
                runs=load_run_evidence(
                    args.evidence, store=store, reference_base=args.output.parent
                ),
                store=store,
            )
        rendered = orrery_json_from_schema(
            schema, title=args.title, execution=execution
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    except (OSError, TypeError, ValueError, RecursionError, GraphRuntimeError) as error:
        parser.error(str(error))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
