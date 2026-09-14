"""Pure presentation adapter from compiler metadata to graph-explorer/v1.

The input is a microcosm.graph.schema.v1 export, not a Frame or RunManifest.
This module checks presentation references, not compilation or authenticity.
It never imports a kernel, opens a source, or infers an execution verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections import Counter
from pathlib import Path

from .canonical import canonical_json

__all__ = ["graph_explorer_document", "graph_explorer_json"]

_PROTOCOL = "microcosm.graph.schema.v1"
_SAFE_INTEGER = 2**53 - 1
_MAX_INPUT_BYTES = 32 * 1024 * 1024
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_ITEMS = 2_000_000
_MAX_NODES = 20_000
_MAX_EDGES = 100_000
_FIELD_KEYS = ("population", "entity", "column", "producer", "declared_in")
_READ_KINDS = {"slice", "slice_mask", "output_mask", "rewrite_incumbent"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Graph explorer: {message}.")


def _copy_json(value: object, *, transport: bool = False) -> object:
    """Bound and detach plain JSON; reject custom objects before invoking them."""
    items, charge = 0, 0

    def walk(child: object, depth: int) -> object:
        nonlocal items, charge
        items += 1
        charge += 16
        _require(items <= _MAX_ITEMS and depth <= 64, "metadata complexity")
        kind = type(child)
        if kind is str:
            _require(len(child) <= 16_384, "string length")
            charge += len(child) * 6
            result = child
        elif child is None or kind is bool:
            result = child
        elif kind is int:
            _require(child.bit_length() <= 1024, "integer size")
            charge += child.bit_length()
            result = (
                {"integer_literal": str(child)}
                if transport and abs(child) > _SAFE_INTEGER
                else child
            )
        elif kind is float:
            _require(math.isfinite(child), "finite numbers required")
            # JS cannot distinguish a large integral float from an unsafe int.
            result = (
                {"float_literal": repr(child)}
                if transport and child.is_integer() and abs(child) > _SAFE_INTEGER
                else child
            )
        elif kind is list:
            _require(len(child) <= _MAX_ITEMS - items, "array size")
            result = [walk(item, depth + 1) for item in child]
        elif kind is dict:
            _require(len(child) <= (_MAX_ITEMS - items) // 2, "object size")
            _require(all(type(key) is str for key in child), "string keys required")
            result = {
                walk(key, depth + 1): walk(item, depth + 1)
                for key, item in child.items()
            }
        else:
            raise ValueError("Graph explorer: plain JSON values required.")
        _require(charge <= _MAX_OUTPUT_BYTES, "prospective metadata size")
        return result

    return walk(value, 0)


def _object(value: object) -> dict:
    _require(type(value) is dict, "object required")
    return value


def _array(value: object) -> list:
    _require(type(value) is list, "array required")
    return value


def _text(value: object) -> str:
    _require(type(value) is str and bool(value.strip()), "nonempty text required")
    return value


def _index(values: object, key: str) -> dict[str, dict]:
    result = {}
    for item in _array(values):
        name = _text(_object(item).get(key))
        _require(name not in result, f"duplicate {key}")
        result[name] = item
    return result


def _id(*parts: str) -> str:
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def _field_id(field: dict) -> str:
    return _id("field", *(_text(field.get(key)) for key in _FIELD_KEYS))


def _revision(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _bounded_json(value: object, limit: int) -> str:
    parts, size = [], 0
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    for part in encoder.iterencode(value):
        size += len(part.encode("utf-8"))
        _require(size <= limit, "serialized metadata size")
        parts.append(part)
    return "".join(parts)


def graph_explorer_document(schema: dict, *, title: str | None = None) -> dict:
    """Export the entire supplied schema snapshot; never silently truncate.

    Use ``graph_schema(compiled)`` as the producer where available. Imported
    dictionaries remain declarations: reference checks and content digests do
    not authenticate them or establish that a compiler or kernel actually ran.
    The document revision binds all supplied metadata, including compiled tables;
    node/edge revisions bind their presentation records, not runtime cache keys.

    Pre-rewrite inputs have distinct field IDs in the same population. Structural
    edges mean ancestry, not equality. Reads apply to an operation, without
    claiming that each input mathematically determines each individual output.
    """
    doc = _object(_copy_json(schema))
    _bounded_json(doc, _MAX_INPUT_BYTES)
    _require(doc.get("protocol") == _PROTOCOL, "unsupported schema protocol")
    country = _text(doc.get("country"))
    graph = _object(doc.get("graph"))
    _require(graph.get("country") == country, "country mismatch")
    digest = hashlib.sha256(canonical_json(graph)).hexdigest()
    _require(doc.get("graph_sha256") == digest, "declaration digest mismatch")
    operations = _index(graph.get("nodes"), "id")
    sources = _index(graph.get("sources"), "name")
    _require(len(operations) <= 256 and len(sources) <= 256, "declaration count")
    compiled = _object(doc.get("compiled"))
    order = _array(compiled.get("order"))
    _require(all(type(item) is str for item in order), "order IDs")
    _require(len(order) == len(operations) and set(order) == set(operations), "order")
    predecessors = _object(compiled.get("predecessors"))
    versions = _object(compiled.get("versions"))
    _require(set(predecessors) == set(operations) == set(versions), "compiled IDs")
    positions = {name: index for index, name in enumerate(order)}
    populations = {
        name for name, op in operations.items() if op.get("structural") != "none"
    }
    for name, op in operations.items():
        _require(
            _text(op.get("structural"))
            in {"none", "create", "filter", "expand", "reweight"},
            "structural kind",
        )
        _require(_text(versions[name]) in populations, "population reference")
        _require(
            versions[name] == (name if name in populations else op.get("population")),
            "population binding",
        )
        base = op.get("base")
        _require(base is None or _text(base) in populations, "base reference")
        parents = _array(predecessors[name])
        _require(all(type(parent) is str for parent in parents), "predecessor IDs")
        _require(len(set(parents)) == len(parents), "duplicate predecessor")
        _require(
            all(
                parent in positions and positions[parent] < positions[name]
                for parent in parents
            ),
            "predecessor order",
        )

    nodes, edges = {}, {}
    # Reserve wrapper metadata, then charge each detached transport record before
    # retaining it. Counts alone would allow long repeated IDs to over-expand.
    transport_doc = _copy_json(doc, transport=True)
    used = len(_bounded_json(transport_doc, _MAX_OUTPUT_BYTES).encode("utf-8")) + 65_536

    def presentation_record(record: dict) -> dict:
        nonlocal used
        record = _object(_copy_json(record, transport=True))
        record["revision"] = _revision(record)
        encoded = _bounded_json(record, _MAX_OUTPUT_BYTES - used)
        used += len(encoded.encode("utf-8")) + 1
        return record

    def add_node(node: dict) -> None:
        _require(node["id"] not in nodes, "duplicate presentation node")
        _require(len(nodes) < _MAX_NODES, "node limit")
        nodes[node["id"]] = presentation_record(node)

    def add_edge(source: str, target: str, kind: str, data: dict | None = None) -> None:
        facts = {} if data is None else data
        identity = _id(
            "edge", kind, source, target, _bounded_json(facts, _MAX_INPUT_BYTES)
        )
        if identity in edges:
            return
        _require(len(edges) < _MAX_EDGES, "edge limit")
        edge = {
            "id": identity,
            "source": source,
            "target": target,
            "kind": kind,
            "category": "dependency",
            "data": facts,
        }
        edges[identity] = presentation_record(edge)

    declarations = {}
    for name, op in operations.items():
        add_node(
            {
                "id": _id("operation", name),
                "label": name,
                "kind": "operation",
                "data": {"declaration": op, "population": versions[name]},
            }
        )
        for output in _array(op.get("outputs")):
            output = _object(output)
            key = (name, _text(output.get("entity")), _text(output.get("column")))
            _require(key not in declarations, "duplicate Owned declaration")
            # The real serializer omits rewrite=False. Normalize only this
            # internal lookup; preserve the authored declaration byte-for-byte.
            rewrite = output.get("rewrite", False)
            _require(type(rewrite) is bool, "rewrite flag")
            declarations[key] = {**output, "rewrite": rewrite}
    for name, source in sources.items():
        _text(source.get("codec"))
        add_node(
            {
                "id": _id("source", name),
                "label": name,
                "kind": "source",
                "data": {"declaration": source},
            }
        )

    fields, visible = {}, {}

    def add_field(field: dict, *, visible_field: bool) -> str:
        identity = _field_id(field)
        key = (field["declared_in"], field["entity"], field["column"])
        _require(key in declarations, "field declaration reference")
        owned = declarations[key]
        _require(
            field["population"] in populations and field["producer"] in operations,
            "field reference",
        )
        _require(
            versions[field["producer"]] == field["population"],
            "field provider population",
        )
        for prop in ("dtype", "rows", "ownership", "rewrite"):
            _require(
                field.get(prop) == owned.get(prop), "field declaration disagreement"
            )
        if identity not in fields:
            fields[identity] = field
            add_node(
                {
                    "id": identity,
                    "label": f"{field['entity']}.{field['column']}",
                    "kind": "field",
                    "data": {**field, "visible_in_schema": visible_field},
                }
            )
            add_edge(_id("operation", field["producer"]), identity, "produced")
        else:
            _require(fields[identity] == field, "ambiguous field value")
        return identity

    for field in _array(doc.get("schema")):
        field = _object(field)
        coordinate = tuple(_text(field.get(key)) for key in _FIELD_KEYS[:3])
        _require(coordinate not in visible, "duplicate visible field")
        visible[coordinate] = add_field(field, visible_field=True)

    # Check role coverage against declarations, without reimplementing the
    # compiler's provider resolution. Preserve repeated declared roles too.
    expected_reads = Counter()
    for name, op in operations.items():
        if op["structural"] == "create":
            continue
        population = versions[name] if op["structural"] == "none" else op["base"]
        for slice_ in _array(op.get("inputs")):
            slice_ = _object(slice_)
            entity, rows = _text(slice_.get("entity")), _text(slice_.get("rows"))
            for column in _array(slice_.get("columns")):
                expected_reads[
                    name, population, entity, _text(column), rows, "slice"
                ] += 1
            if rows != "all":
                expected_reads[name, population, entity, rows, "all", "slice_mask"] += 1
        for output in _array(op.get("outputs")):
            entity, column, rows = (
                _text(output.get(key)) for key in ("entity", "column", "rows")
            )
            if output.get("rewrite", False):
                expected_reads[
                    name, population, entity, column, rows, "rewrite_incumbent"
                ] += 1
            if rows != "all":
                expected_reads[
                    name, population, entity, rows, "all", "output_mask"
                ] += 1
    reads = _array(doc.get("input_bindings"))
    _require(len(reads) <= 100_000, "read count")
    actual_reads = Counter(
        tuple(
            _text(_object(read).get(key))
            for key in ("node", "population", "entity", "column", "rows", "kind")
        )
        for read in reads
    )
    _require(actual_reads == expected_reads, "declared read coverage")
    for read in reads:
        read = _object(read)
        _require(
            read.get("node") in operations and read.get("kind") in _READ_KINDS,
            "read reference or kind",
        )
        _text(read.get("rows"))
        key = tuple(_text(read.get(k)) for k in ("declared_in", "entity", "column"))
        _require(key in declarations, "read declaration reference")
        owned = declarations[key]
        field = {key: read[key] for key in _FIELD_KEYS}
        field.update(
            {key: owned[key] for key in ("dtype", "rows", "ownership", "rewrite")}
        )
        identity = add_field(field, visible_field=False)
        # Preserve explicit role labels alongside the supplied compiled edges.
        # This adapter does not infer a new compiled dependency from a read.
        add_edge(
            identity,
            _id("operation", read["node"]),
            "declared_read",
            {"read_kind": read["kind"], "rows": read["rows"]},
        )

    for name, op in operations.items():
        target = _id("operation", name)
        for parent in predecessors[name]:
            add_edge(_id("operation", parent), target, "compiled_predecessor")
        if op.get("base") is not None:
            # Every field in the completed base is carried. Values may change
            # during structural materialization; this is not an identity edge.
            for (population, _entity, _column), field_id in visible.items():
                if population == op["base"]:
                    add_edge(field_id, target, "structural_input")
        for source in _array(op.get("sources")):
            _require(type(source) is str and source in sources, "source reference")
            add_edge(_id("source", source), target, "source")
        for binding in _array(op.get("artifact_inputs", [])):
            binding = _object(binding)
            producer = binding.get("producer")
            _require(
                type(producer) is str and producer in operations, "artifact producer"
            )
            outputs = _index(operations[producer].get("artifact_outputs", []), "name")
            artifact = _text(binding.get("artifact"))
            _require(
                artifact in outputs
                and outputs[artifact].get("type") == binding.get("type"),
                "artifact type or output",
            )
            _require(producer in predecessors[name], "artifact predecessor")
            add_edge(_id("operation", producer), target, "artifact", binding)

    _require(
        all(
            edge["source"] in nodes and edge["target"] in nodes
            for edge in edges.values()
        ),
        "dangling edge",
    )
    result = {
        "schemaVersion": "graph-explorer/v1",
        "id": _id("microcosm", country),
        "title": _text(title)
        if title is not None
        else f"{country.upper()} graph schema",
        "revision": _revision(doc),
        "description": "Complete supplied declaration metadata; no execution or release verdict.",
        "nodes": [nodes[key] for key in sorted(nodes)],
        "edges": [edges[key] for key in sorted(edges)],
        "metadata": {
            "adapter": "microcosm.graph.explorer.v1",
            "scope": "complete_supplied_schema",
            "truncated": False,
            "evidence_scope": "Imported declaration metadata; not recompiled or authenticated by this adapter.",
            "revision_scope": "Content digests of supplied metadata/presentation records; not runtime cache keys.",
            "edge_scope": "Operation-level reads and structural ancestry; not individual-output formulas or value equality.",
            "microcosm": transport_doc,
            "missing_schema": [
                "entity IDs and memberships",
                "units",
                "period semantics",
                "source preparation internals",
                "implicit runtime reads",
            ],
        },
    }
    result = _copy_json(result, transport=True)
    _bounded_json(result, _MAX_OUTPUT_BYTES)
    return result


def graph_explorer_json(schema: dict, *, title: str | None = None) -> str:
    """Deterministic UTF-8-ready JSON, with exact large-number transport tags.

    Write these bytes directly for host-side snapshot hashing. HTML bundling and
    Receipt verification belong to the shared explorer, not this adapter.
    """
    return (
        _bounded_json(graph_explorer_document(schema, title=title), _MAX_OUTPUT_BYTES)
        + "\n"
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise ValueError("Graph explorer: finite JSON numbers required.")


def main(argv: list[str] | None = None) -> int:
    """Convert saved schema metadata for Orrery without loading a saved run."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--input", type=Path, required=True, help="saved schema JSON")
    parser.add_argument(
        "--output", type=Path, required=True, help="new graph-explorer/v1 JSON file"
    )
    parser.add_argument("--title", help="viewer document title")
    args = parser.parse_args(argv)
    try:
        # Refuse streams/devices before reading; a FIFO must not block the CLI.
        descriptor = os.open(args.input, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            _require(stat.S_ISREG(info.st_mode), "input must be a regular file")
            _require(info.st_size <= _MAX_INPUT_BYTES, "input byte limit")
            raw = stream.read(_MAX_INPUT_BYTES + 1)
        _require(len(raw) <= _MAX_INPUT_BYTES, "input byte limit")
        schema = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_invalid_json_constant,
        )
        rendered = graph_explorer_json(schema, title=args.title)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents overwriting the source through an alias.
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    except (OSError, ValueError, RecursionError) as error:
        parser.error(str(error))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
