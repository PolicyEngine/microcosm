"""Explicit parked-reference exemptions, validated before any row is filtered.

The resource grammar is ``{schema_version: 1, resources: {resource: {exempt:
[{name, activation_status, reason}]}}}``. Resource names accept the usual JSON
filename or stem. A compiler reads a document without selecting all its rows;
comparators select their declared rows, and receipt targets, quantile-band
references, mass rows and annualisation keys explicitly select named rows.
Composition checks that selection closure before passing a per-resource
``{exempt: [...]}`` document to a compiler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text.")
    return value


def exemption_rows(selection: Mapping) -> tuple[dict, ...]:
    """Validate the closed per-resource declaration independently of its rows."""
    if not isinstance(selection, Mapping) or set(selection) != {"exempt"}:
        raise ValueError("A reference selection declares exactly an exempt list.")
    rows = selection["exempt"]
    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise ValueError("Reference selection exempt must be a list.")
    result = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "activation_status",
            "reason",
        }:
            raise ValueError(
                "A reference exemption declares name, activation_status and reason."
            )
        entry = {
            key: _text(row[key], f"Reference exemption {key}")
            for key in ("name", "activation_status", "reason")
        }
        if entry["name"] in seen:
            raise ValueError(f"Repeated reference exemption {entry['name']!r}.")
        seen.add(entry["name"])
        result.append(entry)
    return tuple(result)


def selection_resources(document: Mapping) -> dict[str, dict]:
    """Return declarations keyed by canonical resource stem."""
    if (
        not isinstance(document, Mapping)
        or set(document) != {"schema_version", "resources"}
        or document["schema_version"] != 1
        or not isinstance(document["resources"], Mapping)
    ):
        raise ValueError(
            "Reference selection requires schema_version 1 and a resources object."
        )
    result = {}
    for name, selection in document["resources"].items():
        resource = _text(name, "Reference selection resource").removesuffix(".json")
        if not resource or resource in result:
            raise ValueError(f"Repeated reference selection resource {resource!r}.")
        result[resource] = {"exempt": list(exemption_rows(selection))}
    return result


def validate_exemptions(
    document: Mapping,
    selection: Mapping,
    *,
    resource: str = "references",
    selected: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict, ...]:
    """Validate against the full effective document, before removing any row.

    ``selected`` maps reference names to the instantiated nodes that select
    them. Checking it first ensures a stale or otherwise invalid declaration
    can never hide the more fundamental selected-and-exempt contradiction.
    """
    exemptions = exemption_rows(selection)
    for row in exemptions:
        nodes = (selected or {}).get(row["name"], ())
        if nodes:
            raise ValueError(
                f"Node {nodes[0]!r} selects exempt reference {resource}.{row['name']}."
            )
    raw = document.get("target_references")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise ValueError(f"{resource}.target_references must be a list.")
    references = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError(f"{resource} reference must be an object.")
        name = row.get("name")
        if name in references:
            raise ValueError(f"Repeated reference name {name!r} in {resource}.")
        references[name] = row
    for row in exemptions:
        name = row["name"]
        if name not in references:
            raise ValueError(f"Unknown exempt reference {resource}.{name}.")
        metadata = references[name].get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{resource}.{name} metadata must be an object.")
        status = metadata.get("activation_status", "active")
        if status in {"", "active"}:
            raise ValueError(f"Exempt reference {resource}.{name} is active.")
        if status != row["activation_status"]:
            raise ValueError(
                f"Stale exemption for {resource}.{name}: declared status "
                f"{row['activation_status']!r}, effective status {status!r}."
            )
    return exemptions
