"""Vendor hash-pinned Chronicle facts as UK build resources.

Spine stages need publisher values (DfT bus receipts, NTS trips, Ofgem cap
levels, NEED consumption, HMRC clearances, DESNZ prices) without reading the
Chronicle feed at build time. This module copies the exact consumer facts a
committed selection register names into per-concern JSON resources, carrying
each fact's keys, source digest and value verbatim and the identity of the
pinned feed they were taken from. Nothing here derives, uprates, scales or
totals a value: the stage that consumes a resource declares its own arithmetic.

The selection register (``ledger_fact_vendor_selections.json``) uses the same
closed selector vocabulary as the target references, matched by the same
function, so a selection reads like a reference selector and fails the same
way on an unknown field.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from microcosm.build.ledger_targets import (
    fact_matches_selector,
    selector_field_is_supported,
)
from microcosm.build.target_reference_authoring import _json_safe_ledger_id
from microcosm.build.uk_runtime.chronicle_feed import (
    UKChronicleFeed,
)

VENDOR_SELECTIONS_RESOURCE = "ledger_fact_vendor_selections.json"
VENDOR_SELECTIONS_KIND = "uk_ledger_fact_vendor_selections"
VENDORED_RESOURCE_KIND = "uk_vendored_ledger_facts"
VENDOR_TOOL = "tools/vendor_uk_ledger_facts.py"
VENDOR_TOOL_VERSION = 1
_SELECTION_KEYS = frozenset({"label", "selector", "expected_row_count"})
_RESOURCE_KEYS = frozenset(
    {"resource", "purpose", "consumers", "planned_consumers", "selections"}
)
_ROW_ORDER_KEYS = (
    "concept",
    "geography_id",
    "period_type",
    "period_value",
    "groupby_value_id",
    "dimensions",
    "measure_id",
    "aggregate_fact_key",
)


@dataclass(frozen=True)
class VendoredResource:
    """One generated resource and the bytes it serialises to."""

    resource: str
    payload: dict[str, Any]

    @property
    def content(self) -> bytes:
        return (json.dumps(self.payload, indent=2) + "\n").encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @property
    def row_count(self) -> int:
        return len(self.payload["rows"])


def load_vendor_selections() -> dict[str, Any]:
    """Load the committed selection register from the UK country package."""

    register = json.loads(
        files("microcosm.build.uk")
        .joinpath(VENDOR_SELECTIONS_RESOURCE)
        .read_text(encoding="utf-8")
    )
    validate_vendor_selections(register)
    return register


def validate_vendor_selections(register: Mapping[str, Any]) -> None:
    """Refuse a register whose shape or selector vocabulary is not closed."""

    if register.get("schema_version") != 1:
        raise ValueError("Vendor selection register schema_version must be 1.")
    if register.get("country") != "uk":
        raise ValueError("Vendor selection register country must be 'uk'.")
    if register.get("kind") != VENDOR_SELECTIONS_KIND:
        raise ValueError(
            f"Vendor selection register kind must be {VENDOR_SELECTIONS_KIND!r}."
        )
    resources = register.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError("Vendor selection register must list at least one resource.")
    seen: set[str] = set()
    for entry in resources:
        if not isinstance(entry, Mapping) or set(entry) != _RESOURCE_KEYS:
            raise ValueError(
                "Each vendor selection resource must carry exactly "
                f"{sorted(_RESOURCE_KEYS)}; got {sorted(entry) if isinstance(entry, Mapping) else entry!r}."
            )
        name = entry["resource"]
        if not isinstance(name, str) or not name.endswith(".json") or "/" in name:
            raise ValueError(
                f"Vendored resource name {name!r} must be a bare .json filename."
            )
        if name in seen:
            raise ValueError(f"Vendored resource {name!r} is declared twice.")
        seen.add(name)
        if not isinstance(entry["purpose"], str) or not entry["purpose"].strip():
            raise ValueError(f"Vendored resource {name!r} needs a non-blank purpose.")
        for key in ("consumers", "planned_consumers"):
            consumers = entry.get(key, [])
            if not isinstance(consumers, list) or not all(
                isinstance(item, str) and item for item in consumers
            ):
                raise ValueError(
                    f"Vendored resource {name!r} {key} must be a list of names."
                )
        selections = entry["selections"]
        if not isinstance(selections, list) or not selections:
            raise ValueError(
                f"Vendored resource {name!r} needs at least one selection."
            )
        labels: set[str] = set()
        for selection in selections:
            if not isinstance(selection, Mapping) or set(selection) != _SELECTION_KEYS:
                raise ValueError(
                    f"Vendored resource {name!r}: each selection must carry exactly "
                    f"{sorted(_SELECTION_KEYS)}."
                )
            label = selection["label"]
            if not isinstance(label, str) or not label.strip() or label in labels:
                raise ValueError(
                    f"Vendored resource {name!r}: selection labels must be unique, non-blank strings."
                )
            labels.add(label)
            selector = selection["selector"]
            if not isinstance(selector, Mapping) or not selector:
                raise ValueError(
                    f"Vendored resource {name!r} selection {label!r}: selector must be a non-empty mapping."
                )
            for field in selector:
                if not selector_field_is_supported(str(field)):
                    raise ValueError(
                        f"Vendored resource {name!r} selection {label!r}: "
                        f"unsupported selector field {field!r}."
                    )
            count = selection["expected_row_count"]
            if type(count) is not int or count <= 0:
                raise ValueError(
                    f"Vendored resource {name!r} selection {label!r}: "
                    "expected_row_count must be a positive integer."
                )


def feed_identity(pin: UKChronicleFeed) -> dict[str, Any]:
    """The pinned feed identity every vendored resource records."""

    return {
        "source_repo": pin.source_repo,
        "source_commit": pin.source_commit,
        "build": pin.build,
        "artifact_schema_version": pin.artifact_schema_version,
        "fact_row_count": pin.fact_row_count,
        "facts_sha256": pin.facts_sha256,
        "manifest_sha256": pin.manifest_sha256,
    }


def vendor_row(fact: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the identifying keys, provenance and value of one consumer fact."""

    observed = fact.get("observed_measure") or {}
    source = fact.get("source") or {}
    layout = fact.get("layout") or {}
    geography = fact.get("geography") or {}
    period = fact.get("period") or {}
    coverage = fact.get("period_coverage") or {}
    entity = fact.get("entity") or {}
    lineage = fact.get("lineage") or {}
    aggregation = fact.get("aggregation") or {}
    dimensions = fact.get("dimensions") or {}
    value = fact.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            f"Consumer fact {fact.get('aggregate_fact_key')!r} has a non-numeric value."
        )
    # Fact keys are stored JSON-safe (dots and colons to underscores), the
    # membership register's spelling, so a country-package resource never
    # carries a module:attribute-shaped string.
    row = {
        "aggregate_fact_key": _json_safe_ledger_id(
            _required(fact, "aggregate_fact_key")
        ),
        "semantic_fact_key": _json_safe_ledger_id(_required(fact, "semantic_fact_key")),
        "source_record_id": str(lineage.get("source_record_id") or ""),
        "concept": str(observed.get("source_concept") or ""),
        "measure_id": str(observed.get("source_measure_id") or ""),
        "unit": observed.get("unit"),
        "aggregation": aggregation.get("method"),
        "entity": {"name": entity.get("name"), "role": entity.get("role")},
        "geography": {
            "id": geography.get("id"),
            "level": geography.get("level"),
            "vintage": geography.get("vintage"),
        },
        "period": {"type": period.get("type"), "value": period.get("value")},
        "period_coverage": {
            "start_date": coverage.get("start_date"),
            "end_date": coverage.get("end_date"),
            "basis": coverage.get("basis"),
        },
        "layout": {
            "record_set_id": layout.get("record_set_id"),
            "groupby_dimension": layout.get("groupby_dimension"),
            "groupby_value_id": layout.get("groupby_value_id"),
            "table_record_kind": layout.get("table_record_kind"),
        },
        "dimensions": dict(sorted(dimensions.items())),
        "source": {
            "source_name": source.get("source_name"),
            "source_table": source.get("source_table"),
            "source_file": source.get("source_file"),
            "source_sha256": source.get("source_sha256"),
            "vintage": source.get("vintage"),
            "raw_r2_key": source.get("raw_r2_key"),
        },
        "value": value,
    }
    if not row["concept"]:
        raise ValueError(
            f"Consumer fact {row['aggregate_fact_key']!r} carries no source concept."
        )
    return row


def vendor_resource(
    entry: Mapping[str, Any],
    facts: Iterable[Mapping[str, Any]],
    *,
    pin: UKChronicleFeed,
    strict_counts: bool = True,
) -> VendoredResource:
    """Select and copy the facts one register entry names."""

    fact_list = list(facts)
    selections_out: list[dict[str, Any]] = []
    rows: dict[str, dict[str, Any]] = {}
    for selection in entry["selections"]:
        selector = dict(selection["selector"])
        matched = [fact for fact in fact_list if fact_matches_selector(fact, selector)]
        expected = int(selection["expected_row_count"])
        if strict_counts and len(matched) != expected:
            raise ValueError(
                f"Vendored resource {entry['resource']!r} selection "
                f"{selection['label']!r} matched {len(matched)} facts, "
                f"expected {expected}."
            )
        for fact in matched:
            row = vendor_row(fact)
            key = row["aggregate_fact_key"]
            previous = rows.get(key)
            if previous is not None and previous != row:
                raise ValueError(
                    f"Vendored resource {entry['resource']!r}: aggregate fact "
                    f"{key!r} matched two selections with different rows."
                )
            rows[key] = row
        selections_out.append(
            {
                "label": selection["label"],
                "selector": selector,
                "row_count": len(matched),
            }
        )
    ordered = sorted(rows.values(), key=_row_sort_key)
    payload = {
        "schema_version": 1,
        "country": "uk",
        "kind": VENDORED_RESOURCE_KIND,
        "resource": entry["resource"],
        "purpose": entry["purpose"],
        "consumers": list(entry["consumers"]),
        "planned_consumers": list(entry.get("planned_consumers", [])),
        "policy": (
            "Rows are copied verbatim from the pinned Chronicle consumer feed named "
            "in source_fact_feed; every value is a publisher-stated fact. Consumers "
            "declare their own arithmetic on these rows and never edit this file by "
            f"hand: regenerate it with {VENDOR_TOOL}."
        ),
        "generator": {"tool": VENDOR_TOOL, "version": VENDOR_TOOL_VERSION},
        "source_fact_feed": feed_identity(pin),
        "selections": selections_out,
        "row_count": len(ordered),
        "rows": ordered,
    }
    return VendoredResource(resource=entry["resource"], payload=payload)


def vendor_all(
    register: Mapping[str, Any],
    facts: Sequence[Mapping[str, Any]],
    *,
    pin: UKChronicleFeed,
    only: Iterable[str] | None = None,
    strict_counts: bool = True,
) -> list[VendoredResource]:
    wanted = set(only) if only is not None else None
    results = []
    for entry in register["resources"]:
        if wanted is not None and entry["resource"] not in wanted:
            continue
        results.append(
            vendor_resource(entry, facts, pin=pin, strict_counts=strict_counts)
        )
    if wanted is not None:
        missing = wanted - {result.resource for result in results}
        if missing:
            raise ValueError(f"Unknown vendored resource(s): {sorted(missing)}.")
    return results


def load_vendored_resource(name: str) -> dict[str, Any]:
    """Load a committed vendored resource and check its header shape."""

    payload = json.loads(
        files("microcosm.build.uk").joinpath(name).read_text(encoding="utf-8")
    )
    if payload.get("kind") != VENDORED_RESOURCE_KIND or payload.get("resource") != name:
        raise ValueError(f"{name} is not a vendored Chronicle facts resource.")
    if payload.get("row_count") != len(payload.get("rows", ())):
        raise ValueError(f"{name}: row_count disagrees with the rows carried.")
    return payload


def rows_matching(
    payload: Mapping[str, Any], **criteria: object
) -> list[dict[str, Any]]:
    """Filter vendored rows by top-level scalar fields or dotted sub-keys.

    ``rows_matching(payload, concept="x", period_type="fiscal_year")`` filters
    on ``row["concept"]`` and ``row["period"]["type"]``; ``geography_id`` and
    ``period_value`` follow the same spelling as the selector vocabulary.
    """

    aliases = {
        "period_type": ("period", "type"),
        "period_value": ("period", "value"),
        "geography_id": ("geography", "id"),
        "geography_level": ("geography", "level"),
        "groupby_value_id": ("layout", "groupby_value_id"),
        "record_set_id": ("layout", "record_set_id"),
    }
    result = []
    for row in payload["rows"]:
        keep = True
        for key, expected in criteria.items():
            path = aliases.get(key, (key,))
            actual: Any = row
            for part in path:
                actual = actual.get(part) if isinstance(actual, Mapping) else None
            if isinstance(expected, (list, tuple, set, frozenset)):
                if actual not in expected and str(actual) not in {
                    str(e) for e in expected
                }:
                    keep = False
                    break
            elif actual != expected and str(actual) != str(expected):
                keep = False
                break
        if keep:
            result.append(row)
    return result


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row["concept"]),
        str(row["geography"]["id"]),
        str(row["period"]["type"]),
        str(row["period"]["value"]),
        str(row["layout"]["groupby_value_id"]),
        json.dumps(row["dimensions"], sort_keys=True),
        str(row["measure_id"]),
        str(row["aggregate_fact_key"]),
    )


def _required(fact: Mapping[str, Any], key: str) -> str:
    value = fact.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Consumer fact is missing {key!r}.")
    return value
