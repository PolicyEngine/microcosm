"""Canonical logical-member closure for final HDF5 artifacts.

The inventory is intentionally country-neutral and contains no HDF5 reader.
Country code declares the tables, columns, weights, and scalar authority pins;
this module gives that declaration one closed, canonical wire representation.
Raw artifact comparison and physical HDF5 inspection remain separate concerns.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence

from .canonical import sha256_json

FINAL_H5_INVENTORY_SCHEMA_VERSION = 1
FINAL_H5_INVENTORY_SEMANTICS = "canonical_member_set"
FINAL_H5_MEMBER_KINDS = (
    "entity_table",
    "entity_column",
    "weight_vector",
)

_INVENTORY_KEYS = frozenset(
    {
        "schema_version",
        "semantics",
        "authority",
        "member_count",
        "members_sha256",
        "inventory_sha256",
        "tables",
        "columns",
        "weights",
    }
)
_WEIGHT_KEYS = frozenset({"entity", "column"})


class FinalH5InventoryError(ValueError):
    """A final-H5 logical member inventory is incomplete or noncanonical."""


def _mapping(value: object, *, location: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise FinalH5InventoryError(f"{location}: mapping required")
    return value


def _sequence(value: object, *, location: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(
        value, str | bytes | bytearray
    ):
        raise FinalH5InventoryError(f"{location}: array required")
    return value


def _exact_string_keys(
    value: Mapping[object, object],
    expected: frozenset[str],
    *,
    location: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise FinalH5InventoryError(f"{location}: string keys required")
    actual = set(value)
    if actual != expected:
        raise FinalH5InventoryError(
            f"{location}: keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _canonical_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise FinalH5InventoryError(f"{location}: non-empty string required")
    if unicodedata.normalize("NFC", value) != value:
        raise FinalH5InventoryError(f"{location}: NFC string required")
    return value


def _sha256(value: object, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FinalH5InventoryError(f"{location}: lowercase sha256 required")
    return value


def _canonical_scalar(value: object, *, location: str) -> object:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            raise FinalH5InventoryError(f"{location}: NFC string required")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FinalH5InventoryError(f"{location}: finite JSON number required")
        if value == 0 or value.is_integer():
            raise FinalH5InventoryError(
                f"{location}: noncanonical number; use canonical integer form"
            )
        return value
    raise FinalH5InventoryError(f"{location}: JSON scalar required")


def _authority(
    value: object,
    *,
    location: str,
    require_canonical_order: bool,
) -> dict[str, object]:
    authority = _mapping(value, location=location)
    keys: list[str] = []
    for key in authority:
        keys.append(_canonical_string(key, location=f"{location}/<key>"))
    if not keys:
        raise FinalH5InventoryError(f"{location}: must not be empty")
    if require_canonical_order and keys != sorted(keys):
        raise FinalH5InventoryError(f"{location}: keys must be sorted")
    return {
        key: _canonical_scalar(authority[key], location=f"{location}/{key}")
        for key in sorted(keys)
    }


def _string_set(
    value: object,
    *,
    location: str,
    require_canonical_order: bool,
) -> list[str]:
    result = [
        _canonical_string(item, location=f"{location}/{index}")
        for index, item in enumerate(_sequence(value, location=location))
    ]
    if len(result) != len(set(result)):
        raise FinalH5InventoryError(f"{location}: duplicate values forbidden")
    if require_canonical_order and result != sorted(result):
        raise FinalH5InventoryError(f"{location}: values must be sorted")
    return sorted(result)


def _columns(
    value: object,
    *,
    location: str,
    require_canonical_order: bool,
) -> dict[str, list[str]]:
    columns = _mapping(value, location=location)
    entity_keys: list[str] = []
    for entity in columns:
        entity_keys.append(
            _canonical_string(entity, location=f"{location}/<entity>")
        )
    if require_canonical_order and entity_keys != sorted(entity_keys):
        raise FinalH5InventoryError(f"{location}: entity keys must be sorted")
    result: dict[str, list[str]] = {}
    for entity in sorted(entity_keys):
        entity_columns = _string_set(
            columns[entity],
            location=f"{location}/{entity}",
            require_canonical_order=require_canonical_order,
        )
        if not entity_columns:
            raise FinalH5InventoryError(
                f"{location}/{entity}: empty column sets are noncanonical"
            )
        result[entity] = entity_columns
    if not result:
        raise FinalH5InventoryError(f"{location}: must not be empty")
    return result


def _weights(
    value: object,
    *,
    location: str,
    require_canonical_order: bool,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index, item in enumerate(_sequence(value, location=location)):
        item_location = f"{location}/{index}"
        weight = _mapping(item, location=item_location)
        _exact_string_keys(weight, _WEIGHT_KEYS, location=item_location)
        result.append(
            {
                "entity": _canonical_string(
                    weight["entity"], location=f"{item_location}/entity"
                ),
                "column": _canonical_string(
                    weight["column"], location=f"{item_location}/column"
                ),
            }
        )
    keys = [(row["entity"], row["column"]) for row in result]
    if len(keys) != len(set(keys)):
        raise FinalH5InventoryError(f"{location}: duplicate weights forbidden")
    if require_canonical_order and keys != sorted(keys):
        raise FinalH5InventoryError(
            f"{location}: weights must be sorted by entity and column"
        )
    return sorted(result, key=lambda row: (row["entity"], row["column"]))


def _member_descriptors(
    *,
    tables: list[str],
    columns: Mapping[str, list[str]],
    weights: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Expand the compact wire in its sole canonical descriptor order."""

    return [
        *(
            {"kind": "entity_table", "entity": entity}
            for entity in tables
        ),
        *(
            {
                "kind": "entity_column",
                "entity": entity,
                "column": column,
            }
            for entity, entity_columns in columns.items()
            for column in entity_columns
        ),
        *(
            {
                "kind": "weight_vector",
                "entity": weight["entity"],
                "column": weight["column"],
            }
            for weight in weights
        ),
    ]


def _validate_relations(
    *,
    tables: list[str],
    columns: Mapping[str, list[str]],
    weights: list[dict[str, str]],
) -> None:
    table_set = set(tables)
    dangling_columns = sorted(set(columns) - table_set)
    if dangling_columns:
        raise FinalH5InventoryError(
            "columns: entities without table members " f"{dangling_columns!r}"
        )
    dangling_weights = sorted(
        {row["entity"] for row in weights if row["entity"] not in table_set}
    )
    if dangling_weights:
        raise FinalH5InventoryError(
            "weights: entities without table members " f"{dangling_weights!r}"
        )
    owned_tables = set(columns) | {row["entity"] for row in weights}
    empty_tables = sorted(table_set - owned_tables)
    if empty_tables:
        raise FinalH5InventoryError(
            "tables: every table requires at least one column or weight member "
            f"{empty_tables!r}"
        )
    column_members = {
        (entity, column)
        for entity, entity_columns in columns.items()
        for column in entity_columns
    }
    weight_members = {(row["entity"], row["column"]) for row in weights}
    collisions = sorted(column_members & weight_members)
    if collisions:
        raise FinalH5InventoryError(
            "columns/weights: physical member declared twice " f"{collisions!r}"
        )
    misclassified_columns = sorted(
        (entity, column)
        for entity, column in column_members
        if column == f"{entity}_weight"
    )
    invalid_weights = sorted(
        (entity, column)
        for entity, column in weight_members
        if column != f"{entity}_weight"
    )
    if misclassified_columns or invalid_weights:
        raise FinalH5InventoryError(
            "columns/weights: member kind differs from the H5 selector naming "
            f"protocol; columns={misclassified_columns!r}, "
            f"weights={invalid_weights!r}"
        )


def _inventory_payload(
    *,
    authority: Mapping[str, object],
    member_count: int,
    members_sha256: str,
    tables: list[str],
    columns: Mapping[str, list[str]],
    weights: list[dict[str, str]],
) -> dict[str, object]:
    """Return the exact projection covered by ``inventory_sha256``."""

    return {
        "schema_version": FINAL_H5_INVENTORY_SCHEMA_VERSION,
        "semantics": FINAL_H5_INVENTORY_SEMANTICS,
        "authority": dict(authority),
        "member_count": member_count,
        "members_sha256": members_sha256,
        "tables": tables,
        "columns": dict(columns),
        "weights": weights,
    }


def build_final_h5_member_inventory(
    *,
    authority: Mapping[str, object],
    tables: Iterable[str],
    columns: Mapping[str, Iterable[str]],
    weights: Iterable[Mapping[str, str]],
) -> dict[str, object]:
    """Build one canonical, digest-bound logical final-H5 member inventory."""

    canonical_authority = _authority(
        authority,
        location="authority",
        require_canonical_order=False,
    )
    canonical_tables = _string_set(
        tuple(tables),
        location="tables",
        require_canonical_order=False,
    )
    if not canonical_tables:
        raise FinalH5InventoryError("tables: must not be empty")
    canonical_columns = _columns(
        {entity: tuple(entity_columns) for entity, entity_columns in columns.items()},
        location="columns",
        require_canonical_order=False,
    )
    canonical_weights = _weights(
        tuple(weights),
        location="weights",
        require_canonical_order=False,
    )
    _validate_relations(
        tables=canonical_tables,
        columns=canonical_columns,
        weights=canonical_weights,
    )
    members = _member_descriptors(
        tables=canonical_tables,
        columns=canonical_columns,
        weights=canonical_weights,
    )
    member_count = len(members)
    members_sha256 = sha256_json(members)
    payload = _inventory_payload(
        authority=canonical_authority,
        member_count=member_count,
        members_sha256=members_sha256,
        tables=canonical_tables,
        columns=canonical_columns,
        weights=canonical_weights,
    )
    inventory = {**payload, "inventory_sha256": sha256_json(payload)}
    return validate_final_h5_member_inventory(inventory)


def validate_final_h5_member_inventory(value: object) -> dict[str, object]:
    """Validate and detach a canonical final-H5 member inventory wire value."""

    inventory = _mapping(value, location="final_h5_member_inventory")
    _exact_string_keys(
        inventory,
        _INVENTORY_KEYS,
        location="final_h5_member_inventory",
    )
    schema_version = inventory["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != FINAL_H5_INVENTORY_SCHEMA_VERSION
    ):
        raise FinalH5InventoryError(
            "final_h5_member_inventory/schema_version: unsupported version"
        )
    if inventory["semantics"] != FINAL_H5_INVENTORY_SEMANTICS:
        raise FinalH5InventoryError(
            "final_h5_member_inventory/semantics: canonical_member_set required"
        )
    authority = _authority(
        inventory["authority"],
        location="final_h5_member_inventory/authority",
        require_canonical_order=True,
    )
    tables = _string_set(
        inventory["tables"],
        location="final_h5_member_inventory/tables",
        require_canonical_order=True,
    )
    if not tables:
        raise FinalH5InventoryError(
            "final_h5_member_inventory/tables: must not be empty"
        )
    columns = _columns(
        inventory["columns"],
        location="final_h5_member_inventory/columns",
        require_canonical_order=True,
    )
    weights = _weights(
        inventory["weights"],
        location="final_h5_member_inventory/weights",
        require_canonical_order=True,
    )
    _validate_relations(tables=tables, columns=columns, weights=weights)

    member_count_value = inventory["member_count"]
    if isinstance(member_count_value, bool) or not isinstance(member_count_value, int):
        raise FinalH5InventoryError(
            "final_h5_member_inventory/member_count: integer required"
        )
    members = _member_descriptors(
        tables=tables,
        columns=columns,
        weights=weights,
    )
    expected_member_count = len(members)
    if member_count_value != expected_member_count:
        raise FinalH5InventoryError(
            "final_h5_member_inventory/member_count: "
            f"expected {expected_member_count}, got {member_count_value}"
        )

    declared_members_sha256 = _sha256(
        inventory["members_sha256"],
        location="final_h5_member_inventory/members_sha256",
    )
    expected_members_sha256 = sha256_json(members)
    if declared_members_sha256 != expected_members_sha256:
        raise FinalH5InventoryError(
            "final_h5_member_inventory/members_sha256: digest mismatch"
        )

    declared_inventory_sha256 = _sha256(
        inventory["inventory_sha256"],
        location="final_h5_member_inventory/inventory_sha256",
    )
    payload = _inventory_payload(
        authority=authority,
        member_count=expected_member_count,
        members_sha256=expected_members_sha256,
        tables=tables,
        columns=columns,
        weights=weights,
    )
    expected_inventory_sha256 = sha256_json(payload)
    if declared_inventory_sha256 != expected_inventory_sha256:
        raise FinalH5InventoryError(
            "final_h5_member_inventory/inventory_sha256: digest mismatch"
        )
    return {**payload, "inventory_sha256": expected_inventory_sha256}


def canonical_final_h5_member_descriptors(
    value: object,
) -> tuple[dict[str, str], ...]:
    """Return validated logical members in their sole canonical order."""

    inventory = validate_final_h5_member_inventory(value)
    return tuple(
        _member_descriptors(
            tables=inventory["tables"],  # type: ignore[arg-type]
            columns=inventory["columns"],  # type: ignore[arg-type]
            weights=inventory["weights"],  # type: ignore[arg-type]
        )
    )


__all__ = [
    "FINAL_H5_INVENTORY_SCHEMA_VERSION",
    "FINAL_H5_INVENTORY_SEMANTICS",
    "FINAL_H5_MEMBER_KINDS",
    "FinalH5InventoryError",
    "build_final_h5_member_inventory",
    "canonical_final_h5_member_descriptors",
    "validate_final_h5_member_inventory",
]
