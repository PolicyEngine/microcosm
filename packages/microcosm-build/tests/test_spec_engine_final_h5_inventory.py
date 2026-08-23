"""Fail-closed tests for the country-neutral final-H5 member inventory."""

from __future__ import annotations

import copy

import pytest

from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.final_h5_inventory import (
    FINAL_H5_INVENTORY_SCHEMA_VERSION,
    FINAL_H5_INVENTORY_SEMANTICS,
    FINAL_H5_MEMBER_KINDS,
    FinalH5InventoryError,
    build_final_h5_member_inventory,
    canonical_final_h5_member_descriptors,
    validate_final_h5_member_inventory,
)
from microcosm.build.spec_engine.model import freeze_json


def _inventory() -> dict[str, object]:
    return build_final_h5_member_inventory(
        authority={
            "typed_catalog_sha256": "b" * 64,
            "source_byte_size": 123,
            "source_sha256": "a" * 64,
        },
        tables=("person", "household"),
        columns={
            "person": ("person_id", "age"),
            "household": ("household_id", "income"),
        },
        weights=(
            {"entity": "person", "column": "person_weight"},
            {"entity": "household", "column": "household_weight"},
        ),
    )


def _resign_inventory(inventory: dict[str, object]) -> None:
    payload = {
        key: value for key, value in inventory.items() if key != "inventory_sha256"
    }
    inventory["inventory_sha256"] = sha256_json(payload)


def test_builder_emits_the_exact_compact_canonical_wire() -> None:
    inventory = _inventory()

    assert list(inventory) == [
        "schema_version",
        "semantics",
        "authority",
        "member_count",
        "members_sha256",
        "tables",
        "columns",
        "weights",
        "inventory_sha256",
    ]
    assert inventory["schema_version"] == FINAL_H5_INVENTORY_SCHEMA_VERSION
    assert inventory["semantics"] == FINAL_H5_INVENTORY_SEMANTICS
    assert list(inventory["authority"]) == [  # type: ignore[arg-type]
        "source_byte_size",
        "source_sha256",
        "typed_catalog_sha256",
    ]
    assert inventory["tables"] == ["household", "person"]
    assert list(inventory["columns"]) == ["household", "person"]  # type: ignore[arg-type]
    assert inventory["columns"] == {
        "household": ["household_id", "income"],
        "person": ["age", "person_id"],
    }
    assert inventory["weights"] == [
        {"entity": "household", "column": "household_weight"},
        {"entity": "person", "column": "person_weight"},
    ]


def test_logical_member_descriptors_have_one_canonical_kind_order() -> None:
    inventory = _inventory()
    descriptors = canonical_final_h5_member_descriptors(inventory)

    assert FINAL_H5_MEMBER_KINDS == (
        "entity_table",
        "entity_column",
        "weight_vector",
    )
    assert descriptors == (
        {"kind": "entity_table", "entity": "household"},
        {"kind": "entity_table", "entity": "person"},
        {
            "kind": "entity_column",
            "entity": "household",
            "column": "household_id",
        },
        {
            "kind": "entity_column",
            "entity": "household",
            "column": "income",
        },
        {"kind": "entity_column", "entity": "person", "column": "age"},
        {"kind": "entity_column", "entity": "person", "column": "person_id"},
        {
            "kind": "weight_vector",
            "entity": "household",
            "column": "household_weight",
        },
        {
            "kind": "weight_vector",
            "entity": "person",
            "column": "person_weight",
        },
    )
    assert inventory["member_count"] == len(descriptors) == 8
    assert inventory["members_sha256"] == sha256_json(list(descriptors))
    payload = {
        key: value for key, value in inventory.items() if key != "inventory_sha256"
    }
    assert inventory["inventory_sha256"] == sha256_json(payload)


def test_validator_accepts_and_detaches_frozen_compiler_wire() -> None:
    inventory = _inventory()
    frozen = freeze_json(inventory)

    validated = validate_final_h5_member_inventory(frozen)

    assert validated == inventory
    assert validated is not inventory
    assert validated["columns"] is not inventory["columns"]


@pytest.mark.parametrize("field", ["authority", "tables", "columns", "weights"])
def test_validator_rejects_noncanonical_order_with_valid_inventory_digest(
    field: str,
) -> None:
    inventory = copy.deepcopy(_inventory())
    if field == "authority":
        authority = inventory[field]
        assert isinstance(authority, dict)
        inventory[field] = dict(reversed(tuple(authority.items())))
    elif field == "tables":
        tables = inventory[field]
        assert isinstance(tables, list)
        tables.reverse()
    elif field == "columns":
        columns = inventory[field]
        assert isinstance(columns, dict)
        inventory[field] = dict(reversed(tuple(columns.items())))
    else:
        weights = inventory[field]
        assert isinstance(weights, list)
        weights.reverse()
    _resign_inventory(inventory)

    with pytest.raises(FinalH5InventoryError, match="must be sorted"):
        validate_final_h5_member_inventory(inventory)


def test_validator_rejects_noncanonical_column_order() -> None:
    inventory = copy.deepcopy(_inventory())
    columns = inventory["columns"]
    assert isinstance(columns, dict)
    household_columns = columns["household"]
    assert isinstance(household_columns, list)
    household_columns.reverse()
    _resign_inventory(inventory)

    with pytest.raises(FinalH5InventoryError, match="values must be sorted"):
        validate_final_h5_member_inventory(inventory)


@pytest.mark.parametrize("change", ["missing", "extra", "non_string"])
def test_inventory_root_is_an_exact_mapping(change: str) -> None:
    inventory = _inventory()
    if change == "missing":
        inventory.pop("semantics")
    elif change == "extra":
        inventory["normalization"] = "forbidden"
    else:
        inventory[1] = "not a string key"  # type: ignore[index]

    with pytest.raises(FinalH5InventoryError, match="keys"):
        validate_final_h5_member_inventory(inventory)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "unsupported version"),
        ("schema_version", True, "unsupported version"),
        ("semantics", "ordered_members", "canonical_member_set required"),
        ("member_count", 7, "expected 8"),
        ("member_count", True, "integer required"),
        ("members_sha256", "0" * 64, "digest mismatch"),
        ("members_sha256", "A" * 64, "lowercase sha256 required"),
        ("inventory_sha256", "0" * 64, "digest mismatch"),
        ("inventory_sha256", "short", "lowercase sha256 required"),
    ],
)
def test_validator_recomputes_counts_and_both_digests(
    field: str,
    value: object,
    message: str,
) -> None:
    inventory = _inventory()
    inventory[field] = value

    with pytest.raises(FinalH5InventoryError, match=message):
        validate_final_h5_member_inventory(inventory)


@pytest.mark.parametrize(
    ("authority", "message"),
    [
        ({}, "must not be empty"),
        ({"": "pin"}, "non-empty string required"),
        ({"pin": {"nested": "forbidden"}}, "JSON scalar required"),
        ({"pin": ["forbidden"]}, "JSON scalar required"),
        ({"pin": float("nan")}, "finite JSON number required"),
        ({"pin": 1.0}, "noncanonical number"),
        ({"pin": "e\N{COMBINING ACUTE ACCENT}"}, "NFC string required"),
    ],
)
def test_builder_rejects_noncanonical_or_structured_authority(
    authority: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(FinalH5InventoryError, match=message):
        build_final_h5_member_inventory(
            authority=authority,
            tables=["person"],
            columns={"person": ["person_id"]},
            weights=[],
        )


@pytest.mark.parametrize("member_kind", ["table", "column", "weight"])
def test_builder_rejects_duplicate_logical_members(member_kind: str) -> None:
    tables = ["person"]
    columns = {"person": ["person_id"]}
    weights = [{"entity": "person", "column": "person_weight"}]
    if member_kind == "table":
        tables.append("person")
    elif member_kind == "column":
        columns["person"].append("person_id")
    else:
        weights.append({"entity": "person", "column": "person_weight"})

    with pytest.raises(FinalH5InventoryError, match="duplicate"):
        build_final_h5_member_inventory(
            authority={"pin": "a" * 64},
            tables=tables,
            columns=columns,
            weights=weights,
        )


@pytest.mark.parametrize("member_kind", ["column", "weight"])
def test_builder_rejects_dangling_entities(member_kind: str) -> None:
    columns = {"person": ["person_id"]}
    weights = [{"entity": "person", "column": "person_weight"}]
    if member_kind == "column":
        columns["household"] = ["household_id"]
    else:
        weights.append({"entity": "household", "column": "household_weight"})

    with pytest.raises(FinalH5InventoryError, match="without table members"):
        build_final_h5_member_inventory(
            authority={"pin": "a" * 64},
            tables=["person"],
            columns=columns,
            weights=weights,
        )


def test_builder_rejects_column_weight_physical_member_collision() -> None:
    with pytest.raises(FinalH5InventoryError, match="physical member declared twice"):
        build_final_h5_member_inventory(
            authority={"pin": "a" * 64},
            tables=["household"],
            columns={"household": ["household_weight"]},
            weights=[{"entity": "household", "column": "household_weight"}],
        )


@pytest.mark.parametrize(
    ("columns", "weights"),
    [
        ({"household": ["household_weight"]}, []),
        (
            {"household": ["household_id"]},
            [{"entity": "household", "column": "weight"}],
        ),
    ],
)
def test_builder_enforces_h5_selector_weight_naming_protocol(
    columns: dict[str, list[str]],
    weights: list[dict[str, str]],
) -> None:
    with pytest.raises(FinalH5InventoryError, match="selector naming protocol"):
        build_final_h5_member_inventory(
            authority={"pin": "a" * 64},
            tables=["household"],
            columns=columns,
            weights=weights,
        )


def test_builder_rejects_table_without_any_physical_member() -> None:
    with pytest.raises(FinalH5InventoryError, match="every table requires"):
        build_final_h5_member_inventory(
            authority={"pin": "a" * 64},
            tables=["household", "person"],
            columns={"person": ["person_id"]},
            weights=[],
        )


def test_weight_rows_are_exact_and_column_sets_cannot_be_empty() -> None:
    with pytest.raises(FinalH5InventoryError, match="keys differ"):
        build_final_h5_member_inventory(
            authority={"pin": "a" * 64},
            tables=["person"],
            columns={"person": ["person_id"]},
            weights=[
                {
                    "entity": "person",
                    "column": "person_weight",
                    "normalization": "forbidden",
                }
            ],
        )
    with pytest.raises(FinalH5InventoryError, match="empty column sets"):
        build_final_h5_member_inventory(
            authority={"pin": "a" * 64},
            tables=["person"],
            columns={"person": []},
            weights=[],
        )
    with pytest.raises(FinalH5InventoryError, match="tables: must not be empty"):
        build_final_h5_member_inventory(
            authority={"pin": "a" * 64},
            tables=[],
            columns={},
            weights=[],
        )
    with pytest.raises(FinalH5InventoryError, match="columns: must not be empty"):
        build_final_h5_member_inventory(
            authority={"pin": "a" * 64},
            tables=["household"],
            columns={},
            weights=[{"entity": "household", "column": "household_weight"}],
        )
