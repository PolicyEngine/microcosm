from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from referencing.exceptions import NoSuchResource

from microcosm.build.spec_engine.errors import (
    SpecSchemaError,
    SpecValidationError,
)
from microcosm.build.spec_engine.schemas import (
    DRAFT_2020_12,
    SCHEMA_FILENAMES,
    SCHEMA_IDS,
    SchemaRegistry,
    assert_schema_id_allowed,
    load_schema_registry,
)

SchemaMutation = Callable[[dict[str, dict[str, Any]]], None]


def _mutated_catalog(
    tmp_path: Path,
    mutation: SchemaMutation,
) -> Path:
    source = Path(load_schema_registry().source)
    documents = {
        filename: json.loads((source / filename).read_text(encoding="utf-8"))
        for filename in SCHEMA_FILENAMES
    }
    mutation(documents)
    for filename, document in documents.items():
        (tmp_path / filename).write_text(
            json.dumps(document, sort_keys=True),
            encoding="utf-8",
        )
    return tmp_path


def test_schema_registry_loads_only_the_approved_closed_world() -> None:
    catalog = load_schema_registry()

    assert frozenset(catalog.schemas) == SCHEMA_IDS
    assert len(catalog.schemas) == 15
    assert catalog.schema("locks.schema.json#/$defs/bundle_lock")["required"] == [
        "grammar_receipt",
        "files",
        "spec_sha256",
    ]
    with pytest.raises(NoSuchResource):
        catalog.registry.get_or_retrieve("https://example.invalid/schema.json")


def test_schema_kind_allowlist_refuses_mismatched_authority() -> None:
    assert_schema_id_allowed("bundle", "bundle.schema.json")
    assert_schema_id_allowed("schema", "locks.schema.json")
    assert_schema_id_allowed("legacy_json", "legacy_json")

    with pytest.raises(
        SpecValidationError,
        match="resource kind 'bundle' cannot use schema_id 'sources.schema.json'",
    ):
        assert_schema_id_allowed("bundle", "sources.schema.json")
    with pytest.raises(SpecValidationError, match="unknown resource kind 'kernel'"):
        assert_schema_id_allowed("kernel", "kernel:example")


def test_validation_reports_all_errors_in_deterministic_path_order() -> None:
    invalid = {"country": "US", "extra": True}

    with pytest.raises(SpecValidationError) as first:
        load_schema_registry().validate(invalid, "bundle.schema.json")
    with pytest.raises(SpecValidationError) as second:
        load_schema_registry().validate(invalid, "bundle.schema.json")

    expected = """bundle.schema.json: 4 schema validation error(s)
  /: Additional properties are not allowed ('extra' was unexpected)
  /: 'identity_generation' is a required property
  /: 'seed_protocol' is a required property
  /country: 'US' does not match '^[a-z][a-z0-9_]*$'"""
    assert str(first.value) == expected
    assert str(second.value) == expected


def test_emitted_bundle_lock_fragment_uses_the_same_registry() -> None:
    digest = "0" * 64
    instance = {
        "grammar_receipt": {
            "schema_version": 1,
            "canonicalizer_version": 1,
            "migration_chain": [],
        },
        "files": {"bundle.yaml": {"sha256": digest, "byte_size": 10}},
        "spec_sha256": digest,
    }

    load_schema_registry().validate(
        instance,
        "locks.schema.json#/$defs/bundle_lock",
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda documents: documents["bundle.schema.json"].__setitem__(
                "$schema", "https://json-schema.org/draft/2019-09/schema"
            ),
            "$schema must be exactly",
        ),
        (
            lambda documents: documents["bundle.schema.json"]["properties"].__setitem__(
                "broken", {"$ref": "https://example.invalid/a.json"}
            ),
            "unresolved closed-world $ref",
        ),
    ],
)
def test_schema_catalog_refuses_wrong_draft_or_external_reference(
    tmp_path: Path,
    mutation: SchemaMutation,
    message: str,
) -> None:
    root = _mutated_catalog(tmp_path, mutation)

    with pytest.raises(SpecSchemaError, match=re.escape(message)):
        SchemaRegistry.from_package_data(root)


def test_schema_catalog_refuses_missing_or_unexpected_json(tmp_path: Path) -> None:
    root = _mutated_catalog(tmp_path, lambda documents: None)
    (root / "bundle.schema.json").unlink()

    with pytest.raises(SpecSchemaError, match="missing=.*bundle.schema.json"):
        SchemaRegistry.from_package_data(root)

    (root / "unexpected.schema.json").write_text(
        json.dumps({"$schema": DRAFT_2020_12, "$id": "unexpected.schema.json"}),
        encoding="utf-8",
    )
    with pytest.raises(SpecSchemaError, match="unexpected=.*unexpected.schema.json"):
        SchemaRegistry.from_package_data(root)


def test_schema_catalog_checks_unique_ids_and_schema_structure(
    tmp_path: Path,
) -> None:
    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()

    def duplicate_id(documents: dict[str, dict[str, Any]]) -> None:
        documents["sources.schema.json"]["$id"] = "bundle.schema.json"

    _mutated_catalog(duplicate_root, duplicate_id)
    with pytest.raises(SpecSchemaError, match=r"duplicate schema \$id"):
        SchemaRegistry.from_package_data(duplicate_root)

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()

    def invalid_schema(documents: dict[str, dict[str, Any]]) -> None:
        documents["bundle.schema.json"]["type"] = 42

    _mutated_catalog(invalid_root, invalid_schema)
    with pytest.raises(SpecSchemaError, match="invalid draft-2020-12 schema"):
        SchemaRegistry.from_package_data(invalid_root)


def test_defaults_follow_local_refs_without_mutating_the_input(
    tmp_path: Path,
) -> None:
    def add_defaults(documents: dict[str, dict[str, Any]]) -> None:
        documents["defs.schema.json"]["$defs"]["defaulted"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"count": {"type": "integer", "default": 3}},
        }
        documents["bundle.schema.json"]["properties"]["settings"] = {
            "$ref": "defs.schema.json#/$defs/defaulted"
        }

    catalog = SchemaRegistry.from_package_data(_mutated_catalog(tmp_path, add_defaults))
    authored = {
        "country": "us",
        "identity_generation": 1,
        "seed_protocol": "legacy-v1",
        "settings": {},
    }

    resolved = catalog.validate_and_inject_defaults(
        authored,
        "bundle.schema.json",
    )

    assert resolved["settings"] == {"count": 3}
    assert authored["settings"] == {}


def test_one_of_injects_defaults_from_exactly_one_matching_branch(
    tmp_path: Path,
) -> None:
    def add_one_of(documents: dict[str, dict[str, Any]]) -> None:
        documents["bundle.schema.json"]["properties"]["choice"] = {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["kind"],
                    "properties": {
                        "kind": {"const": "a"},
                        "amount": {"type": "integer", "default": 7},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["kind"],
                    "properties": {
                        "kind": {"const": "b"},
                        "label": {"type": "string", "default": "bee"},
                    },
                },
            ]
        }

    catalog = SchemaRegistry.from_package_data(_mutated_catalog(tmp_path, add_one_of))
    authored = {
        "country": "us",
        "identity_generation": 1,
        "seed_protocol": "legacy-v1",
        "choice": {"kind": "a"},
    }

    resolved = catalog.validate_and_inject_defaults(
        authored,
        "bundle.schema.json",
    )

    assert resolved["choice"] == {"kind": "a", "amount": 7}
    assert authored["choice"] == {"kind": "a"}


def test_ambiguous_one_of_never_leaks_a_branch_default(tmp_path: Path) -> None:
    def add_ambiguous_one_of(documents: dict[str, dict[str, Any]]) -> None:
        documents["bundle.schema.json"]["properties"]["ambiguous"] = {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"left": {"type": "integer", "default": 1}},
                },
                {
                    "type": "object",
                    "properties": {"right": {"type": "integer", "default": 2}},
                },
            ]
        }

    catalog = SchemaRegistry.from_package_data(
        _mutated_catalog(tmp_path, add_ambiguous_one_of)
    )
    authored = {
        "country": "us",
        "identity_generation": 1,
        "seed_protocol": "legacy-v1",
        "ambiguous": {},
    }

    with pytest.raises(SpecValidationError, match="valid under each of"):
        catalog.validate_and_inject_defaults(authored, "bundle.schema.json")

    assert authored["ambiguous"] == {}
