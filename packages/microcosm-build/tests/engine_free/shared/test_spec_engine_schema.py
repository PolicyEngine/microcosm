"""Tests split from packages/microcosm-build/tests/test_spec_engine_schema.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.spec_engine_schema import *


def test_schema_registry_loads_only_the_approved_closed_world() -> None:
    catalog = load_schema_registry()

    assert frozenset(catalog.schemas) == SCHEMA_IDS
    assert len(catalog.schemas) == 15
    assert catalog.schema("locks.schema.json#/$defs/bundle_lock")["required"] == [
        "grammar_receipt",
        "files",
        "generated_authorities",
        "seed_protocol",
        "seed_site_bindings",
        "vintage_authorities",
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
        "generated_authorities": {},
        "seed_protocol": LEGACY_V1_PROTOCOL.to_wire(),
        "seed_site_bindings": [],
        "vintage_authorities": {"records": {}},
        "spec_sha256": digest,
    }

    load_schema_registry().validate(
        instance,
        "locks.schema.json#/$defs/bundle_lock",
    )


@pytest.mark.parametrize("invalid_default", [-1, 2**64])
def test_seed_lock_schema_rejects_out_of_uint64_defaults(invalid_default: int) -> None:
    protocol = copy.deepcopy(LEGACY_V1_PROTOCOL.to_wire())
    protocol["sites"][0]["default"] = invalid_default
    with pytest.raises(SpecValidationError, match="seed_protocol"):
        load_schema_registry().validate(
            {
                "grammar_receipt": {
                    "schema_version": 1,
                    "canonicalizer_version": 1,
                    "migration_chain": [],
                },
                "files": {},
                "generated_authorities": {},
                "seed_protocol": protocol,
                "seed_site_bindings": [],
                "vintage_authorities": {"records": {}},
                "spec_sha256": "0" * 64,
            },
            "locks.schema.json#/$defs/bundle_lock",
        )


def test_seed_lock_schema_rejects_duplicate_rows_and_missing_rng_version() -> None:
    protocol = copy.deepcopy(LEGACY_V1_PROTOCOL.to_wire())
    protocol["sites"].append(copy.deepcopy(protocol["sites"][0]))
    missing_version = copy.deepcopy(LEGACY_V1_PROTOCOL.to_wire())
    del missing_version["sites"][0]["rng_version"]
    base = {
        "grammar_receipt": {
            "schema_version": 1,
            "canonicalizer_version": 1,
            "migration_chain": [],
        },
        "files": {},
        "generated_authorities": {},
        "seed_site_bindings": [],
        "vintage_authorities": {"records": {}},
        "spec_sha256": "0" * 64,
    }
    for invalid in (protocol, missing_version):
        with pytest.raises(SpecValidationError):
            load_schema_registry().validate(
                {**base, "seed_protocol": invalid},
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
