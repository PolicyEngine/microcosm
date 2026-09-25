"""Tests split from packages/microcosm-build/tests/test_spec_engine_loader.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.spec_engine_loader import *


def test_loads_typed_domains_injects_defaults_and_emits_valid_lock(tmp_path) -> None:
    root = _rich_minimal(tmp_path / "xx")
    spec = load_bundle(
        root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )

    assert spec.country == "xx"
    assert (
        spec.resource(ResourceKind.CATALOGS).domain.to_wire()["columns"][0]["contract"][
            "entity"
        ]
        == "person"
    )
    selection = spec.resource(ResourceKind.SELECTION).domain.to_wire()
    assert selection["exact_k"]["k"]["precedence"] == ("run_request_overrides_default")
    assert spec.columns[0].key == "person.age"
    assert spec.spec_binding.attestation == "mirror-attested"
    lock = bundle_lock_payload(spec)
    load_schema_registry().validate(lock, "locks.schema.json#/$defs/bundle_lock")
    assert bundle_lock_bytes(spec) == bundle_lock_bytes(spec)
    assert spec.seed_protocol is LEGACY_V1_PROTOCOL
    assert lock["seed_protocol"] == spec.seed_protocol.to_wire()


def test_resolved_selector_drives_lock_and_derived_v2_fails_closed(tmp_path) -> None:
    root = _rich_minimal(tmp_path / "xx")
    spec = load_bundle(
        root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )
    first = spec.seed_protocol.sites[0]
    changed = SeedProtocol(
        id=spec.seed_protocol.id,
        implementation_id=spec.seed_protocol.implementation_id,
        kernels=spec.seed_protocol.kernels,
        sites=(
            replace(first, reset_boundary="test_only_changed_boundary"),
            *spec.seed_protocol.sites[1:],
        ),
    )
    changed_spec = replace(spec, seed_protocol=changed)
    changed_lock = bundle_lock_payload(changed_spec)
    assert changed_lock["seed_protocol"] == changed.to_wire()
    assert (
        changed_lock["seed_protocol"]["implementation_sha256"]
        != bundle_lock_payload(spec)["seed_protocol"]["implementation_sha256"]
    )

    bundle_path = root / "bundle.yaml"
    bundle_path.write_text(
        bundle_path.read_text(encoding="utf-8").replace("legacy-v1", "derived-v2"),
        encoding="utf-8",
    )
    with pytest.raises(
        SpecResolutionError, match="unsupported F0 protocol 'derived-v2'"
    ):
        load_bundle(root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS))


def test_resolved_seed_protocol_expansion_is_spec_normative(
    tmp_path, monkeypatch
) -> None:
    root = _rich_minimal(tmp_path / "xx")
    first = load_bundle(
        root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )
    original_resolver = loader_module.resolve_cross_references

    def changed_resolver(*args, **kwargs):
        resolved = original_resolver(*args, **kwargs)
        protocol = resolved.seed_protocol
        changed = SeedProtocol(
            id=protocol.id,
            implementation_id=protocol.implementation_id,
            kernels=protocol.kernels,
            sites=(
                replace(
                    protocol.sites[0],
                    reset_boundary="test_only_normative_protocol_mutation",
                ),
                *protocol.sites[1:],
            ),
        )
        return replace(resolved, seed_protocol=changed)

    monkeypatch.setattr(loader_module, "resolve_cross_references", changed_resolver)
    second = load_bundle(
        root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )

    assert second.package_fingerprint == first.package_fingerprint
    assert second.spec_sha256 != first.spec_sha256
    assert (
        second.seed_protocol.implementation_sha256
        != first.seed_protocol.implementation_sha256
    )


def test_manifest_declared_set_reordering_does_not_change_spec_hash(tmp_path) -> None:
    first_root = _rich_minimal(tmp_path / "xx")
    first = load_bundle(
        first_root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )
    second_root = _rich_minimal(tmp_path / "xy")
    manifest_path = second_root / "country_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["country"] = "xx"
    manifest["resources"] = list(reversed(manifest["resources"]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    second = load_bundle(
        second_root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )
    assert second.spec_sha256 == first.spec_sha256
    assert second.package_fingerprint != first.package_fingerprint


def test_legacy_json_resources_load_through_the_strict_json_path(tmp_path) -> None:
    root = _rich_minimal(tmp_path / "xx")
    _append_legacy_json(root, "extras.json", '{"rows": [1, 2]}\n')
    spec = load_bundle(
        root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )
    assert spec.resource(ResourceKind.LEGACY_JSON).domain.to_wire() == {"rows": [1, 2]}


def test_legacy_json_resource_in_yaml_syntax_refuses(tmp_path) -> None:
    root = _rich_minimal(tmp_path / "xx")
    _append_legacy_json(root, "extras.json", "rows: [1, 2]\n")
    with pytest.raises(SpecParseError, match=r"extras\.json.*invalid JSON"):
        load_bundle(root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS))


def test_cross_reference_resolution_accepts_selected_registry(tmp_path) -> None:
    spec = load_bundle(
        _cross_ref_bundle(tmp_path / "xx"),
        kernel_registry=KernelRegistry.from_ids(
            ["survey_loader", "geo_assign", "geo_validate", "unused_library_kernel"]
        ),
    )
    assert {reference.namespace for reference in spec.references} == {
        "kernel",
        "source",
        "stream",
        "vintage",
    }


@pytest.mark.parametrize(
    ("source_ref", "message"),
    [("missing", "dangling source reference")],
)
def test_dangling_cross_reference_refuses(
    tmp_path, source_ref: str, message: str
) -> None:
    with pytest.raises(SpecResolutionError, match=message):
        load_bundle(
            _cross_ref_bundle(tmp_path / "xx", source_ref=source_ref),
            kernel_registry=KernelRegistry.from_ids(
                ["survey_loader", "geo_assign", "geo_validate"]
            ),
        )


def test_manifest_path_traversal_refuses_before_io(tmp_path) -> None:
    root = _write_bundle(
        tmp_path / "xx",
        {
            "bundle.yaml": "country: xx\nidentity_generation: 1\nseed_protocol: legacy-v1\n"
        },
        [
            {
                "path": "../bundle.yaml",
                "kind": "bundle",
                "schema_id": "bundle.schema.json",
            }
        ],
    )
    with pytest.raises(
        SpecValidationError, match="normalized and relative|invalid portable"
    ):
        load_bundle(root)
