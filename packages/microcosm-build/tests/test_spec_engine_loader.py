from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcosm.build.spec_engine import (
    KernelRegistry,
    ResourceKind,
    SpecResolutionError,
    SpecValidationError,
    bundle_lock_bytes,
    bundle_lock_payload,
    load_bundle,
    load_schema_registry,
)

ZERO_SHA = "0" * 64


def _write_bundle(root: Path, files: dict[str, str], rows: list[dict]) -> Path:
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "country": root.name,
        "resources": rows,
    }
    (root / "country_package.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _row(path: str, kind: str) -> dict[str, str]:
    return {"path": path, "kind": kind, "schema_id": f"{kind}.schema.json"}


def _rich_minimal(root: Path, *, note: str = "first", store: str = "local:a") -> Path:
    files = {
        "bundle.yaml": (
            "country: xx\n"
            "identity_generation: 1\n"
            "seed_protocol: legacy-v1\n"
            "status: documentation only\n"
        ),
        "vintages.yaml": (
            "records:\n"
            "  - id: ty2024\n"
            "    kind: tax_period\n"
            "    value: 2024\n"
        ),
        "catalogs.yaml": (
            "columns:\n"
            "  - key: person.age\n"
            "    contract:\n"
            "      entity: person\n"
            "      dtype: int64\n"
            "      period: vintage:ty2024\n"
            "      nullable: false\n"
            "      stability: public_stable\n"
            "    docs:\n"
            f"      description: {note}\n"
            "      citations: [official]\n"
        ),
        "publication.yaml": (
            "attempts:\n"
            "  model: append_only_events_then_terminal_seal\n"
            "  terminal_states: [landed, failed, expired]\n"
            "promotion:\n"
            "  latest_flip: human_gate\n"
            "  idempotency: required_key\n"
            "  recovery: [seal_ok_append_fail]\n"
            "release:\n"
            "  line:\n"
            "    value: microcosm-xx-2024\n"
            "    normative: true\n"
            "    note: documentation note\n"
            "  pattern: '{line}-stacked-f001-s0'\n"
            "  rungs: [f001]\n"
            "audit_chain:\n"
            "  kind: strict_linear\n"
            f"  store: {store}\n"
            "release_graph:\n"
            "  relations: [derived_from]\n"
        ),
        "selection.yaml": (
            "exact_k:\n"
            "  kernel: kernel:exact_k\n"
            "  k: {default: 100, surface: run_request}\n"
            "  pi_hi: 0.999\n"
            "  group_ids: none\n"
            "  on_infeasible: refuse\n"
            "  post_selection_weights: selected_keep_calibrated\n"
        ),
    }
    rows = [
        _row("bundle.yaml", "bundle"),
        _row("vintages.yaml", "vintages"),
        _row("catalogs.yaml", "catalogs"),
        _row("publication.yaml", "publication"),
        _row("selection.yaml", "selection"),
    ]
    return _write_bundle(root, files, rows)


def test_loads_typed_domains_injects_defaults_and_emits_valid_lock(tmp_path) -> None:
    root = _rich_minimal(tmp_path / "xx")
    spec = load_bundle(root, kernel_registry=KernelRegistry.from_ids(["exact_k"]))

    assert spec.country == "xx"
    assert spec.resource(ResourceKind.CATALOGS).domain.to_wire()["columns"][0][
        "contract"
    ]["entity"] == "person"
    selection = spec.resource(ResourceKind.SELECTION).domain.to_wire()
    assert selection["exact_k"]["k"]["precedence"] == (
        "run_request_overrides_default"
    )
    assert spec.columns[0].key == "person.age"
    assert spec.spec_binding.attestation == "mirror-attested"
    lock = bundle_lock_payload(spec)
    load_schema_registry().validate(lock, "locks.schema.json#/$defs/bundle_lock")
    assert bundle_lock_bytes(spec) == bundle_lock_bytes(spec)


def test_semantic_hash_has_golden_vector_and_surface_separation(tmp_path) -> None:
    first = load_bundle(
        _rich_minimal(tmp_path / "xx", note="first", store="local:a"),
        kernel_registry=KernelRegistry.from_ids(["exact_k"]),
    )
    # Pin the domain separator, normalization rules, schema-set receipt, and
    # exact normative projection as one reviewable golden vector.
    assert first.spec_sha256 == (
        "757ffd4d75810811e547b3e13edbd2b5fde06128a9ec80af13ecfad3f64a7a3e"
    )

    second_root = _rich_minimal(tmp_path / "xy", note="second", store="local:b")
    manifest_path = second_root / "country_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["country"] = "xx"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (second_root / "bundle.yaml").write_text(
        (second_root / "bundle.yaml").read_text(encoding="utf-8").replace(
            "country: xx", "country: xx"
        ),
        encoding="utf-8",
    )
    # Directory name is part of the CountrySpec seam, but load_bundle's typed
    # manifest is the country authority and supports fixture locations.
    second = load_bundle(
        second_root, kernel_registry=KernelRegistry.from_ids(["exact_k"])
    )
    assert second.spec_sha256 == first.spec_sha256
    assert second.documentation_sha256 != first.documentation_sha256
    assert second.package_fingerprint != first.package_fingerprint
    assert second.surfaces.operational != first.surfaces.operational


def test_manifest_declared_set_reordering_does_not_change_spec_hash(tmp_path) -> None:
    first_root = _rich_minimal(tmp_path / "xx")
    first = load_bundle(first_root, kernel_registry=KernelRegistry.from_ids(["exact_k"]))
    second_root = _rich_minimal(tmp_path / "xy")
    manifest_path = second_root / "country_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["country"] = "xx"
    manifest["resources"] = list(reversed(manifest["resources"]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    second = load_bundle(second_root, kernel_registry=KernelRegistry.from_ids(["exact_k"]))
    assert second.spec_sha256 == first.spec_sha256
    assert second.package_fingerprint != first.package_fingerprint


def _cross_ref_bundle(root: Path, *, source_ref: str = "survey") -> Path:
    files = {
        "bundle.yaml": "country: xx\nidentity_generation: 1\nseed_protocol: legacy-v1\n",
        "sources.yaml": (
            "sources:\n"
            "  - id: survey\n"
            "    role: survey\n"
            f"    sha256: '{ZERO_SHA}'\n"
            "    loader: kernel:survey_loader\n"
            "    vintages: [vintage:survey2024]\n"
        ),
        "vintages.yaml": (
            "records:\n"
            "  - id: survey2024\n"
            "    kind: survey_period\n"
            "    value: 2024\n"
        ),
        "spine.yaml": (
            "channels:\n"
            "  - id: survey_channel\n"
            f"    source: {source_ref}\n"
            "    observed_geography: state\n"
            "assembly:\n"
            "  mass_anchor_channel: survey_channel\n"
            "  shared_dtype_policy: canonical_string_storage\n"
            "support_roles: []\n"
        ),
        "geography.yaml": (
            "phase: legacy\n"
            "assignment:\n"
            "  anchor: puma\n"
            "  order: legacy_post_transfer\n"
            "  kernels:\n"
            "    assign: kernel:geo_assign\n"
            "    validate: kernel:geo_validate\n"
            "  draw: {}\n"
            "  identified_county_source: source:survey\n"
            "  derive: [state_fips]\n"
            "  assertions: [observed_preserved]\n"
            "  ladder_source: puma_ladder\n"
            "  seed: stream:geography_legacy\n"
        ),
    }
    rows = [
        _row("bundle.yaml", "bundle"),
        _row("sources.yaml", "sources"),
        _row("vintages.yaml", "vintages"),
        _row("spine.yaml", "spine"),
        _row("geography.yaml", "geography"),
    ]
    return _write_bundle(root, files, rows)


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
        {"bundle.yaml": "country: xx\nidentity_generation: 1\nseed_protocol: legacy-v1\n"},
        [
            {
                "path": "../bundle.yaml",
                "kind": "bundle",
                "schema_id": "bundle.schema.json",
            }
        ],
    )
    with pytest.raises(SpecValidationError, match="normalized and relative|invalid portable"):
        load_bundle(root)
