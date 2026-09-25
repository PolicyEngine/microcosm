"""Tests split from packages/microcosm-build/tests/test_spec_engine_loader.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.spec_engine_loader import *


def test_semantic_hash_has_golden_vector_and_surface_separation(tmp_path) -> None:
    __import__("policyengine_us")
    first = load_bundle(
        _rich_minimal(tmp_path / "xx", note="first", store="local:a"),
        kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS),
    )
    # Pin the domain separator, normalization rules, schema-set receipt, and
    # exact normative projection as one reviewable golden vector.
    assert first.spec_sha256 == (
        "db3fdab1b4f61040ee3f6c6b4ada65435ab130f67f181cb9e80688d40f4b7a9a"
    )

    second_root = _rich_minimal(tmp_path / "xy", note="second", store="local:b")
    manifest_path = second_root / "country_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["country"] = "xx"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (second_root / "bundle.yaml").write_text(
        (second_root / "bundle.yaml")
        .read_text(encoding="utf-8")
        .replace("country: xx", "country: xx"),
        encoding="utf-8",
    )
    # Directory name is part of the CountrySpec seam, but load_bundle's typed
    # manifest is the country authority and supports fixture locations.
    second = load_bundle(
        second_root, kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS)
    )
    assert second.spec_sha256 == first.spec_sha256
    assert second.documentation_sha256 != first.documentation_sha256
    assert second.package_fingerprint != first.package_fingerprint
    assert second.surfaces.operational != first.surfaces.operational
