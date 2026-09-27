"""Round-trip and physical-surface mutation gates for resolved bundles."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest
import yaml
from yaml.tokens import AliasToken, AnchorToken

from microcosm.build.spec_engine import (
    ResourceKind,
    SpecValidationError,
    canonical_yaml_bytes,
    emit_resolved_bundle,
    load_bundle,
    resolved_bundle_bytes,
)
from microcosm.build.spec_engine.legacy_adapter import (
    compile_to_legacy_payload,
    diff_legacy_payloads,
)
from microcosm.build.spec_engine.loader import bundle_lock_bytes
from microcosm.build.spec_engine.plan_lock import assert_plan_lock_current
from microcosm.build.spec_engine.yaml12 import load_yaml12


@pytest.fixture(scope="module")
def resolved_us():
    return load_bundle("us")


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _mutate_yaml(path: Path, mutation) -> None:
    value = load_yaml12(path.read_text(encoding="utf-8"), source=str(path))
    assert isinstance(value, dict)
    mutation(value)
    path.write_bytes(canonical_yaml_bytes(value, source=str(path)))


@pytest.mark.parametrize("country", ["be", "uk", "us"])
def test_resolved_bundle_round_trip_is_lossless_deterministic_and_alias_free(
    tmp_path: Path,
    country: str,
) -> None:
    resolved = load_bundle(country)
    expected_files = resolved_bundle_bytes(resolved)
    assert expected_files == resolved_bundle_bytes(resolved)

    root = emit_resolved_bundle(resolved, tmp_path / "first")
    assert {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == expected_files
    manifest = load_yaml12(
        (root / "country_package.json").read_text(encoding="utf-8"),
        source="country_package.json",
    )
    assert isinstance(manifest, dict)
    assert manifest["resources"] == [
        resource.descriptor.to_wire() for resource in resolved.resources
    ]
    assert not {
        "bundle.lock.json",
        "engine_abi.lock.json",
        "plan.lock.json",
    }.intersection(row["path"] for row in manifest["resources"])

    for resource in resolved.resources:
        if resource.descriptor.path.suffix not in {".yaml", ".yml"}:
            continue
        raw = (root / resource.descriptor.path).read_text(encoding="utf-8")
        assert not any(
            isinstance(token, AnchorToken | AliasToken) for token in yaml.scan(raw)
        )

    round_tripped = load_bundle(root)
    assert round_tripped.spec_sha256 == resolved.spec_sha256
    assert round_tripped.documentation_sha256 == resolved.documentation_sha256
    assert round_tripped.surfaces == resolved.surfaces
    assert round_tripped.generated_authorities == resolved.generated_authorities
    assert [
        (resource.descriptor, resource.domain.to_wire())
        for resource in round_tripped.resources
    ] == [
        (resource.descriptor, resource.domain.to_wire())
        for resource in resolved.resources
    ]
    assert (root / "bundle.lock.json").read_bytes() == bundle_lock_bytes(round_tripped)
    assert_plan_lock_current(round_tripped, root / "plan.lock.json")


def test_normative_mutation_changes_hash_and_names_legacy_payload_field(
    tmp_path: Path,
    resolved_us,
) -> None:
    baseline_root = emit_resolved_bundle(resolved_us, tmp_path / "baseline")
    mutated_root = shutil.copytree(baseline_root, tmp_path / "normative")
    relative = resolved_us.resource(ResourceKind.CALIBRATION).descriptor.path

    def mutate(document: dict[str, object]) -> None:
        solver = _mapping(document["solver"])
        stopping = _mapping(solver["stopping_contract"])
        stopping["max_epochs"] = int(stopping["max_epochs"]) + 1

    _mutate_yaml(mutated_root / relative, mutate)
    mutated = load_bundle(mutated_root)

    assert mutated.spec_sha256 != resolved_us.spec_sha256
    differences = diff_legacy_payloads(
        compile_to_legacy_payload(resolved_us),
        compile_to_legacy_payload(mutated),
    )
    paths = {difference.path for difference in differences}
    assert "/calibration_contract/solver/stopping_contract/max_epochs" in paths
    assert "/calibration_contract/solver/stopping/max_epochs" in paths


def test_documentation_mutation_leaves_spec_hash_unchanged(
    tmp_path: Path,
    resolved_us,
) -> None:
    baseline_root = emit_resolved_bundle(resolved_us, tmp_path / "baseline-docs")
    mutated_root = shutil.copytree(baseline_root, tmp_path / "documentation")
    relative = resolved_us.resource(ResourceKind.TAKE_UP).descriptor.path

    def mutate(document: dict[str, object]) -> None:
        programs = document["programs"]
        assert isinstance(programs, list)
        program = _mapping(programs[0])
        documentation = _mapping(program["documentation"])
        documentation["notes"] = f"{documentation['notes']} Review-only note."

    _mutate_yaml(mutated_root / relative, mutate)
    mutated = load_bundle(mutated_root)

    assert mutated.spec_sha256 == resolved_us.spec_sha256
    assert mutated.documentation_sha256 != resolved_us.documentation_sha256


def test_reemission_refuses_nonempty_destination(
    tmp_path: Path,
    resolved_us,
) -> None:
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "owned.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        emit_resolved_bundle(resolved_us, destination)
    assert (destination / "owned.txt").read_text(encoding="utf-8") == "preserve"


def test_reemission_refuses_resource_path_traversal(
    tmp_path: Path,
    resolved_us,
) -> None:
    resource = resolved_us.resources[0]
    escaped_descriptor = replace(
        resource.descriptor,
        path=PurePosixPath("../escaped.yaml"),
    )
    escaped_resource = replace(resource, descriptor=escaped_descriptor)
    malformed = replace(
        resolved_us,
        resources=(escaped_resource, *resolved_us.resources[1:]),
    )
    destination = tmp_path / "destination"

    with pytest.raises(
        SpecValidationError,
        match="resource path must be normalized and relative",
    ):
        emit_resolved_bundle(malformed, destination)

    assert not destination.exists()
    assert not (tmp_path / "escaped.yaml").exists()
