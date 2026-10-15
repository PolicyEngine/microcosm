"""The release contract's version 1 golden fixtures (docs/release-contract.md).

The fixtures freeze C1 (the UK national line's manifests, and the contract's
inventory for that line) and C5 (the publisher's two pointer payloads and the
dataset registry). These tests hold every byte to ``SHA256SUMS`` and prove each
file is what the contract's own code accepts or writes. The build shard's
``test_release_contract_fixture_generation.py`` proves the generator reproduces
them.

A failure here after a deliberate contract change means: regenerate with
``uv run python tools/generate_release_contract_fixtures.py`` and update
docs/release-contract.md in the same pull request.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from microcosm.data import contract, loader, release
from microcosm.data.contract import ReleaseContractError, validate_release_dir
from microcosm.data.registry import DEFAULT_VARIANT, REGISTRY
from test_support.paths import paths_for

FIXTURES = paths_for("microcosm-data").tests / "fixtures" / "release_contract" / "v1"
UK_RELEASE_ID = contract._UK_NATIONAL_RELEASE_ID
UK_RELEASE_DIR = FIXTURES / "c1" / UK_RELEASE_ID
UK_NATIONAL_SPEC = REGISTRY[("uk", 2025, "national")]
REGENERATE = (
    "regenerate with `uv run python tools/generate_release_contract_fixtures.py` "
    "and update docs/release-contract.md"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


class _PointerHub:
    """The one Hub call the pointer readers make, served from the fixtures."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.requests: list[tuple[str, str, str]] = []

    def hf_hub_download(self, *, repo_id: str, filename: str, repo_type: str):
        self.requests.append((repo_id, filename, repo_type))
        assert filename == self.path.name
        return str(self.path)


def test_every_fixture_byte_is_pinned_by_sha256sums():
    listed = {}
    for line in (FIXTURES / "SHA256SUMS").read_text().splitlines():
        expected, relative_path = line.split("  ", 1)
        listed[relative_path] = expected
    on_disk = {
        path.relative_to(FIXTURES).as_posix()
        for path in FIXTURES.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert set(listed) == on_disk
    for relative_path, expected in listed.items():
        observed = hashlib.sha256((FIXTURES / relative_path).read_bytes()).hexdigest()
        assert observed == expected, f"{relative_path} changed; {REGENERATE}"


# --- C1 -----------------------------------------------------------------------


def test_uk_national_manifests_pass_every_check_but_the_omitted_files(tmp_path):
    """The manifests satisfy the whole release contract; the fixture omits only
    the diagnostics and the signed certification, which bind live gate digests
    (docs/release-contract.md, Golden fixtures)."""

    release_dir = tmp_path / UK_RELEASE_ID
    shutil.copytree(UK_RELEASE_DIR, release_dir)
    with pytest.raises(ReleaseContractError) as refusal:
        validate_release_dir(release_dir)
    assert refusal.value.failures == [
        "required file 'calibration_diagnostics.json' is missing.",
        "required file 'release_certification.json' is missing.",
    ]


def test_uk_national_manifests_pass_the_manifest_checks_directly():
    failures: list[str] = []
    build_manifest = _json(UK_RELEASE_DIR / "build_manifest.json")
    release_manifest = _json(UK_RELEASE_DIR / "release_manifest.json")
    contract._check_build_manifest(build_manifest, UK_RELEASE_ID, failures)
    contract._check_release_manifest(release_manifest, UK_RELEASE_ID, failures)
    contract._check_cross_manifest_consistency(
        build_manifest, release_manifest, None, failures
    )
    assert failures == []


def test_loader_selects_the_national_artifact_through_the_registry():
    release_manifest = _json(UK_RELEASE_DIR / "release_manifest.json")
    cut_tag = release_manifest["build"]["cut_tag"]
    certified = loader._certified_release_from_manifest(
        UK_NATIONAL_SPEC, UK_RELEASE_ID, release_manifest, expected_revision=cut_tag
    )
    assert certified.artifact_repo_id == UK_NATIONAL_SPEC.hf_repo
    assert certified.artifact_path == UK_NATIONAL_SPEC.filename
    assert certified.artifact_revision == cut_tag
    built = release_manifest["build"]
    assert certified.model.name == UK_NATIONAL_SPEC.engine_package
    assert certified.model.specifiers == (
        f"=={built['built_with_model_package']['version']}",
    )
    assert certified.core.specifiers == (
        f"=={built['built_with_core_package']['version']}",
    )


def test_inventory_matches_the_live_contract():
    inventory = _json(FIXTURES / "c1" / "inventory.json")
    certification = inventory["certification"]
    assert inventory["release_id"] == UK_RELEASE_ID
    assert inventory["line"] == contract.line_for_release_id(UK_RELEASE_ID)
    assert inventory["dataset_role"] == contract.dataset_role_for_line(
        inventory["line"]
    )
    assert (
        inventory["release_manifest_schema_version"]
        == contract.RELEASE_MANIFEST_SCHEMA_VERSION
    )
    assert inventory["required_release_files"] == list(
        contract.required_release_files(UK_RELEASE_ID)
    ), REGENERATE
    assert certification == {
        "file": contract._UK_RELEASE_CERTIFICATION_FILE,
        "schema_version": contract._UK_RELEASE_CERTIFICATION_SCHEMA_VERSION,
        "kind": contract._UK_RELEASE_CERTIFICATION_KIND,
        "required_fields": sorted(contract._UK_CERTIFICATION_REQUIRED_FIELDS),
        "part_evidence_files": dict(contract._UK_CERTIFICATION_PART_EVIDENCE_FILES),
        "score_receipt_file": contract._UK_CERTIFICATION_SCORE_RECEIPT_FILE,
        "gate_battery_phases": list(contract._UK_GATE_BATTERY_PHASES),
        "signing_key_env": contract._UK_GATE_BATTERY_SIGNING_KEY_ENV,
    }, REGENERATE


def test_inventory_files_are_the_files_the_manifests_ship():
    inventory = _json(FIXTURES / "c1" / "inventory.json")
    release_manifest = _json(UK_RELEASE_DIR / "release_manifest.json")
    shipped = {
        artifact["path"]
        for artifact in release_manifest["artifacts"].values()
        if artifact["kind"] == "diagnostics"
    }
    directory_files = set(inventory["release_directory_files"])
    assert directory_files == shipped | {"build_manifest.json", "release_manifest.json"}
    assert set(inventory["required_release_files"]) <= directory_files
    certification = inventory["certification"]
    assert {
        certification["file"],
        certification["score_receipt_file"],
        *certification["part_evidence_files"].values(),
    } <= directory_files


# --- C5 -----------------------------------------------------------------------


def test_repository_pointer_is_the_publishers_payload_and_reads_back():
    path = FIXTURES / "c5" / release.LATEST_POINTER_PATH
    payload = _json(path)
    assert (
        path.read_bytes()
        == json.dumps(
            release.latest_pointer_payload(
                payload["release_id"], updated_at=payload["updated_at"]
            ),
            indent=1,
        ).encode()
    )
    hub = _PointerHub(path)
    pointer = release.latest_release("policyengine/populace-us", api=hub)
    assert hub.requests == [("policyengine/populace-us", "latest.json", "dataset")]
    assert pointer.release_id == payload["release_id"]
    assert pointer.revision == pointer.release_id
    assert pointer.line is None
    assert set(pointer.paths) == {
        name.removesuffix(".json")
        for name in contract.required_release_files(pointer.release_id)
    }


def test_line_pointer_is_the_publishers_payload_and_reads_back():
    path = FIXTURES / "c5" / release.line_pointer_path("national")
    assert path == FIXTURES / "c5" / UK_NATIONAL_SPEC.pointer_path
    payload = _json(path)
    assert (
        path.read_bytes()
        == json.dumps(
            release.line_pointer_payload(
                payload["release_id"],
                line=payload["line"],
                revision=payload["revision"],
                updated_at=payload["updated_at"],
            ),
            indent=1,
        ).encode()
    )
    pointer = release.latest_line_release(
        UK_NATIONAL_SPEC.hf_repo, line="national", api=_PointerHub(path)
    )
    assert pointer.line == "national"
    assert pointer.release_id == UK_RELEASE_ID


def test_line_pointer_names_the_c1_release_at_its_cut_tag():
    pointer = _json(FIXTURES / "c5" / UK_NATIONAL_SPEC.pointer_path)
    release_manifest = _json(UK_RELEASE_DIR / "release_manifest.json")
    revisions = {
        artifact["revision"] for artifact in release_manifest["artifacts"].values()
    }
    assert pointer["release_id"] == release_manifest["build"]["build_id"]
    assert revisions == {pointer["revision"]} == {release_manifest["build"]["cut_tag"]}
    assert set(pointer["paths"].values()) == {
        f"releases/{UK_RELEASE_ID}/{name}"
        for name in contract.required_release_files(UK_RELEASE_ID)
    }


def test_registry_snapshot_matches_the_live_registry():
    snapshot = _json(FIXTURES / "c5" / "registry.json")
    assert snapshot["default_variant"] == DEFAULT_VARIANT
    assert snapshot["entries"] == [
        {"key": list(key), **dataclasses.asdict(REGISTRY[key])}
        for key in sorted(REGISTRY)
    ], f"the dataset registry changed; {REGENERATE} (C5)"
