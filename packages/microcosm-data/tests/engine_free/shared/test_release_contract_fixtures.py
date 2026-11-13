"""The release contract's version 1 golden fixtures (docs/release-contract.md).

The fixtures freeze C1 (one build folder's manifests for the UK national release
and for the US national release, and the contract's inventory for the UK
national release) and C5 (each repository's ``releases.json`` and the dataset
registry). These tests hold every byte to ``SHA256SUMS``, hold ``writers.json``
to the files it must name, and prove each file is what the contract's own code
accepts or reads. They import only ``microcosm.data``: this package supports
Python versions the emitter interface does not. The build shard's
``test_release_contract_fixture_generation.py`` proves that the generator
reproduces the bytes and that each ``releases.json`` is the interface writer's
serialization.

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

from microcosm.data import contract, loader
from microcosm.data.contract import ReleaseContractError, validate_release_dir
from microcosm.data.registry import DEFAULT_VARIANT, REGISTRY
from microcosm.data.releases_record import (
    RELEASES_RECORD_PATH,
    build_folder,
    parse_releases_record,
)
from test_support.paths import paths_for

FIXTURES = paths_for("microcosm-data").tests / "fixtures" / "release_contract" / "v1"
GENERATOR = "tools/generate_release_contract_fixtures.py"
UK_RELEASE = contract._UK_NATIONAL_RELEASE_ID
US_RELEASE = contract.US_NATIONAL_RELEASE_ID
#: Each golden build's release, with the registry entry that loads it. The UK
#: national release is the UK 2025 default load since microcosm#1432 (Max's
#: #823 ruling).
GOLDENS = {
    "uk": (UK_RELEASE, REGISTRY[("uk", 2025, DEFAULT_VARIANT)]),
    "us": (US_RELEASE, REGISTRY[("us", 2024, DEFAULT_VARIANT)]),
}
COUNTRIES = sorted(GOLDENS)
REGENERATE = (
    f"regenerate with `uv run python {GENERATOR}` and update docs/release-contract.md"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _listed() -> dict[str, str]:
    """Each file ``SHA256SUMS`` lists, with its digest."""

    listed = {}
    for line in (FIXTURES / "SHA256SUMS").read_text().splitlines():
        expected, relative_path = line.split("  ", 1)
        listed[relative_path] = expected
    return listed


def _manifests(country: str) -> tuple[dict, dict]:
    """A golden build folder's build manifest and release manifest."""

    release_dir = FIXTURES / "c1" / GOLDENS[country][0]
    return (
        _json(release_dir / "build_manifest.json"),
        _json(release_dir / "release_manifest.json"),
    )


def _record(country: str):
    """A golden ``releases.json``, as this package's reader parses it."""

    return parse_releases_record(
        (FIXTURES / "c5" / country / RELEASES_RECORD_PATH).read_bytes()
    )


def _refusals(country: str, tmp_path: Path) -> list[str]:
    """What ``validate_release_dir`` refuses a golden build folder for."""

    release_id = GOLDENS[country][0]
    release_dir = tmp_path / release_id
    shutil.copytree(FIXTURES / "c1" / release_id, release_dir)
    with pytest.raises(ReleaseContractError) as refusal:
        validate_release_dir(release_dir)
    return refusal.value.failures


def _keys(value: object) -> set[str]:
    """Every object key in a JSON document, at any depth."""

    if isinstance(value, dict):
        return set(value).union(*(_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value))
    return set()


def test_every_fixture_byte_is_pinned_by_sha256sums():
    listed = _listed()
    on_disk = {
        path.relative_to(FIXTURES).as_posix()
        for path in FIXTURES.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert set(listed) == on_disk
    for relative_path, expected in listed.items():
        observed = hashlib.sha256((FIXTURES / relative_path).read_bytes()).hexdigest()
        assert observed == expected, f"{relative_path} changed; {REGENERATE}"


def test_writers_json_names_a_writer_for_exactly_the_pinned_files():
    """``writers.json`` names, for every file ``SHA256SUMS`` lists and for no
    other, the calls that produce its bytes. A manifest or a ``releases.json``
    is a golden: its last writer is the function that serializes the same file
    in a real build, never the generator."""

    writers = _json(FIXTURES / "writers.json")["files"]
    assert set(writers) == set(_listed()), REGENERATE
    for relative_path, entry in writers.items():
        assert entry["writers"], relative_path
        for writer in entry["writers"]:
            assert set(writer) == {"module", "function"}, relative_path
            assert all(isinstance(name, str) and name for name in writer.values())
    goldens = {
        *(
            f"c1/{release_id}/{name}"
            for release_id, _ in GOLDENS.values()
            for name in ("build_manifest.json", "release_manifest.json")
        ),
        *(f"c5/{country}/{RELEASES_RECORD_PATH}" for country in COUNTRIES),
    }
    assert goldens <= set(writers)
    for relative_path, entry in writers.items():
        modules = {writer["module"] for writer in entry["writers"]}
        if relative_path in goldens:
            assert GENERATOR not in modules, relative_path
        else:
            # The inventory, the registry snapshot and writers.json itself.
            assert modules == {GENERATOR}, relative_path


def test_no_fixture_holds_a_signed_certification():
    """No fixture file is a certification, and no fixture document carries an
    ``attestation`` or a ``signature`` at any depth."""

    assert contract._UK_RELEASE_CERTIFICATION_FILE not in {
        Path(relative_path).name for relative_path in _listed()
    }
    for path in FIXTURES.rglob("*.json"):
        document = _json(path)
        assert document.get("kind") != contract._UK_RELEASE_CERTIFICATION_KIND, path
        assert not {"attestation", "signature"} & _keys(document), path


# --- C1 -----------------------------------------------------------------------


@pytest.mark.parametrize("country", COUNTRIES)
def test_golden_manifests_pass_the_manifest_checks_directly(country):
    release_id = GOLDENS[country][0]
    build_manifest, release_manifest = _manifests(country)
    failures: list[str] = []
    contract._check_build_manifest(build_manifest, release_id, failures)
    contract._check_release_manifest(release_manifest, release_id, failures)
    contract._check_cross_manifest_consistency(
        build_manifest, release_manifest, None, failures
    )
    assert failures == []


@pytest.mark.parametrize("country", COUNTRIES)
def test_golden_manifests_have_the_build_folder_layout(country):
    """Both manifests name the release and the same build as two strings and
    carry none of the fields that named a build before build folders. Each
    artifact carries its role and no revision."""

    release_id = GOLDENS[country][0]
    assert contract.publishes_to_build_folder(release_id)
    build_manifest, release_manifest = _manifests(country)
    assert isinstance(build_manifest["build"], str) and build_manifest["build"]
    for manifest in (build_manifest, release_manifest):
        assert manifest["release"] == release_id
        assert manifest["build"] == build_manifest["build"]
        assert not set(contract.RETIRED_BUILD_FIELDS) & set(manifest)
    assert (
        release_manifest["schema_version"] == contract.RELEASE_MANIFEST_SCHEMA_VERSION
    )
    artifacts = release_manifest["artifacts"]
    assert not any("revision" in artifact for artifact in artifacts.values())
    assert {key: artifact["role"] for key, artifact in artifacts.items()} == {
        key: contract.release_artifact_role(key, artifact["kind"])
        for key, artifact in artifacts.items()
    }


def test_uk_national_golden_is_refused_only_for_the_files_it_omits(tmp_path):
    """``validate_release_dir`` refuses the UK golden for exactly the two
    required files it omits on purpose: both bind live gate digests
    (docs/release-contract.md, Golden fixtures). The assembler's own tests
    validate a full signed directory."""

    assert _refusals("uk", tmp_path) == [
        "required file 'calibration_diagnostics.json' is missing.",
        "required file 'release_certification.json' is missing.",
    ]


def test_us_national_golden_is_refused_only_for_the_files_it_omits(tmp_path):
    """``validate_release_dir`` refuses the US golden for exactly five things,
    each caused by one of the three files it omits on purpose: the calibration
    diagnostics, the source coverage, and the stored-input verdict, which
    binds a digest of the live stored-column register. The verdict's absence
    draws three refusals: each manifest records ``stored_inputs`` as
    ``not_evaluated``, and the release manifest declares no such artifact."""

    failures = _refusals("us", tmp_path)
    assert failures[:2] == [
        "required file 'calibration_diagnostics.json' is missing.",
        f"required file {contract.US_SOURCE_COVERAGE_DIAGNOSTICS_FILE!r} is missing.",
    ]
    stored_inputs = failures[2:]
    assert [failure.split(" ", 1)[0] for failure in stored_inputs] == [
        "build_manifest.json",
        "release_manifest.json",
        "release_manifest.json",
    ]
    for failure in stored_inputs:
        assert repr(contract.US_STORED_INPUTS_EVIDENCE_KEY) in failure
    assert "'not_evaluated'" in stored_inputs[0]
    assert "'not_evaluated'" in stored_inputs[1]
    assert repr(contract.US_STORED_INPUTS_EVIDENCE_FILE) in stored_inputs[2]


def test_uk_national_gates_are_the_certifications_part_censuses():
    """The build manifest copies each certification part's ``statuses``, which
    the contract requires to be a census of that part's gates, all passed."""

    gates = _manifests("uk")[0]["gates"]
    assert set(gates) == set(contract._UK_CERTIFICATION_PART_EVIDENCE_FILES)
    for part, census in gates.items():
        assert list(census) == ["passed"], part
        assert isinstance(census["passed"], int) and census["passed"] > 0, part


def test_us_national_golden_records_a_passing_calibration_gate():
    """The US golden is a build whose calibration gate passed, and it binds no
    gate verdict: every gate-evidence entry reads ``not_evaluated`` in both
    manifests, and the stored-input verdict is one of those entries."""

    build_manifest, release_manifest = _manifests("us")
    calibration = build_manifest["gates"]["calibration"]
    assert calibration["passed"] is True
    assert calibration["failures"] == []
    for manifest in (build_manifest, release_manifest):
        assert contract.US_STORED_INPUTS_EVIDENCE_KEY in manifest["gate_evidence"]
        assert set(manifest["gate_evidence"].values()) == {"not_evaluated"}


def test_inventory_matches_the_live_contract():
    inventory = _json(FIXTURES / "c1" / "inventory.json")
    certification = inventory["certification"]
    assert inventory["release_id"] == UK_RELEASE
    assert inventory["line"] == contract.line_for_release_id(UK_RELEASE)
    assert inventory["dataset_role"] == contract.dataset_role_for_line(
        inventory["line"]
    )
    assert (
        inventory["release_role"] == contract.RELEASE_ROLES[inventory["dataset_role"]]
    )
    assert inventory["publishes_to_build_folder"] is True
    assert contract.publishes_to_build_folder(UK_RELEASE)
    assert inventory["retired_build_fields"] == list(contract.RETIRED_BUILD_FIELDS)
    assert (
        inventory["release_manifest_schema_version"]
        == contract.RELEASE_MANIFEST_SCHEMA_VERSION
    )
    assert inventory["required_release_files"] == list(
        contract.required_release_files(UK_RELEASE)
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
    release_manifest = _manifests("uk")[1]
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


@pytest.mark.parametrize("country", COUNTRIES)
def test_releases_record_lists_the_golden_build_as_its_releases_current_build(
    country,
):
    """Each golden ``releases.json`` is one repository's record after one
    publication and one promotion: it lists the C1 golden build, not
    withdrawn, as the current build of the national release, gives that
    release the role its manifest's ``dataset_role`` maps to, and names it as
    the default."""

    release_id = GOLDENS[country][0]
    build_manifest, release_manifest = _manifests(country)
    record = _record(country)
    assert record.default == release_id
    assert set(record.releases) == {release_id}
    entry = record.releases[release_id]
    assert entry.current == build_manifest["build"] == release_manifest["build"]
    assert set(entry.builds) == {entry.current}
    assert entry.builds[entry.current].withdrawn is False
    dataset_role = release_manifest.get(
        "dataset_role", contract.NATIONAL_DEFAULT_DATASET_ROLE
    )
    assert entry.role == contract.RELEASE_ROLES[dataset_role]
    if country == "us":
        assert entry.role == contract.US_RELEASES[release_id]


@pytest.mark.parametrize("country", COUNTRIES)
def test_loader_selects_each_golden_build_through_its_releases_record(country):
    """The registry entry names the golden's release. Given that release's
    entry in ``releases.json``, the loader selects the dataset inside the
    current build's folder, at no revision, with the packages the manifest
    says the build ran."""

    release_id, spec = GOLDENS[country]
    assert spec.release == release_id
    entry = _record(country).releases[release_id]
    release_manifest = _manifests(country)[1]
    certified = loader._certified_release_from_manifest(
        spec,
        release_id,
        release_manifest,
        expected_revision=None,
        build=entry.current,
        release_role=entry.role,
    )
    # The file name a new build declares: the registry's ``filename``, or its
    # ``build_folder_filename`` where the two differ (the US national release).
    declared = spec.build_folder_filename or spec.filename
    assert certified.build == entry.current
    assert certified.artifact_repo_id == spec.hf_repo
    assert certified.artifact_path == (
        f"{build_folder(release_id, entry.current)}/{declared}"
    )
    assert certified.artifact_revision is None
    assert certified.model.name == spec.engine_package
    assert certified.model.specifiers == (
        f"=={release_manifest['built_with_model_package']['version']}",
    )
    assert certified.core.specifiers == (
        f"=={release_manifest['built_with_core_package']['version']}",
    )


@pytest.mark.parametrize("country", COUNTRIES)
def test_loader_refuses_a_golden_build_under_the_other_role(country):
    """``releases.json`` gives a release one role, and the loader refuses a
    manifest whose ``dataset_role`` maps to another."""

    release_id, spec = GOLDENS[country]
    entry = _record(country).releases[release_id]
    other_role = contract.RELEASE_ROLES[contract.NON_DEFAULT_LOCAL_AREA_DATASET_ROLE]
    assert entry.role != other_role
    with pytest.raises(ValueError, match="is listed with role"):
        loader._certified_release_from_manifest(
            spec,
            release_id,
            _manifests(country)[1],
            expected_revision=None,
            build=entry.current,
            release_role=other_role,
        )


def test_registry_snapshot_matches_the_live_registry():
    snapshot = _json(FIXTURES / "c5" / "registry.json")
    assert snapshot["default_variant"] == DEFAULT_VARIANT
    assert snapshot["entries"] == [
        {"key": list(key), **dataclasses.asdict(REGISTRY[key])}
        for key in sorted(REGISTRY)
    ], f"the dataset registry changed; {REGENERATE} (C5)"
