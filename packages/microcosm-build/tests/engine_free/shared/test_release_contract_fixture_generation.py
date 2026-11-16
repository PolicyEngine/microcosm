"""The release-contract fixture generator and the writers it freezes.

``tools/generate_release_contract_fixtures.py`` writes the C1 and C5 golden
fixtures (docs/release-contract.md) with each file's own writer. The data
shard's ``test_release_contract_fixtures.py`` checks the committed bytes with
``microcosm.data`` alone. This module checks what needs the build shard: that
the generator reproduces the bytes, that every writer ``writers.json`` names
exists and is the one its real build path calls, that each ``releases.json`` is
the emitter interface's serialization, and that the UK assembler's manifests
satisfy the release contract for every input, not only the fixture's.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from microcosm_emitter_interface.releases import (
    ReleasesRecord,
    add_build,
    make_current,
)

from microcosm.build.staging_v2 import validate_staging_delivery
from microcosm.build.uk_runtime.spi_dependency import (
    UK_SPI_PUBLIC_USE_TAPE_KEY,
    uk_spi_public_use_tape_marker,
)
from microcosm.build.uk_runtime.spi_income import SPI_DONOR_UKDS_STUDY
from microcosm.data import contract, loader
from microcosm.data.registry import DEFAULT_VARIANT, REGISTRY
from microcosm.data.releases_record import (
    RELEASES_RECORD_PATH,
    build_folder,
    parse_releases_record,
)
from test_support.paths import paths_for

ROOT = paths_for("microcosm-build").repository
FIXTURES = paths_for("microcosm-data").tests / "fixtures" / "release_contract" / "v1"
ASSEMBLER_TOOL = "tools/assemble_uk_release_dir.py"
US_BUILDER_TOOL = "tools/build_us_fiscal_refresh_release.py"


def _load_assembler():
    path = ROOT / ASSEMBLER_TOOL
    spec = importlib.util.spec_from_file_location("assemble_uk_release_dir", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ASSEMBLER = _load_assembler()
#: The national release is the UK 2025 default load since microcosm#1432 (Max's
#: #823 ruling).
UK_NATIONAL_SPEC = REGISTRY[("uk", 2025, DEFAULT_VARIANT)]
UK_NATIONAL_ROLE = contract.RELEASE_ROLES[contract.NATIONAL_DEFAULT_DATASET_ROLE]


def test_release_contract_fixtures_are_reproducible():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "generate_release_contract_fixtures.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )


# --- writers.json ---------------------------------------------------------------


def _tool_function(tool: str, function: str) -> ast.FunctionDef | None:
    """A tool's top-level function, read from its source without importing it."""

    for node in ast.parse((ROOT / tool).read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return node
    return None


def _called_in(tool: str, function: str) -> set[str]:
    """The names a tool's top-level function calls."""

    definition = _tool_function(tool, function)
    assert definition is not None, f"{tool} defines no {function}"
    return {
        node.func.id
        for node in ast.walk(definition)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _writer_exists(module: str, function: str) -> bool:
    """Whether ``writers.json``'s writer resolves to a function.

    A tool is named by its path and read as source, so the check imports no
    tool. Any other module is imported.
    """

    if module.endswith(".py"):
        return _tool_function(module, function) is not None
    target: object = importlib.import_module(module)
    for part in function.split("."):
        target = getattr(target, part, None)
    return callable(target)


def test_every_writer_the_fixtures_name_exists():
    writers = json.loads((FIXTURES / "writers.json").read_text())["files"]
    missing = [
        (relative_path, writer)
        for relative_path, entry in writers.items()
        for writer in entry["writers"]
        if not _writer_exists(writer["module"], writer["function"])
    ]
    assert missing == []


def test_each_manifest_golden_names_the_writers_its_real_build_calls():
    """A golden freezes live bytes only while the real build path still calls
    the writers ``writers.json`` names for it: the UK assembler's
    ``_stage_and_finalize`` calls the pure function and its serializer, the US
    builder's ``_main`` calls ``_build_manifests`` and
    ``_finish_build_folder_release``, and that function calls
    ``rewrite_release_dir``."""

    writers = json.loads((FIXTURES / "writers.json").read_text())["files"]

    def named(release_id: str) -> list[tuple[str, str]]:
        entries = {
            json.dumps(writers[f"c1/{release_id}/{name}"]["writers"])
            for name in ("build_manifest.json", "release_manifest.json")
        }
        assert len(entries) == 1, release_id
        return [
            (writer["module"], writer["function"])
            for writer in json.loads(entries.pop())
        ]

    assert named(contract._UK_NATIONAL_RELEASE_ID) == [
        (ASSEMBLER_TOOL, "uk_national_release_manifests"),
        (ASSEMBLER_TOOL, "_write_json"),
    ]
    assert {"uk_national_release_manifests", "_write_json"} <= _called_in(
        ASSEMBLER_TOOL, "_stage_and_finalize"
    )

    assert named(contract.US_NATIONAL_RELEASE_ID) == [
        (US_BUILDER_TOOL, "_build_manifests"),
        (US_BUILDER_TOOL, "_finish_build_folder_release"),
        ("microcosm.data.build_folder_manifests", "rewrite_release_dir"),
    ]
    assert {"_build_manifests", "_finish_build_folder_release"} <= _called_in(
        US_BUILDER_TOOL, "_main"
    )
    assert "rewrite_release_dir" in _called_in(
        US_BUILDER_TOOL, "_finish_build_folder_release"
    )


# --- releases.json --------------------------------------------------------------


@pytest.mark.parametrize("country", ["uk", "us"])
def test_each_releases_record_is_the_interface_writers_serialization(country):
    """Each golden ``releases.json`` is byte for byte what the emitter
    interface serializes, and one publication (``add_build`` without
    ``make_current``) then one promotion (``make_current``) give every field of
    it but ``default``, which no interface change sets. This repository's
    reader, which repeats the interface's rules, reads the same record."""

    raw = (FIXTURES / "c5" / country / RELEASES_RECORD_PATH).read_bytes()
    record = ReleasesRecord.parse(raw)
    assert record.serialize() == raw

    (release, entry) = next(iter(record.releases.items()))
    assert list(record.releases) == [release]
    published = add_build(
        ReleasesRecord.empty(),
        release=release,
        build=entry.current,
        release_role=entry.role,
        published_at=entry.builds[entry.current].published_at,
        make_current=False,
    )
    assert published.releases[release].current is None
    promoted = make_current(published, release=release, build=entry.current)
    assert promoted.default is None
    assert dataclasses.replace(promoted, default=release).serialize() == raw

    read = parse_releases_record(raw)
    assert read.default == record.default == release
    assert set(read.releases) == set(record.releases)
    read_entry = read.releases[release]
    assert (read_entry.role, read_entry.current) == (entry.role, entry.current)
    assert {
        build: (listed.published_at, listed.withdrawn)
        for build, listed in read_entry.builds.items()
    } == {
        build: (listed.published_at, listed.withdrawn)
        for build, listed in entry.builds.items()
    }


# --- The UK assembler's manifests, for every input --------------------------------

_HEX = "0123456789abcdef"
_digest = st.text(_HEX, min_size=64, max_size=64)
_version = st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True)
_attempt_suffix = st.builds(
    "{}{:02d}{:02d}T{:02d}{:02d}{:02d}Z-{}".format,
    st.integers(2026, 2099).map(str),
    st.integers(1, 12),
    st.integers(1, 28),
    st.integers(0, 23),
    st.integers(0, 59),
    st.integers(0, 59),
    st.text(_HEX, min_size=8, max_size=8),
)
#: A shippable part's statuses: the certifier's census of its gates by status
#: (``_status_census``), which the contract requires to be all "passed".
_statuses = st.integers(1, 500).map(lambda count: {"passed": count})
_staging = st.sampled_from(
    [
        {
            "contract_version": 2,
            "enabled": True,
            "mode": "local_and_remote",
            "run_id": "uk-national-property",
            "configured_repository": "policyengine/populace-uk-staging",
            "upload_attempts": 3,
            "upload_successes": 3,
            "read_back": "passed",
            "last_error_code": None,
            "opt_out_reason": None,
        },
        {
            "contract_version": 2,
            "enabled": False,
            "mode": "disabled",
            "run_id": None,
            "configured_repository": None,
            "upload_attempts": 0,
            "upload_successes": 0,
            "read_back": "not_requested",
            "last_error_code": None,
            "opt_out_reason": "--no-staging",
        },
    ]
)
#: The SPI Public Use Tape marker of a run whose stages read the tape or not.
_spi_marker = st.sampled_from(
    [
        uk_spi_public_use_tape_marker([]),
        uk_spi_public_use_tape_marker(
            [
                SimpleNamespace(
                    stage="property_stage",
                    artifacts=[{"ukds_study_number": SPI_DONOR_UKDS_STUDY}],
                )
            ]
        ),
    ]
)


@st.composite
def _manifest_inputs(draw):
    evidence_names = [name for _, _, name in ASSEMBLER._EVIDENCE]
    return {
        "measured": {name: draw(_digest) for name in ("candidate", *evidence_names)},
        "calibration_filename": f"{ASSEMBLER._DATASET_KEY}_calibration.npz",
        "calibration_sha": draw(_digest),
        "household_weight_year": {
            "dataset_year": draw(st.integers(2020, 2040)),
            "basis": draw(st.sampled_from(["dataset_year", "as_solved"])),
        },
        # The build's name is the calibration attempt ID.
        "attempt_id": "uk-frs-calibration-attempt-" + draw(_attempt_suffix),
        "runtime": {package: draw(_version) for package in ASSEMBLER._RUNTIME_PACKAGES},
        "code_pin": draw(st.text(_HEX, min_size=40, max_size=40)),
        "target_surface": {
            "sha256": draw(_digest),
            "n_targets": draw(st.integers(1, 100_000)),
        },
        "target_registry": {
            "version": draw(st.text(_HEX, min_size=1, max_size=16)),
            "n_specs": draw(st.integers(1, 100_000)),
        },
        "spine_part": {"statuses": draw(_statuses)},
        "seam_part": {"statuses": draw(_statuses)},
        "release_cut_part": {"statuses": draw(_statuses)},
        "staging_delivery": validate_staging_delivery(draw(_staging)),
        "spi_marker": draw(_spi_marker),
        "created_at": "2026-01-01T00:00:00+00:00",
        "publisher_labels": draw(
            st.dictionaries(st.text(min_size=1), st.text(min_size=1), max_size=3)
        ),
    }


@settings(max_examples=60, deadline=None)
@given(_manifest_inputs())
def test_national_manifests_satisfy_the_release_contract(inputs):
    """Invariant: for any valid inputs, the assembler's two manifests

    * pass the contract's manifest checks with no failure;
    * name the national release and the same build, and carry none of the
      fields that named a build before build folders;
    * give every artifact its role and no revision;
    * ship exactly the signed evidence as their diagnostics;
    * load through ``releases.json``: once the build is published and
      promoted, the loader selects it inside its own folder, at no revision,
      with the digest and the package versions the inputs gave;
    * restate the SPI Public Use Tape marker;
    * copy each certification part's census;
    * depend on the inputs alone, key order included.
    """

    build_manifest, release_manifest = ASSEMBLER.uk_national_release_manifests(**inputs)
    release_id = contract._UK_NATIONAL_RELEASE_ID
    build = inputs["attempt_id"]

    failures: list[str] = []
    contract._check_build_manifest(build_manifest, release_id, failures)
    contract._check_release_manifest(release_manifest, release_id, failures)
    contract._check_cross_manifest_consistency(
        build_manifest, release_manifest, None, failures
    )
    assert failures == []

    for manifest in (build_manifest, release_manifest):
        assert (manifest["release"], manifest["build"]) == (release_id, build)
        assert not set(contract.RETIRED_BUILD_FIELDS) & set(manifest)

    artifacts = release_manifest["artifacts"]
    assert not any("revision" in artifact for artifact in artifacts.values())
    assert {key: artifact["role"] for key, artifact in artifacts.items()} == {
        key: contract.release_artifact_role(key, artifact["kind"])
        for key, artifact in artifacts.items()
    }
    assert {
        artifact["path"]
        for artifact in artifacts.values()
        if artifact["kind"] == "diagnostics"
    } == {filename for _, filename, _ in ASSEMBLER._EVIDENCE}
    assert set(contract.required_release_files(release_id)) <= {
        "build_manifest.json",
        "release_manifest.json",
        *(filename for _, filename, _ in ASSEMBLER._EVIDENCE),
    }

    # The record a publication and then a promotion of this build give, as the
    # interface serializes it and this repository's reader parses it.
    published = add_build(
        ReleasesRecord.empty(),
        release=release_id,
        build=build,
        release_role=UK_NATIONAL_ROLE,
        published_at=inputs["created_at"],
        make_current=False,
    )
    promoted = make_current(published, release=release_id, build=build)
    entry = parse_releases_record(promoted.serialize()).releases[release_id]
    certified = loader._certified_release_from_manifest(
        UK_NATIONAL_SPEC,
        release_id,
        release_manifest,
        expected_revision=None,
        build=entry.current,
        release_role=entry.role,
    )
    assert certified.artifact_path == (
        f"{build_folder(release_id, build)}/{UK_NATIONAL_SPEC.filename}"
    )
    assert certified.artifact_revision is None
    assert certified.artifact_sha256 == inputs["measured"]["candidate"]
    assert certified.model.built_version == inputs["runtime"]["policyengine-uk"]
    assert certified.core.built_version == inputs["runtime"]["policyengine-core"]

    # The release restates the SPI Public Use Tape marker: the flag on the
    # microdata's own entry, the whole marker under licensed_data.
    microdata = artifacts[release_manifest["default_datasets"]["national"]]
    assert (
        microdata["spi_public_use_tape_dependent"] is inputs["spi_marker"]["dependent"]
    )
    assert release_manifest["licensed_data"] == {
        UK_SPI_PUBLIC_USE_TAPE_KEY: inputs["spi_marker"]
    }

    # The manifest records each part's census exactly as the certification signed it.
    assert build_manifest["gates"] == {
        "spine": inputs["spine_part"]["statuses"],
        "calibration_seam": inputs["seam_part"]["statuses"],
        "release_cut": inputs["release_cut_part"]["statuses"],
    }

    # Deterministic, key order included: the builder is a function of its inputs.
    again = ASSEMBLER.uk_national_release_manifests(**inputs)
    assert [json.dumps(document, indent=1) for document in again] == [
        json.dumps(document, indent=1)
        for document in (build_manifest, release_manifest)
    ]
