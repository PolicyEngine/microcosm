"""Rewriting a release directory into the build-folder layout (microcosm#1570).

Every US release writer builds its manifests in the tagged shape it has
always written, then ``microcosm.data.build_folder_manifests`` rewrites them
for its release: ``release`` and ``build`` as strings, the build object's
fields at the top level, a ``role`` and no ``revision`` on each artifact,
and the checksum ledger's lines for the two manifests refreshed. It also
gives the emitter commands a writer prints.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import pytest

from microcosm.data import build_folder_manifests
from microcosm.data.contract import ReleaseContractError

BUILD = "populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z"
RELEASE = "microcosm-us-2024-national"


def tagged_release_manifest() -> dict:
    """A release manifest in the tagged shape a US writer builds."""
    return {
        "schema_version": 1,
        "data_package": {"name": "microcosm-data", "version": "0.1.0"},
        "default_datasets": {"national": "microcosm_us_2024"},
        "build": {
            "build_id": BUILD,
            "built_at": "2026-10-02T04:59:00+00:00",
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.32.5",
            },
            "built_with_model_package": {"name": "policyengine-us", "version": "2.2.1"},
            "gate_evidence": {"stored_inputs": "bound"},
        },
        "compatible_model_packages": [
            {"name": "policyengine-us", "specifier": "==2.2.1"}
        ],
        "artifacts": {
            "microcosm_us_2024": {
                "kind": "microdata",
                "path": "microcosm_us_2024.h5",
                "repo_id": "policyengine/populace-us",
                "revision": BUILD,
                "sha256": "a" * 64,
            },
            "calibration_diagnostics": {
                "kind": "diagnostics",
                "path": "calibration_diagnostics.json",
                "repo_id": "policyengine/populace-us",
                "revision": BUILD,
                "sha256": "b" * 64,
            },
        },
    }


def write_tagged_release(root: Path, *, ledger: bool = False) -> Path:
    """A writer's output directory, named after the release, in the tagged
    shape; with ``ledger`` it also has a checksum ledger for its files."""
    directory = root / "releases" / RELEASE
    directory.mkdir(parents=True)
    (directory / "release_manifest.json").write_text(
        json.dumps(tagged_release_manifest())
    )
    (directory / "build_manifest.json").write_text(
        json.dumps({"build_id": BUILD, "code": {"git_commit": "c" * 40}})
    )
    (directory / "calibration_diagnostics.json").write_text("{}")
    if ledger:
        lines = [
            f"{hashlib.sha256((directory / name).read_bytes()).hexdigest()}  {name}"
            for name in (
                "build_manifest.json",
                "calibration_diagnostics.json",
                "release_manifest.json",
            )
        ]
        (directory / "sha256sums.txt").write_text("\n".join(lines) + "\n")
    return directory


def test_the_release_manifest_names_the_release_and_the_build() -> None:
    rewritten = build_folder_manifests.build_folder_release_manifest(
        tagged_release_manifest(), release=RELEASE, build=BUILD
    )

    assert (rewritten["release"], rewritten["build"]) == (RELEASE, BUILD)
    # The build object's fields move to the top level; its ID is the build.
    assert rewritten["built_at"] == "2026-10-02T04:59:00+00:00"
    assert rewritten["built_with_model_package"]["name"] == "policyengine-us"
    assert rewritten["gate_evidence"] == {"stored_inputs": "bound"}
    assert "build_id" not in rewritten
    assert rewritten["artifacts"]["microcosm_us_2024"] == {
        "kind": "microdata",
        "path": "microcosm_us_2024.h5",
        "repo_id": "policyengine/populace-us",
        "sha256": "a" * 64,
        "role": "dataset",
    }
    assert rewritten["artifacts"]["calibration_diagnostics"]["role"] == (
        "calibration_diagnostics"
    )


def test_the_build_manifest_names_the_release_and_the_build() -> None:
    rewritten = build_folder_manifests.build_folder_build_manifest(
        {"build_id": BUILD, "code": {"git_commit": "c" * 40}},
        release=RELEASE,
        build=BUILD,
    )

    assert rewritten == {
        "release": RELEASE,
        "build": BUILD,
        "code": {"git_commit": "c" * 40},
    }


def test_a_build_field_that_disagrees_with_the_top_level_is_refused() -> None:
    manifest = tagged_release_manifest()
    manifest["built_at"] = "2026-10-03T00:00:00+00:00"

    with pytest.raises(ReleaseContractError, match="built_at"):
        build_folder_manifests.build_folder_release_manifest(
            manifest, release=RELEASE, build=BUILD
        )


def test_a_manifest_already_in_the_build_folder_layout_is_refused() -> None:
    manifest = build_folder_manifests.build_folder_release_manifest(
        tagged_release_manifest(), release=RELEASE, build=BUILD
    )

    with pytest.raises(ReleaseContractError, match="already"):
        build_folder_manifests.build_folder_release_manifest(
            manifest, release=RELEASE, build=BUILD
        )


def test_rewriting_a_directory_refreshes_its_checksum_ledger(tmp_path) -> None:
    directory = write_tagged_release(tmp_path, ledger=True)

    build_folder_manifests.rewrite_release_dir(directory, build=BUILD)

    manifest = json.loads((directory / "release_manifest.json").read_text())
    build_manifest = json.loads((directory / "build_manifest.json").read_text())
    assert (manifest["release"], manifest["build"]) == (RELEASE, BUILD)
    assert (build_manifest["release"], build_manifest["build"]) == (RELEASE, BUILD)
    recorded = {
        entry.split()[1]: entry.split()[0]
        for entry in (directory / "sha256sums.txt").read_text().splitlines()
    }
    for name in (
        "build_manifest.json",
        "calibration_diagnostics.json",
        "release_manifest.json",
    ):
        assert (
            recorded[name]
            == hashlib.sha256((directory / name).read_bytes()).hexdigest()
        )


def test_rewriting_a_directory_named_for_no_release_is_refused(tmp_path) -> None:
    directory = write_tagged_release(tmp_path)
    renamed = directory.with_name(BUILD)
    directory.rename(renamed)

    with pytest.raises(
        ReleaseContractError, match="does not publish into build folders"
    ):
        build_folder_manifests.rewrite_release_dir(renamed, build=BUILD)


def test_the_publication_commands_publish_then_make_current(tmp_path) -> None:
    directory = write_tagged_release(tmp_path)
    artifact_root = tmp_path / "artifacts"

    commands = build_folder_manifests.publication_commands(
        directory, repo_id="policyengine/populace-us", artifact_root=artifact_root
    )

    assert shlex.split(commands["publish_command"]) == [
        "uv",
        "run",
        "microcosm-publish-release-build",
        str(directory),
        "--repo-id",
        "policyengine/populace-us",
        "--artifact-root",
        str(artifact_root),
    ]
    assert shlex.split(commands["promote_command"]) == [
        "uv",
        "run",
        "microcosm-publish-release-build",
        str(directory),
        "--repo-id",
        "policyengine/populace-us",
        "--promote",
    ]
