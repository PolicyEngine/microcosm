"""The build-folder switch on a repository whose builds have no tags.

BE's private release repository keeps each build in an untagged folder,
``releases/<build>/`` on ``main``, holding only its two manifests and its
calibration diagnostics; its release manifests declare files of the
builder's own output directory that the folder doesn't hold, and its
``latest.json`` names a folder with no revision. ``--include-untagged
FOLDER=RELEASE`` moves such a folder as build ``FOLDER`` of ``RELEASE``.
These tests run the tool against the in-memory repository in
``test_support.microcosm_build.build_folder_switch``, whose names are neutral
fixture names; none reaches the Hub.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from microcosm_emitter_interface.releases import ReleasesRecord, check_repository

from microcosm.build import build_folder_switch
from test_support.microcosm_build.build_folder_switch import (
    NEWER_UNTAGGED_FOLDER,
    NEWER_UNTAGGED_RELEASE,
    OLDER_UNTAGGED_FOLDER,
    OLDER_UNTAGGED_RELEASE,
    UNTAGGED_BUILT_AT,
    UNTAGGED_LAST_CHANGED_AT,
    UNTAGGED_REPOSITORY,
    FakeDatasetRepository,
    dumped,
    sha256,
    untagged_repository,
)

INCLUDES = (
    "--include-untagged",
    f"{OLDER_UNTAGGED_FOLDER}={OLDER_UNTAGGED_RELEASE}",
    "--include-untagged",
    f"{NEWER_UNTAGGED_FOLDER}={NEWER_UNTAGGED_RELEASE}",
)
#: ``latest.json`` names the newer build, so the older one's release is told
#: which build is current.
SWITCH_OPTIONS = (
    *INCLUDES,
    "--current",
    f"{OLDER_UNTAGGED_RELEASE}={OLDER_UNTAGGED_FOLDER}",
)
OLDER_SOURCE = f"releases/{OLDER_UNTAGGED_FOLDER}"
NEWER_SOURCE = f"releases/{NEWER_UNTAGGED_FOLDER}"
OLDER_FOLDER = f"releases/{OLDER_UNTAGGED_RELEASE}/{OLDER_UNTAGGED_FOLDER}"
NEWER_FOLDER = f"releases/{NEWER_UNTAGGED_RELEASE}/{NEWER_UNTAGGED_FOLDER}"
#: When a test changes the repository after the copy step.
LATER = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
MANIFEST_NAMES = {"release_manifest.json", "build_manifest.json"}


@pytest.fixture(autouse=True)
def no_hub_access(monkeypatch):
    """Every test runs offline and against the in-memory repository only."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def no_hub_client():
        pytest.fail("the tool built a Hub client")

    monkeypatch.setattr(build_folder_switch, "_default_api", no_hub_client)


@pytest.fixture
def hub(tmp_path) -> FakeDatasetRepository:
    return untagged_repository(tmp_path / "hub-cache")


def run(
    hub: FakeDatasetRepository, capsys, step: str, *options: str
) -> tuple[int, dict, str]:
    """Run one step on the in-memory repository: exit code, report and stderr.

    An option the parser refuses ends the run with its exit code, as on the
    command line.
    """
    try:
        exit_code = build_folder_switch.main(
            [step, UNTAGGED_REPOSITORY, *options], api=hub
        )
    except SystemExit as refusal:
        exit_code = int(refusal.code or 0)
    captured = capsys.readouterr()
    report = json.loads(captured.out) if captured.out.strip() else {}
    return exit_code, report, captured.err


def commit_made_at(hub: FakeDatasetRepository, moment: datetime) -> str:
    """The commit the fixture made at ``moment``."""
    return next(commit for commit, date in hub.dates.items() if date == moment)


def last_change(hub: FakeDatasetRepository, folder: str) -> str:
    """The commit that last changed an untagged folder on ``main``."""
    return commit_made_at(hub, UNTAGGED_LAST_CHANGED_AT[folder])


def published_at(folder: str) -> str:
    return UNTAGGED_LAST_CHANGED_AT[folder].isoformat()


def main_document(hub: FakeDatasetRepository, path: str) -> dict:
    """A JSON file on ``main``, or an empty object when it is not there."""
    return json.loads(hub.files_at().get(path, b"{}"))


def builds_by_source(report: dict) -> dict[str, dict]:
    return {build.get("source_folder"): build for build in report.get("builds", [])}


def test_copy_without_write_lists_each_untagged_folder_and_changes_nothing(hub, capsys):
    head = hub.main

    exit_code, report, error_output = run(hub, capsys, "copy", *INCLUDES)

    assert exit_code == 0, error_output
    assert [
        (
            build["release"],
            build["build"],
            build.get("tag"),
            build.get("source_folder"),
            build["commit"],
            build["published_at"],
            build["status"],
        )
        for build in report.get("builds", [])
    ] == [
        (
            NEWER_UNTAGGED_RELEASE,
            NEWER_UNTAGGED_FOLDER,
            None,
            NEWER_SOURCE,
            last_change(hub, NEWER_UNTAGGED_FOLDER),
            published_at(NEWER_UNTAGGED_FOLDER),
            "to_copy",
        ),
        (
            OLDER_UNTAGGED_RELEASE,
            OLDER_UNTAGGED_FOLDER,
            None,
            OLDER_SOURCE,
            last_change(hub, OLDER_UNTAGGED_FOLDER),
            published_at(OLDER_UNTAGGED_FOLDER),
            "to_copy",
        ),
    ]
    older = builds_by_source(report).get(OLDER_SOURCE, {})
    diagnostics = hub.files_at()[f"{OLDER_SOURCE}/calibration_diagnostics.json"]
    # Only the folder's own file is copied; the declared files it lacks are not.
    assert older.get("copied") == [
        {
            "name": "calibration_diagnostics.json",
            "from": f"{OLDER_SOURCE}/calibration_diagnostics.json",
            "sha256": sha256(diagnostics),
        }
    ]
    assert report.get("not_moved") == []
    assert hub.made == []
    assert hub.main == head


def test_copy_writes_each_untagged_build_copying_the_files_its_folder_holds(
    hub, capsys
):
    before = dict(hub.files_at())

    exit_code, report, error_output = run(hub, capsys, "copy", "--write", *INCLUDES)

    assert exit_code == 0, error_output
    assert [
        (commit.get("source_folder"), commit.get("folder"))
        for commit in report.get("commits", [])
    ] == [(NEWER_SOURCE, NEWER_FOLDER), (OLDER_SOURCE, OLDER_FOLDER)]
    older = next(
        (commit for commit in hub.made if OLDER_UNTAGGED_FOLDER in commit["message"]),
        {"operations": []},
    )
    # The manifests are written new; the diagnostics are copied from the
    # commit that last changed the folder, byte for byte.
    assert sorted(older["operations"], key=lambda operation: operation[1]) == [
        ("add", f"{OLDER_FOLDER}/build_manifest.json", None, None),
        (
            "copy",
            f"{OLDER_FOLDER}/calibration_diagnostics.json",
            f"{OLDER_SOURCE}/calibration_diagnostics.json",
            last_change(hub, OLDER_UNTAGGED_FOLDER),
        ),
        ("add", f"{OLDER_FOLDER}/release_manifest.json", None, None),
    ]
    files = hub.files_at()
    for source, folder in ((OLDER_SOURCE, OLDER_FOLDER), (NEWER_SOURCE, NEWER_FOLDER)):
        assert (
            files.get(f"{folder}/calibration_diagnostics.json")
            == (before[f"{source}/calibration_diagnostics.json"])
        )
    # The old folders and latest.json stay until the switch and the deletion.
    assert {path: files.get(path) for path in before} == before


def test_copied_untagged_manifests_name_the_release_the_build_and_the_origin(
    hub, capsys
):
    original_bytes = hub.files_at()[f"{OLDER_SOURCE}/release_manifest.json"]
    original = json.loads(original_bytes)
    original_build_manifest = main_document(hub, f"{OLDER_SOURCE}/build_manifest.json")

    exit_code, _, error_output = run(hub, capsys, "copy", "--write", *INCLUDES)

    assert exit_code == 0, error_output
    manifest = main_document(hub, f"{OLDER_FOLDER}/release_manifest.json")
    assert (manifest.get("release"), manifest.get("build")) == (
        OLDER_UNTAGGED_RELEASE,
        OLDER_UNTAGGED_FOLDER,
    )
    assert manifest.get(build_folder_switch.MIGRATED_FROM_FIELD) == {
        "folder": OLDER_SOURCE,
        "commit": last_change(hub, OLDER_UNTAGGED_FOLDER),
        "release_manifest_sha256": sha256(original_bytes),
    }
    # The old build object's details are at the top level, without build_id.
    assert "build_id" not in manifest
    assert manifest.get("built_at") == UNTAGGED_BUILT_AT[OLDER_UNTAGGED_FOLDER]
    assert manifest.get("chronicle") == original["build"]["chronicle"]
    for field in ("description", "data_package", "default_datasets"):
        assert manifest.get(field) == original[field]
    assert manifest.get("external_validations") == original["external_validations"]
    build_manifest = main_document(hub, f"{OLDER_FOLDER}/build_manifest.json")
    assert build_manifest == {
        "release": OLDER_UNTAGGED_RELEASE,
        "build": OLDER_UNTAGGED_FOLDER,
        **{
            key: value
            for key, value in original_build_manifest.items()
            if key != "build_id"
        },
    }


def test_an_untagged_build_s_artifacts_are_the_files_its_folder_holds(hub, capsys):
    """The rewritten manifest lists exactly the files the folder holds, besides
    its two manifests, as the collector's daily check of build folders asks;
    the builder's own declarations are kept apart, unchanged."""
    originals = {
        folder: json.loads(hub.files_at()[f"releases/{folder}/release_manifest.json"])
        for folder in (OLDER_UNTAGGED_FOLDER, NEWER_UNTAGGED_FOLDER)
    }

    exit_code, _, error_output = run(hub, capsys, "copy", "--write", *INCLUDES)

    assert exit_code == 0, error_output
    files = hub.files_at()
    for folder_name, folder in (
        (OLDER_UNTAGGED_FOLDER, OLDER_FOLDER),
        (NEWER_UNTAGGED_FOLDER, NEWER_FOLDER),
    ):
        manifest = main_document(hub, f"{folder}/release_manifest.json")
        diagnostics = files.get(f"{folder}/calibration_diagnostics.json", b"")
        assert manifest.get("artifacts") == {
            "calibration_diagnostics": {
                "path": "calibration_diagnostics.json",
                "repo_id": UNTAGGED_REPOSITORY,
                "role": "calibration_diagnostics",
                "sha256": sha256(diagnostics),
            }
        }
        held = {
            path.removeprefix(f"{folder}/"): sha256(content)
            for path, content in files.items()
            if path.startswith(f"{folder}/")
            and path.removeprefix(f"{folder}/") not in MANIFEST_NAMES
        }
        listed = {
            entry.get("path"): entry.get("sha256")
            for entry in manifest.get("artifacts", {}).values()
        }
        assert listed == held
        assert (
            manifest.get("unpublished_artifacts")
            == (originals[folder_name]["artifacts"])
        )


def test_a_copied_untagged_build_counts_as_copied(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDES)

    exit_code, report, error_output = run(hub, capsys, "copy", *INCLUDES)

    assert exit_code == 0, error_output
    assert {
        source: build.get("status")
        for source, build in builds_by_source(report).items()
    } == {OLDER_SOURCE: "already_copied", NEWER_SOURCE: "already_copied"}


def test_switch_makes_latest_json_s_untagged_build_current_and_its_release_default(
    hub, capsys
):
    run(hub, capsys, "copy", "--write", *INCLUDES)

    exit_code, report, error_output = run(
        hub, capsys, "switch", "--write", *SWITCH_OPTIONS
    )

    assert exit_code == 0, error_output
    assert report.get("pointer_files", {}).get("latest.json", {}).get("current") == (
        NEWER_UNTAGGED_FOLDER
    )
    assert main_document(hub, "releases.json") == {
        "schema_version": 1,
        "default": NEWER_UNTAGGED_RELEASE,
        "releases": {
            NEWER_UNTAGGED_RELEASE: {
                "role": "national",
                "current": NEWER_UNTAGGED_FOLDER,
                "builds": {
                    NEWER_UNTAGGED_FOLDER: {
                        "published_at": published_at(NEWER_UNTAGGED_FOLDER),
                        "withdrawn": False,
                    }
                },
            },
            OLDER_UNTAGGED_RELEASE: {
                "role": "national",
                "current": OLDER_UNTAGGED_FOLDER,
                "builds": {
                    OLDER_UNTAGGED_FOLDER: {
                        "published_at": published_at(OLDER_UNTAGGED_FOLDER),
                        "withdrawn": False,
                    }
                },
            },
        },
    }
    files = hub.files_at()
    assert "latest.json" not in files
    record = ReleasesRecord.parse(files["releases.json"])
    assert check_repository(record, list(files), files.__getitem__) == []


def test_two_untagged_builds_can_move_into_one_release(hub, capsys):
    """If both builds go into one release, ``latest.json``'s build is current."""
    includes = (
        "--include-untagged",
        f"{OLDER_UNTAGGED_FOLDER}={NEWER_UNTAGGED_RELEASE}",
        "--include-untagged",
        f"{NEWER_UNTAGGED_FOLDER}={NEWER_UNTAGGED_RELEASE}",
    )
    copy_code, _, copy_error = run(hub, capsys, "copy", "--write", *includes)
    assert copy_code == 0, copy_error

    exit_code, _, error_output = run(hub, capsys, "switch", "--write", *includes)

    assert exit_code == 0, error_output
    record = main_document(hub, "releases.json")
    assert record.get("default") == NEWER_UNTAGGED_RELEASE
    release = record.get("releases", {}).get(NEWER_UNTAGGED_RELEASE, {})
    assert release.get("current") == NEWER_UNTAGGED_FOLDER
    assert sorted(release.get("builds", {})) == sorted(
        [OLDER_UNTAGGED_FOLDER, NEWER_UNTAGGED_FOLDER]
    )


def test_switch_refuses_an_untagged_folder_changed_since_its_copy(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDES)
    hub.seed(
        {f"{NEWER_SOURCE}/calibration_diagnostics.json": b'{"changed": true}'},
        date=LATER,
    )

    exit_code, report, error_output = run(
        hub, capsys, "switch", "--write", *SWITCH_OPTIONS
    )

    assert exit_code == 1
    assert "releases.json" not in hub.files_at()
    assert any(
        NEWER_FOLDER in problem and NEWER_SOURCE in problem
        for problem in report.get("problems", [error_output])
    ), error_output


def test_switch_refuses_a_pointer_naming_an_untagged_folder_that_does_not_move(
    hub, capsys
):
    older_only = (
        "--include-untagged",
        f"{OLDER_UNTAGGED_FOLDER}={OLDER_UNTAGGED_RELEASE}",
    )
    run(hub, capsys, "copy", "--write", *older_only)

    exit_code, report, _ = run(hub, capsys, "switch", *older_only)

    assert exit_code == 1
    assert any(
        "latest.json" in problem
        and f"--include-untagged {NEWER_UNTAGGED_FOLDER}=<release>" in problem
        for problem in report.get("problems", [])
    )


def _add_main_files(hub: FakeDatasetRepository, files: dict[str, bytes]) -> None:
    hub.seed(files, date=LATER)


def _change_newer_manifest(hub: FakeDatasetRepository, change: Callable) -> None:
    path = f"{NEWER_SOURCE}/release_manifest.json"
    manifest = json.loads(hub.files_at()[path])
    change(manifest)
    _add_main_files(hub, {path: dumped(manifest)})


def _tag_newer_folder(hub: FakeDatasetRepository) -> None:
    hub.tags[NEWER_UNTAGGED_FOLDER] = hub.main


@pytest.mark.parametrize(
    ("change", "named"),
    [
        pytest.param(_tag_newer_folder, "--include", id="a tag of the same name"),
        pytest.param(
            lambda hub: _change_newer_manifest(
                hub, lambda manifest: manifest["build"].update(build_id="another")
            ),
            "build_id",
            id="a manifest naming another build",
        ),
        pytest.param(
            lambda hub: _change_newer_manifest(
                hub,
                lambda manifest: manifest["build"].update(
                    cut_tag=f"{NEWER_UNTAGGED_FOLDER}-tag"
                ),
            ),
            "cut_tag",
            id="a manifest naming a tag",
        ),
        pytest.param(
            lambda hub: _change_newer_manifest(
                hub, lambda manifest: manifest.update(unpublished_artifacts={})
            ),
            "unpublished_artifacts",
            id="a manifest that already has unpublished_artifacts",
        ),
        pytest.param(
            lambda hub: _add_main_files(hub, {f"{NEWER_SOURCE}/sha256sums.txt": b""}),
            "sha256sums.txt",
            id="a checksum ledger",
        ),
        pytest.param(
            lambda hub: _add_main_files(
                hub, {f"{NEWER_SOURCE}/nested/extra.json": b"{}"}
            ),
            "nested/extra.json",
            id="a file in a subfolder",
        ),
        pytest.param(
            lambda hub: _add_main_files(
                hub,
                {
                    f"{NEWER_SOURCE}/calibration_diagnostics.csv": b"",
                },
            ),
            "calibration_diagnostics",
            id="two files with one artifact name",
        ),
    ],
)
def test_copy_refuses_an_untagged_folder_it_cannot_move(hub, capsys, change, named):
    change(hub)

    exit_code, report, error_output = run(hub, capsys, "copy", "--write", *INCLUDES)

    assert exit_code == 1
    problems = [
        problem
        for problem in report.get("problems", [error_output])
        if NEWER_UNTAGGED_FOLDER in problem
    ]
    assert problems, error_output
    assert any(named in problem for problem in problems), problems
    assert hub.made == []


def test_copy_refuses_an_untagged_folder_that_is_not_on_main(hub, capsys):
    exit_code, report, error_output = run(
        hub, capsys, "copy", "--include-untagged", "not-a-folder=example-release"
    )

    assert exit_code == 1
    assert any(
        "releases/not-a-folder/" in problem for problem in report.get("problems", [])
    ), error_output


def test_include_untagged_takes_folder_equals_release(hub, capsys):
    exit_code, _, error_output = run(
        hub, capsys, "copy", "--include-untagged", NEWER_UNTAGGED_FOLDER
    )

    assert exit_code == 1
    assert "--include-untagged takes NAME=VALUE" in error_output


def test_delete_old_layout_takes_no_include_untagged_option(hub, capsys):
    exit_code, _, error_output = run(hub, capsys, "delete-old-layout", *INCLUDES[:2])

    assert exit_code == 1
    assert "--include-untagged" in error_output


def test_delete_old_layout_removes_the_untagged_folders_and_reports_them(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDES)
    switch_code, _, switch_error = run(
        hub, capsys, "switch", "--write", *SWITCH_OPTIONS
    )
    assert switch_code == 0, switch_error
    before = dict(hub.files_at())

    exit_code, report, error_output = run(hub, capsys, "delete-old-layout", "--write")

    assert exit_code == 0, error_output
    assert set(hub.files_at()) == {".gitattributes", "README.md", "releases.json"} | {
        path
        for path in before
        if path.startswith((f"{OLDER_FOLDER}/", f"{NEWER_FOLDER}/"))
    }
    # No tag keeps them; they stay in main's history at the head it read.
    assert report.get("untagged_folders") == [NEWER_SOURCE, OLDER_SOURCE]
