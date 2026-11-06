"""The one-time switch from tags and pointer files to build folders (P3).

The tool copies each tagged build into ``releases/<release>/<build>/`` and
then, in one commit, writes ``releases.json`` and deletes the ``latest*.json``
pointer files. These tests run it against the in-memory repository in
``test_support.microcosm_build.build_folder_switch``, laid out as the UK
private repository is before its switch; none reaches the Hub.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from microcosm_emitter_interface.releases import ReleasesRecord, check_repository

from microcosm.build import build_folder_switch
from microcosm.data import loader
from microcosm.data.registry import REGISTRY
from microcosm.data.releases_record import parse_releases_record
from test_support.microcosm_build.build_folder_switch import (
    DENSE,
    DENSE_BUILD,
    DENSE_TAG,
    JUNE_2023_BUILD_METADATA,
    JUNE_2023_RELEASE_METADATA,
    JUNE_2023_TAG,
    NATIONAL,
    NEWER_NATIONAL_BUILD,
    NEWER_NATIONAL_TAG,
    OLDER_NATIONAL_BUILD,
    OLDER_NATIONAL_TAG,
    UK_REPOSITORY,
    FakeDatasetRepository,
    dumped,
    sha256,
    uk_repository,
)

#: The stable release the June 2023 build moves into, when it moves.
JUNE_2023_RELEASE = "populace-uk-2023"
INCLUDE_JUNE_2023 = ("--include", f"{JUNE_2023_TAG}={JUNE_2023_RELEASE}")
NEWER_NATIONAL_FOLDER = f"releases/{NATIONAL}/{NEWER_NATIONAL_BUILD}"
DENSE_FOLDER = f"releases/{DENSE}/{DENSE_BUILD}"
JUNE_2023_FOLDER = f"releases/{JUNE_2023_RELEASE}/{JUNE_2023_TAG}"
JUNE_2023_MANIFEST = f"releases/{JUNE_2023_TAG}/release_manifest.json"
#: When a test changes the repository after the copy step.
LATER = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
#: The branch the tagged publisher (``microcosm.data.release``) keeps while it
#: publishes a dense build: from before its upload until after it tags it.
DENSE_STAGING_BRANCH = f"release-staging/{DENSE}"


@pytest.fixture(autouse=True)
def no_hub_access(monkeypatch):
    """Every test runs offline and against the in-memory repository only."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def no_hub_client():
        pytest.fail("the tool built a Hub client")

    monkeypatch.setattr(build_folder_switch, "_default_api", no_hub_client)


@pytest.fixture
def hub(tmp_path) -> FakeDatasetRepository:
    return uk_repository(tmp_path / "hub-cache")


def run(
    hub: FakeDatasetRepository, capsys, step: str, *options: str
) -> tuple[int, dict, str]:
    """Run one step on the in-memory repository: exit code, report and stderr."""
    exit_code = build_folder_switch.main([step, UK_REPOSITORY, *options], api=hub)
    captured = capsys.readouterr()
    report = json.loads(captured.out) if captured.out.strip() else {}
    return exit_code, report, captured.err


def copied_file(hub: FakeDatasetRepository, path: str) -> bytes:
    """A file on ``main``, or no bytes when it is not there."""
    return hub.files_at().get(path, b"")


def test_copy_without_write_lists_each_tagged_build_and_changes_nothing(hub, capsys):
    head = hub.main

    exit_code, report, error_output = run(hub, capsys, "copy")

    assert exit_code == 0
    assert [
        (build["release"], build["build"], build["tag"], build["status"])
        for build in report.get("builds", [])
    ] == [
        (DENSE, DENSE_BUILD, DENSE_TAG, "to_copy"),
        (NATIONAL, OLDER_NATIONAL_BUILD, OLDER_NATIONAL_TAG, "to_copy"),
        (NATIONAL, NEWER_NATIONAL_BUILD, NEWER_NATIONAL_TAG, "to_copy"),
    ]
    # The June 2023 release names one release per build, so it moves only
    # when --include says into which release.
    assert report.get("not_moved") == [JUNE_2023_TAG]
    newer = next(
        (
            build
            for build in report.get("builds", [])
            if build["build"] == NEWER_NATIONAL_BUILD
        ),
        {},
    )
    # An earlier build's file that this build's manifest does not name stays.
    assert newer.get("left_behind") == ["score_vs_enhanced_frs.json"]
    assert hub.made == []
    assert hub.main == head
    assert "Nothing was written" in error_output


def test_copy_writes_each_build_into_its_folder_copying_its_tagged_files(hub, capsys):
    exit_code, report, _ = run(hub, capsys, "copy", "--write")

    assert exit_code == 0
    assert [commit["tag"] for commit in report.get("commits", [])] == [
        DENSE_TAG,
        OLDER_NATIONAL_TAG,
        NEWER_NATIONAL_TAG,
    ]
    assert len(hub.made) == 3
    tagged_commit = hub.tags[NEWER_NATIONAL_TAG]
    newer = next(
        (commit for commit in hub.made if NEWER_NATIONAL_BUILD in commit["message"]),
        {"operations": []},
    )
    # The manifests are written new; every other file is copied on the Hub's
    # servers from the build's tagged commit, so nothing large is uploaded.
    assert sorted(newer["operations"], key=lambda operation: operation[1]) == [
        ("add", f"{NEWER_NATIONAL_FOLDER}/build_manifest.json", None, None),
        (
            "copy",
            f"{NEWER_NATIONAL_FOLDER}/calibration_diagnostics.json",
            f"releases/{NATIONAL}/calibration_diagnostics.json",
            tagged_commit,
        ),
        (
            "copy",
            f"{NEWER_NATIONAL_FOLDER}/microcosm_uk_2024_25.h5",
            "microcosm_uk_2024_25.h5",
            tagged_commit,
        ),
        (
            "copy",
            f"{NEWER_NATIONAL_FOLDER}/microcosm_uk_2024_25_calibration.npz",
            "microcosm_uk_2024_25_calibration.npz",
            tagged_commit,
        ),
        (
            "copy",
            f"{NEWER_NATIONAL_FOLDER}/release_certification.json",
            f"releases/{NATIONAL}/release_certification.json",
            tagged_commit,
        ),
        ("add", f"{NEWER_NATIONAL_FOLDER}/release_manifest.json", None, None),
    ]
    assert (
        copied_file(hub, f"{NEWER_NATIONAL_FOLDER}/microcosm_uk_2024_25.h5")
        == b"newer national dataset"
    )
    # Until the switch, the repository still publishes through its pointers.
    files = hub.files_at()
    assert "releases.json" not in files
    assert {"latest.json", "latest-national.json"} <= set(files)


def test_copied_manifests_name_the_release_and_the_build(hub, capsys):
    run(hub, capsys, "copy", "--write")
    tagged_bytes = hub.files_at(NEWER_NATIONAL_TAG)[
        f"releases/{NATIONAL}/release_manifest.json"
    ]
    tagged = json.loads(tagged_bytes)

    manifest = json.loads(
        copied_file(hub, f"{NEWER_NATIONAL_FOLDER}/release_manifest.json") or b"{}"
    )
    build_manifest = json.loads(
        copied_file(hub, f"{NEWER_NATIONAL_FOLDER}/build_manifest.json") or b"{}"
    )

    assert (manifest.get("release"), manifest.get("build")) == (
        NATIONAL,
        NEWER_NATIONAL_BUILD,
    )
    for detail in ("built_at", "built_with_core_package", "built_with_model_package"):
        assert manifest.get(detail) == tagged["build"][detail]
    artifacts = manifest.get("artifacts", {})
    assert {key: entry.get("role") for key, entry in artifacts.items()} == {
        "microcosm_uk_2024_25": "dataset",
        "microcosm_uk_2024_25_calibration": "calibration_weights",
        "calibration_diagnostics": "calibration_diagnostics",
        "release_certification": "release_certification",
    }
    assert not any("revision" in entry for entry in artifacts.values())
    assert {key: entry.get("sha256") for key, entry in artifacts.items()} == {
        key: entry["sha256"] for key, entry in tagged["artifacts"].items()
    }
    assert manifest.get("migrated_from") == {
        "tag": NEWER_NATIONAL_TAG,
        "commit": hub.tags[NEWER_NATIONAL_TAG],
        "release_manifest_sha256": sha256(tagged_bytes),
    }
    assert (build_manifest.get("release"), build_manifest.get("build")) == (
        NATIONAL,
        NEWER_NATIONAL_BUILD,
    )
    assert not {"build_id", "attempt_id", "cut_tag"} & set(build_manifest)


def test_the_loader_certifies_a_copied_build_from_its_folder(hub, capsys):
    run(hub, capsys, "copy", "--write")
    manifest = json.loads(
        copied_file(hub, f"{NEWER_NATIONAL_FOLDER}/release_manifest.json") or b"{}"
    )
    assert manifest.get("build") == NEWER_NATIONAL_BUILD

    certified = loader._certified_release_from_manifest(
        REGISTRY[("uk", 2025, "compact")],
        NATIONAL,
        manifest,
        expected_revision=None,
        build=NEWER_NATIONAL_BUILD,
        release_role="national",
    )

    assert certified.artifact_path == f"{NEWER_NATIONAL_FOLDER}/microcosm_uk_2024_25.h5"
    assert certified.artifact_sha256 == sha256(b"newer national dataset")


def test_copy_rewrites_the_checksum_ledger_for_the_rewritten_manifests(hub, capsys):
    run(hub, capsys, "copy", "--write")

    ledger = {
        name: digest
        for digest, name in (
            line.split("  ", 1)
            for line in copied_file(hub, f"{DENSE_FOLDER}/sha256sums.txt")
            .decode()
            .splitlines()
        )
    }

    assert ledger == {
        name: sha256(copied_file(hub, f"{DENSE_FOLDER}/{name}"))
        for name in (
            "build_manifest.json",
            "calibration_diagnostics.json",
            "gate_summary.json",
            "microcosm_uk_2024_25_dense.h5",
            "release_manifest.json",
        )
    }


def test_copy_refuses_a_build_whose_file_does_not_match_its_manifest(hub, capsys):
    hub.commits[hub.tags[DENSE_TAG]]["microcosm_uk_2024_25_dense.h5"] = b"changed"

    exit_code, _, error_output = run(hub, capsys, "copy", "--write")

    assert exit_code == 1
    assert hub.made == []
    assert "does not match the SHA-256 its manifest records" in error_output


def test_copy_again_skips_the_builds_already_in_their_folders(hub, capsys):
    run(hub, capsys, "copy", "--write")
    assert len(hub.made) == 3

    exit_code, report, _ = run(hub, capsys, "copy", "--write")

    assert exit_code == 0
    assert len(hub.made) == 3
    assert [build["status"] for build in report.get("builds", [])] == [
        "already_copied"
    ] * 3


def test_a_failed_copy_commit_names_the_builds_copied_before_it(
    hub, capsys, monkeypatch
):
    create_commit = hub.create_commit
    attempts = []

    def second_commit_conflicts(*args, **kwargs):
        attempts.append(kwargs["parent_commit"])
        if len(attempts) == 2:
            raise RuntimeError("412: another commit landed first")
        return create_commit(*args, **kwargs)

    monkeypatch.setattr(hub, "create_commit", second_commit_conflicts)

    exit_code, _, error_output = run(hub, capsys, "copy", "--write")

    assert exit_code == 1
    assert len(hub.made) == 1
    assert f"copied before it in this run: ['{DENSE_TAG}']" in error_output


def test_a_tag_whose_manifest_names_another_tag_moves_only_when_excluded(hub, capsys):
    stray_tag = f"{NATIONAL}-20261106T000000Z-0badcafe"
    hub.tags[stray_tag] = hub.tags[NEWER_NATIONAL_TAG]

    exit_code, report, _ = run(hub, capsys, "copy")
    excluded_exit_code, excluded, _ = run(hub, capsys, "copy", "--exclude", stray_tag)

    assert exit_code == 1
    assert [problem for problem in report.get("problems", []) if stray_tag in problem]
    assert excluded_exit_code == 0
    assert {"tag": stray_tag, "reason": "--exclude"} in excluded.get("skipped", [])


def test_switch_without_write_prints_the_record_and_changes_nothing(hub, capsys):
    run(hub, capsys, "copy", "--write")
    made = list(hub.made)

    exit_code, report, error_output = run(
        hub, capsys, "switch", "--retire-pointer", "latest.json"
    )

    assert exit_code == 0
    assert report.get("delete") == ["latest-national.json", "latest.json"]
    record = report.get("releases_json", {})
    assert record.get("default") == NATIONAL
    assert record.get("releases", {}).get(NATIONAL, {}).get("current") == (
        OLDER_NATIONAL_BUILD
    )
    assert hub.made == made
    assert "Nothing was written" in error_output


def test_switch_writes_releases_json_and_deletes_the_pointer_files_in_one_commit(
    hub, capsys
):
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)
    head = hub.main

    exit_code, _, _ = run(hub, capsys, "switch", "--write", *INCLUDE_JUNE_2023)

    assert exit_code == 0
    switch_commit = hub.made[-1] if len(hub.made) == 5 else {"operations": []}
    assert switch_commit.get("parent") == head
    assert sorted(operation[:2] for operation in switch_commit["operations"]) == [
        ("add", "releases.json"),
        ("delete", "latest-national.json"),
        ("delete", "latest.json"),
    ]
    # Each build keeps the date of its tagged commit, and each pointer file's
    # build is its release's current build: latest-national.json names the
    # older national build, though a newer one was published since.
    assert json.loads(copied_file(hub, "releases.json") or b"{}") == {
        "schema_version": 1,
        "default": NATIONAL,
        "releases": {
            DENSE: {
                "role": "local_area",
                "current": None,
                "builds": {
                    DENSE_BUILD: {
                        "published_at": "2026-11-04T13:06:25+00:00",
                        "withdrawn": False,
                    }
                },
            },
            NATIONAL: {
                "role": "national",
                "current": OLDER_NATIONAL_BUILD,
                "builds": {
                    OLDER_NATIONAL_BUILD: {
                        "published_at": "2026-10-19T10:21:04+00:00",
                        "withdrawn": False,
                    },
                    NEWER_NATIONAL_BUILD: {
                        "published_at": "2026-11-05T23:30:42+00:00",
                        "withdrawn": False,
                    },
                },
            },
            JUNE_2023_RELEASE: {
                "role": "national",
                "current": JUNE_2023_TAG,
                "builds": {
                    JUNE_2023_TAG: {
                        "published_at": "2026-06-19T02:38:38+00:00",
                        "withdrawn": False,
                    }
                },
            },
        },
    }
    assert not [
        path for path in hub.files_at() if path.rsplit("/", 1)[-1].startswith("latest")
    ]


def test_the_switched_repository_passes_the_interface_checks_and_the_reader(
    hub, capsys
):
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)
    run(hub, capsys, "switch", "--write", *INCLUDE_JUNE_2023)
    content = copied_file(hub, "releases.json")
    assert content, "the switch wrote no releases.json"

    record = ReleasesRecord.parse(content)
    files = hub.files_at()

    assert check_repository(record, list(files), files.__getitem__) == []
    assert parse_releases_record(content).releases[NATIONAL].current == (
        OLDER_NATIONAL_BUILD
    )


def test_switch_refuses_a_pointer_file_whose_build_does_not_move(hub, capsys):
    run(hub, capsys, "copy", "--write")

    exit_code, _, error_output = run(hub, capsys, "switch", "--write")

    assert exit_code == 1
    assert f"latest.json names '{JUNE_2023_TAG}'" in error_output
    assert "--retire-pointer latest.json" in error_output
    assert "releases.json" not in hub.files_at()


def test_switch_can_retire_a_pointer_file_and_make_a_named_build_current(hub, capsys):
    run(hub, capsys, "copy", "--write")

    exit_code, _, _ = run(
        hub,
        capsys,
        "switch",
        "--write",
        "--retire-pointer",
        "latest.json",
        "--current",
        f"{DENSE}={DENSE_BUILD}",
    )

    assert exit_code == 0
    record = json.loads(copied_file(hub, "releases.json") or b"{}")
    assert sorted(record.get("releases", {})) == [DENSE, NATIONAL]
    assert record.get("releases", {}).get(DENSE, {}).get("current") == DENSE_BUILD
    assert record.get("default") == NATIONAL
    assert "latest.json" not in hub.files_at()


def test_switch_refuses_until_every_build_is_in_its_folder(hub, capsys):
    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", "--retire-pointer", "latest.json"
    )

    assert exit_code == 1
    assert hub.made == []
    assert "run the copy step first" in error_output


@pytest.mark.parametrize("step", ["copy", "switch"])
def test_each_step_refuses_a_repository_that_has_releases_json(hub, capsys, step):
    hub.seed({"releases.json": b"{}"}, date=datetime(2026, 11, 6, tzinfo=UTC))

    exit_code, _, error_output = run(
        hub, capsys, step, "--write", "--retire-pointer", "latest.json"
    )

    assert exit_code == 1
    assert hub.made == []
    assert "already has releases.json" in error_output


@pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_the_tool_refuses_to_run_offline(capsys, monkeypatch, variable):
    # The autouse fixture fails the test if the tool builds a Hub client.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv(variable, "1")

    exit_code = build_folder_switch.main(["copy", UK_REPOSITORY, "--write"])

    assert exit_code == 1
    assert variable in capsys.readouterr().err


def test_copy_keeps_both_metadata_objects_of_the_june_2023_build(hub, capsys):
    exit_code, _, error_output = run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)

    assert exit_code == 0, error_output
    manifest = json.loads(
        copied_file(hub, f"{JUNE_2023_FOLDER}/release_manifest.json") or b"{}"
    )
    # The build object's metadata and the release's describe the same build,
    # so they become one object; neither loses a field.
    assert manifest.get("metadata") == {
        **JUNE_2023_BUILD_METADATA,
        **JUNE_2023_RELEASE_METADATA,
    }
    assert (manifest.get("release"), manifest.get("build")) == (
        JUNE_2023_RELEASE,
        JUNE_2023_TAG,
    )


def test_copy_refuses_build_metadata_that_disagrees_with_the_release(hub, capsys):
    tagged = hub.commits[hub.tags[JUNE_2023_TAG]]
    manifest = json.loads(tagged[JUNE_2023_MANIFEST])
    manifest["build"]["metadata"]["region_datasets"] = {"another": "value"}
    tagged[JUNE_2023_MANIFEST] = dumped(manifest)

    exit_code, _, error_output = run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)

    assert exit_code == 1
    assert hub.made == []
    assert "differ on ['metadata.region_datasets']" in error_output


def test_switch_refuses_a_copied_build_whose_file_changed_since(hub, capsys):
    run(hub, capsys, "copy", "--write")
    changed = f"{NEWER_NATIONAL_FOLDER}/microcosm_uk_2024_25.h5"
    hub.seed({changed: b"another dataset"}, date=LATER)
    made = list(hub.made)

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", "--retire-pointer", "latest.json"
    )

    assert exit_code == 1
    assert hub.made == made
    assert f"{changed} does not match the SHA-256" in error_output
    assert "releases.json" not in hub.files_at()


def test_switch_refuses_a_copied_build_folder_holding_another_file(hub, capsys):
    run(hub, capsys, "copy", "--write")
    hub.seed({f"{DENSE_FOLDER}/notes.txt": b"added by hand"}, date=LATER)
    made = list(hub.made)

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", "--retire-pointer", "latest.json"
    )

    assert exit_code == 1
    assert hub.made == made
    assert f"{DENSE_FOLDER}/ also holds ['notes.txt']" in error_output


def test_switch_takes_a_pointer_naming_an_included_build_by_its_tag(hub, capsys):
    # A release-per-build pointer that names the build's tag as its revision
    # names the release by its old name, the tag.
    hub.seed(
        {
            "latest.json": dumped(
                {
                    "schema_version": 1,
                    "release_id": JUNE_2023_TAG,
                    "revision": JUNE_2023_TAG,
                }
            )
        },
        date=LATER,
    )
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", *INCLUDE_JUNE_2023
    )

    assert exit_code == 0, error_output
    record = json.loads(copied_file(hub, "releases.json") or b"{}")
    assert record.get("releases", {}).get(JUNE_2023_RELEASE, {}).get("current") == (
        JUNE_2023_TAG
    )


def test_an_annotated_tag_moves_from_the_commit_it_names(hub, capsys):
    commit = hub.tags[DENSE_TAG]
    assert hub.tag_objects[DENSE_TAG] != commit

    exit_code, report, error_output = run(hub, capsys, "copy", "--write")

    assert exit_code == 0, error_output
    dense = next(
        (build for build in report.get("builds", []) if build["tag"] == DENSE_TAG), {}
    )
    assert (dense.get("commit"), dense.get("published_at")) == (
        commit,
        "2026-11-04T13:06:25+00:00",
    )
    copy_commit = next(
        (made for made in hub.made if DENSE_BUILD in made["message"]),
        {"operations": []},
    )
    assert {
        operation[3]
        for operation in copy_commit["operations"]
        if operation[0] == "copy"
    } == {commit}
    manifest = json.loads(
        copied_file(hub, f"{DENSE_FOLDER}/release_manifest.json") or b"{}"
    )
    assert manifest.get("migrated_from", {}).get("commit") == commit


def test_copy_reports_a_release_name_the_interface_refuses_without_writing(hub, capsys):
    exit_code, report, _ = run(
        hub, capsys, "copy", "--include", f"{JUNE_2023_TAG}=populace uk 2023"
    )

    assert exit_code == 1
    assert any(JUNE_2023_TAG in problem for problem in report.get("problems", []))
    assert hub.made == []


def test_copy_refuses_a_release_role_conflict_before_any_commit(hub, capsys):
    # The June 2023 build is a national build; the dense release's builds are
    # local-area builds, so one release cannot hold both.
    exit_code, _, error_output = run(
        hub, capsys, "copy", "--write", "--include", f"{JUNE_2023_TAG}={DENSE}"
    )

    assert exit_code == 1
    assert "has role" in error_output
    assert hub.made == []


def test_switch_refuses_a_build_folder_that_it_does_not_list(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)

    # The June 2023 build was copied, but this switch neither moves it nor
    # keeps the pointer file that names it.
    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", "--retire-pointer", "latest.json"
    )

    assert exit_code == 1
    assert f"{JUNE_2023_FOLDER}/" in error_output
    assert "releases.json" not in hub.files_at()


def test_switch_refuses_when_a_tag_appears_while_it_reads(hub, capsys, monkeypatch):
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)
    list_repo_refs = hub.list_repo_refs
    listings = []

    def a_tag_lands_after_the_first_listing(*args, **kwargs):
        refs = list_repo_refs(*args, **kwargs)
        listings.append(refs)
        if len(listings) == 1:
            # A tag-only publication lands while the switch reads the builds:
            # it tags a commit that is not on main, so main does not move.
            hub.seed(
                {}, date=LATER, tag=f"{DENSE}-20261106T120000Z-0badf00d", on_main=False
            )
        return refs

    monkeypatch.setattr(hub, "list_repo_refs", a_tag_lands_after_the_first_listing)

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", *INCLUDE_JUNE_2023
    )

    assert exit_code == 1
    assert "tags changed" in error_output
    assert "releases.json" not in hub.files_at()


def test_switch_refuses_a_current_build_that_conflicts_with_a_pointer_file(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)

    exit_code, report, _ = run(
        hub,
        capsys,
        "switch",
        *INCLUDE_JUNE_2023,
        "--current",
        f"{NATIONAL}={NEWER_NATIONAL_BUILD}",
    )

    assert exit_code == 1
    assert any(
        "conflicts with the pointer files' current build" in problem
        for problem in report.get("problems", [])
    )


def test_switch_can_choose_the_default_release(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)

    exit_code, _, _ = run(
        hub,
        capsys,
        "switch",
        "--write",
        *INCLUDE_JUNE_2023,
        "--current",
        f"{DENSE}={DENSE_BUILD}",
        "--default",
        DENSE,
    )

    assert exit_code == 0
    record = json.loads(copied_file(hub, "releases.json") or b"{}")
    assert record.get("default") == DENSE


def test_switch_refuses_while_a_tagged_publication_is_in_progress(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)
    # A dense publication has made its staging branch and not yet its tag;
    # a switch now would leave its build in neither the record nor a folder.
    hub.branches[DENSE_STAGING_BRANCH] = hub.main

    exit_code, report, _ = run(hub, capsys, "switch", *INCLUDE_JUNE_2023)

    assert exit_code == 1
    assert any(
        DENSE_STAGING_BRANCH in problem for problem in report.get("problems", [])
    )

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", *INCLUDE_JUNE_2023
    )

    assert exit_code == 1
    assert DENSE_STAGING_BRANCH in error_output
    assert "releases.json" not in hub.files_at()


def test_switch_refuses_when_a_tagged_publication_starts_while_it_reads(
    hub, capsys, monkeypatch
):
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)
    list_repo_refs = hub.list_repo_refs
    listings = []

    def a_publication_starts_after_the_first_listing(*args, **kwargs):
        refs = list_repo_refs(*args, **kwargs)
        listings.append(refs)
        if len(listings) == 1:
            # The publisher checked for releases.json before the switch
            # commit and makes its staging branch while the switch reads.
            hub.branches[DENSE_STAGING_BRANCH] = hub.main
        return refs

    monkeypatch.setattr(
        hub, "list_repo_refs", a_publication_starts_after_the_first_listing
    )

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", *INCLUDE_JUNE_2023
    )

    assert exit_code == 1
    assert DENSE_STAGING_BRANCH in error_output
    assert "releases.json" not in hub.files_at()


@pytest.mark.parametrize("change", ["deleted", "moved", "annotated tag moved"])
def test_switch_refuses_when_a_tag_is_deleted_or_moved_while_it_reads(
    hub, capsys, monkeypatch, change
):
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)
    list_repo_refs = hub.list_repo_refs
    listings = []

    def a_tag_changes_before_the_second_listing(*args, **kwargs):
        if len(listings) == 1:
            if change == "deleted":
                del hub.tags[OLDER_NATIONAL_TAG]
            else:
                # The dense tag is annotated; the older national one is not.
                tag = (
                    DENSE_TAG if change == "annotated tag moved" else OLDER_NATIONAL_TAG
                )
                hub.move_tag(tag, hub.seed({}, date=LATER, on_main=False))
        refs = list_repo_refs(*args, **kwargs)
        listings.append(refs)
        return refs

    monkeypatch.setattr(hub, "list_repo_refs", a_tag_changes_before_the_second_listing)

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", *INCLUDE_JUNE_2023
    )

    assert exit_code == 1
    assert "tags changed" in error_output
    assert "releases.json" not in hub.files_at()


def test_switch_refuses_a_folder_copied_earlier_and_left_out_with_exclude(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", *INCLUDE_JUNE_2023, "--exclude", DENSE_TAG
    )

    assert exit_code == 1
    assert f"{DENSE_FOLDER}/ is a build folder that this switch would not list" in (
        error_output
    )
    assert "releases.json" not in hub.files_at()


def test_copy_reports_a_release_role_conflict_without_writing(hub, capsys):
    exit_code, report, _ = run(
        hub, capsys, "copy", "--include", f"{JUNE_2023_TAG}={DENSE}"
    )

    assert exit_code == 1
    assert any("has role" in problem for problem in report.get("problems", []))
    assert hub.made == []


@pytest.mark.parametrize(
    "path",
    [
        "releases/a-release/a-build/release_manifest.json",
        "releases/a-release/a-build/nested/file.json",
        "releases/a-release/release_manifest.json",
        "releases/a-release",
        "elsewhere/a-release/a-build/file.json",
        "latest.json",
        "releases.json",
    ],
)
def test_the_tool_finds_build_folders_by_the_interfaces_rule(path):
    """The switch refuses build folders it would not list by the same rule
    that the interface's ``check_commit`` uses, which is private to it."""
    from microcosm_emitter_interface import releases as interface_releases

    assert build_folder_switch._build_folder_of(
        path
    ) == interface_releases._build_folder_of(path)


def test_the_switch_reads_tags_and_branches_again_in_one_request(
    hub, capsys, monkeypatch
):
    """Just before its commit the switch compares one listing of the tags and
    branches with its first read's, and asks the Hub nothing else, so a tag
    made or deleted after that listing cannot fall between its requests."""
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)
    requests: list[str] = []
    list_repo_refs, repo_info = hub.list_repo_refs, hub.repo_info

    def listing(*args, **kwargs):
        requests.append("list_repo_refs")
        return list_repo_refs(*args, **kwargs)

    def resolution(*args, **kwargs):
        requests.append("repo_info")
        return repo_info(*args, **kwargs)

    monkeypatch.setattr(hub, "list_repo_refs", listing)
    monkeypatch.setattr(hub, "repo_info", resolution)

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", *INCLUDE_JUNE_2023
    )

    assert exit_code == 0, error_output
    second_listing = [
        index for index, request in enumerate(requests) if request == "list_repo_refs"
    ][1]
    assert requests[second_listing + 1 :] == []


def test_switch_ignores_branches_that_are_not_the_tagged_publishers(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDE_JUNE_2023)
    hub.branches["draft-notes"] = hub.main
    hub.branches["release-staging-notes"] = hub.main

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", *INCLUDE_JUNE_2023
    )

    assert exit_code == 0, error_output
    assert "releases.json" in hub.files_at()
