"""The build-folder switch on the US repository's tagged layout.

Every US build is its own release in ``releases/<tag>/``, so each build moves
with ``--include TAG=RELEASE``. The US builds have two shapes the UK builds
did not: an ACS local-area build keeps files that its checksum ledger records
but its release manifest doesn't declare, and an SPM build declares its own
``build_manifest.json`` as an artifact. A July publication also left its
staging branch behind after it finished. These tests run the tool against the
in-memory repository in ``test_support.microcosm_build.build_folder_switch``;
none reaches the Hub.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from microcosm.build import build_folder_switch
from test_support.microcosm_build.build_folder_switch import (
    ACS_TAG,
    ACS_UNDECLARED_FILES,
    FINISHED_STAGING_BRANCH,
    FINISHED_TAG,
    SPM_TAG,
    US_LOCAL,
    US_NATIONAL,
    US_REPOSITORY,
    FakeDatasetRepository,
    sha256,
    us_repository,
)

INCLUDES = (
    "--include",
    f"{SPM_TAG}={US_NATIONAL}",
    "--include",
    f"{ACS_TAG}={US_LOCAL}",
)
#: No pointer file names a current ACS build, so the switch is told which.
SWITCH_OPTIONS = (*INCLUDES, "--current", f"{US_LOCAL}={ACS_TAG}")
FINISHED = ("--finished-branch", FINISHED_STAGING_BRANCH)
#: When a test changes the repository after the copy step.
LATER = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
SPM_FOLDER = f"releases/{US_NATIONAL}/{SPM_TAG}"
ACS_FOLDER = f"releases/{US_LOCAL}/{ACS_TAG}"


@pytest.fixture(autouse=True)
def no_hub_access(monkeypatch):
    """Every test runs offline and against the in-memory repository only."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def no_hub_client():
        pytest.fail("the tool built a Hub client")

    monkeypatch.setattr(build_folder_switch, "_default_api", no_hub_client)


@pytest.fixture
def hub(tmp_path) -> FakeDatasetRepository:
    return us_repository(tmp_path / "hub-cache")


def run(
    hub: FakeDatasetRepository, capsys, step: str, *options: str
) -> tuple[int, dict, str]:
    """Run one step on the in-memory repository: exit code, report and stderr."""
    exit_code = build_folder_switch.main([step, US_REPOSITORY, *options], api=hub)
    captured = capsys.readouterr()
    report = json.loads(captured.out) if captured.out.strip() else {}
    return exit_code, report, captured.err


def ledger_digests(ledger: bytes) -> dict[str, str]:
    """Each file a checksum ledger records, with its SHA-256."""
    digests = {}
    for entry in ledger.decode().splitlines():
        if entry.strip():
            digest, name = entry.split()
            digests[name] = digest
    return digests


def build_status(report: dict, tag: str) -> str | None:
    return next(
        (build["status"] for build in report.get("builds", []) if build["tag"] == tag),
        None,
    )


def test_copy_moves_the_files_an_acs_build_s_checksum_ledger_records(hub, capsys):
    exit_code, report, error_output = run(hub, capsys, "copy", "--write", *INCLUDES)

    assert exit_code == 0, error_output
    files = hub.files_at()
    for name, content in ACS_UNDECLARED_FILES.items():
        assert files.get(f"{ACS_FOLDER}/{name}") == content
    recorded = ledger_digests(files.get(f"{ACS_FOLDER}/sha256sums.txt", b""))
    for name, content in ACS_UNDECLARED_FILES.items():
        assert recorded.get(name) == sha256(content)
    assert recorded.get("release_manifest.json") == sha256(
        files.get(f"{ACS_FOLDER}/release_manifest.json", b"")
    )
    # They stay undeclared: the rewritten manifest declares what the tagged
    # one declared, and nothing of the build's folder is left behind.
    manifest = json.loads(files.get(f"{ACS_FOLDER}/release_manifest.json", b"{}"))
    assert {entry["path"] for entry in manifest.get("artifacts", {}).values()} == {
        "populace_us_2024_acs_local.h5",
        "calibration_diagnostics.json",
    }
    acs = next(
        (build for build in report.get("builds", []) if build["tag"] == ACS_TAG), {}
    )
    assert acs.get("left_behind") == []


def test_a_copied_acs_build_counts_as_copied(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDES)

    exit_code, report, error_output = run(hub, capsys, "copy", *INCLUDES)

    assert exit_code == 0, error_output
    assert build_status(report, ACS_TAG) == "already_copied"


def test_copy_refuses_an_undeclared_file_that_does_not_match_its_ledger(hub, capsys):
    # The ledger records each file in name order; this is the last of them.
    tagged_files = hub.commits[hub.tags[ACS_TAG]]
    tagged_files[f"releases/{ACS_TAG}/run_identity.json"] = b"changed after"

    exit_code, report, _ = run(hub, capsys, "copy", *INCLUDES)

    assert exit_code == 1
    problems = [problem for problem in report.get("problems", []) if ACS_TAG in problem]
    assert len(problems) == 1
    assert "run_identity.json" in problems[0]


def test_copy_moves_an_spm_build_whose_manifest_declares_its_build_manifest(
    hub, capsys
):
    exit_code, _, error_output = run(hub, capsys, "copy", "--write", *INCLUDES)

    assert exit_code == 0, error_output
    files = hub.files_at()
    build_manifest = files.get(f"{SPM_FOLDER}/build_manifest.json", b"")
    assert json.loads(build_manifest or b"{}").get("build") == SPM_TAG
    manifest = json.loads(files.get(f"{SPM_FOLDER}/release_manifest.json", b"{}"))
    declared = manifest.get("artifacts", {}).get("build_manifest", {})
    # The manifest declares the rewritten build manifest, by its new SHA-256.
    assert (declared.get("path"), declared.get("sha256")) == (
        "build_manifest.json",
        sha256(build_manifest),
    )
    parent = f"releases/{SPM_TAG}/parent_build_manifest.json"
    assert (
        files.get(f"{SPM_FOLDER}/parent_build_manifest.json")
        == hub.files_at(SPM_TAG)[parent]
    )


def test_a_copied_spm_build_counts_as_copied(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDES)

    exit_code, report, error_output = run(hub, capsys, "copy", *INCLUDES)

    assert exit_code == 0, error_output
    assert build_status(report, SPM_TAG) == "already_copied"


def test_switch_accepts_a_named_finished_staging_branch(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDES)

    exit_code, report, error_output = run(
        hub, capsys, "switch", "--write", *SWITCH_OPTIONS, *FINISHED
    )

    assert exit_code == 0, error_output
    assert report.get("finished_branches") == [FINISHED_STAGING_BRANCH]
    record = json.loads(hub.files_at().get("releases.json", b"{}"))
    assert record.get("default") == US_NATIONAL
    assert {
        release: entry.get("current")
        for release, entry in record.get("releases", {}).items()
    } == {US_NATIONAL: SPM_TAG, US_LOCAL: ACS_TAG}
    assert "latest.json" not in hub.files_at()
    # The leftover build stays reachable by its tag, and its branch stays.
    assert FINISHED_TAG in report.get("not_moved", [])
    assert FINISHED_STAGING_BRANCH in hub.branches


def test_switch_still_refuses_a_staging_branch_it_was_not_told_about(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDES)

    exit_code, report, _ = run(hub, capsys, "switch", *SWITCH_OPTIONS)

    assert exit_code == 1
    assert any(
        FINISHED_STAGING_BRANCH in problem and "--finished-branch" in problem
        for problem in report.get("problems", [])
    )


def test_switch_refuses_a_named_branch_whose_build_has_no_tag(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDES)
    del hub.tags[FINISHED_TAG]

    exit_code, report, _ = run(hub, capsys, "switch", *SWITCH_OPTIONS, *FINISHED)

    assert exit_code == 1
    assert any(
        FINISHED_STAGING_BRANCH in problem and "no tag" in problem
        for problem in report.get("problems", [])
    )


def test_switch_refuses_a_named_branch_whose_head_is_not_on_main(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDES)
    # As a running publication's branch is between its tag and the branch's
    # deletion: it names the tagged commit, which is not on main.
    hub.branches[FINISHED_STAGING_BRANCH] = hub.tags[FINISHED_TAG]

    exit_code, report, error_output = run(
        hub, capsys, "switch", "--write", *SWITCH_OPTIONS, *FINISHED
    )

    assert exit_code == 1
    assert "not on main" in error_output
    assert "releases.json" not in hub.files_at()


def test_switch_refuses_a_named_branch_that_does_not_exist(hub, capsys):
    run(hub, capsys, "copy", "--write", *INCLUDES)
    del hub.branches[FINISHED_STAGING_BRANCH]

    exit_code, report, _ = run(hub, capsys, "switch", *SWITCH_OPTIONS, *FINISHED)

    assert exit_code == 1
    assert any(
        FINISHED_STAGING_BRANCH in problem
        and "not a release-staging/ branch" in problem
        for problem in report.get("problems", [])
    )


def watch_listings(hub, monkeypatch, after_first_listing):
    """Run ``after_first_listing`` once, right after the switch's first
    listing of the repository's tags and branches."""
    list_repo_refs = hub.list_repo_refs
    listings = []

    def listing(*args, **kwargs):
        refs = list_repo_refs(*args, **kwargs)
        listings.append(refs)
        if len(listings) == 1:
            after_first_listing()
        return refs

    monkeypatch.setattr(hub, "list_repo_refs", listing)


def test_switch_refuses_when_a_named_finished_branch_moves_while_it_reads(
    hub, capsys, monkeypatch
):
    run(hub, capsys, "copy", "--write", *INCLUDES)

    def a_publication_uses_the_branch():
        hub.branches[FINISHED_STAGING_BRANCH] = hub.tags[FINISHED_TAG]

    watch_listings(hub, monkeypatch, a_publication_uses_the_branch)

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", *SWITCH_OPTIONS, *FINISHED
    )

    assert exit_code == 1
    assert FINISHED_STAGING_BRANCH in error_output
    # The operator already named it, so the refusal says it moved instead of
    # asking for --finished-branch again.
    assert "moved" in error_output
    assert "name the branch with --finished-branch" not in error_output
    assert "releases.json" not in hub.files_at()


def test_switch_refuses_when_a_named_finished_branch_disappears_while_it_reads(
    hub, capsys, monkeypatch
):
    run(hub, capsys, "copy", "--write", *INCLUDES)
    # A retry of the build can make its branch only once the leftover one is
    # gone, so a branch that disappears is a reason to read again.
    watch_listings(hub, monkeypatch, lambda: hub.branches.pop(FINISHED_STAGING_BRANCH))

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", *SWITCH_OPTIONS, *FINISHED
    )

    assert exit_code == 1
    assert FINISHED_STAGING_BRANCH in error_output
    assert "was deleted" in error_output
    # A run again must leave the option out, or the branch is refused as one
    # that doesn't exist.
    assert f"without --finished-branch {FINISHED_STAGING_BRANCH}" in error_output
    assert "releases.json" not in hub.files_at()


def test_switch_refuses_a_new_staging_branch_beside_an_accepted_one(
    hub, capsys, monkeypatch
):
    run(hub, capsys, "copy", "--write", *INCLUDES)
    other = "release-staging/populace-us-2024-spm-20261110"

    def a_publication_starts():
        hub.branches[other] = hub.main

    watch_listings(hub, monkeypatch, a_publication_starts)

    exit_code, _, error_output = run(
        hub, capsys, "switch", "--write", *SWITCH_OPTIONS, *FINISHED
    )

    assert exit_code == 1
    assert f"branch {other} appeared" in error_output
    # It appeared during the switch, so it is no finished publication's.
    assert f"--finished-branch {other}" not in error_output
    assert "releases.json" not in hub.files_at()


def test_copy_refuses_a_ledger_entry_for_a_file_the_tagged_folder_lacks(hub, capsys):
    del hub.commits[hub.tags[ACS_TAG]][f"releases/{ACS_TAG}/run_identity.json"]

    exit_code, report, _ = run(hub, capsys, "copy", *INCLUDES)

    assert exit_code == 1
    assert any(
        ACS_TAG in problem
        and "run_identity.json" in problem
        and "does not hold" in problem
        for problem in report.get("problems", [])
    )


def test_copy_refuses_a_ledger_that_records_a_file_twice(hub, capsys):
    tagged_files = hub.commits[hub.tags[ACS_TAG]]
    ledger_path = f"releases/{ACS_TAG}/sha256sums.txt"
    tagged_files[ledger_path] += (
        f"{sha256(b'other bytes')}  consumer_export.json\n".encode()
    )

    exit_code, report, _ = run(hub, capsys, "copy", *INCLUDES)

    assert exit_code == 1
    assert any(
        ACS_TAG in problem and "consumer_export.json twice" in problem
        for problem in report.get("problems", [])
    )


def test_copy_refuses_a_declared_build_manifest_that_does_not_match(hub, capsys):
    tagged_files = hub.commits[hub.tags[SPM_TAG]]
    tagged_files[f"releases/{SPM_TAG}/build_manifest.json"] = b'{"changed": true}'

    exit_code, report, _ = run(hub, capsys, "copy", *INCLUDES)

    assert exit_code == 1
    assert any(
        SPM_TAG in problem and "build_manifest.json" in problem
        for problem in report.get("problems", [])
    )


def test_the_copy_plan_lists_each_build_s_undeclared_files(hub, capsys):
    exit_code, report, error_output = run(hub, capsys, "copy", *INCLUDES)

    assert exit_code == 0, error_output
    assert {
        build["tag"]: build.get("undeclared") for build in report.get("builds", [])
    } == {SPM_TAG: [], ACS_TAG: sorted(ACS_UNDECLARED_FILES)}


def test_an_undeclared_file_changed_after_the_copy_makes_its_folder_occupied(
    hub, capsys
):
    run(hub, capsys, "copy", "--write", *INCLUDES)
    hub.seed({f"{ACS_FOLDER}/run_identity.json": b"changed on main"}, date=LATER)

    exit_code, report, _ = run(hub, capsys, "copy", *INCLUDES)

    assert exit_code == 1
    assert build_status(report, ACS_TAG) == "occupied"
    assert any(
        "run_identity.json" in problem and "sha256sums.txt records" in problem
        for problem in report.get("problems", [])
    )


def test_a_declared_build_manifest_changed_after_the_copy_makes_its_folder_occupied(
    hub, capsys
):
    run(hub, capsys, "copy", "--write", *INCLUDES)
    hub.seed({f"{SPM_FOLDER}/build_manifest.json": b'{"changed": true}'}, date=LATER)

    exit_code, report, _ = run(hub, capsys, "copy", *INCLUDES)

    assert exit_code == 1
    assert build_status(report, SPM_TAG) == "occupied"
    assert any(
        "build_manifest.json" in problem and "its manifest records" in problem
        for problem in report.get("problems", [])
    )
