"""Publishing a UK release build through the emitter (hand-over step H2c).

The emitter's release writer puts each build in its own folder and records
it in ``releases.json``; Microcosm hands the build over with
``publish_release`` and treats it as current only when the receipt says so.
These tests use the in-memory client in ``test_support.microcosm_build``; the
emitter's own tests cover the writer, its queue and the Hub.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import huggingface_hub
import pytest
from microcosm_emitter_interface import CollectorProvenance
from microcosm_emitter_interface.releases import ReleaseRecordError, ReleasesRecord

from microcosm.build import collector_submissions, release_publication
from microcosm.data import release as release_module
from microcosm.data.releases_record import parse_releases_record
from test_support.microcosm_build.emitter import FakeEmitterClient
from test_support.microcosm_data.build_folders import UK_NATIONAL_BUILD
from test_support.microcosm_data.release import (
    _as_uk_national_build_folder_release,
    _as_uk_national_line_release,
    artifact_root,  # noqa: F401  (a fixture)
    release_dir,  # noqa: F401  (a fixture)
)

UK_REPO_ID = "policyengine/populace-uk-private"
UK_NATIONAL_RELEASE_ID = "microcosm-uk-2024-25-national"
UK_EARLIER_BUILD = "uk-frs-calibration-attempt-20260901T000000Z-0badcafe"
#: The command's own listing of the repository's files, before the fixture
#: below replaces it in every test.
REAL_REPOSITORY_FILES = release_publication._repository_files
#: The variables ``huggingface_hub`` reads to refuse every Hub request.
OFFLINE_VARIABLES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
#: What ``check_write_access`` answers when the emitter finds no credential.
NO_CREDENTIAL = {
    "status": "checked",
    "credential": "missing",
    "reachable": None,
    "can_write": None,
    "error_type": None,
}
#: What it answers for a credential that can read but not write.
READ_ONLY_CREDENTIAL = {
    "status": "checked",
    "credential": "present",
    "reachable": True,
    "can_write": False,
    "error_type": None,
}


@dataclass
class HubListing:
    """The dataset repository's file list as the command reads it.

    ``files`` is what it lists, and ``listed`` the repositories it asked for.
    """

    files: list[str] = field(default_factory=list)
    listed: list[str] = field(default_factory=list)


@pytest.fixture(autouse=True)
def hub_listing(monkeypatch) -> HubListing:
    """Run each test as on a build machine that can reach the Hub.

    Local test runs set ``HF_HUB_OFFLINE``, and the command refuses to start
    an emitter with it set. Every emitter here is the in-memory client, and
    the repository's root fixture refuses to start a real one. The file list
    the command reads, to refuse a repository that still has pointer files,
    is in memory too and empty unless a test adds to it.
    """
    for name in OFFLINE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    listing = HubListing()

    def list_files(repository: str) -> list[str]:
        listing.listed.append(repository)
        return list(listing.files)

    monkeypatch.setattr(
        release_publication, "_repository_files", list_files, raising=False
    )
    return listing


@pytest.fixture
def national(release_dir: Path, monkeypatch) -> Path:  # noqa: F811
    """A UK national build; the release contract's own tests cover its checks."""
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: []
    )
    return _as_uk_national_build_folder_release(release_dir)


def _record(client: FakeEmitterClient) -> dict:
    return json.loads(client.hub_files[(UK_REPO_ID, "releases.json")])


class Clock:
    """The command's clock and sleep while it waits for the collector."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(release_publication, "_clock", clock)
    monkeypatch.setattr(release_publication, "_sleep", clock.sleep)
    return clock


def test_publishing_hands_every_file_over_with_its_role(
    national: Path,
    artifact_root: Path,  # noqa: F811
) -> None:
    client = FakeEmitterClient()

    outcome = release_publication.publish_release_build(
        national, artifact_root=artifact_root, client=client
    )

    (request,) = client.release_requests
    assert (request.repository, request.release, request.build) == (
        UK_REPO_ID,
        UK_NATIONAL_RELEASE_ID,
        UK_NATIONAL_BUILD,
    )
    assert (request.release_role, request.make_current) == ("national", False)
    files = {item.name: item for item in request.files}
    assert files["release_manifest.json"].role == "release_manifest"
    assert files["build_manifest.json"].role == "build_manifest"
    assert files["populace_us_2024.h5"].role == "dataset"
    assert files["populace_us_2024.h5"].local_path == (
        artifact_root / "populace_us_2024.h5"
    )
    assert files["calibration_diagnostics.json"].role == "calibration_diagnostics"
    folder = f"releases/{UK_NATIONAL_RELEASE_ID}/{UK_NATIONAL_BUILD}"
    assert client.hub_files[(UK_REPO_ID, f"{folder}/populace_us_2024.h5")] == (
        b"h5 payload"
    )
    release = _record(client)["releases"][UK_NATIONAL_RELEASE_ID]
    assert release["role"] == "national"
    assert release["current"] is None
    assert list(release["builds"]) == [UK_NATIONAL_BUILD]
    assert outcome.published and not outcome.current and outcome.succeeded
    assert client.closed is False, "a client the caller passed stays open"


def test_a_publication_names_its_build_for_the_collector(
    national: Path,
    artifact_root: Path,  # noqa: F811
) -> None:
    """The emitter submits the build's calibration diagnostics to the
    collector after the release commit, under the build's own folder."""
    client = FakeEmitterClient()

    release_publication.publish_release_build(
        national, artifact_root=artifact_root, client=client
    )

    (request,) = client.release_requests
    assert request.collector == CollectorProvenance(
        country_code="UK",
        build_id=UK_NATIONAL_BUILD,
        licensed_data=True,
        release=UK_NATIONAL_RELEASE_ID,
    )


@pytest.mark.parametrize(
    "release",
    [
        UK_NATIONAL_RELEASE_ID,
        "microcosm-uk-2024-25-dense",
        "microcosm-uk-2024-25-local-k1",
        "microcosm-uk-2024-25-local-k12",
    ],
)
def test_every_uk_release_names_its_builds_as_licensed_uk_builds(release) -> None:
    assert release_publication.collector_provenance(
        release, UK_NATIONAL_BUILD
    ) == CollectorProvenance(
        country_code="UK",
        build_id=UK_NATIONAL_BUILD,
        licensed_data=True,
        release=release,
    )


def test_a_release_that_publishes_with_tags_names_no_build() -> None:
    assert (
        release_publication.collector_provenance(
            "populace-uk-2023-dd68c73", UK_NATIONAL_BUILD
        )
        is None
    )


def test_a_published_build_waits_until_the_collector_accepts_its_submissions(
    national: Path,
    artifact_root: Path,  # noqa: F811
    clock: Clock,
    capsys,
) -> None:
    """A short-lived emitter starts no new submission once it is closing, so
    the command waits for the collector to accept them before it closes."""
    client = FakeEmitterClient(
        collector_receipts=[{"status": "pending"}, {"status": "awaiting_final_status"}]
    )

    outcome = release_publication.publish_release_build(
        national, artifact_root=artifact_root, client=client
    )

    (request,) = client.release_requests
    assert client.collector_receipt_requests == [request.handover_id()] * 2
    assert clock.slept == [collector_submissions.COLLECTOR_RECEIPT_POLL_SECONDS]
    assert outcome.collector_receipt is not None
    assert outcome.collector_receipt["status"] == "awaiting_final_status"
    assert outcome.as_json()["collector_submission"] == {
        "status": "awaiting_final_status",
        "error_code": None,
    }
    assert outcome.succeeded
    err = capsys.readouterr().err
    assert (
        f"build {UK_NATIONAL_BUILD} of {UK_NATIONAL_RELEASE_ID} is on Hugging Face; "
        "waiting up to 10 minutes for the collector to accept its submissions..." in err
    )
    assert "warning" not in err


def test_the_collector_receipt_is_asked_for_by_the_emitters_hand_over_id(
    national: Path,
    artifact_root: Path,  # noqa: F811
    clock: Clock,
) -> None:
    """The emitter keeps the collector row under the hand-over ID its receipt
    names, so that is the ID the command asks about."""
    emitter_handover_id = "f" * 32
    client = FakeEmitterClient(
        release_receipts=[{"handover_id": emitter_handover_id}],
        collector_receipts=[{"status": "succeeded", "attempts": 1}],
    )

    outcome = release_publication.publish_release_build(
        national, artifact_root=artifact_root, client=client
    )

    assert client.collector_receipt_requests == [emitter_handover_id]
    assert outcome.collector_receipt["status"] == "succeeded"


def test_a_receipt_without_a_hand_over_id_falls_back_to_the_requests_own(
    national: Path,
    artifact_root: Path,  # noqa: F811
    clock: Clock,
) -> None:
    client = FakeEmitterClient(
        release_receipts=[{"handover_id": ""}],
        collector_receipts=[{"status": "succeeded", "attempts": 1}],
    )

    outcome = release_publication.publish_release_build(
        national, artifact_root=artifact_root, client=client
    )

    (request,) = client.release_requests
    assert client.collector_receipt_requests == [request.handover_id()]
    assert outcome.collector_receipt["status"] == "succeeded"


def test_a_published_build_the_emitter_has_no_collector_row_for_is_not_waited_for(
    national: Path,
    artifact_root: Path,  # noqa: F811
    clock: Clock,
) -> None:
    client = FakeEmitterClient()

    outcome = release_publication.publish_release_build(
        national, artifact_root=artifact_root, client=client
    )

    assert len(client.collector_receipt_requests) == 1
    assert clock.slept == []
    assert outcome.collector_receipt is None
    assert outcome.as_json()["collector_submission"] is None
    assert outcome.succeeded


def test_refused_collector_settings_are_reported_and_the_publication_succeeds(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
) -> None:
    client = FakeEmitterClient(
        collector_receipts=[
            {"status": "skipped", "error_code": "COLLECTOR_SETTINGS_REFUSED"}
        ]
    )
    monkeypatch.setattr(release_publication, "_start_release_emitter", lambda: client)

    code, output, error = _run(
        [str(national), "--artifact-root", str(artifact_root)], capsys
    )

    assert code == 0
    assert output["status"] == "published"
    assert output["collector_submission"] == {
        "status": "skipped",
        "error_code": "COLLECTOR_SETTINGS_REFUSED",
    }
    assert (
        f"warning: the collector receives nothing of build {UK_NATIONAL_BUILD} "
        f"of {UK_NATIONAL_RELEASE_ID}: the emitter refused its collector "
        "settings or the collector-token variables it inherits "
        "(COLLECTOR_SETTINGS_REFUSED)." in error
    )
    # The command asked before it closed the emitter.
    assert client.calls.index("collector_submission_receipt") < client.calls.index(
        "close"
    )


@pytest.mark.parametrize(
    "receipt",
    [
        {"status": "pending", "error_code": "REPOSITORY_UNAVAILABLE"},
        {"status": "refused", "error_code": "BUILD_FOLDER_EXISTS"},
    ],
    ids=["pending", "refused"],
)
def test_a_publication_that_has_not_landed_is_asked_about_once(
    national: Path,
    artifact_root: Path,  # noqa: F811
    clock: Clock,
    receipt: dict,
) -> None:
    client = FakeEmitterClient(
        release_receipts=[receipt],
        collector_receipts=[
            {"status": "pending", "error_code": "WAITING_FOR_HUGGING_FACE"}
        ],
    )

    outcome = release_publication.publish_release_build(
        national, artifact_root=artifact_root, client=client
    )

    assert len(client.collector_receipt_requests) == 1
    assert clock.slept == []
    assert outcome.collector_receipt["status"] == "pending"


def test_a_build_published_as_current_is_current(
    national: Path,
    artifact_root: Path,  # noqa: F811
) -> None:
    client = FakeEmitterClient()

    outcome = release_publication.publish_release_build(
        national, artifact_root=artifact_root, make_current=True, client=client
    )

    assert client.release_requests[0].make_current is True
    assert _record(client)["releases"][UK_NATIONAL_RELEASE_ID]["current"] == (
        UK_NATIONAL_BUILD
    )
    assert outcome.current and outcome.succeeded


@pytest.mark.parametrize(
    ("receipt", "current"),
    [
        ({"status": "published", "current": UK_NATIONAL_BUILD}, True),
        # A promotion requested after this publication landed first.
        ({"status": "published", "current": UK_EARLIER_BUILD}, False),
        ({"status": "published", "current": None}, False),
        ({"status": "pending", "current": UK_NATIONAL_BUILD}, False),
        ({"status": "refused", "current": UK_NATIONAL_BUILD}, False),
        ({"status": "unavailable", "current": UK_NATIONAL_BUILD}, False),
    ],
)
def test_a_build_is_current_only_when_a_published_receipt_names_it(
    national: Path,
    artifact_root: Path,  # noqa: F811
    receipt: dict,
    current: bool,
) -> None:
    client = FakeEmitterClient(release_receipts=[receipt])

    outcome = release_publication.publish_release_build(
        national, artifact_root=artifact_root, make_current=True, client=client
    )

    assert outcome.current is current
    assert outcome.succeeded is current


def test_pointer_files_refuse_a_build_made_current(
    national: Path,
    artifact_root: Path,  # noqa: F811
) -> None:
    """While latest*.json files exist, the writer changes no current build."""
    client = FakeEmitterClient()
    client.hub_files[(UK_REPO_ID, "latest-national.json")] = b"{}"

    outcome = release_publication.publish_release_build(
        national, artifact_root=artifact_root, make_current=True, client=client
    )

    assert outcome.receipt["status"] == "refused"
    assert outcome.receipt["error_code"] == "POINTER_FILES_PRESENT"
    # The command refuses such a repository before handing anything over;
    # this is a pointer file that appeared after that check. The tagged
    # publisher refuses these releases, so the advice is to remove the file.
    explanation = outcome.explanation()
    assert "tools/publish_release.sh" not in explanation
    assert "remove it, then run the command again" in explanation
    assert "without --make-current" not in explanation
    assert (UK_REPO_ID, "releases.json") not in client.hub_files


def test_promoting_makes_a_published_build_current(
    national: Path,
    artifact_root: Path,  # noqa: F811
) -> None:
    client = FakeEmitterClient()
    release_publication.publish_release_build(
        national, artifact_root=artifact_root, client=client
    )

    outcome = release_publication.promote_release_build(national, client=client)

    assert client.calls.count("promote_build") == 1
    (publication, promotion) = client.release_requests
    assert (promotion.release, promotion.build) == (
        UK_NATIONAL_RELEASE_ID,
        UK_NATIONAL_BUILD,
    )
    # A promotion names its build for the collector, as a publication does,
    # so the emitter submits the releases.json its commit wrote, and the
    # command asks what became of that submission. This emitter has no
    # collector row for it.
    assert promotion.collector == CollectorProvenance(
        country_code="UK",
        build_id=UK_NATIONAL_BUILD,
        licensed_data=True,
        release=UK_NATIONAL_RELEASE_ID,
    )
    assert client.collector_receipt_requests == [
        publication.handover_id(),
        promotion.handover_id(),
    ]
    assert outcome.collector_receipt is None
    assert outcome.as_json()["collector_submission"] is None
    assert _record(client)["releases"][UK_NATIONAL_RELEASE_ID]["current"] == (
        UK_NATIONAL_BUILD
    )
    assert outcome.current and outcome.succeeded


def test_a_promotion_waits_until_the_collector_accepts_the_new_releases_json(
    national: Path,
    artifact_root: Path,  # noqa: F811
    clock: Clock,
    capsys,
) -> None:
    """The emitter submits the releases.json the promotion wrote, and a
    closing emitter starts no new submission, so the command waits for the
    collector to accept it before it closes."""
    repository: dict = {}
    release_publication.publish_release_build(
        national,
        artifact_root=artifact_root,
        client=FakeEmitterClient(hub_files=repository),
    )
    client = FakeEmitterClient(
        hub_files=repository,
        collector_receipts=[{"status": "pending"}, {"status": "awaiting_final_status"}],
    )
    capsys.readouterr()

    outcome = release_publication.promote_release_build(national, client=client)

    (promotion,) = client.release_requests
    assert client.collector_receipt_requests == [promotion.handover_id()] * 2
    assert clock.slept == [collector_submissions.COLLECTOR_RECEIPT_POLL_SECONDS]
    assert outcome.collector_receipt is not None
    assert outcome.collector_receipt["status"] == "awaiting_final_status"
    assert outcome.as_json()["collector_submission"] == {
        "status": "awaiting_final_status",
        "error_code": None,
    }
    assert outcome.current and outcome.succeeded
    err = capsys.readouterr().err
    assert (
        f"releases.json with build {UK_NATIONAL_BUILD} current in "
        f"{UK_NATIONAL_RELEASE_ID} is on Hugging Face; waiting up to 10 minutes "
        "for the collector to accept its submissions..." in err
    )
    assert "warning" not in err


def test_the_promote_command_asks_the_collector_before_closing_its_emitter(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
) -> None:
    """A collector that turns the submission down is a warning; the
    promotion, which landed, still succeeds."""
    started, _ = _start_emitters(
        monkeypatch,
        {},
        {
            "collector_receipts": [
                {"status": "rejected", "error_code": "invalid_submission"}
            ]
        },
    )
    assert _run([str(national), "--artifact-root", str(artifact_root)], capsys)[0] == 0

    code, output, error = _run([str(national), "--promote"], capsys)

    assert code == 0, error
    assert output["current"] is True
    assert output["collector_submission"] == {
        "status": "rejected",
        "error_code": "invalid_submission",
    }
    assert (
        "warning: the collector rejected the submissions of releases.json with "
        f"build {UK_NATIONAL_BUILD} current in {UK_NATIONAL_RELEASE_ID} "
        "(invalid_submission)." in error
    )
    promoter = started[1]
    assert promoter.calls.index("collector_submission_receipt") < (
        promoter.calls.index("close")
    )


def test_a_promotion_that_has_not_landed_is_not_waited_for(
    national: Path,
    clock: Clock,
) -> None:
    """Its submission waits for the commit, so the command asks once."""
    client = FakeEmitterClient(
        release_receipts=[
            {"status": "pending", "error_code": "REPOSITORY_UNAVAILABLE"}
        ],
        collector_receipts=[
            {"status": "pending", "error_code": "WAITING_FOR_HUGGING_FACE"}
        ],
    )

    outcome = release_publication.promote_release_build(national, client=client)

    (promotion,) = client.release_requests
    assert client.collector_receipt_requests == [promotion.handover_id()]
    assert clock.slept == []
    assert not outcome.succeeded


def test_a_repeated_publication_is_published_not_refused(
    national: Path,
    artifact_root: Path,  # noqa: F811
) -> None:
    """Running the same command again after a lost receipt is safe."""
    client = FakeEmitterClient()
    for _ in range(2):
        outcome = release_publication.publish_release_build(
            national, artifact_root=artifact_root, client=client
        )
    assert outcome.published


def test_an_emitter_without_its_hugging_face_module_publishes_nothing(
    national: Path,
    artifact_root: Path,  # noqa: F811
) -> None:
    client = FakeEmitterClient(modules=())

    outcome = release_publication.publish_release_build(
        national, artifact_root=artifact_root, client=client
    )

    assert outcome.receipt["status"] == "unavailable"
    assert outcome.problem is not None and not outcome.succeeded
    assert client.release_requests == []
    assert client.collector_receipt_requests == []


def test_a_started_emitter_is_closed_after_the_publication(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
) -> None:
    client = FakeEmitterClient()
    monkeypatch.setattr(release_publication, "_start_release_emitter", lambda: client)

    release_publication.publish_release_build(national, artifact_root=artifact_root)

    assert client.closed is True


def _run(argv: list[str], capsys) -> tuple[int, dict | None, str]:
    code = release_publication.main(argv)
    captured = capsys.readouterr()
    output = json.loads(captured.out) if captured.out.strip() else None
    return code, output, captured.err


def test_the_command_publishes_and_prints_the_receipt(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
) -> None:
    client = FakeEmitterClient()
    monkeypatch.setattr(release_publication, "_start_release_emitter", lambda: client)

    code, output, _ = _run(
        [
            str(national),
            "--repo-id",
            UK_REPO_ID,
            "--artifact-root",
            str(artifact_root),
            "--make-current",
        ],
        capsys,
    )

    assert code == 0
    assert output["status"] == "published"
    assert (output["release"], output["build"]) == (
        UK_NATIONAL_RELEASE_ID,
        UK_NATIONAL_BUILD,
    )
    assert output["current"] is True
    assert client.closed is True


def test_the_command_fails_when_the_build_did_not_become_current(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
) -> None:
    client = FakeEmitterClient(
        release_receipts=[
            {"status": "published", "current": UK_EARLIER_BUILD, "message": "later"}
        ]
    )
    monkeypatch.setattr(release_publication, "_start_release_emitter", lambda: client)

    code, output, error = _run(
        [str(national), "--artifact-root", str(artifact_root), "--make-current"],
        capsys,
    )

    assert code == 1
    assert output["current"] is False
    assert output["current_build"] == UK_EARLIER_BUILD
    assert "is not current" in error


def test_the_command_promotes_with_promote(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
) -> None:
    """Each command starts its own emitter; both write to one repository."""
    repository: dict = {}
    emitters: list[FakeEmitterClient] = []

    def start() -> FakeEmitterClient:
        emitters.append(FakeEmitterClient(hub_files=repository))
        return emitters[-1]

    monkeypatch.setattr(release_publication, "_start_release_emitter", start)
    assert _run([str(national), "--artifact-root", str(artifact_root)], capsys)[0] == 0

    code, output, _ = _run([str(national), "--promote"], capsys)

    assert code == 0
    assert output["current"] is True
    assert [emitter.calls.count("promote_build") for emitter in emitters] == [0, 1]
    assert all(emitter.closed for emitter in emitters)


def _start_emitters(
    monkeypatch, *options: dict
) -> tuple[list[FakeEmitterClient], dict]:
    """Each command run starts the next emitter; all write to one repository.

    ``options`` are the keyword arguments of each emitter in the order the
    commands start them; later emitters take none.
    """
    repository: dict = {}
    started: list[FakeEmitterClient] = []
    remaining = list(options)

    def start() -> FakeEmitterClient:
        settings = remaining.pop(0) if remaining else {}
        started.append(FakeEmitterClient(hub_files=repository, **settings))
        return started[-1]

    monkeypatch.setattr(release_publication, "_start_release_emitter", start)
    return started, repository


@pytest.mark.parametrize(
    ("receipt", "explanations"),
    [
        (
            {"status": "pending", "error_code": "REPOSITORY_UNAVAILABLE"},
            (
                "still trying (REPOSITORY_UNAVAILABLE)",
                "Run the same command again later",
            ),
        ),
        (
            {
                "status": "refused",
                "error_code": "BUILD_FOLDER_EXISTS",
                "message": "the build's folder already exists.",
            },
            (
                "refused it (BUILD_FOLDER_EXISTS): the build's folder already exists.",
                "never changed after publishing",
            ),
        ),
        ({"status": "unavailable"}, ("did not answer",)),
    ],
    ids=["pending", "refused", "unavailable"],
)
def test_the_command_fails_and_explains_a_change_that_did_not_land(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
    receipt: dict,
    explanations: tuple[str, ...],
) -> None:
    client = FakeEmitterClient(release_receipts=[receipt])
    monkeypatch.setattr(release_publication, "_start_release_emitter", lambda: client)

    code, output, error = _run(
        [str(national), "--artifact-root", str(artifact_root)], capsys
    )

    assert code == 1
    assert output["status"] == receipt["status"]
    assert output["error_code"] == receipt.get("error_code")
    assert output["current"] is False
    for explanation in explanations:
        assert explanation in error
    assert client.closed is True


@pytest.mark.parametrize(
    "pointer_file",
    ["latest.json", "latest-national.json", "releases/old/latest-evidence.json"],
)
@pytest.mark.parametrize("promote", [False, True], ids=["publish", "promote"])
def test_the_command_refuses_a_repository_with_pointer_files_before_an_emitter(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
    hub_listing: HubListing,
    pointer_file: str,
    promote: bool,
) -> None:
    """The emitter publishes only into a repository without pointer files.
    In a repository that has switched to releases.json, a pointer file
    appeared after the switch; the tagged publisher refuses these releases,
    so the advice is to remove it, not to publish through it."""
    started, repository = _start_emitters(monkeypatch)
    hub_listing.files.extend(["README.md", "releases.json", pointer_file])
    argv = (
        [str(national), "--promote"]
        if promote
        else [str(national), "--artifact-root", str(artifact_root)]
    )

    code, output, error = _run(argv, capsys)

    assert code == 1
    assert started == []
    assert output is None
    assert hub_listing.listed == [UK_REPO_ID]
    assert "refusing to publish" in error and pointer_file in error
    assert "tools/publish_release.sh" not in error
    assert "remove the file(s) that appeared" in error
    assert repository == {}


@pytest.mark.parametrize("promote", [False, True], ids=["publish", "promote"])
def test_the_command_refuses_a_repository_that_has_not_switched_without_advising_removal(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
    hub_listing: HubListing,
    promote: bool,
) -> None:
    """Before its switch a repository has no releases.json, the loader
    follows its pointer files and the switch reads them to choose each
    release's current build, so the refusal says the repository has not
    switched and does not advise removing them."""
    started, repository = _start_emitters(monkeypatch)
    hub_listing.files.extend(["README.md", "latest.json", "latest-national.json"])
    argv = (
        [str(national), "--promote"]
        if promote
        else [str(national), "--artifact-root", str(artifact_root)]
    )

    code, output, error = _run(argv, capsys)

    assert code == 1
    assert started == []
    assert output is None
    assert "has not switched to releases.json" in error
    assert "latest-national.json" in error
    assert "microcosm.build.build_folder_switch" in error
    assert "remove" not in error
    assert repository == {}


def test_the_command_refuses_a_network_error_while_listing(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
) -> None:
    """httpx's transport errors, such as a refused connection, are not
    OSError; they are refused with the same explanation, not a traceback."""
    started, _ = _start_emitters(monkeypatch)

    def unreachable(repository: str) -> list[str]:
        raise httpx.ConnectError("[Errno 61] Connection refused")

    monkeypatch.setattr(
        release_publication, "_repository_files", unreachable, raising=False
    )

    code, output, error = _run(
        [str(national), "--artifact-root", str(artifact_root)], capsys
    )

    assert code == 1
    assert started == []
    assert output is None
    assert f"could not list the files of {UK_REPO_ID}" in error
    assert "ConnectError" in error


@pytest.mark.parametrize(
    ("environment", "token"),
    [
        ({"HUGGINGFACE_TOKEN": "not-a-real-token-b"}, "not-a-real-token-b"),
        (
            {
                "HF_TOKEN": "not-a-real-token-a",
                "HUGGINGFACE_TOKEN": "not-a-real-token-b",
            },
            "not-a-real-token-a",
        ),
    ],
    ids=["huggingface-token", "hf-token-first"],
)
def test_the_listing_reads_the_credential_the_emitter_reads(
    monkeypatch, environment: dict, token: str
) -> None:
    """The emitter reads HF_TOKEN, then HUGGINGFACE_TOKEN, then the saved
    login; huggingface_hub alone does not read HUGGINGFACE_TOKEN, so the
    listing passes it on."""
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    tokens: list[str | None] = []

    class RecordingApi:
        def __init__(self, token: str | None = None) -> None:
            tokens.append(token)

        def list_repo_files(self, repo_id: str, *, repo_type: str) -> list[str]:
            assert (repo_id, repo_type) == (UK_REPO_ID, "dataset")
            return ["README.md"]

    monkeypatch.setattr(huggingface_hub, "HfApi", RecordingApi)

    assert REAL_REPOSITORY_FILES(UK_REPO_ID) == ["README.md"]
    assert tokens == [token]


@pytest.mark.parametrize("promote", [False, True], ids=["publish", "promote"])
def test_the_command_refuses_a_tagged_uk_release(
    release_dir: Path,  # noqa: F811
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
    hub_listing: HubListing,
    promote: bool,
) -> None:
    """A UK build that the assemblers wrote in the tagged layout, before the
    switch, describes the build in a ``build`` object in its release
    manifest; the command refuses it before it lists the repository or
    starts an emitter."""
    tagged = _as_uk_national_line_release(release_dir)
    started, _ = _start_emitters(monkeypatch)
    argv = (
        [str(tagged), "--promote"]
        if promote
        else [str(tagged), "--artifact-root", str(artifact_root)]
    )

    code, output, error = _run(argv, capsys)

    assert code == 1
    assert started == []
    assert hub_listing.listed == []
    assert output is None
    assert "does not name release" in error


def test_the_command_refuses_when_it_cannot_list_the_repository(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
) -> None:
    started, _ = _start_emitters(monkeypatch)

    def unlistable(repository: str) -> list[str]:
        raise OSError(f"401 Client Error for {repository}")

    monkeypatch.setattr(
        release_publication, "_repository_files", unlistable, raising=False
    )

    code, output, error = _run(
        [str(national), "--artifact-root", str(artifact_root)], capsys
    )

    assert code == 1
    assert started == []
    assert output is None
    assert f"could not list the files of {UK_REPO_ID}" in error
    assert "401 Client Error" in error


@pytest.mark.parametrize("variable", OFFLINE_VARIABLES)
@pytest.mark.parametrize("promote", [False, True], ids=["publish", "promote"])
def test_the_command_refuses_offline_mode_before_starting_an_emitter(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
    variable: str,
    promote: bool,
) -> None:
    """The emitter inherits the variable and could never reach the Hub, so the
    change would wait out the emitter's hour as pending."""
    started, _ = _start_emitters(monkeypatch)
    monkeypatch.setenv(variable, "1")
    argv = (
        [str(national), "--promote"]
        if promote
        else [str(national), "--artifact-root", str(artifact_root)]
    )

    code, output, error = _run(argv, capsys)

    assert code == 1
    assert started == []
    assert output is None
    assert "refusing to publish" in error and variable in error


@pytest.mark.parametrize(
    ("access", "reason"),
    [
        (
            NO_CREDENTIAL,
            "needs a Hugging Face write credential (HF_TOKEN or `hf auth login`)",
        ),
        (READ_ONLY_CREDENTIAL, "read-only"),
    ],
    ids=["no-credential", "read-only"],
)
def test_the_command_checks_write_access_before_publishing(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
    access: dict,
    reason: str,
) -> None:
    started, repository = _start_emitters(monkeypatch, {"write_access": access})

    code, output, error = _run(
        [str(national), "--artifact-root", str(artifact_root)], capsys
    )

    (emitter,) = started
    assert code == 1
    assert "check_write_access" in emitter.calls
    assert emitter.release_requests == []
    assert output["status"] == "unavailable"
    assert f"the release change to {UK_REPO_ID}" in error and reason in error
    assert (UK_REPO_ID, "releases.json") not in repository
    assert emitter.closed is True


def test_the_command_checks_write_access_before_promoting(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
) -> None:
    started, repository = _start_emitters(
        monkeypatch, {}, {"write_access": NO_CREDENTIAL}
    )
    assert _run([str(national), "--artifact-root", str(artifact_root)], capsys)[0] == 0

    code, output, error = _run([str(national), "--promote"], capsys)

    promoter = started[1]
    assert code == 1
    assert "check_write_access" in promoter.calls
    assert promoter.release_requests == []
    assert output["status"] == "unavailable"
    assert "needs a Hugging Face write credential" in error
    record = json.loads(repository[(UK_REPO_ID, "releases.json")])
    assert record["releases"][UK_NATIONAL_RELEASE_ID]["current"] is None


def test_preflight_checks_the_build_without_an_emitter(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
) -> None:
    # A preflight reads only local files, so offline mode does not refuse it.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        release_publication,
        "_start_release_emitter",
        lambda: pytest.fail("preflight starts no emitter"),
    )

    code, output, _ = _run(
        [str(national), "--artifact-root", str(artifact_root), "--preflight-only"],
        capsys,
    )

    assert code == 0
    assert output == {
        "valid": True,
        "published": False,
        "release": UK_NATIONAL_RELEASE_ID,
        "build": UK_NATIONAL_BUILD,
        "repository": UK_REPO_ID,
        "release_role": "national",
    }


def test_the_command_refuses_another_repository(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        release_publication,
        "_start_release_emitter",
        lambda: pytest.fail("a refused command starts no emitter"),
    )

    code, output, error = _run(
        [
            str(national),
            "--artifact-root",
            str(artifact_root),
            "--repo-id",
            "policyengine/populace-uk-staging",
        ],
        capsys,
    )

    assert (code, output) == (1, None)
    assert "names policyengine/populace-uk-private" in error


def test_the_command_does_one_thing_at_a_time(national: Path) -> None:  # noqa: F811
    with pytest.raises(SystemExit):
        release_publication.main([str(national), "--promote", "--make-current"])


def test_the_command_refuses_a_release_that_still_publishes_with_tags(
    release_dir: Path,  # noqa: F811
    artifact_root: Path,  # noqa: F811
    capsys,
) -> None:
    """A directory named after a US build ID is in the tagged layout; the
    refusal names the US releases and says to name the directory after one
    (microcosm#1570), since the tagged publisher refuses the US repository
    after its switch."""
    with pytest.raises(SystemExit):
        release_publication.main(
            [str(release_dir), "--artifact-root", str(artifact_root)]
        )
    err = capsys.readouterr().err
    assert "microcosm-us-2024-national" in err
    assert "named after the release" in err


def test_the_command_says_the_other_uk_releases_take_no_new_builds(
    tmp_path: Path, capsys
) -> None:
    """The June 2023 UK release does not publish into build folders, and the
    tagged publisher refuses the UK repository once it records its builds in
    releases.json, so the refusal says it takes no new builds instead of
    sending it there."""
    june = tmp_path / "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
    june.mkdir()

    with pytest.raises(SystemExit):
        release_publication.main([str(june)])

    assert "take no new builds" in capsys.readouterr().err


def test_the_command_refuses_a_build_whose_staging_never_arrived(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
) -> None:
    path = national / "build_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["staging"] = {"enabled": True, "uploads_succeeded": 0}
    path.write_text(json.dumps(manifest))
    client = FakeEmitterClient()
    monkeypatch.setattr(release_publication, "_start_release_emitter", lambda: client)

    code, _, error = _run(
        [str(national), "--artifact-root", str(artifact_root)], capsys
    )
    assert code == 1
    assert "refusing to publish" in error
    assert client.release_requests == []

    code, _, _ = _run(
        [
            str(national),
            "--artifact-root",
            str(artifact_root),
            "--allow-missing-staging",
        ],
        capsys,
    )
    assert code == 0


_RECORD = {
    "schema_version": 1,
    "default": UK_NATIONAL_RELEASE_ID,
    "releases": {
        UK_NATIONAL_RELEASE_ID: {
            "role": "national",
            "current": UK_NATIONAL_BUILD,
            "builds": {
                UK_NATIONAL_BUILD: {"published_at": "2026-11-04", "withdrawn": False},
                UK_EARLIER_BUILD: {"published_at": "2026-11-01", "withdrawn": True},
            },
        }
    },
}


def _variant(**changes) -> dict:
    record = json.loads(json.dumps(_RECORD))
    release = record["releases"][UK_NATIONAL_RELEASE_ID]
    for key, value in changes.items():
        if key in {"schema_version", "default"}:
            record[key] = value
        elif key == "extra":
            record["extra"] = value
        else:
            release[key] = value
    return record


@pytest.mark.parametrize(
    "document",
    [
        _RECORD,
        _variant(current=None),
        _variant(default=None),
        _variant(current=UK_EARLIER_BUILD),
        _variant(current="not-listed"),
        _variant(default="microcosm-uk-2024-25-dense"),
        _variant(schema_version=2),
        _variant(schema_version=True),
        _variant(schema_version=1.0),
        _variant(role="National"),
        _variant(extra={}),
        _variant(builds={"bad/name": {"published_at": "x", "withdrawn": False}}),
        _variant(builds={UK_NATIONAL_BUILD: {"published_at": "", "withdrawn": False}}),
        _variant(builds={UK_NATIONAL_BUILD: {"published_at": "x", "withdrawn": 0}}),
        [],
    ],
)
def test_the_loader_reads_releases_json_as_the_release_writer_does(
    document: object,
) -> None:
    """microcosm-data cannot depend on the interface package (it supports
    Python 3.11), so it repeats the writer's rules; both accept or refuse
    each record alike."""
    content = json.dumps(document).encode()
    try:
        ReleasesRecord.parse(content)
    except ReleaseRecordError:
        writer_accepts = False
    else:
        writer_accepts = True
    try:
        parse_releases_record(content)
    except ValueError:
        reader_accepts = False
    else:
        reader_accepts = True
    assert reader_accepts is writer_accepts


def test_the_roles_microcosm_names_are_the_interfaces() -> None:
    from microcosm_emitter_interface import bundles, publication

    from microcosm.data import contract

    assert contract.RELEASE_MANIFEST_FILE_ROLE == publication.RELEASE_MANIFEST_ROLE
    assert contract.BUILD_MANIFEST_FILE_ROLE == bundles.BUILD_MANIFEST_ROLE
    assert contract.CHECKSUMS_FILE_ROLE == bundles.CHECKSUMS_ROLE
    assert contract.DATASET_FILE_ROLE == bundles.DATASET_ROLE
    assert contract.CALIBRATION_DIAGNOSTICS_FILE_ROLE == (
        bundles.CALIBRATION_DIAGNOSTICS_ROLE
    )


# Withdrawing a published build ---------------------------------------------

#: A published build of the national release that is not its current build.
UK_OTHER_BUILD = "uk-frs-calibration-attempt-20261001T000000Z-0decade0"


def _two_listed_builds(*, with_current: bool = True) -> dict:
    """A switched repository listing two national builds; the first is current
    unless ``with_current`` is False, when neither is."""
    from microcosm_emitter_interface.releases import add_build

    record = ReleasesRecord.empty()
    files: dict = {}
    for build, current in ((UK_EARLIER_BUILD, with_current), (UK_OTHER_BUILD, False)):
        record = add_build(
            record,
            release=UK_NATIONAL_RELEASE_ID,
            build=build,
            release_role="national",
            published_at="2026-10-01T00:00:00+00:00",
            make_current=current,
        )
        manifest = {"release": UK_NATIONAL_RELEASE_ID, "build": build}
        files[
            (
                UK_REPO_ID,
                f"releases/{UK_NATIONAL_RELEASE_ID}/{build}/release_manifest.json",
            )
        ] = json.dumps(manifest).encode()
    files[(UK_REPO_ID, "releases.json")] = record.serialize()
    return files


def _withdraw_function():
    """``release_publication.withdraw_release_build``, which must exist."""
    withdraw = getattr(release_publication, "withdraw_release_build", None)
    assert withdraw is not None, "release_publication has no withdraw_release_build"
    return withdraw


def _withdraw_command(argv: list[str], capsys) -> tuple[int, dict | None, str]:
    """Run ``microcosm-withdraw-release-build``; its entry point must exist."""
    main = getattr(release_publication, "withdraw_main", None)
    assert main is not None, "release_publication has no withdraw_main"
    code = main(argv)
    captured = capsys.readouterr()
    output = json.loads(captured.out) if captured.out.strip() else None
    return code, output, captured.err


def test_withdrawing_sets_a_published_builds_withdrawn_flag() -> None:
    client = FakeEmitterClient(hub_files=_two_listed_builds())

    outcome = _withdraw_function()(
        UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_OTHER_BUILD, client=client
    )

    assert client.calls.count("withdraw_build") == 1
    (withdrawal,) = client.release_requests
    assert (withdrawal.repository, withdrawal.release, withdrawal.build) == (
        UK_REPO_ID,
        UK_NATIONAL_RELEASE_ID,
        UK_OTHER_BUILD,
    )
    entry = _record(client)["releases"][UK_NATIONAL_RELEASE_ID]
    assert entry["builds"][UK_OTHER_BUILD]["withdrawn"] is True
    # Its folder stays, and the current build is unchanged.
    assert entry["current"] == UK_EARLIER_BUILD
    assert outcome.succeeded and not outcome.current
    assert outcome.as_json()["change"] == "withdraw"
    assert outcome.as_json()["withdrawn"] is True
    assert "is withdrawn" in outcome.explanation()


def test_a_withdrawal_names_the_withdrawn_build_for_the_collector() -> None:
    """The interface takes provenance naming the change's own release and
    build; the emitter then submits the releases.json the withdrawal wrote,
    naming the release's current build, which the collector's latest
    calibration tree follows."""
    client = FakeEmitterClient(hub_files=_two_listed_builds())

    outcome = _withdraw_function()(
        UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_OTHER_BUILD, client=client
    )

    (withdrawal,) = client.release_requests
    assert withdrawal.collector == CollectorProvenance(
        country_code="UK",
        build_id=UK_OTHER_BUILD,
        licensed_data=True,
        release=UK_NATIONAL_RELEASE_ID,
    )
    # This emitter has no collector row for it.
    assert client.collector_receipt_requests == [withdrawal.handover_id()]
    assert outcome.collector_receipt is None
    assert outcome.withdrawn


def test_a_withdrawal_waits_until_the_collector_accepts_the_new_releases_json(
    clock: Clock, capsys
) -> None:
    client = FakeEmitterClient(
        hub_files=_two_listed_builds(),
        collector_receipts=[{"status": "pending"}, {"status": "succeeded"}],
    )

    outcome = _withdraw_function()(
        UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_OTHER_BUILD, client=client
    )

    (withdrawal,) = client.release_requests
    assert client.collector_receipt_requests == [withdrawal.handover_id()] * 2
    assert clock.slept == [collector_submissions.COLLECTOR_RECEIPT_POLL_SECONDS]
    assert outcome.collector_receipt is not None
    assert outcome.collector_receipt["status"] == "succeeded"
    assert outcome.as_json()["collector_submission"] == {
        "status": "succeeded",
        "error_code": None,
    }
    assert outcome.withdrawn
    err = capsys.readouterr().err
    assert (
        f"releases.json with build {UK_OTHER_BUILD} of {UK_NATIONAL_RELEASE_ID} "
        "withdrawn is on Hugging Face; waiting up to 10 minutes for the "
        "collector to accept its submissions..." in err
    )
    assert "warning" not in err


def test_the_withdraw_command_says_why_a_release_without_a_current_build_sends_nothing(
    monkeypatch, capsys
) -> None:
    """The emitter submits nothing when the release has no current build
    after the change (``NO_CURRENT_BUILD``); the withdrawal still succeeds."""
    started, repository = _start_emitters(
        monkeypatch,
        {
            "collector_receipts": [
                {"status": "skipped", "error_code": "NO_CURRENT_BUILD"}
            ]
        },
    )
    repository.update(_two_listed_builds(with_current=False))

    code, output, error = _withdraw_command(
        [UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_OTHER_BUILD], capsys
    )

    assert code == 0, error
    assert output["withdrawn"] is True
    assert output["collector_submission"] == {
        "status": "skipped",
        "error_code": "NO_CURRENT_BUILD",
    }
    assert (
        "the collector receives nothing of releases.json with build "
        f"{UK_OTHER_BUILD} of {UK_NATIONAL_RELEASE_ID} withdrawn: the release has "
        "no current build after the change (NO_CURRENT_BUILD)." in error
    )
    (emitter,) = started
    assert emitter.calls.index("collector_submission_receipt") < (
        emitter.calls.index("close")
    )


def test_the_current_build_cannot_be_withdrawn() -> None:
    client = FakeEmitterClient(hub_files=_two_listed_builds())

    outcome = _withdraw_function()(
        UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_EARLIER_BUILD, client=client
    )

    assert not outcome.succeeded
    assert outcome.receipt["error_code"] == "WITHDRAWING_CURRENT_BUILD"
    assert outcome.as_json()["withdrawn"] is False
    assert "--promote" in outcome.explanation()
    entry = _record(client)["releases"][UK_NATIONAL_RELEASE_ID]
    assert entry["builds"][UK_EARLIER_BUILD]["withdrawn"] is False


def test_a_build_the_record_does_not_list_cannot_be_withdrawn() -> None:
    client = FakeEmitterClient(hub_files=_two_listed_builds())

    outcome = _withdraw_function()(
        UK_REPO_ID,
        UK_NATIONAL_RELEASE_ID,
        "uk-frs-calibration-attempt-20260101T000000Z-00000000",
        client=client,
    )

    assert not outcome.succeeded
    assert outcome.receipt["error_code"] == "BUILD_NOT_LISTED"
    assert "releases.json" in outcome.explanation()


def test_a_withdrawn_build_cannot_be_made_current() -> None:
    """Withdrawing the build first, then promoting it, is refused: a withdrawn
    build is never a release's current build."""
    files = _two_listed_builds()
    client = FakeEmitterClient(hub_files=files)
    _withdraw_function()(
        UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_OTHER_BUILD, client=client
    )

    from microcosm_emitter_interface import PromoteBuild

    promotion = release_publication._hand_over(
        PromoteBuild(
            repository=UK_REPO_ID, release=UK_NATIONAL_RELEASE_ID, build=UK_OTHER_BUILD
        ),
        client=client,
        wait_seconds=0,
        wanted_current=True,
    )

    assert not promotion.succeeded
    assert promotion.receipt["error_code"] == "CURRENT_WITHDRAWN"
    assert "withdrawn" in promotion.explanation()
    assert _record(client)["releases"][UK_NATIONAL_RELEASE_ID]["current"] == (
        UK_EARLIER_BUILD
    )


def test_the_withdraw_command_withdraws_and_prints_the_receipt(
    monkeypatch, capsys
) -> None:
    started, repository = _start_emitters(monkeypatch)
    repository.update(_two_listed_builds())

    code, output, error = _withdraw_command(
        [UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_OTHER_BUILD], capsys
    )

    assert code == 0, error
    assert output["change"] == "withdraw" and output["withdrawn"] is True
    assert (output["release"], output["build"]) == (
        UK_NATIONAL_RELEASE_ID,
        UK_OTHER_BUILD,
    )
    (emitter,) = started
    assert emitter.calls.count("withdraw_build") == 1 and emitter.closed
    assert "is withdrawn" in error


def test_the_withdraw_command_fails_on_a_refused_withdrawal(
    monkeypatch, capsys
) -> None:
    _, repository = _start_emitters(monkeypatch)
    repository.update(_two_listed_builds())

    code, output, error = _withdraw_command(
        [UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_EARLIER_BUILD], capsys
    )

    assert code == 1
    assert output["error_code"] == "WITHDRAWING_CURRENT_BUILD"
    assert "make another build current first" in error


@pytest.mark.parametrize("variable", OFFLINE_VARIABLES)
def test_the_withdraw_command_refuses_offline_mode_before_starting_an_emitter(
    monkeypatch, capsys, variable: str
) -> None:
    started, _ = _start_emitters(monkeypatch)
    monkeypatch.setenv(variable, "1")

    code, output, error = _withdraw_command(
        [UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_OTHER_BUILD], capsys
    )

    assert code == 1
    assert started == [] and output is None
    assert "refusing to withdraw" in error and variable in error


def test_the_withdraw_command_refuses_pointer_files_before_starting_an_emitter(
    monkeypatch, capsys, hub_listing: HubListing
) -> None:
    started, _ = _start_emitters(monkeypatch)
    hub_listing.files = ["releases.json", "latest.json"]

    code, output, error = _withdraw_command(
        [UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_OTHER_BUILD], capsys
    )

    assert code == 1
    assert started == [] and output is None
    assert "refusing to withdraw" in error and "latest.json" in error


def test_the_withdraw_command_checks_write_access_before_withdrawing(
    monkeypatch, capsys
) -> None:
    started, repository = _start_emitters(monkeypatch, {"write_access": NO_CREDENTIAL})
    repository.update(_two_listed_builds())

    code, output, error = _withdraw_command(
        [UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_OTHER_BUILD], capsys
    )

    (emitter,) = started
    assert code == 1
    assert "check_write_access" in emitter.calls
    assert emitter.release_requests == []
    assert output["status"] == "unavailable"
    assert "needs a Hugging Face write credential" in error


def test_a_promotion_of_an_unlisted_build_is_explained_as_unpublished() -> None:
    """A promotion takes its names from the build's own manifest, so a build
    releases.json does not list has not been published (or its publication
    has not landed); mistyped names are a withdrawal's problem."""
    from microcosm_emitter_interface import PromoteBuild

    client = FakeEmitterClient(hub_files=_two_listed_builds())
    promotion = release_publication._hand_over(
        PromoteBuild(
            repository=UK_REPO_ID,
            release=UK_NATIONAL_RELEASE_ID,
            build="uk-frs-calibration-attempt-20260101T000000Z-00000000",
        ),
        client=client,
        wait_seconds=0,
        wanted_current=True,
    )

    assert promotion.receipt["error_code"] == "BUILD_NOT_LISTED"
    explanation = promotion.explanation()
    assert "has not been published" in explanation
    assert "check the names" not in explanation


def test_only_a_withdrawal_reports_whether_the_build_is_withdrawn(
    national: Path,
    artifact_root: Path,  # noqa: F811
) -> None:
    """A publication or promotion cannot tell whether the build was withdrawn
    since, so its outcome reports ``withdrawn`` as unknown."""
    client = FakeEmitterClient()

    outcome = release_publication.publish_release_build(
        national, artifact_root=artifact_root, client=client
    )

    assert outcome.as_json()["change"] == "publish"
    assert outcome.as_json()["withdrawn"] is None


def test_withdrawing_a_withdrawn_build_again_succeeds() -> None:
    client = FakeEmitterClient(hub_files=_two_listed_builds())
    withdraw = _withdraw_function()
    withdraw(UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_OTHER_BUILD, client=client)

    again = withdraw(UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_OTHER_BUILD, client=client)

    assert again.withdrawn and again.succeeded


def test_the_withdraw_command_refuses_a_name_the_interface_refuses(
    monkeypatch, capsys
) -> None:
    started, _ = _start_emitters(monkeypatch)

    code, output, error = _withdraw_command(
        [UK_REPO_ID, UK_NATIONAL_RELEASE_ID, "not a build"], capsys
    )

    assert code == 1
    assert started == [] and output is None
    assert "refusing to withdraw" in error


def test_the_withdraw_command_fails_on_a_withdrawal_still_pending(
    monkeypatch, capsys
) -> None:
    _start_emitters(
        monkeypatch,
        {
            "release_receipts": [
                {"status": "pending", "error_code": "REPOSITORY_UNAVAILABLE"}
            ]
        },
    )

    code, output, error = _withdraw_command(
        [UK_REPO_ID, UK_NATIONAL_RELEASE_ID, UK_OTHER_BUILD], capsys
    )

    assert code == 1
    assert output["status"] == "pending" and output["withdrawn"] is False
    assert "still trying (REPOSITORY_UNAVAILABLE)" in error
