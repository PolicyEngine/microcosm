"""Publishing a UK release build through the emitter (hand-over step H2c).

The emitter's release writer puts each build in its own folder and records
it in ``releases.json``; Microcosm hands the build over with
``publish_release`` and treats it as current only when the receipt says so.
These tests use the in-memory client in ``test_support.microcosm_build``; the
emitter's own tests cover the writer, its queue and the Hub.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from microcosm_emitter_interface.releases import ReleaseRecordError, ReleasesRecord

from microcosm.build import release_publication
from microcosm.data import release as release_module
from microcosm.data.releases_record import parse_releases_record
from test_support.microcosm_build.emitter import FakeEmitterClient
from test_support.microcosm_data.build_folders import UK_NATIONAL_BUILD
from test_support.microcosm_data.release import (
    _as_uk_national_build_folder_release,
    artifact_root,  # noqa: F401  (a fixture)
    release_dir,  # noqa: F401  (a fixture)
)

UK_REPO_ID = "policyengine/populace-uk-private"
UK_NATIONAL_RELEASE_ID = "microcosm-uk-2024-25-national"
UK_EARLIER_BUILD = "uk-frs-calibration-attempt-20260901T000000Z-0badcafe"
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


@pytest.fixture(autouse=True)
def _hub_online(monkeypatch) -> None:
    """Run each test as on a build machine that can reach the Hub.

    Local test runs set ``HF_HUB_OFFLINE``, and the command refuses to start
    an emitter with it set. Every emitter here is the in-memory client, and
    the repository's root fixture refuses to start a real one.
    """
    for name in OFFLINE_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def national(release_dir: Path, monkeypatch) -> Path:  # noqa: F811
    """A UK national build; the release contract's own tests cover its checks."""
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: []
    )
    return _as_uk_national_build_folder_release(release_dir)


def _record(client: FakeEmitterClient) -> dict:
    return json.loads(client.hub_files[(UK_REPO_ID, "releases.json")])


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
    assert "without --make-current" in outcome.explanation()
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
    (_, promotion) = client.release_requests
    assert (promotion.release, promotion.build) == (
        UK_NATIONAL_RELEASE_ID,
        UK_NATIONAL_BUILD,
    )
    assert _record(client)["releases"][UK_NATIONAL_RELEASE_ID]["current"] == (
        UK_NATIONAL_BUILD
    )
    assert outcome.current and outcome.succeeded


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


def test_a_promotion_refused_for_pointer_files_says_to_promote_again_later(
    national: Path,
    artifact_root: Path,  # noqa: F811
    monkeypatch,
    capsys,
) -> None:
    """The build is published already, so publishing it again cannot help."""
    _, repository = _start_emitters(monkeypatch)
    repository[(UK_REPO_ID, "latest-national.json")] = b"{}"
    assert _run([str(national), "--artifact-root", str(artifact_root)], capsys)[0] == 0

    code, output, error = _run([str(national), "--promote"], capsys)

    assert code == 1
    assert (output["status"], output["error_code"]) == (
        "refused",
        "POINTER_FILES_PRESENT",
    )
    assert "without --make-current" not in error
    assert "stays published but is not current" in error
    assert "Run this --promote command again once the pointer files" in error
    record = json.loads(repository[(UK_REPO_ID, "releases.json")])
    assert record["releases"][UK_NATIONAL_RELEASE_ID]["current"] is None


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
    with pytest.raises(SystemExit):
        release_publication.main(
            [str(release_dir), "--artifact-root", str(artifact_root)]
        )
    assert "tools/publish_release.sh" in capsys.readouterr().err


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
