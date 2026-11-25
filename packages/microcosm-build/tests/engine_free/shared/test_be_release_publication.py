"""``microcosm-publish-release-build`` for BE's releases (tracker 5b · 3).

BE's builds publish into BE's two releases through this command, as the UK's
do into the UK releases, and each publication, promotion and withdrawal
names its build for the collector as a BE build. BE's release repository is
private, so, as for a UK build, the collector keeps the Orrery graph files
of a BE build private. These tests never start an emitter or reach the Hub.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from microcosm_emitter_interface import CollectorProvenance

from microcosm.build import release_publication
from test_support.microcosm_build.emitter import FakeEmitterClient
from test_support.microcosm_data.build_folders import BE_BUILD, BE_RELEASE_IDS
from test_support.microcosm_data.release import (
    BE_FIXTURE_REPO_ID,
    _as_be_build_folder_release,
    artifact_root,  # noqa: F401  (a fixture)
    release_dir,  # noqa: F401  (a fixture)
)

#: The variables ``huggingface_hub`` reads to refuse every Hub request.
OFFLINE_VARIABLES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
#: The build the receipts of the promotion and withdrawal tests name as
#: current.
BE_CURRENT_BUILD = "microcosm-be-2025-chronicle-0000000-20261101T000000Z"


def _be_build(release: str, build: str = BE_BUILD) -> CollectorProvenance:
    """The collector's provenance of a build of a BE release."""
    return CollectorProvenance(
        country_code="BE", build_id=build, licensed_data=True, release=release
    )


@pytest.fixture
def on_a_build_machine(monkeypatch) -> None:
    """Run as on a machine that can reach the Hub, with a repository that has
    switched to ``releases.json``: no pointer file is listed."""
    for name in OFFLINE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        release_publication, "_repository_files", lambda _repository: []
    )


def test_the_command_takes_a_be_release(tmp_path, monkeypatch, capsys) -> None:
    """A directory named after a BE release passes the command's first check;
    with the Hub offline the command then refuses before any emitter starts."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    be_release_dir = tmp_path / BE_RELEASE_IDS[0]
    be_release_dir.mkdir()

    try:
        exit_code = release_publication.main([str(be_release_dir), "--promote"])
    except SystemExit as refusal:
        # The command's argument parser refuses a release it does not take.
        exit_code = refusal.code

    error_output = capsys.readouterr().err
    assert "does not publish into build folders" not in error_output
    assert exit_code == 1
    assert "HF_HUB_OFFLINE" in error_output


@pytest.mark.parametrize("release", BE_RELEASE_IDS)
def test_every_be_release_names_its_builds_as_licensed_be_builds(release) -> None:
    """The collector registers BE under ``be`` with its private release
    repository, and keeps a licensed build's Orrery graph files private."""
    assert release_publication.collector_provenance(release, BE_BUILD) == _be_build(
        release
    )


@pytest.mark.usefixtures("on_a_build_machine")
def test_a_be_publication_hands_its_files_over_and_names_its_build(
    release_dir: Path,  # noqa: F811
    artifact_root: Path,  # noqa: F811
) -> None:
    be_release_dir = _as_be_build_folder_release(release_dir)
    client = FakeEmitterClient()

    outcome = release_publication.publish_release_build(
        be_release_dir, artifact_root=artifact_root, client=client
    )

    (request,) = client.release_requests
    assert (request.repository, request.release, request.build) == (
        BE_FIXTURE_REPO_ID,
        BE_RELEASE_IDS[0],
        BE_BUILD,
    )
    assert (request.release_role, request.make_current) == ("national", False)
    files = {item.name: item.role for item in request.files}
    assert files["calibration_diagnostics.json"] == "calibration_diagnostics"
    assert {"source_coverage.json", "demographics.json", "reform_validation.json"} <= (
        set(files)
    )
    assert request.collector == _be_build(BE_RELEASE_IDS[0])
    record = json.loads(client.hub_files[(BE_FIXTURE_REPO_ID, "releases.json")])
    assert list(record["releases"][BE_RELEASE_IDS[0]]["builds"]) == [BE_BUILD]
    assert outcome.published and outcome.succeeded


@pytest.mark.usefixtures("on_a_build_machine")
def test_a_be_promotion_names_its_build_for_the_collector(
    tmp_path, monkeypatch
) -> None:
    """The emitter submits the releases.json the promotion wrote under the BE
    country code, as for the build's publication."""
    release = BE_RELEASE_IDS[1]
    be_release_dir = tmp_path / release
    be_release_dir.mkdir()
    monkeypatch.setattr(
        release_publication,
        "build_folder_identity",
        lambda _path: (BE_FIXTURE_REPO_ID, release, BE_BUILD),
    )
    client = FakeEmitterClient(release_receipts=[{"current": BE_BUILD}])

    outcome = release_publication.promote_release_build(be_release_dir, client=client)

    (promotion,) = client.release_requests
    assert promotion.collector == _be_build(release)
    assert client.collector_receipt_requests == [promotion.handover_id()]
    assert outcome.current


@pytest.mark.usefixtures("on_a_build_machine")
def test_a_be_withdrawal_names_its_build_for_the_collector() -> None:
    release = BE_RELEASE_IDS[0]
    client = FakeEmitterClient(release_receipts=[{"current": BE_CURRENT_BUILD}])

    outcome = release_publication.withdraw_release_build(
        BE_FIXTURE_REPO_ID, release, BE_BUILD, client=client
    )

    (withdrawal,) = client.release_requests
    assert withdrawal.collector == _be_build(release)
    assert client.collector_receipt_requests == [withdrawal.handover_id()]
    assert outcome.withdrawn
