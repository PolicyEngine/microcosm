"""tools/resolve_uk_reference.py: a line's current build, checked or fetched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcosm.build.uk_runtime.cut_reference import read_uk_cut_reference
from test_support.microcosm_build.uk_cut_reference import (
    UK_DENSE_RELEASE,
    UK_NATIONAL_RELEASE,
    UKReferenceHub,
    sha256_of,
    stub_reference_evidence,
)
from tools import resolve_uk_reference as tool

DENSE_BUILD = "uk-local-candidate-f100-s42-20261107T173210Z-38605444"
NATIONAL_BUILD = "uk-frs-calibration-attempt-20261107T020434Z-eeb6093e"


@pytest.fixture
def hub(tmp_path: Path) -> UKReferenceHub:
    hub = UKReferenceHub(tmp_path / "hub-root")
    hub.publish(UK_NATIONAL_RELEASE, NATIONAL_BUILD, dataset=b"national")
    hub.publish(UK_DENSE_RELEASE, DENSE_BUILD, dataset=b"dense")
    return hub


def test_check_prints_the_current_build_and_downloads_no_dataset(hub, capsys) -> None:
    assert tool.main(["--release", UK_DENSE_RELEASE, "--check"], hub_download=hub) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["build"] == DENSE_BUILD
    assert not any(call.endswith(".h5") for call in hub.calls)


def test_output_dir_keeps_the_dataset_and_its_record(hub, tmp_path: Path) -> None:
    out = tmp_path / "reference"

    assert (
        tool.main(
            ["--release", UK_DENSE_RELEASE, "--output-dir", str(out)],
            hub_download=hub,
            write_evidence=stub_reference_evidence,
        )
        == 0
    )

    reference = read_uk_cut_reference(out / "reference.json")
    assert reference.build == DENSE_BUILD
    assert (out / reference.filename).read_bytes() == b"dense"
    assert reference.sha256 == sha256_of(b"dense")
    assert (out / "input_mass_reference.json").exists()


def test_a_directory_that_holds_a_reference_is_refused(
    hub, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "reference"
    args = ["--release", UK_NATIONAL_RELEASE, "--output-dir", str(out)]
    stub = stub_reference_evidence
    assert tool.main(args, hub_download=hub, write_evidence=stub) == 0

    assert tool.main(args, hub_download=hub, write_evidence=stub) == 1
    assert "keeps the reference it started with" in capsys.readouterr().err


def test_a_release_without_a_certified_current_build_is_refused(
    tmp_path: Path, capsys
) -> None:
    hub = UKReferenceHub(tmp_path)
    hub.publish(
        UK_NATIONAL_RELEASE,
        NATIONAL_BUILD,
        dataset=b"national",
        evidence={"kind": "uk_release_certification", "shippable": False},
    )

    assert (
        tool.main(["--release", UK_NATIONAL_RELEASE, "--check"], hub_download=hub) == 1
    )
    assert "refused" in capsys.readouterr().err


def test_offline_mode_is_refused(monkeypatch, capsys) -> None:
    """A cached releases.json can name a build that is no longer current (the
    review of microcosm#1574); run_uk_cut refuses offline mode too."""

    from microcosm.build import staging_dataset

    monkeypatch.setattr(staging_dataset, "hub_offline", lambda: True)
    assert tool.main(["--release", "microcosm-uk-2024-25-national", "--check"]) == 1
    assert "HF_HUB_OFFLINE" in capsys.readouterr().err


def test_a_build_can_be_pinned_from_the_command_line(tmp_path, capsys) -> None:
    hub = UKReferenceHub(tmp_path / "hub-root")
    hub.publish(UK_NATIONAL_RELEASE, "uk-frs-calibration-attempt-older", dataset=b"o")
    hub.publish(UK_NATIONAL_RELEASE, "uk-frs-calibration-attempt-newer", dataset=b"n")
    assert (
        tool.main(
            [
                "--release",
                UK_NATIONAL_RELEASE,
                "--build",
                "uk-frs-calibration-attempt-older",
                "--check",
            ],
            hub_download=hub,
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["build"] == "uk-frs-calibration-attempt-older"
    assert summary["selection"] == "pinned"
