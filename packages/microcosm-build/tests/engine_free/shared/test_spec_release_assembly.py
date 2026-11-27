"""Assembling a spec country's transport build into a release build (5b · 13).

``microcosm.build.spec_release_assembly`` reads what a transport build wrote
under ``--out`` (its H5, export descriptor, calibration diagnostics, run
manifest and ``build-status.json``), as the UK national assembler reads its
candidate's outputs, and writes the release directory that
``microcosm-publish-release-build`` publishes: BE's six contract files, with
the H5 and its weights beside the build. These tests use fixture build
folders; no build runs and nothing reaches the Hub.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from microcosm.build import release_publication, spec_release_assembly
from microcosm.build.spec_release_assembly import (
    SpecReleaseAssemblyError,
    assemble_spec_release,
    main,
)
from microcosm.build.transport.cli import EXPORT_FILENAME
from microcosm.build.transport.compose import load_transport_spec
from microcosm.data.contract import (
    BE_REQUIRED_RELEASE_FILES,
    ReleaseContractError,
    validate_release_dir,
)
from microcosm.data.release import publication_refusal
from test_support.microcosm_build.emitter import FakeEmitterClient
from test_support.microcosm_build.spec_release import (
    BUILD_COMMIT,
    BUILD_RUNTIME,
    CLEAN_CODE,
    FINISHED_STAMP,
    GATE_NODES,
    HOUSEHOLD_WEIGHTS,
    demography_references,
    run_manifest,
    write_transport_build_folder,
)

CREATED_AT = "2026-12-02T08:00:00+00:00"
#: The files BE's release contract adds to the base three, by artifact key.
ADDED_FILES = {
    "source_coverage": "source_coverage.json",
    "demographics": "demographics.json",
    "reform_validation": "reform_validation.json",
}


@pytest.fixture(scope="module")
def be_spec() -> dict:
    return load_transport_spec("be")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _expected_build(be_spec: dict, period: int = 2026) -> str:
    chronicle = be_spec["resources"]["chronicle_feed"]["source_commit"]
    return f"microcosm-be-{period}-chronicle-{chronicle[:7]}-{FINISHED_STAMP}"


def test_a_be_build_assembles_into_its_release_with_all_six_files(
    tmp_path: Path, be_spec: dict
) -> None:
    build = write_transport_build_folder(tmp_path / "out")
    releases = tmp_path / "releases"

    assembled = assemble_spec_release(
        build.path, country="be", out_dir=releases, created_at=CREATED_AT
    )

    # The build's population year picks the release, and the build is named
    # as BE's builder names it: year, Chronicle pin, when the run finished.
    assert assembled.release == "microcosm-be-2026-national"
    assert assembled.build == _expected_build(be_spec)
    assert assembled.release_dir == releases / "microcosm-be-2026-national"
    assert sorted(path.name for path in assembled.release_dir.iterdir()) == sorted(
        BE_REQUIRED_RELEASE_FILES
    )
    validate_release_dir(assembled.release_dir)
    manifest = _read(assembled.release_dir / "release_manifest.json")
    for key, filename in ADDED_FILES.items():
        assert manifest["artifacts"][key]["path"] == filename
        assert manifest["artifacts"][key]["role"] == key
        assert manifest["artifacts"][key]["sha256"] == _sha256(
            assembled.release_dir / filename
        )
    assert (manifest["release"], manifest["build"]) == (
        assembled.release,
        assembled.build,
    )
    assert manifest["default_datasets"] == {"national": "populace_be_2026"}


def test_the_dataset_and_its_weights_land_beside_the_build(tmp_path: Path) -> None:
    build = write_transport_build_folder(tmp_path / "out")

    assembled = assemble_spec_release(
        build.path, country="be", out_dir=tmp_path / "releases", created_at=CREATED_AT
    )

    # The publisher reads each declared file from the artifact root under
    # the name the manifest gives it, as for the UK's candidate H5.
    assert assembled.artifact_root == build.path
    dataset = build.path / "populace_be_2026.h5"
    assert dataset.read_bytes() == (build.path / EXPORT_FILENAME).read_bytes()
    weights = np.load(build.path / "populace_be_2026_calibration.npz")
    assert weights["household_weight"].tolist() == list(HOUSEHOLD_WEIGHTS)
    assert weights["household_id"].tolist() == [1, 2, 3]
    manifest = _read(assembled.release_dir / "release_manifest.json")
    assert manifest["artifacts"]["populace_be_2026"]["sha256"] == _sha256(dataset)
    assert manifest["artifacts"]["populace_be_2026"]["kind"] == "microdata"
    assert manifest["artifacts"]["populace_be_2026_calibration"]["sha256"] == (
        _sha256(build.path / "populace_be_2026_calibration.npz")
    )


def test_the_build_manifest_records_what_the_build_ran_on(
    tmp_path: Path, be_spec: dict
) -> None:
    build = write_transport_build_folder(tmp_path / "out")

    assembled = assemble_spec_release(
        build.path, country="be", out_dir=tmp_path / "releases", created_at=CREATED_AT
    )

    assert (assembled.release_dir / "build_manifest.json").is_file()
    manifest = _read(assembled.release_dir / "build_manifest.json")
    assert (manifest["release"], manifest["build"]) == (
        assembled.release,
        assembled.build,
    )
    assert manifest["code"] == CLEAN_CODE
    assert manifest["build_sha"] == BUILD_COMMIT[:7]
    assert manifest["runtime"] == BUILD_RUNTIME
    packages = {name: version for name, version in BUILD_RUNTIME.items()}
    python = packages.pop("python")
    assert manifest["engines"] == {
        "axiom_rules_engine": {"python": python, "packages": packages}
    }
    assert manifest["gates"]["outcomes"] == dict.fromkeys(GATE_NODES, "pass")
    assert manifest["gates"]["known_failures"] == []
    assert manifest["population_year"] == 2026
    assert (
        manifest["chronicle"]["commit"]
        == (be_spec["resources"]["chronicle_feed"]["source_commit"])
    )
    assert manifest["created_at"] == CREATED_AT
    assert manifest["spec"]["resource_hashes"] == be_spec["resource_hashes"]
    # The run manifest's start is that of the command's last pass over the
    # stored graph, not of the build, so only its key and end are recorded.
    assert set(manifest["run"]) == {"key", "finished_at"}
    assert manifest["run"]["finished_at"] == "2026-12-01T10:20:30+00:00"
    assert manifest["dataset"] == {
        "filename": "populace_be_2026.h5",
        "sha256": _sha256(build.path / EXPORT_FILENAME),
    }


@pytest.fixture
def on_a_build_machine(monkeypatch) -> None:
    """Run as on a machine that can reach the Hub, with a repository that has
    switched to ``releases.json``: no pointer file is listed."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        release_publication, "_repository_files", lambda _repository: []
    )


@pytest.mark.usefixtures("on_a_build_machine")
def test_an_assembled_be_build_publishes_through_the_emitter(
    tmp_path: Path, be_spec: dict
) -> None:
    build = write_transport_build_folder(tmp_path / "out")
    assembled = assemble_spec_release(
        build.path, country="be", out_dir=tmp_path / "releases", created_at=CREATED_AT
    )
    client = FakeEmitterClient()
    assert assembled.release_dir.is_dir()

    outcome = release_publication.publish_release_build(
        assembled.release_dir, artifact_root=assembled.artifact_root, client=client
    )

    (request,) = client.release_requests
    assert (
        request.repository
        == be_spec["resources"]["release_contract"]["hf"]["artifact_repo"]
    )
    assert (request.release, request.build) == (assembled.release, assembled.build)
    roles = {item.name: item.role for item in request.files}
    for key, filename in ADDED_FILES.items():
        assert roles[filename] == key
    assert roles["populace_be_2026.h5"] == "dataset"
    assert roles["populace_be_2026_calibration.npz"] == "calibration_weights"
    assert outcome.published and outcome.succeeded


def test_the_printed_publish_command_passes_the_publishers_own_checks(
    tmp_path: Path,
) -> None:
    # microcosm-publish-release-build runs these checks before anything else,
    # with only the flags the printed command passes (none of them).
    build = write_transport_build_folder(tmp_path / "out")
    assembled = assemble_spec_release(
        build.path, country="be", out_dir=tmp_path / "releases", created_at=CREATED_AT
    )

    assert "--allow" not in assembled.publish_command
    assert publication_refusal(assembled.release_dir) is None


@pytest.mark.parametrize(
    ("folder_options", "refusal"),
    [
        ({"not_yet_produced": ["The rent ageing ruling"]}, "still owes"),
        ({"code": None}, "commit"),
        ({"code": {**CLEAN_CODE, "git_dirty": True}}, "dirty"),
        ({"code": {**CLEAN_CODE, "git_dirty": None}}, "dirty"),
        ({"code": {**CLEAN_CODE, "git_commit": "abc1234"}}, "commit"),
        ({"runtime": None}, "runtime"),
        ({"runtime": {"python": "3.13.5"}}, "microcosm-data"),
        ({"gate_outcome": "fail"}, "gate"),
        ({"period": 2024}, "microcosm-be-2024-national"),
    ],
    ids=[
        "owed-outputs",
        "no-code-record",
        "dirty-checkout",
        "unmeasured-checkout",
        "short-commit",
        "no-runtime-record",
        "no-data-package-version",
        "failed-gate",
        "year-without-a-release",
    ],
)
def test_assembly_refuses_a_build_it_cannot_publish_and_writes_nothing(
    tmp_path: Path, folder_options: dict, refusal: str
) -> None:
    build = write_transport_build_folder(tmp_path / "out", **folder_options)
    releases = tmp_path / "releases"
    before = sorted(path.name for path in build.path.iterdir())

    with pytest.raises(SpecReleaseAssemblyError, match=refusal):
        assemble_spec_release(
            build.path, country="be", out_dir=releases, created_at=CREATED_AT
        )

    assert not releases.exists() or not any(releases.iterdir())
    assert sorted(path.name for path in build.path.iterdir()) == before


def test_assembly_refuses_a_dataset_that_changed_after_the_build(
    tmp_path: Path,
) -> None:
    build = write_transport_build_folder(tmp_path / "out")
    dataset = build.path / EXPORT_FILENAME
    dataset.write_bytes(dataset.read_bytes() + b"\0")

    with pytest.raises(SpecReleaseAssemblyError, match=EXPORT_FILENAME):
        assemble_spec_release(build.path, country="be", out_dir=tmp_path / "releases")


def test_assembly_refuses_an_export_descriptor_that_changed_after_the_build(
    tmp_path: Path,
) -> None:
    # The descriptor names the weight entity whose weights the NPZ carries,
    # so it must be the one the build wrote.
    build = write_transport_build_folder(tmp_path / "out")
    descriptor_path = build.path / "export-descriptor.json"
    descriptor = _read(descriptor_path)
    descriptor["bindings"] = {"changed": "after the build"}
    descriptor_path.write_text(json.dumps(descriptor))

    with pytest.raises(SpecReleaseAssemblyError, match="export-descriptor.json"):
        assemble_spec_release(build.path, country="be", out_dir=tmp_path / "releases")


def test_assembly_refuses_a_run_manifest_that_changed_after_the_build(
    tmp_path: Path,
) -> None:
    # The run manifest decides the gate refusal and the build's name, so it
    # must be the one the build wrote, not merely a valid one.
    build = write_transport_build_folder(tmp_path / "out")
    (build.path / "manifest.json").write_bytes(
        run_manifest("be", gate_outcome="not_applicable").to_json_bytes()
    )

    with pytest.raises(SpecReleaseAssemblyError, match="manifest.json"):
        assemble_spec_release(build.path, country="be", out_dir=tmp_path / "releases")


def test_assembly_refuses_a_build_that_left_out_an_active_reference(
    tmp_path: Path,
) -> None:
    omitted = demography_references("be")[0]["name"]
    build = write_transport_build_folder(tmp_path / "out", omit_targets=[omitted])

    # The source coverage gate is a release gate: a hard-target reference the
    # build did not calibrate stops the assembly.
    with pytest.raises(SpecReleaseAssemblyError, match=omitted):
        assemble_spec_release(build.path, country="be", out_dir=tmp_path / "releases")


def test_assembly_refuses_a_build_of_another_spec(tmp_path: Path) -> None:
    # The release files, the build's name and the source coverage check read
    # the spec the assembler loads, so it must be the spec the build ran on.
    build = write_transport_build_folder(
        tmp_path / "out", spec_hash_changes={"target_references": "0" * 64}
    )

    with pytest.raises(SpecReleaseAssemblyError, match="target_references"):
        assemble_spec_release(build.path, country="be", out_dir=tmp_path / "releases")


def test_assembly_refuses_a_build_that_records_no_spec(tmp_path: Path) -> None:
    build = write_transport_build_folder(tmp_path / "out", record_spec=False)

    with pytest.raises(SpecReleaseAssemblyError, match="spec"):
        assemble_spec_release(build.path, country="be", out_dir=tmp_path / "releases")


def test_a_refusal_by_the_release_contract_leaves_nothing_behind(
    tmp_path: Path, monkeypatch
) -> None:
    build = write_transport_build_folder(tmp_path / "out")
    releases = tmp_path / "releases"
    before = sorted(path.name for path in build.path.iterdir())

    def refuse(release_dir, **_options):
        raise ReleaseContractError(Path(release_dir), ["fixture refusal"])

    monkeypatch.setattr(spec_release_assembly, "validate_release_dir", refuse)

    with pytest.raises(SpecReleaseAssemblyError, match="fixture refusal"):
        assemble_spec_release(build.path, country="be", out_dir=releases)

    # The staged directory, the H5 copy and the weights are all removed.
    assert list(releases.iterdir()) == []
    assert sorted(path.name for path in build.path.iterdir()) == before


def test_assembly_refuses_to_replace_an_earlier_assembly(tmp_path: Path) -> None:
    build = write_transport_build_folder(tmp_path / "out")
    releases = tmp_path / "releases"
    assembled = assemble_spec_release(build.path, country="be", out_dir=releases)
    assert (assembled.release_dir / "release_manifest.json").is_file()
    manifest = (assembled.release_dir / "release_manifest.json").read_bytes()

    with pytest.raises(SpecReleaseAssemblyError, match="already exists"):
        assemble_spec_release(build.path, country="be", out_dir=releases)

    assert (assembled.release_dir / "release_manifest.json").read_bytes() == manifest


def test_the_command_prints_the_publication_and_promotion_commands(
    tmp_path: Path, capsys
) -> None:
    build = write_transport_build_folder(tmp_path / "out")
    releases = tmp_path / "releases"

    exit_code = main(
        ["--build-dir", str(build.path), "--country", "be", "--out-dir", str(releases)]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    release_dir = releases / "microcosm-be-2026-national"
    assert summary["release_dir"] == str(release_dir)
    assert "microcosm-publish-release-build" in summary["publish_command"]
    assert f"--artifact-root {build.path}" in summary["publish_command"]
    assert "--promote" not in summary["publish_command"]
    assert "--promote" in summary["promote_command"]


def test_the_command_exits_one_on_a_refused_build(tmp_path: Path, capsys) -> None:
    build = write_transport_build_folder(
        tmp_path / "out", not_yet_produced=["An owed output"]
    )

    exit_code = main(
        [
            "--build-dir",
            str(build.path),
            "--country",
            "be",
            "--out-dir",
            str(tmp_path / "releases"),
        ]
    )

    assert exit_code == 1
    assert "still owes" in capsys.readouterr().err
