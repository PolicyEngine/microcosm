"""Staging a UK build's outputs once they are copied back from a Modal run.

The Modal runner runs every tool with ``HF_HUB_OFFLINE=1``, so a build there
uploads nothing: it verifies its bundle, writes the sidecars and records the
bundle as ``local_only``. After ``modal volume get`` copies the run back,
``tools/stage_uk_rowwise_candidate.py --modal-receipt`` checks every output
the manifest registers against the run's receipt, then hands the bundle to the
operator's emitter with the code that re-stages a local build that kept its
bundle local.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from microcosm.build import staging_dataset
from test_support.microcosm_build.emitter import FakeEmitterClient, FakeHandoverQueue
from test_support.microcosm_build.modal_copy_back import (
    copy_back,
    rewrite_receipt,
    write_receipt,
)
from test_support.microcosm_build.uk_full_build_cli import (
    STEM,
    _edit_an_output,
    _remote_hub,
    load_tool,
    run_dense_main,
    single_run_id,
)

PRIVATE_REPOSITORY = "policyengine/populace-uk-private"


def _modal_build(tmp_path, monkeypatch):
    """A dense build run as the Modal runner runs it, then copied back.

    Returns the Hub, the copied-back run directory and the stage's receipt.
    """
    pytest.importorskip("tables")
    hub = _remote_hub(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    status, out = run_dense_main(tmp_path, monkeypatch, staging=None)
    assert status == 0
    monkeypatch.delenv("HF_HUB_OFFLINE")
    assert hub.paths(PRIVATE_REPOSITORY) == []
    copied, receipt = copy_back(out, tmp_path / "modal-runs")
    return hub, copied, receipt


def _workstation_emitters(monkeypatch, stager, **options):
    """Every emitter the command starts on the workstation, in order."""
    started: list[FakeEmitterClient] = []

    def start():
        started.append(FakeEmitterClient(**options))
        return started[-1]

    monkeypatch.setattr(stager, "_start_bundle_emitter", start)
    return started


def _staging_argv(copied: Path, receipt: Path, *extra: str) -> list[str]:
    return [
        "--run-dir",
        str(copied),
        "--modal-receipt",
        str(receipt),
        "--wait-seconds",
        "60",
        *extra,
    ]


def _manifest(copied: Path) -> dict:
    stager = load_tool("stage_uk_rowwise_candidate")
    return json.loads((copied / stager.MANIFEST_FILENAME).read_text())


def _files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_copied_back_outputs_are_handed_over_as_a_local_build_is_restaged(
    tmp_path, monkeypatch, capsys
):
    """The workstation's emitter stages the bundle the Modal build verified,
    under the build's own run id, and its receipt becomes the manifest's
    record exactly as when a local build is re-staged."""
    hub, copied, receipt = _modal_build(tmp_path, monkeypatch)
    at_build = _manifest(copied)["staged_dataset"]
    assert (at_build["mode"], at_build["status"]) == ("local_only", "skipped")
    stager = load_tool("stage_uk_rowwise_candidate")
    started = _workstation_emitters(
        monkeypatch, stager, hub_files=hub.files, hub_api=hub
    )
    handed_over = []
    shared_stage_bundle = stager.stage_bundle

    def stage_bundle(bundle, **options):
        handed_over.append((bundle, options))
        return shared_stage_bundle(bundle, **options)

    monkeypatch.setattr(stager, "stage_bundle", stage_bundle)
    capsys.readouterr()

    assert stager.main(_staging_argv(copied, receipt)) == 0

    # One emitter of the command's own, closed when it finished.
    (client,) = started
    assert client.closed
    (request,) = client.bundle_requests
    roles = {item.name: item.role for item in request.files}
    assert roles[f"{STEM}.h5"] == "dataset"
    assert roles[stager.MANIFEST_FILENAME] == "build_manifest"
    assert roles["staged_manifest.json"] == "staged_manifest"
    assert roles["sha256sums.txt"] == "checksums"
    # Through the re-stage command's own hand-over.
    ((bundle, options),) = handed_over
    assert bundle.run_dir == copied.resolve()
    assert options["repository"] == PRIVATE_REPOSITORY
    assert options["wait_seconds"] == 60.0
    run_id = single_run_id(copied)
    manifest = _manifest(copied)
    staged = manifest["staged_dataset"]
    assert staged == staging_dataset.validate_staged_dataset_delivery(staged)
    assert (staged["status"], staged["error_code"]) == ("uploaded", None)
    assert staged["revision"] == hub.sha
    assert (staged["repository"], staged["prefix"], staged["run_id"]) == (
        PRIVATE_REPOSITORY,
        f"staged/{run_id}",
        run_id,
    )
    assert staged["files"] == bundle.digests()
    assert manifest["staged_dataset_at_build"] == {
        "mode": "local_only",
        "status": "skipped",
        "error_code": None,
    }
    record = manifest["staged_dataset_handover"]
    assert record["run_dir"] == str(copied.resolve())
    assert record["repository"] == PRIVATE_REPOSITORY
    captured = capsys.readouterr()
    assert json.loads(captured.out) == staged
    # The Modal build's staging run and Logbook row keep what it recorded.
    assert "still records skipped" in captured.err
    assert "phase dataset_stage_skipped" in captured.err
    assert len(hub.commits) == 1
    assert sorted(hub.paths(PRIVATE_REPOSITORY)) == sorted(
        f"staged/{run_id}/{name}" for name in roles
    )
    for line in (copied / "sha256sums.txt").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((copied / name).read_bytes()).hexdigest() == digest


def _edit_after_copy_back(copied, run_root, receipt):
    _edit_an_output(copied)


def _change_without_the_manifest(copied, run_root, receipt):
    dataset = copied / f"{STEM}.h5"
    dataset.write_bytes(dataset.read_bytes() + b"\0")


def _drop_the_dataset_from_the_receipt(copied, run_root, receipt):
    rewrite_receipt(
        receipt,
        lambda content: content.update(
            outputs=[
                item
                for item in content["outputs"]
                if not item["path"].endswith(f"{STEM}.h5")
            ]
        ),
    )


def _unfinished_stage(copied, run_root, receipt):
    write_receipt(run_root, status="FAILED")


def _stage_stopped_at_its_budget(copied, run_root, receipt):
    write_receipt(run_root, status="FAILED", stopped_at_budget=True)


def _another_kind_of_file(copied, run_root, receipt):
    receipt.write_text(json.dumps({"schema": "microcosm-modal-us-stage-plan/1"}))


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        pytest.param(
            _edit_after_copy_back,
            r"is not the file the Modal run wrote",
            id="an output edited after the copy-back",
        ),
        pytest.param(
            _change_without_the_manifest,
            r"does not match its manifest digest",
            id="an output that no longer matches the manifest",
        ),
        pytest.param(
            _drop_the_dataset_from_the_receipt,
            rf"the receipt lists no out/{STEM}\.h5",
            id="an output the receipt does not list",
        ),
        pytest.param(
            _unfinished_stage,
            r"status is 'FAILED', not 'COMPLETED'",
            id="a stage that did not finish",
        ),
        pytest.param(
            _stage_stopped_at_its_budget,
            r"stopped at its budget",
            id="a stage stopped at its budget",
        ),
        pytest.param(
            _another_kind_of_file,
            r"is not a microcosm-modal-us-stage-receipt/1",
            id="a file that is not a receipt",
        ),
    ],
)
def test_a_copied_back_directory_that_does_not_verify_is_refused(
    tmp_path, monkeypatch, change, refusal
):
    """Nothing is handed over, no emitter is started and no file changes
    unless every output the manifest registers is the file the finished
    stage wrote."""
    hub, copied, receipt = _modal_build(tmp_path, monkeypatch)
    change(copied, receipt.parent.parent, receipt)
    before = _files(copied)
    stager = load_tool("stage_uk_rowwise_candidate")
    started = _workstation_emitters(
        monkeypatch, stager, hub_files=hub.files, hub_api=hub
    )

    with pytest.raises(SystemExit, match=refusal):
        stager.main(_staging_argv(copied, receipt))

    assert started == []
    assert _files(copied) == before
    assert hub.paths(PRIVATE_REPOSITORY) == []


def test_a_directory_outside_the_receipts_state_is_refused(tmp_path, monkeypatch):
    """The receipt describes the run's copied-back state directory, beside
    its receipts directory unless ``--modal-state-root`` names it; a copy of
    the run directory anywhere else is not what it describes."""
    hub, copied, receipt = _modal_build(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere" / "out"
    shutil.copytree(copied, elsewhere)
    stager = load_tool("stage_uk_rowwise_candidate")
    started = _workstation_emitters(
        monkeypatch, stager, hub_files=hub.files, hub_api=hub
    )

    with pytest.raises(SystemExit, match=r"is not inside the Modal run's state"):
        stager.main(_staging_argv(elsewhere, receipt))
    with pytest.raises(SystemExit, match=rf"the receipt lists no {STEM}\.h5"):
        stager.main(_staging_argv(copied, receipt, "--modal-state-root", str(copied)))

    assert started == []
    assert hub.paths(PRIVATE_REPOSITORY) == []
    # Naming the state directory itself is the default's equivalent.
    assert (
        stager.main(
            _staging_argv(
                copied,
                receipt,
                "--modal-state-root",
                str(receipt.parent.parent / "state"),
            )
        )
        == 0
    )


def test_the_modal_state_root_needs_a_receipt():
    stager = load_tool("stage_uk_rowwise_candidate")
    with pytest.raises(SystemExit):
        stager._parse_args(["--run-dir", "run", "--modal-state-root", "state"])
    parsed = stager._parse_args(["--run-dir", "run", "--modal-receipt", "receipt.json"])
    assert parsed.modal_receipt == Path("receipt.json")
    assert parsed.modal_state_root is None


def test_a_copied_back_bundle_that_landed_is_not_handed_over_again(
    tmp_path, monkeypatch, capsys
):
    """Staging rewrites the manifest and ``sha256sums.txt``, so they no longer
    match the receipt; the outputs still do, so a second run with the receipt,
    and a re-stage without it, both find the record and change nothing."""
    hub, copied, receipt = _modal_build(tmp_path, monkeypatch)
    stager = load_tool("stage_uk_rowwise_candidate")
    started = _workstation_emitters(
        monkeypatch, stager, hub_files=hub.files, hub_api=hub
    )
    assert stager.main(_staging_argv(copied, receipt)) == 0
    listed = {
        item["path"]: item["sha256"]
        for item in json.loads(receipt.read_text())["outputs"]
    }
    manifest_path = copied / stager.MANIFEST_FILENAME
    assert (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != listed[f"out/{stager.MANIFEST_FILENAME}"]
    )
    landed = _files(copied)

    for argv in (
        _staging_argv(copied, receipt),
        ["--run-dir", str(copied), "--wait-seconds", "60"],
    ):
        capsys.readouterr()
        assert stager.main(argv) == 0
        assert "nothing to do" in capsys.readouterr().err

    assert _files(copied) == landed
    assert len(hub.commits) == 1
    assert sum(len(client.bundle_requests) for client in started) == 1


def test_a_copied_back_bundle_whose_first_hand_over_failed_is_staged_again(
    tmp_path, monkeypatch, capsys
):
    """The operator's first attempt finds no credential; the emitter keeps the
    bundle queued, and the next attempt with the receipt records it landed.
    The Modal build's own record stays the one the note names."""
    hub, copied, receipt = _modal_build(tmp_path, monkeypatch)
    queue = FakeHandoverQueue()
    stager = load_tool("stage_uk_rowwise_candidate")
    _workstation_emitters(monkeypatch, stager, hub_files=hub.files, queue=queue)
    assert stager.main(_staging_argv(copied, receipt, "--wait-seconds", "0")) == 1
    manifest = _manifest(copied)
    assert manifest["staged_dataset"]["error_code"] == "CREDENTIAL_MISSING"
    assert manifest["staged_dataset_at_build"]["status"] == "skipped"
    _workstation_emitters(
        monkeypatch, stager, hub_files=hub.files, hub_api=hub, queue=queue
    )
    capsys.readouterr()

    assert stager.main(_staging_argv(copied, receipt)) == 0

    manifest = _manifest(copied)
    assert manifest["staged_dataset"]["status"] == "uploaded"
    assert manifest["staged_dataset"]["revision"] == hub.sha
    assert manifest["staged_dataset_at_build"]["status"] == "skipped"
    err = capsys.readouterr().err
    assert "still records skipped" in err
    assert "CREDENTIAL_MISSING" not in err
    assert len(hub.commits) == 1


def test_hf_hub_offline_on_the_workstation_stages_nothing(
    tmp_path, monkeypatch, capsys
):
    """``HF_HUB_OFFLINE`` left set on the workstation, as the runner sets it
    for every tool, means the emitter could never upload: the command refuses
    before it starts one. ``--local-only`` still checks the receipt and keeps
    the bundle local."""
    hub, copied, receipt = _modal_build(tmp_path, monkeypatch)
    before = _files(copied)
    stager = load_tool("stage_uk_rowwise_candidate")
    started = _workstation_emitters(
        monkeypatch, stager, hub_files=hub.files, hub_api=hub
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(SystemExit, match="HF_HUB_OFFLINE is set"):
        stager.main(_staging_argv(copied, receipt))
    assert _files(copied) == before

    capsys.readouterr()
    assert stager.main(_staging_argv(copied, receipt, "--local-only")) == 0
    staged = _manifest(copied)["staged_dataset"]
    assert (staged["mode"], staged["status"], staged["repository"]) == (
        "local_only",
        "skipped",
        None,
    )
    _edit_an_output(copied)
    with pytest.raises(SystemExit, match="is not the file the Modal run wrote"):
        stager.main(_staging_argv(copied, receipt, "--local-only"))

    assert started == []
    assert hub.paths(PRIVATE_REPOSITORY) == []
