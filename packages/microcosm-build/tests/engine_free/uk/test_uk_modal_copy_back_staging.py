"""Staging a UK build's outputs once they are copied back from a Modal run.

No UK program is registered with the Modal runner yet. The runner runs every
tool with ``HF_HUB_OFFLINE=1``, so a UK build run there would upload nothing:
it would verify its bundle, write the sidecars and record the bundle as
``local_only``. These tests run such a build locally with that variable set
and lay it out as ``modal volume get`` copies a run back. Then
``tools/stage_uk_rowwise_candidate.py --modal-receipt`` checks the build
manifest and every output it registers against the stage's receipt, and hands
the bundle to the operator's emitter with the code that re-stages a local
build that kept its bundle local.
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
MANIFEST = "rowwise_candidate_manifest.json"


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


def _refusal(stager, argv: list[str]) -> str:
    """The error message the command exits with; fails unless it refuses so."""
    try:
        status = stager.main(argv)
    except SystemExit as stop:
        assert isinstance(stop.code, str), f"the command exited with {stop.code!r}"
        return stop.code
    except Exception as error:
        raise AssertionError(
            f"the command raised {error!r} instead of exiting with a message"
        ) from error
    raise AssertionError(f"the command did not refuse; it returned {status}")


def _completes(stager, argv: list[str]) -> int:
    """The command's exit status; fails if it refuses."""
    try:
        return stager.main(argv)
    except SystemExit as stop:
        raise AssertionError(f"the command refused: {stop.code}") from None


def _edit_the_build_manifest(copied: Path, change) -> None:
    """Apply ``change`` to the build manifest, written as the build writes it."""
    path = copied / MANIFEST
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )


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


def _receipt_records_other_bytes(copied, run_root, receipt):
    def change(content):
        for item in content["outputs"]:
            if item["path"] == f"out/{STEM}.h5":
                item["sha256"] = "0" * 64

    rewrite_receipt(receipt, change)


def _dataset_never_mirrored(copied, run_root, receipt):
    (copied / f"{STEM}.h5").unlink()
    rewrite_receipt(
        receipt,
        lambda content: content.update(outputs_not_mirrored=[f"out/{STEM}.h5"]),
    )


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
            rf"out/{MANIFEST} is not the build manifest the Modal stage wrote",
            id="an output edited after the copy-back",
        ),
        pytest.param(
            _receipt_records_other_bytes,
            rf"out/{STEM}\.h5 is not the file the Modal stage wrote",
            id="an output the receipt records with other bytes",
        ),
        pytest.param(
            _dataset_never_mirrored,
            rf"out/{STEM}\.h5 was hashed in the Modal container but never copied",
            id="an output the stage never copied to the runs volume",
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
    unless the build manifest and every output it registers are the files the
    finished stage wrote."""
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
    """The operator's first attempt finds no credential, and the emitter keeps
    the bundle queued. The next emitter on the workstation uploads it, so the
    next attempt reads the earlier hand-over's receipt and records it as
    ``already_staged`` with its commit, as for a local build's queued bundle.
    The Modal build's own record stays the one the note names."""
    hub, copied, receipt = _modal_build(tmp_path, monkeypatch)
    queue = FakeHandoverQueue()
    stager = load_tool("stage_uk_rowwise_candidate")
    _workstation_emitters(monkeypatch, stager, hub_files=hub.files, queue=queue)
    assert stager.main(_staging_argv(copied, receipt)) == 1
    manifest = _manifest(copied)
    assert manifest["staged_dataset"]["error_code"] == "CREDENTIAL_MISSING"
    assert manifest["staged_dataset_at_build"]["status"] == "skipped"
    _workstation_emitters(
        monkeypatch, stager, hub_files=hub.files, hub_api=hub, queue=queue
    )
    capsys.readouterr()

    assert stager.main(_staging_argv(copied, receipt)) == 0

    manifest = _manifest(copied)
    assert manifest["staged_dataset"]["status"] == "already_staged"
    assert manifest["staged_dataset"]["revision"] == hub.sha
    assert manifest["staged_dataset_at_build"]["status"] == "skipped"
    err = capsys.readouterr().err
    assert "now records the bundle as already_staged" in err
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
    with pytest.raises(
        SystemExit, match="is not the build manifest the Modal stage wrote"
    ):
        stager.main(_staging_argv(copied, receipt, "--local-only"))

    assert started == []
    assert hub.paths(PRIVATE_REPOSITORY) == []


def _change_a_field(manifest, copied):
    manifest["release_posture"] = "edited after the copy-back"


def _register_another_file_the_stage_wrote(manifest, copied):
    registered = {
        Path(str(entry["path"])).name for entry in manifest["outputs"].values()
    }
    unregistered = sorted(
        path.name
        for path in copied.iterdir()
        if path.is_file()
        and path.name
        not in registered | {MANIFEST, "staged_manifest.json", "sha256sums.txt"}
    )
    assert unregistered, "the stage wrote no other file in the run directory"
    content = (copied / unregistered[0]).read_bytes()
    manifest["outputs"]["another_file"] = {
        "path": unregistered[0],
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _rename_a_role(manifest, copied):
    key = next(
        key
        for key, entry in manifest["outputs"].items()
        if str(entry["path"]).endswith(".h5")
    )
    manifest["outputs"]["renamed_dataset"] = manifest["outputs"].pop(key)


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(_change_a_field, id="a field changed"),
        pytest.param(
            _register_another_file_the_stage_wrote,
            id="another file the stage wrote registered as an output",
        ),
        pytest.param(_rename_a_role, id="an output's role renamed"),
    ],
)
def test_a_build_manifest_edited_after_the_copy_back_is_refused(
    tmp_path, monkeypatch, edit
):
    """The build manifest says which files are staged and with which roles,
    and it is uploaded as the build's record. Until a hand-over rewrites it,
    it must be the file the stage wrote, even when every output still
    matches the receipt."""
    hub, copied, receipt = _modal_build(tmp_path, monkeypatch)
    _edit_the_build_manifest(copied, lambda manifest: edit(manifest, copied))
    before = _files(copied)
    stager = load_tool("stage_uk_rowwise_candidate")
    started = _workstation_emitters(
        monkeypatch, stager, hub_files=hub.files, hub_api=hub
    )

    message = _refusal(stager, _staging_argv(copied, receipt))

    assert f"out/{MANIFEST} is not the build manifest the Modal stage wrote" in message
    assert started == []
    assert _files(copied) == before
    assert hub.paths(PRIVATE_REPOSITORY) == []


@pytest.mark.parametrize(
    "case", ["a stage that did not finish", "a directory outside the state"]
)
def test_the_receipt_and_the_directory_are_checked_before_any_output_is_hashed(
    tmp_path, monkeypatch, case
):
    """A UK dataset is about 2.4 GB, so the command refuses a receipt of an
    unfinished stage, or a directory the receipt does not describe, before it
    reads any output."""
    hub, copied, receipt = _modal_build(tmp_path, monkeypatch)
    run_dir = copied
    if case == "a stage that did not finish":
        write_receipt(receipt.parent.parent, status="FAILED")
        expected = "status is 'FAILED', not 'COMPLETED'"
    else:
        run_dir = tmp_path / "elsewhere" / "out"
        shutil.copytree(copied, run_dir)
        expected = "is not inside the Modal run's state directory"
    stager = load_tool("stage_uk_rowwise_candidate")
    _workstation_emitters(monkeypatch, stager, hub_files=hub.files, hub_api=hub)
    hashed = []
    from_manifest = staging_dataset.StagedDatasetBundle.from_manifest

    def recording_from_manifest(*arguments, **options):
        hashed.append(arguments)
        return from_manifest(*arguments, **options)

    monkeypatch.setattr(
        staging_dataset.StagedDatasetBundle,
        "from_manifest",
        staticmethod(recording_from_manifest),
    )

    message = _refusal(stager, _staging_argv(run_dir, receipt))

    assert expected in message
    assert hashed == []


def test_a_state_directory_reached_through_a_symbolic_link_is_the_default_root(
    tmp_path, monkeypatch
):
    """The default state root, ``state/`` beside the receipt's directory, is
    resolved as ``--modal-state-root`` is, so a ``state`` directory that is a
    symbolic link to where the files were copied still describes them."""
    hub, copied, receipt = _modal_build(tmp_path, monkeypatch)
    run_root = receipt.parent.parent
    volume_copy = tmp_path / "volume-copy"
    shutil.move(str(run_root / "state"), str(volume_copy))
    (run_root / "state").symlink_to(volume_copy, target_is_directory=True)
    stager = load_tool("stage_uk_rowwise_candidate")
    _workstation_emitters(monkeypatch, stager, hub_files=hub.files, hub_api=hub)

    assert _completes(stager, _staging_argv(run_root / "state" / "out", receipt)) == 0

    staged = _manifest(volume_copy / "out")["staged_dataset"]
    assert (staged["status"], staged["revision"]) == ("uploaded", hub.sha)


@pytest.mark.parametrize(
    "route",
    [
        "a symbolic link to the run directory",
        "a path through ..",
        "a link in the state directory to a copy outside it",
    ],
)
def test_the_run_directory_is_checked_where_it_really_is(tmp_path, monkeypatch, route):
    """The command compares the run directory's real location with the state
    directory, so a link to it or a path through ``..`` stages it, and a link
    placed in the state directory that leads outside it is refused."""
    hub, copied, receipt = _modal_build(tmp_path, monkeypatch)
    if route == "a symbolic link to the run directory":
        named = tmp_path / "link-to-the-run"
        named.symlink_to(copied, target_is_directory=True)
    elif route == "a path through ..":
        named = copied / ".." / copied.name
    else:
        outside = tmp_path / "outside" / "out"
        shutil.copytree(copied, outside)
        named = copied.parent / "linked-out"
        named.symlink_to(outside, target_is_directory=True)
    stager = load_tool("stage_uk_rowwise_candidate")
    started = _workstation_emitters(
        monkeypatch, stager, hub_files=hub.files, hub_api=hub
    )

    if route == "a link in the state directory to a copy outside it":
        message = _refusal(stager, _staging_argv(named, receipt))
        assert "is not inside the Modal run's state directory" in message
        assert started == []
        return
    assert _completes(stager, _staging_argv(named, receipt)) == 0
    manifest = _manifest(copied)
    assert manifest["staged_dataset"]["status"] == "uploaded"
    assert manifest["staged_dataset_handover"]["run_dir"] == str(copied.resolve())


def test_a_local_only_check_keeps_the_build_manifest_the_stage_wrote(
    tmp_path, monkeypatch
):
    """``--local-only`` writes the build's own ``local_only`` record into the
    build manifest again, so the manifest keeps the bytes the receipt lists
    and a later run with the receipt still stages it."""
    hub, copied, receipt = _modal_build(tmp_path, monkeypatch)
    manifest_bytes = (copied / MANIFEST).read_bytes()
    stager = load_tool("stage_uk_rowwise_candidate")
    started = _workstation_emitters(
        monkeypatch, stager, hub_files=hub.files, hub_api=hub
    )

    assert _completes(stager, _staging_argv(copied, receipt, "--local-only")) == 0
    assert started == []
    assert (copied / MANIFEST).read_bytes() == manifest_bytes
    assert _completes(stager, _staging_argv(copied, receipt)) == 0

    assert _manifest(copied)["staged_dataset"]["status"] == "uploaded"


@pytest.mark.parametrize(
    ("content", "says"),
    [
        pytest.param("{not json", "is not valid JSON", id="invalid JSON"),
        pytest.param("[]", "must be a JSON object", id="JSON that is not an object"),
    ],
)
def test_a_malformed_build_manifest_is_an_error_message(tmp_path, content, says):
    """Every refusal of the build manifest is an error message, as the
    bundle's own checks of it are, never a traceback."""
    stager = load_tool("stage_uk_rowwise_candidate")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / MANIFEST).write_text(content)

    message = _refusal(stager, ["--run-dir", str(run_dir), "--local-only"])

    assert message.startswith("error: ")
    assert says in message
