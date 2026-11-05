"""Staged dataset bundles handed to the installed microcosm-emitter.

Every other workspace test hands bundles to the in-memory emitter client. These
start the separately installed ``microcosm-emitter`` command with
``start_bundle_emitter()``, as the UK re-stage command
(``tools/stage_uk_rowwise_candidate.py``) starts it: the Hugging Face module
only, so the telemetry module never starts and nothing contacts the collector.
A build's own emitter also runs the telemetry and Orrery modules; its Hugging
Face module is the same code. The emitter's Hugging Face Hub requests go to the
loopback stand-in Hub (``HF_ENDPOINT``), which takes the dataset through the
LFS upload a real Hub requires for it, since Xet is turned off.

``tools/run_installed_emitter_tests.sh`` installs the minimum release builds
accept and the newest pinned release, and runs this file once for each, with
``MICROCOSM_EMITTER_COMMAND`` naming the installed command and
``INSTALLED_EMITTER_VERSION`` its release.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from microcosm_emitter_interface import EmitterClient, launcher
from microcosm_emitter_interface import client as emitter_interface_client

from microcosm.build import telemetry_emitter
from microcosm.build.staging_dataset import (
    SHA256SUMS_FILENAME,
    STAGED_MANIFEST_FILENAME,
    StagedDatasetBundle,
    at_build_record,
    handover_record,
    local_only_staged_dataset,
    resume_hand_over,
    stage_bundle,
    write_access_problem,
    write_sidecars,
)
from microcosm.build.telemetry_emitter import (
    OutdatedEmitterClient,
    start_bundle_emitter,
)
from test_support.microcosm_build.modal_copy_back import copy_back
from test_support.microcosm_build.stand_in_hub import CommitAttempt, StandInHub
from test_support.microcosm_build.uk_full_build_cli import load_tool

REPOSITORY = "microcosm-tests/staged-bundles"
RUN_ID = "installed-emitter-20261102T000000Z"
PREFIX = f"staged/{RUN_ID}"
MANIFEST = "build_manifest.json"
DATASET = "microcosm_test_2025.h5"
CALIBRATION_DIAGNOSTICS = "calibration_diagnostics.json"
#: A placeholder the stand-in Hub accepts; never a Hugging Face credential.
TOKEN = "stand-in-token-not-a-credential"
#: The emitter copies bundle files of at most 32 MiB before it answers, so a
#: dataset one byte larger is uploaded from disk and must keep its bytes.
UNCOPIED_DATASET_BYTES = 32 * 1024 * 1024 + 1
#: A bound on the build's wait for an upload that succeeds or is refused, so
#: a broken hand-over fails the test instead of waiting 30 minutes.
UPLOAD_WAIT_SECONDS = 60.0
#: The build's wait in the test of a Hub that never answers.
NO_ANSWER_WAIT_SECONDS = 3.0
#: The emitter's shutdown allowance (15 seconds) and some margin.
EMITTER_EXIT_SECONDS = 30.0
BUNDLE_NAMES = frozenset(
    {
        DATASET,
        CALIBRATION_DIAGNOSTICS,
        MANIFEST,
        STAGED_MANIFEST_FILENAME,
        SHA256SUMS_FILENAME,
    }
)
#: Lists the repository with the emitter's own huggingface_hub, as JSON.
LIST_REPOSITORY = """\
import json
import sys

from huggingface_hub import HfApi

api = HfApi()
repository, folder = sys.argv[1], sys.argv[2]
files = api.list_repo_files(repository, repo_type="dataset")
entries = api.list_repo_tree(repository, path_in_repo=folder, repo_type="dataset")
print(json.dumps(
    {
        "files": sorted(files),
        "folder": [[type(entry).__name__, entry.path] for entry in entries],
    }
))
"""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_bundle(
    root: Path, *, dataset_bytes: int = 4096, manifest_name: str = MANIFEST
) -> StagedDatasetBundle:
    """A finished run's directory: a dataset, its diagnostics and a manifest.

    The bundle has every role a build hands over: ``dataset`` and
    ``calibration_diagnostics`` outputs, the build manifest, and the staged
    manifest and checksums that ``write_sidecars`` adds.
    """
    run_directory = root / "run"
    run_directory.mkdir()
    pattern = b"microcosm-installed-emitter-test-dataset\n"
    contents = {
        "dataset": (
            DATASET,
            (pattern * (dataset_bytes // len(pattern) + 1))[:dataset_bytes],
        ),
        "calibration_diagnostics": (
            CALIBRATION_DIAGNOSTICS,
            json.dumps({"targets": [{"name": "households", "error": 0.0}]}).encode(),
        ),
    }
    outputs = {}
    for key, (name, content) in contents.items():
        (run_directory / name).write_bytes(content)
        outputs[key] = {
            "path": name,
            "sha256": _sha256(content),
            "bytes": len(content),
        }
    manifest = {
        "build_kind": "installed_emitter_test",
        "releasable": False,
        "outputs": outputs,
    }
    (run_directory / manifest_name).write_text(json.dumps(manifest, indent=2))
    bundle = StagedDatasetBundle.from_manifest(
        run_directory, run_id=RUN_ID, manifest_name=manifest_name
    )
    write_sidecars(bundle, repository=REPOSITORY)
    return bundle


def _local_digests(bundle: StagedDatasetBundle) -> dict[str, str]:
    """Each bundle file's path under the staged folder, and its SHA-256 now."""
    return {
        f"{PREFIX}/{name}": _sha256((bundle.run_dir / name).read_bytes())
        for name in BUNDLE_NAMES
    }


def _stage(bundle: StagedDatasetBundle, client: EmitterClient, **options: Any):
    return stage_bundle(
        bundle,
        emitter=client,
        repository=REPOSITORY,
        wait_seconds=options.pop("wait_seconds", UPLOAD_WAIT_SECONDS),
        **options,
    )


@dataclass
class InstalledEmitter:
    """Starts the installed emitter as the re-stage command does, and stops it."""

    command: str
    version: str
    monkeypatch: pytest.MonkeyPatch
    processes: list[launcher.EmitterProcess] = field(default_factory=list)
    clients: list[EmitterClient] = field(default_factory=list)

    def start(self, *, queue_directory: Path | None = None) -> EmitterClient:
        """``start_bundle_emitter()``, optionally with a queue of its own.

        ``queue_directory`` stands for another machine: the emitter keeps its
        event queue file and hand-over database under it instead of beside
        the earlier emitters'.
        """
        if queue_directory is not None:
            self.monkeypatch.setenv("XDG_CACHE_HOME", str(queue_directory))
        client = start_bundle_emitter()
        self.clients.append(client)
        return client

    def record_start(self, *arguments: Any, **options: Any) -> launcher.EmitterProcess:
        process = launcher.start_emitter(*arguments, **options)
        self.processes.append(process)
        return process

    def run_hub_client(self, script: str, *arguments: str) -> Any:
        """Run ``script`` with the emitter's own Python and huggingface_hub.

        The uv tool directory keeps the emitter's interpreter beside its
        command. The script inherits this process's environment, so its
        requests go to the stand-in too; it prints JSON.
        """
        python = Path(self.command).resolve().parent / "python"
        answer = subprocess.run(
            [str(python), "-I", "-c", script, *arguments],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return json.loads(answer.stdout)

    def stop(self) -> None:
        for client in self.clients:
            client.close()
        for emitter in self.processes:
            try:
                emitter.process.wait(timeout=EMITTER_EXIT_SECONDS)
            except subprocess.TimeoutExpired:
                emitter.process.kill()
                emitter.process.wait(timeout=5)


@pytest.fixture
def stand_in_hub():
    hub = StandInHub(token=TOKEN)
    hub.create_repository(REPOSITORY)
    try:
        yield hub
    finally:
        hub.close()


@pytest.fixture
def installed_emitter(fake_telemetry_emitters, monkeypatch, tmp_path, stand_in_hub):
    """The installed emitter command, with its Hub requests sent to the stand-in.

    The emitter inherits this process's environment, which the fixture sets.
    The workspace's own fixture makes any attempt to start an emitter fail
    the test; this one puts the interface's launcher back, records every
    emitter process it starts, and waits for each to exit afterwards. Then it
    fails the test if the stand-in saw a request it has no route for, a
    request its handler failed on, or a proxy request meant for another host,
    naming each (``StandInHub.stray_requests``).
    """
    command = os.environ.get("MICROCOSM_EMITTER_COMMAND", "").strip()
    version = os.environ.get("INSTALLED_EMITTER_VERSION", "").strip()
    if not command or not version or shutil.which(command) is None:
        pytest.fail(
            "Run these tests with tools/run_installed_emitter_tests.sh, which sets "
            "MICROCOSM_EMITTER_COMMAND and INSTALLED_EMITTER_VERSION."
        )
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HUGGINGFACE_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    environment = {
        "HF_ENDPOINT": stand_in_hub.url,
        "HF_TOKEN": TOKEN,
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HOME": str(tmp_path / "hugging-face-home"),
        "XDG_CACHE_HOME": str(tmp_path / "emitter-cache"),
        # A client that honours the proxy variables, as huggingface_hub's
        # does, sends a request meant for any other host to the stand-in,
        # which refuses and records it. A client that ignored them would
        # reach that host unseen.
        "HTTP_PROXY": stand_in_hub.url,
        "HTTPS_PROXY": stand_in_hub.url,
        "ALL_PROXY": stand_in_hub.url,
        "NO_PROXY": "127.0.0.1",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    emitter = InstalledEmitter(
        command=command, version=version, monkeypatch=monkeypatch
    )
    monkeypatch.setattr(emitter_interface_client, "start_emitter", emitter.record_start)
    try:
        yield emitter
    finally:
        emitter.stop()
    assert stand_in_hub.stray_requests() == [], (
        "the stand-in Hub could not answer every request as a Hub would"
    )


def test_the_build_starts_the_installed_emitter_with_only_its_hub_module(
    installed_emitter: InstalledEmitter,
) -> None:
    client = installed_emitter.start()

    assert client.available
    assert client.emitter_version == installed_emitter.version
    assert client.modules == {"huggingface"}
    assert len(installed_emitter.processes) == 1


def test_an_emitter_older_than_the_minimum_is_never_started(
    installed_emitter: InstalledEmitter,
    stand_in_hub: StandInHub,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    major, minor, patch = (int(part) for part in installed_emitter.version.split("."))
    required = f"{major}.{minor}.{patch + 1}"
    # The adapter compares the installed release with ``_MINIMUM_RELEASE``,
    # computed from ``MINIMUM_EMITTER_VERSION`` when it is imported.
    monkeypatch.setattr(telemetry_emitter, "MINIMUM_EMITTER_VERSION", required)
    monkeypatch.setattr(
        telemetry_emitter, "_MINIMUM_RELEASE", (major, minor, patch + 1)
    )

    client = installed_emitter.start()
    problem = write_access_problem(client, REPOSITORY)

    assert isinstance(client, OutdatedEmitterClient)
    assert not client.available
    assert installed_emitter.processes == []
    assert stand_in_hub.requests == []
    warning = capsys.readouterr().err
    assert f"reports version {installed_emitter.version}" in warning
    assert f"needs {required} or later" in warning
    assert problem is not None
    assert f"reports version {installed_emitter.version}" in problem


def test_the_installed_emitter_uploads_the_bundle_in_one_commit(
    tmp_path: Path, stand_in_hub: StandInHub, installed_emitter: InstalledEmitter
) -> None:
    bundle = _write_bundle(tmp_path, dataset_bytes=UNCOPIED_DATASET_BYTES)
    first_head = stand_in_hub.head(REPOSITORY)
    client = installed_emitter.start()

    assert write_access_problem(client, REPOSITORY) is None
    handover = _stage(bundle, client)

    commits = stand_in_hub.commits(REPOSITORY)
    assert len(commits) == 1
    commit = commits[0]
    # The commit names the head it was built on, and that was the head.
    assert stand_in_hub.commit_attempts(REPOSITORY) == [
        CommitAttempt(parent_commit=first_head, head=first_head, status=200)
    ]
    assert commit.parent == first_head
    assert commit.title == f"Stage {RUN_ID} under {PREFIX}"
    assert {path: _sha256(content) for path, content in commit.added.items()} == (
        _local_digests(bundle)
    )
    # The dataset went through the LFS upload, as a real Hub requires for an
    # .h5 file over 10 MB; the other files travelled inside the commit.
    dataset_sha256 = _sha256((bundle.run_dir / DATASET).read_bytes())
    assert commit.uploaded_as == {
        f"{PREFIX}/{name}": "lfs" if name == DATASET else "regular"
        for name in BUNDLE_NAMES
    }
    assert set(stand_in_hub.lfs_objects) == {dataset_sha256}
    for kind in ("preupload", "lfs_batch", "lfs_upload", "lfs_verify", "commit"):
        assert stand_in_hub.statuses(kind) == [200], kind
    assert handover.delivery == {
        "contract_version": 1,
        "mode": "local_and_remote",
        "repository": REPOSITORY,
        "prefix": PREFIX,
        "run_id": RUN_ID,
        "revision": commit.oid,
        "status": "uploaded",
        "error_code": None,
        "opt_out_reason": None,
        "files": bundle.digests(),
    }
    # The emitter copied every file within its limits before it answered, and
    # uploaded the larger dataset from disk: only copied files may change.
    assert handover.handed_over == BUNDLE_NAMES
    assert handover.copied_files == BUNDLE_NAMES - {DATASET}
    assert not handover.may_rewrite(DATASET)
    for name in BUNDLE_NAMES - {DATASET}:
        assert handover.may_rewrite(name)


def test_a_modal_runs_copied_back_outputs_are_staged_by_the_re_stage_command(
    tmp_path: Path, stand_in_hub: StandInHub, installed_emitter: InstalledEmitter
) -> None:
    """After ``modal volume get``, the UK re-stage command checks the build
    manifest and the outputs against the stage's receipt, starts the installed
    emitter as it does for a local directory, and records the commit the
    emitter made of them, under the run id the build recorded."""
    stager = load_tool("stage_uk_rowwise_candidate")
    bundle = _write_bundle(tmp_path, manifest_name=stager.MANIFEST_FILENAME)
    # A build run with HF_HUB_OFFLINE, as on Modal, records its bundle as
    # local_only under its run id, and the command stages it under that id.
    manifest_path = bundle.run_dir / stager.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    record = local_only_staged_dataset(bundle, prefix="staged")
    manifest.update(
        staged_dataset=record, staged_dataset_at_build=at_build_record(record)
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    copied, receipt = copy_back(bundle.run_dir, tmp_path / "modal-runs")

    status = stager.main(
        [
            "--run-dir",
            str(copied),
            "--repo-id",
            REPOSITORY,
            "--modal-receipt",
            str(receipt),
            "--wait-seconds",
            str(UPLOAD_WAIT_SECONDS),
        ]
    )

    assert status == 0
    assert len(installed_emitter.processes) == 1
    (commit,) = stand_in_hub.commits(REPOSITORY)
    names = {
        DATASET,
        CALIBRATION_DIAGNOSTICS,
        stager.MANIFEST_FILENAME,
        STAGED_MANIFEST_FILENAME,
        SHA256SUMS_FILENAME,
    }
    assert set(commit.added) == {f"{PREFIX}/{name}" for name in names}
    listed = {
        item["path"]: item["sha256"]
        for item in json.loads(receipt.read_text())["outputs"]
    }
    for name in (DATASET, CALIBRATION_DIAGNOSTICS):
        assert _sha256(commit.added[f"{PREFIX}/{name}"]) == listed[f"out/{name}"]
    manifest = json.loads((copied / stager.MANIFEST_FILENAME).read_text())
    staged = manifest["staged_dataset"]
    assert (staged["status"], staged["error_code"]) == ("uploaded", None)
    assert (staged["repository"], staged["prefix"], staged["revision"]) == (
        REPOSITORY,
        PREFIX,
        commit.oid,
    )
    assert staged["run_id"] == RUN_ID
    assert manifest["staged_dataset_handover"]["run_dir"] == str(copied.resolve())


def test_a_commit_on_a_moved_head_is_refused_then_made_on_the_new_head(
    tmp_path: Path, stand_in_hub: StandInHub, installed_emitter: InstalledEmitter
) -> None:
    bundle = _write_bundle(tmp_path)
    first_head = stand_in_hub.head(REPOSITORY)
    other_file = "other-writer/notes.json"
    stand_in_hub.land_before_next_commit(REPOSITORY, {other_file: b"{}\n"})
    client = installed_emitter.start()

    handover = _stage(bundle, client)

    other_commit, bundle_commit = stand_in_hub.commits(REPOSITORY)
    assert stand_in_hub.commit_attempts(REPOSITORY) == [
        CommitAttempt(parent_commit=first_head, head=other_commit.oid, status=412),
        CommitAttempt(
            parent_commit=other_commit.oid, head=other_commit.oid, status=200
        ),
    ]
    assert bundle_commit.parent == other_commit.oid
    assert bundle_commit.files[other_file] == b"{}\n"
    # Each commit request was prepared again, but the dataset's LFS object was
    # uploaded and verified once; the second batch answer offered no upload,
    # and the second commit refers to the stored object.
    assert stand_in_hub.statuses("lfs_batch") == [200, 200]
    assert stand_in_hub.statuses("lfs_upload") == [200]
    assert stand_in_hub.statuses("lfs_verify") == [200]
    assert bundle_commit.uploaded_as[f"{PREFIX}/{DATASET}"] == "lfs"
    assert handover.delivery["status"] == "uploaded"
    assert handover.delivery["revision"] == bundle_commit.oid


def test_the_tree_lists_the_staged_folder(
    tmp_path: Path, stand_in_hub: StandInHub, installed_emitter: InstalledEmitter
) -> None:
    bundle = _write_bundle(tmp_path)
    assert _stage(bundle, installed_emitter.start()).delivery["status"] == "uploaded"

    listing = installed_emitter.run_hub_client(LIST_REPOSITORY, REPOSITORY, "staged")

    assert listing["files"] == sorted(_local_digests(bundle))
    assert listing["folder"] == [["RepoFolder", PREFIX]]
    assert stand_in_hub.statuses("tree") == [200, 200]


def test_staging_the_same_bundle_again_records_already_staged(
    tmp_path: Path, stand_in_hub: StandInHub, installed_emitter: InstalledEmitter
) -> None:
    bundle = _write_bundle(tmp_path)
    client = installed_emitter.start()
    first = _stage(bundle, client)
    assert first.delivery["status"] == "uploaded"

    # Through the same queue, the identical hand-over returns its receipt.
    repeated = _stage(bundle, client)
    # An emitter with its own queue finds the bundle on the Hub.
    second_client = installed_emitter.start(queue_directory=tmp_path / "second-queue")
    again = _stage(bundle, second_client)

    assert repeated.delivery == first.delivery
    assert again.delivery == {**first.delivery, "status": "already_staged"}
    assert len(stand_in_hub.commits(REPOSITORY)) == 1
    assert stand_in_hub.statuses("paths_info") == [200]


def test_a_read_only_credential_is_refused_before_the_build_stages(
    stand_in_hub: StandInHub, installed_emitter: InstalledEmitter
) -> None:
    stand_in_hub.refuse_writes()
    client = installed_emitter.start()

    problem = write_access_problem(client, REPOSITORY)

    assert problem is not None
    assert "needs a write credential" in problem
    assert stand_in_hub.statuses("whoami") == [200]


def test_a_refused_commit_is_recorded_and_the_build_continues(
    tmp_path: Path, stand_in_hub: StandInHub, installed_emitter: InstalledEmitter
) -> None:
    bundle = _write_bundle(tmp_path)
    stand_in_hub.refuse_writes()
    client = installed_emitter.start()

    handover = _stage(bundle, client)

    assert handover.delivery["status"] == "failed"
    assert handover.delivery["error_code"] == "UPLOAD_FAILED"
    assert handover.delivery["revision"] is None
    assert stand_in_hub.commits(REPOSITORY) == []
    assert stand_in_hub.statuses("preupload") == [403]
    assert stand_in_hub.lfs_objects == {}


def test_a_rejected_token_is_refused_and_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stand_in_hub: StandInHub,
    installed_emitter: InstalledEmitter,
) -> None:
    bundle = _write_bundle(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "a-token-the-stand-in-rejects")
    client = installed_emitter.start()

    problem = write_access_problem(client, REPOSITORY)
    handover = _stage(bundle, client)

    # This records today's mapping: the emitter reports a token the Hub rejects
    # (HTTP 401) as an unreachable repository, and records the hand-over as
    # REPOSITORY_UNAVAILABLE, which it keeps retrying, though no retry can
    # succeed with the same token. An emitter error code for a rejected
    # credential is a planned emitter follow-up; when it arrives, this test
    # fails, and Microcosm's error codes need it.
    assert problem is not None
    assert f"cannot reach {REPOSITORY} (HfHubHTTPError)" in problem
    assert handover.delivery["status"] == "failed"
    assert handover.delivery["error_code"] == "REPOSITORY_UNAVAILABLE"
    assert stand_in_hub.requests
    assert {request.status for request in stand_in_hub.requests} == {401}
    assert stand_in_hub.commits(REPOSITORY) == []


def test_a_missing_token_is_refused_without_a_hub_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stand_in_hub: StandInHub,
    installed_emitter: InstalledEmitter,
) -> None:
    bundle = _write_bundle(tmp_path)
    monkeypatch.delenv("HF_TOKEN")
    client = installed_emitter.start()

    problem = write_access_problem(client, REPOSITORY)
    handover = _stage(bundle, client)

    assert problem is not None
    assert "needs a Hugging Face write credential" in problem
    assert handover.delivery["status"] == "failed"
    assert handover.delivery["error_code"] == "CREDENTIAL_MISSING"
    assert stand_in_hub.requests == []


def test_a_hub_that_never_answers_leaves_the_upload_pending(
    tmp_path: Path,
    stand_in_hub: StandInHub,
    installed_emitter: InstalledEmitter,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = _write_bundle(tmp_path)
    stand_in_hub.stall("commit")
    client = installed_emitter.start()

    started = time.monotonic()
    handover = _stage(bundle, client, wait_seconds=NO_ANSWER_WAIT_SECONDS)
    waited = time.monotonic() - started

    # The build waited until its limit, not until a failed receipt request.
    assert waited >= NO_ANSWER_WAIT_SECONDS
    assert "(UPLOAD_PENDING)" in capsys.readouterr().err
    assert handover.delivery["status"] == "failed"
    assert handover.delivery["error_code"] == "UPLOAD_PENDING"
    assert handover.delivery["revision"] is None
    assert handover.handed_over == BUNDLE_NAMES
    record = handover_record(handover)
    assert record is not None
    assert stand_in_hub.wait_for_stalled("commit", timeout=UPLOAD_WAIT_SECONDS)
    assert stand_in_hub.commits(REPOSITORY) == []

    # The emitter keeps the bundle queued and finishes once the Hub answers;
    # a re-stage then records the outcome of the hand-over the build's
    # manifest names (its ``staged_dataset_handover`` block).
    stand_in_hub.release()
    resumed = resume_hand_over(
        bundle,
        emitter=client,
        record=record,
        repository=REPOSITORY,
        wait_seconds=UPLOAD_WAIT_SECONDS,
    )

    commits = stand_in_hub.commits(REPOSITORY)
    assert len(commits) == 1
    assert {path: _sha256(content) for path, content in commits[0].added.items()} == (
        _local_digests(bundle)
    )
    assert resumed is not None
    assert resumed.delivery == {
        **handover.delivery,
        "status": "already_staged",
        "revision": commits[0].oid,
        "error_code": None,
    }
