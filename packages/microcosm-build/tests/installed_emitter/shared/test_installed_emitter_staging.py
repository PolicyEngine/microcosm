"""Staged dataset bundles handed to the installed microcosm-emitter.

Every other workspace test hands bundles to the in-memory emitter client. These
start the separately installed ``microcosm-emitter`` command, as a build does
(``start_bundle_emitter``: the Hugging Face module only, so the telemetry
module never starts and nothing contacts the collector), and send its Hugging
Face Hub requests to the loopback stand-in Hub (``HF_ENDPOINT``).
``tools/run_installed_emitter_tests.sh`` installs the minimum release builds
accept and the newest pinned release, and runs this file once for each, with
``MICROCOSM_EMITTER_COMMAND`` naming the installed command and
``INSTALLED_EMITTER_VERSION`` its release.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
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
    handover_record,
    resume_hand_over,
    stage_bundle,
    write_access_problem,
    write_sidecars,
)
from microcosm.build.telemetry_emitter import (
    OutdatedEmitterClient,
    start_bundle_emitter,
)
from test_support.microcosm_build.stand_in_hub import StandInHub

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


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_bundle(root: Path, *, dataset_bytes: int = 4096) -> StagedDatasetBundle:
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
    (run_directory / MANIFEST).write_text(json.dumps(manifest, indent=2))
    bundle = StagedDatasetBundle.from_manifest(
        run_directory, run_id=RUN_ID, manifest_name=MANIFEST
    )
    write_sidecars(bundle, repository=REPOSITORY)
    return bundle


def _local_digests(bundle: StagedDatasetBundle) -> dict[str, str]:
    """Each bundle file's path under the staged folder, and its SHA-256 now."""
    return {
        f"{PREFIX}/{name}": _sha256((bundle.run_dir / name).read_bytes())
        for name in BUNDLE_NAMES
    }


@dataclass
class InstalledEmitter:
    """Starts the installed emitter as a build does, and stops what it started."""

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
    emitter process it starts, and waits for each to exit afterwards.
    """
    command = os.environ.get("MICROCOSM_EMITTER_COMMAND", "").strip()
    version = os.environ.get("INSTALLED_EMITTER_VERSION", "").strip()
    if not command or not version:
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
        # A request meant for any host but the stand-in goes to the stand-in
        # as a proxy request, which it refuses and records.
        "HTTP_PROXY": stand_in_hub.url,
        "HTTPS_PROXY": stand_in_hub.url,
        "ALL_PROXY": stand_in_hub.url,
        "NO_PROXY": "127.0.0.1",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    emitter = InstalledEmitter(version=version, monkeypatch=monkeypatch)
    monkeypatch.setattr(emitter_interface_client, "start_emitter", emitter.record_start)
    try:
        yield emitter
    finally:
        emitter.stop()
    assert stand_in_hub.proxied_requests == [], "a request left the loopback stand-in"


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
    handover = stage_bundle(
        bundle, emitter=client, repository=REPOSITORY, wait_seconds=UPLOAD_WAIT_SECONDS
    )

    commits = stand_in_hub.commits(REPOSITORY)
    assert len(commits) == 1
    commit = commits[0]
    assert commit.parent == first_head
    assert commit.title == f"Stage {RUN_ID} under {PREFIX}"
    assert {path: _sha256(content) for path, content in commit.added.items()} == (
        _local_digests(bundle)
    )
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


def test_staging_the_same_bundle_again_records_already_staged(
    tmp_path: Path, stand_in_hub: StandInHub, installed_emitter: InstalledEmitter
) -> None:
    bundle = _write_bundle(tmp_path)
    client = installed_emitter.start()
    first = stage_bundle(
        bundle, emitter=client, repository=REPOSITORY, wait_seconds=UPLOAD_WAIT_SECONDS
    )
    assert first.delivery["status"] == "uploaded"
    revision = first.delivery["revision"]

    # Through the same queue, the identical hand-over returns its receipt.
    repeated = stage_bundle(
        bundle, emitter=client, repository=REPOSITORY, wait_seconds=UPLOAD_WAIT_SECONDS
    )
    # An emitter with its own queue finds the bundle on the Hub.
    second_client = installed_emitter.start(queue_directory=tmp_path / "second-queue")
    again = stage_bundle(
        bundle,
        emitter=second_client,
        repository=REPOSITORY,
        wait_seconds=UPLOAD_WAIT_SECONDS,
    )

    assert repeated.delivery == first.delivery
    assert again.delivery == {**first.delivery, "status": "already_staged"}
    assert again.delivery["revision"] == revision
    assert len(stand_in_hub.commits(REPOSITORY)) == 1


def test_a_read_only_credential_is_refused_before_the_build_stages(
    stand_in_hub: StandInHub, installed_emitter: InstalledEmitter
) -> None:
    stand_in_hub.refuse_writes()
    client = installed_emitter.start()

    problem = write_access_problem(client, REPOSITORY)

    assert problem is not None
    assert "needs a write credential" in problem
    assert ("GET", "whoami") in stand_in_hub.requests


def test_a_refused_commit_is_recorded_and_the_build_continues(
    tmp_path: Path, stand_in_hub: StandInHub, installed_emitter: InstalledEmitter
) -> None:
    bundle = _write_bundle(tmp_path)
    stand_in_hub.refuse_writes()
    client = installed_emitter.start()

    handover = stage_bundle(
        bundle, emitter=client, repository=REPOSITORY, wait_seconds=UPLOAD_WAIT_SECONDS
    )

    assert handover.delivery["status"] == "failed"
    assert handover.delivery["error_code"] == "UPLOAD_FAILED"
    assert handover.delivery["revision"] is None
    assert stand_in_hub.commits(REPOSITORY) == []
    assert ("POST", "preupload") in stand_in_hub.requests


def test_a_hub_that_never_answers_leaves_the_upload_pending(
    tmp_path: Path, stand_in_hub: StandInHub, installed_emitter: InstalledEmitter
) -> None:
    bundle = _write_bundle(tmp_path)
    stand_in_hub.stall("commit")
    client = installed_emitter.start()

    handover = stage_bundle(
        bundle,
        emitter=client,
        repository=REPOSITORY,
        wait_seconds=NO_ANSWER_WAIT_SECONDS,
    )

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
