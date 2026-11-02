"""The staged-dataset lane: bundle from a manifest, emitter hand-over, fetch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build import staging_dataset
from microcosm.build.staging_dataset import (
    SHA256SUMS_FILENAME,
    STAGED_MANIFEST_FILENAME,
    StagedDatasetBundle,
    StagedDatasetError,
    delivery_covers_bundle,
    disabled_staged_dataset,
    fetch_bundle,
    local_only_staged_dataset,
    parse_sha256sums,
    refresh_sha256sums_entry,
    validate_staged_dataset_delivery,
    write_sidecars,
)
from microcosm.build.staging_storage import HuggingFaceDatasetStorage
from test_support.microcosm_build.emitter import FakeEmitterClient, FakeHandoverQueue

MANIFEST = "rowwise_candidate_manifest.json"
RUN_ID = "uk-local-candidate-f100-s7-20260917T180000Z-0badcafe"
REPOSITORY = "policyengine/populace-uk-private"
DATASET = "microcosm_uk_2025_local.h5"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_run(root: Path, *, manifest_extra: dict | None = None) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    files = {
        "dataset": (DATASET, b"h5-bytes" * 100),
        "diagnostics": ("solve_diagnostics.csv", b"a,b\n1,2\n"),
        "calibration_diagnostics": (
            "calibration_diagnostics.json",
            json.dumps({"targets": []}).encode(),
        ),
    }
    outputs = {}
    for key, (name, data) in files.items():
        (run_dir / name).write_bytes(data)
        outputs[key] = {
            "path": str(run_dir / name),
            "sha256": _sha(data),
            "bytes": len(data),
        }
    (run_dir / "run.log").write_text("not staged\n")
    manifest = {
        "build_kind": "uk_rowwise_calibrated_candidate",
        "releasable": False,
        "git_commit": "b" * 40,
        "outputs": outputs,
        **(manifest_extra or {}),
    }
    (run_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return run_dir


class FakeHub:
    """One private dataset repository, as ``HfApi`` serves it.

    The fake emitter commits bundles with it and ``fetch_bundle`` reads them
    back; it records each path's last commit, as ``get_paths_info`` does.
    """

    def __init__(self, root: Path, *, fail_commit: bool = False) -> None:
        self.root = root
        self.files: dict[str, bytes] = {}
        self.commits: list[dict] = []
        self.commit_of: dict[str, str] = {}
        self.fail_commit = fail_commit
        self.sha = "0" * 40

    def file_exists(self, *, repo_id, filename, repo_type):
        assert repo_type == "dataset"
        return filename in self.files

    def repo_info(self, *, repo_id, repo_type):
        assert repo_type == "dataset"
        return SimpleNamespace(sha=self.sha)

    def create_commit(
        self, *, repo_id, operations, commit_message, repo_type, parent_commit
    ):
        assert repo_type == "dataset"
        if self.fail_commit:
            raise RuntimeError("401 Unauthorized token=do-not-record")
        assert parent_commit == self.sha
        for operation in operations:
            content = operation.path_or_fileobj
            self.files[operation.path_in_repo] = (
                content if isinstance(content, bytes) else Path(content).read_bytes()
            )
        self.commits.append(
            {
                "message": commit_message,
                "paths": sorted(op.path_in_repo for op in operations),
                "parent": parent_commit,
            }
        )
        self.sha = _sha(f"{commit_message}{len(self.commits)}".encode())[:40]
        for operation in operations:
            self.commit_of[operation.path_in_repo] = self.sha
        return SimpleNamespace(oid=self.sha)

    def get_paths_info(self, *, repo_id, paths, expand, repo_type):
        assert expand and repo_type == "dataset"
        return [
            SimpleNamespace(path=p, last_commit=SimpleNamespace(oid=self.commit_of[p]))
            for p in paths
            if p in self.commit_of
        ]

    def hf_hub_download(self, *, repo_id, filename, repo_type, **kwargs):
        if filename not in self.files:
            raise FileNotFoundError(filename)
        destination = self.root / "cache" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[filename])
        return str(destination)


def _storage(hub: FakeHub) -> HuggingFaceDatasetStorage:
    return HuggingFaceDatasetStorage(REPOSITORY, api=hub)


def _bundle(tmp_path: Path) -> StagedDatasetBundle:
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    write_sidecars(bundle, repository=REPOSITORY)
    return bundle


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _stage(bundle: StagedDatasetBundle, emitter, **options):
    """Hand the bundle to ``emitter``, waiting on a test clock."""
    clock = Clock()
    return staging_dataset.stage_bundle(
        bundle,
        emitter=emitter,
        repository=REPOSITORY,
        clock=clock,
        sleep=clock.sleep,
        **options,
    )


def test_bundle_takes_exactly_the_manifest_outputs(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    assert [item.name for item in bundle.files] == [
        "calibration_diagnostics.json",
        DATASET,
        "solve_diagnostics.csv",
    ]
    # Each output's role is its manifest key, so readers find the dataset
    # and the calibration diagnostics by role rather than by file name.
    assert [item.role for item in bundle.files] == [
        "calibration_diagnostics",
        "dataset",
        "diagnostics",
    ]
    assert "run.log" not in bundle.digests()
    assert bundle.summary == {
        "build_kind": "uk_rowwise_calibrated_candidate",
        "git_commit": "b" * 40,
        "releasable": False,
    }
    assert bundle.remote_prefix() == f"staged/{RUN_ID}"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: (d / "solve_diagnostics.csv").unlink(), "missing"),
        (lambda d: (d / "solve_diagnostics.csv").write_bytes(b"x"), "digest"),
    ],
)
def test_bundle_refuses_missing_or_changed_outputs(tmp_path, mutation, message):
    run_dir = _write_run(tmp_path)
    mutation(run_dir)
    with pytest.raises(StagedDatasetError, match=message):
        StagedDatasetBundle.from_manifest(
            run_dir, run_id=RUN_ID, manifest_name=MANIFEST
        )


def test_bundle_refuses_an_unsafe_run_id_and_an_empty_manifest(tmp_path):
    run_dir = _write_run(tmp_path)
    with pytest.raises(StagedDatasetError, match="run_id"):
        StagedDatasetBundle.from_manifest(
            run_dir, run_id="../escape", manifest_name=MANIFEST
        )
    (run_dir / MANIFEST).write_text(json.dumps({"outputs": {}}))
    with pytest.raises(StagedDatasetError, match="nothing to stage"):
        StagedDatasetBundle.from_manifest(
            run_dir, run_id=RUN_ID, manifest_name=MANIFEST
        )


def test_sidecars_list_every_file_and_verify_themselves(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    sums, staged_manifest = write_sidecars(
        bundle, repository="policyengine/populace-uk-private"
    )
    entries = dict((name, sha) for sha, name in parse_sha256sums(sums.read_text()))
    assert set(entries) == {
        "calibration_diagnostics.json",
        DATASET,
        "solve_diagnostics.csv",
        MANIFEST,
        STAGED_MANIFEST_FILENAME,
    }
    for name, sha in entries.items():
        assert _sha((run_dir / name).read_bytes()) == sha
    payload = json.loads(staged_manifest.read_text())
    assert payload["schema_name"] == "microcosm.staged-dataset.manifest"
    assert payload["prefix"] == f"staged/{RUN_ID}"
    assert payload["files"] == bundle.digests()
    assert payload["manifest"]["sha256"] == _sha((run_dir / MANIFEST).read_bytes())
    # Byte-identical on a second write: no timestamps, no absolute paths.
    before = staged_manifest.read_bytes()
    write_sidecars(bundle, repository="policyengine/populace-uk-private")
    assert staged_manifest.read_bytes() == before
    assert str(run_dir) not in staged_manifest.read_text()


@pytest.mark.parametrize(
    ("key", "role"),
    [
        ("dataset", "dataset"),
        ("calibration_diagnostics", "calibration_diagnostics"),
        ("local_registry", "local_registry"),
        ("orrery/graph.schema.json", "orrery_graph_schema_json"),
        (
            "orrery/evidence-0e07dea6cca04063a47a8166793d9a0c-6f8356b6990586505573"
            "13f43b8875214a46246d990ce7e3f574d8af6dd52e9f-binding.json",
            "orrery_evidence_binding_json",
        ),
        ("dense_reference_deadbeef", "dense_reference_deadbeef"),
        ("Score-Receipt", "score_receipt"),
        ("2024_weights", "output_2024_weights"),
        ("checksums", "output_checksums"),
        ("...", "output"),
    ],
)
def test_an_outputs_role_is_its_manifest_key_as_a_role_name(key, role):
    assert staging_dataset.output_role(key) == role


def test_stage_bundle_hands_every_file_to_the_emitter_with_its_role_digest_and_size(
    tmp_path,
):
    bundle = _bundle(tmp_path)
    hub = FakeHub(tmp_path)
    emitter = FakeEmitterClient(hub_api=hub)
    staged = _stage(bundle, emitter)

    (request,) = emitter.bundle_requests
    assert request.repository == REPOSITORY
    assert request.run_id == RUN_ID
    assert request.prefix == "staged"
    assert request.commit_message is None
    run_dir = bundle.run_dir
    assert [
        (item.name, item.role, Path(item.local_path), item.sha256, item.bytes)
        for item in request.files
    ] == [
        (name, role, run_dir / name, _sha(data), len(data))
        for name, role, data in (
            (name, role, (run_dir / name).read_bytes())
            for name, role in (
                ("calibration_diagnostics.json", "calibration_diagnostics"),
                (DATASET, "dataset"),
                ("solve_diagnostics.csv", "diagnostics"),
                (MANIFEST, "build_manifest"),
                (STAGED_MANIFEST_FILENAME, "staged_manifest"),
                (SHA256SUMS_FILENAME, "checksums"),
            )
        )
    ]
    # The build made no Hub call: the emitter committed the bundle once.
    assert emitter.calls == ["stage_dataset_bundle", "dataset_bundle_receipt"]
    assert len(hub.commits) == 1
    assert hub.commits[0]["paths"] == sorted(
        f"staged/{RUN_ID}/{item.name}" for item in request.files
    )
    delivery = staged.delivery
    assert delivery == {
        "contract_version": 1,
        "mode": "local_and_remote",
        "repository": REPOSITORY,
        "prefix": f"staged/{RUN_ID}",
        "run_id": RUN_ID,
        "revision": hub.sha,
        "status": "uploaded",
        "error_code": None,
        "opt_out_reason": None,
        "files": bundle.digests(),
    }
    assert validate_staged_dataset_delivery(delivery) == delivery


def _receipt(status, *, attempts=1, error_code=None, revision=None, copied=()):
    return {
        "status": status,
        "error_code": error_code,
        "revision": revision,
        "attempts": attempts,
        "repository": REPOSITORY,
        "prefix": f"staged/{RUN_ID}",
        "copied_files": list(copied),
    }


QUEUED = _receipt("pending", attempts=0)


@pytest.mark.parametrize(
    ("outcome", "status", "error_code", "revision"),
    [
        (_receipt("uploaded", revision="a" * 40), "uploaded", None, "a" * 40),
        (
            _receipt("already_staged", revision="b" * 40),
            "already_staged",
            None,
            "b" * 40,
        ),
        (
            _receipt("failed", error_code="REMOTE_DIFFERS"),
            "failed",
            "REMOTE_DIFFERS",
            None,
        ),
        (
            _receipt("failed", error_code="PRESERVED_BYTES_CHANGED"),
            "failed",
            "PRESERVED_BYTES_CHANGED",
            None,
        ),
        # Still queued after a failed attempt: the build records the code,
        # and the emitter keeps retrying.
        (
            _receipt("pending", error_code="CREDENTIAL_MISSING"),
            "failed",
            "CREDENTIAL_MISSING",
            None,
        ),
        (
            _receipt("pending", error_code="REPOSITORY_UNAVAILABLE"),
            "failed",
            "REPOSITORY_UNAVAILABLE",
            None,
        ),
        (
            _receipt("pending", error_code="PRESERVED_FILES_MISSING"),
            "failed",
            "PRESERVED_FILES_MISSING",
            None,
        ),
        (
            _receipt("pending", error_code="A_LATER_EMITTER_CODE"),
            "failed",
            "UPLOAD_FAILED",
            None,
        ),
        # Settled answers Microcosm does not recognize are final, with their
        # own code, never one the emitter keeps retrying.
        (
            _receipt("failed", error_code="A_LATER_EMITTER_CODE"),
            "failed",
            "RECEIPT_UNRECOGNIZED",
            None,
        ),
        (
            _receipt("uploaded", revision=None),
            "failed",
            "RECEIPT_UNRECOGNIZED",
            None,
        ),
        # A code the emitter retries never arrives on a settled failure.
        (
            _receipt("failed", error_code="UPLOAD_FAILED"),
            "failed",
            "RECEIPT_UNRECOGNIZED",
            None,
        ),
        (
            _receipt("failed", error_code="UPLOAD_PENDING"),
            "failed",
            "RECEIPT_UNRECOGNIZED",
            None,
        ),
    ],
)
def test_the_emitters_receipt_becomes_the_delivery_record(
    tmp_path, capsys, outcome, status, error_code, revision
):
    bundle = _bundle(tmp_path)
    emitter = FakeEmitterClient(bundle_receipts=[QUEUED, outcome])
    delivery = _stage(bundle, emitter).delivery
    assert delivery["mode"] == "local_and_remote"
    assert delivery["repository"] == REPOSITORY
    assert delivery["prefix"] == f"staged/{RUN_ID}"
    assert delivery["run_id"] == RUN_ID
    assert delivery["files"] == bundle.digests()
    assert (delivery["status"], delivery["error_code"], delivery["revision"]) == (
        status,
        error_code,
        revision,
    )
    assert validate_staged_dataset_delivery(delivery) == delivery
    if status == "failed":
        assert error_code in capsys.readouterr().err


@pytest.mark.parametrize(
    ("emitter", "error_code"),
    [
        (None, "EMITTER_UNAVAILABLE"),
        (FakeEmitterClient(modules=()), "EMITTER_UNAVAILABLE"),
        (FakeEmitterClient(modules={"telemetry", "orrery"}), "EMITTER_UNAVAILABLE"),
        (
            FakeEmitterClient(
                bundle_receipts=[
                    {
                        **_receipt("refused", attempts=0),
                        "handover_id": "",
                        "error_code": "FILE_DIGEST_DIFFERS",
                    }
                ]
            ),
            "HANDOVER_REFUSED",
        ),
    ],
)
def test_a_hand_over_no_emitter_queued_is_a_failed_delivery(
    tmp_path, capsys, emitter, error_code
):
    bundle = _bundle(tmp_path)
    staged = _stage(bundle, emitter)
    assert staged.delivery["status"] == "failed"
    assert staged.delivery["error_code"] == error_code
    assert staged.delivery["revision"] is None
    assert error_code in capsys.readouterr().err
    # No emitter reads the bundle from disk, so the build may rewrite any file.
    assert staged.may_rewrite(MANIFEST) and staged.may_rewrite(SHA256SUMS_FILENAME)


def test_the_wait_limit_for_a_hand_over_is_thirty_minutes():
    assert staging_dataset.STAGED_DATASET_WAIT_SECONDS == 1800.0


def test_a_hand_over_that_never_settles_stops_the_wait_at_the_limit(tmp_path, capsys):
    bundle = _bundle(tmp_path)
    clock = Clock()
    emitter = FakeEmitterClient(bundle_receipts=[QUEUED])
    staged = staging_dataset.stage_bundle(
        bundle,
        emitter=emitter,
        repository=REPOSITORY,
        clock=clock,
        sleep=clock.sleep,
    )
    waited = clock.now - 1000.0
    assert waited >= staging_dataset.STAGED_DATASET_WAIT_SECONDS
    assert waited < (
        staging_dataset.STAGED_DATASET_WAIT_SECONDS
        + staging_dataset.RECEIPT_POLL_SECONDS
    )
    assert set(clock.slept) == {staging_dataset.RECEIPT_POLL_SECONDS}
    delivery = staged.delivery
    assert (delivery["status"], delivery["error_code"]) == ("failed", "UPLOAD_PENDING")
    assert validate_staged_dataset_delivery(delivery) == delivery
    err = capsys.readouterr().err
    assert "UPLOAD_PENDING" in err and "30 minutes" in err and "queued" in err


def test_the_wait_ends_with_the_first_attempt_after_the_hand_over(tmp_path):
    bundle = _bundle(tmp_path)
    clock = Clock()
    emitter = FakeEmitterClient(
        bundle_receipts=[
            QUEUED,
            QUEUED,
            QUEUED,
            _receipt("uploaded", revision="c" * 40),
            _receipt("uploaded", revision="d" * 40),
        ]
    )
    staged = staging_dataset.stage_bundle(
        bundle, emitter=emitter, repository=REPOSITORY, clock=clock, sleep=clock.sleep
    )
    assert staged.delivery["revision"] == "c" * 40
    assert emitter.calls.count("dataset_bundle_receipt") == 3
    assert len(clock.slept) == 2

    # A bundle handed over again joins its queued hand-over, whose receipt
    # already counts earlier attempts: the wait is for the next attempt.
    history = _receipt("pending", attempts=3, error_code="UPLOAD_FAILED")
    emitter = FakeEmitterClient(
        bundle_receipts=[
            history,
            history,
            _receipt("uploaded", attempts=4, revision="e" * 40),
        ]
    )
    staged = staging_dataset.stage_bundle(
        bundle, emitter=emitter, repository=REPOSITORY, clock=clock, sleep=clock.sleep
    )
    assert staged.delivery["status"] == "uploaded"
    assert staged.delivery["revision"] == "e" * 40


def test_the_build_may_rewrite_only_the_files_the_emitter_copied(tmp_path):
    bundle = _bundle(tmp_path)
    hub = FakeHub(tmp_path)
    copied = _stage(bundle, FakeEmitterClient(hub_api=hub))
    # Every file of this bundle is small, so the emitter copied them all.
    assert copied.copied_files == {item.name for item in bundle.files} | {
        MANIFEST,
        STAGED_MANIFEST_FILENAME,
        SHA256SUMS_FILENAME,
    }
    assert copied.may_rewrite(MANIFEST) and copied.may_rewrite(SHA256SUMS_FILENAME)

    # A file the emitter uploads from disk stays as handed over, whether or
    # not the hand-over settled: only a copy is safe to change.
    for outcome in (
        _receipt("uploaded", revision="a" * 40, copied=[SHA256SUMS_FILENAME]),
        _receipt("pending", error_code="UPLOAD_FAILED", copied=[SHA256SUMS_FILENAME]),
    ):
        emitter = FakeEmitterClient(
            bundle_receipts=[{**QUEUED, "copied_files": [SHA256SUMS_FILENAME]}, outcome]
        )
        staged = _stage(bundle, emitter)
        assert staged.copied_files == {SHA256SUMS_FILENAME}
        assert not staged.may_rewrite(MANIFEST)
        assert not staged.may_rewrite(DATASET)
        assert staged.may_rewrite(SHA256SUMS_FILENAME)
        # A file outside the bundle is the build's own.
        assert staged.may_rewrite("build.json")

    # An emitter that lists no copies leaves every bundle file as handed over.
    receipt = {key: value for key, value in QUEUED.items() if key != "copied_files"}
    staged = _stage(bundle, FakeEmitterClient(bundle_receipts=[receipt]))
    assert not staged.may_rewrite(MANIFEST)
    assert not staged.may_rewrite(SHA256SUMS_FILENAME)

    # Nothing was handed over in local-only mode.
    local = staging_dataset.StagedBundleHandOver(local_only_staged_dataset(bundle))
    assert local.may_rewrite(MANIFEST) and local.may_rewrite(SHA256SUMS_FILENAME)


def test_stage_bundle_requires_the_sidecars_first(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    emitter = FakeEmitterClient(hub_api=FakeHub(tmp_path))
    with pytest.raises(StagedDatasetError, match="write_sidecars"):
        _stage(bundle, emitter)
    assert emitter.bundle_requests == []


@pytest.mark.parametrize(
    ("environment", "offline"),
    [
        ({"HF_HUB_OFFLINE": "1"}, True),
        ({"HF_HUB_OFFLINE": "on"}, True),
        ({"HF_HUB_OFFLINE": "Yes"}, True),
        ({"HF_HUB_OFFLINE": "TRUE"}, True),
        ({"TRANSFORMERS_OFFLINE": "1"}, True),
        ({"HF_HUB_OFFLINE": "0"}, False),
        ({"HF_HUB_OFFLINE": ""}, False),
        ({}, False),
    ],
)
def test_hub_offline_reads_the_variable_huggingface_hub_reads(environment, offline):
    assert staging_dataset.hub_offline(environment) is offline


def _emitter_that_never_started(reason: str) -> FakeEmitterClient:
    from microcosm_emitter_interface import EmitterStartupError

    client = FakeEmitterClient(modules=())
    client.startup_error = EmitterStartupError(reason)
    return client


def _emitter_that_timed_out() -> FakeEmitterClient:
    from microcosm_emitter_interface import RequestFailure
    from microcosm_emitter_interface.bundles import unavailable_write_access

    client = FakeEmitterClient(write_access=unavailable_write_access())
    client.last_failure = RequestFailure("staging", "not_accepted", "TimeoutError")
    return client


def _outdated_emitter() -> object:
    from microcosm.build import telemetry_emitter

    return telemetry_emitter.OutdatedEmitterClient("0.5.1")


def test_a_refused_remote_stage_says_why_and_how_to_fix_it():
    from microcosm.build.telemetry_emitter import MINIMUM_EMITTER_VERSION

    # Each cause has its own reason and its own remedy.
    missing = staging_dataset.write_access_problem(
        _emitter_that_never_started("command_not_found"), REPOSITORY
    )
    assert "no microcosm-emitter command" in missing
    assert "`uv tool install microcosm-emitter`" in missing
    outdated = staging_dataset.write_access_problem(_outdated_emitter(), REPOSITORY)
    assert "0.5.1" in outdated and MINIMUM_EMITTER_VERSION in outdated
    upgrade = (
        f"uv tool install --upgrade 'microcosm-emitter>={MINIMUM_EMITTER_VERSION}'"
    )
    assert upgrade in outdated
    failed = staging_dataset.write_access_problem(
        _emitter_that_never_started("readiness_timeout"), REPOSITORY
    )
    assert "readiness_timeout" in failed and "upgrade" not in failed
    timed_out = staging_dataset.write_access_problem(
        _emitter_that_timed_out(), REPOSITORY
    )
    assert "did not answer" in timed_out and "TimeoutError" in timed_out
    assert "install" not in timed_out
    malformed = staging_dataset.write_access_problem(
        FakeEmitterClient(), "not a repository"
    )
    assert "owner/name" in malformed and "install" not in malformed
    module = staging_dataset.write_access_problem(
        FakeEmitterClient(modules={"telemetry", "orrery"}), REPOSITORY
    )
    assert "Hugging Face module" in module
    # Without an emitter a build can still keep its bundle local.
    for problem in (missing, outdated, failed, timed_out, module):
        assert "uploads the bundle" in problem


@pytest.mark.parametrize(
    ("emitter", "message"),
    [
        (None, "emitter"),
        (FakeEmitterClient(modules={"telemetry", "orrery"}), "emitter"),
        (
            FakeEmitterClient(
                write_access={
                    "status": "checked",
                    "credential": "missing",
                    "reachable": None,
                    "can_write": None,
                    "error_type": None,
                }
            ),
            "write credential",
        ),
        (
            FakeEmitterClient(
                write_access={
                    "status": "checked",
                    "credential": "present",
                    "reachable": False,
                    "can_write": None,
                    "error_type": "HfHubHTTPError",
                }
            ),
            "cannot reach policyengine/populace-uk-private (HfHubHTTPError)",
        ),
        (
            FakeEmitterClient(
                write_access={
                    "status": "checked",
                    "credential": "present",
                    "reachable": True,
                    "can_write": False,
                    "error_type": None,
                }
            ),
            "read-only",
        ),
    ],
)
def test_a_remote_stage_the_emitter_cannot_write_is_refused(emitter, message):
    problem = staging_dataset.write_access_problem(emitter, REPOSITORY)
    assert problem is not None and message in problem


def test_an_unread_write_scope_is_a_warning_not_a_refusal(capsys):
    unread = FakeEmitterClient(
        write_access={
            "status": "checked",
            "credential": "present",
            "reachable": True,
            "can_write": None,
            "error_type": None,
        }
    )
    assert staging_dataset.write_access_problem(unread, REPOSITORY) is None
    assert "write scope could not be read" in capsys.readouterr().err
    assert staging_dataset.write_access_problem(FakeEmitterClient(), REPOSITORY) is None
    assert capsys.readouterr().err == ""


def test_a_remote_bundle_that_differs_is_left_for_another_run_id(tmp_path, capsys):
    bundle = _bundle(tmp_path)
    emitter = FakeEmitterClient(
        bundle_receipts=[QUEUED, _receipt("failed", error_code="REMOTE_DIFFERS")]
    )
    assert _stage(bundle, emitter).delivery["error_code"] == "REMOTE_DIFFERS"
    err = capsys.readouterr().err
    assert "already holds a different bundle" in err
    assert "another run id" in err and "can be re-staged" not in err


def test_the_codes_of_a_bundle_that_may_still_arrive_are_record_codes():
    retrying = staging_dataset.RETRYING_ERROR_CODES
    assert retrying <= set(staging_dataset._ERROR_CODES)
    assert retrying == {
        "UPLOAD_PENDING",
        "REPOSITORY_UNAVAILABLE",
        "UPLOAD_FAILED",
        "CREDENTIAL_MISSING",
        "PRESERVED_FILES_MISSING",
    }
    for final in (
        "REMOTE_DIFFERS",
        "PRESERVED_BYTES_CHANGED",
        "RECEIPT_UNRECOGNIZED",
        "HANDOVER_REFUSED",
        "EMITTER_UNAVAILABLE",
    ):
        assert final in staging_dataset._ERROR_CODES and final not in retrying


def test_a_remote_bundle_this_directory_handed_over_is_not_sent_to_another_run_id(
    tmp_path, capsys
):
    """When the directory's outputs changed after an earlier hand-over, the
    folder may hold that earlier bundle: a second copy is not advised."""
    bundle = _bundle(tmp_path)
    earlier = {
        **bundle.digests(),
        "build_record.json": {"sha256": "c" * 64, "bytes": 1},
    }
    emitter = FakeEmitterClient(
        bundle_receipts=[QUEUED, _receipt("failed", error_code="REMOTE_DIFFERS")]
    )
    staged = _stage(bundle, emitter, earlier_files=earlier)
    assert staged.delivery["error_code"] == "REMOTE_DIFFERS"
    err = capsys.readouterr().err
    assert "may hold the bundle handed over earlier" in err
    assert "another run id" not in err


def _first_hand_over(bundle, hub, queue):
    """The build's emitter took the hand-over and stopped."""
    return _stage(
        bundle,
        FakeEmitterClient(
            hub_api=hub, queue=queue, deliver=False, answer_receipts=False
        ),
    )


def _resume(bundle, emitter, record):
    clock = Clock()
    return staging_dataset.resume_hand_over(
        bundle,
        emitter=emitter,
        record=record,
        repository=REPOSITORY,
        clock=clock,
        sleep=clock.sleep,
    )


def test_a_bundle_staged_again_resumes_the_earlier_hand_over(tmp_path):
    """The emitter still holds the earlier hand-over: its outcome is the
    record, and nothing is handed over again."""
    bundle = _bundle(tmp_path)
    hub = FakeHub(tmp_path)
    queue = FakeHandoverQueue()
    first = _first_hand_over(bundle, hub, queue)
    assert first.delivery["error_code"] == "UPLOAD_PENDING"
    assert first.handover_id in queue.jobs
    record = staging_dataset.handover_record(first)
    assert record == {
        "handover_id": first.handover_id,
        "repository": REPOSITORY,
        "run_id": RUN_ID,
        "run_dir": str(bundle.run_dir),
        "files": bundle.digests(),
    }
    # What the build recorded in its staging run is kept on its own.
    assert staging_dataset.at_build_record(first.delivery) == {
        "mode": "local_and_remote",
        "status": "failed",
        "error_code": "UPLOAD_PENDING",
    }
    # Rewriting the local sidecars does not change what was handed over.
    (bundle.run_dir / STAGED_MANIFEST_FILENAME).write_text('{"files": {}}')

    later = FakeEmitterClient(hub_api=hub, queue=queue)
    resumed = _resume(bundle, later, record)
    assert resumed is not None
    assert later.bundle_requests == []
    assert (resumed.delivery["status"], resumed.delivery["revision"]) == (
        "already_staged",
        hub.sha,
    )
    assert resumed.delivery["files"] == record["files"]
    assert resumed.handover_id == first.handover_id
    # The manifest was copied when the bundle was handed over.
    assert resumed.may_rewrite(MANIFEST)

    # An emitter that never had the hand-over cannot resume it.
    assert _resume(bundle, FakeEmitterClient(hub_api=hub), record) is None
    manifest = {"staged_dataset_handover": record}
    assert (
        staging_dataset.recorded_handover(
            manifest, repository=REPOSITORY, run_id=RUN_ID
        )
        == record
    )
    for repository, run_id in (("other/repo", RUN_ID), (REPOSITORY, "other-run")):
        assert (
            staging_dataset.recorded_handover(
                manifest, repository=repository, run_id=run_id
            )
            is None
        )


def test_a_hand_over_whose_files_are_gone_or_changed_is_not_resumed(tmp_path):
    """A job that has not landed and can never finish from its recorded paths
    is left to the emitter; the caller hands the directory over as a new job.
    A job that landed is recorded wherever the directory is now."""
    bundle = _bundle(tmp_path)
    hub = FakeHub(tmp_path)
    queue = FakeHandoverQueue()
    record = staging_dataset.handover_record(_first_hand_over(bundle, hub, queue))
    # The directory moved: the job names its files at the old paths, where
    # they are gone.
    moved = {**record, "run_dir": str(tmp_path / "elsewhere")}
    assert _resume(bundle, FakeEmitterClient(hub_api=hub, queue=queue), moved) is None
    # The emitter settled the job for good because its files changed.
    changed = FakeEmitterClient(
        bundle_receipts=[_receipt("failed", error_code="PRESERVED_BYTES_CHANGED")]
    )
    assert _resume(bundle, changed, record) is None
    # Files missing at the last attempt are back at their recorded paths (the
    # command only runs on a directory with every output): the job can finish.
    back = FakeEmitterClient(
        bundle_receipts=[
            _receipt("pending", error_code="PRESERVED_FILES_MISSING"),
            _receipt("uploaded", attempts=2, revision="d" * 40),
        ]
    )
    resumed = _resume(bundle, back, record)
    assert resumed is not None and resumed.delivery["revision"] == "d" * 40
    # A job that landed is recorded even after the directory moved.
    landed = FakeEmitterClient(
        bundle_receipts=[_receipt("uploaded", revision="e" * 40)]
    )
    resumed = _resume(bundle, landed, moved)
    assert resumed is not None
    assert (resumed.delivery["status"], resumed.delivery["revision"]) == (
        "already_staged",
        "e" * 40,
    )
    # The moved directory's files are not the job's, so any may change.
    assert resumed.may_rewrite(DATASET)


def test_the_builds_own_hand_over_refused_as_remote_differs_is_recorded(
    tmp_path, capsys
):
    """The emitter settled the build's own job: the folder holds another bundle
    for this run, for example one a re-stage of a copy of the directory sent
    first. That final code is the record, and no second run id is advised."""
    bundle = _bundle(tmp_path)
    record = staging_dataset.handover_record(
        _first_hand_over(bundle, FakeHub(tmp_path), FakeHandoverQueue())
    )
    settled = FakeEmitterClient(
        bundle_receipts=[_receipt("failed", error_code="REMOTE_DIFFERS")]
    )
    resumed = _resume(bundle, settled, record)
    assert resumed is not None
    assert (resumed.delivery["status"], resumed.delivery["error_code"]) == (
        "failed",
        "REMOTE_DIFFERS",
    )
    err = capsys.readouterr().err
    assert "the build's own hand-over" in err
    assert "another run id" not in err


def test_fetch_bundle_verifies_every_digest_and_copies_out_of_the_cache(tmp_path):
    bundle = _bundle(tmp_path)
    hub = FakeHub(tmp_path)
    _stage(bundle, FakeEmitterClient(hub_api=hub))

    dest = tmp_path / "fetched"
    paths = fetch_bundle(storage=_storage(hub), run_id=RUN_ID, dest=dest)
    assert {path.name for path in paths} == {
        "calibration_diagnostics.json",
        DATASET,
        "solve_diagnostics.csv",
        MANIFEST,
        STAGED_MANIFEST_FILENAME,
    }
    assert (dest / DATASET).read_bytes() == b"h5-bytes" * 100
    assert (dest / SHA256SUMS_FILENAME).is_file()

    only = fetch_bundle(
        storage=_storage(hub), run_id=RUN_ID, dest=tmp_path / "h5", h5_only=True
    )
    assert [path.name for path in only] == [DATASET]

    hub.files[f"staged/{RUN_ID}/solve_diagnostics.csv"] = b"tampered"
    with pytest.raises(StagedDatasetError, match="does not match"):
        fetch_bundle(storage=_storage(hub), run_id=RUN_ID, dest=tmp_path / "bad")
    assert not (tmp_path / "bad" / "solve_diagnostics.csv").exists()


def test_delivery_records_follow_their_mode(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    disabled = disabled_staged_dataset("--no-staging")
    assert disabled["mode"] == "disabled" and disabled["status"] == "skipped"
    assert disabled["files"] == {}
    local = local_only_staged_dataset(bundle)
    assert local["mode"] == "local_only" and local["status"] == "skipped"
    assert local["repository"] is None and local["files"] == bundle.digests()
    with pytest.raises(StagedDatasetError):
        disabled_staged_dataset("")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"status": "uploaded", "revision": None}, "records its revision"),
        ({"status": "failed", "error_code": None}, "error code"),
        ({"status": "skipped"}, "uploaded, already staged or failed"),
        ({"opt_out_reason": "x"}, "opt-out reason"),
        ({"error_code": "BOOM", "status": "failed"}, "unknown"),
        (
            {
                "mode": "local_only",
                "repository": None,
                "revision": None,
                "status": "uploaded",
            },
            "local-only",
        ),
        ({"extra": 1}, "keys differ"),
        ({"files": {"a.h5": {"sha256": "zz", "bytes": 1}}}, "invalid"),
    ],
)
def test_delivery_validation_refuses_contradictions(overrides, message):
    base = {
        "contract_version": 1,
        "mode": "local_and_remote",
        "repository": "policyengine/populace-uk-private",
        "prefix": f"staged/{RUN_ID}",
        "run_id": RUN_ID,
        "revision": "a" * 40,
        "status": "uploaded",
        "error_code": None,
        "opt_out_reason": None,
        "files": {"microcosm_uk_2025_local.h5": {"sha256": "0" * 64, "bytes": 3}},
    }
    assert validate_staged_dataset_delivery(base) == base
    with pytest.raises(StagedDatasetError, match=message):
        validate_staged_dataset_delivery({**base, **overrides})


def test_parse_sha256sums_rejects_paths_and_duplicates():
    good = f"{'a' * 64}  file.h5\n{'b' * 64}  other.json\n"
    assert parse_sha256sums(good) == [("a" * 64, "file.h5"), ("b" * 64, "other.json")]
    with pytest.raises(StagedDatasetError, match="malformed"):
        parse_sha256sums(f"{'a' * 64}  ../escape.h5\n")
    with pytest.raises(StagedDatasetError, match="twice"):
        parse_sha256sums(f"{'a' * 64}  file.h5\n{'b' * 64}  file.h5\n")
    with pytest.raises(StagedDatasetError, match="no files"):
        parse_sha256sums("\n")


def test_refresh_sha256sums_entry_redigests_one_listed_file(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    sums, _ = write_sidecars(bundle, repository=None)
    before = dict((name, sha) for sha, name in parse_sha256sums(sums.read_text()))
    (run_dir / MANIFEST).write_text(
        json.dumps(
            {**json.loads((run_dir / MANIFEST).read_text()), "staged_dataset": {}}
        )
    )
    digest = refresh_sha256sums_entry(run_dir, MANIFEST)
    after = dict((name, sha) for sha, name in parse_sha256sums(sums.read_text()))
    assert digest == _sha((run_dir / MANIFEST).read_bytes()) == after[MANIFEST]
    assert after[MANIFEST] != before[MANIFEST]
    assert {k: v for k, v in after.items() if k != MANIFEST} == {
        k: v for k, v in before.items() if k != MANIFEST
    }
    with pytest.raises(StagedDatasetError, match="does not list"):
        refresh_sha256sums_entry(run_dir, "run.log")


def test_delivery_covers_bundle_only_for_a_landed_record_of_these_outputs(tmp_path):
    bundle = _bundle(tmp_path)
    landed = {
        "contract_version": 1,
        "mode": "local_and_remote",
        "repository": REPOSITORY,
        "prefix": f"staged/{RUN_ID}",
        "run_id": RUN_ID,
        "revision": "a" * 40,
        "status": "uploaded",
        "error_code": None,
        "opt_out_reason": None,
        "files": bundle.digests(),
    }
    assert delivery_covers_bundle(landed, bundle, repository=REPOSITORY)
    assert delivery_covers_bundle(
        {**landed, "status": "already_staged"}, bundle, repository=REPOSITORY
    )
    assert not delivery_covers_bundle(landed, bundle, repository="other/repo")
    for error_code in ("UPLOAD_FAILED", "UPLOAD_PENDING", "EMITTER_UNAVAILABLE"):
        assert not delivery_covers_bundle(
            {**landed, "status": "failed", "revision": None, "error_code": error_code},
            bundle,
            repository=REPOSITORY,
        )
    assert not delivery_covers_bundle(None, bundle, repository=REPOSITORY)
