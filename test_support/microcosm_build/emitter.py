"""An in-memory stand-in for the interface package's ``EmitterClient``.

Build tests hand telemetry events, staging run files and Orrery inputs to it
instead of to the separately installed ``microcosm-emitter`` command. It
answers as the emitter documents those calls, without a process, socket,
database or network request:

- each accepted ``stage_run_files`` hand-over is committed at once, and
  ``hub_files`` holds what the dataset repository would then contain;
- ``read_back_run_files`` compares ``hub_files`` with the bytes this fake
  committed, so a test can alter ``hub_files`` to stand for a Hub copy that
  differs;
- Orrery inventories measure the named files, and hand-overs answer
  ``pending``;
- a staged dataset bundle is checked and its small files copied as the
  emitter does before it answers ``pending``. It becomes a job in a
  ``FakeHandoverQueue``, which several clients can share as emitters on one
  machine share their hand-over database: a hand-over identical to a queued
  one returns that job's receipt. Whenever a delivering client takes a
  hand-over or answers for a receipt, the queue's clock moves on one step
  and the client makes one upload attempt for each job that is due, the
  earliest due first, as the emitter claims them; a failed attempt makes the
  job due again after a wait that doubles each time. An attempt goes through ``hub_api``,
  an ``HfApi``-like fake Hub (without one, the job stays pending with
  ``CREDENTIAL_MISSING``, as on an emitter that finds no credential).
  ``deliver=False`` and ``answer_receipts=False`` stand for an emitter that
  took the hand-over and then stopped; ``bundle_receipts`` scripts the
  answers instead, and ``never_copy`` names files the emitter leaves on
  disk.

The emitter's own tests cover the process, its queues and the Hub.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from microcosm_emitter_interface import StageDatasetBundle, StageRunFiles
from microcosm_emitter_interface.bundles import (
    BUNDLE_INVENTORY_ROLE,
    SIDECAR_ROLES,
    DatasetBundleReceipt,
    WriteAccess,
    unavailable_bundle_receipt,
    unavailable_write_access,
)
from microcosm_emitter_interface.orrery import (
    PublicationFileRole,
    PublicationInventory,
    PublicationReceipt,
    publication_id_of,
    unavailable_receipt,
)
from microcosm_emitter_interface.staging import (
    ReadBackResult,
    RunFilesReceipt,
    unavailable_read_back,
    unavailable_run_files_receipt,
)
from microcosm_emitter_interface.telemetry import (
    TelemetryCommand,
    TelemetryRecorder,
    telemetry_message,
)

ALL_MODULES = frozenset({"telemetry", "orrery", "huggingface"})
LATEST_POINTER_PATH = "latest_staging.json"
RUNS_INDEX_PATH = "runs.json"
#: The emitter copies a bundle's smallest files, each of at most 32 MiB and
#: together at most 64 MiB, before it answers the hand-over.
MAX_COPIED_FILE_BYTES = 32 * 1024 * 1024
MAX_COPIED_BYTES = 64 * 1024 * 1024
#: What ``check_write_access`` answers for a credential that can write.
WRITABLE = {
    "status": "checked",
    "credential": "present",
    "reachable": True,
    "can_write": True,
    "error_type": None,
}


class _BundleRefusedError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class FakeHandoverQueue:
    """Bundle jobs by hand-over ID, with the step at which each is due.

    ``clock`` counts delivery steps. A new job is due at once; a failed
    attempt makes it due again after ``2 ** attempts`` steps (the emitter
    waits 2 seconds, doubling up to 300).
    """

    def __init__(self) -> None:
        #: Each job's request, copied bytes, latest receipt and due step.
        self.jobs: dict[str, dict[str, Any]] = {}
        self.clock = 0

    def receipt(self, handover_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(handover_id)
        return None if job is None else dict(job["receipt"])

    def due_jobs(self) -> list[dict[str, Any]]:
        """Pending jobs due now, the earliest due first, then the oldest."""
        pending = [
            (job["due"], order, job)
            for order, job in enumerate(self.jobs.values())
            if job["receipt"]["status"] == "pending" and job["due"] <= self.clock
        ]
        return [job for _, _, job in sorted(pending, key=lambda item: item[:2])]


def fake_publication_inventory(
    directory: Path,
    roles: Mapping[str, PublicationFileRole],
    publication_id: str | None = None,
) -> PublicationInventory:
    """Measure the named files as the emitter's inventory does, in name order."""
    return {
        "version": 1,
        "publication_id": publication_id or uuid.uuid4().hex,
        "files": [
            {
                "name": name,
                "role": role,
                "bytes": (Path(directory) / name).stat().st_size,
                "sha256": hashlib.sha256(
                    (Path(directory) / name).read_bytes()
                ).hexdigest(),
            }
            for name, role in sorted(roles.items())
        ],
    }


class FakeEmitterClient:
    """The ``EmitterClient`` calls Microcosm makes, answered in memory.

    ``accept_run_files=False`` stands for an emitter that does not answer a
    staging hand-over in time: each one returns an ``unavailable`` receipt,
    as the interface does after a timeout.
    """

    emitter_version = "0.11.9"

    def __init__(
        self,
        *,
        modules: Iterable[str] = ALL_MODULES,
        accept_run_files: bool = True,
        hub_files: dict[tuple[str, str], bytes] | None = None,
        hub_api: Any = None,
        bundle_receipts: Sequence[Mapping[str, Any]] | None = None,
        never_copy: Iterable[str] = (),
        write_access: Mapping[str, Any] | None = None,
        queue: FakeHandoverQueue | None = None,
        deliver: bool = True,
        answer_receipts: bool = True,
    ) -> None:
        self._modules = frozenset(modules)
        self.accept_run_files = accept_run_files
        #: The ``HfApi``-like fake Hub bundles are committed with, or None.
        self.hub_api = hub_api
        #: Scripted bundle receipts: the hand-over's answer, then each receipt
        #: asked for (the last one repeats).
        self.bundle_receipts = (
            None if bundle_receipts is None else list(bundle_receipts)
        )
        #: Files never copied at hand-over, whatever their size.
        self.never_copy = frozenset(never_copy)
        self.write_access = dict(WRITABLE if write_access is None else write_access)
        self.bundle_requests: list[StageDatasetBundle] = []
        #: Each hand-over's copied bytes, by hand-over ID and file name.
        self.copied_bundle_files: dict[str, dict[str, bytes]] = {}
        self._bundle_answers: dict[str, list[dict[str, Any]]] = {}
        #: The bundle jobs this emitter shares with others on the machine.
        self.queue = queue if queue is not None else FakeHandoverQueue()
        #: Whether this emitter makes upload attempts and answers receipts.
        self.deliver = deliver
        self.answer_receipts = answer_receipts
        self.closed = False
        self.startup_error = None
        self.last_failure = None
        #: Every call's name, in order.
        self.calls: list[str] = []
        self.telemetry_messages: list[dict[str, Any]] = []
        self.run_file_requests: list[StageRunFiles] = []
        self.orrery_requests: list[dict[str, Any]] = []
        #: What the dataset repositories hold: ``(repository, path)`` to bytes.
        #: A test may pass a fake Hub's own mapping to share it.
        self.hub_files = hub_files if hub_files is not None else {}
        #: Override for ``read_back_run_files``, when a test needs one.
        self.read_back_answer: ReadBackResult | None = None
        self._committed: dict[tuple[str, str], tuple[str, str]] = {}
        self._deliveries: dict[tuple[str, str], dict[str, Any]] = {}
        self.telemetry = TelemetryRecorder(self)

    @property
    def available(self) -> bool:
        return bool(self._modules) and not self.closed

    @property
    def modules(self) -> frozenset[str]:
        return self._modules if self.available else frozenset()

    def paths(self, repository: str) -> list[str]:
        return sorted(path for repo, path in self.hub_files if repo == repository)

    def hub_json(self, repository: str, path: str) -> Any:
        return json.loads(self.hub_files[(repository, path)])

    def committed_paths(self, repository: str, run_id: str) -> set[str]:
        return {
            path
            for (repo, path), (owner, _) in self._committed.items()
            if repo == repository and owner == run_id
        }

    # Telemetry ---------------------------------------------------------------

    def emit_telemetry(self, command: TelemetryCommand) -> bool:
        self.calls.append("emit_telemetry")
        if "telemetry" not in self.modules:
            return False
        self.telemetry_messages.append(telemetry_message(command))
        return True

    # Staging run files ---------------------------------------------------------

    def _delivery(self, repository: str, run_id: str) -> dict[str, Any]:
        return self._deliveries.setdefault(
            (repository, run_id),
            {
                "files_attempted": 0,
                "files_committed": 0,
                "commit_attempts": 0,
                "commits": 0,
                "last_error_code": None,
                "last_commit": None,
            },
        )

    def _receipt(self, repository: str, run_id: str) -> RunFilesReceipt:
        return {
            "status": "recorded",
            "repository": repository,
            "run_id": run_id,
            "pending_files": 0,
            **self._delivery(repository, run_id),
        }

    def stage_run_files(self, request: StageRunFiles) -> RunFilesReceipt:
        self.calls.append("stage_run_files")
        self.run_file_requests.append(request)
        if "huggingface" not in self.modules or not self.accept_run_files:
            return unavailable_run_files_receipt(request.repository, request.run_id)
        targets = [
            (f"{request.prefix}/{request.run_id}/{item.relative_path}", item.local_path)
            for item in request.files
        ]
        if request.latest_pointer is not None:
            targets.append((LATEST_POINTER_PATH, request.latest_pointer))
        for path_in_repo, local_path in targets:
            content = Path(local_path).read_bytes()
            self.hub_files[(request.repository, path_in_repo)] = content
            self._committed[(request.repository, path_in_repo)] = (
                request.run_id,
                hashlib.sha256(content).hexdigest(),
            )
        committed = len(targets)
        if request.run_index_entry is not None:
            self._merge_runs_index(request.repository, dict(request.run_index_entry))
            committed += 1
        delivery = self._delivery(request.repository, request.run_id)
        if committed:
            delivery["files_attempted"] += committed
            delivery["files_committed"] += committed
            delivery["commit_attempts"] += 1
            delivery["commits"] += 1
            delivery["last_commit"] = hashlib.sha1(
                str(len(self.run_file_requests)).encode()
            ).hexdigest()
        return self._receipt(request.repository, request.run_id)

    def _merge_runs_index(self, repository: str, entry: dict[str, Any]) -> None:
        existing = self.hub_files.get((repository, RUNS_INDEX_PATH))
        runs = [] if existing is None else json.loads(existing).get("runs", [])
        runs = [run for run in runs if run.get("run_id") != entry["run_id"]]
        runs.append(entry)
        runs.sort(
            key=lambda run: str(
                run.get("updated_at") or run.get("started_at") or run.get("run_id")
            ),
            reverse=True,
        )
        self.hub_files[(repository, RUNS_INDEX_PATH)] = json.dumps(
            {"schema_version": 1, "updated_at": entry.get("updated_at"), "runs": runs},
            indent=1,
        ).encode()

    def run_files_receipt(self, repository: str, run_id: str) -> RunFilesReceipt:
        self.calls.append("run_files_receipt")
        if "huggingface" not in self.modules:
            return unavailable_run_files_receipt(repository, run_id)
        return self._receipt(repository, run_id)

    def read_back_run_files(
        self, repository: str, run_id: str, *, wait_seconds: float = 30.0
    ) -> ReadBackResult:
        self.calls.append("read_back_run_files")
        if "huggingface" not in self.modules:
            return unavailable_read_back()
        if self.read_back_answer is not None:
            return self.read_back_answer
        committed = {
            path: digest
            for (repo, path), (owner, digest) in self._committed.items()
            if repo == repository and owner == run_id
        }
        if not committed:
            return {
                "status": "failed",
                "files": 0,
                "mismatched_paths": [],
                "error_code": "NO_COMMITTED_FILES",
            }
        mismatched = [
            path
            for path, digest in sorted(committed.items())
            if hashlib.sha256(self.hub_files.get((repository, path), b"")).hexdigest()
            != digest
        ]
        return {
            "status": "failed" if mismatched else "passed",
            "files": len(committed),
            "mismatched_paths": mismatched,
            "error_code": None,
        }

    # Orrery inputs -------------------------------------------------------------

    def orrery_inventory(
        self,
        directory: Path,
        roles: Mapping[str, PublicationFileRole],
        *,
        publication_id: str | None = None,
    ) -> PublicationInventory | None:
        self.calls.append("orrery_inventory")
        if "orrery" not in self.modules:
            return None
        return fake_publication_inventory(directory, roles, publication_id)

    def hand_over_orrery_inputs(
        self,
        directory: Path,
        inventory: PublicationInventory,
        *,
        wait_seconds: float = 30.0,
    ) -> PublicationReceipt:
        self.calls.append("hand_over_orrery_inputs")
        publication_id = publication_id_of(inventory)
        if "orrery" not in self.modules:
            return unavailable_receipt(publication_id)
        self.orrery_requests.append(
            {
                "action": "graph_publication",
                "directory": str(directory),
                "inventory": inventory,
            }
        )
        return {
            "version": 1,
            "publication_id": publication_id,
            "status": "pending",
            "error_code": None,
        }

    # Staged dataset bundles -----------------------------------------------------

    def stage_dataset_bundle(
        self, request: StageDatasetBundle, *, wait_seconds: float = 0
    ) -> DatasetBundleReceipt:
        self.calls.append("stage_dataset_bundle")
        self.bundle_requests.append(request)
        if "huggingface" not in self.modules:
            return unavailable_bundle_receipt(request)
        handover_id = request.handover_id()
        if self.bundle_receipts is not None:
            answers = [
                {"handover_id": handover_id, **dict(receipt)}
                for receipt in self.bundle_receipts
            ]
            self._bundle_answers[handover_id] = answers[1:] or answers
            return answers[0]  # type: ignore[return-value]
        try:
            copied = self._copy_small_files(request)
        except _BundleRefusedError as refusal:
            return {
                "handover_id": "",
                "status": "refused",
                "error_code": refusal.code,
                "revision": None,
                "attempts": 0,
                "repository": request.repository,
                "prefix": "",
                "copied_files": [],
            }
        self.copied_bundle_files[handover_id] = copied
        known = self.queue.receipt(handover_id)
        if known is not None:
            # The emitter applies a repeated hand-over once.
            return known  # type: ignore[return-value]
        queued: dict[str, Any] = {
            "handover_id": handover_id,
            "status": "pending",
            "error_code": None,
            "revision": None,
            "attempts": 0,
            "repository": request.repository,
            "prefix": request.remote_prefix,
            "copied_files": sorted(copied),
        }
        self.queue.jobs[handover_id] = {
            "request": request,
            "copied": copied,
            "receipt": dict(queued),
            "due": self.queue.clock,
        }
        self._deliver_pending_jobs()
        return queued  # type: ignore[return-value]

    def dataset_bundle_receipt(self, handover_id: str) -> DatasetBundleReceipt | None:
        self.calls.append("dataset_bundle_receipt")
        if "huggingface" not in self.modules or not self.answer_receipts:
            return None
        if self.bundle_receipts is not None and handover_id not in self._bundle_answers:
            # A scripted emitter that holds a hand-over made elsewhere.
            self._bundle_answers[handover_id] = [
                {"handover_id": handover_id, **dict(receipt)}
                for receipt in self.bundle_receipts
            ]
        answers = self._bundle_answers.get(handover_id)
        if answers:
            return dict(answers.pop(0) if len(answers) > 1 else answers[0])  # type: ignore[return-value]
        self._deliver_pending_jobs()
        return self.queue.receipt(handover_id)  # type: ignore[return-value]

    def _deliver_pending_jobs(self) -> None:
        """One delivery step: an upload attempt for each job that is due."""
        if not self.deliver:
            return
        self.queue.clock += 1
        for job in self.queue.due_jobs():
            receipt = job["receipt"]
            status, error_code, revision = self._upload_bundle(
                job["request"], job["copied"]
            )
            attempts = receipt["attempts"] + 1
            receipt.update(
                status=status,
                error_code=error_code,
                revision=revision,
                attempts=attempts,
            )
            job["due"] = self.queue.clock + 2**attempts

    def deliver_now(self) -> None:
        """Stand for this emitter's worker running once, with no caller."""
        self._deliver_pending_jobs()

    def check_write_access(self, repository: str) -> WriteAccess:
        self.calls.append("check_write_access")
        if "huggingface" not in self.modules:
            return unavailable_write_access()
        return dict(self.write_access)  # type: ignore[return-value]

    def _copy_small_files(self, request: StageDatasetBundle) -> dict[str, bytes]:
        """Check each file's size, then copy the smallest, as the emitter does."""
        for item in request.files:
            path = Path(item.local_path)
            if not path.is_file():
                raise _BundleRefusedError("LOCAL_FILE_MISSING")
            if path.stat().st_size != item.bytes:
                raise _BundleRefusedError("FILE_SIZE_DIFFERS")
        copied: dict[str, bytes] = {}
        total = 0
        for item in sorted(request.files, key=lambda entry: (entry.bytes, entry.name)):
            if item.name in self.never_copy:
                continue
            if (
                item.bytes > MAX_COPIED_FILE_BYTES
                or total + item.bytes > MAX_COPIED_BYTES
            ):
                break
            content = Path(item.local_path).read_bytes()
            if hashlib.sha256(content).hexdigest() != item.sha256:
                raise _BundleRefusedError("FILE_DIGEST_DIFFERS")
            copied[item.name] = content
            total += item.bytes
        return copied

    def _upload_bundle(
        self, request: StageDatasetBundle, copied: Mapping[str, bytes]
    ) -> tuple[str, str | None, str | None]:
        """One upload attempt as the emitter documents it: status, code, revision.

        The bundle goes under ``<prefix>/<run_id>/`` in one commit; a Hub copy
        of its ``staged_manifest`` with the same outputs is ``already_staged``
        with that copy's commit, and a different one ``REMOTE_DIFFERS``.
        """
        api = self.hub_api
        if api is None:
            return "pending", "CREDENTIAL_MISSING", None
        for item in request.files:
            if item.name in copied:
                continue
            path = Path(item.local_path)
            if not path.is_file():
                return "pending", "PRESERVED_FILES_MISSING", None
            if hashlib.sha256(path.read_bytes()).hexdigest() != item.sha256:
                return "failed", "PRESERVED_BYTES_CHANGED", None
        repository, prefix = request.repository, request.remote_prefix
        inventory = next(
            item for item in request.files if item.role == BUNDLE_INVENTORY_ROLE
        )
        remote_inventory = f"{prefix}/{inventory.name}"
        outputs = {
            item.name: item.sha256
            for item in request.files
            if item.role not in SIDECAR_ROLES
        }
        try:
            if api.file_exists(
                repo_id=repository, filename=remote_inventory, repo_type="dataset"
            ):
                local = api.hf_hub_download(
                    repo_id=repository, filename=remote_inventory, repo_type="dataset"
                )
                remote = json.loads(Path(local).read_text())
                remote_files = remote.get("files") if isinstance(remote, dict) else None
                if (
                    not isinstance(remote_files, dict)
                    or {
                        name: (entry or {}).get("sha256")
                        for name, entry in remote_files.items()
                    }
                    != outputs
                ):
                    return "failed", "REMOTE_DIFFERS", None
                return (
                    "already_staged",
                    None,
                    self._last_commit(api, repository, remote_inventory),
                )
            parent = api.repo_info(repo_id=repository, repo_type="dataset").sha
            info = api.create_commit(
                repo_id=repository,
                operations=[
                    SimpleNamespace(
                        path_in_repo=f"{prefix}/{item.name}",
                        path_or_fileobj=copied.get(item.name, str(item.local_path)),
                    )
                    for item in request.files
                ],
                commit_message=request.commit_message
                or f"Stage {request.run_id} under {prefix}",
                repo_type="dataset",
                parent_commit=parent,
            )
        except Exception:
            return "pending", "UPLOAD_FAILED", None
        return "uploaded", None, info.oid

    @staticmethod
    def _last_commit(api: Any, repository: str, path: str) -> str | None:
        paths_info = getattr(api, "get_paths_info", None)
        if callable(paths_info):
            for entry in paths_info(
                repo_id=repository, paths=[path], expand=True, repo_type="dataset"
            ):
                return entry.last_commit.oid
        return api.repo_info(repo_id=repository, repo_type="dataset").sha

    def close(self) -> None:
        self.calls.append("close")
        self.closed = True
