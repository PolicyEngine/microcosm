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
  ``pending``.

The emitter's own tests cover the process, its queues and the Hub.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from microcosm_emitter_interface import StageRunFiles
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

    emitter_version = "0.7.0"

    def __init__(
        self,
        *,
        modules: Iterable[str] = ALL_MODULES,
        accept_run_files: bool = True,
    ) -> None:
        self._modules = frozenset(modules)
        self.accept_run_files = accept_run_files
        self.closed = False
        self.startup_error = None
        self.last_failure = None
        #: Every call's name, in order.
        self.calls: list[str] = []
        self.telemetry_messages: list[dict[str, Any]] = []
        self.run_file_requests: list[StageRunFiles] = []
        self.orrery_requests: list[dict[str, Any]] = []
        #: What the dataset repositories hold: ``(repository, path)`` to bytes.
        self.hub_files: dict[tuple[str, str], bytes] = {}
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

    def close(self) -> None:
        self.calls.append("close")
        self.closed = True
