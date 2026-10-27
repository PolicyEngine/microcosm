"""Microcosm's build adapter for the separately installed local emitter.

A build imports only ``microcosm-emitter-interface``. Starting the adapter
starts the ``microcosm-emitter`` command, installed on its own (for example
``uv tool install microcosm-emitter``), as this build's child process, with
three modules: telemetry, Orrery graph publication, and the Hugging Face
hand-over of staging run files. The interface finds the command through
``MICROCOSM_EMITTER_COMMAND``, then ``PATH``.

The child inherits the build's environment variables and runs as the same
user, so it finds the build's Hugging Face credential (``HF_TOKEN``, then
``HUGGINGFACE_TOKEN``, then the token saved by ``hf auth login``). Its module
settings travel on its standard input, never in environment variables or
process arguments. No call raises into the build: a missing, slow,
refusing or outdated emitter leaves the build running without it.

The adapter stops an emitter older than ``MINIMUM_EMITTER_VERSION`` as soon
as it has started, after reading its version from the emitter's ``hello``
answer (``EmitterClient.emitter_version``).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from microcosm_emitter_interface import (
    EmitterClient,
    HuggingFaceSettings,
    OrrerySettings,
    TelemetryRun,
    TelemetrySettings,
    default_spool_path,
)
from microcosm_emitter_interface.orrery import (
    PublicationFileRole,
    PublicationInventory,
    PublicationReceipt,
    publication_id_of,
    unavailable_receipt,
)
from microcosm_emitter_interface.telemetry import (
    DEFAULT_FAILURE_CLASS,
    DEFAULT_HEARTBEAT_SECONDS,
    TelemetryEventType,
    TelemetryStatus,
)

from microcosm.build.telemetry_identity import runtime_identity

__all__ = [
    "DEFAULT_HEARTBEAT_SECONDS",
    "MINIMUM_EMITTER_VERSION",
    "EmitterClient",
    "LocalTelemetryEmitter",
    "TelemetryRun",
    "publication_inventory",
    "publish_graph",
    "start_local_telemetry_emitter_service",
]

SETUP_WARNING = (
    "Local Microcosm emitter was not started ({error_type}); the build will continue."
)
#: The emitter took about four seconds to become ready on an M1 Max; this is
#: only a ceiling on the wait.
STARTUP_TIMEOUT_SECONDS = 15.0
#: The first emitter release with everything a build hands over: the Hugging
#: Face module (``stage_run_files``, ``run_files_receipt`` and
#: ``read_back_run_files``) and its queue in ``events.handovers.sqlite3``.
#: Releases 0.4.0 to 0.5.1 accept the same protocol version but start without
#: that module; 0.7.0 added only release publication, which builds do not use.
MINIMUM_EMITTER_VERSION = "0.6.0"
OUTDATED_EMITTER_WARNING = (
    "The installed microcosm-emitter {reported}; this build needs {required} "
    "or later, so it stopped that emitter and will continue without it. "
    "Upgrade it with `uv tool install --upgrade 'microcosm-emitter>={required}'`, "
    "or point MICROCOSM_EMITTER_COMMAND at a newer one."
)
_RELEASE_NUMBER = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_MINIMUM_RELEASE = tuple(int(part) for part in MINIMUM_EMITTER_VERSION.split("."))


def _release_number(version: object) -> tuple[int, ...] | None:
    """``(major, minor, patch)`` of a release version, or None if it has none."""
    found = _RELEASE_NUMBER.match(version) if isinstance(version, str) else None
    return None if found is None else tuple(int(part) for part in found.groups())


def _start_emitter(**options: Any) -> EmitterClient:
    """Start the installed emitter; stop it again if it is too old.

    ``EmitterClient.start`` returns only after the emitter's ``hello`` answer,
    which names its release. An emitter that did not start is returned as it
    is; an outdated one is closed, so every call reports it unavailable.
    """
    client = EmitterClient.start(**options)
    if not client.available:
        return client
    installed = _release_number(client.emitter_version)
    if installed is not None and installed >= _MINIMUM_RELEASE:
        return client
    reported = (
        "reports no version"
        if client.emitter_version is None
        else f"reports version {client.emitter_version}"
    )
    print(
        OUTDATED_EMITTER_WARNING.format(
            reported=reported, required=MINIMUM_EMITTER_VERSION
        ),
        file=sys.stderr,
    )
    client.close()
    return client


class LocalTelemetryEmitter:
    """One build's emitter: its telemetry run and the client that reaches it.

    ``client`` is the interface's ``EmitterClient`` for the emitter this build
    started, or ``None`` when the build could not prepare one. Staging writers
    hand their run files to the same client.

    Recording the run's outcome (``complete``, ``fail`` or ``block``) shuts
    the emitter down unless ``close_service_on_finish`` is false; a build that
    still hands work over afterwards passes false and calls ``close``.
    """

    def __init__(
        self,
        *,
        run: TelemetryRun,
        client: EmitterClient | None,
        close_service_on_finish: bool = True,
    ) -> None:
        self.run = run
        self.client = client
        self._close_service_on_finish = close_service_on_finish
        self._closed = False

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        country_code: str,
        pipeline: str,
        candidate_id: str | None = None,
        release_id: str | None = None,
        run_kind: str = "build",
        development_collector_url: str | None = None,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        startup_timeout_seconds: float = STARTUP_TIMEOUT_SECONDS,
        spool_path: Path | str | None = None,
        close_service_on_finish: bool = True,
    ) -> LocalTelemetryEmitter:
        """Start the installed emitter; a failure leaves an unavailable adapter."""
        run = TelemetryRun(
            run_id=run_id,
            country_code=country_code,
            pipeline=pipeline,
            candidate_id=candidate_id,
            release_id=release_id,
            run_kind=run_kind,
        )
        try:
            path = Path(spool_path) if spool_path else default_spool_path()
            identity = runtime_identity()
        except Exception as error:
            print(
                SETUP_WARNING.format(error_type=type(error).__name__),
                file=sys.stderr,
            )
            return cls(
                run=run, client=None, close_service_on_finish=close_service_on_finish
            )
        client = _start_emitter(
            telemetry=TelemetrySettings(
                run=run,
                identity=identity,
                development_collector_url=development_collector_url,
                heartbeat_seconds=heartbeat_seconds,
            ),
            orrery=OrrerySettings(development_collector_url=development_collector_url),
            # Its queue is the hand-over database beside the spool database
            # (``events.handovers.sqlite3``), so emitters of every release can
            # share the spool database on one machine.
            hugging_face=HuggingFaceSettings(),
            spool_path=path,
            startup_timeout=startup_timeout_seconds,
        )
        return cls(
            run=run, client=client, close_service_on_finish=close_service_on_finish
        )

    @property
    def available(self) -> bool:
        """Whether this build's emitter is running and not yet closed."""
        return self.client is not None and self.client.available and not self._closed

    def _telemetry(self):
        return self.client.telemetry if self.available else None

    def emit(
        self,
        *,
        event_type: TelemetryEventType,
        status: TelemetryStatus,
        stage_id: str | None = None,
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if (telemetry := self._telemetry()) is not None:
            telemetry.emit(
                event_type=event_type,
                status=status,
                stage_id=stage_id,
                message=message,
                details=details,
            )

    def stage(
        self,
        stage_id: str,
        *,
        status: TelemetryStatus = "started",
        message: str | None = None,
        **details: Any,
    ) -> None:
        if (telemetry := self._telemetry()) is not None:
            telemetry.stage(stage_id, status=status, message=message, **details)

    def transition_stage(
        self,
        stage_id: str,
        *,
        status: str = "running",
        message: str | None = None,
        **details: Any,
    ) -> None:
        if (telemetry := self._telemetry()) is not None:
            telemetry.transition_stage(
                stage_id, status=status, message=message, **details
            )

    def progress(
        self,
        stage_id: str,
        *,
        done: int,
        total: int,
        unit: str | None = None,
        **details: Any,
    ) -> None:
        if (telemetry := self._telemetry()) is not None:
            telemetry.progress(stage_id, done=done, total=total, unit=unit, **details)

    def calibration_progress(self, event: Mapping[str, Any]) -> None:
        if (telemetry := self._telemetry()) is not None:
            telemetry.calibration_progress(event)

    def transition_calibration_progress(self, event: Mapping[str, Any]) -> None:
        if (telemetry := self._telemetry()) is not None:
            telemetry.transition_calibration_progress(event)

    def fail(
        self,
        error: BaseException,
        *,
        failed_during: str | None = None,
        failure_class: str = DEFAULT_FAILURE_CLASS,
        error_code: str | None = None,
    ) -> None:
        try:
            if (telemetry := self._telemetry()) is not None:
                telemetry.fail(
                    error,
                    failed_during=failed_during,
                    failure_class=failure_class,
                    error_code=error_code,
                )
        finally:
            self._finish()

    def block(
        self,
        *,
        phase: str,
        blocking_gate_ids: Sequence[str],
        blocking_failure_count: int | None = None,
        gate_statuses: Mapping[str, str] | None = None,
    ) -> None:
        """Record that the build's validation checks refused its candidate."""
        try:
            if (telemetry := self._telemetry()) is not None:
                telemetry.refuse(
                    phase=phase,
                    refusing_check_ids=list(blocking_gate_ids),
                    refusing_check_count=blocking_failure_count,
                    check_statuses=gate_statuses,
                )
        finally:
            self._finish()

    def complete(self) -> None:
        try:
            if (telemetry := self._telemetry()) is not None:
                telemetry.complete()
        finally:
            self._finish()

    def _finish(self) -> None:
        if self._close_service_on_finish:
            self.close()

    def close(self) -> None:
        """Ask the emitter to finish its hand-overs and stop; repeats do nothing."""
        if self._closed:
            return
        self._closed = True
        if self.client is not None:
            self.client.close()

    def publication_inventory(
        self, directory: Path, roles: Mapping[str, PublicationFileRole]
    ) -> PublicationInventory | None:
        """Ask this build's emitter to measure selected public files."""
        if not self.available:
            return None
        return self.client.orrery_inventory(directory, roles)

    def publish_graph(
        self,
        directory: Path,
        inventory: PublicationInventory,
        *,
        wait_seconds: float = 30.0,
    ) -> PublicationReceipt:
        """Queue the measured files; wait only after the emitter queued them."""
        if not self.available:
            return unavailable_receipt(publication_id_of(inventory))
        return self.client.hand_over_orrery_inputs(
            directory, inventory, wait_seconds=wait_seconds
        )


def _orrery_emitter() -> EmitterClient:
    """A short-lived emitter that runs only the Orrery module."""
    return _start_emitter(
        orrery=OrrerySettings(), startup_timeout=STARTUP_TIMEOUT_SECONDS
    )


def publication_inventory(
    directory: Path, roles: Mapping[str, PublicationFileRole]
) -> PublicationInventory | None:
    """Measure graph files when the build has no emitter of its own."""
    client = _orrery_emitter()
    try:
        return client.orrery_inventory(directory, roles)
    finally:
        client.close()


def publish_graph(
    directory: Path,
    inventory: PublicationInventory,
    *,
    wait_seconds: float = 30.0,
) -> PublicationReceipt:
    """Queue graph files when the build has no emitter of its own.

    The emitter's queue is durable, so a pending job is delivered by a later
    emitter or by ``microcosm-emitter-publish-graph``.
    """
    client = _orrery_emitter()
    try:
        return client.hand_over_orrery_inputs(
            directory, inventory, wait_seconds=wait_seconds
        )
    finally:
        client.close()


def start_local_telemetry_emitter_service(**run: Any) -> LocalTelemetryEmitter:
    """Start this build's emitter through the interface."""
    return LocalTelemetryEmitter.start(**run)
