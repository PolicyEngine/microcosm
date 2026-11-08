"""Microcosm's build adapter for the separately installed local emitter.

A build imports only ``microcosm-emitter-interface``. Starting the adapter
starts the ``microcosm-emitter`` command, installed on its own (for example
``uv tool install microcosm-emitter``), as this build's child process, with
three modules: telemetry, Orrery graph publication, and the Hugging Face
hand-over of staging run files and staged dataset bundles. The interface
finds the command through ``MICROCOSM_EMITTER_COMMAND``, then ``PATH``.

The child inherits the build's environment variables and runs as the same
user, so it finds the build's Hugging Face credential (``HF_TOKEN``, then
``HUGGINGFACE_TOKEN``, then the token saved by ``hf auth login``). Its module
settings travel on its standard input, never in environment variables or
process arguments. No call raises into the build: a missing, slow,
refusing or outdated emitter leaves the build running without it.

Starting the adapter is the exception, and only through
``MICROCOSM_TELEMETRY_REQUIRED``. With ``1`` (a build that must send
telemetry) it raises ``TelemetryRequiredError`` when the emitter cannot be
used, so the build stops before doing any work; any value other than ``1``,
``0`` or unset raises ``ValueError``. For a startup failure that
``EmitterClient.start`` reports itself, the interface first prints that the
build will continue; the adapter's error says that line does not apply.
Leaving the requirement off (unset or ``0``) exists only for the
convenience of people outside PolicyEngine who run Microcosm without access
to the collector; PolicyEngine's own builds keep telemetry required.

The adapter never starts an emitter older than ``MINIMUM_EMITTER_VERSION``.
It reads the version from ``microcosm-emitter version`` first, because the
emitter's telemetry module records the run's start as soon as the emitter
starts, before its ``hello`` answer can name its release.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Literal, Protocol

from microcosm_emitter_interface import (
    COMMAND_ENVIRONMENT_VARIABLE,
    DEFAULT_COMMAND,
    PROTOCOL_VERSION,
    EmitterClient,
    EmitterStartupError,
    HuggingFaceSettings,
    OrrerySettings,
    RequestFailure,
    StageDatasetBundle,
    TelemetryRun,
    TelemetrySettings,
    default_spool_path,
)
from microcosm_emitter_interface.bundles import DatasetBundleReceipt, WriteAccess
from microcosm_emitter_interface.launcher import StartupFailureReason, resolve_command
from microcosm_emitter_interface.orrery import (
    PublicationFileRole,
    PublicationInventory,
    PublicationReceipt,
    publication_id_of,
    unavailable_receipt,
)
from microcosm_emitter_interface.protocol import HUGGING_FACE_MODULE, TELEMETRY_MODULE
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
    "TELEMETRY_REQUIRED_VARIABLE",
    "BundleEmitter",
    "EmitterClient",
    "LocalTelemetryEmitter",
    "OutdatedEmitterClient",
    "TelemetryRequiredError",
    "TelemetryRun",
    "emitter_problem",
    "publication_inventory",
    "publish_graph",
    "start_bundle_emitter",
    "start_local_telemetry_emitter_service",
]

SETUP_WARNING = (
    "Local Microcosm emitter was not started ({error_type}); the build will continue."
)
#: The emitter took about four seconds to become ready on an M1 Max; this is
#: only a ceiling on the wait.
STARTUP_TIMEOUT_SECONDS = 15.0
#: The package data file that holds the oldest emitter release a build
#: starts, ``MINIMUM_EMITTER_VERSION``. It is the one place the release is
#: written: the Modal stage runner (``tools/modal_us_stage_plan.py``), which
#: imports nothing from the workspace, reads the same file by its path in the
#: checkout that launches it, and installs at least that release in its image.
MINIMUM_EMITTER_VERSION_RESOURCE = "minimum_emitter_version.json"
_RELEASE_NUMBER = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _packaged_minimum_emitter_version() -> str:
    """The release ``MINIMUM_EMITTER_VERSION_RESOURCE`` names, as ``X.Y.Z``.

    A file without exactly that is a packaging defect, so importing this
    module fails rather than start an emitter of any release.
    """
    resource = importlib_resources.files("microcosm.build").joinpath(
        MINIMUM_EMITTER_VERSION_RESOURCE
    )
    data = json.loads(resource.read_text(encoding="utf-8"))
    version = data.get("minimum_emitter_version") if isinstance(data, dict) else None
    if not isinstance(version, str) or _RELEASE_NUMBER.fullmatch(version) is None:
        raise ValueError(
            f"microcosm/build/{MINIMUM_EMITTER_VERSION_RESOURCE} must hold "
            '{"minimum_emitter_version": "X.Y.Z"}'
        )
    return version


#: The oldest emitter release a build starts: 0.13.0 since 2026-11-05, the
#: newest release then. 0.13.0 added ``microcosm-emitter status``, which the
#: Modal stage runner runs in the container after a stage's tool to record how
#: many telemetry events each run left undelivered, so the image needs it; a
#: build requires the same release, so that the release is written once
#: (``MINIMUM_EMITTER_VERSION_RESOURCE``). Practically no machine ran the
#: separately installed emitter yet, so refusing 0.11.9 to 0.12.5 cost
#: nothing then. Earlier floors: 0.11.9 from 2026-11-02, because since 0.10.0
#: the emitter sends to the collector's hostname by default, which stays the
#: same when the collector moves to another Google Cloud project, while
#: earlier releases send to its Cloud Run address, which changes with the
#: move; 0.11.1 fixed two cases in which the telemetry module stopped. A
#: refused emitter costs a build its hosted telemetry, its staging hand-overs
#: and its Orrery graph publication, fails a UK build given
#: ``--staging-read-back``, and refuses a UK build that asks for remote
#: staging of its dataset bundle (``OUTDATED_EMITTER_WARNING``), so a later
#: rise weighs that against how many machines would have to upgrade.
#: 0.6.0 added the Hugging Face module (``stage_run_files``,
#: ``run_files_receipt`` and ``read_back_run_files`` for staging run files;
#: ``stage_dataset_bundle``, ``dataset_bundle_receipt`` and
#: ``check_write_access`` for staged dataset bundles) and its queue in
#: ``events.handovers.sqlite3``; releases 0.4.0 to 0.5.1 accept the same
#: protocol version but start without that module.
MINIMUM_EMITTER_VERSION = _packaged_minimum_emitter_version()
UPGRADE_EMITTER_COMMAND = (
    f"uv tool install --upgrade 'microcosm-emitter>={MINIMUM_EMITTER_VERSION}'"
)
INSTALL_EMITTER_COMMAND = "uv tool install microcosm-emitter"
#: Printed instead of starting an older emitter. Without one, staging run
#: files are still written locally, the UK graph step records a failed
#: receipt (``local_service_unavailable``) and creates no publication job,
#: ``StagingRunBundleWriterV2.verify_remote`` fails because no files were
#: handed over, and the UK pre-flight refuses a remote stage of the dataset
#: bundle (``rowwise_staging.preflight_staged_dataset``), naming the upgrade
#: command (``emitter_problem``).
OUTDATED_EMITTER_WARNING = (
    "The installed microcosm-emitter {reported}; this build needs {required} "
    "or later, so it did not start that emitter. The build continues without "
    "hosted telemetry or staging hand-overs and queues no Orrery graph "
    "publication; a UK build given --staging-read-back will fail at the "
    "read-back. But a UK build that asks for remote staging of its dataset "
    "bundle is refused before it starts; UK builds ask for it by default, and "
    "--staging-local-only or --no-staged-dataset keeps the bundle local. "
    "Upgrade it with `{upgrade}`, or point MICROCOSM_EMITTER_COMMAND at a "
    "newer one."
)
_MINIMUM_RELEASE = tuple(int(part) for part in MINIMUM_EMITTER_VERSION.split("."))

#: ``1``: this build must send telemetry, so it stops when its emitter cannot
#: be used. Unset or ``0``: it continues without one, which exists only for
#: the convenience of people outside PolicyEngine who run Microcosm without
#: access to the collector; PolicyEngine's own builds keep telemetry
#: required. Whoever starts the build sets it; the adapter reads it each time
#: it starts.
TELEMETRY_REQUIRED_VARIABLE = "MICROCOSM_TELEMETRY_REQUIRED"
REQUIRED_TELEMETRY_ERROR = (
    "{variable}=1, so this build must send telemetry, but {problem}"
    "{interface_line} The build stops before doing any work."
)
#: Added for a startup failure ``EmitterClient.start`` reports: interface
#: 0.7.0 prints "Local Microcosm emitter failed to start (<reason>); the
#: build will continue." first, and has no setting to leave it out.
INTERFACE_LINE_SUPERSEDED = (
    ' The line above saying "the build will continue" comes from '
    "microcosm-emitter-interface, which cannot tell that this build must send "
    "telemetry; it does not apply."
)
#: The reasons the adapter finds itself, before or after
#: ``EmitterClient.start``; the interface's own are ``StartupFailureReason``.
type AdapterFailureReason = Literal[
    "setup_failure",
    "version_timeout",
    "version_failed",
    "outdated_emitter",
    "hello_failed",
    "telemetry_module_not_loaded",
]
type TelemetryRequiredReason = StartupFailureReason | AdapterFailureReason
#: What each startup failure the interface reports means, in the error of a
#: build that must send telemetry; ``{command}`` is the command the adapter
#: resolved and the interface started. The interface names a fixed reason
#: and, for an emitter that exited, its exit code; never the emitter's
#: output. ``microcosm-emitter version`` has listed protocol version 1 by
#: then, so an ``unsupported_protocol_version`` is a failed ``hello``.
_STARTUP_PROBLEMS: Mapping[StartupFailureReason, str] = {
    "command_not_found": (
        "the emitter command (`{command}`) was no longer found when the "
        f"interface started it. Install it with `{INSTALL_EMITTER_COMMAND}`, or "
        "point MICROCOSM_EMITTER_COMMAND at it."
    ),
    "unsupported_protocol_version": (
        "its emitter (`{command}`) became ready, but its `hello` answer failed, "
        "timed out or could not be read, although `{command} version` lists "
        "protocol version 1 (the interface reports unsupported_protocol_version)."
    ),
    "child_exit": (
        "its emitter (`{command}`) exited before it was ready "
        '(child_exit{exit_detail}). Emitter 0.11.0 and later first print "The '
        'Microcosm emitter cannot start: …" when MICROCOSM_COLLECTOR_TOKEN, '
        "MICROCOSM_COLLECTOR_TOKEN_EXPIRES_AT and MICROCOSM_COLLECTOR_ORIGIN are "
        "set only in part, cannot be read, or name another collector. For any "
        'other cause in its host, the emitter prints "Service startup failed '
        '(startup_failure).", without the cause; an emitter installation that '
        "cannot be imported prints a Python traceback instead."
    ),
    "readiness_timeout": (
        "its emitter (`{command}`) was not ready within its startup time limit "
        "(readiness_timeout)."
    ),
    "configuration_timeout": (
        "its emitter (`{command}`) did not take its module settings within its "
        "startup time limit (configuration_timeout)."
    ),
    "invalid_configuration": (
        "the module settings for its emitter (`{command}`) are invalid or too "
        "large (invalid_configuration)."
    ),
    "startup_failure": (
        "its emitter (`{command}`) could not be started (startup_failure)."
    ),
}
#: An emitter killed by a signal before it was ready prints nothing of its
#: own (``child_exit`` with a negative exit code).
_KILLED_PROBLEM = (
    "`{command}` was killed by signal {signal} before it was ready (child_exit, "
    "exit code {exit_code}), so it printed nothing of its own."
)
#: The reason a required build stops with, for an interface reason that
#: means something else once ``version`` has answered.
_REASONS_AFTER_VERSION_CHECK: Mapping[StartupFailureReason, AdapterFailureReason] = {
    "unsupported_protocol_version": "hello_failed"
}
#: How ``microcosm-emitter version`` gave no usable answer: it did not answer
#: in time, did not run, exited with an error, or answered with something no
#: emitter release prints.
type VersionFailureKind = Literal["timeout", "not_run", "exited", "not_an_answer"]
_NOT_STARTED = "so the emitter was not started"
#: For a ``version`` that ran and answered with something no emitter
#: release prints: the command may be another program.
_NOT_AN_EMITTER_ANSWER = (
    "every microcosm-emitter release from 0.4.0 answers with a JSON object "
    "naming its emitter_version and protocol_versions, so `{command}` may not "
    "be microcosm-emitter. Point MICROCOSM_EMITTER_COMMAND at microcosm-emitter, "
    f"or install it with `{INSTALL_EMITTER_COMMAND}`."
)


class TelemetryRequiredError(RuntimeError):
    """A build that must send telemetry has no emitter it can use.

    ``LocalTelemetryEmitter.start`` raises it only with
    ``MICROCOSM_TELEMETRY_REQUIRED=1``. ``reason`` is the interface's startup
    reason (``StartupFailureReason``) or one the adapter finds itself
    (``AdapterFailureReason``). ``exit_code`` is the emitter's exit code when
    it exited before it was ready. With ``interface_warned`` the message
    says that the interface's line above, which says the build will
    continue, does not apply. The message holds no setting's or credential's
    value.
    """

    def __init__(
        self,
        reason: TelemetryRequiredReason,
        problem: str,
        *,
        exit_code: int | None = None,
        interface_warned: bool = False,
    ) -> None:
        self.reason: TelemetryRequiredReason = reason
        self.exit_code = exit_code
        super().__init__(
            REQUIRED_TELEMETRY_ERROR.format(
                variable=TELEMETRY_REQUIRED_VARIABLE,
                problem=problem,
                interface_line=INTERFACE_LINE_SUPERSEDED if interface_warned else "",
            )
        )


def _startup_refusal(
    error: EmitterStartupError | None, *, command: str, interface_warned: bool
) -> TelemetryRequiredError:
    """The error of a build that must send telemetry, for a startup failure.

    ``command`` is the emitter command the interface started.
    """
    interface_reason: StartupFailureReason = (
        "startup_failure" if error is None else error.reason
    )
    exit_code = None if error is None else error.exit_code
    reason: TelemetryRequiredReason = interface_reason
    if interface_warned:
        reason = _REASONS_AFTER_VERSION_CHECK.get(interface_reason, interface_reason)
    if interface_reason == "child_exit" and exit_code is not None and exit_code < 0:
        problem = _KILLED_PROBLEM.format(
            command=command, signal=-exit_code, exit_code=exit_code
        )
    else:
        template = (
            _STARTUP_PROBLEMS.get(interface_reason)
            or _STARTUP_PROBLEMS["startup_failure"]
        )
        exit_detail = "" if exit_code is None else f", exit code {exit_code}"
        problem = template.format(command=command, exit_detail=exit_detail)
    return TelemetryRequiredError(
        reason, problem, exit_code=exit_code, interface_warned=interface_warned
    )


def _missing_command_refusal(error: EmitterStartupError) -> TelemetryRequiredError:
    """A required build's refusal when the adapter finds no emitter command.

    It names what was looked for: ``MICROCOSM_EMITTER_COMMAND``'s value, or
    ``microcosm-emitter`` on ``PATH``.
    """
    named = os.environ.get(COMMAND_ENVIRONMENT_VARIABLE, "").strip()
    looked_for = (
        f"`{named}`, named by {COMMAND_ENVIRONMENT_VARIABLE}"
        if named
        else f"`{DEFAULT_COMMAND}`, on PATH"
    )
    return TelemetryRequiredError(
        error.reason,
        f"the emitter command ({looked_for}) was not found. Install it with "
        f"`{INSTALL_EMITTER_COMMAND}`, or point {COMMAND_ENVIRONMENT_VARIABLE} at "
        "it.",
    )


def _telemetry_required() -> bool:
    """Whether this build must send telemetry (``MICROCOSM_TELEMETRY_REQUIRED``).

    ``1``: it must. Unset or ``0``: it continues without an emitter, which
    is only for people outside PolicyEngine without access to the collector;
    PolicyEngine's own builds keep telemetry required. Any other value raises
    ``ValueError``, so a mistyped setting never turns the requirement off.
    """
    setting = os.environ.get(TELEMETRY_REQUIRED_VARIABLE)
    if setting is None or setting == "0":
        return False
    if setting == "1":
        return True
    raise ValueError(
        f"{TELEMETRY_REQUIRED_VARIABLE} must be 1 or 0, or unset; it is {setting!r}."
    )


def _release_number(version: object) -> tuple[int, ...] | None:
    """``(major, minor, patch)`` of a release version, or None if it has none."""
    found = _RELEASE_NUMBER.match(version) if isinstance(version, str) else None
    return None if found is None else tuple(int(part) for part in found.groups())


def _version_description(
    command: str, timeout: float
) -> tuple[dict[str, Any] | None, tuple[VersionFailureKind, str] | None]:
    """``command version``'s answer, or None and how ``version`` failed.

    ``microcosm-emitter version`` (every release since 0.4.0, the first with
    the command) prints the same description as the emitter's ``hello``
    answer, and starts none of its modules. A failure is its kind and a
    phrase for an error message.
    """
    try:
        answer = subprocess.run(
            [command, "version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, ("timeout", f"did not answer within {timeout:g} s")
    except Exception as error:
        return None, ("not_run", f"could not be run ({type(error).__name__})")
    if answer.returncode != 0:
        return None, ("exited", f"exited {answer.returncode}")
    try:
        description = json.loads(answer.stdout)
    except ValueError:
        description = None
    if not isinstance(description, dict):
        return None, ("not_an_answer", "printed no JSON object")
    return description, None


def _version_refusal(
    command: str,
    version: str | None,
    failure: tuple[VersionFailureKind, str] | None,
) -> TelemetryRequiredError:
    """Why a required build does not start the emitter its ``version`` describes.

    Only a ``version`` that did not run is advice to check that the command
    runs, and only an emitter older than the minimum is advice to upgrade.
    """
    if failure is None and _release_number(version) is not None:
        return TelemetryRequiredError(
            "outdated_emitter",
            f"the installed microcosm-emitter (`{command}`) reports version "
            f"{version}, and this build needs {MINIMUM_EMITTER_VERSION} or later. "
            f"Upgrade it with `{UPGRADE_EMITTER_COMMAND}`, or point "
            f"{COMMAND_ENVIRONMENT_VARIABLE} at a newer one.",
        )
    if failure is None:
        failure = (
            "not_an_answer",
            "printed no emitter_version"
            if version is None
            else f"reported version {version!r}, which is not a release number",
        )
    kind, detail = failure
    if kind == "timeout":
        return TelemetryRequiredError(
            "version_timeout", f"`{command} version` {detail}, {_NOT_STARTED}."
        )
    if kind == "not_run":
        advice = f"Check that `{command}` runs on this machine."
    elif kind == "exited":
        advice = (
            "Releases before 0.4.0 have no `version` command; for a later one, "
            f"`{command} version` prints its own error."
        )
    else:
        detail = f"ran but {detail}"
        advice = _NOT_AN_EMITTER_ANSWER.format(command=command)
    return TelemetryRequiredError(
        "version_failed", f"`{command} version` {detail}, {_NOT_STARTED}. {advice}"
    )


def _protocol_refusal(
    command: str, description: Mapping[str, Any]
) -> TelemetryRequiredError | None:
    """A required build's refusal of an emitter that lists no protocol version 1."""
    versions = description.get("protocol_versions")
    if not isinstance(versions, list) or not all(
        type(version) is int for version in versions
    ):
        return TelemetryRequiredError(
            "version_failed",
            f"`{command} version` ran but printed no list of protocol versions, "
            f"{_NOT_STARTED}. " + _NOT_AN_EMITTER_ANSWER.format(command=command),
        )
    if PROTOCOL_VERSION not in versions:
        return TelemetryRequiredError(
            "unsupported_protocol_version",
            f"`{command} version` lists protocol versions {versions}, not protocol "
            f"version {PROTOCOL_VERSION}, the one this build sends, {_NOT_STARTED}.",
        )
    return None


def _start_emitter(*, required: bool = False, **options: Any) -> EmitterClient:
    """Start the installed emitter, unless it is older than the minimum.

    The emitter's telemetry module records the run's start as soon as the
    emitter starts, before ``hello`` can name its release, so the release is
    read from ``microcosm-emitter version`` first. An emitter stopped after
    starting would leave the collector a run that started and never
    finished. An outdated emitter is never started: the client returned is
    unavailable, so every call reports that. A missing command is left to
    ``EmitterClient.start``, which reports it.

    With ``required`` (a build that must send telemetry, which asks for the
    telemetry module), each of these failures, every other one
    ``EmitterClient.start`` reports, and an emitter that started without the
    telemetry module raise ``TelemetryRequiredError`` instead. The adapter
    finds a missing command, a ``version`` that times out or fails, an
    outdated emitter and one that lists no protocol version 1 before
    anything starts, and prints nothing then. For a failure
    ``EmitterClient.start`` reports, the interface has already printed that
    the build will continue, and the error says that does not apply. Each
    message names the command the adapter resolved.
    """
    command: str | None = None
    missing: EmitterStartupError | None = None
    try:
        command = resolve_command()
    except EmitterStartupError as error:
        missing = error
    except Exception:
        missing = EmitterStartupError("startup_failure")
    if command is None:
        if required:
            raise _missing_command_refusal(
                missing or EmitterStartupError("command_not_found")
            )
        return EmitterClient.start(**options)
    timeout = options.get("startup_timeout", STARTUP_TIMEOUT_SECONDS)
    description, failure = _version_description(command, timeout)
    reported_version = (
        None if description is None else description.get("emitter_version")
    )
    version = reported_version if isinstance(reported_version, str) else None
    installed = _release_number(version)
    if installed is not None and installed >= _MINIMUM_RELEASE:
        if required and description is not None:
            refusal = _protocol_refusal(command, description)
            if refusal is not None:
                raise refusal
        client = EmitterClient.start(command=command, **options)
        if required and not client.available:
            raise _startup_refusal(
                client.startup_error, command=command, interface_warned=True
            )
        if required and TELEMETRY_MODULE not in client.modules:
            # It started without telemetry, so it recorded no run; stop it.
            client.close()
            raise TelemetryRequiredError(
                "telemetry_module_not_loaded",
                f"its emitter (`{command}`) started without its telemetry module.",
            )
        return client
    if required:
        raise _version_refusal(command, version, failure)
    reported = "reports no version" if version is None else f"reports version {version}"
    print(
        OUTDATED_EMITTER_WARNING.format(
            reported=reported,
            required=MINIMUM_EMITTER_VERSION,
            upgrade=UPGRADE_EMITTER_COMMAND,
        ),
        file=sys.stderr,
    )
    return OutdatedEmitterClient(reported)


class OutdatedEmitterClient(EmitterClient):
    """The client for an installed emitter too old to start: never available.

    ``reported`` is what its ``version`` command answered (for example
    ``reports version 0.5.1``), so a refusal can name it.
    """

    def __init__(self, reported: str) -> None:
        super().__init__()
        self.reported = reported


class BundleEmitter(Protocol):
    """The interface ``EmitterClient``'s staged-bundle calls and state.

    ``startup_error`` says why an emitter did not start, and ``last_failure``
    why its latest call reported failure.
    """

    startup_error: EmitterStartupError | None
    last_failure: RequestFailure | None

    @property
    def available(self) -> bool: ...

    @property
    def modules(self) -> frozenset[str]: ...

    def stage_dataset_bundle(
        self, request: StageDatasetBundle, *, wait_seconds: float = 0
    ) -> DatasetBundleReceipt: ...

    def dataset_bundle_receipt(
        self, handover_id: str
    ) -> DatasetBundleReceipt | None: ...

    def check_write_access(self, repository: str) -> WriteAccess: ...

    def close(self) -> None: ...


def emitter_problem(client: BundleEmitter | None) -> str | None:
    """Why ``client`` cannot take a Hugging Face hand-over, and what fixes it.

    ``None`` when it is running with its Hugging Face module.
    """

    if isinstance(client, OutdatedEmitterClient):
        return (
            f"the installed microcosm-emitter {client.reported}, and builds need "
            f"{MINIMUM_EMITTER_VERSION} or later, so the build did not start it; "
            f"upgrade it with `{UPGRADE_EMITTER_COMMAND}`, or point "
            "MICROCOSM_EMITTER_COMMAND at a newer one"
        )
    if client is None:
        return "the build could not prepare its emitter (see the warning above)"
    if not client.available:
        failure = client.startup_error
        reason = None if failure is None else failure.reason
        if reason == "command_not_found":
            return (
                "no microcosm-emitter command was found (MICROCOSM_EMITTER_COMMAND, "
                f"then PATH); install it with `{INSTALL_EMITTER_COMMAND}`"
            )
        if reason is not None:
            return (
                f"the emitter did not start ({reason}); check that "
                "`microcosm-emitter version` runs, then try again"
            )
        return "the build's emitter is not running"
    if HUGGING_FACE_MODULE not in client.modules:
        return (
            "the running emitter has no Hugging Face module; upgrade it with "
            f"`{UPGRADE_EMITTER_COMMAND}`"
        )
    return None


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
        """Start the installed emitter; a failure leaves an unavailable adapter.

        With ``MICROCOSM_TELEMETRY_REQUIRED=1`` a failure raises
        ``TelemetryRequiredError`` instead, before the build does any work:
        the emitter could not be prepared or was not found, its ``version``
        timed out or failed, it is older than ``MINIMUM_EMITTER_VERSION``, it
        lists no protocol version 1, its ``hello`` failed, it exited or timed
        out before it was ready, or it started without its telemetry module.
        The version check and the start each wait at most
        ``startup_timeout_seconds``. An emitter that cannot use the
        collector-token variables it inherits exits before it is ready, so it
        is refused too. Whether the emitter later authenticates to the
        collector is not visible here: the interface has no call that reports
        it. Any value other than ``1``, ``0`` or unset raises ``ValueError``.
        """
        required = _telemetry_required()
        run = TelemetryRun(
            run_id=run_id,
            country_code=country_code,
            pipeline=pipeline,
            candidate_id=candidate_id,
            release_id=release_id,
            run_kind=run_kind,
        )
        setup_error: str | None = None
        try:
            path = Path(spool_path) if spool_path else default_spool_path()
            identity = runtime_identity()
        except Exception as error:
            # Only the type: the message can name private paths.
            setup_error = type(error).__name__
        if setup_error is not None:
            if required:
                raise TelemetryRequiredError(
                    "setup_failure",
                    f"its emitter could not be prepared ({setup_error}).",
                )
            print(SETUP_WARNING.format(error_type=setup_error), file=sys.stderr)
            return cls(
                run=run, client=None, close_service_on_finish=close_service_on_finish
            )
        client = _start_emitter(
            required=required,
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


def start_bundle_emitter() -> EmitterClient:
    """A short-lived emitter with only the Hugging Face module.

    A command that stages a finished run's bundle, outside any build, uses
    it; the caller closes it. Its hand-over queue is the one builds on this
    machine share (``events.handovers.sqlite3``), so it also resumes their
    queued hand-overs.
    """
    return _start_emitter(
        hugging_face=HuggingFaceSettings(), startup_timeout=STARTUP_TIMEOUT_SECONDS
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
