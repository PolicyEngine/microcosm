"""One build program's telemetry run, through the build adapter.

A build program (the registered US Modal programs, for one) opens
``program_telemetry`` once, right after it parses its arguments, so that the
emitter starts before the program does any work. Inside it the program
records a stage change before each of its stages. Leaving it records the
outcome: completion when the body returns or exits with status 0, and
failure, with the error, when it raises anything else. A SIGTERM inside the
body (the Modal runner's budget, a supervisor, ``kill``) is turned into
``BuildTerminatedError`` (``microcosm.build.termination``), recorded as a
termination in the stage that was running, and the program then exits 143,
as the UK builds do. Starting the emitter
follows ``microcosm.build.telemetry_emitter``: with
``MICROCOSM_TELEMETRY_REQUIRED=1`` an emitter that cannot be used stops the
program before its first stage; unset or ``0``, the program runs without it.
Leaving telemetry not required exists only for people outside PolicyEngine
who run Microcosm without access to the collector; PolicyEngine's own builds
keep it required.

The telemetry run's ID comes from ``MICROCOSM_TELEMETRY_RUN_ID`` when that is
set. The Modal stage runner sets it to ``<plan run_id>.<stage>``, so every
attempt of a plan's stage reports under one run, which the collector shows
as a restart of that run, and the part before the last ``.`` (the plan's
``run_id``) becomes the run's ``candidate_id``. Unset, as on a workstation,
each attempt makes its own ID, ``<pipeline>-<UTC time>-<8 hex digits>`` (the
pipeline with ``-`` for ``_``), as the US fiscal refresh does, and has no
candidate ID.

A program that runs its stages in child interpreters of itself records their
telemetry in the parent: it starts its children with
``child_stage_environment``, and a child for which
``telemetry_recorded_by_parent`` holds opens no telemetry run of its own.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime

from microcosm.build.run_outcome import classify_failure
from microcosm.build.telemetry_emitter import LocalTelemetryEmitter
from microcosm.build.termination import BuildTerminatedError, raise_on_sigterm

__all__ = [
    "TELEMETRY_PARENT_PID_VARIABLE",
    "TELEMETRY_RUN_ID_VARIABLE",
    "child_stage_environment",
    "program_telemetry",
    "telemetry_recorded_by_parent",
    "telemetry_run_identity",
]

#: The telemetry run a build program reports under, when whoever starts the
#: program chooses it. The Modal stage runner sets ``<plan run_id>.<stage>``.
TELEMETRY_RUN_ID_VARIABLE = "MICROCOSM_TELEMETRY_RUN_ID"
#: Set by a program for the child interpreters it runs its stages in: the
#: parent's process ID. A child whose parent has that ID starts no emitter,
#: because the parent records the child's stage and outcome. A value left in
#: a shell names no parent of a program started by hand, so it does nothing.
TELEMETRY_PARENT_PID_VARIABLE = "MICROCOSM_TELEMETRY_PARENT_PID"
#: The collector's rule for a run ID (``SAFE_IDENTIFIER`` in
#: microcosm-collector's models). A given ID that breaks it is refused here,
#: before any work, rather than by the collector after the build started.
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")


def telemetry_run_identity(
    pipeline: str,
    *,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> tuple[str, str | None]:
    """This attempt's telemetry run ID and candidate ID.

    The run ID is ``MICROCOSM_TELEMETRY_RUN_ID`` when set, and its part
    before the last ``.``, if any, is the candidate ID. Unset or empty, the
    run ID is ``<pipeline>-<UTC time>-<8 hex digits>`` and there is no
    candidate ID. A set value the collector would refuse raises
    ``ValueError``.
    """

    environment = os.environ if environ is None else environ
    given = environment.get(TELEMETRY_RUN_ID_VARIABLE, "")
    if given:
        if not _RUN_ID.fullmatch(given):
            raise ValueError(
                f"{TELEMETRY_RUN_ID_VARIABLE} must be 1 to 192 letters, digits, "
                "'.', '_', ':' or '-', starting with a letter or digit"
            )
        candidate, separator, _stage = given.rpartition(".")
        return given, candidate if separator and candidate else None
    instant = (now or datetime.now(UTC)).astimezone(UTC)
    prefix = pipeline.replace("_", "-")
    return f"{prefix}-{instant:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}", None


def child_stage_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """``environment`` for a child stage whose telemetry this process records."""

    return {**environment, TELEMETRY_PARENT_PID_VARIABLE: str(os.getpid())}


def telemetry_recorded_by_parent(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this process is a child stage whose parent records its telemetry."""

    environment = os.environ if environ is None else environ
    return environment.get(TELEMETRY_PARENT_PID_VARIABLE) == str(os.getppid())


def _exited_cleanly(error: BaseException) -> bool:
    return isinstance(error, SystemExit) and error.code in (None, 0)


@contextmanager
def program_telemetry(
    *,
    pipeline: str,
    country_code: str = "US",
    run_kind: str = "build",
    environ: Mapping[str, str] | None = None,
) -> Iterator[LocalTelemetryEmitter]:
    """Start this program's emitter, and record its outcome on the way out.

    Yields the adapter, unavailable when the emitter could not be used and
    telemetry is not required; every call on it is then a no-op.
    ``TelemetryRequiredError`` from the adapter, and ``ValueError`` for a
    run ID the collector would refuse, propagate before the body runs. A
    SIGTERM in the body ends it with ``SystemExit(143)`` after the
    termination is recorded.
    """

    run_id, candidate_id = telemetry_run_identity(pipeline, environ=environ)
    emitter = LocalTelemetryEmitter.start(
        run_id=run_id,
        country_code=country_code,
        pipeline=pipeline,
        candidate_id=candidate_id,
        run_kind=run_kind,
    )
    try:
        with raise_on_sigterm():
            yield emitter
    except BuildTerminatedError as terminated:
        classified = classify_failure(terminated)
        emitter.fail(
            terminated,
            failure_class=classified.failure_class or "terminated",
            error_code=classified.error_code,
        )
        raise SystemExit(terminated.exit_code) from terminated
    except BaseException as error:
        if _exited_cleanly(error):
            emitter.complete()
        else:
            emitter.fail(error)
        raise
    emitter.complete()
