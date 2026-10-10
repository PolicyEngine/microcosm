"""Configuration and protocol values for the telemetry emitter service."""

from __future__ import annotations

from typing import Final

RETENTION_DAYS: Final = 7
MAX_QUEUED_BYTES: Final = 100 * 1024 * 1024
BATCH_SIZE: Final = 100
PRODUCTION_COLLECTOR_URL: Final = (
    "https://microcosm-telemetry-389282473430.us-central1.run.app"
)

LOOPBACK_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})
TOKEN_EXCHANGE_PATH: Final = "/v1/auth/huggingface/exchange"
RUN_REGISTRATION_PATH: Final = "/v1/runs"
RUN_EVENTS_PATH_TEMPLATE: Final = "/v1/runs/{run_id}/events"
HTTP_USER_AGENT: Final = "microcosm-telemetry-emitter/1"
MAX_HTTP_RESPONSE_BYTES: Final = 1_048_576
HTTP_TIMEOUT_SECONDS: Final = 5.0

UPLOAD_STATE_PENDING: Final = "pending"
UPLOAD_STATE_LOCAL_ONLY: Final = "local_only"
LOCAL_ONLY_MISSING_CREDENTIAL: Final = "missing_huggingface_credential"
LOCAL_ONLY_REJECTED_CREDENTIAL: Final = "huggingface_credential_rejected"
LOCAL_ONLY_REJECTED_REGISTRATION: Final = "run_registration_rejected"
LOCAL_ONLY_REJECTED_COLLECTOR_AUTHORIZATION: Final = "collector_authorization_rejected"
LOCAL_ONLY_REJECTED_EVENTS: Final = "collector_rejected_events"
LOCAL_ONLY_PRE_ELIGIBILITY: Final = "created_before_upload_eligibility"

NO_CREDENTIAL_MESSAGE: Final = (
    "Microcosm telemetry is local-only: no ambient Hugging Face credential was "
    "found. The dataset build will continue."
)
REJECTED_CREDENTIAL_MESSAGE: Final = (
    "Microcosm telemetry is local-only for this run: the ambient Hugging Face "
    "credential was not accepted as a PolicyEngine organization member. The "
    "dataset build will continue."
)
REJECTED_EVENTS_MESSAGE: Final = (
    "Microcosm telemetry is local-only for this run: the collector rejected its "
    "events (HTTP {status}), so they will not be retried. The dataset build will "
    "continue."
)
COLLECTOR_URL_HTTPS_ERROR: Final = "collector URL must be an HTTPS origin"
COLLECTOR_URL_ORIGIN_ERROR: Final = (
    "collector URL must be an origin without credentials or path data"
)
DEVELOPMENT_COLLECTOR_LOOPBACK_ERROR: Final = (
    "development collector URL must use a loopback address"
)
COLLECTOR_RESPONSE_TOO_LARGE_ERROR: Final = "collector response exceeds 1 MiB"

DEFAULT_TOKEN_LIFETIME_SECONDS: Final = 3_600
MINIMUM_TOKEN_LIFETIME_SECONDS: Final = 60
TOKEN_REFRESH_MARGIN_SECONDS: Final = 30
INITIAL_RETRY_SECONDS: Final = 1.0
MAX_RETRY_SECONDS: Final = 60.0

DATABASE_TIMEOUT_SECONDS: Final = 5
PRUNE_INTERVAL_SECONDS: Final = 60.0
PRUNE_BATCH_ROWS: Final = 500
# SQLite's busy handler polls rather than queueing, so another process rarely
# gets the lock between two back-to-back transactions. Prune pauses between
# batches and works at most PRUNE_STEP_SECONDS per worker tick.
PRUNE_BATCH_PAUSE_SECONDS: Final = 0.025
PRUNE_STEP_SECONDS: Final = 1.0
# Every concurrent build on a host shares one spool, so a write can find it
# locked for longer than SQLite's own busy wait. Startup retries such errors
# with jittered, doubling waits until shortly before the build stops waiting
# for readiness. While it does, each SQLite statement waits at most
# STARTUP_BUSY_TIMEOUT_SECONDS, so an attempt begun before that deadline ends
# within the margin, which also covers binding the socket and the build's ping.
STARTUP_BUSY_TIMEOUT_SECONDS: Final = 0.25
SPOOL_RETRY_INITIAL_SECONDS: Final = 0.05
SPOOL_RETRY_MAX_SECONDS: Final = 0.25
READY_DEADLINE_MARGIN_SECONDS: Final = 2.0
STARTUP_RETRY_LIMIT_SECONDS: Final = 60.0
SPOOL_LOCKED_EXIT_STATUS: Final = 75  # EX_TEMPFAIL from sysexits.h

# The socket hands each accepted event to a writer thread through an ordered
# in-memory queue, so a spool that another process keeps locked never delays
# the build's sends. The queue holds at most QUEUE_MAX_EVENTS events and
# QUEUE_MAX_BYTES of their encoded JSON. The last QUEUE_RESERVED_EVENTS and
# QUEUE_RESERVED_BYTES of that room take only run events (started, completed,
# failed, blocked), so a queue that progress updates have filled still records
# how the build ended.
QUEUE_MAX_EVENTS: Final = 10_000
QUEUE_MAX_BYTES: Final = 16 * 1024 * 1024
QUEUE_RESERVED_EVENTS: Final = 16
QUEUE_RESERVED_BYTES: Final = 256 * 1024
# The writer appends up to WRITER_BATCH_EVENTS queued events per transaction.
# One append's statements together wait at most WRITER_BUSY_TIMEOUT_SECONDS
# for other processes' locks, and the append waits no longer than that for
# another thread of this process to release the spool. So the writer's retry
# loop decides how long to keep trying, and during a shutdown drain no append
# touches the database after the deadline less that wait.
WRITER_BATCH_EVENTS: Final = 100
WRITER_BUSY_TIMEOUT_SECONDS: Final = 0.25

DEFAULT_HEARTBEAT_SECONDS: Final = 60.0
DEFAULT_DRAIN_SECONDS: Final = 15.0
MINIMUM_HEARTBEAT_SECONDS: Final = 1.0
SOCKET_LISTEN_BACKLOG: Final = 16
SOCKET_ACCEPT_TIMEOUT_SECONDS: Final = 0.5
SOCKET_CONNECTION_TIMEOUT_SECONDS: Final = 0.25
WORKER_INTERVAL_SECONDS: Final = 1.0
DRAIN_RETRY_SECONDS: Final = 0.5

EVENT_OBJECT_ERROR: Final = "event must be an object"
EVENT_FIELDS_ERROR: Final = "event must have an event_type and a status"
QUEUE_FULL_ERROR: Final = "local telemetry queue is full"
QUEUE_CLOSED_ERROR: Final = "local telemetry queue is closed"
SPOOL_BUSY_ERROR: Final = "the spool is busy in this process"
UNSUPPORTED_ACTION_ERROR: Final = "unsupported local telemetry action"
LOCAL_MESSAGE_TOO_LARGE_ERROR: Final = "local telemetry message exceeds 1 MiB"
FAILURE_CLASS_UNEXPECTED_PROCESS_EXIT: Final = "unexpected_process_exit"

READY_DEADLINE_ERROR: Final = "ready deadline must be a number of seconds"
UNKNOWN_SPOOL_REVISION_ERROR: Final = (
    "telemetry spool schema revision {revision!r} is not in this microcosm's "
    "migration history, whose head is {head!r}"
)
SPOOL_LOCKED_WARNING: Final = (
    "warning: the local telemetry emitter service could not register this build "
    "in its spool {spool}: another process kept the spool locked for "
    "{waited_seconds:.1f} s ({error})."
)
SERVICE_ARGUMENTS_WARNING: Final = (
    "warning: the local telemetry emitter service could not start: {error}"
)
SERVICE_FAILED_WARNING: Final = (
    "warning: the local telemetry emitter service stopped: {error_type}: {error}"
)
WORKER_STEP_WARNING: Final = (
    "warning: the local telemetry emitter service's delivery worker hit "
    "{error_type} ({error}) and will keep running."
)
QUEUE_FULL_WARNING: Final = (
    "warning: the local telemetry emitter service's queue is full ({events} "
    "updates, {megabytes:.1f} MiB, waiting for its spool); it will refuse "
    "updates until the spool takes them."
)
UNWRITTEN_EVENTS_WARNING: Final = (
    "warning: the local telemetry emitter service could not write {count} "
    "queued update(s) to its spool before its {seconds:g} s shutdown drain "
    "ended ({reason})."
)
DRAIN_TIME_REASON: Final = "the drain ran out of time"
EVENT_DROPPED_WARNING: Final = (
    "warning: the local telemetry emitter service's spool refused an update, "
    "which was dropped: {error_type} ({error})."
)
WRITER_STOPPED_WARNING: Final = (
    "warning: the local telemetry emitter service's spool writer stopped: "
    "{error_type}: {error}"
)
MAX_WARNING_ERROR_CHARS: Final = 300
