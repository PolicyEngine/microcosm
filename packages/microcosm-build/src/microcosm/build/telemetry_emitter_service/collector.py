"""Collector authentication and best-effort event delivery."""

from __future__ import annotations

import enum
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit

from huggingface_hub import get_token

from microcosm.build.telemetry_emitter_service.constants import (
    COLLECTOR_RESPONSE_TOO_LARGE_ERROR,
    COLLECTOR_URL_HTTPS_ERROR,
    COLLECTOR_URL_ORIGIN_ERROR,
    DEFAULT_TOKEN_LIFETIME_SECONDS,
    DELIVERY_RUN_WARNING,
    DEVELOPMENT_COLLECTOR_LOOPBACK_ERROR,
    HTTP_TIMEOUT_SECONDS,
    HTTP_USER_AGENT,
    INITIAL_RETRY_SECONDS,
    LOCAL_ONLY_MISSING_CREDENTIAL,
    LOCAL_ONLY_REJECTED_COLLECTOR_AUTHORIZATION,
    LOCAL_ONLY_REJECTED_CREDENTIAL,
    LOCAL_ONLY_REJECTED_REGISTRATION,
    LOOPBACK_HOSTS,
    MAX_HTTP_RESPONSE_BYTES,
    MAX_RETRY_SECONDS,
    MINIMUM_TOKEN_LIFETIME_SECONDS,
    NO_CREDENTIAL_MESSAGE,
    PRODUCTION_COLLECTOR_URL,
    REJECTED_CREDENTIAL_MESSAGE,
    RUN_EVENTS_PATH_TEMPLATE,
    RUN_REGISTRATION_PATH,
    TOKEN_EXCHANGE_PATH,
    TOKEN_EXCHANGE_WARNING,
    TOKEN_REFRESH_MARGIN_SECONDS,
)
from microcosm.build.telemetry_emitter_service.contention import (
    is_transient_spool_error,
)
from microcosm.build.telemetry_emitter_service.diagnostics import (
    describe_error,
    write_warning,
)
from microcosm.build.telemetry_emitter_service.spool import EventSpool, RunKey


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep bearer credentials on the explicitly configured origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _collector_origin(value: str, *, allow_loopback_http: bool = False) -> str:
    parsed = urlsplit(value)
    loopback = parsed.hostname in LOOPBACK_HOSTS
    valid_scheme = parsed.scheme == "https" or (
        allow_loopback_http and loopback and parsed.scheme == "http"
    )
    if not parsed.hostname or not valid_scheme:
        raise ValueError(COLLECTOR_URL_HTTPS_ERROR)
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(COLLECTOR_URL_ORIGIN_ERROR)
    return value.rstrip("/")


def _development_collector_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.hostname not in LOOPBACK_HOSTS:
        raise ValueError(DEVELOPMENT_COLLECTOR_LOOPBACK_ERROR)
    return _collector_origin(value, allow_loopback_http=True)


def _decode_response(body: bytes) -> dict[str, Any]:
    try:
        response = json.loads(body) if body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return response if isinstance(response, dict) else {}


def _http_post(
    url: str,
    payload: Mapping[str, Any],
    bearer_token: str,
    *,
    timeout: float = HTTP_TIMEOUT_SECONDS,
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
            "User-Agent": HTTP_USER_AGENT,
        },
        method="POST",
    )
    try:
        opener = urllib.request.build_opener(_NoRedirectHandler)
        with opener.open(request, timeout=timeout) as response:
            body = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
            if len(body) > MAX_HTTP_RESPONSE_BYTES:
                raise OSError(COLLECTOR_RESPONSE_TOO_LARGE_ERROR)
            return response.status, _decode_response(body)
    except urllib.error.HTTPError as error:
        body = error.read(MAX_HTTP_RESPONSE_BYTES + 1)
        if len(body) > MAX_HTTP_RESPONSE_BYTES:
            return error.code, {}
        return error.code, _decode_response(body)


def _huggingface_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
        or get_token()
    )


@dataclass
class RetryDelay:
    """When one delivery scope may next contact the collector after failing.

    Each failure postpones the next attempt by the current delay, then doubles
    the delay from ``INITIAL_RETRY_SECONDS`` up to ``MAX_RETRY_SECONDS``.
    """

    next_attempt_at: float = 0.0
    seconds: float = INITIAL_RETRY_SECONDS

    def due(self, now: float) -> bool:
        return now >= self.next_attempt_at

    def defer(self, now: float) -> None:
        self.next_attempt_at = now + self.seconds
        self.seconds = min(MAX_RETRY_SECONDS, self.seconds * 2)


class _RunOutcome(enum.Enum):
    DELIVERED = "delivered"
    FAILED = "failed"
    # Nothing was attempted for the run itself: it has just been made
    # local-only, its batch was empty, or it is waiting for a collector token.
    UNCHANGED = "unchanged"


class CollectorDelivery:
    """Authenticate queued runs and deliver idempotent event batches.

    Every service on a host delivers every pending run in the host's shared
    spool, so one run's failure must not hold up the others. Each run waits out
    its own failures in its own ``RetryDelay``. The collector token is shared
    by every run, so a failed token exchange delays only further exchanges.
    """

    def __init__(
        self,
        spool: EventSpool,
        *,
        development_collector_url: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.collector_url = (
            _development_collector_url(development_collector_url)
            if development_collector_url is not None
            else _collector_origin(PRODUCTION_COLLECTOR_URL)
        )
        self.spool = spool
        self._clock = clock
        self._session_token: tuple[str, float] | None = None
        self._registered: set[RunKey] = set()
        self._warned_no_token = False
        self._warned_denied: set[RunKey] = set()
        self._reported_errors: set[tuple[str, str]] = set()
        self._run_delays: dict[RunKey, RetryDelay] = {}
        # Batches the collector accepted whose removal from the spool failed.
        # They are removed before the run sends anything else, so contention
        # never makes the same batch go out again.
        self._accepted: dict[RunKey, list[str]] = {}
        self._exchange_delay = RetryDelay()

    def flush_once(self) -> bool:
        """Attempt one delivery pass without waiting for retry deadlines.

        Every pending run whose own retry delay has passed is attempted. A run
        whose delivery fails, whatever the cause, is retried after its delay
        and does not stop the pass; an unexpected error is reported once per
        error type. Lock contention on the spool is not a failure of any run:
        it ends the pass and propagates, and the worker retries on its next
        tick. So does any error listing the pending runs. A batch the
        collector accepted but contention kept in the spool is removed before
        its run sends anything else, so it is never sent twice.
        """

        run_keys = self.spool.pending_run_keys()
        # A run that left the queue (delivered, local-only or pruned) starts
        # afresh if it returns, and these tables stay as small as the queue.
        self._run_delays = {
            run_key: self._run_delays[run_key]
            for run_key in run_keys
            if run_key in self._run_delays
        }
        self._accepted = {
            run_key: self._accepted[run_key]
            for run_key in run_keys
            if run_key in self._accepted
        }
        made_progress = False
        for run_key in run_keys:
            delay = self._run_delays.get(run_key)
            if delay is not None and not delay.due(self._clock()):
                continue
            try:
                outcome = self._deliver_run(run_key)
            except Exception as error:
                if is_transient_spool_error(error):
                    raise
                self._report(DELIVERY_RUN_WARNING, error)
                outcome = _RunOutcome.FAILED
            if outcome is _RunOutcome.DELIVERED:
                made_progress = True
                self._run_delays.pop(run_key, None)
            elif outcome is _RunOutcome.FAILED:
                self._run_delays.setdefault(run_key, RetryDelay()).defer(self._clock())
        return made_progress

    def _deliver_run(self, run_key: RunKey) -> _RunOutcome:
        run_id, producer_id = run_key
        if run_key in self._accepted:
            self.spool.acknowledge(self._accepted[run_key])
            del self._accepted[run_key]
            return _RunOutcome.DELIVERED
        token = self._collector_token(run_key)
        if token is None:
            return _RunOutcome.UNCHANGED
        if run_key not in self._registered:
            unregistered = self._register(run_key, token)
            if unregistered is not None:
                return unregistered
        events = self.spool.batch(run_id, producer_id)
        if not events:
            return _RunOutcome.UNCHANGED
        try:
            status, _ = _http_post(
                self.collector_url + RUN_EVENTS_PATH_TEMPLATE.format(run_id=run_id),
                {"events": events},
                token,
            )
        except (OSError, TimeoutError):
            return _RunOutcome.FAILED
        if status == HTTPStatus.ACCEPTED:
            self._accepted[run_key] = [event["event_id"] for event in events]
            self.spool.acknowledge(self._accepted[run_key])
            del self._accepted[run_key]
            return _RunOutcome.DELIVERED
        if status == HTTPStatus.FORBIDDEN:
            self._make_local_only(run_key, LOCAL_ONLY_REJECTED_COLLECTOR_AUTHORIZATION)
            return _RunOutcome.UNCHANGED
        if status == HTTPStatus.UNAUTHORIZED:
            self._session_token = None
        return _RunOutcome.FAILED

    def _collector_token(self, run_key: RunKey) -> str | None:
        if (
            self._session_token is not None
            and self._session_token[1] > self._clock() + TOKEN_REFRESH_MARGIN_SECONDS
        ):
            return self._session_token[0]
        hf_token = _huggingface_token()
        if not hf_token:
            if not self._warned_no_token:
                write_warning(NO_CREDENTIAL_MESSAGE)
                self._warned_no_token = True
            self._make_local_only(run_key, LOCAL_ONLY_MISSING_CREDENTIAL)
            return None
        if not self._exchange_delay.due(self._clock()):
            return None
        status = None
        try:
            status, response = _http_post(
                self.collector_url + TOKEN_EXCHANGE_PATH,
                {},
                hf_token,
            )
            if status == HTTPStatus.OK and isinstance(
                response.get("access_token"), str
            ):
                expires_in = max(
                    MINIMUM_TOKEN_LIFETIME_SECONDS,
                    int(response.get("expires_in", DEFAULT_TOKEN_LIFETIME_SECONDS)),
                )
                token = response["access_token"]
                self._session_token = (token, self._clock() + expires_in)
                self._exchange_delay = RetryDelay()
                return token
        except (OSError, TimeoutError):
            status = None
        except Exception as error:
            # The exchange is the same for every run, so a response it cannot
            # use is not this run's failure: retrying it for the next run in
            # the pass would only repeat the request.
            self._report(TOKEN_EXCHANGE_WARNING, error)
            status = None
        if status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            self._make_local_only(run_key, LOCAL_ONLY_REJECTED_CREDENTIAL)
        else:
            self._exchange_delay.defer(self._clock())
        return None

    def _register(self, run_key: RunKey, token: str) -> _RunOutcome | None:
        """Register the run; return ``None`` once registered, else its outcome."""

        # Read here, inside the run's own error handling, so a stored
        # registration that cannot be decoded fails only its own run.
        registration = self.spool.registration(*run_key)
        try:
            status, _ = _http_post(
                self.collector_url + RUN_REGISTRATION_PATH,
                registration,
                token,
            )
        except (OSError, TimeoutError):
            return _RunOutcome.FAILED
        if status == HTTPStatus.CREATED:
            self._registered.add(run_key)
            return None
        if status in {HTTPStatus.FORBIDDEN, HTTPStatus.CONFLICT}:
            self._make_local_only(run_key, LOCAL_ONLY_REJECTED_REGISTRATION)
            return _RunOutcome.UNCHANGED
        if status == HTTPStatus.UNAUTHORIZED:
            self._session_token = None
        return _RunOutcome.FAILED

    def _make_local_only(self, run_key: RunKey, reason: str) -> None:
        self.spool.make_local_only(*run_key, reason)
        if reason != LOCAL_ONLY_MISSING_CREDENTIAL:
            self._warn_denied(run_key)

    def _report(self, warning: str, error: Exception) -> None:
        """Print one line per warning and error type, never a traceback."""

        error_type = type(error).__name__
        if (warning, error_type) in self._reported_errors:
            return
        self._reported_errors.add((warning, error_type))
        write_warning(
            warning.format(error_type=error_type, error=describe_error(error))
        )

    def _warn_denied(self, run_key: RunKey) -> None:
        if run_key in self._warned_denied:
            return
        write_warning(REJECTED_CREDENTIAL_MESSAGE)
        self._warned_denied.add(run_key)
