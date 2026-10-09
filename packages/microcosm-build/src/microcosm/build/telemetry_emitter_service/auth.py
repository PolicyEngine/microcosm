"""HF-to-collector session shared by telemetry and graph publication."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from http import HTTPStatus
from typing import Any, Literal, NamedTuple
from urllib.parse import urlsplit

from huggingface_hub import get_token

from microcosm.build.telemetry_emitter_service.constants import (
    COLLECTOR_RESPONSE_TOO_LARGE_ERROR,
    DEFAULT_TOKEN_LIFETIME_SECONDS,
    DEVELOPMENT_COLLECTOR_LOOPBACK_ERROR,
    HTTP_TIMEOUT_SECONDS,
    HTTP_USER_AGENT,
    LOOPBACK_HOSTS,
    MAX_HTTP_RESPONSE_BYTES,
    MINIMUM_TOKEN_LIFETIME_SECONDS,
    SERVICE_ORIGIN_FORMAT_ERROR,
    SERVICE_ORIGIN_HTTPS_ERROR,
    TOKEN_EXCHANGE_PATH,
    TOKEN_REFRESH_MARGIN_SECONDS,
)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep bearer credentials on the explicitly configured origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


type CredentialStatus = Literal["ok", "missing", "rejected", "unavailable"]


class CollectorCredential(NamedTuple):
    """A session token, or the reason none is available for this attempt."""

    status: CredentialStatus
    token: str | None


def validated_origin(value: str, *, allow_loopback_http: bool = False) -> str:
    """Return an HTTPS origin, or plain HTTP only for an allowed loopback host."""
    parsed = urlsplit(value)
    loopback = parsed.hostname in LOOPBACK_HOSTS
    valid_scheme = parsed.scheme == "https" or (
        allow_loopback_http and loopback and parsed.scheme == "http"
    )
    if not parsed.hostname or not valid_scheme:
        raise ValueError(SERVICE_ORIGIN_HTTPS_ERROR)
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(SERVICE_ORIGIN_FORMAT_ERROR)
    return value.rstrip("/")


def _development_collector_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.hostname not in LOOPBACK_HOSTS:
        raise ValueError(DEVELOPMENT_COLLECTOR_LOOPBACK_ERROR)
    return validated_origin(value, allow_loopback_http=True)


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


type HttpPost = Callable[[str, Mapping[str, Any], str], tuple[int, dict[str, Any]]]


def _huggingface_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
        or get_token()
    )


class CollectorSession:
    """One thread-safe short-lived credential shared by independent publishers."""

    def __init__(
        self,
        origin: str,
        *,
        post: HttpPost | None = None,
        credential: Callable[[], str | None] | None = None,
    ) -> None:
        self.origin = origin
        self._post = post or _http_post
        self._credential = credential or _huggingface_token
        self._session_token: tuple[str, float] | None = None
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        with self._lock:
            self._session_token = None

    def credential(self) -> CollectorCredential:
        with self._lock:
            if (
                self._session_token is not None
                and self._session_token[1]
                > time.monotonic() + TOKEN_REFRESH_MARGIN_SECONDS
            ):
                return CollectorCredential("ok", self._session_token[0])
            hf_token = self._credential()
            if not hf_token:
                return CollectorCredential("missing", None)
            try:
                status, response = self._post(
                    self.origin + TOKEN_EXCHANGE_PATH, {}, hf_token
                )
            except (OSError, TimeoutError):
                return CollectorCredential("unavailable", None)
            if status == HTTPStatus.OK and isinstance(
                response.get("access_token"), str
            ):
                try:
                    expires_in = max(
                        MINIMUM_TOKEN_LIFETIME_SECONDS,
                        int(response.get("expires_in", DEFAULT_TOKEN_LIFETIME_SECONDS)),
                    )
                except (ValueError, TypeError):
                    return CollectorCredential("unavailable", None)
                token = response["access_token"]
                self._session_token = (token, time.monotonic() + expires_in)
                return CollectorCredential("ok", token)
            if status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
                return CollectorCredential("rejected", None)
            return CollectorCredential("unavailable", None)
