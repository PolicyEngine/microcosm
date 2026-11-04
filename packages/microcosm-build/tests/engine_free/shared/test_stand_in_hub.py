"""The stand-in Hub's report of requests it could not answer as a Hub would.

The installed-emitter tests fail when ``StandInHub.stray_requests`` names
anything; these requests come from ``http.client`` in the test process, with
no emitter.
"""

from __future__ import annotations

import http.client
from collections.abc import Iterator

import pytest

from test_support.microcosm_build.stand_in_hub import StandInHub

REPOSITORY = "microcosm-tests/staged-bundles"
TOKEN = "stand-in-token-not-a-credential"


@pytest.fixture
def stand_in_hub() -> Iterator[StandInHub]:
    hub = StandInHub(token=TOKEN)
    hub.create_repository(REPOSITORY)
    try:
        yield hub
    finally:
        hub.close()


def _send(
    hub: StandInHub, method: str, target: str, body: bytes = b""
) -> tuple[int, bytes]:
    host, port = hub.url.removeprefix("http://").split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=10)
    try:
        connection.request(
            method, target, body=body, headers={"Authorization": f"Bearer {TOKEN}"}
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_a_handler_failure_is_answered_with_500_and_reported(
    stand_in_hub: StandInHub,
) -> None:
    status, _ = _send(
        stand_in_hub,
        "POST",
        f"/api/datasets/{REPOSITORY}/preupload/main",
        b"not json",
    )

    assert status == 500
    assert stand_in_hub.statuses("preupload") == [500]
    assert len(stand_in_hub.handler_failures) == 1
    failure = stand_in_hub.handler_failures[0]
    assert failure.startswith(f"POST /api/datasets/{REPOSITORY}/preupload/main: ")
    assert "JSONDecodeError: Expecting value" in failure
    assert stand_in_hub.stray_requests() == [f"handler failure: {failure}"]


def test_unanswered_and_proxy_requests_are_reported(stand_in_hub: StandInHub) -> None:
    unanswered, _ = _send(stand_in_hub, "GET", "/api/not-a-route")
    proxied, _ = _send(stand_in_hub, "GET", "http://127.0.0.1:9/elsewhere")

    assert (unanswered, proxied) == (404, 403)
    assert stand_in_hub.stray_requests() == [
        "unanswered: GET /api/not-a-route",
        "proxy request for http://127.0.0.1:9/elsewhere",
    ]


def test_answered_requests_are_not_reported(stand_in_hub: StandInHub) -> None:
    status, _ = _send(stand_in_hub, "GET", "/api/whoami-v2")

    assert status == 200
    assert stand_in_hub.stray_requests() == []
