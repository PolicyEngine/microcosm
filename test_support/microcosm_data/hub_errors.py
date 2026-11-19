"""Download errors as huggingface_hub 1.x raises them, for the fake Hubs.

``hf_hub_download`` answers a file the Hub reports missing (HTTP 404) with
``RemoteEntryNotFoundError``. When it could not ask the Hub, for example
after a 5xx outlasted its retries or on a 403, and nothing is cached for that
revision, it raises ``LocalEntryNotFoundError`` instead, which also inherits
from ``EntryNotFoundError`` and ``FileNotFoundError``. Only the first means
the file is not there.
"""

from __future__ import annotations

import httpx
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
)

#: A host that never resolves; no fake makes a request to it.
_FAKE_HUB = "https://hub.invalid"


class MissingOnHubError(RemoteEntryNotFoundError, FileNotFoundError):
    """The Hub's 404 for a file, as the fake Hubs raise it.

    It is also a ``FileNotFoundError``, as the fakes' earlier error was, so
    tests that expect one keep reading the same failure.
    """


def _request(repo_id: str, filename: str, revision: str | None) -> httpx.Request:
    return httpx.Request(
        "HEAD",
        f"{_FAKE_HUB}/datasets/{repo_id}/resolve/{revision or 'main'}/{filename}",
    )


def missing_on_hub(
    repo_id: str,
    filename: str,
    revision: str | None = None,
    *,
    message: str | None = None,
) -> MissingOnHubError:
    """The error for a file the Hub reports missing at ``revision``."""
    return MissingOnHubError(
        message
        or (
            f"404 Client Error: entry not found for {repo_id}/{filename} at "
            f"revision {revision!r}"
        ),
        response=httpx.Response(404, request=_request(repo_id, filename, revision)),
    )


def hub_unreachable(
    repo_id: str,
    filename: str,
    revision: str | None = None,
    *,
    status_code: int | None = None,
) -> LocalEntryNotFoundError:
    """The error ``hf_hub_download`` raises when it could not ask the Hub.

    With ``status_code`` (a 5xx after the retries, or a 403), it is raised
    from the Hub's HTTP error, as huggingface_hub raises it.
    """
    error = LocalEntryNotFoundError(
        f"could not reach the Hub for {repo_id}/{filename} and it is not in the "
        "local cache"
    )
    if status_code is not None:
        error.__cause__ = HfHubHTTPError(
            f"{status_code} Server Error for {repo_id}/{filename}",
            response=httpx.Response(
                status_code, request=_request(repo_id, filename, revision)
            ),
        )
    return error
