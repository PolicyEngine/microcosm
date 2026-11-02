"""A loopback HTTP server that answers the emitter's Hugging Face Hub requests.

The installed-emitter tests (``tests/installed_emitter/``) start the separately
installed ``microcosm-emitter`` with ``HF_ENDPOINT`` set to this server. The
emitter's Hugging Face module builds ``HfApi(token=...)`` without an endpoint,
so ``huggingface_hub`` sends every request here. The server answers the HTTP
requests that ``huggingface_hub`` 1.33.0 (the release
``tools/installed-emitter/constraints.txt`` installs) sends for the seven
``HfApi`` calls the emitter makes, for dataset repositories:

- ``whoami``: ``GET /api/whoami-v2``;
- ``repo_info``: ``GET /api/datasets/<repository>/revision/<revision>``;
- ``list_repo_files``: ``GET /api/datasets/<repository>/tree/<revision>``;
- ``get_paths_info``: ``POST /api/datasets/<repository>/paths-info/<revision>``,
  a form with ``paths`` and ``expand``;
- ``file_exists``: ``HEAD /datasets/<repository>/resolve/<revision>/<path>``;
- ``hf_hub_download``: the same ``HEAD``, then ``GET`` on that address;
- ``create_commit``: ``POST /api/datasets/<repository>/preupload/<revision>``,
  then ``POST /api/datasets/<repository>/commit/<revision>`` with one JSON
  object per line.

Repositories follow the emitter's own in-memory Hub
(``tests/support/fake_hugging_face.py`` in microcosm-emitter): each starts with
one commit; a commit whose ``parentCommit`` is no longer the head gets HTTP
412; a missing repository, revision or file gets HTTP 404 with the error code
``huggingface_hub`` turns into ``RepositoryNotFoundError``,
``RevisionNotFoundError`` or ``RemoteEntryNotFoundError``; and every commit
keeps the exact bytes it added. The preupload answer gives every file the
``regular`` upload mode, so file contents travel inside the commit request
and nothing is uploaded through LFS or Xet; a commit that names an LFS file
is refused with HTTP 400.

A request without the configured token gets HTTP 401. A test can refuse
writes, as the Hub does for a read-only token (``refuse_writes``), and hold
one kind of request without answering until it calls ``release``
(``stall``). The server also refuses every proxy request with HTTP 403 and
records its target (``proxied_requests``): a test that sets the proxy
variables to this server's address sees any request meant for another host.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

__all__ = ["REQUEST_KINDS", "StandInCommit", "StandInHub"]

#: The kinds of request the server answers, as ``requests`` records them.
REQUEST_KINDS = (
    "whoami",
    "repo_info",
    "tree",
    "paths_info",
    "resolve",
    "preupload",
    "commit",
)
#: How long a stalled request waits for ``release`` before it is answered
#: anyway, so a test that never releases it cannot hold the server forever.
STALL_LIMIT_SECONDS = 120.0
_REPOSITORY = r"(?P<repository>[^/]+/[^/]+)"
_REVISION = r"(?P<revision>[^/]+)"
_ROUTES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("GET", "whoami", re.compile(r"^/api/whoami-v2$")),
    (
        "GET",
        "repo_info",
        re.compile(rf"^/api/datasets/{_REPOSITORY}/revision/{_REVISION}$"),
    ),
    (
        "GET",
        "tree",
        re.compile(
            rf"^/api/datasets/{_REPOSITORY}/tree/{_REVISION}(?:/(?P<path>.+))?$"
        ),
    ),
    (
        "POST",
        "paths_info",
        re.compile(rf"^/api/datasets/{_REPOSITORY}/paths-info/{_REVISION}$"),
    ),
    (
        "POST",
        "preupload",
        re.compile(rf"^/api/datasets/{_REPOSITORY}/preupload/{_REVISION}$"),
    ),
    (
        "POST",
        "commit",
        re.compile(rf"^/api/datasets/{_REPOSITORY}/commit/{_REVISION}$"),
    ),
    (
        "HEAD",
        "resolve",
        re.compile(rf"^/datasets/{_REPOSITORY}/resolve/{_REVISION}/(?P<path>.+)$"),
    ),
    (
        "GET",
        "resolve",
        re.compile(rf"^/datasets/{_REPOSITORY}/resolve/{_REVISION}/(?P<path>.+)$"),
    ),
)
_TRUE_VALUES = frozenset({"1", "true"})


def git_blob_id(content: bytes) -> str:
    """The Git object ID the Hub reports for a file (its ``oid`` and ETag)."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def _hub_date(moment: datetime) -> str:
    """A date in the Hub's format, which ``huggingface_hub`` parses."""
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


@dataclass(frozen=True)
class StandInCommit:
    """One commit: the repository's files after it, and what it wrote."""

    oid: str
    parent: str | None
    title: str
    files: Mapping[str, bytes]
    added: Mapping[str, bytes]
    date: datetime


@dataclass
class _Repository:
    commits: list[StandInCommit] = field(default_factory=list)

    @property
    def head(self) -> StandInCommit:
        return self.commits[-1]


class HubError(Exception):
    """An HTTP error answer, with the headers ``huggingface_hub`` reads."""

    def __init__(
        self, status: int, message: str, *, error_code: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_code = error_code
        self.headers: dict[str, str] = {"X-Error-Message": message}
        if error_code is not None:
            self.headers["X-Error-Code"] = error_code


@dataclass(frozen=True)
class _Answer:
    status: int
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)


def _json_answer(payload: Any, status: int = 200) -> _Answer:
    return _Answer(
        status,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )


class StandInHub:
    """In-memory dataset repositories behind a ``huggingface_hub`` HTTP surface.

    The server listens on ``127.0.0.1`` at a free port from construction until
    ``close``; ``url`` is the value for ``HF_ENDPOINT``.
    """

    def __init__(self, *, token: str) -> None:
        self.token = token
        self._lock = threading.Lock()
        self._repositories: dict[str, _Repository] = {}
        self._sequence = 0
        self._writes_refused = False
        self._stalled_kinds: set[str] = set()
        self._stalled_count: dict[str, int] = {}
        self._release = threading.Event()
        self._stall_changed = threading.Condition(self._lock)
        self._requests: list[tuple[str, str]] = []
        self._proxied: list[str] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="stand-in-hub",
            daemon=True,
        )
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        """Answer any stalled request, then stop the server."""
        self.release()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    # Test controls ---------------------------------------------------------

    def create_repository(self, repository: str) -> str:
        """Create a dataset repository with one empty commit; return its ID."""
        with self._lock:
            self._repositories[repository] = _Repository()
            return self._append_commit(
                repository, files={}, added={}, title="Initial commit", parent=None
            ).oid

    def head(self, repository: str) -> str:
        with self._lock:
            return self._repositories[repository].head.oid

    def commits(self, repository: str) -> list[StandInCommit]:
        """Every commit after the repository's first, oldest first."""
        with self._lock:
            return list(self._repositories[repository].commits[1:])

    def refuse_writes(self) -> None:
        """Answer as the Hub does for a read-only token.

        ``whoami`` reports the token's role as ``read``, and the preupload and
        commit requests get HTTP 403. Reads still succeed.
        """
        with self._lock:
            self._writes_refused = True

    def stall(self, kind: str) -> None:
        """Hold every request of ``kind`` without answering until ``release``."""
        if kind not in REQUEST_KINDS:
            raise ValueError(f"unknown request kind: {kind}")
        with self._lock:
            self._release.clear()
            self._stalled_kinds.add(kind)

    def wait_for_stalled(self, kind: str, *, timeout: float) -> bool:
        """Whether a request of ``kind`` is being held, waiting up to ``timeout``."""
        with self._stall_changed:
            return self._stall_changed.wait_for(
                lambda: self._stalled_count.get(kind, 0) > 0, timeout=timeout
            )

    def release(self) -> None:
        """Answer every held request, and hold no more."""
        with self._lock:
            self._stalled_kinds.clear()
            self._release.set()

    @property
    def requests(self) -> list[tuple[str, str]]:
        """Each request answered or held, as ``(method, kind)``, in order."""
        with self._lock:
            return list(self._requests)

    @property
    def proxied_requests(self) -> list[str]:
        """The target of each request sent to this server as a proxy."""
        with self._lock:
            return list(self._proxied)

    # Request handling ------------------------------------------------------

    def _record_proxied(self, target: str) -> None:
        with self._lock:
            self._proxied.append(target)

    def _hold_if_stalled(self, kind: str) -> None:
        with self._lock:
            if kind not in self._stalled_kinds:
                return
            self._stalled_count[kind] = self._stalled_count.get(kind, 0) + 1
            self._stall_changed.notify_all()
        self._release.wait(timeout=STALL_LIMIT_SECONDS)
        with self._lock:
            self._stalled_count[kind] -= 1

    def answer(
        self,
        method: str,
        target: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> _Answer:
        address = urlsplit(target)
        path = address.path
        for route_method, kind, pattern in _ROUTES:
            found = pattern.fullmatch(path)
            if route_method != method or found is None:
                continue
            with self._lock:
                self._requests.append((method, kind))
            self._hold_if_stalled(kind)
            try:
                self._authenticate(headers)
                handler = self._handlers()[kind]
                return handler(
                    method=method,
                    match=found,
                    query=parse_qs(address.query),
                    headers=headers,
                    body=body,
                )
            except HubError as error:
                return _Answer(
                    error.status,
                    json.dumps({"error": error.message}).encode(),
                    {"Content-Type": "application/json", **error.headers},
                )
        return _Answer(
            404,
            json.dumps({"error": f"the stand-in Hub has no {method} {path}"}).encode(),
            {"Content-Type": "application/json"},
        )

    def _handlers(self) -> dict[str, Callable[..., _Answer]]:
        return {
            "whoami": self._whoami,
            "repo_info": self._repo_info,
            "tree": self._tree,
            "paths_info": self._paths_info,
            "resolve": self._resolve,
            "preupload": self._preupload,
            "commit": self._commit,
        }

    def _authenticate(self, headers: Mapping[str, str]) -> None:
        if headers.get("authorization") != f"Bearer {self.token}":
            raise HubError(401, "Invalid credentials in Authorization header")

    def _repository(self, repository: str) -> _Repository:
        found = self._repositories.get(repository)
        if found is None:
            raise HubError(404, "Repository not found", error_code="RepoNotFound")
        return found

    def _commit_at(self, repository: str, revision: str) -> StandInCommit:
        commits = self._repository(repository).commits
        if revision == "main":
            return commits[-1]
        for commit in commits:
            if commit.oid == revision:
                return commit
        raise HubError(404, "Invalid rev id", error_code="RevisionNotFound")

    def _append_commit(
        self,
        repository: str,
        *,
        files: Mapping[str, bytes],
        added: Mapping[str, bytes],
        title: str,
        parent: str | None,
    ) -> StandInCommit:
        self._sequence += 1
        commit = StandInCommit(
            oid=hashlib.sha1(f"{repository}:{self._sequence}".encode()).hexdigest(),
            parent=parent,
            title=title,
            files=dict(files),
            added=dict(added),
            date=datetime.now(UTC),
        )
        self._repositories[repository].commits.append(commit)
        return commit

    def _last_commit(self, repository: str, path: str) -> StandInCommit | None:
        """The latest commit that changed ``path``, file or folder."""
        last = None
        previous: dict[str, bytes] = {}
        for commit in self._repository(repository).commits:
            current = {
                name: content
                for name, content in commit.files.items()
                if name == path or name.startswith(f"{path}/")
            }
            if current and current != previous:
                last = commit
            previous = current
        return last

    def _entry(
        self, repository: str, commit: StandInCommit, path: str, *, expand: bool
    ) -> dict[str, Any] | None:
        """A file or folder as the tree and paths-info endpoints describe it."""
        if path in commit.files:
            content = commit.files[path]
            entry: dict[str, Any] = {
                "type": "file",
                "path": path,
                "oid": git_blob_id(content),
                "size": len(content),
            }
        elif any(name.startswith(f"{path}/") for name in commit.files):
            entry = {
                "type": "directory",
                "path": path,
                "oid": hashlib.sha1(f"tree:{commit.oid}:{path}".encode()).hexdigest(),
            }
        else:
            return None
        if expand:
            last = self._last_commit(repository, path)
            if last is not None:
                entry["lastCommit"] = {
                    "id": last.oid,
                    "title": last.title,
                    "date": _hub_date(last.date),
                }
        return entry

    def _whoami(self, **_: Any) -> _Answer:
        with self._lock:
            role = "read" if self._writes_refused else "write"
        return _json_answer(
            {
                "type": "user",
                "name": "stand-in-user",
                "auth": {
                    "type": "access_token",
                    "accessToken": {"displayName": "stand-in token", "role": role},
                },
            }
        )

    def _repo_info(self, *, match: re.Match[str], **_: Any) -> _Answer:
        repository = match["repository"]
        with self._lock:
            commit = self._commit_at(repository, unquote(match["revision"]))
        return _json_answer(
            {
                "id": repository,
                "sha": commit.oid,
                "private": True,
                "lastModified": _hub_date(commit.date),
            }
        )

    def _tree(
        self, *, match: re.Match[str], query: Mapping[str, list[str]], **_: Any
    ) -> _Answer:
        repository = match["repository"]
        folder = unquote(match["path"] or "").strip("/")
        recursive = query.get("recursive", ["false"])[0].lower() in _TRUE_VALUES
        expand = query.get("expand", ["false"])[0].lower() in _TRUE_VALUES
        with self._lock:
            commit = self._commit_at(repository, unquote(match["revision"]))
            base = f"{folder}/" if folder else ""
            if folder and folder in commit.files:
                raise HubError(404, "Entry not found", error_code="EntryNotFound")
            below = sorted(name for name in commit.files if name.startswith(base))
            if folder and not below:
                raise HubError(404, "Entry not found", error_code="EntryNotFound")
            paths: list[str] = []
            for name in below:
                parts = name[len(base) :].split("/")
                depths = range(1, len(parts) + 1) if recursive else (1,)
                for depth in depths:
                    path = base + "/".join(parts[:depth])
                    if path not in paths:
                        paths.append(path)
            entries = [
                self._entry(repository, commit, path, expand=expand) for path in paths
            ]
        return _json_answer([entry for entry in entries if entry is not None])

    def _paths_info(
        self, *, match: re.Match[str], headers: Mapping[str, str], body: bytes, **_: Any
    ) -> _Answer:
        repository = match["repository"]
        fields = parse_qs(body.decode(), keep_blank_values=True)
        paths = fields.get("paths", [])
        expand = fields.get("expand", ["false"])[0].lower() in _TRUE_VALUES
        with self._lock:
            commit = self._commit_at(repository, unquote(match["revision"]))
            entries = [
                self._entry(repository, commit, path.strip("/"), expand=expand)
                for path in paths
            ]
        return _json_answer([entry for entry in entries if entry is not None])

    def _resolve(self, *, method: str, match: re.Match[str], **_: Any) -> _Answer:
        repository = match["repository"]
        path = unquote(match["path"])
        with self._lock:
            commit = self._commit_at(repository, unquote(match["revision"]))
            content = commit.files.get(path)
        if content is None:
            error = HubError(404, "Entry not found", error_code="EntryNotFound")
            error.headers["X-Repo-Commit"] = commit.oid
            raise error
        return _Answer(
            200,
            content if method == "GET" else b"",
            {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(content)),
                "ETag": f'"{git_blob_id(content)}"',
                "X-Repo-Commit": commit.oid,
            },
        )

    def _refuse_if_read_only(self) -> None:
        with self._lock:
            refused = self._writes_refused
        if refused:
            raise HubError(403, "You don't have the rights to write to this repository")

    def _preupload(self, *, match: re.Match[str], body: bytes, **_: Any) -> _Answer:
        self._refuse_if_read_only()
        repository = match["repository"]
        with self._lock:
            commit = self._commit_at(repository, unquote(match["revision"]))
        files = json.loads(body)["files"]
        return _json_answer(
            {
                "files": [
                    {
                        "path": item["path"],
                        "uploadMode": "regular",
                        "shouldIgnore": False,
                    }
                    for item in files
                ],
                "commitOid": commit.oid,
            }
        )

    def _commit(self, *, match: re.Match[str], body: bytes, **_: Any) -> _Answer:
        self._refuse_if_read_only()
        repository = match["repository"]
        revision = unquote(match["revision"])
        if revision != "main":
            raise HubError(400, "the stand-in Hub commits only to main")
        header: Mapping[str, Any] | None = None
        added: dict[str, bytes] = {}
        for line in body.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key, value = item.get("key"), item.get("value") or {}
            if key == "header":
                header = value
            elif key == "file" and value.get("encoding") == "base64":
                added[value["path"]] = base64.b64decode(value["content"])
            else:
                raise HubError(
                    400, f"the stand-in Hub accepts only regular files, not {key!r}"
                )
        if header is None or not header.get("summary"):
            raise HubError(400, "the commit has no summary")
        with self._lock:
            head = self._repository(repository).head
            parent = header.get("parentCommit")
            if parent is not None and parent != head.oid:
                raise HubError(412, "A commit has happened since. Please refresh.")
            commit = self._append_commit(
                repository,
                files={**head.files, **added},
                added=added,
                title=str(header["summary"]).split("\n", 1)[0],
                parent=head.oid,
            )
        return _json_answer(
            {
                "success": True,
                "commitOid": commit.oid,
                "commitUrl": f"{self.url}/datasets/{repository}/commit/{commit.oid}",
                "hookOutput": "",
                "pullRequestUrl": None,
            }
        )


def _handler_for(hub: StandInHub) -> type[BaseHTTPRequestHandler]:
    class StandInHubRequestHandler(BaseHTTPRequestHandler):
        """Pass each request to ``hub``; refuse proxy requests."""

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def _send(self, answer: _Answer) -> None:
            self.send_response(answer.status)
            headers = {
                key.lower(): (key, value) for key, value in answer.headers.items()
            }
            headers.setdefault(
                "content-length", ("Content-Length", str(len(answer.body)))
            )
            for key, value in headers.values():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD" and answer.body:
                self.wfile.write(answer.body)

        def _refuse_proxy(self) -> None:
            hub._record_proxied(self.path)
            self._send(_Answer(403, b"the stand-in Hub is not a proxy"))

        def _handle(self) -> None:
            if not self.path.startswith("/"):
                self._refuse_proxy()
                return
            body = self._body()
            headers = {key.lower(): value for key, value in self.headers.items()}
            self._send(hub.answer(self.command, self.path, headers, body))

        def do_CONNECT(self) -> None:  # noqa: N802 (http.server's naming)
            self._refuse_proxy()

        do_GET = do_HEAD = do_POST = _handle  # noqa: N815

        def log_message(self, message_format: str, *args: Any) -> None:
            """Keep the test output free of one line per request."""

    return StandInHubRequestHandler
