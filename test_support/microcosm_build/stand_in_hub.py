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
- ``create_commit``: ``POST /api/datasets/<repository>/preupload/<revision>``;
  for each file the preupload answer sends through LFS, ``POST
  /datasets/<repository>.git/info/lfs/objects/batch``, a ``PUT`` of its bytes
  to the address the batch answer gives, and ``POST
  /datasets/<repository>.git/info/lfs/objects/verify``; then ``POST
  /api/datasets/<repository>/commit/<revision>`` with one JSON object per
  line of text, where an LFS file is a reference to its stored object.

A real Hub stores files over 10 MB, and files that match its default
``.gitattributes`` (``*.h5`` among them), through LFS or Xet, never inside
the commit request. The tests turn Xet off (``HF_HUB_DISABLE_XET=1``), so
``huggingface_hub`` takes the LFS path with its ``basic`` transfer, which the
server offers as a real Hub does; machines with ``hf_xet`` installed upload
through Xet instead, which the server does not model. Every other file
travels inside the commit request (the ``regular`` upload mode). Downloads are
answered directly, without the redirect a real Hub sends for an LFS file,
and listings do not describe LFS pointers.

Repositories follow the emitter's own in-memory Hub
(``tests/support/fake_hugging_face.py`` in microcosm-emitter): each starts with
one commit; a commit whose ``parentCommit`` is no longer the head gets HTTP
412; a missing repository, revision or file gets HTTP 404 with the error code
``huggingface_hub`` turns into ``RepositoryNotFoundError``,
``RevisionNotFoundError`` or ``RemoteEntryNotFoundError``; and every commit
keeps the exact bytes it added. Each commit request is recorded with the
``parentCommit`` it sent (``commit_attempts``).

A request without the configured token gets HTTP 401. A test can refuse
writes, as the Hub does for a read-only token (``refuse_writes``), let
another writer's commit land just before the next commit request
(``land_before_next_commit``), and hold one kind of request without
answering until it calls ``release`` (``stall``). The server records every
request it answers (``requests``) and every one it has no route for
(``unmatched_requests``). It refuses every proxy request with HTTP 403 and
records its target (``proxied_requests``): a test that sets the proxy
variables to this server's address sees a request meant for another host
from any client that honours those variables.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlsplit

__all__ = [
    "REQUEST_KINDS",
    "CommitAttempt",
    "RecordedRequest",
    "StandInCommit",
    "StandInHub",
]

#: How long a stalled request waits for ``release`` before it is answered
#: anyway, so a test that never releases it cannot hold the server forever.
STALL_LIMIT_SECONDS = 120.0
#: The size above which the Hub stores a file through LFS (10 MB).
LFS_SIZE_LIMIT_BYTES = 10_000_000
#: Default dataset ``.gitattributes`` patterns that this server stores
#: through LFS whatever their size.
LFS_PATTERNS = ("*.h5", "*.hdf5", "*.parquet", "*.arrow")
_LFS_CONTENT_TYPE = "application/vnd.git-lfs+json"
_REPOSITORY = r"(?P<repository>[^/]+/[^/]+?)"
_REVISION = r"(?P<revision>[^/]+)"
_TRUE_VALUES = frozenset({"1", "true"})

type UploadMode = Literal["regular", "lfs"]


@dataclass(frozen=True)
class _Route:
    method: str
    kind: str
    pattern: re.Pattern[str]
    #: Whether the request must carry the configured token. The LFS upload
    #: address carries a one-time key instead, as a signed address would.
    authenticated: bool = True


_ROUTES: tuple[_Route, ...] = (
    _Route("GET", "whoami", re.compile(r"^/api/whoami-v2$")),
    _Route(
        "GET",
        "repo_info",
        re.compile(rf"^/api/datasets/{_REPOSITORY}/revision/{_REVISION}$"),
    ),
    _Route(
        "GET",
        "tree",
        re.compile(
            rf"^/api/datasets/{_REPOSITORY}/tree/{_REVISION}(?:/(?P<path>.+))?$"
        ),
    ),
    _Route(
        "POST",
        "paths_info",
        re.compile(rf"^/api/datasets/{_REPOSITORY}/paths-info/{_REVISION}$"),
    ),
    _Route(
        "POST",
        "preupload",
        re.compile(rf"^/api/datasets/{_REPOSITORY}/preupload/{_REVISION}$"),
    ),
    _Route(
        "POST",
        "lfs_batch",
        re.compile(rf"^/datasets/{_REPOSITORY}\.git/info/lfs/objects/batch$"),
    ),
    _Route(
        "PUT",
        "lfs_upload",
        re.compile(r"^/lfs-uploads/(?P<oid>[0-9a-f]{64})$"),
        authenticated=False,
    ),
    _Route(
        "POST",
        "lfs_verify",
        re.compile(rf"^/datasets/{_REPOSITORY}\.git/info/lfs/objects/verify$"),
    ),
    _Route(
        "POST",
        "commit",
        re.compile(rf"^/api/datasets/{_REPOSITORY}/commit/{_REVISION}$"),
    ),
    _Route(
        "HEAD",
        "resolve",
        re.compile(rf"^/datasets/{_REPOSITORY}/resolve/{_REVISION}/(?P<path>.+)$"),
    ),
    _Route(
        "GET",
        "resolve",
        re.compile(rf"^/datasets/{_REPOSITORY}/resolve/{_REVISION}/(?P<path>.+)$"),
    ),
)
#: The kinds of request the server answers, as ``requests`` records them.
REQUEST_KINDS = tuple(dict.fromkeys(route.kind for route in _ROUTES))


def git_blob_id(content: bytes) -> str:
    """The Git object ID the Hub reports for a file (its ``oid`` and ETag)."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def _hub_date(moment: datetime) -> str:
    """A date in the Hub's format, which ``huggingface_hub`` parses."""
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _upload_mode(path: str, size: int) -> UploadMode:
    name = PurePosixPath(path).name
    if size > LFS_SIZE_LIMIT_BYTES or any(fnmatch(name, item) for item in LFS_PATTERNS):
        return "lfs"
    return "regular"


@dataclass(frozen=True)
class StandInCommit:
    """One commit: the repository's files after it, and what it wrote.

    ``uploaded_as`` says how each added file travelled: inside the commit
    request (``regular``) or as a reference to a stored LFS object (``lfs``).
    """

    oid: str
    parent: str | None
    title: str
    files: Mapping[str, bytes]
    added: Mapping[str, bytes]
    uploaded_as: Mapping[str, UploadMode]
    date: datetime


@dataclass(frozen=True)
class CommitAttempt:
    """One commit request: the ``parentCommit`` it sent, the head then, and
    the HTTP status it was answered with."""

    parent_commit: str | None
    head: str
    status: int


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    kind: str
    path: str
    status: int


@dataclass
class _Repository:
    commits: list[StandInCommit] = field(default_factory=list)
    attempts: list[CommitAttempt] = field(default_factory=list)
    #: Another writer's files, committed when the next commit request comes.
    land_before_next_commit: dict[str, bytes] | None = None

    @property
    def head(self) -> StandInCommit:
        return self.commits[-1]


@dataclass(frozen=True)
class HubRequest:
    """One request, as a route's handler sees it."""

    method: str
    path: str
    query: Mapping[str, list[str]]
    headers: Mapping[str, str]
    body: bytes
    match: re.Match[str]

    @property
    def repository(self) -> str:
        return self.match["repository"]

    @property
    def revision(self) -> str:
        return unquote(self.match["revision"])

    def json(self) -> Any:
        return json.loads(self.body)


class HubError(Exception):
    """An HTTP error answer, with the headers ``huggingface_hub`` reads."""

    def __init__(
        self, status: int, message: str, *, error_code: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.headers: dict[str, str] = {"X-Error-Message": message}
        if error_code is not None:
            self.headers["X-Error-Code"] = error_code


@dataclass(frozen=True)
class _Answer:
    status: int
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)


def _json_answer(
    payload: Any, *, status: int = 200, content_type: str = "application/json"
) -> _Answer:
    return _Answer(status, json.dumps(payload).encode(), {"Content-Type": content_type})


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
        self._requests: list[RecordedRequest] = []
        self._unmatched: list[str] = []
        self._proxied: list[str] = []
        #: Stored LFS objects by SHA-256, and the key each upload address takes.
        self._lfs_objects: dict[str, bytes] = {}
        self._upload_keys: dict[str, str] = {}
        self._handlers: dict[str, Callable[[HubRequest], _Answer]] = {
            "whoami": self._whoami,
            "repo_info": self._repo_info,
            "tree": self._tree,
            "paths_info": self._paths_info,
            "preupload": self._preupload,
            "lfs_batch": self._lfs_batch,
            "lfs_upload": self._lfs_upload,
            "lfs_verify": self._lfs_verify,
            "commit": self._commit,
            "resolve": self._resolve,
        }
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
                repository, added={}, uploaded_as={}, title="Initial commit"
            ).oid

    def head(self, repository: str) -> str:
        with self._lock:
            return self._repositories[repository].head.oid

    def commits(self, repository: str) -> list[StandInCommit]:
        """Every commit after the repository's first, oldest first."""
        with self._lock:
            return list(self._repositories[repository].commits[1:])

    def commit_attempts(self, repository: str) -> list[CommitAttempt]:
        """Each commit request to ``repository`` checked against its head:
        accepted (200), or refused for naming an older head (412)."""
        with self._lock:
            return list(self._repositories[repository].attempts)

    def land_before_next_commit(
        self, repository: str, files: Mapping[str, bytes]
    ) -> None:
        """Commit ``files`` as another writer just before the next commit
        request is checked, so a request pinned to the old head gets 412."""
        with self._lock:
            self._repositories[repository].land_before_next_commit = dict(files)

    def refuse_writes(self) -> None:
        """Answer as the Hub does for a read-only token.

        ``whoami`` reports the token's role as ``read``, and the preupload,
        LFS batch and commit requests get HTTP 403. Reads still succeed.
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
    def requests(self) -> list[RecordedRequest]:
        """Each request answered, in order of its answer."""
        with self._lock:
            return list(self._requests)

    def statuses(self, kind: str) -> list[int]:
        """The HTTP status of each answered request of ``kind``, in order."""
        if kind not in REQUEST_KINDS:
            raise ValueError(f"unknown request kind: {kind}")
        return [request.status for request in self.requests if request.kind == kind]

    @property
    def unmatched_requests(self) -> list[str]:
        """``METHOD path`` of each request the server has no route for."""
        with self._lock:
            return list(self._unmatched)

    @property
    def proxied_requests(self) -> list[str]:
        """The target of each request sent to this server as a proxy."""
        with self._lock:
            return list(self._proxied)

    @property
    def lfs_objects(self) -> dict[str, bytes]:
        """The stored LFS objects, by SHA-256."""
        with self._lock:
            return dict(self._lfs_objects)

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
        self, method: str, target: str, headers: Mapping[str, str], body: bytes
    ) -> _Answer:
        address = urlsplit(target)
        for route in _ROUTES:
            found = route.pattern.fullmatch(address.path)
            if route.method != method or found is None:
                continue
            self._hold_if_stalled(route.kind)
            request = HubRequest(
                method=method,
                path=address.path,
                query=parse_qs(address.query),
                headers=headers,
                body=body,
                match=found,
            )
            try:
                if route.authenticated:
                    self._authenticate(headers)
                answer = self._handlers[route.kind](request)
            except HubError as error:
                answer = _Answer(
                    error.status,
                    json.dumps({"error": error.message}).encode(),
                    {"Content-Type": "application/json", **error.headers},
                )
            with self._lock:
                self._requests.append(
                    RecordedRequest(method, route.kind, address.path, answer.status)
                )
            return answer
        with self._lock:
            self._unmatched.append(f"{method} {address.path}")
        return _Answer(
            404,
            json.dumps(
                {"error": f"the stand-in Hub has no {method} {address.path}"}
            ).encode(),
            {"Content-Type": "application/json"},
        )

    def _authenticate(self, headers: Mapping[str, str]) -> None:
        if headers.get("authorization") != f"Bearer {self.token}":
            raise HubError(401, "Invalid credentials in Authorization header")

    def _refuse_if_read_only(self) -> None:
        with self._lock:
            refused = self._writes_refused
        if refused:
            raise HubError(403, "You don't have the rights to write to this repository")

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
        added: Mapping[str, bytes],
        uploaded_as: Mapping[str, UploadMode],
        title: str,
    ) -> StandInCommit:
        """A new head with ``added`` written over the current head's files."""
        commits = self._repositories[repository].commits
        head = commits[-1] if commits else None
        self._sequence += 1
        commit = StandInCommit(
            oid=hashlib.sha1(f"{repository}:{self._sequence}".encode()).hexdigest(),
            parent=None if head is None else head.oid,
            title=title,
            files={**({} if head is None else head.files), **added},
            added=dict(added),
            uploaded_as=dict(uploaded_as),
            date=datetime.now(UTC),
        )
        commits.append(commit)
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

    def _whoami(self, request: HubRequest) -> _Answer:
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

    def _repo_info(self, request: HubRequest) -> _Answer:
        with self._lock:
            commit = self._commit_at(request.repository, request.revision)
        return _json_answer(
            {
                "id": request.repository,
                "sha": commit.oid,
                "private": True,
                "lastModified": _hub_date(commit.date),
            }
        )

    def _tree(self, request: HubRequest) -> _Answer:
        repository = request.repository
        folder = unquote(request.match["path"] or "").strip("/")
        recursive = request.query.get("recursive", ["false"])[0].lower() in _TRUE_VALUES
        expand = request.query.get("expand", ["false"])[0].lower() in _TRUE_VALUES
        with self._lock:
            commit = self._commit_at(repository, request.revision)
            base = f"{folder}/" if folder else ""
            below = sorted(name for name in commit.files if name.startswith(base))
            if folder and (folder in commit.files or not below):
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

    def _paths_info(self, request: HubRequest) -> _Answer:
        fields = parse_qs(request.body.decode(), keep_blank_values=True)
        expand = fields.get("expand", ["false"])[0].lower() in _TRUE_VALUES
        with self._lock:
            commit = self._commit_at(request.repository, request.revision)
            entries = [
                self._entry(request.repository, commit, path.strip("/"), expand=expand)
                for path in fields.get("paths", [])
            ]
        return _json_answer([entry for entry in entries if entry is not None])

    def _resolve(self, request: HubRequest) -> _Answer:
        path = unquote(request.match["path"])
        with self._lock:
            commit = self._commit_at(request.repository, request.revision)
            content = commit.files.get(path)
        if content is None:
            error = HubError(404, "Entry not found", error_code="EntryNotFound")
            error.headers["X-Repo-Commit"] = commit.oid
            raise error
        return _Answer(
            200,
            content if request.method == "GET" else b"",
            {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(content)),
                "ETag": f'"{git_blob_id(content)}"',
                "X-Repo-Commit": commit.oid,
            },
        )

    def _preupload(self, request: HubRequest) -> _Answer:
        self._refuse_if_read_only()
        with self._lock:
            commit = self._commit_at(request.repository, request.revision)
        return _json_answer(
            {
                "files": [
                    {
                        "path": item["path"],
                        "uploadMode": _upload_mode(item["path"], int(item["size"])),
                        "shouldIgnore": False,
                    }
                    for item in request.json()["files"]
                ],
                "commitOid": commit.oid,
            }
        )

    def _lfs_batch(self, request: HubRequest) -> _Answer:
        """Offer the ``basic`` transfer for each object not yet stored."""
        self._refuse_if_read_only()
        payload = request.json()
        if payload.get("operation") != "upload":
            raise HubError(400, "the stand-in Hub accepts only LFS uploads")
        if "basic" not in (payload.get("transfers") or ["basic"]):
            raise HubError(400, "the stand-in Hub offers only the basic transfer")
        with self._lock:
            self._repository(request.repository)
            objects = []
            for item in payload["objects"]:
                oid, size = str(item["oid"]), int(item["size"])
                answer: dict[str, Any] = {"oid": oid, "size": size}
                stored = self._lfs_objects.get(oid)
                if stored is None or len(stored) != size:
                    key = self._upload_keys.setdefault(oid, secrets.token_hex(16))
                    answer["authenticated"] = True
                    answer["actions"] = {
                        "upload": {"href": f"{self.url}/lfs-uploads/{oid}?key={key}"},
                        "verify": {
                            "href": f"{self.url}/datasets/{request.repository}.git"
                            "/info/lfs/objects/verify"
                        },
                    }
                objects.append(answer)
        return _json_answer(
            {"transfer": "basic", "objects": objects, "hash_algo": "sha256"},
            content_type=_LFS_CONTENT_TYPE,
        )

    def _lfs_upload(self, request: HubRequest) -> _Answer:
        """Store an object's bytes, sent to the address the batch answer gave."""
        oid = request.match["oid"]
        with self._lock:
            expected_key = self._upload_keys.get(oid)
        if expected_key is None or request.query.get("key") != [expected_key]:
            raise HubError(403, "the upload address is not valid")
        if hashlib.sha256(request.body).hexdigest() != oid:
            raise HubError(400, "the uploaded bytes do not match the object ID")
        with self._lock:
            self._lfs_objects[oid] = request.body
        return _Answer(200)

    def _lfs_verify(self, request: HubRequest) -> _Answer:
        payload = request.json()
        with self._lock:
            stored = self._lfs_objects.get(str(payload.get("oid")))
        if stored is None or len(stored) != payload.get("size"):
            raise HubError(404, "the LFS object is not stored")
        return _json_answer({"success": True}, content_type=_LFS_CONTENT_TYPE)

    def _commit_contents(
        self, body: bytes
    ) -> tuple[Mapping[str, Any], dict[str, bytes], dict[str, UploadMode]]:
        """The commit's header, and each added file with how it travelled."""
        header: Mapping[str, Any] | None = None
        added: dict[str, bytes] = {}
        uploaded_as: dict[str, UploadMode] = {}
        for text in body.splitlines():
            if not text.strip():
                continue
            item = json.loads(text)
            key, value = item.get("key"), item.get("value") or {}
            if key == "header":
                header = value
            elif key == "file" and value.get("encoding") == "base64":
                added[value["path"]] = base64.b64decode(value["content"])
                uploaded_as[value["path"]] = "regular"
            elif key == "lfsFile" and value.get("algo") == "sha256":
                with self._lock:
                    stored = self._lfs_objects.get(str(value.get("oid")))
                if stored is None or len(stored) != value.get("size"):
                    raise HubError(
                        400, f"no stored LFS object for {value.get('path')!r}"
                    )
                added[value["path"]] = stored
                uploaded_as[value["path"]] = "lfs"
            else:
                raise HubError(
                    400, f"the stand-in Hub accepts only file additions, not {key!r}"
                )
        if header is None or not header.get("summary"):
            raise HubError(400, "the commit has no summary")
        return header, added, uploaded_as

    def _commit(self, request: HubRequest) -> _Answer:
        self._refuse_if_read_only()
        if request.revision != "main":
            raise HubError(400, "the stand-in Hub commits only to main")
        header, added, uploaded_as = self._commit_contents(request.body)
        parent = header.get("parentCommit")
        with self._lock:
            repository = self._repository(request.repository)
            if repository.land_before_next_commit is not None:
                other_files = repository.land_before_next_commit
                repository.land_before_next_commit = None
                self._append_commit(
                    request.repository,
                    added=other_files,
                    uploaded_as=dict.fromkeys(other_files, "regular"),
                    title="Another writer's commit",
                )
            head = repository.head.oid
            if parent is not None and parent != head:
                repository.attempts.append(CommitAttempt(parent, head, 412))
                raise HubError(412, "A commit has happened since. Please refresh.")
            commit = self._append_commit(
                request.repository,
                added=added,
                uploaded_as=uploaded_as,
                title=str(header["summary"]).split("\n", 1)[0],
            )
            repository.attempts.append(CommitAttempt(parent, head, 200))
        return _json_answer(
            {
                "success": True,
                "commitOid": commit.oid,
                "commitUrl": f"{self.url}/datasets/{request.repository}/commit/"
                f"{commit.oid}",
                "hookOutput": "",
                "pullRequestUrl": None,
            }
        )


def _handler_for(hub: StandInHub) -> type[BaseHTTPRequestHandler]:
    class StandInHubRequestHandler(BaseHTTPRequestHandler):
        """Pass each request to ``hub``; refuse proxy requests."""

        def _body(self) -> bytes:
            # huggingface_hub 1.33.0 sends every body, the LFS upload's
            # included, with a Content-Length.
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

        do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_PATCH = _handle  # noqa: N815

        def log_message(self, message_format: str, *args: Any) -> None:
            """Keep the test output free of one line per request."""

    return StandInHubRequestHandler
