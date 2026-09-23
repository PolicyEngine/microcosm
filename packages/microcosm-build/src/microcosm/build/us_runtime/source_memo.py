"""Persistent, content-addressed memo for deterministic US source derivations.

Most of a native run's CPU outside the graph goes to US source owners
re-deriving the same values from the same pinned archives: decompressing and
parsing every ACS member, inventorying them and assembling the full raw
catalogue (see ``experiments/native-source-auth-memo``). Each derivation is a
deterministic function of exact source bytes, explicit parameters and code.
This module lets an owner record such a value once and reuse it later -- in
the same run or a later one -- only when all three are exactly what produced
it.

It is not a source of authority. Owners still capture their current source
bytes and check them against the pins, still construct, register and seal
every issued object, and still run every live-object, mutation and producer
check. An entry only stands in for the pure computation those same bytes
would have produced under that same code. A refusal is never recorded: only
a computation that returned is.

Keys
    ``sha256("microcosm.us.source-memo.v1\\0" + canonical(document))`` where the
    document holds the namespace, a runtime identity (interpreter, platform,
    optimisation level, zlib, the digests of the stdlib parser modules, of the
    binary that provides the C ``_csv``/``_json``/``zlib`` code and of this
    module), the owner's code identity, every input as ``{role, sha256,
    bytes}`` and the canonical parameters. ``canonical`` is sorted compact JSON
    encoded as strict UTF-8, so distinct values have distinct keys; a value it
    cannot encode exactly (a lone surrogate, a NaN) never forms a key. File
    inputs are hashed through one no-follow descriptor at call time; stat
    metadata and modification times are never trusted.

Code identity
    Each owner passes its own recorded producer or implementation identity
    (the one its receipts already carry) plus ``live_code(*modules)``: the
    SHA-256 of each module's file, a check that every function and method the
    module defines still runs the code compiled from those bytes, the identity
    of every function bound in its namespace and its immutable constants
    (limits, pins, field lists). A monkeypatched constant or function
    therefore changes the key, and loaded code that no longer matches its file
    bypasses the memo entirely.

Entries fail closed
    ``index/<kk>/<key>.json`` names content-addressed ``blobs/<dd>/<sha256>``
    and carries an HMAC-SHA256 made with a per-user key kept outside the memo
    root, so a copied or foreign memo directory confers nothing. A lookup
    accepts an entry only if the index is a bounded regular non-symlink file
    in canonical form, its MAC verifies, its key document equals the requested
    one byte for byte and every blob is a regular non-symlink file of the
    recorded size and SHA-256. Anything else is a counted miss, never a
    partial trust: the owner recomputes from source and the fresh entry
    atomically replaces the rejected one. On a miss the code identity and the
    file inputs are recomputed after the computation; a value is stored only
    if neither moved and its encoding is proven to decode to exactly the
    computed value.

Activation
    Off by default, so an ordinary call runs exactly the unmemoized path.
    ``with source_memo(root):`` enables it for the current context;
    ``source_memo(None)`` disables it inside that context. Otherwise the
    ``MICROCOSM_US_SOURCE_MEMO`` environment variable names the root.
    ``MICROCOSM_US_SOURCE_MEMO_KEY`` (or ``key_path=``) names the HMAC key
    file, by default ``$XDG_CONFIG_HOME/microcosm/us-source-memo.key`` or
    ``~/.config/microcosm/us-source-memo.key``, created 0600 on first use.
    The memo is a cache: deleting its root is always safe.
"""

from __future__ import annotations

import _csv
import _json
import contextvars
import csv
import hashlib
import hmac
import importlib
import json
import json.decoder
import json.encoder
import json.scanner
import math
import os
import platform
import re
import secrets
import stat
import struct
import sys
import sysconfig
import tempfile
import threading
import time
import zipfile
import zlib
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FunctionType

import numpy as np
import pandas as pd

PROTOCOL = "microcosm.us.source-memo.v1"
ROOT_ENV = "MICROCOSM_US_SOURCE_MEMO"
KEY_ENV = "MICROCOSM_US_SOURCE_MEMO_KEY"
_KEY_DOMAIN = PROTOCOL.encode("ascii") + b"\0"
_MAC_DOMAIN = PROTOCOL.encode("ascii") + b"\0index\0"
_LAYOUT = "v1"
_SECRET_BYTES = 32
_INDEX_MAX_BYTES = 1024**2
_BLOB_MAX_BYTES = 16 * 1024**3
_BLOBS_MAX = 64
_CHUNK = 1024**2
_COMPILED_MAX = 64
_HEX = frozenset("0123456789abcdef")
_DISABLED = object()
_SKIP = object()
_ACTIVE = contextvars.ContextVar("microcosm_us_source_memo", default=None)
_LOCK = threading.RLock()
_STATISTICS = Counter()
_ENVIRONMENT_MEMOS = {}
_COMPILED = {}
_RUNTIME = None
_SELF_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class SourceMemoError(ValueError):
    """Static refusal code; never names a path, key or source value."""


def _require(condition, code):
    if not condition:
        raise SourceMemoError(code)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value) -> bytes:
    """Sorted compact JSON as strict UTF-8: the only encoding of keys and indexes.

    Strict UTF-8 refuses a lone surrogate instead of escaping it, so two
    distinct strings never share an encoding.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def ordered(value) -> bytes:
    """Compact strict-UTF-8 JSON preserving mapping order, for JSON-native values."""
    return json.dumps(
        value, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _is_hex(value, size=64):
    return type(value) is str and len(value) == size and set(value) <= _HEX


def _count(namespace, event, amount=1):
    with _LOCK:
        _STATISTICS[f"{namespace}:{event}"] += amount
        _STATISTICS[f"*:{event}"] += amount


def statistics() -> dict:
    """Process-wide counters and seconds, by ``namespace:event`` and ``*:event``."""
    with _LOCK:
        return {key: _STATISTICS[key] for key in sorted(_STATISTICS)}


def reset_statistics() -> None:
    with _LOCK:
        _STATISTICS.clear()


# -- identities -------------------------------------------------------------


def _binary_sha(module) -> str:
    """Digest of the file providing a C module, libpython when it is built in.

    The interpreter is usually reached through a virtual environment's
    symlink, so this one path is resolved before it is hashed.
    """
    path = getattr(module, "__file__", None)
    if path is None:
        path = (
            Path(sysconfig.get_config_var("LIBDIR"))
            / sysconfig.get_config_var("LDLIBRARY")
            if sysconfig.get_config_var("Py_ENABLE_SHARED")
            else Path(sys.executable)
        )
    return file_sha256(os.path.realpath(path))[0]


def runtime_identity() -> dict:
    """Interpreter, platform and parser-provider identity, once per process."""
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is None:
            _RUNTIME = {
                "memo_sha256": _SELF_SHA256,
                "python": sys.version,
                "hexversion": sys.hexversion,
                "cache_tag": sys.implementation.cache_tag,
                "optimize": sys.flags.optimize,
                "machine": platform.machine(),
                "system": platform.system(),
                "byteorder": sys.byteorder,
                "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
                "c_providers_sha256": {
                    name: _binary_sha(module)
                    for name, module in (
                        ("_csv", _csv),
                        ("_json", _json),
                        ("zlib", zlib),
                    )
                },
                "stdlib_sha256": {
                    name: _sha(Path(module.__file__).read_bytes())
                    for name, module in (
                        ("csv", csv),
                        ("json", json),
                        ("json.decoder", json.decoder),
                        ("json.encoder", json.encoder),
                        ("json.scanner", json.scanner),
                        ("zipfile", zipfile),
                    )
                },
            }
        return json.loads(canonical(_RUNTIME))


def _self_unchanged() -> bool:
    try:
        return _sha(Path(__file__).read_bytes()) == _SELF_SHA256
    except OSError:
        return False


def _compiled(path: str, payload: bytes) -> dict:
    """Code objects compiled from exact module bytes, by qualified name."""
    key = (path, _sha(payload))
    with _LOCK:
        codes = _COMPILED.get(key)
    if codes is None:
        codes = {}

        def visit(code):
            codes.setdefault(code.co_qualname, []).append(code)
            for value in code.co_consts:
                if isinstance(value, CodeType):
                    visit(value)

        visit(compile(payload, path, "exec", dont_inherit=True))
        with _LOCK:
            while len(_COMPILED) >= _COMPILED_MAX:
                del _COMPILED[next(iter(_COMPILED))]
            _COMPILED[key] = codes
    return codes


def _immutable(value, depth=0):
    """A JSON form of an immutable constant, or _SKIP for anything else."""
    _require(depth <= 32, "MEMO_CONSTANT_DEPTH")
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return [kind.__name__, value]
    if kind is float:
        return ["float", value.hex()]
    if kind is bytes:
        return ["bytes", value.hex()]
    if kind is re.Pattern:
        return ["pattern", _immutable(value.pattern, depth + 1), value.flags]
    if kind in (tuple, frozenset):
        items = [_immutable(item, depth + 1) for item in value]
        if any(item is _SKIP for item in items):
            return _SKIP
        if kind is frozenset:
            items = sorted(items, key=canonical)
        return [kind.__name__, items]
    return _SKIP


def _function(value, module, codes: dict, seen: set) -> list:
    """Identity of one bound function, checking code the module itself defines."""
    if id(value) in seen:
        return ["seen", value.__qualname__]
    seen.add(id(value))
    code = value.__code__
    own = value.__globals__ is vars(module) and code.co_filename != "<string>"
    if own:
        candidates = codes.get(code.co_qualname, ())
        _require(any(code == item for item in candidates), "MEMO_LIVE_CODE")
    return [
        value.__module__,
        value.__qualname__,
        code.co_qualname,
        own,
        [
            _function(cell.cell_contents, module, codes, seen)
            for cell in value.__closure__ or ()
            if isinstance(cell.cell_contents, FunctionType)
        ],
    ]


def live_code(*modules) -> dict:
    """Current file bytes, checked loaded code and constants of ``modules``.

    Raises ``SourceMemoError`` when a function or method a module defines no
    longer runs the code compiled from its file, so the memo is bypassed.
    """
    result = {}
    for module in modules:
        path = module.__file__
        payload = Path(path).read_bytes()
        codes = _compiled(path, payload)
        functions, constants, seen = {}, {}, set()
        for name, value in sorted(vars(module).items()):
            if name.startswith("__"):
                continue
            if isinstance(value, FunctionType):
                functions[name] = _function(value, module, codes, seen)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                for method_name, method in sorted(vars(value).items()):
                    if isinstance(method, (staticmethod, classmethod)):
                        method = method.__func__
                    parts = (
                        (method.fget, method.fset, method.fdel)
                        if isinstance(method, property)
                        else (method,)
                    )
                    for index, part in enumerate(parts):
                        if isinstance(part, FunctionType):
                            functions[f"{name}.{method_name}/{index}"] = _function(
                                part, module, codes, seen
                            )
            else:
                form = _immutable(value)
                if form is not _SKIP:
                    constants[name] = form
        result[module.__name__] = {
            "sha256": _sha(payload),
            "functions": functions,
            "constants": constants,
        }
    return result


@dataclass(frozen=True)
class FileInput:
    """A source file hashed at call time through a no-follow descriptor."""

    role: str
    path: object


@dataclass(frozen=True)
class DigestInput:
    """Bytes the owner already holds and hashed itself (never a claimed digest)."""

    role: str
    sha256: str
    size: int


def _stat_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def file_sha256(path, *, maximum=None) -> tuple[str, int]:
    """SHA-256 and size of one regular non-symlink file, unchanged while read."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), "MEMO_INPUT_KIND")
        _require(maximum is None or before.st_size <= maximum, "MEMO_INPUT_SIZE")
        digest, count = hashlib.sha256(), 0
        with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as stream:
            while chunk := stream.read(_CHUNK):
                count += len(chunk)
                _require(count <= before.st_size, "MEMO_INPUT_CHANGED")
                digest.update(chunk)
        _require(
            count == before.st_size
            and _stat_identity(before) == _stat_identity(os.fstat(descriptor)),
            "MEMO_INPUT_CHANGED",
        )
        return digest.hexdigest(), count
    finally:
        os.close(descriptor)


def _inputs(inputs):
    result = []
    for item in inputs:
        if type(item) is FileInput:
            digest, size = file_sha256(item.path)
        else:
            _require(type(item) is DigestInput, "MEMO_INPUT_TYPE")
            _require(
                _is_hex(item.sha256) and type(item.size) is int and item.size >= 0,
                "MEMO_INPUT_DIGEST",
            )
            digest, size = item.sha256, item.size
        _require(type(item.role) is str and item.role, "MEMO_INPUT_ROLE")
        result.append({"role": item.role, "sha256": digest, "bytes": size})
    return result


# -- strict value comparison --------------------------------------------------


def strict_equal(left, right) -> bool:
    """Type-exact deep equality, including mapping order and signed zeros."""
    stack = [(left, right)]
    while stack:
        a, b = stack.pop()
        if type(a) is not type(b):
            return False
        kind = type(a)
        if kind in (list, tuple):
            if len(a) != len(b):
                return False
            stack.extend(zip(a, b, strict=True))
        elif kind is dict:
            if list(a) != list(b):
                return False
            stack.extend(zip(a.values(), b.values(), strict=True))
            stack.extend(zip(a, b, strict=True))
        elif kind is float:
            if not (a == b and math.copysign(1.0, a) == math.copysign(1.0, b)):
                return False
        elif kind in (str, int, bool, bytes, type(None)):
            if a != b:
                return False
        else:
            return False
    return True


def exact_text(value) -> bool:
    """A str that strict UTF-8 JSON round-trips exactly: no surrogate code points.

    Strictly decoded UTF-8 never contains one, and ``ordered`` refuses to
    encode one, so this only lets a proof refuse early instead of at encoding.
    """
    if value.isascii():
        return True
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def json_native(value) -> bool:
    """Only dict(str keys)/list/str/int/bool/None: ``ordered`` round-trips exactly.

    Strings must be surrogate-free (``exact_text``). Mapping order survives
    because ``ordered`` does not sort keys and the decoder preserves order.
    """
    stack = [value]
    while stack:
        item = stack.pop()
        kind = type(item)
        if kind is str:
            if not exact_text(item):
                return False
        elif kind is list:
            stack.extend(item)
        elif kind is dict:
            if not all(type(key) is str and exact_text(key) for key in item):
                return False
            stack.extend(item.values())
        elif kind not in (int, bool, type(None)):
            return False
    return True


# -- the store ------------------------------------------------------------------


def _default_key_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "microcosm" / "us-source-memo.key"


def _private_directory(path: Path) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = path.lstat()
    _require(stat.S_ISDIR(info.st_mode), "MEMO_DIRECTORY")
    _require(info.st_uid == os.getuid(), "MEMO_DIRECTORY_OWNER")
    _require(not info.st_mode & 0o022, "MEMO_DIRECTORY_MODE")
    return path


def _load_secret(path: Path) -> bytes:
    path = Path(os.path.abspath(path))
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Write a complete key under a private name, then link it into place:
        # a concurrent first use either wins the link or reads the winner's key.
        descriptor, temporary = tempfile.mkstemp(prefix=".key-", dir=path.parent)
        try:
            os.write(descriptor, secrets.token_bytes(_SECRET_BYTES))
            os.fsync(descriptor)
            os.close(descriptor)
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        _require(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
            and not info.st_mode & 0o077
            and info.st_size == _SECRET_BYTES,
            "MEMO_KEY_FILE",
        )
        secret = os.read(descriptor, _SECRET_BYTES + 1)
        _require(len(secret) == _SECRET_BYTES, "MEMO_KEY_FILE")
        return secret
    finally:
        os.close(descriptor)


def _read_regular(path: Path, maximum: int, exact=None) -> bytes | None:
    """Whole regular non-symlink file, or None if absent. Raises on anything odd."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), "MEMO_ENTRY_KIND")
        _require(before.st_size <= maximum, "MEMO_ENTRY_SIZE")
        _require(exact is None or before.st_size == exact, "MEMO_ENTRY_SIZE")
        # One read per GiB: a blob up to 1 GiB is a single allocation, never
        # copied again. One byte past the recorded size detects growth.
        parts, remaining = [], before.st_size + 1
        with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as stream:
            while remaining > 0:
                chunk = stream.read(min(remaining, 1024**3))
                if not chunk:
                    break
                parts.append(chunk)
                remaining -= len(chunk)
        value = parts[0] if len(parts) == 1 else b"".join(parts)
        del parts
        _require(
            len(value) == before.st_size
            and _stat_identity(before) == _stat_identity(os.fstat(descriptor)),
            "MEMO_ENTRY_CHANGED",
        )
        return value
    finally:
        os.close(descriptor)


class _Memo:
    def __init__(self, root, key_path=None):
        root = Path(os.path.realpath(os.path.abspath(root)))
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _private_directory(root)
        key_path = Path(
            os.path.realpath(
                os.path.abspath(
                    key_path or os.environ.get(KEY_ENV) or _default_key_path()
                )
            )
        )
        _require(not key_path.is_relative_to(root), "MEMO_KEY_INSIDE_ROOT")
        self._secret = _load_secret(key_path)
        self.root = _private_directory(root / _LAYOUT)
        for name in ("index", "blobs", "tmp"):
            _private_directory(self.root / name)

    # Paths are derived only from validated lowercase hex digests.
    def _index_path(self, key: str) -> Path:
        return self.root / "index" / key[:2] / f"{key}.json"

    def _blob_path(self, digest: str) -> Path:
        return self.root / "blobs" / digest[:2] / digest

    def _mac(self, body: bytes) -> str:
        return hmac.new(self._secret, _MAC_DOMAIN + body, hashlib.sha256).hexdigest()

    def lookup(self, namespace, key, document_bytes):
        """Exact verified blobs, or None. Never raises for a bad entry."""
        try:
            raw = _read_regular(self._index_path(key), _INDEX_MAX_BYTES)
            if raw is None:
                return None
            entry = json.loads(raw)
            _require(type(entry) is dict and canonical(entry) == raw, "INDEX_FORM")
            _require(
                set(entry) == {"protocol", "key", "document", "blobs", "mac"},
                "INDEX_FIELDS",
            )
            mac = entry.pop("mac")
            _require(
                _is_hex(mac) and hmac.compare_digest(mac, self._mac(canonical(entry))),
                "INDEX_MAC",
            )
            _require(
                entry["protocol"] == PROTOCOL
                and entry["key"] == key
                and canonical(entry["document"]) == document_bytes,
                "INDEX_KEY",
            )
            blobs = entry["blobs"]
            _require(
                type(blobs) is list
                and 0 < len(blobs) <= _BLOBS_MAX
                and all(
                    type(item) is dict
                    and set(item) == {"sha256", "bytes"}
                    and _is_hex(item["sha256"])
                    and type(item["bytes"]) is int
                    and 0 <= item["bytes"] <= _BLOB_MAX_BYTES
                    for item in blobs
                ),
                "INDEX_BLOBS",
            )
            values = []
            for item in blobs:
                value = _read_regular(
                    self._blob_path(item["sha256"]), _BLOB_MAX_BYTES, item["bytes"]
                )
                _require(value is not None, "BLOB_MISSING")
                _require(_sha(value) == item["sha256"], "BLOB_DIGEST")
                values.append(value)
            return values
        except SourceMemoError as error:
            _count(namespace, f"rejected:{error}")
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            _count(namespace, "rejected:ENTRY_UNREADABLE")
        return None

    def _publish(self, target: Path, data: bytes):
        _private_directory(target.parent)
        descriptor, temporary = tempfile.mkstemp(prefix=".memo-", dir=self.root / "tmp")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o400)
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def _blob_present(self, path, digest, size):
        try:
            return file_sha256(path, maximum=size) == (digest, size)
        except (OSError, SourceMemoError):
            return False

    def store(self, namespace, key, document, blobs):
        # lookup() refuses any other count, so such an entry would never hit.
        _require(type(blobs) is list and 0 < len(blobs) <= _BLOBS_MAX, "BLOB_COUNT")
        described = []
        for value in blobs:
            _require(
                type(value) is bytes and len(value) <= _BLOB_MAX_BYTES, "BLOB_SIZE"
            )
            digest = _sha(value)
            described.append({"sha256": digest, "bytes": len(value)})
            path = self._blob_path(digest)
            if not self._blob_present(path, digest, len(value)):
                self._publish(path, value)
        entry = {
            "protocol": PROTOCOL,
            "key": key,
            "document": document,
            "blobs": described,
        }
        entry["mac"] = self._mac(canonical(entry))
        body = canonical(entry)
        _require(len(body) <= _INDEX_MAX_BYTES, "INDEX_SIZE")
        self._publish(self._index_path(key), body)
        _count(namespace, "stored")
        _count(namespace, "stored_bytes", sum(len(value) for value in blobs))


@contextmanager
def source_memo(root, *, key_path=None):
    """Enable the memo at ``root`` for this context; ``None`` disables it."""
    memo = _DISABLED if root is None else _Memo(root, key_path)
    token = _ACTIVE.set(memo)
    try:
        yield None if memo is _DISABLED else memo
    finally:
        _ACTIVE.reset(token)


def _current():
    memo = _ACTIVE.get()
    if memo is _DISABLED:
        return None
    if memo is not None:
        return memo
    root = os.environ.get(ROOT_ENV)
    if not root:
        return None
    selector = (root, os.environ.get(KEY_ENV))
    with _LOCK:
        memo = _ENVIRONMENT_MEMOS.get(selector)
        if memo is None:
            memo = _ENVIRONMENT_MEMOS[selector] = _Memo(root, selector[1])
    return memo


def enabled() -> bool:
    return _current() is not None


def memoized(
    namespace, *, code, inputs, parameters, compute, encode, decode, proof=None
):
    """``compute()``, or the value an identical earlier computation recorded.

    ``code`` returns the owner's JSON code identity and is evaluated before the
    lookup and again after a miss's computation. ``inputs`` are FileInput or
    DigestInput items and ``parameters`` a canonical JSON value; either may be
    a zero-argument callable, evaluated only when the memo is enabled.
    ``encode(value)`` returns a list of bytes blobs and ``decode(blobs)``
    rebuilds the value. ``proof(value, blobs)`` must return True only if
    ``decode(blobs)`` is exactly ``value``; by default the decoded value is
    compared with ``strict_equal``. Exceptions from ``compute`` always
    propagate unchanged, so owner refusals keep their codes and timing, and
    nothing is recorded for them.
    """
    memo = _current()
    if memo is None:
        return compute()
    started = time.perf_counter()
    if not _self_unchanged():
        _count(namespace, "bypassed:MEMO_CODE_CHANGED")
        return compute()
    try:
        # Callables keep a disabled memo free of any key-forming work.
        inputs = tuple(inputs() if callable(inputs) else inputs)
        parameters = parameters() if callable(parameters) else parameters
        code_before = code()
        document = {
            "protocol": PROTOCOL,
            "namespace": namespace,
            "runtime": runtime_identity(),
            "code": code_before,
            "inputs": _inputs(inputs),
            "parameters": parameters,
        }
        document_bytes = canonical(document)
    except Exception as error:
        # An identity that cannot be formed is never a key. The owner's own
        # path then runs unchanged and refuses, if it must, with its own code
        # at its own point.
        _count(namespace, f"bypassed:{type(error).__name__}")
        return compute()
    key = _sha(_KEY_DOMAIN + document_bytes)
    blobs = memo.lookup(namespace, key, document_bytes)
    _count(namespace, "key_seconds", time.perf_counter() - started)
    if blobs is not None:
        decoding = time.perf_counter()
        try:
            value = decode(blobs)
        except Exception:
            _count(namespace, "rejected:DECODE")
        else:
            _count(namespace, "hit")
            _count(namespace, "decode_seconds", time.perf_counter() - decoding)
            return value
        del blobs
    _count(namespace, "miss")
    computing = time.perf_counter()
    value = compute()
    _count(namespace, "compute_seconds", time.perf_counter() - computing)
    storing = time.perf_counter()
    try:
        if code() != code_before:
            _count(namespace, "not_stored:CODE_CHANGED")
            return value
        if _inputs(inputs) != document["inputs"] or not _self_unchanged():
            _count(namespace, "not_stored:INPUT_CHANGED")
            return value
        encoded = encode(value)
        _require(
            type(encoded) is list and all(type(item) is bytes for item in encoded),
            "ENCODED_TYPE",
        )
        verified = (
            proof(value, encoded)
            if proof is not None
            else strict_equal(decode(encoded), value)
        )
        if verified is not True:
            _count(namespace, "not_stored:ROUND_TRIP")
            return value
        memo.store(namespace, key, document, encoded)
    except Exception as error:
        # Storing is an optimization for later calls: a failure to record
        # never changes the value this call returns.
        _count(namespace, f"not_stored:{type(error).__name__}")
    finally:
        _count(namespace, "store_seconds", time.perf_counter() - storing)
    return value


# -- small codecs shared by owners ---------------------------------------------


def encode_json(value):
    """One blob of order-preserving JSON; refuses anything not JSON-native."""
    _require(json_native(value), "NOT_JSON_NATIVE")
    return [ordered(value)]


def decode_json(blobs: list[bytes]):
    _require(len(blobs) == 1, "BLOB_COUNT")
    return json.loads(blobs[0])


def prove_json(value, blobs: list[bytes]) -> bool:
    """``encode_json`` admits only JSON-native values, which ``ordered`` round-trips
    exactly; decoding again would only repeat that guarantee at the cost of a
    second copy of the value."""
    return len(blobs) == 1 and json_native(value)


# -- exact typed values ------------------------------------------------------------

_TYPED_DEPTH = 64


def _to_typed(value, depth: int):
    _require(depth <= _TYPED_DEPTH, "TYPED_DEPTH")
    kind = type(value)
    if value is None or kind in (bool, int):
        return value
    if kind is str:
        _require(exact_text(value), "TYPED_TEXT")
        return value
    if kind is float:
        return ["F", struct.pack(">d", value).hex()]
    if kind is bytes:
        return ["B", value.hex()]
    if kind in (list, tuple):
        tag = "L" if kind is list else "T"
        return [tag, *(_to_typed(item, depth + 1) for item in value)]
    if kind is dict:
        _require(
            all(type(key) is str and exact_text(key) for key in value), "TYPED_KEY"
        )
        return [
            "D",
            *([key, _to_typed(item, depth + 1)] for key, item in value.items()),
        ]
    raise SourceMemoError("TYPED_VALUE")


def _from_typed(value, depth: int):
    _require(depth <= _TYPED_DEPTH, "TYPED_DEPTH")
    if type(value) is not list:
        _require(value is None or type(value) in (bool, int, str), "TYPED_TAG")
        return value
    _require(bool(value), "TYPED_TAG")
    tag, items = value[0], value[1:]
    if tag in ("F", "B"):
        _require(len(items) == 1 and type(items[0]) is str, "TYPED_TAG")
        raw = bytes.fromhex(items[0])
        return struct.unpack(">d", raw)[0] if tag == "F" else raw
    if tag in ("L", "T"):
        decoded = [_from_typed(item, depth + 1) for item in items]
        return decoded if tag == "L" else tuple(decoded)
    _require(tag == "D", "TYPED_TAG")
    result = {}
    for pair in items:
        _require(type(pair) is list and len(pair) == 2, "TYPED_KEY")
        key = pair[0]
        _require(type(key) is str and key not in result, "TYPED_KEY")
        result[key] = _from_typed(pair[1], depth + 1)
    return result


def encode_typed(value) -> bytes:
    """Exact JSON of None/bool/int/str/float/bytes/list/tuple/dict (str keys).

    Every container is a tagged list, so tuples stay tuples, mapping order
    survives and floats retain their IEEE-754 bits, including NaN payloads.
    """
    return ordered(_to_typed(value, 0))


def decode_typed(raw: bytes):
    return _from_typed(json.loads(raw), 0)


# -- exact data frames ------------------------------------------------------------

_NUMPY_KINDS = frozenset("biuf")
_STRING_STORAGES = frozenset({"python", "pyarrow"})


def _string_form(dtype) -> list | None:
    """``[storage, na]`` of an exact pandas StringDtype, else None."""
    if type(dtype) is not pd.StringDtype or dtype.storage not in _STRING_STORAGES:
        return None
    na = dtype.na_value
    if na is pd.NA:
        return [dtype.storage, "NA"]
    if type(na) is float and math.isnan(na):
        return [dtype.storage, "nan"]
    return None


def _string_dtype(form: list):
    _require(
        type(form) is list
        and len(form) == 2
        and form[0] in _STRING_STORAGES
        and form[1] in ("NA", "nan"),
        "FRAME_DTYPE",
    )
    return pd.StringDtype(form[0], na_value=pd.NA if form[1] == "NA" else np.nan)


def _numpy_dtype(form: str):
    _require(type(form) is str, "FRAME_DTYPE")
    dtype = np.dtype(form)
    _require(dtype.kind in _NUMPY_KINDS and dtype.str == form, "FRAME_DTYPE")
    return dtype


def _string_values(series) -> list:
    values = series.array.to_numpy(dtype=object, na_value=None).tolist()
    missing = series.isna().to_numpy(dtype=bool)
    _require(len(values) == len(missing), "FRAME_VALUES")
    for value, absent in zip(values, missing.tolist(), strict=True):
        if absent:
            _require(value is None, "FRAME_VALUES")
        else:
            _require(type(value) is str and exact_text(value), "FRAME_VALUES")
    return values


def _object_values(series) -> list:
    """Encode only inert scalar cells; never serialize arbitrary Python objects.

    Object columns can contain distinct missing markers. Keep None, pd.NA
    and every float's bits separate rather than normalizing through isna().
    """
    values = []
    for value in series.to_numpy(copy=False):
        if value is pd.NA:
            values.append(["NA"])
        else:
            _require(
                value is None or type(value) in (bool, int, float, str, bytes),
                "FRAME_OBJECT",
            )
            values.append(_to_typed(value, 0))
    return values


def _decode_object_values(raw: bytes, rows: int) -> list:
    values = json.loads(raw)
    _require(type(values) is list and len(values) == rows, "FRAME_ROWS")
    decoded = []
    for value in values:
        if value == ["NA"]:
            decoded.append(pd.NA)
        else:
            item = _from_typed(value, 0)
            _require(
                item is None or type(item) in (bool, int, float, str, bytes),
                "FRAME_OBJECT",
            )
            decoded.append(item)
    return decoded


def encode_frame(frame) -> list[bytes]:
    """A header blob, then one blob per column; refuses any other frame shape.

    Admits exactly: a plain DataFrame with no attrs, a RangeIndex, unique str
    column labels held in an object or StringDtype index, and columns that are
    numpy bool/int/uint/float (stored as their raw bytes, so NaN payloads and
    signed zeros survive) or a python/pyarrow StringDtype with either missing
    marker (stored as JSON text or null), or object columns of exact Python
    scalar cells (None/bool/int/float/str/bytes/pd.NA, with tagged encoding).
    """
    _require(type(frame) is pd.DataFrame, "FRAME_TYPE")
    _require(not frame.attrs and frame.flags.allows_duplicate_labels, "FRAME_FLAGS")
    index = frame.index
    _require(
        type(index) is pd.RangeIndex
        and (index.name is None or type(index.name) is str),
        "FRAME_INDEX",
    )
    columns = frame.columns
    names = list(columns)
    _require(
        type(columns) is pd.Index
        and columns.name is None
        and all(type(name) is str and exact_text(name) for name in names)
        and len(set(names)) == len(names),
        "FRAME_COLUMNS",
    )
    if isinstance(columns.dtype, np.dtype) and columns.dtype.kind == "O":
        _require(columns.dtype.metadata is None, "FRAME_DTYPE")
        labels = "object"
    else:
        labels = _string_form(columns.dtype)
        _require(labels is not None, "FRAME_COLUMNS")
    header = {
        "format": "microcosm.us.source-memo.frame.v1",
        "rows": len(frame),
        "index": [index.start, index.stop, index.step, index.name],
        "labels": labels,
        "columns": [],
    }
    blobs = []
    for position, name in enumerate(names):
        series = frame.iloc[:, position]
        dtype = series.dtype
        if isinstance(dtype, np.dtype):
            _require(dtype.metadata is None, "FRAME_DTYPE")
        if isinstance(dtype, np.dtype) and dtype.kind == "O":
            header["columns"].append([name, "object", None])
            blobs.append(ordered(_object_values(series)))
            continue
        if type(dtype) is not pd.StringDtype:
            _require(
                isinstance(dtype, np.dtype) and dtype.kind in _NUMPY_KINDS,
                "FRAME_DTYPE",
            )
            array = series.to_numpy(copy=False)
            _require(array.dtype == dtype and array.ndim == 1, "FRAME_VALUES")
            header["columns"].append([name, "numpy", dtype.str])
            blobs.append(np.ascontiguousarray(array).tobytes())
            continue
        form = _string_form(dtype)
        _require(form is not None, "FRAME_DTYPE")
        header["columns"].append([name, "string", form])
        blobs.append(ordered(_string_values(series)))
    return [ordered(header), *blobs]


def _frame_header(raw: bytes) -> dict:
    header = json.loads(raw)
    _require(
        type(header) is dict
        and list(header) == ["format", "rows", "index", "labels", "columns"]
        and header["format"] == "microcosm.us.source-memo.frame.v1"
        and type(header["rows"]) is int
        and header["rows"] >= 0
        and type(header["index"]) is list
        and len(header["index"]) == 4
        and all(type(v) is int for v in header["index"][:3])
        and (header["index"][3] is None or type(header["index"][3]) is str)
        and type(header["columns"]) is list
        and all(
            type(column) is list and len(column) == 3 and type(column[0]) is str
            for column in header["columns"]
        ),
        "FRAME_HEADER",
    )
    return header


def decode_frame(blobs: list[bytes]):
    """Rebuild exactly the frame ``encode_frame`` described, one block per column."""
    _require(type(blobs) is list and blobs, "FRAME_BLOBS")
    header = _frame_header(blobs[0])
    columns = header["columns"]
    _require(len(blobs) == 1 + len(columns), "FRAME_BLOBS")
    start, stop, step, index_name = header["index"]
    index = pd.RangeIndex(start, stop, step, name=index_name)
    rows = header["rows"]
    _require(len(index) == rows, "FRAME_ROWS")
    names = [column[0] for column in columns]
    _require(len(set(names)) == len(names), "FRAME_COLUMNS")
    data = {}
    for (name, kind, form), raw in zip(columns, blobs[1:], strict=True):
        if kind == "numpy":
            dtype = _numpy_dtype(form)
            _require(len(raw) == rows * dtype.itemsize, "FRAME_ROWS")
            array = np.frombuffer(raw, dtype=dtype).copy()
            data[name] = pd.Series(array, index=index, name=name, copy=False)
        elif kind == "object":
            _require(form is None, "FRAME_DTYPE")
            data[name] = pd.Series(
                _decode_object_values(raw, rows),
                dtype=object,
                index=index,
                name=name,
            )
        else:
            _require(kind == "string", "FRAME_DTYPE")
            values = json.loads(raw)
            _require(
                type(values) is list
                and len(values) == rows
                and all(v is None or type(v) is str for v in values),
                "FRAME_VALUES",
            )
            data[name] = pd.Series(
                pd.array(values, dtype=_string_dtype(form)),
                index=index,
                name=name,
                copy=False,
            )
    labels = header["labels"]
    label_dtype = object if labels == "object" else _string_dtype(labels)
    return pd.DataFrame(
        data, index=index, columns=pd.Index(names, dtype=label_dtype), copy=False
    )


def decode_frames(blobs: list[bytes]) -> list:
    """Consecutive ``encode_frame`` encodings, consumed exactly."""
    frames, position = [], 0
    while position < len(blobs):
        width = 1 + len(_frame_header(blobs[position])["columns"])
        _require(position + width <= len(blobs), "FRAME_BLOBS")
        frames.append(decode_frame(blobs[position : position + width]))
        position += width
    return frames


def frames_identical(left, right) -> bool:
    """Type-, dtype-, label- and bit-exact equality of two data frames.

    Beyond ``encode_frame``'s shapes this is simply False, never an error.
    """
    if type(left) is not pd.DataFrame or type(right) is not pd.DataFrame:
        return False
    if left.shape != right.shape or left.attrs or right.attrs:
        return False
    if left.flags.allows_duplicate_labels != right.flags.allows_duplicate_labels:
        return False
    a_index, b_index = left.index, right.index
    if not (
        type(a_index) is pd.RangeIndex
        and type(b_index) is pd.RangeIndex
        and (a_index.start, a_index.stop, a_index.step, a_index.name)
        == (b_index.start, b_index.stop, b_index.step, b_index.name)
    ):
        return False
    a_columns, b_columns = left.columns, right.columns
    if any(
        isinstance(columns.dtype, np.dtype) and columns.dtype.metadata is not None
        for columns in (a_columns, b_columns)
    ):
        return False
    if not (
        type(a_columns) is type(b_columns)
        and a_columns.name is None
        and b_columns.name is None
        and list(a_columns) == list(b_columns)
        and all(type(name) is str for name in a_columns)
        and all(type(name) is str for name in b_columns)
        and type(a_columns.dtype) is type(b_columns.dtype)
        and a_columns.dtype == b_columns.dtype
        and repr(a_columns.dtype) == repr(b_columns.dtype)
    ):
        return False
    for position in range(left.shape[1]):
        a, b = left.iloc[:, position], right.iloc[:, position]
        if any(
            isinstance(series.dtype, np.dtype) and series.dtype.metadata is not None
            for series in (a, b)
        ):
            return False
        if not (
            type(a.dtype) is type(b.dtype)
            and a.dtype == b.dtype
            and repr(a.dtype) == repr(b.dtype)
            and type(a.array) is type(b.array)
            and a.name == b.name
        ):
            return False
        if isinstance(a.dtype, np.dtype) and a.dtype.kind == "O":
            try:
                if ordered(_object_values(a)) != ordered(_object_values(b)):
                    return False
            except SourceMemoError:
                return False
        elif type(a.dtype) is pd.StringDtype:
            if _string_form(a.dtype) is None:
                return False
            try:
                if _string_values(a) != _string_values(b):
                    return False
            except SourceMemoError:
                return False
        else:
            if not (isinstance(a.dtype, np.dtype) and a.dtype.kind in _NUMPY_KINDS):
                return False
            x, y = a.to_numpy(copy=False), b.to_numpy(copy=False)
            if (
                x.dtype != y.dtype
                or np.ascontiguousarray(x).tobytes()
                != np.ascontiguousarray(y).tobytes()
            ):
                return False
    return True


# -- third-party parser identity ----------------------------------------------------

# pandas options that choose the string dtypes a CSV read produces.
_PANDAS_OPTIONS = ("future.infer_string", "mode.string_storage")
# Parsing, type inference, hashing (isin/duplicated) and sorting code of the
# libraries a frame derivation runs, digested once per process as loaded.
_LIBRARY_MODULES = (
    "numpy._core._multiarray_umath",
    "pandas._libs.algos",
    "pandas._libs.hashtable",
    "pandas._libs.lib",
    "pandas._libs.parsers",
    "pandas.io.parsers.base_parser",
    "pandas.io.parsers.c_parser_wrapper",
    "pandas.io.parsers.readers",
    "pyarrow.lib",
)
_LIBRARIES = None


def _parser_value(value):
    """Exact JSON identity of the defaults used by the CSV parser."""
    from enum import Enum

    immutable = _immutable(value)
    if immutable is not _SKIP:
        return immutable
    if isinstance(value, Enum):
        return [
            "enum",
            type(value).__module__,
            type(value).__qualname__,
            value.name,
            _parser_value(value.value),
        ]
    if type(value) is dict:
        _require(all(type(key) is str for key in value), "MEMO_PARSER_DEFAULT")
        return ["dict", [[key, _parser_value(value[key])] for key in sorted(value)]]
    if type(value) in (list, tuple, set, frozenset):
        items = [_parser_value(item) for item in value]
        if type(value) in (set, frozenset):
            items.sort(key=canonical)
        return [type(value).__name__, items]
    raise SourceMemoError("MEMO_PARSER_DEFAULT")


def _parser_module_identity(module, class_name):
    """Bind source-defined parser functions/methods to their live code.

    A replacement is refused even when it forges the original function's
    name. Check the defining globals as well as the code, since the same code
    executed with other globals need not parse the same bytes. Overloads share
    a qualified name; only the final definition is the runtime implementation.
    """
    payload = Path(module.__file__).read_bytes()
    codes = _compiled(module.__file__, payload)
    owner = getattr(module, class_name)
    _require(
        isinstance(owner, type) and type(owner) is type(owner.__bases__[0]),
        "MEMO_LIVE_PARSER",
    )
    functions = {}
    for name, candidates in codes.items():
        parts = name.split(".")
        if (
            "<" in name
            or parts[-1] == "__annotate__"
            or not candidates[-1].co_flags & 1  # CO_OPTIMIZED: function, not class
            or not (len(parts) == 1 or len(parts) == 2 and parts[0] == class_name)
        ):
            continue
        value = vars(module if len(parts) == 1 else owner).get(parts[-1])
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        _require(
            type(value) is FunctionType
            and value.__globals__ is vars(module)
            and value.__code__ == candidates[-1]
            and (
                not value.__closure__
                or value.__code__.co_freevars == ("__class__",)
                and len(value.__closure__) == 1
                and value.__closure__[0].cell_contents is owner
            ),
            "MEMO_LIVE_PARSER",
        )
        functions[name] = {
            "defaults": _parser_value(value.__defaults__),
            "kwdefaults": _parser_value(value.__kwdefaults__),
        }
    # An added special method (for example __getattribute__) can alter the
    # dispatch of all the checked methods without replacing any of them.
    for name, value in vars(owner).items():
        if isinstance(value, FunctionType):
            _require(f"{class_name}.{name}" in functions, "MEMO_LIVE_PARSER")
    return {"sha256": _sha(payload), "functions": functions}


def _live_parser_identity():
    """The pandas C-engine dispatch chain used by ACS archive reads.

    Provider file digests alone cannot authenticate live Python bindings.
    Verify these on every lookup, including calls made before a first memo
    entry exists, rather than trusting a lazily captured reference snapshot.
    """
    from collections.abc import Iterator

    import pandas.io.parsers as exports
    import pandas.io.parsers.base_parser as base
    import pandas.io.parsers.c_parser_wrapper as cparser
    import pandas.io.parsers.readers as readers
    from pandas._libs import parsers

    _require(
        pd.read_csv is readers.read_csv
        and exports.read_csv is readers.read_csv
        and exports.TextFileReader is readers.TextFileReader
        and readers.CParserWrapper is cparser.CParserWrapper
        and readers.ParserBase is base.ParserBase
        and cparser.ParserBase is base.ParserBase
        and cparser.parsers is parsers
        and readers.TextFileReader.__bases__ == (Iterator,)
        and cparser.CParserWrapper.__bases__ == (base.ParserBase,)
        and base.ParserBase.__bases__ == (object,),
        "MEMO_LIVE_PARSER",
    )
    # Native constructors must still belong to this exact extension type,
    # rejecting a Python substitute or subclass. Cython may expose a mutable
    # extension type, so check its read methods as well as its module binding.
    native = parsers.TextReader
    _require(
        type(native) is type
        and native.__module__ == "pandas._libs.parsers"
        and native.__name__ == "TextReader"
        and getattr(native.__new__, "__self__", None) is native
        and getattr(native.__init__, "__objclass__", None) is native,
        "MEMO_LIVE_PARSER",
    )
    for name in ("read", "read_low_memory", "close"):
        method = getattr(native, name)
        _require(
            getattr(method, "__objclass__", None) is native
            or (
                type(method).__name__ == "cython_function_or_method"
                and type(method).__module__.startswith("_cython_")
                and method.__module__ == "pandas._libs.parsers"
                and method.__qualname__ == f"TextReader.{name}"
            ),
            "MEMO_LIVE_PARSER",
        )
    return {
        "modules": {
            module.__name__: _parser_module_identity(module, class_name)
            for module, class_name in (
                (readers, "TextFileReader"),
                (cparser, "CParserWrapper"),
                (base, "ParserBase"),
            )
        },
        "defaults": {
            "readers.parser_defaults": _parser_value(readers.parser_defaults),
            "readers._c_parser_defaults": _parser_value(readers._c_parser_defaults),
            "base.parser_defaults": _parser_value(base.parser_defaults),
            "readers.STR_NA_VALUES": _parser_value(readers.STR_NA_VALUES),
        },
    }


def library_identity() -> dict:
    """numpy/pandas/pyarrow versions and file digests plus the current pandas
    string options and source-verified live CSV parser bindings/defaults.
    Mutable parser state is checked on every call, including the first one."""
    global _LIBRARIES
    with _LOCK:
        if _LIBRARIES is None:
            import pyarrow  # an absent pyarrow leaves the identity unformable

            _LIBRARIES = {
                "versions": {
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "pyarrow": pyarrow.__version__,
                },
                "files_sha256": {
                    name: file_sha256(
                        os.path.realpath(importlib.import_module(name).__file__)
                    )[0]
                    for name in _LIBRARY_MODULES
                },
            }
        static = json.loads(canonical(_LIBRARIES))
    return {
        **static,
        "pandas_options": {name: pd.get_option(name) for name in _PANDAS_OPTIONS},
        "live_csv_parser": _live_parser_identity(),
    }
