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


def _function(value, module, codes, seen):
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
    def _index_path(self, key):
        return self.root / "index" / key[:2] / f"{key}.json"

    def _blob_path(self, digest):
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


def decode_json(blobs):
    _require(len(blobs) == 1, "BLOB_COUNT")
    return json.loads(blobs[0])


def prove_json(value, blobs):
    """``encode_json`` admits only JSON-native values, which ``ordered`` round-trips
    exactly; decoding again would only repeat that guarantee at the cost of a
    second copy of the value."""
    return len(blobs) == 1 and json_native(value)
