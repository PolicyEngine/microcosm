"""Pinned engine parameter files and the urllib3 probe: invented hooks only.

The derive phase imports policyengine-us, which reads eight public parameter
CSVs at module import. The boundary may open exactly those files, only when
their bytes hash to the pins verified before the hook is armed; every other
data-suffixed open still refuses. urllib3's import-time IPv6 probe creates one
socket and binds it to loopback from ``_has_ipv6``; nothing else on a socket is
accepted.
"""

import base64
import hashlib
import importlib.util
import os
import types
from importlib import metadata
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def diagnostic():
    source = (
        Path(__file__).resolve().parents[3] / "tools/spec_seed_identity_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("engine_parameter_controls", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _boundary(diagnostic, monkeypatch, allowed):
    hooks, first, distinct, contexts = [], [], [], {}
    monkeypatch.setattr(diagnostic.sys, "addaudithook", hooks.append)
    monkeypatch.setattr(Path, "resolve", lambda path: path)
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: Path("/invented/repo")))
    diagnostic.install_boundary(
        Path("/invented/repo"),
        Path("/invented/owned"),
        Path("/invented/output"),
        first_refusal=first,
        distinct_refusals=distinct,
        code_contexts=contexts,
        allowed_data_files=allowed,
    )
    assert len(hooks) == 1
    return hooks[0], distinct


@pytest.mark.requires_us
def test_pins_match_the_installed_distribution_record(diagnostic):
    """Every pin is a regular file in the distribution whose RECORD digest agrees."""

    dist = metadata.distribution(diagnostic.ENGINE_PUBLIC_PARAMETER_DISTRIBUTION)
    root = Path(dist.locate_file("")).resolve()
    record = {}
    for entry in dist.files or ():
        if entry.hash is not None and entry.hash.mode == "sha256":
            padded = entry.hash.value + "=" * (-len(entry.hash.value) % 4)
            record[str(entry)] = base64.urlsafe_b64decode(padded).hex()
    assert len(diagnostic.ENGINE_PUBLIC_PARAMETER_FILES) == 8
    for relative, expected, size in diagnostic.ENGINE_PUBLIC_PARAMETER_FILES:
        path = root / relative
        assert path.is_file() and not path.is_symlink(), relative
        raw = path.read_bytes()
        assert len(raw) == size and hashlib.sha256(raw).hexdigest() == expected
        assert record[relative] == expected
    report, allowed = diagnostic.pinned_engine_parameter_files()
    assert report["policy"] == "pinned_engine_public_parameter_files_v1"
    assert report["count"] == 8 and report["version"] == dist.version
    assert allowed == frozenset(
        (root / relative).resolve()
        for relative, _digest, _size in diagnostic.ENGINE_PUBLIC_PARAMETER_FILES
    )


def test_a_pin_whose_bytes_differ_refuses_before_arming(diagnostic, tmp_path):
    """Verification fails closed on bytes, size, RECORD, or a missing file."""

    relative = "invented_engine/parameters/table.csv"
    root = tmp_path / "site-packages"
    (root / "invented_engine/parameters").mkdir(parents=True)
    raw = b"invented,public,parameter\n1,2,3\n"
    (root / relative).write_bytes(raw)
    good = hashlib.sha256(raw).hexdigest()
    encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=")

    class _Hash:
        mode = "sha256"

        def __init__(self, value: str) -> None:
            self.value = value

    class _Entry(str):
        hash = _Hash(encoded.decode())

    class _Distribution:
        version = "0.0.0-invented"
        files = [_Entry(relative)]

        def locate_file(self, _name):
            return root

    def distribution(name):
        assert name == "invented-engine"
        return _Distribution()

    real = metadata.distribution
    try:
        metadata.distribution = distribution
        report, allowed = diagnostic.pinned_engine_parameter_files(
            distribution="invented-engine", pins=((relative, good, len(raw)),)
        )
        assert report["count"] == 1 and allowed == {(root / relative).resolve()}
        for pins in (
            ((relative, "0" * 64, len(raw)),),
            ((relative, good, len(raw) + 1),),
            (("invented_engine/parameters/absent.csv", good, len(raw)),),
            (("../outside.csv", good, len(raw)),),
        ):
            with pytest.raises(diagnostic.RefusalError, match="^ENGINE_PARAMETER_PIN$"):
                diagnostic.pinned_engine_parameter_files(
                    distribution="invented-engine", pins=pins
                )
        _Entry.hash = _Hash("AAAA")  # RECORD disagrees with the file bytes.
        with pytest.raises(diagnostic.RefusalError, match="^ENGINE_PARAMETER_PIN$"):
            diagnostic.pinned_engine_parameter_files(
                distribution="invented-engine", pins=((relative, good, len(raw)),)
            )
    finally:
        metadata.distribution = real


def test_hook_opens_only_the_allowed_data_files(diagnostic, monkeypatch):
    allowed = frozenset({Path("/invented/site-packages/engine/parameters/pinned.csv")})
    hook, distinct = _boundary(diagnostic, monkeypatch, allowed)
    hook("open", (str(next(iter(allowed))), "r", os.O_RDONLY))
    assert distinct == []
    for path in (
        "/invented/site-packages/engine/parameters/other.csv",
        "/invented/site-packages/engine/parameters/pinned.csv.bak.csv",
        "/invented/repo/pinned.csv",
    ):
        with pytest.raises(diagnostic.RefusalError, match="^DATA_FILE$"):
            hook("open", (path, "r", os.O_RDONLY))
    assert distinct == ["DATA_FILE"]


def test_proc_stat_is_readable_metadata(diagnostic, monkeypatch):
    hook, distinct = _boundary(diagnostic, monkeypatch, frozenset())
    hook("open", ("/proc/stat", "r", os.O_RDONLY))
    assert distinct == []
    with pytest.raises(diagnostic.RefusalError, match="^READ_SCOPE$"):
        hook("open", ("/proc/version", "r", os.O_RDONLY))


def _call_as_has_ipv6(diagnostic, event, args):
    """Invoke the hook from a frame named like urllib3's probe function."""

    code = compile(
        "def _has_ipv6(hook, event, args):\n    return hook(event, args)\n",
        "/invented/site-packages/urllib3/util/connection.py",
        "exec",
    )
    namespace: dict[str, object] = {}
    exec(code, namespace)
    return namespace["_has_ipv6"](diagnostic, event, args)


def test_urllib3_ipv6_probe_is_the_only_accepted_socket_use(diagnostic, monkeypatch):
    hook, distinct = _boundary(diagnostic, monkeypatch, frozenset())
    sock = types.SimpleNamespace()
    _call_as_has_ipv6(hook, "socket.__new__", (sock, 10, 1, 0))
    _call_as_has_ipv6(hook, "socket.bind", (sock, ("::1", 0)))
    _call_as_has_ipv6(hook, "socket.bind", (sock, ("127.0.0.1", 0)))
    assert distinct == []
    with pytest.raises(diagnostic.RefusalError, match="^NETWORK_OR_CHILD$"):
        _call_as_has_ipv6(hook, "socket.bind", (sock, ("0.0.0.0", 0)))
    with pytest.raises(diagnostic.RefusalError, match="^NETWORK_OR_CHILD$"):
        _call_as_has_ipv6(hook, "socket.connect", (sock, ("::1", 80)))
    with pytest.raises(diagnostic.RefusalError, match="^NETWORK_OR_CHILD$"):
        hook("socket.__new__", (sock, 10, 1, 0))
    with pytest.raises(diagnostic.RefusalError, match="^NETWORK_OR_CHILD$"):
        hook("socket.bind", (sock, ("::1", 0)))
    assert distinct == ["NETWORK_OR_CHILD"]
