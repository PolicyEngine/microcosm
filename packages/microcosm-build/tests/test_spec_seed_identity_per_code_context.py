"""Invented audit calls only; no real hook, source read, child or derivation."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def diagnostic():
    source = (
        Path(__file__).resolve().parents[3] / "tools/spec_seed_identity_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("per_code_context_controls", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def audited(diagnostic, monkeypatch):
    hooks, first, distinct, contexts = [], [], [], {}
    monkeypatch.setattr(diagnostic.sys, "addaudithook", hooks.append)
    monkeypatch.setattr(Path, "resolve", lambda path: path)
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: Path("/invented/repo")))
    refusals = diagnostic.install_boundary(
        Path("/invented/repo"),
        Path("/invented/owned"),
        Path("/invented/output"),
        first_refusal=first,
        distinct_refusals=distinct,
        code_contexts=contexts,
    )
    assert len(hooks) == 1
    return hooks[0], refusals, first, distinct, contexts


def test_caught_middle_child_denial_keeps_own_context(diagnostic, audited):
    hook, refusals, first, distinct, contexts = audited
    terminal = None
    secret = "invented-secret-never-retain"
    for event, args, code in (
        ("open", ("/proc/version", "r", os.O_RDONLY), "READ_SCOPE"),
        (
            "subprocess.Popen",
            (secret, [secret], secret, {secret: secret}),
            "NETWORK_OR_CHILD",
        ),
        ("open", ("/invented/repo/never-open.csv", "r", os.O_RDONLY), "DATA_FILE"),
    ):
        with pytest.raises(diagnostic.RefusalError, match="^" + code + "$") as error:
            hook(event, args)
        terminal = error.value
    # The historical refusal counter remains first-code-only. Every operation
    # still raises; retaining context must not turn a swallowed denial into success.
    assert refusals == ["READ_SCOPE"]
    assert distinct == ["READ_SCOPE", "NETWORK_OR_CHILD", "DATA_FILE"]
    assert list(contexts) == distinct
    assert first == [contexts["READ_SCOPE"]]
    assert terminal.boundary_context == contexts["DATA_FILE"]
    child = json.loads(contexts["NETWORK_OR_CHILD"])
    assert child["code"] == "NETWORK_OR_CHILD" and child["event"] == "subprocess.Popen"
    assert child["frames"] and len(child["frames"]) <= 8
    assert child["requested"]["path"] is None and child["resolved"]["path"] is None
    for payload in contexts.values():
        assert type(payload) is bytes and len(payload) <= 4096
        assert secret.encode() not in payload and b"never-open.csv" not in payload


def test_repeated_code_retains_first_record_but_exception_has_fresh_context(
    diagnostic, audited
):
    hook, refusals, first, distinct, contexts = audited

    def original_caller():
        hook("subprocess.Popen", ("invented", [], None, None))

    def later_caller():
        hook("subprocess.Popen", ("invented", [], None, None))

    with pytest.raises(diagnostic.RefusalError) as original:
        original_caller()
    retained = contexts["NETWORK_OR_CHILD"]
    with pytest.raises(diagnostic.RefusalError) as later:
        later_caller()
    assert contexts == {"NETWORK_OR_CHILD": retained}
    assert first == [retained] and refusals == distinct == ["NETWORK_OR_CHILD"]
    assert original.value.boundary_context == retained
    assert later.value.boundary_context != retained
    assert "original_caller" in [f["function"] for f in json.loads(retained)["frames"]]
    assert "later_caller" in [
        f["function"] for f in json.loads(later.value.boundary_context)["frames"]
    ]


def test_eight_record_capacity_never_discards_or_grows(diagnostic, audited):
    hook, refusals, first, distinct, contexts = audited
    # A full invented collector exercises the bound even though the current
    # boundary emits only seven different fixed codes.
    codes = (
        "WRITE_SCOPE",
        "DATA_FILE",
        "FILESYSTEM_PATH",
        "FILESYSTEM_DESCRIPTOR",
        "FILESYSTEM_LINK",
        "NETWORK_OR_CHILD",
        "WALL",
        "OUTPUT_CHANGED",
    )
    contexts.update({code: diagnostic.encoded({"code": code}) for code in codes})
    retained = tuple(contexts.items())
    with pytest.raises(diagnostic.RefusalError, match="^READ_SCOPE$") as error:
        hook("open", ("/proc/version", "r", os.O_RDONLY))
    assert tuple(contexts.items()) == retained and len(contexts) == 8
    assert refusals == distinct == ["READ_SCOPE"]
    assert first == [error.value.boundary_context]


@pytest.mark.parametrize(
    ("event", "label"),
    (
        ("os.posix_spawn", "os.posix_spawn"),
        ("socket.connect", "network_or_child_event"),
        ("subprocess.invented-secret-event", "network_or_child_event"),
    ),
)
def test_network_context_event_is_fixed_and_never_serializes_arguments(
    diagnostic, audited, event, label
):
    hook, _, _, _, contexts = audited
    secret = "invented-secret-arguments"
    with pytest.raises(diagnostic.RefusalError, match="^NETWORK_OR_CHILD$"):
        hook(event, (secret, {secret: secret}))
    payload = contexts["NETWORK_OR_CHILD"]
    assert json.loads(payload)["event"] == label
    assert secret.encode() not in payload and b"invented-secret-event" not in payload


def test_context_capture_reads_no_source_files_or_locals(
    diagnostic, audited, monkeypatch
):
    hook, _, _, _, contexts = audited
    private_local = "invented-private-local-not-evidence"

    def forbidden(*args, **kwargs):
        pytest.fail("context formatting attempted a source or filesystem read")

    with monkeypatch.context() as patch:
        for name in ("open", "read_bytes", "read_text", "stat", "lstat"):
            patch.setattr(Path, name, forbidden)
        with pytest.raises(diagnostic.RefusalError, match="^NETWORK_OR_CHILD$"):
            hook("subprocess.Popen", (private_local,))
    payload = contexts["NETWORK_OR_CHILD"]
    assert private_local.encode() not in payload
    assert all(
        set(frame) == {"file", "function", "line"}
        for frame in json.loads(payload)["frames"]
    )


def test_context_capture_failure_does_not_replace_denial_with_exception_text(
    diagnostic, audited, monkeypatch
):
    hook, refusals, first, _, contexts = audited

    def unavailable(*args):
        raise RuntimeError("invented-sensitive-exception-text")

    monkeypatch.setattr(diagnostic, "read_refusal_context", unavailable)
    with pytest.raises(diagnostic.RefusalError, match="^NETWORK_OR_CHILD$") as error:
        hook("subprocess.Popen", ("invented",))
    expected = diagnostic.encoded(
        {"code": "NETWORK_OR_CHILD", "context": "unavailable"}
    )
    assert refusals == ["NETWORK_OR_CHILD"]
    assert first == [expected] and contexts == {"NETWORK_OR_CHILD": expected}
    assert error.value.boundary_context == expected


def test_all_maximum_records_fit_status_cap_and_preserve_aggregate_cap(diagnostic):
    base = diagnostic.encoded({"code": "READ_SCOPE", "context": ""})
    maximum = diagnostic.encoded(
        {"code": "READ_SCOPE", "context": "x" * (4096 - len(base))}
    )
    assert len(maximum) == 4096
    status = diagnostic.encoded(
        {
            "completed": False,
            "coverage_pass": False,
            "first_boundary_refusal": json.loads(maximum),
            "terminal_boundary_refusal": json.loads(maximum),
            "boundary_refusal_contexts": [json.loads(maximum)] * 8,
            "boundary_refusal_context_limit": {"codes": 8, "bytes_per_code": 4096},
        }
    )
    assert 16 * 1024 < len(status) < 64 * 1024
    assert diagnostic.CAPS == {
        "candidate-digests.json": 64 * 1024,
        "seed-protocol.json": 256 * 1024,
        "seed-map.json": 256 * 1024,
        "seed-bindings.json": 128 * 1024,
        "environment-and-source.json": 128 * 1024,
        "diagnostic-status.json": 64 * 1024,
    }
    assert diagnostic.MAX_TOTAL == 1024 * 1024
    payloads = {name: b"{}" for name in diagnostic.CAPS}
    payloads["diagnostic-status.json"] = status
    diagnostic.validate_payloads(payloads)


@pytest.mark.parametrize(
    "name",
    (
        "candidate-digests.json",
        "seed-protocol.json",
        "seed-map.json",
        "seed-bindings.json",
        "environment-and-source.json",
        "diagnostic-status.json",
    ),
)
def test_each_existing_output_cap_still_refuses_before_publication(diagnostic, name):
    payloads = {key: b"{}" for key in diagnostic.CAPS}
    payloads[name] = b"x" * (diagnostic.CAPS[name] + 1)
    with pytest.raises(diagnostic.RefusalError, match="^OUTPUT_CAP$"):
        diagnostic.validate_payloads(payloads)
