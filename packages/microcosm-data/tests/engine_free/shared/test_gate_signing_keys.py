"""The committed UK gate signing key registry binds verification to named keys."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess

import pytest
from hypothesis import given
from hypothesis import strategies as st

from microcosm.data import contract
from microcosm.data import gate_signing_keys as registry
from microcosm.data.gate_signing_keys import (
    GATE_SIGNING_KEY_STATUSES,
    UK_GATE_SIGNING_KEY_ENV,
    UK_GATE_SIGNING_KEYS,
    GateSigningKey,
    GateSigningKeyError,
    decode_gate_signing_key,
    gate_signing_key_fingerprint,
    require_registered_gate_signing_key,
    unregistered_gate_signing_key_problem,
)

#: The key that signed every UK cut on record (T19, T21, the 10/19 dense cut).
CONTINUITY_FINGERPRINT = (
    "66bf0e29dbdabc2e3d7a79f21f43ac28d17b592faaf91e67c9bcf9b30c09a6ff"
)
UNREGISTERED_KEY = b"\x11" * 32


def _entry(key: bytes, status: str) -> GateSigningKey:
    return GateSigningKey(
        key_sha256=gate_signing_key_fingerprint(key),
        label=f"test-{status}",
        status=status,  # type: ignore[arg-type]
        custodian="test",
        held_in="test",
        registered="2026-10-22",
        evidence="test",
        retired_on="2026-10-22" if status != "active" else None,
    )


def _use_registry(monkeypatch, *entries: GateSigningKey) -> None:
    table = {entry.key_sha256: entry for entry in entries}
    monkeypatch.setattr(
        registry, "_registry", lambda country: table if country == "uk" else None
    )


@pytest.mark.committed_gate_key_registry
def test_the_committed_registry_is_well_formed():
    assert UK_GATE_SIGNING_KEYS, "the UK registry must name at least one key"
    labels = set()
    for fingerprint, entry in UK_GATE_SIGNING_KEYS.items():
        assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        assert entry.key_sha256 == fingerprint
        assert entry.status in GATE_SIGNING_KEY_STATUSES
        assert (entry.retired_on is None) == (entry.status == "active")
        assert entry.label not in labels
        labels.add(entry.label)
        for field in ("custodian", "held_in", "registered", "evidence"):
            assert getattr(entry, field).strip()
    assert any(entry.status == "active" for entry in UK_GATE_SIGNING_KEYS.values())


@pytest.mark.committed_gate_key_registry
def test_the_key_that_signed_every_cut_on_record_still_verifies():
    """Continuity: re-registering must never strand the published cuts."""

    entry = UK_GATE_SIGNING_KEYS[CONTINUITY_FINGERPRINT]
    assert entry.status in {"active", "retired"}


@pytest.mark.committed_gate_key_registry
def test_an_unregistered_key_neither_signs_nor_verifies():
    for purpose in ("sign", "verify"):
        problem = unregistered_gate_signing_key_problem(
            "uk", UNREGISTERED_KEY, purpose=purpose
        )
        assert problem is not None
        assert gate_signing_key_fingerprint(UNREGISTERED_KEY) in problem
        with pytest.raises(GateSigningKeyError, match="not registered"):
            require_registered_gate_signing_key("uk", UNREGISTERED_KEY, purpose=purpose)


@pytest.mark.committed_gate_key_registry
def test_countries_without_a_registry_keep_the_old_behaviour():
    assert (
        unregistered_gate_signing_key_problem("us", UNREGISTERED_KEY, purpose="sign")
        is None
    )
    assert registry.registered_gate_signing_key("us", "0" * 64) is None


@given(
    key=st.binary(min_size=32, max_size=32),
    status=st.sampled_from(sorted(GATE_SIGNING_KEY_STATUSES)),
)
def test_status_decides_purpose(key, status):
    """For every key and status: active signs and verifies, retired only
    verifies, revoked does neither, and signing implies verifying."""

    entry = _entry(key, status)
    with pytest.MonkeyPatch.context() as patch:
        _use_registry(patch, entry)
        may_sign = (
            unregistered_gate_signing_key_problem("uk", key, purpose="sign") is None
        )
        may_verify = (
            unregistered_gate_signing_key_problem("uk", key, purpose="verify") is None
        )
    assert may_sign == (status == "active")
    assert may_verify == (status in {"active", "retired"})
    assert not may_sign or may_verify


@given(
    key=st.binary(min_size=32, max_size=32),
    other=st.binary(min_size=32, max_size=32),
)
def test_registration_is_by_exact_fingerprint(key, other):
    """A registered key admits only itself: no other key verifies under it."""

    with pytest.MonkeyPatch.context() as patch:
        _use_registry(patch, _entry(key, "active"))
        other_problem = unregistered_gate_signing_key_problem(
            "uk", other, purpose="verify"
        )
    assert (other_problem is None) == (other == key)


@given(key=st.binary(min_size=32, max_size=32))
def test_the_fingerprint_is_the_attested_sha256(key):
    assert gate_signing_key_fingerprint(key) == hashlib.sha256(key).hexdigest()
    encoded = base64.b64encode(key).decode()
    assert decode_gate_signing_key(encoded, env="E") == key


@given(raw=st.binary(max_size=64).filter(lambda value: len(value) != 32))
def test_a_key_of_any_other_length_is_refused(raw):
    with pytest.raises(GateSigningKeyError, match="exactly 32 bytes|must contain"):
        decode_gate_signing_key(base64.b64encode(raw).decode(), env="E")


def test_malformed_base64_is_refused():
    with pytest.raises(GateSigningKeyError, match="valid base64"):
        decode_gate_signing_key("not base64!", env="E")


# ---------------------------------------------------------------------------
# The publisher's verification key
# ---------------------------------------------------------------------------


def _set_key(monkeypatch, key: bytes) -> None:
    monkeypatch.setenv(UK_GATE_SIGNING_KEY_ENV, base64.b64encode(key).decode())


@pytest.mark.parametrize(
    ("status", "verifies"), [("active", True), ("retired", True), ("revoked", False)]
)
def test_the_publisher_verifies_only_registered_keys(monkeypatch, status, verifies):
    key = b"\x22" * 32
    _set_key(monkeypatch, key)
    _use_registry(monkeypatch, _entry(key, status))
    failures: list[str] = []
    assert (contract._uk_gate_battery_verification_key(failures) == key) is verifies
    assert (failures == []) is verifies
    if not verifies:
        assert "revoked" in failures[0]


def test_the_legacy_schema_3_trust_root_is_bound_too(monkeypatch):
    legacy_env = contract._UK_TERMINAL_GATE_SIGNING_KEY_ENV
    monkeypatch.setenv(legacy_env, base64.b64encode(UNREGISTERED_KEY).decode())
    _use_registry(monkeypatch, _entry(b"\x22" * 32, "active"))
    failures: list[str] = []
    assert contract._uk_terminal_verification_key(failures) is None
    assert len(failures) == 1 and "not registered" in failures[0]
    _use_registry(monkeypatch, _entry(UNREGISTERED_KEY, "retired"))
    failures = []
    assert contract._uk_terminal_verification_key(failures) == UNREGISTERED_KEY
    assert failures == []


def test_the_publisher_refuses_an_unregistered_key(monkeypatch):
    _set_key(monkeypatch, UNREGISTERED_KEY)
    _use_registry(monkeypatch, _entry(b"\x22" * 32, "active"))
    failures: list[str] = []
    assert contract._uk_gate_battery_verification_key(failures) is None
    assert len(failures) == 1 and "not registered" in failures[0]


# ---------------------------------------------------------------------------
# The operator CLI
# ---------------------------------------------------------------------------


def test_check_reports_the_fingerprint_and_status(monkeypatch, capsys):
    key = b"\x33" * 32
    _set_key(monkeypatch, key)
    _use_registry(monkeypatch, _entry(key, "retired"))
    assert registry.main(["check", "--purpose", "verify"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["key_sha256"] == gate_signing_key_fingerprint(key)
    assert report["registered"]["status"] == "retired"
    assert registry.main(["check"]) == 1  # a retired key may not sign
    assert "retired" in json.loads(capsys.readouterr().out)["problem"]


def test_check_fails_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv(UK_GATE_SIGNING_KEY_ENV, raising=False)
    assert registry.main(["check"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_generate_pipes_a_fresh_key_and_its_fingerprint(capsys):
    assert registry.main(["generate"]) == 0
    captured = capsys.readouterr()
    key = decode_gate_signing_key(captured.out.strip(), env="E")
    assert captured.err.strip() == f"key_sha256 {gate_signing_key_fingerprint(key)}"
    assert registry.main(["generate"]) == 0
    assert capsys.readouterr().out.strip() != captured.out.strip()


def test_generate_never_prints_a_key_to_a_terminal(capsys, monkeypatch):
    monkeypatch.setattr(registry.sys.stdout, "isatty", lambda: True)
    assert registry.main(["generate"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--keychain-service" in captured.err


def test_generate_stores_the_key_without_it_in_any_argv(capsys, monkeypatch):
    calls = []
    store = {}

    def run(argv, **kwargs):
        calls.append((argv, kwargs.get("input")))
        if argv[:2] == ["security", "-i"]:
            store["value"] = kwargs["input"].split(" -w ", 1)[1].strip()
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, store["value"] + "\n", "")

    monkeypatch.setattr(registry.subprocess, "run", run)
    monkeypatch.setattr(registry.getpass, "getuser", lambda: "builder")
    monkeypatch.setattr(registry.sys.stdout, "isatty", lambda: True)
    assert registry.main(["generate", "--keychain-service", "uk-gate-key"]) == 0
    printed = capsys.readouterr().out.strip()
    key = base64.b64decode(store["value"])
    assert printed == f"key_sha256 {gate_signing_key_fingerprint(key)}"
    assert store["value"] not in printed
    for argv, _ in calls:
        assert store["value"] not in " ".join(argv)
    assert calls[0][1].startswith("add-generic-password -U -a builder -s uk-gate-key ")


def test_a_key_that_does_not_read_back_is_reported(capsys, monkeypatch):
    monkeypatch.setattr(
        registry.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "other\n", ""),
    )
    monkeypatch.setattr(registry.getpass, "getuser", lambda: "builder")
    assert registry.main(["generate", "--keychain-service", "uk-gate-key"]) == 1
    assert "did not read back" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["two words", "quo'te", ""])
def test_keychain_names_must_be_plain(name):
    with pytest.raises(GateSigningKeyError, match="must be plain"):
        registry.store_in_keychain(b"k" * 32, service=name, account="builder")


@pytest.mark.committed_gate_key_registry
def test_list_prints_the_committed_entries(capsys):
    assert registry.main(["list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {row["key_sha256"] for row in listed} == set(UK_GATE_SIGNING_KEYS)
