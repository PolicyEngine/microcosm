"""Every UK signer and verifier consults the committed signing key registry."""

from __future__ import annotations

import base64
import json

import pytest

from microcosm.build import gate_battery
from microcosm.build.uk_runtime import (
    calibration_run,
    national_role,
    release_certification,
)
from microcosm.data import contract
from microcosm.data import gate_signing_keys as registry
from microcosm.data.gate_signing_keys import (
    UK_GATE_SIGNING_KEY_ENV,
    GateSigningKey,
    GateSigningKeyError,
    gate_signing_key_fingerprint,
)

ACTIVE = b"\x41" * 32
RETIRED = b"\x52" * 32
REVOKED = b"\x56" * 32
UNREGISTERED = b"\x55" * 32


def _entry(key: bytes, status: str) -> GateSigningKey:
    return GateSigningKey(
        key_sha256=gate_signing_key_fingerprint(key),
        label=f"test-{status}",
        status=status,  # type: ignore[arg-type]
        custodian="test",
        held_in="test",
        registered="2026-10-22",
        evidence="test",
        retired_on=None if status == "active" else "2026-10-22",
    )


@pytest.fixture
def strict_registry(monkeypatch):
    """A registry with one key of each status, and nothing else admitted."""

    table = {
        entry.key_sha256: entry
        for entry in (
            _entry(ACTIVE, "active"),
            _entry(RETIRED, "retired"),
            _entry(REVOKED, "revoked"),
        )
    }
    monkeypatch.setattr(
        registry, "_registry", lambda country: table if country == "uk" else None
    )
    return table


def _use_key(monkeypatch, key: bytes) -> None:
    monkeypatch.setenv(UK_GATE_SIGNING_KEY_ENV, base64.b64encode(key).decode())


def test_every_layer_reads_one_variable():
    assert gate_battery.gate_signing_key_env("uk") == UK_GATE_SIGNING_KEY_ENV
    assert contract._UK_GATE_BATTERY_SIGNING_KEY_ENV == UK_GATE_SIGNING_KEY_ENV


def test_a_registry_refusal_is_a_signing_error_not_a_crash():
    """The battery turns a RuntimeError into an unsigned, unshippable report;
    the registry's refusal must be one."""

    assert issubclass(GateSigningKeyError, RuntimeError)


@pytest.mark.parametrize(
    ("key", "signs"),
    [(ACTIVE, True), (RETIRED, False), (REVOKED, False), (UNREGISTERED, False)],
)
def test_the_battery_signs_only_with_an_active_key(
    strict_registry, monkeypatch, key, signs
):
    _use_key(monkeypatch, key)
    if signs:
        assert gate_battery._signing_key("uk") == key
    else:
        with pytest.raises(GateSigningKeyError):
            gate_battery._signing_key("uk")


@pytest.mark.parametrize(
    ("key", "certifies"),
    [(ACTIVE, True), (RETIRED, False), (REVOKED, False), (UNREGISTERED, False)],
)
def test_the_certifier_needs_an_active_key(
    strict_registry, monkeypatch, key, certifies
):
    _use_key(monkeypatch, key)
    if certifies:
        assert release_certification._require_signing_key() == key
    else:
        with pytest.raises(release_certification.UKReleaseCertificationError):
            release_certification._require_signing_key()


def test_the_national_role_refuses_an_unusable_key_before_loading_inputs(
    strict_registry, monkeypatch
):
    _use_key(monkeypatch, ACTIVE)
    national_role.require_signing_key()
    for key in (RETIRED, UNREGISTERED):
        _use_key(monkeypatch, key)
        with pytest.raises(ValueError, match="registered UK gate signing key"):
            national_role.require_signing_key()
    monkeypatch.delenv(UK_GATE_SIGNING_KEY_ENV)
    with pytest.raises(ValueError, match=UK_GATE_SIGNING_KEY_ENV):
        national_role.require_signing_key()


def test_resigning_a_scoped_report_needs_an_active_key(strict_registry, monkeypatch):
    payload = {"attestation": {"signing_error": "unsigned at battery time"}}
    _use_key(monkeypatch, UNREGISTERED)
    with pytest.raises(GateSigningKeyError, match="not registered"):
        calibration_run.resign_uk_gate_report(payload)
    assert payload["attestation"] == {"signing_error": "unsigned at battery time"}


def test_the_local_preflight_names_an_unregistered_key(
    strict_registry, tmp_path, monkeypatch
):
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[5]
        / "tools/preflight_uk_local_release_candidate.py"
    )
    spec = importlib.util.spec_from_file_location("preflight_uk_local", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failures = module.check_env(
        pins_path=tmp_path / "missing-pins",
        spine_h5=tmp_path / "spine.h5",
        ladder_npz=tmp_path / "ladder.npz",
        ledger_dir=tmp_path,
        environ={module.SIGNING_KEY_ENV: base64.b64encode(UNREGISTERED).decode()},
        verify_digests=False,
    )
    assert any("not registered" in failure for failure in failures)
    failures = module.check_env(
        pins_path=tmp_path / "missing-pins",
        spine_h5=tmp_path / "spine.h5",
        ladder_npz=tmp_path / "ladder.npz",
        ledger_dir=tmp_path,
        environ={module.SIGNING_KEY_ENV: base64.b64encode(ACTIVE).decode()},
        verify_digests=False,
    )
    assert not any(module.SIGNING_KEY_ENV in failure for failure in failures)


def test_a_signed_report_verifies_until_its_key_is_revoked(
    strict_registry, tmp_path, monkeypatch
):
    """Round trip: the battery signs under the active key and the publisher's
    dense check accepts it; retiring keeps it verifiable; revoking does not."""

    from microcosm.build.uk_runtime.graph_national import replay_uk_dense_gate_battery
    from test_support.microcosm_build.uk_full_build_cli import STEM, gate_payload

    _use_key(monkeypatch, ACTIVE)
    report = replay_uk_dense_gate_battery(
        json.loads(gate_payload("terminal", release_candidate=True)),
        report_path=tmp_path / f"{STEM}.local_gates.json",
        release_id="uk-local-candidate-f100-s0-test",
        release_candidate=True,
    )
    assert report["shippable"] is True
    assert report["attestation"]["signing_key_sha256"] == (
        gate_signing_key_fingerprint(ACTIVE)
    )

    def failures() -> list[str]:
        found: list[str] = []
        contract._check_uk_dense_gate_report(
            report, failures=found, attempt_id="uk-local-candidate-f100-s0-test"
        )
        return found

    assert failures() == []
    for status, verifies in (("retired", True), ("revoked", False)):
        strict_registry[gate_signing_key_fingerprint(ACTIVE)] = _entry(ACTIVE, status)
        assert (failures() == []) is verifies
    # An unregistered key that signed is refused even though the HMAC is valid.
    strict_registry.pop(gate_signing_key_fingerprint(ACTIVE))
    assert any("not registered" in line for line in failures())


def test_an_unregistered_signer_writes_an_unshippable_report(
    strict_registry, tmp_path, monkeypatch
):
    from microcosm.build.uk_runtime.graph_national import replay_uk_dense_gate_battery
    from test_support.microcosm_build.uk_full_build_cli import STEM, gate_payload

    _use_key(monkeypatch, UNREGISTERED)
    report = replay_uk_dense_gate_battery(
        json.loads(gate_payload("terminal", release_candidate=True)),
        report_path=tmp_path / f"{STEM}.local_gates.json",
        release_id="uk-local-candidate-f100-s0-test",
        release_candidate=True,
    )
    assert report["shippable"] is False
    assert report["attestation"]["signature"] is None
    assert "not registered" in report["attestation"]["signing_error"]
