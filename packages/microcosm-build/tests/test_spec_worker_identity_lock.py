"""Actual lock validation without worker, engine or identity-probe execution."""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def worker_identity(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    source = root / (
        "packages/microcosm-build/src/microcosm/build/us_runtime/worker_identity.py"
    )
    spec = importlib.util.spec_from_file_location("worker_lock_controls", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    # Locate the real checked-in lock without importing the PUF worker.
    monkeypatch.setattr(module, "_repository_root", lambda: root)
    return module


def test_checked_in_lock_passes_actual_worker_validation(worker_identity):
    root = Path(__file__).resolve().parents[3]
    assert (
        worker_identity._approved_uv_lock_sha256()
        == hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    )


def test_actual_worker_validation_refuses_an_incorrect_pin(
    worker_identity, monkeypatch
):
    monkeypatch.setattr(worker_identity, "APPROVED_UV_LOCK_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="unapproved uv.lock digest"):
        worker_identity._approved_uv_lock_sha256()


@pytest.mark.parametrize(
    "digest",
    [
        # The superseded current lock must not become an implicit legacy alias.
        "751d5ef5d25406bbae1798667f0e29890d4aad933d323c12912c7d45d8809bb9",
        "0" * 64,
    ],
)
def test_unapproved_explicit_lock_refuses_before_worker_probe(
    worker_identity, monkeypatch, digest
):
    def unexpected_probe():
        raise AssertionError("An unapproved lock reached the worker source probe")

    monkeypatch.setattr(worker_identity, "_worker_source_identity", unexpected_probe)
    with pytest.raises(ValueError, match="worker lock is not approved"):
        worker_identity._uncached_primary_qrf_worker_semantic_identity(
            uv_lock_sha256=digest
        )


def test_wheel_without_checkout_uses_approved_lock(worker_identity, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    expected = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    monkeypatch.setattr(worker_identity, "_repository_root", lambda: None)
    assert worker_identity._approved_uv_lock_sha256() == expected


def test_explicit_legacy_campaign_remains_accepted_before_probe(
    worker_identity, monkeypatch
):
    class StopBeforeProbeError(Exception):
        pass

    def stop_before_probe():
        raise StopBeforeProbeError

    monkeypatch.setattr(worker_identity, "_worker_source_identity", stop_before_probe)
    with pytest.raises(StopBeforeProbeError):
        worker_identity._uncached_primary_qrf_worker_semantic_identity(
            uv_lock_sha256=worker_identity.LEGACY_CAMPAIGN_UV_LOCK_SHA256
        )
