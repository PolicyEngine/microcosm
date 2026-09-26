"""Checked-in source hashing only; no diagnostic bootstrap or runtime execution."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def diagnostic():
    source = (
        Path(__file__).resolve().parents[3] / "tools/spec_seed_identity_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("seed_lock_controls", source)
    module = importlib.util.module_from_spec(spec)
    # Read the checked-in source directly; do not consume a cached bytecode file.
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    return module


def test_checked_in_lock_passes_actual_source_stamps(diagnostic):
    root = Path(__file__).resolve().parents[3]
    stamps = diagnostic.source_stamps(root)
    assert (
        stamps["uv.lock"] == hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    )
    assert set(stamps) == {
        *diagnostic.SOURCE_PATHS,
        "tools/spec_seed_identity_diagnostics.py",
        "uv.lock",
        ".github/workflows/test.yml",
    }


def test_actual_source_stamps_refuses_an_incorrect_lock_pin(diagnostic, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(diagnostic, "LOCK_SHA256", "0" * 64)
    with pytest.raises(diagnostic.RefusalError, match="^LOCK$"):
        diagnostic.source_stamps(root)
