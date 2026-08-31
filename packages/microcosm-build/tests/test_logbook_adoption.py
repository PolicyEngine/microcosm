"""Contracts for UK/US Logbook adoption helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from microcosm.build.logbook import (
    LOGBOOK_ROW_FIELDS,
    canonical_json_bytes,
    load_logbook_row,
)
from microcosm.build.logbook_adoption import (
    AttemptState,
    append_phase,
    apply_error_verdict,
    atomic_write_json,
    error_receipt_path,
    git_code_pin,
    local_artifact_reference,
    preflight_digest,
    record_terminal_attempt,
    resolve_predecessor,
    role_pins_digest,
    sha256_argument,
    write_error_receipt,
)


@pytest.fixture(autouse=True)
def _spool_only_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never inherit operator Logbook configuration."""

    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)


def _state(**overrides: object) -> AttemptState:
    values: dict[str, object] = {
        "build_id": "uk-fixture-attempt",
        "identity_digest": "1" * 64,
        "input_pins_digest": "2" * 64,
        "phases_reached": ["attempt_started"],
        "gate_verdicts": {
            "pipeline": {
                "verdict": "running",
                "receipt": "pending-build-scoped-terminal-receipt",
            }
        },
    }
    values.update(overrides)
    return AttemptState(**values)


def test_git_code_pin_resolves_and_rejects_malformed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin = "a" * 40

    def fake_check_output(
        command: list[str],
        *,
        cwd: Path,
        stderr: int,
        text: bool,
    ) -> str:
        assert command == ["git", "rev-parse", "HEAD"]
        assert cwd == tmp_path
        assert stderr == subprocess.DEVNULL
        assert text is True
        return f"{pin}\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    assert git_code_pin(tmp_path) == pin

    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "bad\n")
    with pytest.raises(ValueError, match="malformed"):
        git_code_pin(tmp_path)

    def raise_git(*args: object, **kwargs: object) -> str:
        raise subprocess.CalledProcessError(1, ["git"])

    monkeypatch.setattr(subprocess, "check_output", raise_git)
    with pytest.raises(RuntimeError, match="Could not resolve"):
        git_code_pin(tmp_path)


def test_local_artifact_reference_repo_home_and_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    artifact = repo / "out" / "dataset.h5"
    artifact.parent.mkdir()
    artifact.touch()

    def repo_check_output(
        command: list[str],
        *,
        cwd: Path,
        stderr: int,
        text: bool,
    ) -> str:
        assert command == ["git", "rev-parse", "--show-toplevel"]
        assert cwd == repo
        assert stderr == subprocess.DEVNULL
        assert text is True
        return f"{repo}\n"

    monkeypatch.setattr(subprocess, "check_output", repo_check_output)
    assert local_artifact_reference(artifact, repository_hint=repo) == (
        "local://out/dataset.h5"
    )

    home = tmp_path / "home"
    home.mkdir()
    home_artifact = home / "exports" / "row.json"
    home_artifact.parent.mkdir()
    home_artifact.touch()
    monkeypatch.setattr(Path, "home", lambda: home)

    def no_repo(*args: object, **kwargs: object) -> str:
        raise subprocess.CalledProcessError(1, ["git"])

    monkeypatch.setattr(subprocess, "check_output", no_repo)
    assert local_artifact_reference(home_artifact, repository_hint=repo) == (
        "local://~/exports/row.json"
    )

    outside = tmp_path / "outside" / "receipt.json"
    outside.parent.mkdir()
    outside.touch()
    assert local_artifact_reference(outside, repository_hint=repo) == (
        f"local://{outside.resolve().as_posix().lstrip('/')}"
    )


def test_resolve_predecessor_cli_env_agree_conflict_and_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "b" * 64
    assert resolve_predecessor(None) is None
    assert resolve_predecessor(digest) == digest

    monkeypatch.setenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", digest)
    assert resolve_predecessor(None) == digest
    assert resolve_predecessor(digest) == digest

    with pytest.raises(ValueError, match="disagrees"):
        resolve_predecessor("c" * 64)

    monkeypatch.setenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", "not-a-digest")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        resolve_predecessor(None)

    with pytest.raises(argparse.ArgumentTypeError, match="lowercase SHA-256"):
        sha256_argument("ABC")
    assert sha256_argument(digest) == digest


def test_role_pins_digest_golden_and_rejects_unnormalized_bytes_key() -> None:
    pins = {
        "dataset": {"sha256": "0" * 64, "size_bytes": 123},
        "ladder": {"sha256": "f" * 64, "size_bytes": 456},
    }
    expected = hashlib.sha256(canonical_json_bytes(pins)).hexdigest()

    assert role_pins_digest(pins) == expected

    with pytest.raises(ValueError, match="size_bytes"):
        role_pins_digest({"dataset": {"sha256": "0" * 64, "bytes": 123}})

    with pytest.raises(ValueError, match="sha256"):
        role_pins_digest({"dataset": {"sha256": "not-a-digest", "size_bytes": 123}})


def test_error_receipt_and_placeholder_drop_only_when_sole_key(tmp_path: Path) -> None:
    state = _state()
    path = write_error_receipt(
        error_receipt_path(tmp_path, build_id=state.build_id),
        state=state,
        pipeline="uk-local-rowwise",
        error=RuntimeError("boom"),
    )
    apply_error_verdict(
        state,
        f"{local_artifact_reference(path, repository_hint=tmp_path)}#/error_type",
    )

    assert state.gate_verdicts == {
        "pipeline_error": {
            "verdict": "error",
            "receipt": f"{local_artifact_reference(path, repository_hint=tmp_path)}#/error_type",
        }
    }
    assert state.phases_reached == ["attempt_started", "error"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pipeline"] == "uk-local-rowwise"
    assert payload["error_type"] == "builtins.RuntimeError"
    assert payload["message"] == "boom"

    state = _state(
        gate_verdicts={
            "pipeline": {
                "verdict": "running",
                "receipt": "pending-build-scoped-terminal-receipt",
            },
            "gate": {"verdict": "failed", "receipt": "local://receipt#/gate"},
        }
    )
    apply_error_verdict(state, "local://error.json#/error_type")
    assert "pipeline" in state.gate_verdicts
    assert state.gate_verdicts["pipeline_error"]["verdict"] == "error"


def test_record_terminal_attempt_writes_one_validated_row_and_refuses_double_record(
    tmp_path: Path,
) -> None:
    state = _state(
        gate_verdicts={
            "uk_gate": {"verdict": "passed", "receipt": "local://gate.json"}
        },
        artifact_location="local://out/dataset.h5",
    )
    append_phase(state, "configured")
    append_phase(state, "configured")

    spool_path = record_terminal_attempt(
        state=state,
        started_at=1.0,
        now=lambda: 3.5,
        started_ts=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
        pipeline="uk-local-rowwise",
        rung="f100",
        seed=42,
        code_pin="a" * 40,
        disposition="iterating",
        predecessor=None,
        spool_dir=tmp_path / "logbook-spool",
    )

    assert state.spool_path == spool_path
    row = load_logbook_row(spool_path)
    assert frozenset(row.to_mapping()) == LOGBOOK_ROW_FIELDS
    assert row.build_id == "uk-fixture-attempt"
    assert row.pipeline == "uk-local-rowwise"
    assert row.phases_reached == ("attempt_started", "configured")
    assert row.wall_seconds == 2.5

    with pytest.raises(RuntimeError, match="already recorded"):
        record_terminal_attempt(
            state=state,
            started_at=1.0,
            now=lambda: 4.0,
            started_ts=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
            pipeline="uk-local-rowwise",
            rung="f100",
            seed=42,
            code_pin="a" * 40,
            disposition="iterating",
            predecessor=None,
            spool_dir=tmp_path / "logbook-spool",
        )


def test_preflight_digest_and_atomic_write_json_are_canonical(tmp_path: Path) -> None:
    assert (
        preflight_digest("uk-frs-staging")
        == hashlib.sha256(
            canonical_json_bytes(
                {"pipeline": "uk-frs-staging", "state": "preflight"}
            )
        ).hexdigest()
    )

    path = tmp_path / "nested" / "receipt.json"
    atomic_write_json(path, {"b": 1, "a": {"x": 2}})
    assert path.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "x": 2\n  },\n  "b": 1\n}\n'
    )


def test_atomic_write_json_preserves_nulls_and_rejects_non_string_keys(
    tmp_path: Path,
) -> None:
    """Receipts are audit artifacts: an explicit null must survive.

    Review finding on #666 (PR #670): the extracted writer silently dropped
    None-valued keys, so "explicitly null" and "never set" became
    indistinguishable in immutable receipts. The US driver's writer keeps
    nulls and rejects non-string keys; the shared module must match.
    """

    path = tmp_path / "receipt.json"
    atomic_write_json(
        path,
        {"gate": {"verdict": "failed", "reason": None}},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["gate"] == {"verdict": "failed", "reason": None}

    with pytest.raises(ValueError, match="string JSON keys"):
        atomic_write_json(tmp_path / "bad.json", {"gate": {1: "x"}})


def test_sha256_argument_message() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="lowercase SHA-256"):
        sha256_argument("1" * 63)
