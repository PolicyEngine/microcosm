"""Formatting-only outer-failure controls; no main/audit/dependency execution."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def diagnostic():
    source = (
        Path(__file__).resolve().parents[3] / "tools/spec_seed_identity_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("outer_failure_controls", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind", [TypeError, PermissionError])
def test_outer_error_keeps_fixed_type_and_code_coordinates_without_message(
    diagnostic, kind
):
    try:
        raise kind("secret-password.csv user-input-payload")
    except kind as error:
        payload = diagnostic.outer_failure_context(error)
    value = json.loads(payload)
    assert value["exception_type"] == kind.__name__
    assert value["code"] == "OUTER_FAILURE"
    assert value["completed"] is value["coverage_pass"] is False
    assert value["artifact_status_not_modified"] is True
    assert b"secret-password" not in payload
    assert b"user-input-payload" not in payload
    assert len(payload) <= 8192


def test_actual_short_named_traceback_frame_is_retained(diagnostic):
    def cleanup_probe():
        raise TypeError("not emitted")

    try:
        cleanup_probe()
    except TypeError as error:
        result = json.loads(diagnostic.outer_failure_context(error))
    assert result["traceback_code_frames"][-1]["function"] == "cleanup_probe"
    assert result["traceback_code_frames"][-1]["line"] > 0


def test_fixed_refusal_retains_attached_sanitized_context(diagnostic):
    context = {"code": "READ_SCOPE", "event": "open", "requested": {"path": None}}
    error = diagnostic.RefusalError(
        "READ_SCOPE", boundary_context=diagnostic.encoded(context)
    )
    result = json.loads(diagnostic.outer_failure_context(error))
    assert result["exception_type"] == "RefusalError"
    assert result["code"] == "READ_SCOPE"
    assert result["boundary_refusal"] == context


def test_unknown_refusal_and_exception_names_are_fixed_labels(diagnostic):
    class SecretNamedError(Exception):
        pass

    errors = (
        diagnostic.RefusalError("arbitrary-secret-code"),
        SecretNamedError("secret-message"),
    )
    for error in errors:
        payload = diagnostic.outer_failure_context(error)
        assert b"secret" not in payload.lower()
        assert b"SecretNamedError" not in payload
        result = json.loads(payload)
        assert result["code"] in {"OUTER_FAILURE", "OTHER_FIXED_REFUSAL"}


def test_traceback_scan_and_output_are_bounded(diagnostic):
    def recurse(depth):
        if depth:
            recurse(depth - 1)
        else:
            raise ValueError("unprinted")

    try:
        recurse(40)
    except ValueError as error:
        payload = diagnostic.outer_failure_context(error)
    result = json.loads(payload)
    assert result["traceback_truncated"] is True
    assert len(result["traceback_code_frames"]) <= 8
    assert len(payload) <= 8192


def test_exception_only_metadata_does_not_open_or_resolve_files(
    diagnostic, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("outer report attempted file I/O or resolution")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    payload = diagnostic.outer_failure_context(TypeError("no traceback"))
    assert json.loads(payload)["traceback_code_frames"] == []
