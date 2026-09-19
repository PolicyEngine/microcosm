"""Synthetic checks of the prepared harness; no population files are opened."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "experiments/acs_spm_source_512a.py"
_SPEC = importlib.util.spec_from_file_location("acs_spm_source_harness", _PATH)
assert _SPEC and _SPEC.loader
harness = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(harness)


@pytest.mark.parametrize(
    "actual,expected",
    [
        (["id"], ["id", "amount"]),
        (["id", "amount", "extra"], ["id", "amount"]),
        (["id", "id"], ["id", "amount"]),
        (["id"], []),
        (["id"], ["id", "id"]),
    ],
)
def test_complete_schema_refuses_dropped_extra_and_duplicate_columns(actual, expected):
    with pytest.raises(harness.HarnessRefusalError):
        harness.require_columns(actual, expected)


@pytest.mark.parametrize(
    "records",
    [
        [{"id": 1}],
        [{"id": 1, "amount": 2, "extra": 3}],
        [{"id": 1, "amount": 2}, {"id": 2}],
        ["raw private data"],
        {},
    ],
)
def test_every_golden_row_must_have_exact_pinned_schema(records):
    with pytest.raises(harness.HarnessRefusalError):
        harness.require_record_schema(records, ["id", "amount"])


def test_empty_golden_needs_explicit_historical_columns_and_zero_row_count():
    harness.require_record_schema([], ["id", "amount"], row_count=0)
    with pytest.raises(
        harness.HarnessRefusalError, match="invalid_pinned_column_schema"
    ):
        harness.require_record_schema([], [], row_count=0)
    with pytest.raises(harness.HarnessRefusalError, match="golden_row_count_mismatch"):
        harness.require_record_schema([], ["id", "amount"], row_count=1)


def test_schema_allows_column_order_only_to_change():
    harness.require_columns(["amount", "id"], ["id", "amount"])
    harness.require_record_schema([{"amount": 2, "id": 1}], ["id", "amount"])


@pytest.mark.parametrize(
    "head,status,reason",
    [
        ("reviewed", "", None),
        ("other", "", "source_head_mismatch"),
        ("reviewed", " M source.py", "source_worktree_dirty"),
        ("reviewed", "?? unreviewed.py", "source_worktree_dirty"),
    ],
)
def test_checkout_requires_reviewed_head_and_no_uncommitted_files(
    monkeypatch, tmp_path, head, status, reason
):
    calls = []

    def git(command, *, cwd, text):
        assert cwd == tmp_path and text is True
        calls.append(command)
        return head if command == ["git", "rev-parse", "HEAD"] else status

    monkeypatch.setattr(harness.subprocess, "check_output", git)
    if reason:
        with pytest.raises(harness.HarnessRefusalError, match=reason):
            harness.require_checkout(tmp_path, "reviewed", "source")
    else:
        assert harness.require_checkout(tmp_path, "reviewed", "source") == "reviewed"
        assert calls == [
            ["git", "rev-parse", "HEAD"],
            ["git", "status", "--porcelain=v1"],
        ]


def test_failure_receipt_only_discloses_own_static_reason():
    refused = harness.failure_report(
        harness.HarnessRefusalError("source_head_mismatch")
    )
    assert refused["reason"] == "source_head_mismatch"
    failure = harness.failure_report(AssertionError("raw private data"))
    assert failure == {
        "status": "fail",
        "scope": "development_source_only",
        "error_type": "AssertionError",
    }
