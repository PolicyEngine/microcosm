"""Synthetic checks of the prepared harness; no population files are opened."""

import importlib.util
import json
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


def nullable_comparison_fixture():
    import pandas as pd

    actual = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "role": pd.Series([True, False, pd.NA], dtype="boolean"),
            "label": pd.Series(["one", "two", pd.NA], dtype="string"),
            "amount": pd.Series([1.25, 0, pd.NA], dtype="Float64"),
        }
    )
    golden = pd.DataFrame(json.loads(actual.to_json(orient="records")))
    return actual, golden


def test_json_null_matches_nullable_role_string_and_amount():
    actual, golden = nullable_comparison_fixture()
    before = actual.copy(deep=True)
    harness.assert_golden_equal(actual, golden)
    assert actual.equals(before)


@pytest.mark.parametrize(
    "column,row,value",
    [
        ("role", 2, False),  # Missing role cannot become a child role.
        ("role", 0, False),
        ("role", 0, 1),  # Python's True == 1 must not hide a changed value type.
        ("role", 1, 0),
        ("amount", 2, 0),  # Missing amount cannot become an observed zero.
        ("amount", 0, 1.5),
        ("label", 0, "changed"),
        ("person_id", 0, 99),
    ],
)
def test_missing_sentinel_normalization_retains_real_mismatches(column, row, value):
    actual, golden = nullable_comparison_fixture()
    golden.loc[row, column] = value
    with pytest.raises(AssertionError):
        harness.assert_golden_equal(actual, golden)


@pytest.mark.parametrize("golden_id", [2**53, float(2**53 + 1)])
def test_golden_comparison_preserves_integer_precision_above_float_boundary(golden_id):
    import pandas as pd

    actual = pd.DataFrame({"person_id": [2**53 + 1]})
    golden = pd.DataFrame({"person_id": [golden_id]})
    with pytest.raises(AssertionError):
        harness.assert_golden_equal(actual, golden)
