"""Schema-2 contract for US by-origin reviewed exclusions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from microcosm.build.us_runtime.reviewed_exclusions import (
    US_BY_ORIGIN_EXCLUSION_REGISTER_RESOURCE,
    USReviewedExclusion,
    load_us_by_origin_reviewed_exclusion_policy,
    us_default_by_origin_reviewed_exclusion_policy,
    us_exclusion_evaluation_date,
)

_CHECK_ID = "person/puf_tax_itemization/business_is_sstb[clone_0]#dead"


def _entry(**updates: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "reason": "Independently adjudicated measurement scope.",
        "approved_by": "Release owner",
        "adjudication": "owner-review-123",
        "approved_on": "2026-08-01",
        "expires_on": "2026-09-01",
    }
    entry.update(updates)
    return entry


def _payload(*, exclusions: object | None = None) -> dict[str, object]:
    return {
        "schema_version": 2,
        "description": "Test-only US by-origin exclusions.",
        "exclusions": {_CHECK_ID: _entry()} if exclusions is None else exclusions,
    }


def _write_override(path: Path, payload: object) -> tuple[Path, str]:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_committed_register_is_schema_2_empty_and_immutable() -> None:
    policy = us_default_by_origin_reviewed_exclusion_policy()

    assert US_BY_ORIGIN_EXCLUSION_REGISTER_RESOURCE == (
        "by_origin_reviewed_exclusions.json"
    )
    assert policy.policy_source == "committed"
    assert policy.policy_payload()["schema_version"] == 2
    assert policy.exclusions == {}
    assert len(policy.policy_sha256) == 64
    assert policy is us_default_by_origin_reviewed_exclusion_policy()
    with pytest.raises(TypeError):
        policy.exclusions[_CHECK_ID] = USReviewedExclusion(**_entry())  # type: ignore[index,arg-type]


def test_authenticated_override_preserves_all_five_sealed_fields(
    tmp_path: Path,
) -> None:
    path, digest = _write_override(tmp_path / "reviewed.json", _payload())

    policy = load_us_by_origin_reviewed_exclusion_policy(
        path,
        expected_sha256=digest,
    )

    assert policy.policy_source == "override"
    assert policy.policy_payload()["exclusions"] == {_CHECK_ID: _entry()}
    assert len(policy.policy_sha256) == 64


@pytest.mark.parametrize(
    ("source", "digest"),
    (("reviewed.json", None), (None, "0" * 64)),
)
def test_override_refuses_partial_arming(
    source: str | None,
    digest: str | None,
) -> None:
    with pytest.raises(ValueError, match="source and expected_sha256 together"):
        load_us_by_origin_reviewed_exclusion_policy(
            source,
            expected_sha256=digest,
        )


def test_override_refuses_digest_mismatch(tmp_path: Path) -> None:
    path, _digest = _write_override(tmp_path / "reviewed.json", _payload())

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_us_by_origin_reviewed_exclusion_policy(
            path,
            expected_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "payload",
    (
        [],
        {"schema_version": 2, "description": "x"},
        {
            "schema_version": True,
            "description": "x",
            "exclusions": {},
        },
        {"schema_version": 2, "description": "", "exclusions": {}},
        {"schema_version": 2, "description": "x", "exclusions": []},
    ),
)
def test_loader_refuses_top_level_schema_drift(
    tmp_path: Path,
    payload: object,
) -> None:
    path, digest = _write_override(tmp_path / "reviewed.json", payload)

    with pytest.raises((TypeError, ValueError)):
        load_us_by_origin_reviewed_exclusion_policy(path, expected_sha256=digest)


@pytest.mark.parametrize(
    "check_id",
    (
        "",
        " business#dead",
        "business",
        "business#dead#incidence",
        "business#unknown",
    ),
)
def test_loader_requires_exact_leg_plus_criterion_keys(
    tmp_path: Path,
    check_id: str,
) -> None:
    path, digest = _write_override(
        tmp_path / "reviewed.json",
        _payload(exclusions={check_id: _entry()}),
    )

    with pytest.raises(ValueError, match="leg#criterion"):
        load_us_by_origin_reviewed_exclusion_policy(path, expected_sha256=digest)


@pytest.mark.parametrize(
    "entry",
    (
        None,
        {},
        _entry(reason=""),
        _entry(approved_by=" owner "),
        _entry(approved_on="20260801"),
        _entry(expires_on="2026-08-01"),
        _entry(expires_on="2026-07-31"),
    ),
)
def test_loader_refuses_incomplete_or_invalid_receipts(
    tmp_path: Path,
    entry: object,
) -> None:
    path, digest = _write_override(
        tmp_path / "reviewed.json",
        _payload(exclusions={_CHECK_ID: entry}),
    )

    with pytest.raises((TypeError, ValueError)):
        load_us_by_origin_reviewed_exclusion_policy(path, expected_sha256=digest)


@pytest.mark.parametrize(
    "raw",
    (
        '{"schema_version":2,"description":"x","exclusions":{},"exclusions":{}}',
        '{"schema_version":2,"description":"x","exclusions":{},"x":NaN}',
        "{",
    ),
)
def test_loader_refuses_duplicate_nonfinite_or_malformed_json(
    tmp_path: Path,
    raw: str,
) -> None:
    path = tmp_path / "reviewed.json"
    raw_bytes = raw.encode("utf-8")
    path.write_bytes(raw_bytes)

    with pytest.raises(ValueError):
        load_us_by_origin_reviewed_exclusion_policy(
            path,
            expected_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )


def test_receipt_force_window_is_inclusive() -> None:
    receipt = USReviewedExclusion(**_entry())  # type: ignore[arg-type]

    assert receipt.premature(date(2026, 7, 31))
    assert not receipt.premature(date(2026, 8, 1))
    assert not receipt.expired(date(2026, 9, 1))
    assert receipt.expired(date(2026, 9, 2))


def test_exclusion_clock_refuses_datetime_and_nondate() -> None:
    expected = date(2026, 8, 19)

    assert us_exclusion_evaluation_date(expected) is expected
    with pytest.raises(TypeError, match="datetime.date"):
        us_exclusion_evaluation_date(datetime(2026, 8, 19, tzinfo=UTC))
    with pytest.raises(TypeError, match="datetime.date"):
        us_exclusion_evaluation_date("2026-08-19")  # type: ignore[arg-type]
