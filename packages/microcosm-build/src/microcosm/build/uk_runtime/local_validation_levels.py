"""Loader and fail-closed validation for UK report-only local benchmarks."""

from __future__ import annotations

import copy
import json
from importlib.resources import files
from typing import Any

from microcosm.build.uk_runtime.chronicle_feed import load_uk_chronicle_feed

LOCAL_VALIDATION_LEVELS_RESOURCE = "local_validation_levels.json"
VALID_LEVEL_STATUSES = frozenset({"available", "awaiting_facts"})


def load_uk_local_validation_levels() -> dict[str, Any]:
    payload = json.loads(
        files("microcosm.build.uk")
        .joinpath(LOCAL_VALIDATION_LEVELS_RESOURCE)
        .read_text()
    )
    _validate(payload)
    return copy.deepcopy(payload)


#: The one UK Chronicle pin (uk/chronicle_feed.json), read once at import so
#: register validation compares against the declaration without re-parsing it.
_EXPECTED_FEED_PIN = load_uk_chronicle_feed()


def _validate(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("UK local validation-level register must be an object.")
    if payload.get("schema_version") != 1:
        raise ValueError("UK local validation-level schema_version must be 1.")
    if payload.get("register_kind") != "uk_local_validation_levels":
        raise ValueError("UK local validation-level register_kind is invalid.")
    feed = payload.get("source_feed")
    pin = _EXPECTED_FEED_PIN
    if (
        not isinstance(feed, dict)
        or feed.get("sha256") != pin.facts_sha256
        or feed.get("manifest_sha256") != pin.manifest_sha256
        or feed.get("source_commit") != pin.source_commit
    ):
        raise ValueError(
            "UK local validation-level feed pin is missing or stale against "
            "uk/chronicle_feed.json."
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("UK local validation-level register needs rows.")
    ids = [str(row.get("id", "")) for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or any(not row_id for row_id in ids):
        raise ValueError("Every UK local validation-level row needs an id.")
    if len(ids) != len(set(ids)):
        raise ValueError("UK local validation-level row ids must be unique.")
    for row in rows:
        status = row.get("status")
        if status not in VALID_LEVEL_STATUSES:
            raise ValueError(f"validation row {row['id']!r} has invalid status.")
        if not isinstance(row.get("in_sample"), bool):
            raise ValueError(f"validation row {row['id']!r} needs in_sample bool.")
        for key in ("name", "comparison_kind", "geography_id", "estimate", "evidence"):
            if not row.get(key):
                raise ValueError(f"validation row {row['id']!r} needs {key}.")
        if status == "available" and not row.get("benchmark_selector"):
            raise ValueError(
                f"available validation row {row['id']!r} needs benchmark_selector."
            )
        if status == "awaiting_facts" and not row.get("needs_facts"):
            raise ValueError(
                f"awaiting validation row {row['id']!r} needs needs_facts."
            )
        if "benchmark_value" in row or "value" in row:
            raise ValueError(f"validation row {row['id']!r} must remain value-free.")
