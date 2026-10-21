"""OBR's NICs total sits beside the bound class rows (microcosm#1095)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from microcosm.build.uk_runtime.local_targets import load_uk_population_contract
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
STABLE_UK_FACT_FEED_NAME = ".codex-work/consumer_facts_uk.jsonl"
CLASS_ROWS = ("obr.ni_employee", "obr.ni_employer", "obr.ni_self_employed")


def test_the_nics_total_is_a_diagnostic_beside_the_class_rows() -> None:
    """OBR's table 3.8 total is cash, the class rows accrued, and it carries
    statutory-payment recoveries, Class 1A, 1B and 3 and settlements no
    household carries, so it leaves the fit (María's ruling of 2026-10-09)."""

    from microcosm.build.uk_runtime.ledger_targets import (
        UK_REQUIRED_TARGET_DIAGNOSTICS,
    )

    contract = load_uk_population_contract()
    target_ids = {target["target_id"] for target in contract["targets"]}
    assert "obr.ni" not in target_ids
    assert set(CLASS_ROWS) <= target_ids
    parity = contract["registry_parity"]
    assert "obr/ni" in parity["excluded"]
    assert "obr/ni" not in parity["mapped"]
    assert "obr.ni" not in parity["scope_target_ids"]
    declaration = contract["diagnostic_references"]["obr.ni"]
    assert declaration["attach_to_target"] == "obr.ni_employee"
    assert UK_REQUIRED_TARGET_DIAGNOSTICS["obr.ni_employee"] == ("obr.ni",)
    reference = declaration["reference"]
    assert reference["period_match_policy"] == "exact"
    assert reference["ledger_selector"]["source_concept"] == "obr.ni_total"
    assert reference["ledger_fact_key"]


def test_the_nics_diagnostic_carries_the_fy2025_26_forecast() -> None:
    """Feed-gated: the diagnostic resolves its exact FY2025-26 fact."""

    from microcosm.build.uk_runtime.ledger_targets import (
        _attached_diagnostic_metadata,
    )

    root = _TEST_PATHS.repository
    configured = os.environ.get("CHRONICLE_UK_FACTS")
    feed = Path(configured) if configured else root / STABLE_UK_FACT_FEED_NAME
    if feed.is_dir():
        feed = feed / "consumer_facts.jsonl"
    if not feed.is_file():
        pytest.skip("pinned UK Chronicle consumer feed is not present")
    with feed.open(encoding="utf-8") as handle:
        facts = tuple(json.loads(line) for line in handle if '"obr.ni_total"' in line)

    metadata = _attached_diagnostic_metadata(facts, "obr.ni_employee")

    prefix = "obr_ni_total_diagnostic"
    assert metadata[f"{prefix}_role"] == "diagnostic_only_not_in_fit"
    assert metadata[f"{prefix}_status"] == "available"
    assert float(metadata[f"{prefix}_value_gbp"]) == pytest.approx(200_081_395_082.71)
