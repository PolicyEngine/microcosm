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
    """OBR's total NICs line is cash, the class rows accrued, and it carries
    statutory-payment recoveries, Class 1B and 3, settlements, and Class 1A on
    every benefit in kind, of which households carry only the company cars and
    fuel, so it leaves the fit (María's ruling of 2026-10-09)."""

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


def test_the_employer_row_binds_class_1_alone() -> None:
    """OBR's employer line is table 4.4 row 18, Class 1 alone; Class 1A sits in
    row 21 ('Other NIC'). policyengine-uk's ni_employer adds ni_class_1a since
    2.165.0, so the row binds ni_class_1_employer, and the exclusion that holds
    it out names that measure in its rebind condition (microcosm#1430)."""

    contract = load_uk_population_contract()
    row = next(t for t in contract["targets"] if t["target_id"] == "obr.ni_employer")
    binding = row["bindings"]["policyengine"]
    assert binding["value_variable"] == "ni_class_1_employer"
    assert "value_expression" not in binding
    register = json.loads(
        (
            _TEST_PATHS.package
            / "src/microcosm/build/uk/calibration_measure_exclusions.json"
        ).read_text(encoding="utf-8")
    )
    exclusion = next(
        entry for entry in register["exclusions"] if entry["name"] == "obr.ni_employer"
    )
    assert "binds policyengine-uk's ni_class_1_employer" in exclusion["reason"]
    assert (
        "nets the allowance out of the measure the row binds, ni_class_1_employer"
        in exclusion["reason"]
    )


def test_the_nics_diagnostic_carries_the_fy2025_26_outturn() -> None:
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
    # The October 2026 EFO's outturn: March's forecast left the feed with
    # PolicyEngine/chronicle#397 (microcosm#1430).
    assert float(metadata[f"{prefix}_value_gbp"]) == pytest.approx(200_829_000_000)
    assert metadata[f"{prefix}_ledger_assertion"] == "observation"
