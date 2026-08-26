"""The committed spine acceptance receipt binds to the production plan.

microcosm#771: the previous acceptance evidence quietly described a 24-stage
build after the plan had grown to 25. This binder makes that class of drift a
CI failure: the receipt's stage roster must equal the roster the spine driver
actually executes, its verdicts must be the accepted ones, and its identity
bases must name the stage-time contract the instruments verify.
"""

from __future__ import annotations

import importlib.util
import json
from importlib.resources import files
from pathlib import Path

_DRIVER_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "build_uk_frs_spine.py"
)


def _receipt() -> dict:
    return json.loads(
        files("microcosm.build.uk")
        .joinpath("spine_candidate_acceptance.json")
        .read_text()
    )


def _driver_stage_names() -> tuple[str, ...]:
    spec = importlib.util.spec_from_file_location("build_uk_frs_spine", _DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module._STAGE_NAMES)


def test_receipt_roster_is_the_production_plan():
    receipt = _receipt()
    roster = tuple(receipt["candidate"]["stage_roster"])
    assert roster == _driver_stage_names()
    assert receipt["candidate"]["stage_count"] == len(roster)


def test_receipt_identity_and_verdicts_are_the_accepted_ones():
    receipt = _receipt()
    assert len(receipt["candidate"]["sha256"]) == 64
    assert int(receipt["candidate"]["entity_row_counts"]["household"]) == 52846
    assert receipt["twin"]["payload_identical"] is True
    ladder = receipt["identity_ladder"]
    assert set(ladder) == {"e4", "e5", "e6", "e7", "e8"}
    for check, row in ladder.items():
        assert row["identical_under_permutation"] is True, check
        assert row["matches_stored_columns"] is True, check
    assert ladder["e6"]["nhs_age_basis"] == "stage_time_top_coded"
    assert ladder["e8"]["donor_age_basis"] == "stage_time_top_coded"
    parity = receipt["strict_parity"]
    assert parity["verdict"] == "signed_parity"
    assert parity["unsigned_differences"] == 0
    assert parity["strict_failure"] is False
    assert parity["share_band"]["effective"] == parity["share_band"]["contract"]
