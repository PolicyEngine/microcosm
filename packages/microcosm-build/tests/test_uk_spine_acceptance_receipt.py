"""The committed spine acceptance receipt binds to the production plan.

microcosm#771: the previous acceptance evidence quietly described a 24-stage
build after the plan had grown to 25. This binder makes that class of drift a
CI failure. The #828 and #832 stages are deliberately pending the licensed I5
rebuild, so the historical receipt stays truthful while the test pins the two
reviewed roster differences from the current driver. I5 restores strict roster
equality when it re-mints the receipt.
"""

from __future__ import annotations

import json
from importlib.resources import files

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.graph import (
    UK_SPINE_EXCLUSIONS,
    uk_spine_graph,
)
from microcosm.graph import compile_graph


def _receipt() -> dict:
    return json.loads(
        files("microcosm.build.uk")
        .joinpath("spine_candidate_acceptance.json")
        .read_text()
    )


def _production_graph_stage_names() -> tuple[str, ...]:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    declared = {
        stage.stage
        for stage in spec.sources.stages
        if stage.stage not in UK_SPINE_EXCLUSIONS
    }
    compiled = compile_graph(uk_spine_graph(spec))
    return tuple(node_id for node_id in compiled.order if node_id in declared)


def test_receipt_roster_is_the_production_plan():
    receipt = _receipt()
    accepted_roster = tuple(receipt["candidate"]["stage_roster"])
    production_roster = _production_graph_stage_names()
    expected_extra_stages = {"uc_capital_coherence", "uc_reporter_redraw"}

    assert set(production_roster) - set(accepted_roster) == expected_extra_stages
    assert tuple(
        stage for stage in production_roster if stage not in expected_extra_stages
    ) == accepted_roster
    assert receipt["candidate"]["stage_count"] == len(accepted_roster)


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
    battery = receipt["spine_battery"]
    assert battery["blocked_at_phase"] is None
    assert battery["statuses"] == {"passed": 14}
    assert len(battery["report_sha256"]) == 64
