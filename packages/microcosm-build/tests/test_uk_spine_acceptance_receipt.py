"""The committed spine acceptance receipt binds to the production plan.

microcosm#771: the previous acceptance evidence quietly described a 24-stage
build after the plan had grown to 25. This binder makes that class of drift a
CI failure. The #828 stage is deliberately pending the licensed I7 rebuild, so
the historical receipt stays truthful while the test pins its one reviewed
roster difference from the current driver.
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
    coherence_index = production_roster.index("uc_capital_coherence")

    assert production_roster == (
        *accepted_roster[:coherence_index],
        "uc_capital_coherence",
        *accepted_roster[coherence_index:],
    )
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
