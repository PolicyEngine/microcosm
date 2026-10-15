"""Published local cells below the small-cell floor are deferred by rule (#1123)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from importlib import resources as importlib_resources

import pytest

from microcosm.build.uk_runtime.ledger_targets import (
    assert_uk_local_deferrals_in_force,
    uk_small_cell_deferred_cells,
    uk_small_cell_rule_is_signed,
)
from microcosm.build.uk_runtime.uprating_holds import UK_HOLD_EXPIRY_NOTICE_DAYS
from tools.generate_uk_local_target_references import _small_cell_deferrals


def _rule_register(tmp_path, **overrides):
    """The packaged rule, signed for the test unless a field is overridden."""

    register = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_small_cell_rule.json")
        .read_text(encoding="utf-8")
    )
    for rule in register["rules"]:
        rule.update(
            {
                "approved_by": "reviewer",
                "approved_on": "2026-10-01",
                "status": "approved",
                **overrides,
            }
        )
    path = tmp_path / "local_small_cell_rule.json"
    path.write_text(json.dumps(register), encoding="utf-8")
    return path


def _candidate(area: str, value: float, status: str = "active", **extra) -> dict:
    return {
        "geography_id": area,
        "status": status,
        "resolved_value": value,
        **extra,
    }


def _membership(*candidates: dict) -> dict:
    return {
        "targets": {
            "welshgov.council_tax_stock.by_area.band_h": {
                "geography_levels": {
                    "local_authority": {"candidates": list(candidates)}
                }
            },
            "ons.census.households": {
                "geography_levels": {
                    "local_authority": {"candidates": [_candidate("W06000019", 3.0)]}
                }
            },
        }
    }


def test_cells_below_the_floor_are_deferred_by_rule_with_its_expiry(tmp_path):
    deferrals = _small_cell_deferrals(
        _membership(_candidate("W06000019", 3.0), _candidate("W06000015", 900.0)),
        _rule_register(tmp_path),
    )
    (deferral,) = deferrals
    assert deferral.target_id == "welshgov.council_tax_stock.by_area.band_h"
    assert deferral.area_ids == ("W06000019",)
    assert deferral.defer_if_compiles is True
    assert deferral.reason_id == "council_tax_band_household_floor"
    assert deferral.expires_on == "2027-04-01"


def test_a_target_outside_the_rule_is_never_deferred(tmp_path):
    membership = _membership(_candidate("W06000015", 900.0))
    assert _small_cell_deferrals(membership, _rule_register(tmp_path)) == ()


@pytest.mark.parametrize(
    "unsigned",
    [
        {"approved_by": None},
        {"approved_on": None},
        {"status": "proposed"},
        {"status": "rejected"},
    ],
)
def test_a_rule_nobody_signed_defers_nothing(tmp_path, unsigned):
    membership = _membership(_candidate("W06000019", 3.0))
    assert _small_cell_deferrals(membership, _rule_register(tmp_path, **unsigned)) == ()


def test_a_membership_deferring_under_an_unsigned_rule_is_refused():
    membership = {
        "signed_deferrals": [
            {
                "target_id": "welshgov.council_tax_stock.by_area.band_h",
                "geography_level": "local_authority",
                "reason_id": "council_tax_band_household_floor",
                "expires_on": "2027-04-01",
            }
        ]
    }
    rule = {"rule_id": "council_tax_band_household_floor", "status": "proposed"}
    assert not uk_small_cell_rule_is_signed(rule)
    with pytest.raises(ValueError, match="unsigned small-cell rule"):
        assert_uk_local_deferrals_in_force(membership, date(2026, 10, 7), rules=[rule])
    signed = {
        **rule,
        "approved_by": "r",
        "approved_on": "2026-10-01",
        "status": "approved",
    }
    assert_uk_local_deferrals_in_force(membership, date(2026, 10, 7), rules=[signed])
    # A status other than ``approved`` leaves a signed-looking rule out of force.
    with pytest.raises(ValueError, match="unsigned small-cell rule"):
        assert_uk_local_deferrals_in_force(
            membership, date(2026, 10, 7), rules=[{**signed, "status": "withdrawn"}]
        )


def test_a_deferral_under_a_rule_no_longer_in_the_register_is_refused():
    membership = {
        "signed_deferrals": [
            {
                "target_id": "welshgov.council_tax_stock.by_area.band_h",
                "geography_level": "local_authority",
                "reason_id": "council_tax_band_household_floor",
                "expires_on": "2027-04-01",
            }
        ]
    }
    with pytest.raises(ValueError, match="not in uk/local_small_cell_rule.json"):
        assert_uk_local_deferrals_in_force(membership, date(2026, 10, 7), rules=[])


def test_deferred_cells_keep_their_values_for_reconciliation():
    membership = _membership(
        _candidate(
            "W06000019",
            3.0,
            status="signed_deferred",
            signed_reason_id="council_tax_band_household_floor",
            deferred_value=3.0,
        )
    )
    # The membership binds the authority's census households, so a whole run
    # carries it: a run whose ladder lacks it is a narrowed one.
    assert uk_small_cell_deferred_cells(membership) == (
        {
            "target_id": "welshgov.council_tax_stock.by_area.band_h",
            "geography_level": "local_authority",
            "geography_id": "W06000019",
            "value": 3.0,
            "area_has_bound_cells": True,
        },
    )


def test_an_expired_deferral_is_refused_at_its_review_date():
    membership = {
        "signed_deferrals": [
            {
                "target_id": "welshgov.council_tax_stock.by_area.band_h",
                "geography_level": "local_authority",
                "reason_id": "council_tax_band_household_floor",
                "expires_on": "2027-04-01",
            }
        ]
    }
    rules = [
        {
            "rule_id": "council_tax_band_household_floor",
            "approved_by": "r",
            "approved_on": "2026-10-01",
            "status": "approved",
        }
    ]
    assert_uk_local_deferrals_in_force(membership, date(2027, 4, 1), rules=rules)
    with pytest.raises(ValueError, match="expired"):
        assert_uk_local_deferrals_in_force(membership, date(2027, 4, 2), rules=rules)


def _committed(resource: str):
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(resource)
        .read_text(encoding="utf-8")
    )


def test_every_committed_rule_is_signed():
    # An unsigned rule defers nothing, so a proposed rule left in the register
    # would leave the cells it names bound without a word. Both rules were
    # signed on 2026-10-14 (microcosm#1278).
    rules = _committed("local_small_cell_rule.json")["rules"]
    assert [
        rule["rule_id"] for rule in rules if not uk_small_cell_rule_is_signed(rule)
    ] == []


def test_no_committed_rule_is_about_to_lapse():
    # A clock on purpose, as for the uprating holds: every UK build refuses a
    # deferral after its rule's expiry, and feed-backed compiles are skipped in
    # CI, so this fails UK_HOLD_EXPIRY_NOTICE_DAYS days before the earliest
    # expiry, while the builds still pass. Clear it by a new review and
    # signature in uk/local_small_cell_rule.json.
    earliest = min(
        date.fromisoformat(rule["expires_on"])
        for rule in _committed("local_small_cell_rule.json")["rules"]
    )
    days_left = (earliest - date.today()).days
    assert days_left > UK_HOLD_EXPIRY_NOTICE_DAYS, (
        f"A UK small-cell rule expires on {earliest.isoformat()}, in {days_left} "
        "day(s); every UK build refuses its deferrals from the next day."
    )


def test_the_committed_membership_defers_each_rules_cells():
    # The regenerated references carry the signed rules' deferrals: the
    # council-tax band cells under 50 dwellings (band G 3, band H 7, band I 6)
    # and the higher rate taxpayer cells HMRC publishes at 5,000 or fewer.
    cells: dict[str, Counter] = {}
    for deferral in _committed("local_target_reference_membership.json")[
        "signed_deferrals"
    ]:
        if deferral["reason_id"] in {
            "council_tax_band_household_floor",
            "hmrc_rounded_thousand_floor",
        }:
            cells.setdefault(deferral["reason_id"], Counter())[
                deferral["geography_level"]
            ] += len(deferral["area_ids"])
    assert cells == {
        "council_tax_band_household_floor": Counter({"local_authority": 16}),
        "hmrc_rounded_thousand_floor": Counter(
            {"constituency": 155, "local_authority": 37}
        ),
    }


def test_the_parity_receipt_names_the_rule_on_a_small_cell_deferral():
    # A cell the incumbent carries but a signed rule defers is fixture-only in
    # the local parity receipt, and its reason is the deferral's own signed
    # rationale, which names the rule (microcosm#1278 review).
    reason_by_cell = {
        (deferral["target_id"], area): deferral["reason_id"]
        for deferral in _committed("local_target_reference_membership.json")[
            "signed_deferrals"
        ]
        if deferral["reason_id"]
        in {"council_tax_band_household_floor", "hmrc_rounded_thousand_floor"}
        for area in deferral["area_ids"]
    }
    rows = [
        row
        for row in _committed(
            "ledger_compile_parity_local_incumbent_2025_signed_differences.json"
        )["differences"]
        if tuple(row["name"].split("@", 1)) in reason_by_cell
    ]
    assert rows
    for row in rows:
        assert row["kind"] == "fixture_only"
        assert reason_by_cell[tuple(row["name"].split("@", 1))] in row["reason"]
