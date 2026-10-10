"""Published local cells below the small-cell floor are deferred by rule (#1123)."""

from __future__ import annotations

from datetime import date

import pytest

from microcosm.build.uk_runtime.ledger_targets import (
    assert_uk_local_deferrals_in_force,
    uk_small_cell_deferred_cells,
)
from tools.generate_uk_local_target_references import _small_cell_deferrals


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


def test_cells_below_the_floor_are_deferred_by_rule_with_its_expiry():
    deferrals = _small_cell_deferrals(
        _membership(_candidate("W06000019", 3.0), _candidate("W06000015", 900.0))
    )
    (deferral,) = deferrals
    assert deferral.target_id == "welshgov.council_tax_stock.by_area.band_h"
    assert deferral.area_ids == ("W06000019",)
    assert deferral.defer_if_compiles is True
    assert deferral.reason_id == "council_tax_band_household_floor"
    assert deferral.expires_on == "2027-04-01"


def test_a_target_outside_the_rule_is_never_deferred():
    membership = _membership(_candidate("W06000015", 900.0))
    assert _small_cell_deferrals(membership) == ()


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
    assert uk_small_cell_deferred_cells(membership) == (
        {
            "target_id": "welshgov.council_tax_stock.by_area.band_h",
            "geography_level": "local_authority",
            "geography_id": "W06000019",
            "value": 3.0,
        },
    )


def test_an_expired_deferral_is_refused_at_its_review_date():
    membership = {
        "signed_deferrals": [
            {
                "target_id": "welshgov.council_tax_stock.by_area.band_h",
                "geography_level": "local_authority",
                "expires_on": "2027-04-01",
            }
        ]
    }
    assert_uk_local_deferrals_in_force(membership, date(2027, 4, 1))
    with pytest.raises(ValueError, match="expired"):
        assert_uk_local_deferrals_in_force(membership, date(2027, 4, 2))
