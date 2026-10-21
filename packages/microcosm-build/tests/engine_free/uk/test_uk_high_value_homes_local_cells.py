"""The high-value homes area cells: the constituency deferral and the LA floor.

microcosm#1301: the family's region rows control it on the local line, so the
local reference generator defers its constituency level whole and binds its
local-authority cells down to the signed ``hvcts_la_household_floor`` rule.
These tests run on the committed contract, crosswalk and rule register and on
synthetic membership reports; the regenerated references and their counts are
pinned with the other rules in test_uk_local_small_cell_rule.py.
"""

from __future__ import annotations

import copy
import json
from importlib import resources as importlib_resources

import pytest

from microcosm.build.uk_runtime.ledger_targets import uk_small_cell_rule_is_signed
from microcosm.build.uk_runtime.local_targets import load_uk_population_contract
from tools.generate_uk_local_target_references import (
    HIGH_VALUE_HOMES_LOCAL_TARGET_IDS,
    LOCAL_GEOGRAPHY_LEVELS,
    _area_scope_by_target_id,
    _area_signed_deferrals,
    _filter_contract_by_geography_levels,
    _small_cell_deferrals,
)

RULE_ID = "hvcts_la_household_floor"
DEFERRAL_REASON_ID = "high_value_homes_constituency_grain_deferred"


def _committed(resource: str) -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(resource)
        .read_text(encoding="utf-8")
    )


def _crosswalk() -> dict:
    return _committed("local_area_crosswalk.json")


def _local_contract() -> dict:
    # What the generator authors from: the targets with a local level.
    return _filter_contract_by_geography_levels(
        load_uk_population_contract(), allowed_levels=LOCAL_GEOGRAPHY_LEVELS
    )


def _rule() -> dict:
    (rule,) = (
        rule
        for rule in _committed("local_small_cell_rule.json")["rules"]
        if rule["rule_id"] == RULE_ID
    )
    return rule


def test_the_constituency_level_is_deferred_whole() -> None:
    contract = _local_contract()
    crosswalk = _crosswalk()
    declarations = _area_signed_deferrals(contract, crosswalk)
    england_and_wales = tuple(
        area_id
        for area_id in crosswalk["levels"]["constituency"]["area_ids"]
        if area_id[:1] in {"E", "W"}
    )
    assert len(england_and_wales) == 575
    rows = [row for row in declarations if row.reason_id == DEFERRAL_REASON_ID]
    assert [(row.target_id, row.geography_level, row.area_ids) for row in rows] == [
        (target_id, "constituency", england_and_wales)
        for target_id in HIGH_VALUE_HOMES_LOCAL_TARGET_IDS
    ]
    # Every candidate the target has at the level: its area scope is England
    # and Wales, so the deferral leaves no constituency cell to bind.
    scopes = _area_scope_by_target_id(contract, crosswalk)
    for target_id in HIGH_VALUE_HOMES_LOCAL_TARGET_IDS:
        assert scopes[target_id]["constituency"] == frozenset(england_and_wales)
    for row in rows:
        # The cells compile from the derived facts, so the deferral must
        # override a compiled cell; a hand-written deferral carries no expiry,
        # its time bound being the family's binding adjudication.
        assert row.defer_if_compiles is True
        assert row.expires_on is None
        assert RULE_ID in row.rationale
        assert "microcosm.high_value_homes.over_*_by_region" in row.rationale
        assert "microcosm#1301" in row.rationale
    # The local-authority cells stay bound but for the two support-floor areas;
    # the small-cell rule defers the rest in the generator's second pass.
    assert {
        row.reason_id
        for row in declarations
        if row.target_id in HIGH_VALUE_HOMES_LOCAL_TARGET_IDS
        and row.geography_level == "local_authority"
    } == {"local_authority_support_floor_excluded"}


def test_the_constituency_mask_refuses_roster_drift() -> None:
    crosswalk = copy.deepcopy(_crosswalk())
    area_ids = crosswalk["levels"]["constituency"]["area_ids"]
    area_ids.remove(next(area_id for area_id in area_ids if area_id[:1] == "W"))
    with pytest.raises(
        ValueError, match=r"high-value homes.*expected 575.*measured 574"
    ):
        _area_signed_deferrals(_local_contract(), crosswalk)


def test_the_floor_is_signed_at_400_homes_to_april_2027() -> None:
    rule = _rule()
    assert uk_small_cell_rule_is_signed(rule)
    assert rule["target_id_prefixes"] == ["microcosm.high_value_homes."]
    assert rule["absolute_floor"] == 400
    assert rule["approved_by"] == "juaristi22"
    assert rule["approved_on"] == "2026-10-16"
    assert rule["status"] == "approved"
    assert rule["expires_on"] == "2027-04-01"
    assert rule["tracking"] == "microcosm#1301"
    assert DEFERRAL_REASON_ID in rule["reason"]


def test_the_floor_reaches_only_the_three_local_targets() -> None:
    # The prefix also names the region rows, but they bind at country and
    # region level, so they are no local target and the generator never sees
    # them; no other local family shares the prefix.
    (prefix,) = _rule()["target_id_prefixes"]
    local = {
        target["target_id"]
        for target in _local_contract()["targets"]
        if target["target_id"].startswith(prefix)
    }
    assert local == set(HIGH_VALUE_HOMES_LOCAL_TARGET_IDS)
    regions = {
        target["target_id"]
        for target in load_uk_population_contract()["targets"]
        if target["target_id"].startswith(prefix)
    } - local
    assert regions == {f"{target_id}_by_region" for target_id in local}


def _candidate(area: str, value: float, status: str = "active", **extra) -> dict:
    return {"geography_id": area, "status": status, "resolved_value": value, **extra}


def test_the_floor_defers_the_active_local_authority_cells_below_400() -> None:
    # A synthetic first-pass report: the constituency cells are already
    # signed-deferred level-wide, and one authority is a support-floor area.
    # The committed register's other rules name other targets.
    deferred = {"status": "signed_deferred"}
    report = {
        "targets": {
            "microcosm.high_value_homes.over_2m": {
                "geography_levels": {
                    "constituency": {
                        "candidates": [
                            _candidate(
                                "PCON_SMALL",
                                12.0,
                                **deferred,
                                signed_reason_id=DEFERRAL_REASON_ID,
                            ),
                            _candidate(
                                "PCON_LARGE",
                                9000.0,
                                **deferred,
                                signed_reason_id=DEFERRAL_REASON_ID,
                            ),
                        ]
                    },
                    "local_authority": {
                        "candidates": [
                            _candidate("LA_ZERO", 0.0),
                            _candidate("LA_BELOW", 399.9),
                            _candidate("LA_AT_FLOOR", 400.0),
                            _candidate("LA_LARGE", 9000.0),
                            _candidate(
                                "LA_SUPPORT",
                                3.0,
                                **deferred,
                                signed_reason_id="local_authority_support_floor_excluded",
                            ),
                        ]
                    },
                }
            },
            "ons.census.households": {
                "geography_levels": {
                    "local_authority": {"candidates": [_candidate("LA_BELOW", 3.0)]}
                }
            },
        }
    }
    (deferral,) = _small_cell_deferrals(report)
    assert deferral.target_id == "microcosm.high_value_homes.over_2m"
    assert deferral.geography_level == "local_authority"
    assert deferral.area_ids == ("LA_BELOW", "LA_ZERO")
    assert deferral.reason_id == RULE_ID
    assert deferral.defer_if_compiles is True
    assert deferral.expires_on == "2027-04-01"
    assert "below 400" in deferral.rationale
    assert "approved by juaristi22 on 2026-10-16" in deferral.rationale
