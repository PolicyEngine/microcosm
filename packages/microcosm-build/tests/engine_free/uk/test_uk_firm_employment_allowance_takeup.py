"""Opt-in Employment Allowance claim draw at HMRC's 2024-25 take-up by size."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_employment_allowance_takeup import (
    UK_FIRM_EA_LIABILITY_LIMIT_GBP,
    UK_FIRM_EA_PROXY_SECONDARY_RATE,
    UK_FIRM_EA_PROXY_SECONDARY_THRESHOLD_GBP,
    UK_FIRM_EA_TAKEUP_FILES,
    UK_FIRM_EA_TAKEUP_STATUS,
    assign_uk_firm_employment_allowance_takeup,
    draw_uk_firm_employment_allowance_takeup,
    read_uk_firm_employment_allowance_takeup,
    uk_firm_employer_nic_proxy,
    uk_firm_employment_allowance_eligible,
    uk_firm_employment_allowance_takeup_bands,
    uk_firm_employment_allowance_takeup_held_out,
)
from test_support.paths import paths_for

FIXTURE = (
    paths_for("microcosm-build").package
    / "tests"
    / "fixtures"
    / "uk"
    / "firm_employment_allowance_takeup_2024_25"
)
SOURCE_SHA256 = "3848abed310515a0882c5f59a34b0b5e2f350155965f22ff3b173b55a0ceaaf0"


@pytest.fixture(scope="module")
def tables():
    return read_uk_firm_employment_allowance_takeup(FIXTURE)


def _firms(n: int, employees, *, seed: int = 0, **overrides) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    employees = np.broadcast_to(np.asarray(employees), (n,)).astype(np.int64)
    frame = pd.DataFrame(
        {
            "firm_id": np.arange(n, dtype=np.int64) * 7 + 3,
            "firm_weight": np.full(n, 1.0),
            "firm_employees": employees,
            "firm_claims_employment_allowance": np.ones(n, dtype=bool),
            "firm_is_single_director_company": np.zeros(n, dtype=bool),
            "firm_is_public_body": np.zeros(n, dtype=bool),
            "firm_paybill": employees * 20_000.0,
            "firm_sic_section": rng.choice(["A", "C", "F", "B", "UNKNOWN"], n),
            "firm_region": rng.choice(["LONDON", "WALES", "SCOTLAND"], n),
        }
    )
    return frame.assign(**overrides)


def test_fixture_matches_provenance_and_source(tables) -> None:
    for name in UK_FIRM_EA_TAKEUP_FILES.values():
        data = (FIXTURE / name).read_bytes()
        record = json.loads((FIXTURE / f"{name}.provenance.json").read_text())
        assert record["sha256"] == SOURCE_SHA256
        assert record["published"] == "2025-06-25"
        assert record["accessed"] == "2026-11-22"
        assert record["file"]["sha256"] == hashlib.sha256(data).hexdigest()
        assert tables.sha256[name] == record["file"]["sha256"]
    assert tables.total == 1_223_000
    assert [row[:3] for row in tables.size] == [
        ("1-9", "1-9", 1_049_000.0),
        ("10-49", "10-49", 166_000.0),
        ("50-249", "50-249", 7_000.0),
        ("250+", "250+", None),
    ]
    assert len(tables.sector) == 17 and len(tables.region) == 12


def test_suppressed_marker_is_kept_not_a_number(tables) -> None:
    top = tables.size[-1]
    assert top[2] is None and top[3] == "<500"
    assert all(row[3] is None for row in tables.size[:-1])
    other = next(row for row in tables.sector if row[0] == "B,D,E,O,U")
    assert other[2] == 6_000.0


def test_bands_pool_from_fifty_and_ignore_the_suppressed_count(tables) -> None:
    bands = uk_firm_employment_allowance_takeup_bands(tables)
    assert [b.lower for b in bands] == [1, 10, 50]
    assert [b.hmrc_claimants for b in bands] == [1_049_000, 166_000, 7_000]
    assert bands[-1].pooled == ("50-249", "250+")
    assert bands[-1].ignored_suppressed == (("250+", "<500"),)
    # Pooling below a suppressed band would need a number for it: refused.
    with pytest.raises(ValueError, match="suppressed"):
        uk_firm_employment_allowance_takeup_bands(tables, pooled_from=250)
    with pytest.raises(ValueError, match="band edge"):
        uk_firm_employment_allowance_takeup_bands(tables, pooled_from=20)


def test_nic_proxy() -> None:
    threshold = UK_FIRM_EA_PROXY_SECONDARY_THRESHOLD_GBP
    rate = UK_FIRM_EA_PROXY_SECONDARY_RATE
    proxy = uk_firm_employer_nic_proxy(
        np.array([0.0, 5_000.0, 50_000.0]), np.array([0, 1, 2])
    )
    assert proxy == pytest.approx([0.0, 0.0, rate * (50_000 - 2 * threshold)])
    # One of three employees below the threshold, paid 4,000.
    with_below = uk_firm_employer_nic_proxy(
        np.array([64_000.0]), np.array([3]), np.array([4_000.0]), np.array([1])
    )
    assert with_below == pytest.approx([rate * (60_000 - 2 * threshold)])
    with pytest.raises(ValueError):
        uk_firm_employer_nic_proxy(np.ones(1), np.ones(1), np.ones(1), None)


def test_eligibility_rules() -> None:
    limit_pay = UK_FIRM_EA_LIABILITY_LIMIT_GBP / UK_FIRM_EA_PROXY_SECONDARY_RATE
    firms = pd.DataFrame(
        {
            "firm_employees": [3, 0, 3, 3, 3, 3, 1],
            "firm_claims_employment_allowance": [1, 1, 0, 1, 1, 1, 1],
            "firm_is_single_director_company": [0, 0, 0, 1, 0, 0, 0],
            "firm_is_public_body": [0, 0, 0, 0, 1, 0, 0],
            "firm_paybill": [60_000, 0, 60_000, 60_000, 60_000, 0, 0],
        }
    ).astype(
        {
            "firm_claims_employment_allowance": bool,
            "firm_is_single_director_company": bool,
            "firm_is_public_body": bool,
            "firm_paybill": float,
        }
    )
    firms.loc[5, "firm_paybill"] = 3 * UK_FIRM_EA_PROXY_SECONDARY_THRESHOLD_GBP + (
        limit_pay * 1.01
    )
    firms.loc[6, "firm_paybill"] = UK_FIRM_EA_PROXY_SECONDARY_THRESHOLD_GBP + (
        limit_pay * 0.99
    )
    parts = uk_firm_employment_allowance_eligible(firms)
    assert parts["eligible"].tolist() == [True, False, False, False, False, False, True]
    assert parts["no_employees"].tolist()[1]
    assert parts["proxy_nic_over_limit"].tolist() == [0, 0, 0, 0, 0, 1, 0]
    with pytest.raises(ValueError, match="firm_paybill"):
        uk_firm_employment_allowance_eligible(firms.drop(columns="firm_paybill"))


def test_band_totals_match_hmrc_on_a_large_frame(tables) -> None:
    # 300,000 rows; weights chosen so every band holds more weighted eligible
    # firms than HMRC's claimants.
    n = 300_000
    employees = np.repeat([1, 5, 12, 30, 60, 400], n // 6)
    firms = _firms(n, employees, firm_weight=np.repeat([30.0, 30, 8, 8, 2, 2], n // 6))
    draw = draw_uk_firm_employment_allowance_takeup(firms, tables, seed=11)
    bands = {row["band"]: row for row in draw.diagnostics["bands"]}
    for label, hmrc in (("1-9", 1_049_000), ("10-49", 166_000), ("50+", 7_000)):
        row = bands[label]
        assert not row["capped"] and row["cap_shortfall"] == 0
        assert 0 < row["propensity"] < 1
        assert row["weighted_claimants"] == pytest.approx(hmrc, rel=0.05)
    assert draw.diagnostics["hmrc_band_sum"] == 1_222_000
    assert draw.diagnostics["hmrc_total"] == 1_223_000
    assert draw.diagnostics["status"] == UK_FIRM_EA_TAKEUP_STATUS == "assumed"


def test_propensity_caps_at_one_and_reports_the_shortfall(tables) -> None:
    firms = _firms(1_000, 3)
    draw = draw_uk_firm_employment_allowance_takeup(firms, tables, seed=1)
    small = draw.diagnostics["bands"][0]
    assert small["capped"] and small["propensity"] == 1.0
    assert small["cap_shortfall"] == pytest.approx(1_049_000 - 1_000)
    assert draw.claims.all()
    empty = draw.diagnostics["bands"][-1]
    assert empty["weighted_eligible"] == 0 and empty["propensity"] == 0
    assert empty["cap_shortfall"] == 7_000
    assert ((draw.propensity >= 0) & (draw.propensity <= 1)).all()


def test_drawn_claims_are_a_subset_of_the_rule_flag(tables) -> None:
    n = 20_000
    rng = np.random.default_rng(5)
    firms = _firms(
        n,
        rng.integers(0, 80, n),
        firm_weight=np.full(n, 40.0),
        firm_claims_employment_allowance=rng.random(n) < 0.8,
        firm_is_public_body=rng.random(n) < 0.05,
    )
    out, diagnostics = assign_uk_firm_employment_allowance_takeup(firms, tables, seed=2)
    drawn = out["firm_claims_employment_allowance"].to_numpy()
    rule = firms["firm_claims_employment_allowance"].to_numpy()
    assert not (drawn & ~rule).any()
    assert not (drawn & firms["firm_is_public_body"].to_numpy()).any()
    assert diagnostics["rule_claimants_ineligible_weighted"]["public_body"] > 0
    assert diagnostics["rule_claimants_ineligible_weighted"]["no_employees"] > 0
    assert out.drop(columns="firm_claims_employment_allowance").equals(
        firms.drop(columns="firm_claims_employment_allowance")
    )


def test_draw_is_deterministic_and_row_order_independent(tables) -> None:
    n = 5_000
    firms = _firms(n, np.random.default_rng(3).integers(1, 70, n), firm_weight=300.0)
    first = draw_uk_firm_employment_allowance_takeup(firms, tables, seed=9)
    again = draw_uk_firm_employment_allowance_takeup(firms, tables, seed=9)
    assert np.array_equal(first.claims, again.claims)
    other = draw_uk_firm_employment_allowance_takeup(firms, tables, seed=10)
    assert not np.array_equal(first.claims, other.claims)
    shuffled = firms.sample(frac=1.0, random_state=4)
    drawn = draw_uk_firm_employment_allowance_takeup(shuffled, tables, seed=9)
    by_id = dict(zip(firms["firm_id"], first.claims, strict=True))
    assert [by_id[i] for i in shuffled["firm_id"]] == drawn.claims.tolist()
    with pytest.raises(ValueError, match="unique"):
        draw_uk_firm_employment_allowance_takeup(
            firms.assign(firm_id=1), tables, seed=9
        )


def test_held_out_sector_and_region_mapping(tables) -> None:
    firms = _firms(6, 3, firm_weight=[1.0, 2, 4, 8, 16, 32])
    firms["firm_sic_section"] = ["A", "B", "D", "UNKNOWN", "F", "F"]
    firms["firm_region"] = ["LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND"] + [
        "NORTH_EAST",
        "LONDON",
    ]
    claims = np.array([True, True, True, True, True, False])
    result = uk_firm_employment_allowance_takeup_held_out(firms, claims, tables)
    assert result["model_weighted_claimants"] == 31
    sector = {row["hmrc_code"]: row for row in result["sector"]["rows"]}
    assert sector["A"]["model_weighted_claimants"] == 1
    assert sector["B,D,E,O,U"]["model_weighted_claimants"] == 6
    assert sector["F"]["model_weighted_claimants"] == 16
    assert result["sector"]["unmapped_model_labels"] == {"UNKNOWN": 8.0}
    assert result["sector"]["unmapped_hmrc_codes"] == []
    region = {row["hmrc_code"]: row for row in result["region"]["rows"]}
    assert region["W99999999"]["model_weighted_claimants"] == 2
    assert region["S99999999"]["model_weighted_claimants"] == 4
    assert region["N99999999"]["model_weighted_claimants"] == 8
    assert region["E12000007"]["model_weighted_claimants"] == 1
    assert region["E12000007"]["hmrc_claimants"] == 219_000
    assert result["region"]["unmapped_model_labels"] == {}
    assert result["region"]["unmapped_hmrc_codes"] == []


def test_a_large_low_pay_employer_falls_in_the_pooled_band(tables) -> None:
    # 400 employees at the threshold: no proxy NIC, so eligible, and pooled
    # with the 50-249 band whose published count is the only one used.
    firms = _firms(
        2,
        [400, 60],
        firm_paybill=[400 * UK_FIRM_EA_PROXY_SECONDARY_THRESHOLD_GBP, 0.0],
    )
    draw = draw_uk_firm_employment_allowance_takeup(firms, tables, seed=0)
    pooled = draw.diagnostics["bands"][-1]
    assert pooled["band"] == "50+" and pooled["eligible_rows"] == 2
    assert pooled["hmrc_claimants"] == 7_000 and pooled["propensity"] == 1.0
    assert draw.claims.tolist() == [True, True]


def test_tables_are_frozen(tables) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        tables.total = 0  # type: ignore[misc]
