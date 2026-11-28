"""UK firm R&D: an opt-in allocation of HMRC R&D tax credit claims."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from functools import cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_rd_tax_credits import (
    UK_FIRM_RD_FILES,
    UK_FIRM_RD_SECTIONS,
    UK_FIRM_RD_STATUS,
    UKFirmRDSource,
    assign_uk_firm_rd_tax_credits,
    draw_uk_firm_rd_tax_credits,
    read_uk_firm_rd_source,
    uk_firm_rd_expenditure_table,
)
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
FIXTURE = PATHS.tests / "fixtures" / "uk" / "firm_rd_tax_credits_2024_25"
TOOL = PATHS.repository / "tools" / "extract_uk_firm_rd_tax_credits_tables.py"


@cache
def _source() -> UKFirmRDSource:
    return read_uk_firm_rd_source(FIXTURE)


@cache
def _tool():
    name = "extract_uk_firm_rd_tax_credits_tables"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _firms(rows_per_section: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sections = np.repeat(np.array(UK_FIRM_RD_SECTIONS), rows_per_section)
    n = sections.size
    employees = np.floor(rng.lognormal(2.0, 1.5, n))
    return pd.DataFrame(
        {
            "firm_id": np.arange(n) * 7 + 11,
            "firm_weight": rng.uniform(5.0, 60.0, n),
            "firm_sic_section": sections,
            "firm_employees": employees,
            "firm_turnover": 80_000.0 * (1.0 + employees) * rng.uniform(0.5, 2.0, n),
            "ons_legal_status": rng.choice([1, 1, 1, 2, 3], n),
        }
    )


def test_source_reads_published_2024_25_totals() -> None:
    source = _source()
    manufacturing = source.sections["C"]
    assert manufacturing.claims == 11_090
    assert manufacturing.expenditure_gbp == 14_195e6
    assert source.sections["L"].claims == 40
    assert source.unknown_claims == 95
    assert source.unknown_expenditure_gbp == 135e6
    assert source.national_claims == 40_325
    assert source.national_companies == 37_545
    assert source.national_expenditure_gbp["all_schemes"] == 50_970e6
    # Independent rounding to 5: the sections plus unknown miss RD1/RD4 by 5.
    claims = sum(s.claims for s in source.sections.values()) + source.unknown_claims
    spend = sum(s.expenditure_gbp for s in source.sections.values())
    assert claims == pytest.approx(source.national_claims, abs=5)
    assert abs(spend + source.unknown_expenditure_gbp - 50_970e6) <= 10e6


def test_provenance_hashes_match_fixture() -> None:
    record = json.loads((FIXTURE / "provenance.json").read_text(encoding="utf-8"))
    assert record["accessed"] == "2026-11-23"
    assert "none was filled from memory" in record["method"].lower()
    assert record["source"]["sha256"] == (
        "efc2238695c40dc4273acc8846f9057bbe582ff04c951c75085c32632c004324"
    )
    for name in UK_FIRM_RD_FILES:
        payload = (FIXTURE / name).read_bytes()
        assert record["files"][name]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_extract_table_rejects_moved_header() -> None:
    tool = _tool()
    spec = tool.RD4
    rows = [
        (spec[2] + ", 2016-17 to 2024-25 (£ million)",),
        tuple(reversed(spec[3])),
    ]
    with pytest.raises(ValueError, match="header row"):
        tool.extract_table(rows, spec)


def test_extract_table_keeps_published_text() -> None:
    tool = _tool()
    spec = tool.RD6
    labels = [f"{letter}. Sector {letter}" for letter in tool.RD6_SECTIONS]
    body = [(label, "1,234", "<5", "N/A", None) for label in labels]
    rows = [
        (spec[2],),
        spec[3],
        *body,
        (tool.RD6_UNKNOWN, 5.0),
        ("End of worksheet",),
    ]
    out = tool.extract_table(rows, spec)
    assert out[0][:6] == ["A. Sector A", "1,234", "<5", "N/A", "", ""]
    assert out[-1][:2] == [tool.RD6_UNKNOWN, "5"]


def test_expenditure_table_rakes_to_rd6_and_rd4() -> None:
    source = _source()
    table = uk_firm_rd_expenditure_table(source)
    for letter, spec in source.sections.items():
        assert table.loc[letter].sum() == pytest.approx(spec.expenditure_gbp)
    sections = sum(s.expenditure_gbp for s in source.sections.values())
    large_share = 30_245e6 / (30_245e6 + 20_725e6)
    assert table["large"].sum() == pytest.approx(sections * large_share)
    assert source.national_large_expenditure_gbp == 30_245e6
    assert source.national_sme_expenditure_gbp == 20_725e6
    assert source.sections["C"].large_claims == 1_100 + 215
    # Suppressed large cells count as 0: section L has no large claims.
    assert source.sections["L"].large_claims == 0
    assert table.at["L", "large"] == 0


def test_strata_and_expenditure_match_targets() -> None:
    firms = _firms(rows_per_section=1_000)
    firms["firm_turnover"] *= 100.0
    draw = draw_uk_firm_rd_tax_credits(firms, _source(), seed=7)
    d = draw.diagnostics
    weight = firms["firm_weight"].to_numpy()
    assert d["status"] == UK_FIRM_RD_STATUS
    assert d["companies_per_claim"] == pytest.approx(37_545 / 40_325)
    for row in d["sections"]:
        for stratum in ("large", "sme"):
            if not row[f"{stratum}_draw_capped"]:
                assert (
                    abs(
                        row[f"{stratum}_weighted_claimants"]
                        - row[f"{stratum}_target_companies"]
                    )
                    <= 30.0
                )
            assert row[f"{stratum}_model_expenditure_gbp"] + row[
                f"{stratum}_unallocated_gbp"
            ] == pytest.approx(row[f"{stratum}_target_expenditure_gbp"])
    assert d["unallocated_expenditure_gbp"] == pytest.approx(0.0, abs=1.0)
    assert d["model_large_expenditure_gbp"] + d[
        "model_sme_expenditure_gbp"
    ] == pytest.approx(d["hmrc_section_expenditure_gbp"])
    company = firms["ons_legal_status"].to_numpy() == 1
    assert not draw.claimant[~company].any()
    assert (draw.qualifying_rd_expenditure[~draw.claimant] == 0).all()
    assert float(np.dot(weight, draw.qualifying_rd_expenditure)) == pytest.approx(
        d["model_expenditure_gbp"]
    )


def test_expenditure_never_exceeds_turnover_and_remainder_is_reported() -> None:
    firms = _firms()
    draw = draw_uk_firm_rd_tax_credits(firms, _source(), seed=7)
    turnover = firms["firm_turnover"].to_numpy()
    assert (draw.qualifying_rd_expenditure <= turnover * (1 + 1e-12)).all()
    d = draw.diagnostics
    assert d["unallocated_expenditure_gbp"] > 0
    assert d["rows_at_turnover_cap"] > 0
    assert d["model_expenditure_gbp"] + d["unallocated_expenditure_gbp"] == (
        pytest.approx(d["hmrc_section_expenditure_gbp"])
    )


def test_large_stratum_is_taken_top_down_by_headcount() -> None:
    firms = _firms(rows_per_section=1_000)
    draw = draw_uk_firm_rd_tax_credits(firms, _source(), seed=7)
    employees = firms["firm_employees"].to_numpy()
    large = draw.claimant & ~draw.is_sme
    for letter in ("C", "J", "M"):
        pool = (firms["firm_sic_section"] == letter).to_numpy() & (
            firms["ons_legal_status"].to_numpy() == 1
        )
        in_large = pool & large
        assert in_large.any()
        assert employees[in_large].min() >= employees[pool & ~in_large].max()


def test_small_pool_is_capped_and_reported() -> None:
    firms = _firms(rows_per_section=40)
    draw = draw_uk_firm_rd_tax_credits(firms, _source(), seed=7)
    assert "C:sme" in draw.diagnostics["draw_capped_strata"]
    in_c = (firms["firm_sic_section"] == "C") & (firms["ons_legal_status"] == 1)
    assert draw.claimant[in_c.to_numpy()].all()


def test_draw_is_row_order_invariant_and_seeded() -> None:
    firms = _firms()
    first = assign_uk_firm_rd_tax_credits(firms, _source(), seed=11)
    shuffled = firms.sample(frac=1.0, random_state=0)
    second = assign_uk_firm_rd_tax_credits(shuffled, _source(), seed=11)
    pd.testing.assert_frame_equal(first, second.loc[first.index])
    other = assign_uk_firm_rd_tax_credits(firms, _source(), seed=12)
    assert not np.array_equal(
        first["firm_qualifying_rd_expenditure"], other["firm_qualifying_rd_expenditure"]
    )


def test_sme_flag_follows_stratum_then_headcount_limb() -> None:
    firms = _firms()
    out = assign_uk_firm_rd_tax_credits(firms, _source(), seed=1)
    draw = draw_uk_firm_rd_tax_credits(firms, _source(), seed=1)
    employees = firms["firm_employees"].to_numpy()
    rest = ~draw.claimant
    assert (out["firm_is_sme"].to_numpy()[rest] == (employees[rest] < 500)).all()
    large = draw.claimant & ~draw.is_sme
    assert large.any() and (draw.claimant & draw.is_sme).any()
    assert not out["firm_is_sme"].to_numpy()[large].any()
    edge = firms.copy()
    edge.loc[edge.index[:3], "ons_legal_status"] = 2
    edge.loc[edge.index[:3], "firm_employees"] = [499.0, 500.0, 5_000.0]
    flags = assign_uk_firm_rd_tax_credits(edge, _source(), seed=1)["firm_is_sme"]
    assert flags.iloc[:3].tolist() == [True, False, False]
    assert out.drop(columns=["firm_is_sme", "firm_qualifying_rd_expenditure"]).equals(
        firms
    )


def test_rejects_duplicate_ids_and_missing_columns() -> None:
    firms = _firms(rows_per_section=5)
    with pytest.raises(ValueError, match="firm_turnover"):
        draw_uk_firm_rd_tax_credits(
            firms.drop(columns="firm_turnover"), _source(), seed=1
        )
    firms.loc[1, "firm_id"] = firms.loc[0, "firm_id"]
    with pytest.raises(ValueError, match="one row per firm_id"):
        draw_uk_firm_rd_tax_credits(firms, _source(), seed=1)
