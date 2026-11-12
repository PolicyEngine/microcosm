"""Reference facts, TPA's constants and the stock cell declarations (microcosm#1524)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.uk_runtime import land_reference_facts as facts
from microcosm.build.uk_runtime import stock_cell_declarations as declared
from microcosm.build.uk_runtime import ucl_price_paid_epc as ucl
from microcosm.build.uk_runtime import voa_stock_tables as voa
from microcosm.build.weighted_quantiles import check_knots

_REGIONS = ", ".join(f'"E1200000{i}": 1.0{i}' for i in range(1, 10))
_CONFIG = f"""\
import not_a_module_anywhere
raise SystemExit("config.py must never run")
BASE_COST_PER_M2 = {{"D": 2100, "S": 1950, "T": 1900, "F": 2400}}  # BCIS/ABI-style
REGION_FACTOR = {{{_REGIONS}}}
KAPPA = 0.7054
"""


def test_tpa_constants_are_read_as_syntax_and_never_run() -> None:
    costs, factors = facts.parse_tpa_constants(_CONFIG)
    assert costs == {"D": 2100.0, "S": 1950.0, "T": 1900.0, "F": 2400.0}
    assert factors["E12000007"] == pytest.approx(1.07) and len(factors) == 9
    with pytest.raises(ValueError, match="lacks"):
        facts.parse_tpa_constants("BASE_COST_PER_M2 = {}\n")
    with pytest.raises(ValueError, match="twice"):
        facts.parse_tpa_constants(_CONFIG + "REGION_FACTOR = {}\n")
    with pytest.raises(ValueError, match="D, S, T and F"):
        facts.parse_tpa_constants(_CONFIG.replace('"F": 2400', '"B": 2400'))


def test_tpa_cells_give_each_authority_a_ratio_of_totals() -> None:
    payload = {
        "meta": {"kappa": 0.7054, "land_share": 0.55},
        "las": {
            "E06000001": {
                "bands": {
                    "A": {
                        "count": 3,
                        "value_deciles": [100, 300],
                        "land_deciles": [0, 100],
                    },
                    "B": {"count": 1, "value_deciles": [1000], "land_deciles": [800]},
                }
            }
        },
    }
    meta, out = facts.parse_tpa_cells(json.dumps(payload).encode())
    assert meta == {"kappa": 0.7054, "land_share": 0.55}
    row = out["E06000001"]
    assert row["value_gbp"] == 3 * 200 + 1000 and row["land_gbp"] == 3 * 50 + 800
    assert row["land_share"] == pytest.approx(950 / 1600)  # not the mean of ratios
    payload["las"]["E06000001"]["bands"]["B"]["land_deciles"] = [2000]
    with pytest.raises(ValueError, match="inconsistent"):
        facts.parse_tpa_cells(json.dumps(payload).encode())


def _fact(
    concept, code, value, *, level="local_authority", spec="", period=2026, **dims
):
    return {
        "observed_measure": {"source_concept": concept},
        "geography": {"level": level, "id": code},
        "layout": {"record_set_spec_id": spec},
        "period": {"value": period},
        "dimensions": dims,
        "value": value,
        "aggregate_fact_key": f"key:{concept}:{code}:{period}",
    }


def test_read_feed_facts_parses_only_marked_rows(tmp_path: Path) -> None:
    path = tmp_path / "consumer_facts.jsonl"
    marked = _fact(
        "mhclg.residential_site_density",
        "E06000001",
        40,
        site_density_scenario="medium",
    )
    path.write_text(
        json.dumps(marked) + "\n" + "{not json, and not marked}\n", encoding="utf-8"
    )
    assert facts.read_feed_facts(path) == [marked]


def _land_facts() -> list[dict]:
    concept = declared.MHCLG_SHEETS["floor_adjusted"]
    return [
        _fact(
            concept,
            "E06000001",
            2_000_000,
            new_build_price_percentile="p50",
            site_density_scenario="medium",
        ),
        _fact(
            concept,
            "E06000001",
            3_000_000,
            new_build_price_percentile="p75",
            site_density_scenario="medium",
        ),
        _fact(
            concept,
            "E06000002",
            1_000_000,
            new_build_price_percentile="p50",
            site_density_scenario="medium",
        ),
        _fact(
            "mhclg.residential_site_density",
            "E06000001",
            40,
            site_density_scenario="medium",
        ),
        _fact(
            "mhclg.residential_site_density",
            "E06000002",
            25,
            site_density_scenario="medium",
        ),
    ]


def test_land_per_new_home_is_value_over_density() -> None:
    concept = declared.MHCLG_SHEETS["floor_adjusted"]
    central = facts.land_per_new_home(
        _land_facts(), concept=concept, density="medium", percentile=50
    )
    assert central == {"E06000001": 50_000.0, "E06000002": 40_000.0}
    denser = facts.land_per_new_home(
        _land_facts(),
        concept=concept,
        density="medium",
        percentile=50,
        density_factor=1.25,
    )
    assert denser["E06000001"] == pytest.approx(40_000.0)
    assert facts.land_per_new_home(
        _land_facts(), concept=concept, density="medium", percentile=75
    ) == {"E06000001": 75_000.0}
    with pytest.raises(ValueError, match="no density"):
        facts.land_per_new_home(
            _land_facts()[:3], concept=concept, density="medium", percentile=50
        )


def test_market_values_band_d_and_tax_base_read_their_tables() -> None:
    rows = [
        _fact(
            "mhclg.mvs_market_value_mean",
            "E12000001",
            221_000,
            level="region",
            spec=declared.MVS_REGION_SPEC + "full.v1",
            ehs_tenure="owner_occupiers",
        ),
        _fact(
            "mhclg.mvs_market_value_mean",
            "E12000001",
            93_000,
            level="region",
            spec=declared.MVS_REGION_SPEC + "full.v1",
            ehs_tenure="all_social_renters",
        ),
        _fact(
            "mhclg.mvs_market_value_mean",
            "E12000001",
            175_000,
            level="region",
            spec=declared.MVS_REGION_SPEC + "full.v1",
        ),
        _fact(
            "mhclg.council_tax.average_band_d_billing_authority_area",
            "E07000223",
            2548.28,
            spec=declared.BAND_D_ENGLAND_SPEC + "v1",
        ),
        _fact(
            "mhclg.council_tax.average_band_d_billing_authority_area",
            "E07000223",
            0,
            spec="uk.local_geography.council_tax.levels_england.fy2025.v1",
        ),
        _fact(
            "welshgov.council_tax.average_band_d",
            "W06000001",
            2260.73,
            spec=declared.BAND_D_WALES_SPEC + "v1",
        ),
        _fact(
            "mhclg.council_tax.council_tax_base",
            "E07000223",
            22736.19,
            spec=declared.TAX_BASE_ENGLAND_SPEC + "v1",
        ),
    ]
    assert facts.market_value_means(rows, spec_prefix=declared.MVS_REGION_SPEC) == {
        "E12000001": {
            "owner_occupiers": 221_000,
            "all_social_renters": 93_000,
            "all": 175_000,
        }
    }
    assert facts.band_d_by_authority(
        rows,
        england_spec=declared.BAND_D_ENGLAND_SPEC,
        wales_spec=declared.BAND_D_WALES_SPEC,
    ) == {"E07000223": 2548.28, "W06000001": 2260.73}
    assert facts.tax_base_by_authority(rows, spec=declared.TAX_BASE_ENGLAND_SPEC) == {
        "E07000223": 22736.19
    }


def test_epc_mean_floor_area_counts_a_recoded_council_once() -> None:
    spec = declared.EPC_D3_SPEC + "2025_q4.v1"
    rows = [
        _fact(
            "mhclg.epc_domestic_total_floor_area",
            "E08000016",
            1000.0,
            spec=spec,
            period="2025-Q4",
        ),
        _fact(
            "mhclg.epc_domestic_total_floor_area",
            "E08000038",
            1000.0,
            spec=spec,
            period="2025-Q4",
        ),
        _fact(
            "mhclg.epc_domestic_lodgements",
            "E08000016",
            10,
            spec=spec,
            period="2025-Q4",
        ),
        _fact(
            "mhclg.epc_domestic_lodgements",
            "E08000038",
            10,
            spec=spec,
            period="2025-Q4",
        ),
        _fact(
            "mhclg.epc_domestic_total_floor_area",
            "E08000038",
            900.0,
            spec=spec,
            period="2026-Q1",
        ),
        _fact(
            "mhclg.epc_domestic_lodgements",
            "E08000038",
            10,
            spec=spec,
            period="2026-Q1",
        ),
    ]
    means = facts.epc_mean_floor_area(
        rows, spec_prefix=declared.EPC_D3_SPEC, code_aliases={"E08000038": "E08000016"}
    )
    assert means == {"E08000016": pytest.approx(1900.0 / 20)}


def test_level_inputs_resolve_feed_then_parameter_then_absent() -> None:
    assert facts.resolve_level_input(
        "e46y", feed_value=2.5e12, feed_key="k", parameter=None
    ) == {"name": "e46y", "status": "feed", "value": 2.5e12, "key": "k"}
    assert (
        facts.resolve_level_input(
            "e46y", feed_value=None, feed_key=None, parameter=2.5e12
        )["status"]
        == "parameter"
    )
    assert (
        facts.resolve_level_input(
            "e46y", feed_value=None, feed_key=None, parameter=None
        )["value"]
        is None
    )
    with pytest.raises(ValueError, match="drop the parameter"):
        facts.resolve_level_input(
            "e46y", feed_value=2.5e12, feed_key="k", parameter=2.4e12
        )
    assert facts.feed_inputs_fingerprint(
        {"a": 1, "b": [1]}
    ) == facts.feed_inputs_fingerprint({"b": [1], "a": 1})


# ----------------------------------------------------------------------------
# The declarations


def test_the_declarations_are_consistent() -> None:
    assert declared.POOLING.levels[0] == "msoa_band"
    assert {declared.POOLING.minimum("means"), declared.POOLING.minimum("knots")} == {
        30,
        100,
    }
    grid = np.array(declared.KNOT_GRID)
    check_knots(grid, grid * declared.KNOT_CAP, cap=declared.KNOT_CAP)
    assert len(grid) == 13
    bands = declared.EPC_AGE_BANDS
    assert all(b[1] == a[2] + 1 for a, b in zip(bands, bands[1:], strict=False))
    assert tuple(declared.BAND_D_RATIOS) == declared.BANDS == voa.BANDS
    assert declared.PROPERTY_TYPES == voa.PRICE_PAID_TYPES
    for family in (
        declared.STRUCTURE_VARIANTS,
        declared.VALUE_VARIANTS,
        declared.MHCLG_VARIANTS,
        declared.DECISION_RULE,
        declared.MEASUREMENTS,
    ):
        names = [entry.get("name", entry.get("id")) for entry in family]
        assert len(names) == len(set(names)) and all(names)
    structure = {v["name"] for v in declared.STRUCTURE_VARIANTS}
    (rank,) = [r for r in declared.DECISION_RULE if r["id"] == "rank_stability"]
    assert set(rank["variants"]) <= structure
    assert {"central", "kappa_1"} <= structure
    for profile in declared.DEPRECIATION_PROFILES.values():
        assert profile["shape"] == "none" or 0 < profile["cap"] <= 1
    # UCL's column declarations build the reader's own declarations.
    linked = ucl.LinkedColumns(**declared.UCL_LINKED_COLUMNS)
    assert linked.has_address
    ucl.CertificateColumns(**declared.UCL_CERTIFICATE_COLUMNS)
    assert declared.UCL_COLUMNS_VERIFIED_AGAINST is None  # until discover runs
