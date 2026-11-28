"""UK firm VAT composition by SIC section from the Supply and Use Tables 2024."""

from __future__ import annotations

import hashlib
import json
from functools import cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_consumption import UKFirmSupplyUseSource
from microcosm.build.uk_runtime.firm_vat_composition import (
    UK_FIRM_INPUT_RATIO_BOUNDS,
    UK_FIRM_VAT_COMPOSITION_FILES,
    UK_FIRM_VAT_COMPOSITION_RULE_COLUMNS,
    UK_FIRM_VAT_COMPOSITION_SHARE_COLUMNS,
    UK_FIRM_VAT_SUPPLY_COLUMNS,
    UK_FIRM_VAT_TREATMENTS,
    UKFirmVATCompositionRules,
    UKFirmVATCompositionSource,
    add_uk_firm_vat_composition_columns,
    compose_uk_firm_vat_supplies,
    read_uk_firm_vat_composition_source,
    rescale_uk_firm_input_costs,
    uk_firm_input_ratio_factors,
    uk_firm_vat_product_shares,
    uk_firm_vat_relief_checks,
    uk_firm_vat_section_shares,
    uk_firm_vat_voluntary_registration,
)
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
FIXTURES = PATHS.tests / "fixtures" / "uk"
FIXTURE = FIXTURES / "firm_vat_composition_2024"
SUPPLY_USE = FIXTURES / "firm_supply_use_2024"


@cache
def _source():
    return read_uk_firm_vat_composition_source(FIXTURE, SUPPLY_USE)


@cache
def _shares() -> pd.DataFrame:
    return uk_firm_vat_section_shares(_source())


def test_provenance_pins_every_file_and_both_workbooks():
    provenance = json.loads((FIXTURE / "provenance.json").read_text())
    for name in UK_FIRM_VAT_COMPOSITION_FILES:
        digest = hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest()
        assert provenance["files"][name]["sha256"] == digest
    assert provenance["sources"]["sut"]["sha256"].startswith("716c9e02")
    assert provenance["sources"]["abs"]["sha256"].startswith("9e548c5c")
    assert {
        x["direct_url"].split("/")[-2] for x in provenance["sources"]["legislation"]
    } >= {
        "7A",
        "8",
        "9",
    }


def test_reader_refuses_a_changed_file(tmp_path):
    for name in (*UK_FIRM_VAT_COMPOSITION_FILES, "provenance.json"):
        (tmp_path / name).write_bytes((FIXTURE / name).read_bytes())
    path = tmp_path / "sut_dwellings_gfcf.csv"
    path.write_text(path.read_text().replace("107010", "107011"))
    with pytest.raises(ValueError, match="sha256"):
        read_uk_firm_vat_composition_source(tmp_path, SUPPLY_USE)


def test_treatment_rows_are_labelled_and_sum_to_one():
    table = _source().treatment
    assert set(table["status"]) == {"statute", "assumed"}
    assert table["basis"].str.len().min() > 10
    numeric = table[~table["standard"].isin(["DWELLINGS", "GOODS"])]
    sums = numeric[list(UK_FIRM_VAT_TREATMENTS)].astype(float).sum(axis=1)
    assert np.allclose(sums, 1.0)
    computed = table[table["standard"].isin(["DWELLINGS", "GOODS"])]
    assert set(computed["status"]) == {"assumed"}


def test_section_shares_are_shares_and_match_the_reviewed_values():
    shares = _shares()
    assert np.allclose(shares[list(UK_FIRM_VAT_TREATMENTS)].sum(axis=1), 1.0)
    assert (
        (
            (shares[list(UK_FIRM_VAT_TREATMENTS)] >= 0)
            & (shares[list(UK_FIRM_VAT_TREATMENTS)] <= 1)
        )
        .all()
        .all()
    )
    lo, hi = UK_FIRM_INPUT_RATIO_BOUNDS
    assert shares["input_ratio_target"].between(lo, hi).all()
    # Values reviewed at gate B1 (27 Nov 2026).
    assert shares.loc["K", "exempt"] == pytest.approx(0.5628, abs=1e-4)
    assert shares.loc["D", "reduced"] == pytest.approx(0.2136, abs=1e-4)
    assert shares.loc["A", "zero"] == pytest.approx(0.8142, abs=1e-4)
    assert shares.loc["G", "input_ratio_target"] == pytest.approx(0.8612, abs=1e-4)
    assert shares.loc["G", "input_vat_rate"] == pytest.approx(0.1669, abs=1e-4)
    assert shares.loc["G", "standard"] == pytest.approx(0.6911, abs=1e-4)
    # Exports of hotel and restaurant services are non-residents' spending here.
    assert shares.loc["I", "standard"] == pytest.approx(1.0)
    assert shares.loc["P", "zero"] == pytest.approx(0.0)
    assert shares.loc["O", "exempt"] == pytest.approx(0.9923, abs=1e-4)


def test_section_g_ratio_is_on_the_gross_sales_base():
    source = _source()
    abs_g = source.abs_turnover.set_index("sic_section").loc[
        "G", "turnover_gbp_million"
    ]
    assert abs_g == 1793666
    # ABS 2024 purchases / turnover for G, a cross-check of the SUT-based base.
    purchases = source.abs_turnover.set_index("sic_section").loc[
        "G", "purchases_gbp_million"
    ]
    assert purchases / abs_g == pytest.approx(
        _shares().loc["G", "input_ratio_target"], abs=0.03
    )


def test_compose_splits_turnover_and_keeps_out_of_scope_rows_exempt():
    firms = pd.DataFrame(
        {
            "firm_turnover": [1_000_000.0, 500_000.0],
            "firm_input_costs": [400_000.0, 100_000.0],
            "firm_sic_section": ["K", "C"],
        }
    )
    out = compose_uk_firm_vat_supplies(firms, _shares(), np.array([True, False]))
    total = sum(out[c] for c in UK_FIRM_VAT_SUPPLY_COLUMNS.values())
    assert np.allclose(total, firms["firm_turnover"])
    assert out.loc[0, "firm_exempt_supplies"] == pytest.approx(
        1e6 * _shares().loc["K", "exempt"]
    )
    assert out.loc[1, "firm_exempt_supplies"] == 500_000.0
    assert out.loc[1, "firm_standard_rated_supplies"] == 0.0
    assert out.loc[0, "firm_input_vat"] == pytest.approx(
        400_000 * _shares().loc["K", "input_vat_rate"]
    )
    with pytest.raises(ValueError, match="sections"):
        compose_uk_firm_vat_supplies(
            firms.assign(firm_sic_section=["K", "U"]), _shares()
        )


def test_voluntary_registration_keeps_registered_rows_below_threshold():
    firms = pd.DataFrame(
        {
            "firm_standard_rated_supplies": [50_000.0, 100_000.0, 10_000.0],
            "firm_reduced_rated_supplies": [0.0, 0.0, 0.0],
            "firm_zero_rated_supplies": [10_000.0, 0.0, 0.0],
        }
    )
    got = uk_firm_vat_voluntary_registration(
        firms, np.array([True, True, False]), 90_000.0
    )
    assert got.tolist() == [True, False, False]


def test_input_ratio_factors_hit_the_target_and_keep_the_bounds():
    rng = np.random.default_rng(0)
    ratio = 0.2 + rng.beta(4, 2, 2000) * 0.6
    turnover = rng.lognormal(12, 1, 2000)
    sections = np.array(["G"] * 1000 + ["M"] * 1000)
    targets = pd.Series({"G": 0.86, "M": 0.39})
    factors = uk_firm_input_ratio_factors(sections, ratio, turnover, targets)
    assert factors["G"]["achieved"] == pytest.approx(0.86, abs=1e-9)
    assert factors["M"]["achieved"] == pytest.approx(0.39, abs=1e-9)
    assert factors["G"]["k"] > 1 > factors["M"]["k"]
    new, again = rescale_uk_firm_input_costs(
        turnover, ratio * turnover, sections, targets
    )
    assert again == factors
    lo, hi = UK_FIRM_INPUT_RATIO_BOUNDS
    assert ((new / turnover >= lo - 1e-12) & (new / turnover <= hi + 1e-12)).all()
    zero, _ = rescale_uk_firm_input_costs(
        np.array([0.0]), np.array([0.0]), np.array(["G"]), targets
    )
    assert zero.tolist() == [0.0]


class _FakeSimulation:
    """VAT liability with pe-uk's formula: output VAT less input VAT x rounded recovery."""

    def __init__(self, table: pd.DataFrame, period: int) -> None:
        self.table = table
        assert list(table.columns) == list(UK_FIRM_VAT_COMPOSITION_RULE_COLUMNS)

    def calculate(self, variable: str, period: int):
        t = self.table
        if variable == "firm_id":
            return t["firm_id"].to_numpy()
        taxable = (
            t["firm_standard_rated_supplies"]
            + t["firm_reduced_rated_supplies"]
            + t["firm_zero_rated_supplies"]
        )
        share = (
            np.ceil(taxable / (taxable + t["firm_exempt_supplies"]) * 100 - 1e-6) / 100
        )
        output = (
            0.2 * t["firm_standard_rated_supplies"]
            + 0.05 * t["firm_reduced_rated_supplies"]
        )
        return (output - t["firm_input_vat"] * share).to_numpy()


def test_rules_hand_zero_and_reduced_supplies_to_policyengine_uk():
    candidates = pd.DataFrame(
        {
            "firm_id": [1, 2],
            "firm_turnover": [1e6, 1e6],
            "firm_input_costs": [5e5, 5e5],
            "firm_employees": [5, 5],
            "firm_sic_section": ["K", "C"],
            "firm_standard_rated_supplies": [1e6, 1e6],
            "firm_exempt_supplies": [0.0, 0.0],
            "firm_input_vat": [1e5, 1e5],
        }
    )
    rules = UKFirmVATCompositionRules(_shares(), simulation_factory=_FakeSimulation)
    got = rules.evaluate(candidates, period=2024).vat_liability_if_registered
    k = _shares().loc["K"]
    # Recovery share for section K: (1 - exempt) = 0.4372, rounded up to 0.44.
    share = np.ceil((1 - k["exempt"]) * 100 - 1e-6) / 100
    assert share == 0.44
    hand_k = 0.2 * k["standard"] * 1e6 - k["input_vat_rate"] * 5e5 * share
    c = _shares().loc["C"]
    hand_c = 0.2 * c["standard"] * 1e6 - c["input_vat_rate"] * 5e5
    assert got == pytest.approx([hand_k, hand_c], rel=1e-12)


def test_relief_checks_are_reported_not_targets():
    checks = uk_firm_vat_relief_checks(_source())["ratio"].round(3).to_dict()
    # Every ratio is pinned, the poor ones too; none is a target.
    assert checks == {
        "NS7 food (zero)": 1.044,
        "NS1 printed matter (zero)": 1.563,
        "NS2 children's clothing (zero)": 0.866,
        "NS3 new dwellings (zero)": 0.892,
        "NS4 domestic fuel and power (reduced)": 0.999,
        "NS5 domestic passenger transport (zero)": 0.72,
        "NS6 prescription drugs (zero)": 0.0,
        "NS16 water and sewerage (zero)": 0.846,
    }
    costs = _source().relief_costs.set_index("code")["cost_2024_25_gbp_million"]
    assert costs["VAT - NS7"] == 26300 and costs["VAT - NS16"] == 2900


def _copy_fixture(tmp_path):
    for name in (*UK_FIRM_VAT_COMPOSITION_FILES, "provenance.json"):
        (tmp_path / name).write_bytes((FIXTURE / name).read_bytes())


def _rewrite_treatment(tmp_path, edit):
    path = tmp_path / "vat_product_treatment.csv"
    table = pd.read_csv(path, dtype=str)
    table = edit(table)
    table.to_csv(path, index=False, lineterminator="\n")
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance["files"]["vat_product_treatment.csv"]["sha256"] = digest
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))


def _set(column, value):
    def edit(table):
        table.loc[0, column] = value
        return table

    return edit


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_set("user", "tourists"), "user must be"),
        (_set("standard", "0.5"), "sum to 1"),
        (_set("status", "guess"), "status"),
        (lambda table: table.drop(index=0), "Every SUT product"),
        (lambda table: pd.concat([table, table.iloc[[1]]]), "repeats"),
    ],
)
def test_reader_refuses_a_malformed_treatment_table(tmp_path, edit, message):
    _copy_fixture(tmp_path)
    _rewrite_treatment(tmp_path, edit)
    with pytest.raises(ValueError, match=message):
        read_uk_firm_vat_composition_source(tmp_path, SUPPLY_USE)


def _toy_source() -> UKFirmVATCompositionSource:
    """Three products by hand: food (A), a service (M) and retail (G47)."""

    uses = pd.DataFrame(
        {
            "intermediate_consumption": [20.0, 30.0, 0.0],
            "households": [60.0, 10.0, 0.0],
            "npish": 0.0,
            "central_government": 0.0,
            "local_government": 0.0,
            "gfcf": 0.0,
            "valuables": 0.0,
            "changes_in_inventories": 0.0,
            "exports": [20.0, 60.0, 0.0],
            "total_demand": [100.0, 100.0, 0.0],
            "domestic_output": [80.0, 100.0, 40.0],
            "distributors_margins": [40.0, 0.0, -40.0],
        },
        index=pd.Index(["CPA_A01", "CPA_M70", "CPA_G47"], name="product_code"),
    )
    treatment = pd.DataFrame(
        [
            ("CPA_A01", "all", "0.25", "0", "0.75", "0", "toy", "assumed"),
            ("CPA_M70", "all", "1", "0", "0", "0", "toy", "statute"),
            ("CPA_G47", "all", "GOODS", "0", "GOODS", "0", "toy", "assumed"),
        ],
        columns=["product_code", "user", *UK_FIRM_VAT_TREATMENTS, "basis", "status"],
    )
    intermediate = pd.DataFrame(
        {
            "product_code": ["CPA_A01", "CPA_M70", "CPA_M70"],
            "industry_product_code": ["CPA_M70", "CPA_M70", "CPA_A01"],
            "value_gbp_million": [10.0, 30.0, 20.0],
        }
    )
    divisions = pd.DataFrame(
        {
            "product_code": ["CPA_A01", "CPA_M70", "CPA_G47"],
            "sic_division": [1, 70, 47],
            "sic_section": ["A", "M", "G"],
        }
    )
    supply_use = UKFirmSupplyUseSource(
        uses=uses, hhfce=pd.DataFrame(), divisions=divisions, sha256={}
    )
    abs_turnover = pd.DataFrame(
        {
            "sic_section": ["G", "45", "46", "47"],
            "turnover_gbp_million": [200.0, 0.0, 0.0, 200.0],
        }
    )
    return UKFirmVATCompositionSource(
        treatment=treatment,
        intermediate=intermediate,
        dwellings_gfcf_gbp_million=0.0,
        abs_turnover=abs_turnover,
        relief_costs=pd.DataFrame(),
        supply_use=supply_use,
        sha256={},
    )


def test_product_and_section_shares_by_hand():
    source = _toy_source()
    products = uk_firm_vat_product_shares(source)
    # Food: 80 of 100 sold at home at 0.25/0.75, 20 exported zero-rated.
    assert products.loc["CPA_A01", "zero"] == pytest.approx((80 * 0.75 + 20) / 100)
    assert products.loc["CPA_A01", "export_share"] == pytest.approx(0.2)
    # Service: 40 at home standard, 60 exported zero-rated.
    assert products.loc["CPA_M70", "standard"] == pytest.approx(0.4)
    # Retail resells household goods (food only here): the food household mix.
    assert products.loc["CPA_G47", "zero"] == pytest.approx(0.75)
    # Industry M70 buys 10 of food at 0.2 x 0.25 and 30 of M at 0.2.
    assert products.loc["CPA_M70", "input_vat_rate"] == pytest.approx(
        (10 * 0.05 + 30 * 0.2) / 40
    )
    assert products.loc["CPA_A01", "input_vat_rate"] == pytest.approx(0.2)
    sections = uk_firm_vat_section_shares(source)
    assert sections.loc["M", "input_ratio_target"] == pytest.approx(40 / 100)
    # Section G: gross sales 200, margin 40, no IC, so the ratio is
    # 1 - 40 / 200; goods resold are priced at food's business rate 0.2 x 0.25.
    assert sections.loc["G", "input_ratio_target"] == pytest.approx(0.8)
    assert sections.loc["G", "input_vat_rate"] == pytest.approx(0.05)


def test_rescale_leaves_sections_without_a_target_alone():
    turnover = np.array([100.0, 100.0, 0.0])
    inputs = np.array([50.0, 5.0, 3.0])
    new, factors = rescale_uk_firm_input_costs(
        turnover, inputs, np.array(["G", "Z", "G"]), pd.Series({"G": 0.6})
    )
    assert new[0] == pytest.approx(60.0)
    assert new[1] == 5.0 and new[2] == 3.0
    assert factors["Z"]["rescaled"] == 0.0


def test_columns_only_mode_adds_shares_and_leaves_scored_columns_alone():
    firms = pd.DataFrame(
        {
            "firm_id": [1, 2],
            "firm_turnover": [1e6, 2e5],
            "firm_input_costs": [6e5, 1e5],
            "firm_sic_section": ["G", "Q"],
            "firm_standard_rated_supplies": [1e6, 0.0],
            "firm_exempt_supplies": [0.0, 2e5],
            "firm_input_vat": [1.2e5, 2e4],
        }
    )
    out = add_uk_firm_vat_composition_columns(firms, _shares())
    pd.testing.assert_frame_equal(out[firms.columns], firms)
    assert set(out.columns) - set(firms.columns) == set(
        UK_FIRM_VAT_COMPOSITION_SHARE_COLUMNS.values()
    )
    assert out.loc[1, "firm_sut_exempt_share"] == pytest.approx(
        _shares().loc["Q", "exempt"]
    )
    with pytest.raises(ValueError, match="already"):
        add_uk_firm_vat_composition_columns(out, _shares())


def test_input_ratio_bounds_follow_the_generator_clamp():
    from microcosm.build.uk_runtime import firm_generation

    assert UK_FIRM_INPUT_RATIO_BOUNDS == (
        firm_generation.INPUT_RATIO_LOWER,
        firm_generation.INPUT_RATIO_UPPER,
    )
