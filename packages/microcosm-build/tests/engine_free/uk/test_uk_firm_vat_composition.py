"""UK firm VAT composition by SIC section from the Supply and Use Tables 2024."""

from __future__ import annotations

import hashlib
import json
from functools import cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_vat_composition import (
    UK_FIRM_INPUT_RATIO_BOUNDS,
    UK_FIRM_VAT_COMPOSITION_FILES,
    UK_FIRM_VAT_COMPOSITION_RULE_COLUMNS,
    UK_FIRM_VAT_SUPPLY_COLUMNS,
    UK_FIRM_VAT_TREATMENTS,
    UKFirmVATCompositionRules,
    compose_uk_firm_vat_supplies,
    read_uk_firm_vat_composition_source,
    rescale_uk_firm_input_costs,
    uk_firm_input_ratio_factors,
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
    assert shares.loc["G", "input_vat_rate"] == pytest.approx(0.1462, abs=1e-4)
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
    checks = uk_firm_vat_relief_checks(_source())
    assert checks.loc["NS7 food (zero)", "ratio"] == pytest.approx(1.044, abs=1e-3)
    assert checks.loc[
        "NS4 domestic fuel and power (reduced)", "ratio"
    ] == pytest.approx(0.999, abs=1e-3)
    assert checks.loc["NS6 prescription drugs (zero)", "model_base_gbp_bn"] == 0.0
