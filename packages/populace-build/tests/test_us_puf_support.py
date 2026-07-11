"""US PUF support-channel expansion tests."""

import importlib

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.puf_support as puf_support_module
from populace.build.us_runtime import (
    BASE_ASEC_SUPPORT_CHANNEL,
    CPS_CARRIED_FORMULA_OWNED_COLUMNS,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    derive_us_cps_carried_inputs,
    impute_us_puf_tax_detail_support,
    puf_tax_unit_donor_from_arrays,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from populace.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS,
    assert_formula_owned_blocklist_current,
    resolve_formula_owned_outputs,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _minimal_us_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 10, 20], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
            "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
            "person_marital_unit_id": np.asarray([10000, 10000, 20000], dtype="int64"),
            "employment_income_before_lsr": np.asarray(
                [50_000, 20_000, 125_000],
                dtype="int64",
            ),
            "partnership_income": [1_000.0, 2_000.0, 3_000.0],
            "s_corp_income": [4_000.0, 5_000.0, 6_000.0],
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype="int64"),
                "state_fips": np.asarray([6, 36], dtype="int64"),
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([10, 20], dtype="int64"),
                "filing_status_input": ["JOINT", "SINGLE"],
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
        "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
        "marital_unit": pd.DataFrame({"marital_unit_id": np.asarray([10000, 20000])}),
    }
    strata = pd.Series(
        ["asec_2024", "asec_2024", "asec_2023"],
        name="stratum",
    )
    weights = {
        "household": Weights(
            values=np.asarray([100.0, 300.0]),
            kind=WeightKind.DESIGN,
        )
    }
    return Frame(tables, US_SCHEMA, weights, strata)


def _raw_asec_predictor_frame() -> Frame:
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.asarray([1, 2, 3], dtype="int64"),
                "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
                "person_tax_unit_id": np.asarray([10, 10, 20], dtype="int64"),
                "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
                "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
                "person_marital_unit_id": np.asarray(
                    [10000, 10000, 20000], dtype="int64"
                ),
                "A_AGE": [65, 40, 61],
                "A_SEX": [1, 2, 2],
                "WSAL_VAL": [100.0, 200.0, 0.0],
                "SEMP_VAL": [10.0, 0.0, 30.0],
                "INT_VAL": [1_000.0, 0.0, 200.0],
                "DIV_VAL": [100.0, 0.0, 50.0],
                "CAP_VAL": [50.0, 0.0, 25.0],
                "SS_VAL": [1_000.0, 900.0, 800.0],
                "RESNSS1": [1, 2, 0],
                "RESNSS2": [0, 0, 0],
                "PNSN_VAL": [1_000.0, 0.0, 0.0],
                "ANN_VAL": [100.0, 0.0, 0.0],
                "DST_SC1": [4, 3, 0],
                "DST_VAL1": [500.0, 200.0, 0.0],
                "NOW_MRK": [1, 2, 1],
                "NOW_NONM": [2, 1, 2],
                "NOW_MCAID": [1, 2, 2],
                "NOW_GRP": [2, 1, 1],
                "NOW_CHAMPVA": [2, 2, 1],
                "NOW_MIL": [1, 2, 2],
                "NOW_VACARE": [2, 1, 2],
                "NOW_OTHMT": [2, 1, 2],
                "NOW_IHSFLG": [1, 2, 2],
                "RNT_VAL": [20.0, 0.0, 0.0],
                "FRSE_VAL": [5.0, 0.0, 0.0],
                "UC_VAL": [0.0, 70.0, 0.0],
                "OI_OFF": [19, 0, 0],
                "OI_VAL": [3.0, 0.0, 0.0],
                "PHIP_VAL": [400.0, 0.0, 50.0],
                "PEMCPREM": [100.0, 0.0, 25.0],
                "PMED_VAL": [200.0, 0.0, 40.0],
                "POTC_VAL": [30.0, 0.0, 10.0],
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype="int64"),
                "state_fips": np.asarray([6, 36], dtype="int64"),
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([10, 20], dtype="int64"),
                "filing_status_input": ["JOINT", "SINGLE"],
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
        "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
        "marital_unit": pd.DataFrame({"marital_unit_id": np.asarray([10000, 20000])}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray([100.0, 300.0]), WeightKind.DESIGN)},
    )


def test_puf_support_channel_doubles_rows_without_doubling_mass() -> None:
    frame = _minimal_us_frame()

    expanded = clone_us_frame_for_puf_support(frame)

    for entity in frame.entities:
        assert expanded.n(entity) == 2 * frame.n(entity)
    assert expanded.weights_for("household").kind == WeightKind.DESIGN
    assert (
        expanded.weights_for("household").total == frame.weights_for("household").total
    )
    assert expanded.weights_for("household").values.tolist() == [
        50.0,
        150.0,
        50.0,
        150.0,
    ]
    assert expanded.strata.tolist() == frame.strata.tolist() + frame.strata.tolist()


def test_puf_support_channel_preserves_provenance_and_remaps_linked_ids() -> None:
    expanded = clone_us_frame_for_puf_support(_minimal_us_frame())

    person = expanded.table("person")
    tax_unit = expanded.table("tax_unit")
    puf_people = person[
        person[support_channel_column("person")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]
    puf_tax_units = tax_unit[
        tax_unit[support_channel_column("tax_unit")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]

    assert person[support_channel_column("person")].tolist() == [
        BASE_ASEC_SUPPORT_CHANNEL,
        BASE_ASEC_SUPPORT_CHANNEL,
        BASE_ASEC_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    ]
    assert person[support_clone_index_column("person")].tolist() == [0, 0, 0, 1, 1, 1]
    assert person[support_source_id_column("person")].tolist() == [1, 2, 3, 1, 2, 3]
    assert set(puf_people["person_tax_unit_id"]).issubset(
        set(puf_tax_units["tax_unit_id"])
    )
    assert set(puf_people["person_tax_unit_id"]).isdisjoint(
        set(
            tax_unit.loc[
                tax_unit[support_channel_column("tax_unit")] == "asec", "tax_unit_id"
            ]
        )
    )
    assert puf_people["employment_income_before_lsr"].tolist() == [
        50_000.0,
        20_000.0,
        125_000.0,
    ]
    assert puf_tax_units[support_source_id_column("tax_unit")].tolist() == [10, 20]


def test_puf_support_channel_refuses_duplicate_or_missing_puf_channel() -> None:
    frame = _minimal_us_frame()

    with pytest.raises(ValueError, match="must be unique"):
        clone_us_frame_for_puf_support(frame, channels=("asec", "asec"))

    with pytest.raises(ValueError, match="must start with 'asec'"):
        clone_us_frame_for_puf_support(frame, channels=("puf_tax_detail", "tail"))

    with pytest.raises(ValueError, match="must include 'puf_tax_detail'"):
        clone_us_frame_for_puf_support(frame, channels=("asec", "tail"))


def test_puf_support_channel_refuses_to_run_twice() -> None:
    expanded = clone_us_frame_for_puf_support(_minimal_us_frame())

    with pytest.raises(ValueError, match="should run exactly once"):
        clone_us_frame_for_puf_support(expanded)


def test_puf_tax_unit_donor_from_arrays_aggregates_person_values() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 10, 20],
            "employment_income": [5.0, 7.0, 11.0],
            "non_qualified_dividend_income": [1.0, 2.0, 3.0],
            "qualified_dividend_income": [4.0, 5.0, 6.0],
            "home_mortgage_interest": [10.0, 20.0, 30.0],
            "educator_expense": [100.0, 200.0, 300.0],
            "social_security": [100.0, 200.0, 300.0],
            "taxable_unemployment_compensation": [13.0, 17.0, 19.0],
            "state_and_local_sales_or_income_tax": [40.0, 50.0],
        },
        person_outputs=(
            "employment_income_before_lsr",
            "qualified_dividend_income",
            "non_qualified_dividend_income",
            "home_mortgage_interest",
            "educator_expense",
            "social_security_retirement",
            "social_security_disability",
            "social_security_dependents",
            "social_security_survivors",
            "unemployment_compensation",
        ),
        tax_unit_outputs=(),
    )

    assert donor["employment_income_before_lsr"].tolist() == [12.0, 11.0]
    assert "employment_income" not in donor
    assert donor["qualified_dividend_income"].tolist() == [9.0, 6.0]
    assert donor["non_qualified_dividend_income"].tolist() == [3.0, 3.0]
    assert donor["puf_predictor_dividend_income"].tolist() == [12.0, 9.0]
    assert donor["social_security_retirement"].tolist() == [300.0, 300.0]
    assert donor["social_security_disability"].tolist() == [0.0, 0.0]
    assert donor["social_security_dependents"].tolist() == [0.0, 0.0]
    assert donor["social_security_survivors"].tolist() == [0.0, 0.0]
    assert "social_security" not in donor
    assert donor["unemployment_compensation"].tolist() == [30.0, 19.0]
    assert donor["home_mortgage_interest"].tolist() == [30.0, 30.0]
    assert donor["educator_expense"].tolist() == [300.0, 300.0]
    assert "interest_deduction" not in donor
    assert "state_withheld_income_tax" not in donor
    assert donor["puf_predictor_employment_income"].tolist() == [12.0, 11.0]
    assert donor["puf_predictor_filing_status_code"].tolist() == [1.0, 2.0]


def test_puf_tax_detail_default_person_outputs_are_engine_leaves() -> None:
    assert "qualified_dividend_income" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "non_qualified_dividend_income" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "dividend_income" not in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "ordinary_dividend_income" not in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "social_security_retirement" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "social_security_disability" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "social_security_dependents" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "social_security_survivors" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "social_security" not in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "employment_income_before_lsr" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "employment_income" not in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "self_employment_income_before_lsr" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "self_employment_income" not in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "tax_exempt_interest_income" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "unemployment_compensation" not in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert (
        "long_term_capital_gains_before_response"
        in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    )
    assert "long_term_capital_gains" not in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "taxable_private_pension_income" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "taxable_pension_income" not in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "qualified_tuition_expenses" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "educator_expense" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "casualty_loss" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert (
        "unreimbursed_business_employee_expenses"
        in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    )
    assert "medical_expense_deduction" not in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert "interest_deduction" not in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    assert "domestic_production_ald" in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    assert "first_home_mortgage_interest" in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    assert "second_home_mortgage_interest" in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    assert "first_home_mortgage_balance" in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    assert "second_home_mortgage_balance" in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    assert (
        "first_home_mortgage_origination_year"
        in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    )
    assert (
        "second_home_mortgage_origination_year"
        in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    )
    # Issue #278: the deduction bases the sparse release zeroed must ride the
    # donor as engine input leaves (the desired-contribution inputs), never
    # as the formula-owned realized amounts or ALD aggregates.
    assert (
        "traditional_ira_contributions_desired" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    )
    assert "traditional_ira_contributions" not in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert (
        "self_employed_pension_contributions_desired"
        in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    )
    assert (
        "self_employed_pension_contributions"
        not in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    )
    assert (
        "self_employed_pension_contribution_ald"
        not in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    )
    assert (
        "self_employed_pension_contribution_ald"
        not in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    )
    assert "health_savings_account_ald" in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS


def test_puf_tax_unit_donor_derives_qualified_tuition_from_raw_fields() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 10, 20],
            "E03230": [500.0, 2_000.0, -10.0],
            "E87530": [1_000.0, 1_500.0, 4_000.0],
        },
        person_outputs=("qualified_tuition_expenses",),
        tax_unit_outputs=(),
    )

    assert donor["qualified_tuition_expenses"].tolist() == [3_000.0, 4_000.0]


def test_puf_tax_unit_donor_carries_person_educator_expense() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 10, 20],
            "educator_expense": [250.0, 50.0, 0.0],
        },
        person_outputs=("educator_expense",),
        tax_unit_outputs=(),
    )

    assert donor["educator_expense"].tolist() == [300.0, 0.0]


def test_puf_tax_unit_donor_carries_person_casualty_losses() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 10, 20],
            "casualty_loss": [1_000.0, 2_000.0, 4_000.0],
        },
        person_outputs=("casualty_loss",),
        tax_unit_outputs=(),
    )

    assert donor["casualty_loss"].tolist() == [3_000.0, 4_000.0]


def test_puf_tax_unit_donor_carries_both_person_alimony_leaves() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 10, 20],
            "alimony_income": [1_000.0, 2_000.0, 4_000.0],
            "alimony_expense": [500.0, 700.0, 3_000.0],
        },
        person_outputs=("alimony_income", "alimony_expense"),
        tax_unit_outputs=(),
    )

    assert donor["alimony_income"].tolist() == [3_000.0, 4_000.0]
    assert donor["alimony_expense"].tolist() == [1_200.0, 3_000.0]


def test_puf_tax_unit_donor_carries_person_misc_itemized_expenses() -> None:
    output = "unreimbursed_business_employee_expenses"
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 10, 20],
            output: [1_000.0, 2_000.0, 4_000.0],
        },
        person_outputs=(output,),
        tax_unit_outputs=(),
    )

    assert donor[output].tolist() == [3_000.0, 4_000.0]


def test_puf_tax_unit_donor_carries_ald_contribution_leaves() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 10, 20],
            "traditional_ira_contributions": [1_000.0, 2_000.0, 3_000.0],
            "self_employed_pension_contribution_ald": [4_000.0, 5_000.0],
            "health_savings_account_ald": [600.0, 700.0],
        },
        person_outputs=(
            "traditional_ira_contributions_desired",
            "self_employed_pension_contributions_desired",
        ),
        tax_unit_outputs=("health_savings_account_ald",),
    )

    assert donor["traditional_ira_contributions_desired"].tolist() == [
        3_000.0,
        3_000.0,
    ]
    assert donor["self_employed_pension_contributions_desired"].tolist() == [
        4_000.0,
        5_000.0,
    ]
    assert donor["health_savings_account_ald"].tolist() == [600.0, 700.0]


def test_puf_tax_detail_refuses_self_employed_ald_aggregates_as_outputs() -> None:
    arrays = {
        "tax_unit_id": [10],
        "household_weight": [100.0],
        "filing_status": [b"SINGLE"],
        "person_tax_unit_id": [10],
        "self_employed_pension_contribution_ald": [4_000.0],
    }

    with pytest.raises(ValueError, match="formula-owned aggregate outputs"):
        puf_tax_unit_donor_from_arrays(
            arrays,
            person_outputs=("self_employed_pension_contribution_ald",),
            tax_unit_outputs=(),
        )


def test_puf_tax_detail_refuses_formula_owned_outputs() -> None:
    arrays = {
        "tax_unit_id": [10],
        "household_weight": [100.0],
        "filing_status": [b"SINGLE"],
        "person_tax_unit_id": [10],
        "home_mortgage_interest": [10.0],
    }

    with pytest.raises(ValueError, match="formula-owned aggregate outputs"):
        puf_tax_unit_donor_from_arrays(
            arrays,
            person_outputs=(),
            tax_unit_outputs=("interest_deduction",),
        )

    expanded = clone_us_frame_for_puf_support(_minimal_us_frame())
    donor = pd.DataFrame(
        {
            "puf_predictor_filing_status_code": [1.0, 2.0, 4.0, 1.0],
            "puf_predictor_tax_unit_person_count": [1.0, 2.0, 1.0, 2.0],
            "interest_deduction": [100.0, 100.0, 100.0, 100.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="formula-owned aggregate outputs"):
        impute_us_puf_tax_detail_support(
            expanded,
            donor,
            predictors=(
                "puf_predictor_filing_status_code",
                "puf_predictor_tax_unit_person_count",
            ),
            person_outputs=(),
            tax_unit_outputs=("interest_deduction",),
            n_estimators=4,
            seed=0,
        )


class _FakeFormulaOwnedEngine:
    """Minimal metadata source for the formula-owned guard (issue #301).

    Reports exactly the names in ``formula_owned`` as formula-owned, restricted
    to the requested set — the contract
    :func:`resolve_formula_owned_outputs` and
    :func:`assert_formula_owned_blocklist_current` depend on. Injecting it keeps
    these tests deterministic whether or not ``policyengine_us`` happens to be
    installed in the test environment.
    """

    def __init__(self, formula_owned: set[str]) -> None:
        self._formula_owned = set(formula_owned)

    def formula_owned_outputs(self, names) -> set[str]:
        return set(names) & self._formula_owned


def test_resolve_formula_owned_outputs_unions_static_seed_with_engine() -> None:
    # #301: a name the engine reports as formula-owned is rejected even though
    # it is absent from the static seed set — the derived source keeps the guard
    # current as PolicyEngine-US adds variables, with no edit to the seed set.
    assert "income_tax" not in PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS
    engine = _FakeFormulaOwnedEngine({"income_tax"})
    requested = {
        "income_tax",  # engine-only (not in static seed)
        "interest_deduction",  # static seed only (engine below does not flag it)
        "employment_income_before_lsr",  # legitimate leaf input
    }

    rejected = resolve_formula_owned_outputs(requested, engine=engine)

    assert rejected == {"income_tax", "interest_deduction"}


def test_resolve_formula_owned_outputs_always_applies_static_seed() -> None:
    # Even when the engine flags nothing, the static seed is always enforced, so
    # the guard still rejects known formula-owned aggregates when metadata is
    # unavailable.
    empty_engine = _FakeFormulaOwnedEngine(set())
    requested = {"interest_deduction", "employment_income_before_lsr"}

    assert resolve_formula_owned_outputs(requested, engine=empty_engine) == {
        "interest_deduction",
    }


def test_assert_formula_owned_blocklist_current_passes_when_engine_agrees() -> None:
    # Reverse-direction check: every static-seed entry the engine still reports
    # as formula-owned is fine — no drift, no error.
    agreeing_engine = _FakeFormulaOwnedEngine(set(PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS))
    assert_formula_owned_blocklist_current(agreeing_engine)


def test_assert_formula_owned_blocklist_current_flags_stale_entries() -> None:
    # A static-seed entry the engine no longer treats as formula-owned is stale
    # (e.g. the engine turned it into a plain input or renamed it away) and must
    # be surfaced by name so it cannot silently linger and wrongly reject a
    # legitimate leaf.
    stale = next(iter(PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS))
    still_owned = set(PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS) - {stale}
    drifted_engine = _FakeFormulaOwnedEngine(still_owned)

    with pytest.raises(ValueError, match=stale):
        assert_formula_owned_blocklist_current(drifted_engine)


def test_resolve_formula_owned_outputs_engine_none_falls_back_to_static() -> None:
    # With no engine passed and metadata unavailable, resolution falls back to
    # the static seed. The fallback is exercised by monkeypatching the lazy
    # engine resolver to report no engine, so the test is deterministic even
    # where policyengine_us is installed.
    puf_support_module._formula_owned_engine = lambda: None
    try:
        requested = {"interest_deduction", "employment_income_before_lsr"}
        assert resolve_formula_owned_outputs(requested) == {"interest_deduction"}
    finally:
        importlib.reload(puf_support_module)


class _ImportErrorEngine:
    """An adapter whose lazy policyengine_us import is missing at call time."""

    def formula_owned_outputs(self, names):
        raise ImportError("No module named 'policyengine_us'")


def test_resolve_formula_owned_outputs_degrades_on_engine_import_error() -> None:
    # The adapter module ships with populace-frame, so construction succeeds
    # even without policyengine_us installed; the missing [us] extra surfaces
    # as an ImportError at call time (the CI environment). That must degrade
    # to the static seed exactly like having no engine, not abort the build.
    requested = {"interest_deduction", "employment_income_before_lsr"}

    rejected = resolve_formula_owned_outputs(requested, engine=_ImportErrorEngine())

    assert rejected == {"interest_deduction"}


def test_blocklist_current_check_noops_on_engine_import_error() -> None:
    # Same degradation for the reverse-direction check: a call-time
    # ImportError means no metadata is available, so the check is a no-op
    # rather than a build failure.
    assert_formula_owned_blocklist_current(_ImportErrorEngine())


def test_resolve_formula_owned_outputs_catches_engine_output_off_static_list() -> None:
    # #301, against the real engine: a genuinely formula-owned output that is NOT
    # on the static seed set (income_tax) is still rejected, and every legitimate
    # leaf input passes through. This is the failure a stale hand-written
    # blocklist would silently allow.
    pytest.importorskip("policyengine_us")

    requested = {
        "income_tax",  # formula-owned, deliberately not in the static seed
        "employment_income_before_lsr",  # leaf input
        "qualified_dividend_income",  # leaf input
    }
    assert "income_tax" not in PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS

    rejected = resolve_formula_owned_outputs(requested)

    assert "income_tax" in rejected
    assert "employment_income_before_lsr" not in rejected
    assert "qualified_dividend_income" not in rejected


def test_static_seed_is_subset_of_engine_derived_formula_owned_set() -> None:
    # #301, against the real engine: the static seed set is a SUBSET of the set
    # the engine derives as formula-owned, so it never diverges into rejecting a
    # name the engine treats as an input. Equivalently, the drift check passes
    # against live metadata.
    pytest.importorskip("policyengine_us")

    from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

    engine = PolicyEngineUSEngine()
    engine_derived = engine.formula_owned_outputs(PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS)

    assert PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS <= set(engine_derived)
    # And the guard's own consistency check agrees.
    assert_formula_owned_blocklist_current(engine)


def test_cps_carried_derivations_create_leaf_inputs_not_aggregates() -> None:
    frame = derive_us_cps_carried_inputs(_raw_asec_predictor_frame())

    person = frame.table("person")
    assert not CPS_CARRIED_FORMULA_OWNED_COLUMNS.intersection(person.columns)
    assert person["age"].tolist() == [65.0, 40.0, 61.0]
    assert person["is_female"].tolist() == [False, True, True]
    assert person["employment_income_before_lsr"].tolist() == [100.0, 200.0, 0.0]
    assert person["self_employment_income_before_lsr"].tolist() == [10.0, 0.0, 30.0]
    assert person["taxable_interest_income"].tolist() == [680.0, 0.0, 136.0]
    assert "tax_exempt_interest_income" not in person
    np.testing.assert_allclose(
        person["qualified_dividend_income"].to_numpy(),
        [44.8, 0.0, 22.4],
    )
    np.testing.assert_allclose(
        person["non_qualified_dividend_income"].to_numpy(),
        [55.2, 0.0, 27.6],
    )
    np.testing.assert_allclose(
        person["long_term_capital_gains_before_response"].to_numpy(),
        [44.0, 0.0, 22.0],
    )
    np.testing.assert_allclose(
        person["short_term_capital_gains"].to_numpy(),
        [6.0, 0.0, 3.0],
    )
    assert person["social_security_retirement"].tolist() == [1_000.0, 0.0, 0.0]
    assert person["social_security_disability"].tolist() == [0.0, 900.0, 800.0]
    assert person["taxable_private_pension_income"].tolist() == [649.0, 0.0, 0.0]
    assert person["taxable_ira_distributions"].tolist() == [500.0, 0.0, 0.0]
    assert person["rental_income"].tolist() == [20.0, 0.0, 0.0]
    assert person["farm_income"].tolist() == [5.0, 0.0, 0.0]
    assert person["unemployment_compensation"].tolist() == [0.0, 70.0, 0.0]
    assert person["alimony_income"].tolist() == [0.0, 0.0, 0.0]
    assert person["miscellaneous_income"].tolist() == [3.0, 0.0, 0.0]
    assert person["health_insurance_premiums_without_medicare_part_b"].tolist() == [
        400.0,
        0.0,
        50.0,
    ]
    assert person["medicare_part_b_premiums"].tolist() == [100.0, 0.0, 25.0]
    assert person["other_medical_expenses"].tolist() == [200.0, 0.0, 40.0]
    assert person["over_the_counter_health_expenses"].tolist() == [30.0, 0.0, 10.0]
    assert person["has_marketplace_health_coverage_at_interview"].tolist() == [
        True,
        False,
        True,
    ]
    assert person["has_marketplace_health_coverage"].tolist() == [
        True,
        False,
        True,
    ]
    assert person[
        "has_non_marketplace_direct_purchase_health_coverage_at_interview"
    ].tolist() == [
        False,
        True,
        False,
    ]
    assert person["has_medicaid_health_coverage_at_interview"].tolist() == [
        True,
        False,
        False,
    ]
    assert person["has_esi"].tolist() == [False, True, True]
    assert person["has_champva_health_coverage_at_interview"].tolist() == [
        False,
        False,
        True,
    ]
    assert person["has_tricare_health_coverage_at_interview"].tolist() == [
        True,
        False,
        False,
    ]
    assert person["has_va_health_coverage_at_interview"].tolist() == [
        False,
        True,
        False,
    ]
    assert person["has_other_means_tested_health_coverage_at_interview"].tolist() == [
        False,
        True,
        False,
    ]
    assert person["has_indian_health_service_coverage_at_interview"].tolist() == [
        True,
        False,
        False,
    ]


def test_cps_carried_derivations_reject_formula_owned_input_columns() -> None:
    frame = _raw_asec_predictor_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["dividend_income"] = [1.0, 2.0, 3.0]
    bad = Frame(tables, frame.schema, {"household": frame.weights_for("household")})

    with pytest.raises(ValueError, match="formula-owned aggregate columns"):
        derive_us_cps_carried_inputs(bad)


def test_cps_carried_derives_spm_unit_childcare_from_replicated_asec_column() -> None:
    frame = _raw_asec_predictor_frame()
    frame.table("person")["SPM_CHILDCAREXPNS"] = [800.0, 800.0, 0.0]

    derived = derive_us_cps_carried_inputs(frame)

    assert derived.table("spm_unit")[
        "spm_unit_pre_subsidy_childcare_expenses"
    ].tolist() == [800.0, 0.0]


def test_cps_carried_preserves_existing_spm_unit_childcare_values() -> None:
    frame = _raw_asec_predictor_frame()
    frame.table("person")["SPM_CHILDCAREXPNS"] = [800.0, 800.0, 0.0]
    frame.table("spm_unit")["spm_unit_pre_subsidy_childcare_expenses"] = [
        111.0,
        222.0,
    ]

    derived = derive_us_cps_carried_inputs(frame)

    assert derived.table("spm_unit")[
        "spm_unit_pre_subsidy_childcare_expenses"
    ].tolist() == [111.0, 222.0]


def test_cps_carried_derivations_unblock_default_puf_predictors() -> None:
    expanded = clone_us_frame_for_puf_support(
        derive_us_cps_carried_inputs(_raw_asec_predictor_frame())
    )
    donor = pd.DataFrame(
        {
            "puf_predictor_filing_status_code": [1.0, 2.0, 4.0, 1.0],
            "puf_predictor_tax_unit_person_count": [1.0, 2.0, 1.0, 2.0],
            "puf_predictor_employment_income": [100.0, 200.0, 300.0, 400.0],
            "puf_predictor_self_employment_income": [10.0, 20.0, 30.0, 40.0],
            "puf_predictor_taxable_interest_income": [1.0, 2.0, 3.0, 4.0],
            "puf_predictor_dividend_income": [5.0, 6.0, 7.0, 8.0],
            "puf_predictor_short_term_capital_gains": [13.0, 14.0, 15.0, 16.0],
            "puf_predictor_long_term_capital_gains": [17.0, 18.0, 19.0, 20.0],
            "taxable_interest_income": [100.0, 200.0, 300.0, 400.0],
            "qualified_dividend_income": [10.0, 20.0, 30.0, 40.0],
            "non_qualified_dividend_income": [50.0, 60.0, 70.0, 80.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        person_outputs=(
            "taxable_interest_income",
            "qualified_dividend_income",
            "non_qualified_dividend_income",
        ),
        tax_unit_outputs=(),
        n_estimators=4,
        seed=0,
    )

    puf_people = imputed.table("person")[
        imputed.table("person")[support_channel_column("person")]
        == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]
    assert {"taxable_interest_income", "qualified_dividend_income"}.issubset(
        puf_people.columns
    )


def test_puf_tax_detail_snaps_sparse_taxable_interest_to_observed_zero(
    monkeypatch,
) -> None:
    class TinyPositiveQRF:
        def __init__(self, *, n_estimators: int, seed: int) -> None:
            pass

        def fit(
            self,
            frame,
            predictors,
            outputs,
            *,
            weights,
        ) -> "TinyPositiveQRF":
            return self

        def predict(self, features: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "taxable_interest_income": [0.25, 9.6],
                    "qualified_dividend_income": [0.25, 9.6],
                },
                index=features.index,
            )

    monkeypatch.setattr(puf_support_module, "QRF", TinyPositiveQRF)
    expanded = clone_us_frame_for_puf_support(
        derive_us_cps_carried_inputs(_raw_asec_predictor_frame())
    )
    donor = pd.DataFrame(
        {
            "puf_predictor_filing_status_code": [1.0, 2.0],
            "puf_predictor_tax_unit_person_count": [1.0, 1.0],
            "taxable_interest_income": [0.0, 10.0],
            "qualified_dividend_income": [0.0, 10.0],
            "weight": [1.0, 1.0],
        }
    )

    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=(
            "puf_predictor_filing_status_code",
            "puf_predictor_tax_unit_person_count",
        ),
        person_outputs=(
            "taxable_interest_income",
            "qualified_dividend_income",
        ),
        tax_unit_outputs=(),
        n_estimators=4,
        seed=0,
    )

    puf_people = imputed.table("person")[
        imputed.table("person")[support_channel_column("person")]
        == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]
    assert puf_people["taxable_interest_income"].tolist() == [0.0, 0.0, 10.0]
    assert puf_people["qualified_dividend_income"].tolist() == [0.25, 0.0, 9.6]


def test_puf_tax_detail_preserves_sparse_educator_rate_and_earnings_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EducatorExpenseQRF:
        def __init__(self, *, n_estimators: int, seed: int) -> None:
            pass

        def fit(
            self,
            frame,
            predictors,
            outputs,
            *,
            weights,
        ) -> "EducatorExpenseQRF":
            assert outputs == ["educator_expense"]
            assert weights == "design"
            return self

        def predict(self, features: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {"educator_expense": [300.0, 200.0]},
                index=features.index,
            )

    monkeypatch.setattr(puf_support_module, "QRF", EducatorExpenseQRF)
    expanded = clone_us_frame_for_puf_support(_minimal_us_frame())
    donor = pd.DataFrame(
        {
            "puf_predictor_filing_status_code": [1.0, 2.0, 4.0],
            "puf_predictor_tax_unit_person_count": [1.0, 2.0, 1.0],
            "educator_expense": [300.0, 200.0, 0.0],
            # Weighted donor positive rate = (0.5 + 0.5) / 4 = 25%.
            "weight": [0.5, 0.5, 3.0],
        }
    )

    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=(
            "puf_predictor_filing_status_code",
            "puf_predictor_tax_unit_person_count",
        ),
        person_outputs=("educator_expense",),
        tax_unit_outputs=(),
        n_estimators=4,
        seed=0,
    )

    person = imputed.table("person")
    asec_people = person[
        person[support_channel_column("person")] == BASE_ASEC_SUPPORT_CHANNEL
    ]
    puf_people = person[
        person[support_channel_column("person")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ].sort_values(support_source_id_column("person"))
    assert asec_people["educator_expense"].tolist() == [0.0, 0.0, 0.0]

    puf_totals = puf_people.groupby("person_tax_unit_id", sort=False)[
        "educator_expense"
    ].sum()
    np.testing.assert_allclose(puf_totals.to_numpy(), [900.0, 0.0])
    np.testing.assert_allclose(
        puf_people.iloc[:2]["educator_expense"].to_numpy(),
        [900.0 * 50_000.0 / 70_000.0, 900.0 * 20_000.0 / 70_000.0],
    )

    household = imputed.table("household")
    puf_household_mask = (
        household[support_channel_column("household")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    )
    puf_household_weights = pd.Series(
        imputed.weights_for("household").values[puf_household_mask],
        index=household.loc[puf_household_mask, "household_id"],
    )
    tax_unit_household = puf_people.groupby("person_tax_unit_id", sort=False)[
        "person_household_id"
    ].first()
    tax_unit_weights = tax_unit_household.map(puf_household_weights)
    puf_positive_rate = float(
        tax_unit_weights[puf_totals > 0.0].sum() / tax_unit_weights.sum()
    )
    donor_positive_rate = float(
        donor.loc[donor["educator_expense"] > 0.0, "weight"].sum()
        / donor["weight"].sum()
    )
    assert donor_positive_rate == pytest.approx(0.25)
    assert puf_positive_rate == pytest.approx(donor_positive_rate)


def test_puf_tax_unit_donor_derives_partnership_and_s_corp_split_from_raw_fields() -> (
    None
):
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 20],
            "E25980": [50.0, 70.0],
            "E25960": [5.0, 10.0],
            "E26190": [80.0, 110.0],
            "E26180": [20.0, 30.0],
            "E17500": [1_000.0, 2_000.0],
        },
        person_outputs=(
            "partnership_income",
            "s_corp_income",
            "partnership_self_employment_net_earnings",
            "health_insurance_premiums_without_medicare_part_b",
            "medicare_part_b_premiums",
            "other_medical_expenses",
            "over_the_counter_health_expenses",
        ),
        tax_unit_outputs=(),
    )

    assert donor["partnership_income"].tolist() == [45.0, 60.0]
    assert donor["s_corp_income"].tolist() == [60.0, 80.0]
    assert donor["partnership_self_employment_net_earnings"].tolist() == [25.0, 40.0]
    assert donor["health_insurance_premiums_without_medicare_part_b"].tolist() == [
        453.0,
        906.0,
    ]
    assert donor["medicare_part_b_premiums"].tolist() == [137.0, 274.0]
    assert donor["other_medical_expenses"].tolist() == [325.0, 650.0]
    assert donor["over_the_counter_health_expenses"].tolist() == [85.0, 170.0]
    assert "tax_unit_partnership_s_corp_income" not in donor


def test_puf_tax_unit_donor_carries_structural_mortgage_leaves() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 20],
            "first_home_mortgage_balance": [250_000.0, 500_000.0],
            "second_home_mortgage_balance": [0.0, 125_000.0],
            "first_home_mortgage_interest": [10_000.0, 20_000.0],
            "second_home_mortgage_interest": [0.0, 5_000.0],
            "first_home_mortgage_origination_year": [2018, 2016],
            "second_home_mortgage_origination_year": [0, 2020],
            "health_savings_account_ald": [1_500.0, 0.0],
            "domestic_production_ald": [7_500.0, 0.0],
        },
        person_outputs=(),
        tax_unit_outputs=PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    )

    assert donor["health_savings_account_ald"].tolist() == [1_500.0, 0.0]
    assert donor["domestic_production_ald"].tolist() == [7_500.0, 0.0]
    assert donor["first_home_mortgage_balance"].tolist() == [250_000.0, 500_000.0]
    assert donor["second_home_mortgage_balance"].tolist() == [0.0, 125_000.0]
    assert donor["first_home_mortgage_interest"].tolist() == [10_000.0, 20_000.0]
    assert donor["second_home_mortgage_interest"].tolist() == [0.0, 5_000.0]
    assert donor["first_home_mortgage_origination_year"].tolist() == [
        2018.0,
        2016.0,
    ]
    assert donor["second_home_mortgage_origination_year"].tolist() == [0.0, 2020.0]
    assert "interest_deduction" not in donor


def test_puf_tax_unit_donor_carries_processed_partnership_s_corp_as_leaves() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 20],
            "partnership_s_corp_income": [1_000.0, 2_000.0],
            "partnership_se_income": [100.0, 200.0],
        },
        person_outputs=(
            "partnership_income",
            "s_corp_income",
            "partnership_self_employment_net_earnings",
        ),
        tax_unit_outputs=(),
    )

    assert donor["partnership_income"].tolist() == [1_000.0, 2_000.0]
    assert donor["s_corp_income"].tolist() == [0.0, 0.0]
    assert donor["partnership_self_employment_net_earnings"].tolist() == [100.0, 200.0]
    assert "tax_unit_partnership_s_corp_income" not in donor


def test_puf_tax_detail_imputation_writes_only_puf_channel() -> None:
    expanded = clone_us_frame_for_puf_support(_minimal_us_frame())
    donor = pd.DataFrame(
        {
            "filing_status_code": [1.0, 2.0, 4.0, 1.0],
            "tax_unit_person_count": [1.0, 2.0, 1.0, 2.0],
            "employment_income_before_lsr": [1_000.0, 1_000.0, 1_000.0, 1_000.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=(
            "puf_predictor_filing_status_code",
            "puf_predictor_tax_unit_person_count",
        ),
        person_outputs=("employment_income_before_lsr",),
        tax_unit_outputs=(),
        n_estimators=4,
        seed=0,
    )

    person = imputed.table("person")
    asec_people = person[
        person[support_channel_column("person")] == BASE_ASEC_SUPPORT_CHANNEL
    ]
    puf_people = person[
        person[support_channel_column("person")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]

    assert asec_people["employment_income_before_lsr"].tolist() == [
        50_000.0,
        20_000.0,
        125_000.0,
    ]
    np.testing.assert_allclose(
        puf_people.groupby("person_tax_unit_id")["employment_income_before_lsr"]
        .sum()
        .to_numpy(),
        [1_000.0, 1_000.0],
    )
    assert "tax_unit_partnership_s_corp_income" not in imputed.table("tax_unit")


def test_puf_tax_detail_imputation_distributes_contributions_by_earnings() -> None:
    frame = _minimal_us_frame()
    frame.table("person")["self_employment_income_before_lsr"] = [
        0.0,
        40_000.0,
        10_000.0,
    ]
    expanded = clone_us_frame_for_puf_support(frame)
    donor = pd.DataFrame(
        {
            "filing_status_code": [1.0, 2.0, 4.0, 1.0],
            "tax_unit_person_count": [1.0, 2.0, 1.0, 2.0],
            "traditional_ira_contributions_desired": [900.0, 900.0, 900.0, 900.0],
            "self_employed_pension_contributions_desired": [
                500.0,
                500.0,
                500.0,
                500.0,
            ],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=(
            "puf_predictor_filing_status_code",
            "puf_predictor_tax_unit_person_count",
        ),
        person_outputs=(
            "traditional_ira_contributions_desired",
            "self_employed_pension_contributions_desired",
        ),
        tax_unit_outputs=(),
        n_estimators=4,
        seed=0,
    )

    person = imputed.table("person")
    puf_people = person[
        person[support_channel_column("person")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ].sort_values("person_id")
    joint_unit = puf_people.iloc[:2]
    # The cloned ASEC channel carries no contribution values, so the unit
    # total must follow the earnings basis: pension contributions follow
    # self-employment earnings (all on the second person), IRA contributions
    # follow total earnings (50k vs 20k + 40k).
    np.testing.assert_allclose(
        joint_unit["self_employed_pension_contributions_desired"].to_numpy(),
        [0.0, 500.0],
    )
    np.testing.assert_allclose(
        joint_unit["traditional_ira_contributions_desired"].to_numpy(),
        [900.0 * 50_000.0 / 110_000.0, 900.0 * 60_000.0 / 110_000.0],
    )
    asec_people = person[
        person[support_channel_column("person")] == BASE_ASEC_SUPPORT_CHANNEL
    ]
    assert asec_people["traditional_ira_contributions_desired"].tolist() == [
        0.0,
        0.0,
        0.0,
    ]


def test_puf_tax_detail_imputation_reconciles_social_security_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TotalSocialSecurityQRF:
        def __init__(self, **_: object) -> None:
            pass

        def fit(
            self,
            *_: object,
            **__: object,
        ) -> "TotalSocialSecurityQRF":
            return self

        def predict(self, features: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "social_security_retirement": [400.0, 800.0],
                    "social_security_disability": [0.0, 0.0],
                    "social_security_dependents": [0.0, 0.0],
                    "social_security_survivors": [0.0, 0.0],
                },
                index=features.index,
            )

    monkeypatch.setattr(puf_support_module, "QRF", TotalSocialSecurityQRF)

    base = _minimal_us_frame()
    person = base.table("person")
    person["social_security_retirement"] = [100.0, 0.0, 0.0]
    person["social_security_disability"] = [0.0, 100.0, 0.0]
    person["social_security_dependents"] = [0.0, 0.0, 0.0]
    person["social_security_survivors"] = [0.0, 0.0, 0.0]
    expanded = clone_us_frame_for_puf_support(base)
    donor = pd.DataFrame(
        {
            "filing_status_code": [1.0, 2.0],
            "tax_unit_person_count": [1.0, 2.0],
            "social_security_retirement": [1_000.0, 2_000.0],
            "social_security_disability": [0.0, 0.0],
            "social_security_dependents": [0.0, 0.0],
            "social_security_survivors": [0.0, 0.0],
            "weight": [1.0, 1.0],
        }
    )

    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=(
            "puf_predictor_filing_status_code",
            "puf_predictor_tax_unit_person_count",
        ),
        person_outputs=(
            "social_security_retirement",
            "social_security_disability",
            "social_security_dependents",
            "social_security_survivors",
        ),
        tax_unit_outputs=(),
        n_estimators=4,
        seed=0,
    )

    person = imputed.table("person")
    puf_people = person[
        person[support_channel_column("person")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]
    puf_totals = puf_people.groupby("person_tax_unit_id")[
        [
            "social_security_retirement",
            "social_security_disability",
            "social_security_dependents",
            "social_security_survivors",
        ]
    ].sum()
    np.testing.assert_allclose(
        puf_totals.to_numpy(),
        [
            [200.0, 200.0, 0.0, 0.0],
            [200.0, 200.0, 200.0, 200.0],
        ],
    )
    assert "social_security" not in person.columns


def test_puf_tax_detail_imputation_snaps_origination_years_to_donor_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FractionalYearQRF:
        def __init__(self, **_: object) -> None:
            pass

        def fit(
            self,
            *_: object,
            **__: object,
        ) -> "FractionalYearQRF":
            return self

        def predict(self, features: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "first_home_mortgage_origination_year": [2008.4, 2017.6],
                    "second_home_mortgage_origination_year": [0.2, 2019.5],
                },
                index=features.index,
            )

    monkeypatch.setattr(puf_support_module, "QRF", FractionalYearQRF)
    expanded = clone_us_frame_for_puf_support(_minimal_us_frame())
    donor = pd.DataFrame(
        {
            "filing_status_code": [1.0, 2.0, 4.0],
            "tax_unit_person_count": [1.0, 2.0, 1.0],
            "first_home_mortgage_origination_year": [0.0, 2008.0, 2018.0],
            "second_home_mortgage_origination_year": [0.0, 2020.0, 2020.0],
            "weight": [1.0, 1.0, 1.0],
        }
    )

    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=(
            "puf_predictor_filing_status_code",
            "puf_predictor_tax_unit_person_count",
        ),
        person_outputs=(),
        tax_unit_outputs=(
            "first_home_mortgage_origination_year",
            "second_home_mortgage_origination_year",
        ),
        n_estimators=4,
        seed=0,
    )

    puf_tax_units = imputed.table("tax_unit")[
        imputed.table("tax_unit")[support_channel_column("tax_unit")]
        == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]
    assert puf_tax_units["first_home_mortgage_origination_year"].tolist() == [
        2008.0,
        2018.0,
    ]
    assert puf_tax_units["second_home_mortgage_origination_year"].tolist() == [
        0.0,
        2020.0,
    ]


class TestPufSupportWeightsAuditWiring:
    """The PUF-support production fit emits a weights-audit record.

    This is what makes the build-level weights audit (populace #300) real rather
    than dead code: the actual production imputation records the weight kind it
    resolved, so a release manifest carries it and a ``"none"`` fit fails the
    release. The fit runs on a synthetic frame with no ``policyengine_us``, so
    this proves the wiring end to end in CI's engine-less environment.
    """

    def _donor(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "filing_status_code": [1.0, 2.0, 4.0, 1.0],
                "tax_unit_person_count": [1.0, 2.0, 1.0, 2.0],
                "employment_income_before_lsr": [1_000.0, 1_000.0, 1_000.0, 1_000.0],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        )

    def _impute(self, fit_records):
        return impute_us_puf_tax_detail_support(
            clone_us_frame_for_puf_support(_minimal_us_frame()),
            self._donor(),
            predictors=(
                "puf_predictor_filing_status_code",
                "puf_predictor_tax_unit_person_count",
            ),
            person_outputs=("employment_income_before_lsr",),
            tax_unit_outputs=(),
            n_estimators=4,
            seed=0,
            fit_records=fit_records,
        )

    def test_production_fit_records_design_weight_kind(self) -> None:
        from populace.build import FitWeightRecord
        from populace.build.us_runtime import US_PUF_SUPPORT_FIT_NAME

        fit_records: list[FitWeightRecord] = []
        self._impute(fit_records)

        assert fit_records == [FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, "design")]

    def test_recorded_fit_passes_the_weights_audit_gate(self) -> None:
        from populace.build import weights_audit_gate

        fit_records = []
        self._impute(fit_records)

        result = weights_audit_gate(fit_records)
        assert result.passed
        from populace.build.us_runtime import US_PUF_SUPPORT_FIT_NAME

        assert result.details["resolved_weight_kinds"] == {
            US_PUF_SUPPORT_FIT_NAME: "design"
        }

    def test_wired_gate_would_fail_a_none_fit(self) -> None:
        # Prove the wired gate can actually find something: swap the resolved
        # kind to "none" and the release-blocking gate fails, naming the fit.
        from populace.build import FitWeightRecord, weights_audit_gate
        from populace.build.us_runtime import US_PUF_SUPPORT_FIT_NAME

        result = weights_audit_gate([FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, "none")])
        assert not result.passed
        assert US_PUF_SUPPORT_FIT_NAME in result.failures[0]
        assert "unweighted" in result.failures[0]

    def test_records_are_only_emitted_when_a_sink_is_provided(self) -> None:
        # The out-parameter is opt-in: existing callers that pass nothing get
        # the same Frame return and are unaffected.
        imputed = impute_us_puf_tax_detail_support(
            clone_us_frame_for_puf_support(_minimal_us_frame()),
            self._donor(),
            predictors=(
                "puf_predictor_filing_status_code",
                "puf_predictor_tax_unit_person_count",
            ),
            person_outputs=("employment_income_before_lsr",),
            tax_unit_outputs=(),
            n_estimators=4,
            seed=0,
        )
        assert imputed.table("person") is not None

    def test_production_fit_records_design_kind_under_the_real_engine(self) -> None:
        # The engine-less tests above run the imputation with a trivial output
        # that never trips the formula-owned guard. This gated test runs the same
        # audited seam with the LIVE PolicyEngine-US metadata guard active
        # (assert_formula_owned_blocklist_current + resolve_formula_owned_outputs
        # both call real engine metadata), over real leaf-input outputs, so the
        # seam is proven end to end on the production code path an actual build
        # takes: the guard passes on genuine leaves, the DESIGN-weighted fit
        # records "design", and the release-blocking gate passes carrying it.
        pytest.importorskip("policyengine_us")
        from populace.build import FitWeightRecord, weights_audit_gate
        from populace.build.us_runtime import US_PUF_SUPPORT_FIT_NAME

        donor = pd.DataFrame(
            {
                "filing_status_code": [1.0, 2.0, 4.0, 1.0],
                "tax_unit_person_count": [1.0, 2.0, 1.0, 2.0],
                "employment_income_before_lsr": [1_000.0, 2_000.0, 3_000.0, 4_000.0],
                "qualified_dividend_income": [10.0, 20.0, 30.0, 40.0],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        fit_records: list = []
        impute_us_puf_tax_detail_support(
            clone_us_frame_for_puf_support(_minimal_us_frame()),
            donor,
            predictors=(
                "puf_predictor_filing_status_code",
                "puf_predictor_tax_unit_person_count",
            ),
            person_outputs=(
                "employment_income_before_lsr",
                "qualified_dividend_income",
            ),
            tax_unit_outputs=(),
            n_estimators=4,
            seed=0,
            fit_records=fit_records,
        )

        assert fit_records == [FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, "design")]
        result = weights_audit_gate(fit_records)
        assert result.passed
        assert result.details["resolved_weight_kinds"] == {
            US_PUF_SUPPORT_FIT_NAME: "design"
        }
