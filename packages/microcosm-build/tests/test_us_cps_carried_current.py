"""Corrected CPS monetary leaves: legacy oracles, routing, and refusals."""

import struct

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import asec_current_money_resources as loader
from microcosm.build.us_runtime import cps_carried
from microcosm.build.us_runtime._asec_current_money_codec import (
    current_money_content_sha256,
    encode_current_money,
)
from microcosm.build.us_runtime.asec_current_money import (
    AsecMoneyScope,
    AsecMoneyViews,
    SyntheticAsecSource,
    classify_asec_money,
    compile_asec_current_money_spec,
    require_complete_current_money,
    restate_asec_current_money,
)
from microcosm.build.us_runtime.asec_current_money_selection import (
    parse_current_money_body,
    select_current_money,
)
from microcosm.build.us_runtime.cps_carried_current import (
    CPS_CARRIED_CURRENT_PERSON_LEAVES,
    CPS_CARRIED_CURRENT_ROUTING_COLUMNS,
    CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES,
    CpsCarriedCurrentRefusalError,
    cps_carried_current_leaf_contract,
    derive_cps_carried_current_leaves,
)

PIN = "a" * 64
DIGEST = "b" * 64


def invented_money(person_amounts=None):
    """Six invented persons in three single-year households, one per cohort."""
    resources = loader.packaged_current_money_resources(
        inspected_version=money.MICROUNIT_VERSION,
        inspected_module_sha256=money.MICROUNIT_SHA256,
    )
    source = SyntheticAsecSource(
        PIN,
        PIN,
        ((2022, PIN), (2023, PIN), (2024, PIN)),
        (("ED_VAL.archive", PIN), ("ED_VAL.member", PIN)),
        PIN,
    )
    spec = compile_asec_current_money_spec(resources, source)
    scope = AsecMoneyScope(
        (10, 11, 12, 13, 14, 15),
        (100, 101, 102),
        (100, 100, 101, 101, 102, 102),
        (200, 200, 201, 201, 202, 202),
        (2022, 2022, 2023, 2023, 2024, 2024),
        (2022, 2023, 2024),
        ("p0", "p1", "p2", "p3", "p4", "p5"),
        ("h0", "h1", "h2"),
    )
    amounts = dict(person_amounts or {})
    person = pd.DataFrame(
        {
            field.column: np.asarray(
                amounts.get(field.name, np.zeros(6)), dtype="float64"
            )
            for field in spec.fields
            if field.entity == "person"
        }
    )
    household = pd.DataFrame(
        {
            field.column: np.zeros(3)
            for field in spec.fields
            if field.entity == "household"
        }
    )
    views = AsecMoneyViews(person, household, scope)
    ready = require_complete_current_money(
        restate_asec_current_money(classify_asec_money(views, spec, scope), spec), spec
    )
    return spec, ready


def body_of(ready):
    return parse_current_money_body(
        encode_current_money(ready),
        expected_header=ready.header,
        expected_content_sha256=current_money_content_sha256(ready),
        field_entities=tuple(
            (domain.name, domain.entity) for domain in ready.bindings.spec.fields
        ),
    )


def selected_of(body, persons, households):
    return select_current_money(
        body,
        person_positions=np.asarray(persons, dtype="int64"),
        household_positions=np.asarray(households, dtype="int64"),
        prepared_receipt_sha256=DIGEST,
        selection_sha256="c" * 64,
        person_identity_sha256="d" * 64,
        household_identity_sha256="e" * 64,
    )


# Six invented persons: two per source-year cohort, one household each pair.
ROUTING = {
    "A_AGE": [70, 40, 66, 20, 61, 35],
    "RESNSS1": [1, 0, 3, 4, 0, 2],
    "RESNSS2": [0, 0, 0, 0, 0, 0],
    "OI_OFF": [20, 12, 0, 3, 20, 0],
    "DST_SC1": [4, 1, 4, 0, 0, 4],
    "DST_SC2": [0, 4, 0, 0, 0, 0],
    "DST_SC1_YNG": [0, 0, 0, 4, 0, 0],
    "DST_SC2_YNG": [0, 0, 0, 0, 0, 0],
}
AMOUNTS = {
    "WSAL_VAL": [1000, 2000, 3000, 0, 5000, 6000],
    "SEMP_VAL": [-100, 0, 250, 0, 0, 77],
    "INT_VAL": [11, 22, 33, 44, 55, 66],
    "DIV_VAL": [101, 0, 303, 0, 505, 0],
    "CAP_VAL": [7, 0, 9, 0, 11, 0],
    "SS_VAL": [900, 0, 800, 700, 600, 500],
    "PNSN_VAL": [10, 20, 30, 40, 50, 60],
    "ANN_VAL": [-1, 5, 0, 0, 7, 0],
    "OI_VAL": [13, 17, 19, 23, 29, 31],
    "RNT_VAL": [-5, 0, 15, 0, 0, 25],
    "FRSE_VAL": [0, -9, 0, 0, 0, 3],
    "UC_VAL": [1, 2, 3, 4, 5, 6],
    "PHIP_VAL": [2, 4, 6, 8, 10, 12],
    "PMED_VAL": [3, 6, 9, 12, 15, 18],
    "POTC_VAL": [1, 1, 2, 3, 5, 8],
    "DST_VAL1": [500, 600, 700, 0, 0, 900],
    "DST_VAL2": [0, 650, 0, 0, 0, 0],
    "DST_VAL1_YNG": [0, 0, 0, 800, 0, 0],
    "DST_VAL2_YNG": [0, 0, 0, 0, 0, 0],
    "SPM_CHILDCAREXPNS": [40, 40, 60, 60, 80, 80],
}
SPM_MEMBERSHIP = np.asarray([200, 200, 201, 201, 202, 202], dtype="int64")
SPM_IDS = np.asarray([200, 201, 202], dtype="int64")
PERSON_YEARS = (2022, 2022, 2023, 2023, 2024, 2024)


def routing_frame(rows=(0, 1, 2, 3, 4, 5), **overrides):
    frame = pd.DataFrame(
        {
            name: pd.array([ROUTING[name][index] for index in rows], dtype="int64")
            for name in CPS_CARRIED_CURRENT_ROUTING_COLUMNS
        }
    )
    for name, values in overrides.items():
        frame[name] = values
    return frame


def leaves(rows=(0, 1, 2, 3, 4, 5), spm_rows=(0, 1, 2)):
    _, ready = invented_money(AMOUNTS)
    body = body_of(ready)
    selected = selected_of(body, persons=tuple(rows), households=tuple(spm_rows))
    membership = SPM_MEMBERSHIP[list(rows)]
    ids = np.unique(membership)
    derived = derive_cps_carried_current_leaves(
        selected,
        routing=routing_frame(rows),
        spm_membership=membership,
        spm_ids=ids,
    )
    return body, selected, derived


def factor(year):
    for pinned_year, _expression, bits in invented_money()[0].factors:
        if pinned_year == year:
            return struct.unpack("<d", bytes.fromhex(bits))[0]
    raise AssertionError(year)


def legacy_tables():
    person = pd.DataFrame(
        {name: np.zeros(6) for name in cps_carried.CPS_CARRIED_RAW_PERSON_COLUMNS}
    )
    for name, values in AMOUNTS.items():
        person[name] = np.asarray(values, dtype="float64")
    for name, values in ROUTING.items():
        person[name] = np.asarray(values, dtype="float64")
    person["person_spm_unit_id"] = SPM_MEMBERSHIP
    spm_unit = pd.DataFrame({"spm_unit_id": SPM_IDS})
    return cps_carried.derive_us_cps_carried_tables(person, spm_unit)


def test_owned_leaves_are_the_declared_monetary_surface_plus_age():
    _body, _selected, derived = leaves()
    assert tuple(sorted(derived.person)) == CPS_CARRIED_CURRENT_PERSON_LEAVES
    assert tuple(sorted(derived.spm_unit)) == CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES
    assert len(CPS_CARRIED_CURRENT_PERSON_LEAVES) == 24
    owned = set(CPS_CARRIED_CURRENT_PERSON_LEAVES) | set(
        CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES
    )
    # Every owned leaf is an incumbent CPS-carried surface, never a new column.
    assert owned <= set(cps_carried.CPS_CARRIED_PERSON_INPUTS) | set(
        cps_carried.CPS_CARRIED_SPM_UNIT_INPUTS
    )
    assert not owned & set(cps_carried.CPS_CARRIED_FORMULA_OWNED_COLUMNS)


def test_2024_rows_match_the_legacy_helper_byte_for_byte():
    rows = (4, 5)
    _body, _selected, derived = leaves(rows=rows, spm_rows=(2,))
    assert all(PERSON_YEARS[index] == 2024 for index in rows)
    legacy = legacy_tables()
    for name in CPS_CARRIED_CURRENT_PERSON_LEAVES:
        expected = legacy.person[name].to_numpy(dtype="float64")[list(rows)]
        assert derived.person[name].tobytes() == expected.tobytes(), name
    expected = legacy.spm_unit["spm_unit_pre_subsidy_childcare_expenses"].to_numpy(
        dtype="float64"
    )[[2]]
    assert (
        derived.spm_unit["spm_unit_pre_subsidy_childcare_expenses"].tobytes()
        == expected.tobytes()
    )


@pytest.mark.parametrize("row,year", [(0, 2022), (2, 2023)])
def test_prior_cohort_rows_restate_before_splitting(row, year):
    body, _selected, derived = leaves(rows=(row,), spm_rows=(row // 2,))
    scale = factor(year)
    interest = AMOUNTS["INT_VAL"][row] * scale
    assert body.field("INT_VAL").amounts[row] == interest
    assert derived.person["taxable_interest_income"][0] == (
        interest * cps_carried.TAXABLE_INTEREST_FRACTION
    )
    dividends = AMOUNTS["DIV_VAL"][row] * scale if AMOUNTS["DIV_VAL"][row] else 0.0
    assert derived.person["qualified_dividend_income"][0] == (
        dividends * cps_carried.QUALIFIED_DIVIDEND_FRACTION
    )
    assert derived.person["non_qualified_dividend_income"][0] == (
        dividends * (1 - cps_carried.QUALIFIED_DIVIDEND_FRACTION)
    )


def test_restating_before_splitting_is_observable():
    """The two orders are not interchangeable; some amounts separate them."""
    scale = factor(2022)
    fraction = cps_carried.TAXABLE_INTEREST_FRACTION
    separating = [
        amount
        for amount in range(1, 200)
        if (amount * scale) * fraction != (amount * fraction) * scale
    ]
    assert separating, "the fixture cannot distinguish the two orders"
    amount = separating[0]
    _body, _selected, derived = leaves_for_interest(amount)
    assert derived.person["taxable_interest_income"][0] == (amount * scale) * fraction
    assert derived.person["taxable_interest_income"][0] != (amount * fraction) * scale


def leaves_for_interest(amount):
    _, ready = invented_money({**AMOUNTS, "INT_VAL": [amount, 0, 0, 0, 0, 0]})
    body = body_of(ready)
    selected = selected_of(body, persons=(0,), households=(0,))
    derived = derive_cps_carried_current_leaves(
        selected,
        routing=routing_frame((0,)),
        spm_membership=SPM_MEMBERSHIP[[0]],
        spm_ids=np.asarray([200], dtype="int64"),
    )
    return body, selected, derived


def test_social_security_routing_precedence_and_age_fallback():
    _body, _selected, derived = leaves()
    retirement = derived.person["social_security_retirement"]
    disability = derived.person["social_security_disability"]
    survivors = derived.person["social_security_survivors"]
    dependents = derived.person["social_security_dependents"]
    total = retirement + disability + survivors + dependents
    _body2, selected, _d = leaves()
    assert total.tolist() == pytest.approx(selected.amounts("SS_VAL").tolist())
    assert retirement[0] > 0 and disability[0] == 0  # RESNSS1 == 1
    assert survivors[2] > 0  # RESNSS1 == 3
    assert dependents[3] > 0  # RESNSS1 == 4
    # Unclassified positive receipt at 61 falls back to disability, not retirement.
    assert ROUTING["A_AGE"][4] == 61 and ROUTING["RESNSS1"][4] == 0
    assert disability[4] > 0 and retirement[4] == 0


def test_annuity_niu_contributes_zero_and_losses_stay_signed():
    _body, selected, derived = leaves()
    assert selected.amounts("ANN_VAL")[0] == 0.0
    pensions = (
        derived.person["taxable_private_pension_income"]
        + derived.person["tax_exempt_private_pension_income"]
    )
    assert pensions[0] == pytest.approx(selected.amounts("PNSN_VAL")[0])
    assert derived.person["self_employment_income_before_lsr"][0] < 0
    assert derived.person["rental_income"][0] < 0
    assert derived.person["farm_operations_income"][1] < 0


def test_other_income_conserves_and_ira_takes_code_four_only():
    _body, selected, derived = leaves()
    total = (
        derived.person["alimony_income"]
        + derived.person["strike_benefits"]
        + derived.person["miscellaneous_income"]
    )
    assert total.tolist() == selected.amounts("OI_VAL").tolist()
    assert derived.person["alimony_income"][0] == selected.amounts("OI_VAL")[0]
    assert derived.person["strike_benefits"][1] == selected.amounts("OI_VAL")[1]
    assert derived.person["miscellaneous_income"][2] == selected.amounts("OI_VAL")[2]
    ira = derived.person["taxable_ira_distributions"]
    assert ira[0] == selected.amounts("DST_VAL1")[0]
    assert ira[1] == selected.amounts("DST_VAL2")[1]  # DST_SC1 == 1 contributes 0
    assert ira[3] == selected.amounts("DST_VAL1_YNG")[3]
    assert ira[4] == 0.0


def test_spm_childcare_carries_the_agreed_member_value():
    _body, selected, derived = leaves()
    values = derived.spm_unit["spm_unit_pre_subsidy_childcare_expenses"]
    assert len(values) == 3
    assert values.tolist() == [
        selected.amounts("SPM_CHILDCAREXPNS")[0],
        selected.amounts("SPM_CHILDCAREXPNS")[2],
        selected.amounts("SPM_CHILDCAREXPNS")[4],
    ]


def test_routing_refuses_raw_dollar_columns_and_malformed_codes():
    _, ready = invented_money(AMOUNTS)
    selected = selected_of(body_of(ready), persons=(0, 1), households=(0,))
    membership = SPM_MEMBERSHIP[[0, 1]]
    ids = np.unique(membership)

    def derive(routing, **kwargs):
        return derive_cps_carried_current_leaves(
            selected,
            routing=routing,
            spm_membership=kwargs.get("membership", membership),
            spm_ids=kwargs.get("ids", ids),
        )

    money_columns = routing_frame((0, 1))
    money_columns["WSAL_VAL"] = np.zeros(2, dtype="int64")
    with pytest.raises(CpsCarriedCurrentRefusalError, match="RAW_MONEY_INPUT_REFUSED"):
        derive(money_columns)
    absent = routing_frame((0, 1)).drop(columns=["RESNSS2"])
    with pytest.raises(CpsCarriedCurrentRefusalError, match="ROUTING_ROSTER"):
        derive(absent)
    floats = routing_frame((0, 1))
    floats["A_AGE"] = np.asarray([70.0, 40.0])
    with pytest.raises(CpsCarriedCurrentRefusalError, match="ROUTING_DTYPE"):
        derive(floats)
    nullable = routing_frame((0, 1))
    nullable["OI_OFF"] = pd.array([20, None], dtype="Int64")
    with pytest.raises(CpsCarriedCurrentRefusalError, match="ROUTING_MISSING"):
        derive(nullable)
    with pytest.raises(CpsCarriedCurrentRefusalError, match="ROUTING_ROW_ALIGNMENT"):
        derive(routing_frame((0,)))


def test_inconsistent_spm_members_and_bad_unit_ids_refuse():
    # The source decoder already refuses an inconsistent SPM amount, so this
    # exercises the remaining risk: a membership vector that claims two rows
    # from different units belong to one, which must refuse rather than reduce.
    _, ready = invented_money(AMOUNTS)
    selected = selected_of(body_of(ready), persons=(0, 2), households=(0,))
    with pytest.raises(CpsCarriedCurrentRefusalError, match="INCONSISTENT_SPM_AMOUNT"):
        derive_cps_carried_current_leaves(
            selected,
            routing=routing_frame((0, 2)),
            spm_membership=np.asarray([200, 200], dtype="int64"),
            spm_ids=np.asarray([200], dtype="int64"),
        )
    _, clean = invented_money(AMOUNTS)
    ok = selected_of(body_of(clean), persons=(0, 1), households=(0,))
    with pytest.raises(CpsCarriedCurrentRefusalError, match="SPM_MEMBERSHIP_COVERAGE"):
        derive_cps_carried_current_leaves(
            ok,
            routing=routing_frame((0, 1)),
            spm_membership=SPM_MEMBERSHIP[[0, 1]],
            spm_ids=np.asarray([200, 201], dtype="int64"),
        )
    with pytest.raises(CpsCarriedCurrentRefusalError, match="SPM_IDS"):
        derive_cps_carried_current_leaves(
            ok,
            routing=routing_frame((0, 1)),
            spm_membership=SPM_MEMBERSHIP[[0, 1]],
            spm_ids=np.asarray([200.0]),
        )


def test_leaf_contract_records_the_reused_model_assumptions():
    contract = cps_carried_current_leaf_contract()
    assert contract["raw_dollar_alternates"] == []
    assert contract["amount_basis"] == "target_current_2024"
    assert contract["release_eligible"] is False
    assumptions = contract["model_assumptions"]
    assert assumptions["qualified_dividend_fraction"] == (
        cps_carried.QUALIFIED_DIVIDEND_FRACTION
    )
    assert assumptions["annuity_niu_contributes_zero"] is True
    assert assumptions["ira_distribution_code"] == 4
    assert assumptions["social_security_age_fallback"] == 62
    assert set(contract["money_fields"]) <= set(money.FIELDS)
