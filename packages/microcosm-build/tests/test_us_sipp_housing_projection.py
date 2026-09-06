"""Behavioral contracts for the December SIPP housing projection recipe.

Every row here is invented. No SIPP byte is opened, no parent is authenticated,
nothing is fitted and no amount below came from the real file. The tests assert
what the recipe must refuse and how it must classify evidence it is given.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
import subprocess
import sys
import types
from decimal import Decimal

import pytest

from microcosm.build import sipp_december_source as source
from microcosm.build import sipp_housing_measurements as hm
from microcosm.build import sipp_housing_predictors as predictors
from microcosm.build import sipp_housing_projection as hp
from microcosm.build import sipp_measurements as m

SOURCE = m.SippSource(source._PINS.sha256, source._PINS.metadata_sha256, 2023, 2022)

# One invented renter household that satisfies every admission condition. Each
# test starts from this and changes exactly the evidence it is about.
RENTER = {
    "ETENURE": "2",
    "ATENURE": "1",
    "TRENTMORT": "900",
    "ARENTMORT": "1",
    "TUTILS": "150",
    "AUTILS": "1",
    "ERENTSUB": "2",
    "ARENTSUB": "1",
    "EVOUCHER": "",
    "AVOUCHER": "0",
    "EPRDEBT": "",
    "APRDEBT": "0",
    "EMHDEBT": "",
    "AMHDEBT": "0",
    "EEHC_TEN": "2",
    "AEHC_TEN": "1",
    "EEHC_RENTSUB": "2",
    "AEHC_RENTSUB": "1",
    "EEHC_VOUCHER": "",
    "AEHC_VOUCHER": "0",
}
# Version 3 predictor columns. A single-member household built from these
# agrees with its own roster on both counts, so a test that is about something
# else never trips a disagreement diagnostic by accident.
PREDICTORS = {
    "TEHC_ST": "24",
    "AEHC_ST": "1",
    "RHNUMPER": "1",
    "AHNUMPER": "1",
    "RHNUMU18": "0",
    "AHNUMU18": "1",
    "THTOTINC": "4200",
    "AHTOTINC": "1",
    "TAGE_EHC": "44",
}
OWNER = {
    **RENTER,
    "ETENURE": "1",
    "TRENTMORT": "1800",
    "ERENTSUB": "",
    "ARENTSUB": "0",
    "EEHC_TEN": "1",
    "EEHC_RENTSUB": "",
    "AEHC_RENTSUB": "0",
    "EPRDEBT": "1",
    "APRDEBT": "1",
}


def _row(ssuid="0001", pnum=101, residence="100001", weight="1000", **overrides):
    values = dict.fromkeys(hp.PROJECTION_FIELDS, "")
    values.update(
        SSUID=ssuid,
        PNUM=str(pnum),
        MONTHCODE="12",
        ERESIDENCEID=residence,
        WPFINWGT=weight,
        **RENTER,
        **PREDICTORS,
    )
    values.update(overrides)
    return m.SippRow(values)


def _bare_row(ssuid="0001", pnum=101, residence="100001", weight="1000", **overrides):
    """The same household with the nine predictor columns absent entirely."""
    values = {
        k: v
        for k, v in _row(ssuid, pnum, residence, weight, **overrides).values.items()
        if k not in predictors.PREDICTOR_FIELDS
    }
    return m.SippRow(values)


def _household(*rows):
    return hm.december_housing(list(rows), source=SOURCE, complete_membership=True)[0]


def _project(*rows, selection="first_32_ssuids"):
    return hp.project(list(rows), source=SOURCE, selection=selection)


# --- recipe identity and refusals ------------------------------------------


def test_recipe_bytes_match_the_committed_pin():
    payload = hp.recipe_bytes()
    assert hashlib.sha256(payload).hexdigest() == hp.RECIPE_SHA256
    assert hp.verify_recipe() == hp.recipe_digest()


def test_recipe_document_is_a_fresh_value_a_caller_cannot_poison():
    document = hp.recipe_document()
    document["fields"] = ["TAMPERED"]
    document["parent"]["sha256"] = "0" * 64
    assert hp.recipe_document()["fields"] == list(hp.PROJECTION_FIELDS)
    assert hashlib.sha256(hp.recipe_bytes()).hexdigest() == hp.RECIPE_SHA256


def test_a_tampered_recipe_pin_refuses_before_any_measurement(monkeypatch):
    monkeypatch.setattr(hp, "RECIPE_SHA256", "0" * 64)
    with pytest.raises(hp.HousingProjectionError, match="HOUSING_RECIPE_TAMPERED"):
        hp.verify_recipe()
    with pytest.raises(hp.HousingProjectionError, match="HOUSING_RECIPE_TAMPERED"):
        _project(_row())


def test_a_second_dictionary_authority_refuses(monkeypatch):
    monkeypatch.setitem(hp.DICTIONARY, "sha256", "1" * 64)
    with pytest.raises(hp.HousingProjectionError, match="DICTIONARY_MISMATCH"):
        hp.verify_dictionary()


@pytest.mark.parametrize(
    "replacement",
    [
        m.SippSource("0" * 64, source._PINS.metadata_sha256, 2023, 2022),
        m.SippSource(source._PINS.sha256, "0" * 64, 2023, 2022),
        m.SippSource(source._PINS.sha256, source._PINS.metadata_sha256, 2022, 2022),
        m.SippSource(source._PINS.sha256, source._PINS.metadata_sha256, 2023, 2023),
    ],
)
def test_a_tampered_source_identity_refuses(replacement):
    with pytest.raises(hp.HousingProjectionError, match="HOUSING_SOURCE"):
        hp.verify_source(replacement)
    with pytest.raises(hp.HousingProjectionError, match="HOUSING_SOURCE"):
        hp.project([_row()], source=replacement, selection="first_32_ssuids")


def test_a_header_missing_a_declared_column_refuses():
    header = [f for f in hp.PROJECTION_FIELDS if f != "ARENTMORT"]
    with pytest.raises(hp.HousingProjectionError, match="HEADER"):
        hp.header_positions(header)
    with pytest.raises(hp.HousingProjectionError, match="HEADER"):
        hp.header_positions([*hp.PROJECTION_FIELDS, "ETENURE"])
    positions = hp.header_positions(["FILLER", *hp.PROJECTION_FIELDS])
    assert positions == tuple(range(1, len(hp.PROJECTION_FIELDS) + 1))


def test_an_unsupported_selection_refuses():
    with pytest.raises(hp.HousingProjectionError, match="UNSUPPORTED_SELECTION"):
        _project(_row(), selection="first_64_ssuids")


def test_the_recipe_does_not_introduce_a_second_household_identity():
    assert "SHHADID" not in hp.PROJECTION_FIELDS
    assert hp.recipe_document()["residence_key"]["field"] == "ERESIDENCEID"


def test_the_frozen_sixteen_field_producer_contract_is_untouched():
    assert source.FIELDS == (
        "SSUID",
        "PNUM",
        "MONTHCODE",
        "ERESIDENCEID",
        "WPFINWGT",
        "THEQ_VEH",
        "AHEQ_VEH",
        "THVAL_VEH",
        "AHVAL_VEH",
        "THDEBT_VEH",
        "AHDEBT_VEH",
        "TVEH_NUM",
        "AVEH_NUM",
        "TVAL_BANK",
        "AVAL_BANK",
        "TAGE_EHC",
    )
    assert source._COPIED == ("THEQ_VEH", "THVAL_VEH", "THDEBT_VEH", "TVEH_NUM")
    assert source._PINS.expected_counts == (39513, 17646, 17314, 318, 29)
    assert tuple(m._HOUSEHOLD) == (
        "TVEH_NUM",
        "THVAL_VEH",
        "THEQ_VEH",
        "THDEBT_VEH",
    )


def test_the_one_column_shared_with_the_frozen_producer_has_a_single_contract():
    # Version 3 carries TAGE_EHC, which the frozen sixteen-field producer also
    # reads to gate its retirement-income and bank-asset universes. Sharing a
    # column is allowed; reading it two different ways is not.
    shared = set(source.FIELDS) & set(hp.PROJECTION_FIELDS[5:])
    assert shared == {predictors.SHARED_WITH_FROZEN_PRODUCER} == {"TAGE_EHC"}
    spec = predictors._PREDICTOR_AMOUNTS["TAGE_EHC"]
    assert {
        "low": spec.low,
        "high": spec.high,
        "sentinel": spec.sentinel,
    } == predictors.FROZEN_PRODUCER_AGE_READING
    # The frozen producer's own reading, asserted against its source rather than
    # against a copy of the numbers.
    frozen = pathlib.Path(m.__file__).read_text()
    assert '_number(row.values, "TAGE_EHC", 0, 90, -999)' in frozen
    assert hp.recipe_document()["prospective_predictors"][
        "shared_with_frozen_producer"
    ] == {
        "field": "TAGE_EHC",
        "frozen_producer_reading": {"low": 0, "high": 90, "sentinel": -999},
        "read_identically_here": True,
        "frozen_producer_fields_mutated": False,
        "second_contract_for_the_shared_column_introduced": False,
    }


def test_the_frozen_producer_modules_are_byte_identical():
    # These two modules are declared frozen: their digests sit inside the
    # producer binding that already-published receipts carry. A change to either
    # is a review, not a refactor, so the bytes are pinned here.
    digests = {
        pathlib.Path(module.__file__).name: hashlib.sha256(
            pathlib.Path(module.__file__).read_bytes()
        ).hexdigest()
        for module in (source, m)
    }
    assert digests == {
        "sipp_december_source.py": (
            "faa97d8699028111de48c7b065e0463def3007e985f1ae5cca35cf13ff129d4c"
        ),
        "sipp_measurements.py": (
            "81fac9866b72cf42dbd98a2eb24a9c1087893ede3fa57c8ccdbd411812510b2c"
        ),
    }


# --- transcription fidelity -------------------------------------------------


@pytest.mark.parametrize(
    ("field", "flag", "page", "grain"),
    [
        ("TRENTMORT", "ARENTMORT", 379, "household"),
        ("TUTILS", "AUTILS", 380, "household"),
        ("ETENURE", "ATENURE", 784, "household"),
        ("ERENTSUB", "ARENTSUB", 788, "household"),
        ("EVOUCHER", "AVOUCHER", 789, "household"),
        ("EEHC_RENTSUB", "AEHC_RENTSUB", 790, "person_month"),
        ("EEHC_VOUCHER", "AEHC_VOUCHER", 791, "person_month"),
    ],
)
def test_each_mandated_field_carries_its_transcribed_flag_page_and_grain(
    field, flag, page, grain
):
    entry = next(e for e in hp.field_contracts() if e["name"] == field)
    assert entry["status_flag"] == flag
    assert entry["dictionary_zero_based_page"] == page
    assert entry["dictionary_printed_page"] == page + 1
    assert entry["grain"] == grain
    assert entry["documented_numeric_missing_sentinel"] is None
    assert flag in hp.PROJECTION_FIELDS


def test_the_amount_domains_and_category_domains_are_the_published_ones():
    contracts = {e["name"]: e for e in hp.field_contracts()}
    for name in ("TRENTMORT", "TUTILS"):
        assert (contracts[name]["minimum"], contracts[name]["maximum"]) == (
            0,
            9_999_999,
        )
        assert contracts[name]["unit"] == "USD_nominal_2022_monthly"
    assert contracts["ETENURE"]["codes"] == [1, 2, 3]
    assert contracts["ERENTSUB"]["codes"] == [1, 2]
    assert contracts["EVOUCHER"]["universe"] == "ETENURE in (2,3) and ERENTSUB = 1"
    assert contracts["ERENTSUB"]["universe"] == "ETENURE in (2,3)"
    assert (
        contracts["TRENTMORT"]["universe"]
        == "ETENURE = 2 or (ETENURE = 1 and (EPRDEBT = 1 or EMHDEBT = 1))"
    )
    assert contracts["EEHC_VOUCHER"]["universe"] == "EEHC_RENTSUB = 1"
    assert contracts["TUTILS"]["universe"] == "ETENURE in (1,2,3)"


def test_utilities_stay_a_separate_measurement_from_rent():
    household = _household(_row())
    assert household.rent.value == Decimal(900)
    assert household.utilities.value == Decimal(150)
    assert household.rent.concept == "TRENTMORT_on_renter_tenure"
    assert household.utilities.concept == "TUTILS"
    assert hp.NON_CLAIMS["utilities_added_into_any_rent_quantity"] is False


def test_the_declared_period_and_price_basis_are_never_applied_here():
    declarations = hp.recipe_document()["period_and_basis_declarations"]
    assert declarations["amount_period"] == "December_2022_monthly"
    assert declarations["annualization"] == "not_applied"
    assert declarations["price_basis"] == "USD_nominal_2022_original"
    assert hp.NON_CLAIMS["observes_landlord_subsidy_amount"] is False
    assert hp.NON_CLAIMS["unassisted_counterfactual_asserted"] is False


# --- copy grain -------------------------------------------------------------


def test_a_household_copy_disagreement_makes_the_measurement_unknown():
    household = _household(
        _row(pnum=101),
        _row(pnum=102, TRENTMORT="1200"),
    )
    rent_amount = household.household("TRENTMORT")
    assert rent_amount.value is None
    assert "household_copy_conflict" in rent_amount.reasons
    assert hm.measurement_class(rent_amount) == "copy_conflict"
    assert household.rent.value is None
    assert household.exclusion == "rent_copy_conflict"


def test_a_tenure_copy_disagreement_excludes_before_any_rent_reading():
    household = _household(_row(pnum=101), _row(pnum=102, ETENURE="1"))
    assert household.tenure == "tenure_copy_conflict"
    assert household.exclusion == "tenure_copy_conflict"
    assert household.rent.value is None


def test_a_disagreeing_allocation_flag_alone_is_a_copy_conflict():
    household = _household(_row(pnum=101), _row(pnum=102, ARENTMORT="2"))
    assert "household_copy_conflict" in household.household("TRENTMORT").reasons


def test_person_month_indicators_are_not_copied_across_members():
    household = _household(
        _row(pnum=101),
        _row(pnum=102, EEHC_RENTSUB="1", EEHC_VOUCHER="2", AEHC_VOUCHER="1"),
    )
    receipts = household.person_month("EEHC_RENTSUB")
    assert [v.value for v in receipts] == [Decimal(2), Decimal(1)]
    assert all("household_copy_conflict" not in v.reasons for v in receipts)
    assert household.diagnostics["eehc_rentsub_member_disagreement"] is True
    # The household stays classified by its household indicators.
    assert household.assistance == "explicit_unassisted"
    assert household.exclusion is None
    diagnostics = _project(
        _row(pnum=101),
        _row(pnum=102, EEHC_RENTSUB="1", EEHC_VOUCHER="2", AEHC_VOUCHER="1"),
    ).diagnostics
    assert diagnostics["person_month_receipt_in_explicit_unassisted_households"] == 1
    assert diagnostics["person_month_receipt_in_admitted_households"] == 1


def test_the_housing_roster_must_agree_with_the_frozen_grouping(monkeypatch):
    def wrong(rows, **kwargs):
        households = original(rows, **kwargs)
        return households[:0]

    original = m.december_households
    monkeypatch.setattr(m, "december_households", wrong)
    with pytest.raises(hm.HousingMeasurementError, match="HOUSING_ROSTER_DISAGREEMENT"):
        _household(_row())


# --- missing, NIU, zero and negative ---------------------------------------


def test_a_blank_amount_is_not_a_zero_amount():
    blank = _household(_row(TRENTMORT="", ARENTMORT="0"))
    assert blank.household("TRENTMORT").value is None
    assert "TRENTMORT:blank" in blank.household("TRENTMORT").reasons
    assert blank.rent.value is None
    assert blank.exclusion == "rent_unknown"
    zero = _household(_row(TRENTMORT="0"))
    assert zero.household("TRENTMORT").value == 0
    assert zero.rent.value == 0
    assert zero.exclusion is None
    assert zero.diagnostics["rent_zero"] is True


def test_a_negative_amount_is_an_invalid_domain_not_a_documented_sentinel():
    household = _household(_row(TRENTMORT="-999"))
    observation = household.household("TRENTMORT").observations[0]
    assert observation.lexical_class == "invalid_domain"
    assert observation.value is None
    contract = next(e for e in hp.field_contracts() if e["name"] == "TRENTMORT")
    assert contract["documented_numeric_missing_sentinel"] is None


def test_an_out_of_universe_flag_never_yields_an_observed_value():
    household = _household(_row(TRENTMORT="900", ARENTMORT="0"))
    amount = household.household("TRENTMORT")
    assert amount.value is None
    assert "ARENTMORT:not_in_universe" in amount.reasons
    assert hm.measurement_class(amount) == "not_in_universe"


def test_an_absent_column_is_distinct_from_a_blank_one():
    values = dict.fromkeys(hp.PROJECTION_FIELDS, "")
    values.update(
        SSUID="0001",
        PNUM="101",
        MONTHCODE="12",
        ERESIDENCEID="100001",
        WPFINWGT="1000",
        **RENTER,
    )
    del values["TUTILS"]
    household = _household(m.SippRow(values))
    utilities = household.household("TUTILS")
    assert "TUTILS:absent" in utilities.reasons
    raw = {r.name: r for r in utilities.observations[0].raw_fields}
    assert raw["TUTILS"].present is False
    assert raw["AUTILS"].present is True


def test_an_out_of_domain_category_code_refuses_the_value():
    household = _household(_row(ETENURE="4"))
    tenure = household.household("ETENURE")
    assert tenure.value is None
    assert tenure.observations[0].lexical_class == "invalid_domain"
    assert household.tenure == "tenure_unknown"
    assert household.exclusion == "tenure_unknown"


def test_the_domain_check_is_set_membership_not_only_a_range(monkeypatch):
    gapped = hm._Category(
        "ATENURE",
        (1, 3),
        ("Owned", "No payment"),
        "household",
        "invented_gapped_domain",
        "invented",
        "invented",
        784,
    )
    monkeypatch.setitem(hm._HOUSEHOLD_CATEGORIES, "ETENURE", gapped)
    household = _household(_row(ETENURE="2"))
    assert household.household("ETENURE").value is None


# --- owner versus renter ----------------------------------------------------


def test_an_owner_amount_is_a_mortgage_and_never_a_zero_rent():
    household = _household(_row(**OWNER))
    assert household.tenure == "owner"
    assert household.household("TRENTMORT").value == Decimal(1800)
    assert household.rent.value is None
    assert "mortgage_excluded" in household.rent.reasons
    assert "tenure_not_renter:owner" in household.rent.reasons
    assert household.exclusion == "tenure_owner"
    assert household.diagnostics["owner_debt"] == "owner_debt_indicated"
    assert household.diagnostics["owner_debt_evidence"] == {
        "EPRDEBT": "observed",
        "EMHDEBT": "not_in_universe",
    }


def test_an_owner_without_reported_debt_is_reported_as_unresolved():
    household = _household(
        _row(**{**OWNER, "EPRDEBT": "", "APRDEBT": "0", "TRENTMORT": ""})
    )
    assert household.diagnostics["owner_debt"] == "owner_debt_unresolved"
    assert household.rent.value is None


def test_occupied_without_payment_of_rent_is_outside_the_rent_universe():
    household = _household(_row(ETENURE="3", EEHC_TEN="3"))
    assert household.tenure == "no_payment"
    assert household.rent.value is None
    assert "tenure_not_renter:no_payment" in household.rent.reasons
    assert household.exclusion == "tenure_no_payment"


def test_utilities_are_measured_for_owners_too_without_entering_rent():
    household = _household(_row(**OWNER))
    assert household.utilities.value == Decimal(150)
    assert household.rent.value is None


# --- assistance -------------------------------------------------------------


def test_a_reported_explicit_negative_is_the_only_unassisted_route():
    assert _household(_row()).assistance == "explicit_unassisted"


def test_an_imputed_negative_is_not_an_explicit_negative():
    household = _household(_row(ARENTSUB="2"))
    assert household.assistance == "unassisted_allocated"
    assert household.exclusion == "assistance_unassisted_allocated"
    assert hm.measurement_quality(household.household("ERENTSUB")) == (
        "hot_deck_imputed"
    )


def test_an_unknown_assistance_status_is_never_recoded_as_unassisted():
    household = _household(_row(ERENTSUB="", ARENTSUB=""))
    assert household.assistance == "assistance_unknown"
    assert household.exclusion == "assistance_unknown"


def test_a_not_in_universe_assistance_status_keeps_its_own_class():
    household = _household(_row(ERENTSUB="", ARENTSUB="0"))
    assert household.assistance == "assistance_not_in_universe"
    # Its own rung, not the shared unknown one: the doc says the class stays
    # distinct, and the exclusive ladder now says the same thing.
    assert household.exclusion == "assistance_not_in_universe"
    assert "assistance_not_in_universe" in hm.EXCLUSION_ORDER


def test_an_assistance_copy_conflict_is_its_own_exclusion():
    household = _household(_row(pnum=101), _row(pnum=102, ERENTSUB="1"))
    assert household.assistance == "assistance_copy_conflict"
    assert household.exclusion == "assistance_copy_conflict"


def test_a_voucher_answer_outside_its_published_universe_is_a_conflict():
    household = _household(_row(EVOUCHER="1", AVOUCHER="1"))
    assert household.assistance == "assistance_universe_conflict"
    assert household.exclusion == "assistance_universe_conflict"


def test_an_owner_answering_the_assistance_item_is_a_conflict():
    household = _household(_row(**{**OWNER, "ERENTSUB": "2", "ARENTSUB": "1"}))
    assert household.assistance == "assistance_universe_conflict"


def test_an_owner_outside_the_assistance_universe_is_not_a_conflict():
    assert _household(_row(**OWNER)).assistance == "not_in_assistance_universe"


@pytest.mark.parametrize(
    ("voucher", "voucher_flag", "expected"),
    [
        ("1", "1", "assisted_with_voucher"),
        ("2", "1", "assisted_without_voucher"),
        ("", "0", "assisted_voucher_unknown"),
    ],
)
def test_an_assisted_household_keeps_its_voucher_detail(
    voucher, voucher_flag, expected
):
    household = _household(_row(ERENTSUB="1", EVOUCHER=voucher, AVOUCHER=voucher_flag))
    assert household.assistance == expected
    assert household.exclusion == "assistance_assisted"
    # An assisted household's TRENTMORT is its post-assistance tenant payment.
    assert household.household("TRENTMORT").value == Decimal(900)
    assert household.rent.value == Decimal(900)


# --- the strict explicit-negative rule (review S1) ---------------------------


@pytest.mark.parametrize(
    ("label", "rows", "assistance", "exclusion"),
    [
        (
            "copy_conflict",
            lambda: (_row(pnum=101), _row(pnum=102, EVOUCHER="1", AVOUCHER="1")),
            "assistance_copy_conflict",
            "assistance_copy_conflict",
        ),
        (
            "suppressed",
            lambda: (_row(AVOUCHER="-99"),),
            "assistance_universe_conflict",
            "assistance_universe_conflict",
        ),
        (
            "value_under_not_in_universe_flag",
            lambda: (_row(EVOUCHER="1", AVOUCHER="0"),),
            "assistance_universe_conflict",
            "assistance_universe_conflict",
        ),
        (
            "blank_under_observed_flag",
            lambda: (_row(EVOUCHER="", AVOUCHER="1"),),
            "assistance_universe_conflict",
            "assistance_universe_conflict",
        ),
    ],
)
def test_an_explicit_negative_admits_only_a_blank_out_of_universe_voucher(
    label, rows, assistance, exclusion
):
    household = _household(*rows())
    assert household.household("ERENTSUB").value == Decimal(2)
    assert hm.measurement_class(household.household("ERENTSUB")) == "observed"
    assert household.assistance == assistance
    assert household.exclusion == exclusion
    assert household.assistance != "explicit_unassisted"


def test_the_expected_blank_voucher_under_flag_zero_is_not_turned_into_missing():
    household = _household(_row())
    voucher = household.household("EVOUCHER")
    assert household.diagnostics["voucher_evidence"] == (
        "expected_blank_not_in_universe"
    )
    # The published universe predicts exactly this state for an unassisted
    # household, so it stays not-in-universe rather than becoming unknown.
    assert hm.measurement_class(voucher) == "not_in_universe"
    assert household.diagnostics["voucher_class"] == "not_in_universe"
    assert voucher.value is None
    assert "EVOUCHER:blank" in voucher.reasons
    assert household.assistance == "explicit_unassisted"
    assert household.exclusion is None


def test_a_discarded_voucher_value_is_distinguished_from_a_blank_one():
    blank = _household(_row())
    populated = _household(_row(EVOUCHER="1", AVOUCHER="0"))
    # measurement_class alone cannot separate them: the value was discarded, so
    # both read as the flag's class. The preserved observations can.
    assert hm.measurement_class(blank.household("EVOUCHER")) == "not_in_universe"
    assert hm.measurement_class(populated.household("EVOUCHER")) == "not_in_universe"
    assert blank.household("EVOUCHER").observations[0].lexical_class == "blank"
    assert populated.household("EVOUCHER").observations[0].lexical_class == "numeric"
    raw = {
        r.name: r for r in populated.household("EVOUCHER").observations[0].raw_fields
    }
    assert raw["EVOUCHER"].value == "1"
    assert raw["AVOUCHER"].value == "0"
    assert blank.diagnostics["voucher_evidence"] == "expected_blank_not_in_universe"
    assert populated.diagnostics["voucher_evidence"] == "answer_present"
    assert blank.exclusion is None
    assert populated.exclusion == "assistance_universe_conflict"


def test_an_unreadable_voucher_answer_is_unresolved_not_a_universe_conflict():
    household = _household(_row(EVOUCHER="9", AVOUCHER="0"))
    assert household.diagnostics["voucher_evidence"] == "unresolved"
    assert household.assistance == "assistance_voucher_unresolved"
    assert household.exclusion == "assistance_voucher_unresolved"


def test_an_absent_voucher_column_is_unresolved_rather_than_admitted():
    values = dict.fromkeys(hp.PROJECTION_FIELDS, "")
    values.update(
        SSUID="0001",
        PNUM="101",
        MONTHCODE="12",
        ERESIDENCEID="100001",
        WPFINWGT="1000",
        **RENTER,
    )
    del values["EVOUCHER"]
    household = _household(m.SippRow(values))
    assert household.household("EVOUCHER").observations[0].lexical_class == "absent"
    assert household.diagnostics["voucher_evidence"] == "unresolved"
    assert household.assistance == "assistance_voucher_unresolved"


def test_an_imputed_negative_with_conflicting_voucher_evidence_stays_excluded():
    household = _household(_row(ARENTSUB="2", EVOUCHER="1", AVOUCHER="1"))
    assert household.assistance == "assistance_universe_conflict"
    assert household.assistance != "explicit_unassisted"


@pytest.mark.parametrize(
    ("overrides", "extra"),
    [
        ({"AVOUCHER": "-99"}, None),
        ({"EVOUCHER": "", "AVOUCHER": "1"}, None),
        ({}, {"ERENTSUB": "1"}),
    ],
)
def test_an_owner_with_conflicting_evidence_is_unresolved_not_out_of_universe(
    overrides, extra
):
    rows = [_row(**{**OWNER, **overrides})]
    if extra is not None:
        rows.append(_row(pnum=102, **{**OWNER, **overrides, **extra}))
    household = _household(*rows)
    assert household.tenure == "owner"
    assert household.assistance == "owner_assistance_evidence_unresolved"
    assert household.assistance != "not_in_assistance_universe"
    # The ladder still reports the owner reason; the class keeps the evidence.
    assert household.exclusion == "tenure_owner"


def test_the_recipe_states_the_only_voucher_evidence_it_admits():
    universe = hp.recipe_document()["assistance_universe"]
    assert universe["voucher_evidence_admitted_with_explicit_negative"] == (
        "expected_blank_not_in_universe"
    )
    assert universe["voucher_blank_under_flag_0_is_expected_not_missing"] is True
    assert universe["voucher_value_under_flag_0_distinguished_from_blank"] is True
    assert set(universe["voucher_evidence_states"]) == set(hm.VOUCHER_EVIDENCE_STATES)


def test_the_projection_counts_how_every_voucher_answer_was_graded():
    diagnostics = _project(
        _row(ssuid="0001", residence="100001"),
        _row(ssuid="0002", residence="100002", EVOUCHER="1", AVOUCHER="0"),
        _row(ssuid="0003", residence="100003", AVOUCHER="-99"),
    ).diagnostics
    assert diagnostics["voucher_evidence_states"] == {
        "answer_present": 1,
        "expected_blank_not_in_universe": 1,
        "suppressed": 1,
    }
    assert diagnostics["voucher_evidence_states_admitted"] == {
        "expected_blank_not_in_universe": 1
    }


def test_an_assisted_households_rent_is_not_a_full_contract_charge():
    document = hp.recipe_document()
    assert (
        document["non_claims"]["observes_full_contract_charge_for_assisted_households"]
        is False
    )


# --- allocation quality -----------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "quality"),
    [
        ("1", "reported"),
        ("2", "hot_deck_imputed"),
        ("3", "logically_imputed"),
        ("4", "model_imputed"),
        ("5", "cold_deck_imputed"),
        ("6", "range_imputed"),
        ("7", "reported_and_imputed_components"),
        ("8", "imputed_components"),
        ("9", "component_derived_allocation_unresolved"),
    ],
)
def test_every_allocation_class_stays_distinct_from_reported(flag, quality):
    household = _household(_row(ARENTMORT=flag))
    amount = household.household("TRENTMORT")
    assert hm.measurement_quality(amount) == quality
    assert hm.measurement_class(amount) == ("observed" if flag == "1" else "allocated")
    assert amount.value == Decimal(900)
    diagnostics = _project(_row(ARENTMORT=flag)).diagnostics
    assert diagnostics["rent_quality_admitted"] == {quality: 1}


def test_a_suppressed_flag_is_neither_observed_nor_a_value():
    household = _household(_row(ARENTMORT="-99"))
    amount = household.household("TRENTMORT")
    assert amount.value is None
    assert hm.measurement_class(amount) == "suppressed"


def test_an_invalid_flag_refuses_the_value():
    household = _household(_row(ARENTMORT="17"))
    amount = household.household("TRENTMORT")
    assert amount.value is None
    assert "ARENTMORT:invalid_status" in amount.reasons


# --- topcodes, boundaries and weights ---------------------------------------


def test_a_topcode_marker_is_recorded_without_inferring_a_threshold():
    household = _household(_row(TRNTMT_MED="3000", TRNTMT_STD="250"))
    markers = household.person_topcode("TRNTMT_MED")
    assert markers[0].value == Decimal(3000)
    assert markers[0].status == "topcode_group_marker"
    assert hm.measurement_class(markers[0]) == "observed_without_status_flag"
    assert household.diagnostics["rent_topcode_marked_records"] == 2
    assert household.diagnostics["rent_topcode_partial_marking"] is False
    assert (
        _project(_row(TRNTMT_MED="3000", TRNTMT_STD="250")).diagnostics["boundaries"][
            "rent_topcode_marked_households"
        ]
        == 1
    )


def test_a_partly_marked_topcode_group_is_reported_as_partial():
    household = _household(_row(TRNTMT_MED="3000"))
    assert household.diagnostics["rent_topcode_partial_marking"] is True


def test_a_domain_maximum_amount_is_counted_separately_from_a_topcode_marker():
    household = _household(_row(TRENTMORT="9999999"))
    assert household.diagnostics["rent_at_domain_maximum"] is True
    assert household.diagnostics["rent_topcode_marked_records"] == 0


def test_zero_and_unknown_weights_stay_explicit():
    zero = _household(_row(weight="0"))
    assert zero.weight.value == 0
    assert zero.exclusion == "weight_not_positive"
    unknown = _household(_row(weight=""))
    assert unknown.weight.value is None
    assert unknown.exclusion == "weight_unknown"
    diagnostics = _project(_row(weight="0")).diagnostics
    assert diagnostics["household_weights"]["zero_count"] == 1
    assert diagnostics["person_weights"]["unknown_count"] == 0


def test_the_household_weight_is_named_as_a_mean_member_weight():
    household = _household(_row(pnum=101, weight="100"), _row(pnum=102, weight="200"))
    assert household.weight.value == Decimal(150)
    assert household.weight.name == "sipp_december_mean_member_weight"
    weights = hp.recipe_document()["weight"]
    assert weights["household_statistic"] == "mean_member_WPFINWGT"
    assert weights["is_publisher_household_sampling_weight"] is False
    assert weights["is_a_rent_model_sampling_weight"] is False
    totals = _project(
        _row(pnum=101, weight="100"), _row(pnum=102, weight="200")
    ).diagnostics["source_weight_totals"]
    assert totals["statistic"] == "sum_of_household_mean_member_WPFINWGT"
    assert totals["not_a_publisher_household_sampling_weight"] is True
    assert totals["by_tenure_class"] == {"renter": "150"}


def test_an_unknown_weight_is_bucketed_rather_than_dropped_from_the_totals():
    totals = _project(_row(weight="")).diagnostics["source_weight_totals"]
    assert totals["by_tenure_class"] == {"unknown_household_weight": "0"}


# --- aggregate diagnostics --------------------------------------------------


def test_every_exclusion_reason_is_declared_in_the_recipe_order():
    rows = [
        _row(ssuid="0001", residence="100001"),
        _row(ssuid="0002", residence="100002", **OWNER),
        _row(ssuid="0003", residence="100003", ETENURE="3", EEHC_TEN="3"),
        _row(ssuid="0004", residence="100004", ERENTSUB="1"),
        _row(ssuid="0005", residence="100005", ARENTSUB="2"),
        _row(ssuid="0006", residence="100006", ERENTSUB="", ARENTSUB=""),
        _row(ssuid="0007", residence="100007", TRENTMORT=""),
        _row(ssuid="0008", residence="100008", weight="0"),
        _row(ssuid="0009", residence="100009", EVOUCHER="1", AVOUCHER="1"),
        _row(ssuid="0010", residence="100010", ETENURE="9"),
        _row(ssuid="0011", residence="100011", ERENTSUB="", ARENTSUB="0"),
        _row(ssuid="0012", residence="100012", EVOUCHER="9", AVOUCHER="0"),
        _row(ssuid="0013", residence="100013", pnum=101),
        _row(ssuid="0013", residence="100013", pnum=102, ETENURE="1"),
        _row(ssuid="0014", residence="100014", pnum=101),
        _row(ssuid="0014", residence="100014", pnum=102, ERENTSUB="1"),
        _row(ssuid="0015", residence="100015", pnum=101),
        _row(ssuid="0015", residence="100015", pnum=102, TRENTMORT="1200"),
        _row(ssuid="0016", residence="100016", weight=""),
    ]
    result = hp.project(rows, source=SOURCE, selection="all_ssuids")
    exclusions = result.diagnostics["exclusions_exclusive"]
    # Every declared rung is reachable, and nothing is reported outside them.
    assert set(exclusions) - {"admitted"} == set(hm.EXCLUSION_ORDER)
    assert exclusions == {
        "admitted": 1,
        "assistance_assisted": 1,
        "assistance_copy_conflict": 1,
        "assistance_not_in_universe": 1,
        "assistance_unassisted_allocated": 1,
        "assistance_universe_conflict": 1,
        "assistance_unknown": 1,
        "assistance_voucher_unresolved": 1,
        "rent_copy_conflict": 1,
        "rent_unknown": 1,
        "tenure_copy_conflict": 1,
        "tenure_no_payment": 1,
        "tenure_owner": 1,
        "tenure_unknown": 1,
        "weight_not_positive": 1,
        "weight_unknown": 1,
    }
    assert result.diagnostics["households"] == 16
    assert result.diagnostics["admitted_unassisted_renter_households"] == 1
    assert result.diagnostics["tenure_classes"] == {
        "no_payment": 1,
        "owner": 1,
        "renter": 12,
        "tenure_copy_conflict": 1,
        "tenure_unknown": 1,
    }
    assert result.diagnostics["household_codes"]["ETENURE"] == {
        "1": 1,
        "2": 12,
        "3": 1,
        "copy_conflict": 1,
        "unknown": 1,
    }


def test_the_receipt_carries_the_recipe_and_no_individual_amount():
    result = _project(_row(pnum=101), _row(pnum=102))
    receipt = result.receipt_json.decode()
    assert result.receipt_sha256 == hashlib.sha256(result.receipt_json).hexdigest()
    assert hp.RECIPE_SHA256 in receipt
    assert result.recipe_digest in receipt
    assert '"900"' not in receipt
    assert '"0001"' not in receipt
    assert '"101"' not in receipt
    assert len(result.diagnostics["exclusions_exclusive"]) == 1


def test_the_receipt_is_stable_for_identical_supplied_rows():
    first = _project(_row(pnum=101), _row(pnum=102))
    second = _project(_row(pnum=101), _row(pnum=102))
    assert first.receipt_json == second.receipt_json


def test_copy_conflicts_and_unknowns_are_counted_per_household_field():
    result = _project(_row(pnum=101), _row(pnum=102, TUTILS="9"))
    assert result.diagnostics["household_copy_conflicts"]["TUTILS"] == 1
    assert result.diagnostics["household_copy_conflicts"]["TRENTMORT"] == 0
    assert result.diagnostics["household_unknown_measurements"]["TUTILS"] == 1


# --- selection evidence and caller assertion (review S3) ---------------------


def test_first_32_selection_refuses_more_than_thirty_two_ssuids():
    rows = [_row(ssuid=f"{i:04d}", residence=str(100001 + i)) for i in range(40)]
    with pytest.raises(hp.HousingProjectionError, match="SELECTION_EXCEEDS_FIRST_32"):
        hp.project(rows, source=SOURCE, selection="first_32_ssuids")
    # The label was the only thing wrong: the same rows measure under all_ssuids.
    every = hp.project(rows, source=SOURCE, selection="all_ssuids")
    assert every.diagnostics["households"] == 40
    assert (
        json.loads(every.receipt_json)["selection_evidence"]["selected_ssuid_count"]
        == 40
    )
    exact = rows[:32]
    bounded = hp.project(exact, source=SOURCE, selection="first_32_ssuids")
    assert (
        json.loads(bounded.receipt_json)["selection_evidence"]["selected_ssuid_count"]
        == 32
    )


def test_the_receipt_records_the_selected_ssuid_count_and_digest():
    result = _project(
        _row(ssuid="0001", residence="100001"),
        _row(ssuid="0002", residence="100002"),
    )
    evidence = json.loads(result.receipt_json)["selection_evidence"]
    assert evidence["mode"] == "first_32_ssuids"
    assert evidence["selected_ssuid_count"] == 2
    assert evidence["first_32_ssuids_maximum"] == 32
    # The digest covers the sorted SSUID set, so row order cannot move it while
    # a different set must.
    reordered = _project(
        _row(ssuid="0002", residence="100002"),
        _row(ssuid="0001", residence="100001"),
    )
    other = _project(
        _row(ssuid="0003", residence="100003"),
        _row(ssuid="0004", residence="100004"),
    )
    assert (
        json.loads(reordered.receipt_json)["selection_evidence"][
            "selected_ssuid_digest"
        ]
        == evidence["selected_ssuid_digest"]
    )
    assert (
        json.loads(other.receipt_json)["selection_evidence"]["selected_ssuid_digest"]
        != evidence["selected_ssuid_digest"]
    )


def test_the_receipt_calls_the_selection_a_caller_assertion_not_a_proof():
    receipt = json.loads(_project(_row()).receipt_json)
    evidence = receipt["selection_evidence"]
    assert evidence["source_authentication"] == "caller_asserted"
    assert evidence["count_is_not_proof_of_a_relative_selection"] is True
    assert evidence["relative_selection_established_by"] == (
        "authenticated_runner_over_the_full_december_roster"
    )
    declared = hp.recipe_document()["selection_authentication"]
    assert declared["first_32_ssuids_maximum"] == 32
    assert declared["recorded_in_receipt_as"] == "caller_asserted"
    assert "not proof of" in declared["count_is_not_proof_of_a_relative_selection"]


def test_the_receipt_keeps_source_identity_apart_from_the_caller_assertion():
    receipt = json.loads(_project(_row()).receipt_json)
    identity = receipt["source_identity"]
    assert identity["data_sha256"] == source._PINS.sha256
    assert identity["metadata_sha256"] == source._PINS.metadata_sha256
    assert identity["survey_year"] == 2023
    assert identity["reference_year"] == 2022
    assert identity["matches_frozen_producer_pins"] is True
    # Rows are supplied, never hashed here; no byte proof is claimed from them.
    assert identity["supplied_rows_verified_against_these_bytes"] is False
    assert identity["new_source_byte_proof_claimed"] is False
    assert hp.NON_CLAIMS["new_source_byte_proof_claimed"] is False
    with pytest.raises(hp.HousingProjectionError, match="HOUSING_SOURCE"):
        hp.source_identity(
            m.SippSource("0" * 64, source._PINS.metadata_sha256, 2023, 2022)
        )


# --- implementation binding (review S4) -------------------------------------


def test_the_receipt_binds_both_housing_modules_and_both_frozen_modules():
    binding = json.loads(_project(_row()).receipt_json)["implementation"]
    assert tuple(binding["modules"]) == hp.IMPLEMENTATION_MODULES
    assert set(binding["module_sha256"]) == set(hp.IMPLEMENTATION_MODULES)
    assert set(hp.IMPLEMENTATION_MODULES) == {
        "microcosm.build.sipp_december_source",
        "microcosm.build.sipp_housing_measurements",
        "microcosm.build.sipp_housing_predictors",
        "microcosm.build.sipp_housing_projection",
        "microcosm.build.sipp_measurements",
    }
    for name, digest in binding["module_sha256"].items():
        path = pathlib.Path(sys.modules[name].__file__)
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert binding["frozen_modules_mutated"] is False
    assert binding["inside_recipe_bytes"] is False
    assert binding["dependency_versions_bound"] is False


def test_the_implementation_binding_is_never_inside_the_bytes_it_hashes():
    recipe = hp.recipe_bytes().decode()
    binding = hp.implementation_binding()
    assert binding["binding"] not in recipe
    for digest in binding["module_sha256"].values():
        assert digest not in recipe
    # The recipe still names what is bound, so the omission is declared, and the
    # pin over those bytes is unaffected by any module edit.
    declared = hp.recipe_document()["implementation_binding"]
    assert list(declared["modules"]) == list(hp.IMPLEMENTATION_MODULES)
    assert declared["inside_recipe_bytes"] is False
    assert hashlib.sha256(hp.recipe_bytes()).hexdigest() == hp.RECIPE_SHA256


def test_a_module_digest_follows_that_module_s_bytes(tmp_path, monkeypatch):
    path = tmp_path / "invented_implementation.py"
    path.write_text("RULE = 'admit'\n")
    module = types.ModuleType("invented_implementation")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, "invented_implementation", module)
    first = hp.module_digests(("invented_implementation",))
    assert first == {
        "invented_implementation": hashlib.sha256(path.read_bytes()).hexdigest()
    }
    path.write_text("RULE = 'refuse'\n")
    assert hp.module_digests(("invented_implementation",)) != first
    with pytest.raises(hp.HousingProjectionError, match="IMPLEMENTATION_MODULES"):
        hp.module_digests("microcosm.build.sipp_housing_projection")


def test_a_changed_classification_module_moves_the_receipt(monkeypatch):
    baseline = _project(_row()).receipt_sha256
    real = hp.module_digests()
    changed = {**real, "microcosm.build.sipp_housing_measurements": "0" * 64}
    monkeypatch.setattr(
        hp, "module_digests", lambda names=hp.IMPLEMENTATION_MODULES: changed
    )
    moved = _project(_row())
    assert moved.receipt_sha256 != baseline
    binding = json.loads(moved.receipt_json)["implementation"]
    assert binding["module_sha256"]["microcosm.build.sipp_housing_measurements"] == (
        "0" * 64
    )
    # The recipe pin is untouched: a rule change moves the receipt, not the pin.
    assert hashlib.sha256(hp.recipe_bytes()).hexdigest() == hp.RECIPE_SHA256


# --- topcode marker states (review S5) --------------------------------------


def test_a_zero_utilities_marker_is_in_domain_and_counted_as_a_zero():
    household = _household(_row(TUTILS_MED="0", TUTILS_STD="0"))
    marker = household.person_topcode("TUTILS_MED")[0]
    assert marker.value == 0
    assert marker.status == "topcode_group_marker"
    assert household.diagnostics["utilities_topcode_marked_records"] == 2
    assert household.diagnostics["utilities_topcode_marker_zero_records"] == 2
    assert household.diagnostics["utilities_topcode_marker_blank_records"] == 0
    assert household.diagnostics["utilities_topcode_marker_invalid_domain_records"] == 0


def test_a_zero_rent_marker_is_outside_its_published_minimum_of_one():
    household = _household(_row(TRNTMT_MED="0", TRNTMT_STD="0"))
    assert household.diagnostics["rent_topcode_marked_records"] == 0
    assert household.diagnostics["rent_topcode_marker_invalid_domain_records"] == 2
    assert household.diagnostics["rent_topcode_marker_zero_records"] == 0
    contracts = {e["name"]: e for e in hp.field_contracts()}
    assert contracts["TRNTMT_MED"]["minimum"] == 1
    assert contracts["TUTILS_MED"]["minimum"] == 0


def test_marker_states_aggregate_without_scoring_a_zero_as_a_cleared_topcode():
    diagnostics = _project(
        _row(ssuid="0001", residence="100001", TUTILS_MED="0", TUTILS_STD="0"),
        _row(ssuid="0002", residence="100002"),
        _row(ssuid="0003", residence="100003", TRNTMT_MED="0"),
    ).diagnostics
    boundaries = diagnostics["boundaries"]
    assert boundaries["utilities_topcode_marker_zero_records"] == 2
    assert boundaries["utilities_topcode_marker_zero_households"] == 1
    assert boundaries["utilities_topcode_marker_blank_records"] == 4
    assert boundaries["utilities_topcode_marked_households"] == 1
    assert boundaries["utilities_topcode_partial_marking_households"] == 0
    assert boundaries["rent_topcode_marker_invalid_domain_records"] == 1
    assert boundaries["rent_topcode_marker_blank_records"] == 5
    assert boundaries["rent_topcode_marked_households"] == 0
    markers = hp.recipe_document()["topcode_markers"]
    assert markers["threshold_inferred"] is False
    assert markers["trntmt_minimum"] == 1 and markers["tutils_minimum"] == 0
    assert (
        "no data row has been read"
        in (markers["zero_marker_is_not_a_cleared_non_topcode"])
    )


# --- month, residence, tenure and universe cases (review S6) ----------------


def test_a_november_row_cannot_contradict_the_december_measurement():
    households = hm.december_housing(
        [
            _row(pnum=101),
            _row(
                pnum=101,
                MONTHCODE="11",
                ETENURE="1",
                TRENTMORT="5",
                TUTILS="7",
                ERENTSUB="1",
                EVOUCHER="1",
                AVOUCHER="1",
            ),
        ],
        source=SOURCE,
        complete_membership=True,
    )
    assert len(households) == 1
    household = households[0]
    assert len(household.members) == 1
    assert household.tenure == "renter"
    assert household.household("TRENTMORT").value == Decimal(900)
    assert household.household("TUTILS").value == Decimal(150)
    assert household.assistance == "explicit_unassisted"
    assert household.exclusion is None


def test_one_ssuid_with_two_residences_becomes_two_classified_households():
    households = hm.december_housing(
        [
            _row(pnum=101, residence="100001"),
            _row(pnum=102, residence="100002", **OWNER),
        ],
        source=SOURCE,
        complete_membership=True,
    )
    assert [h.key for h in households] == [("0001", 100001), ("0001", 100002)]
    assert [h.tenure for h in households] == ["renter", "owner"]
    assert [h.exclusion for h in households] == [None, "tenure_owner"]
    assert [len(h.members) for h in households] == [1, 1]
    diagnostics = hp.project(
        [
            _row(pnum=101, residence="100001"),
            _row(pnum=102, residence="100002", **OWNER),
        ],
        source=SOURCE,
        selection="first_32_ssuids",
    ).diagnostics
    assert diagnostics["households"] == 2
    assert diagnostics["people"] == 2


def test_an_owner_reporting_no_debt_is_recorded_as_owner_debt_absent():
    overrides = {**OWNER, "EPRDEBT": "2", "APRDEBT": "1", "TRENTMORT": ""}
    household = _household(_row(**overrides))
    assert household.diagnostics["owner_debt"] == "owner_debt_absent"
    assert household.diagnostics["owner_debt_evidence"] == {
        "EPRDEBT": "observed",
        "EMHDEBT": "not_in_universe",
    }
    assert household.rent.value is None
    diagnostics = _project(_row(**overrides)).diagnostics
    assert diagnostics["owner_debt_resolution"] == {"owner_debt_absent": 1}
    assert diagnostics["owner_debt_evidence_classes"]["EPRDEBT"] == {"observed": 1}
    assert diagnostics["owner_debt_evidence_classes"]["EMHDEBT"] == {
        "not_in_universe": 1
    }


def test_an_assisted_household_with_no_member_receipt_is_counted():
    diagnostics = _project(
        _row(ERENTSUB="1", EVOUCHER="1", AVOUCHER="1", EEHC_RENTSUB="2")
    ).diagnostics
    assert diagnostics["no_person_month_receipt_in_assisted_households"] == 1
    assert diagnostics["person_month_receipt_in_explicit_unassisted_households"] == 0
    assert diagnostics["exclusions_exclusive"] == {"assistance_assisted": 1}


def test_no_payment_tenure_keeps_its_assistance_class_but_excludes_on_tenure():
    household = _household(_row(ETENURE="3", EEHC_TEN="3", ERENTSUB="2", ARENTSUB="1"))
    assert household.tenure == "no_payment"
    assert household.assistance == "explicit_unassisted"
    assert household.exclusion == "tenure_no_payment"
    assert household.rent.value is None
    assert "no_payment_tenure_outside_trentmort_universe" in household.rent.reasons


def test_a_renter_amount_outside_its_flag_universe_reaches_rent_unknown():
    household = _household(_row(ARENTMORT="0"))
    assert household.tenure == "renter"
    assert household.assistance == "explicit_unassisted"
    assert household.household("TRENTMORT").value is None
    assert household.rent.value is None
    assert "ARENTMORT:not_in_universe" in household.rent.reasons
    assert household.exclusion == "rent_unknown"
    assert household.diagnostics["rent_amount_class"] == "not_in_universe"


def test_person_month_evidence_never_reaches_the_household_measurements():
    household = _household(
        _row(EEHC_RENTSUB="1", AEHC_RENTSUB="1", EEHC_VOUCHER="1", AEHC_VOUCHER="1")
    )
    assert {v.concept for v in household.household_measurements} == set(
        hm.HOUSEHOLD_CATEGORIES
    ) | set(hm.HOUSEHOLD_AMOUNTS)
    for name in hm.PERSON_MONTH_CATEGORIES:
        with pytest.raises(hm.HousingMeasurementError, match="UNKNOWN_HOUSEHOLD_FIELD"):
            household.household(name)
    assert household.diagnostics["eehc_rentsub_receipt_members"] == 1
    assert household.assistance == "explicit_unassisted"
    assert household.exclusion is None
    assert (
        hp.recipe_document()["assistance_universe"][
            "person_month_indicators_determine_household_class"
        ]
        is False
    )


@pytest.mark.parametrize(
    "rentsub",
    [
        {"ERENTSUB": "", "ARENTSUB": "0"},
        {"ERENTSUB": "", "ARENTSUB": "-99"},
        {"ERENTSUB": "", "ARENTSUB": ""},
    ],
)
def test_a_published_voucher_answer_conflicts_whatever_erentsub_carries(rentsub):
    # EVOUCHER needs ERENTSUB = 1 behind it. A discarded value under flag 0 is
    # still a published answer, so the conflict does not depend on the flag.
    household = _household(_row(**rentsub, EVOUCHER="1", AVOUCHER="0"))
    assert household.diagnostics["voucher_evidence"] == "answer_present"
    assert household.assistance == "assistance_universe_conflict"
    assert household.exclusion == "assistance_universe_conflict"
    without = _household(_row(**rentsub))
    assert without.assistance != "assistance_universe_conflict"


# --- version 3 predictor transcription --------------------------------------


@pytest.mark.parametrize(
    ("field", "flag", "page", "grain", "length"),
    [
        ("TEHC_ST", "AEHC_ST", 3310, "person_month", 2),
        ("RHNUMPER", "AHNUMPER", 890, "household_concept_person_month", 2),
        ("RHNUMU18", "AHNUMU18", 891, "household_concept_person_month", 2),
        ("THTOTINC", "AHTOTINC", 2865, "household_concept_person_month", 12),
        ("TAGE_EHC", None, 796, "person_month", 2),
    ],
)
def test_each_predictor_field_carries_its_reread_locator_flag_and_grain(
    field, flag, page, grain, length
):
    entry = next(e for e in hp.field_contracts() if e["name"] == field)
    assert entry["status_flag"] == flag
    assert entry["dictionary_zero_based_page"] == page
    assert entry["dictionary_printed_page"] == page + 1
    assert entry["grain"] == grain
    assert entry["file_class"] == "Person-month"
    assert entry["dictionary_length"] == length
    assert entry["prospective_predictor"] is True
    assert entry["participates_in_exclusion_order"] is False
    assert entry["universe"] == "THHLDSTATUS in (1,2,3,4)"
    assert entry["limitations"]


def test_the_state_domain_is_the_published_fifty_three_code_set():
    entry = next(e for e in hp.field_contracts() if e["name"] == "TEHC_ST")
    codes = entry["codes"]
    assert len(codes) == 53
    # A range would admit 3, 7, 14, 43, 52 and 57-59; the published list does not.
    assert not {3, 7, 14, 43, 52, 57, 58, 59} & set(codes)
    assert codes[:3] == [1, 2, 4] and codes[-2:] == [60, 61]
    assert entry["code_labels"]["60"] == "Puerto Rico and Island Areas"
    assert entry["code_labels"]["61"] == "Foreign Country"
    geography = hp.recipe_document()["prospective_predictors"]["state_geography"]
    assert geography["published_code_count"] == 53
    assert geography["state_and_dc_code_count"] == 51
    assert geography["codes_that_are_not_a_state_or_dc"] == [60, 61]
    assert geography["domain_is_a_published_code_set_not_the_range_01_56"] is True


@pytest.mark.parametrize("code", ["3", "07", "43", "52", "57", "62", "0"])
def test_an_unpublished_state_code_refuses_the_value(code):
    household = _project(_row(TEHC_ST=code)).predictors[0]
    state = household.household("TEHC_ST")
    assert state.value is None
    assert hm.measurement_class(state) == "unknown"
    assert household.status == "predictor_unknown"


def test_the_income_domain_is_signed_and_its_sentinel_lies_outside_it():
    entry = next(e for e in hp.field_contracts() if e["name"] == "THTOTINC")
    assert entry["minimum"] == -10_000_000_000
    assert entry["maximum"] == 10_000_000_000
    assert entry["unit"] == "USD_nominal_2022_monthly"
    assert entry["kind"] == "amount"
    sentinel = entry["out_of_domain_sentinel"]
    assert not entry["minimum"] <= sentinel <= entry["maximum"]
    assert entry["documented_numeric_missing_sentinel"] is None
    # Every predictor amount's sentinel must sit outside its own domain.
    for name, spec in predictors._PREDICTOR_AMOUNTS.items():
        assert not spec.low <= spec.sentinel <= spec.high, name


def test_a_sentinel_inside_a_published_domain_refuses_rather_than_hides_a_value():
    broken = hm._Amount(
        "AHTOTINC",
        -10_000_000_000,
        10_000_000_000,
        "household_concept_person_month",
        "invented",
        "invented",
        "invented",
        "invented",
        0,
        unit="USD_nominal_2022_monthly",
        sentinel=hm._UNDOCUMENTED_SENTINEL,
    )
    with pytest.raises(
        hm.HousingMeasurementError, match="SENTINEL_INSIDE_DOMAIN:THTOTINC"
    ):
        hm._amount_lexical({"THTOTINC": "5"}, "THTOTINC", broken)


def test_the_age_column_publishes_no_status_flag_and_no_allocation_quality():
    entry = next(e for e in hp.field_contracts() if e["name"] == "TAGE_EHC")
    assert entry["status_flag"] is None
    assert entry["minimum"] == 0 and entry["maximum"] == 90
    assert entry["kind"] == "age"
    age = _project(_row()).predictors[0].person_age_measurements[0]
    assert hm.measurement_quality(age) == "no_published_status_flag"
    assert hm.measurement_class(age) == "observed_without_status_flag"
    assert age.observations[0].origin == "published_source_value_without_status_flag"
    declared = hp.recipe_document()["prospective_predictors"]["age"]
    assert declared["allocation_quality_available"] is False
    assert declared["domain_maximum_called_a_topcode"] is False


def test_the_projection_carries_thirty_eight_declared_columns():
    assert len(hp.PROJECTION_FIELDS) == 38
    assert len(set(hp.PROJECTION_FIELDS)) == 38
    assert hp.PROJECTION_FIELDS[-9:] == predictors.PREDICTOR_FIELDS
    assert set(hp.recipe_document()["fields"]) == set(hp.PROJECTION_FIELDS)
    # field_contracts() refuses any column that is not declared or flagged.
    assert {e["name"] for e in hp.field_contracts()} <= set(hp.PROJECTION_FIELDS)


def test_columns_left_out_are_declared_absent_rather_than_silently_dropped():
    absent = hp.recipe_document()["prospective_predictors"][
        "fields_deliberately_absent"
    ]
    assert absent["TEHC_METRO"] == "metro_status_unused_in_this_increment"
    assert absent["AEHC_METRO"] == "metro_status_unused_in_this_increment"
    for variant in ("RHNUMPERWT2", "RHNUMU18WT2", "THTOTINCT2"):
        assert absent[variant] == "type_2_inclusive_variant_not_carried"
    assert absent["THHLDSTATUS"].startswith("published_universe_gate")
    assert not set(absent) & set(hp.PROJECTION_FIELDS)


# --- version 3 predictor grain and copy agreement ---------------------------


def test_a_predictor_copy_disagreement_is_a_conflict_and_never_a_majority():
    result = _project(
        _row(pnum=101, TEHC_ST="24"),
        _row(pnum=102, TEHC_ST="24"),
        _row(pnum=103, TEHC_ST="51"),
    )
    state = result.predictors[0].household("TEHC_ST")
    assert state.value is None
    assert "household_copy_conflict" in state.reasons
    assert hm.measurement_class(state) == "copy_conflict"
    assert result.predictors[0].status == "predictor_copy_conflict"
    diagnostics = result.diagnostics["prospective_predictors"]
    assert diagnostics["predictor_copy_conflicts"]["TEHC_ST"] == 1
    assert diagnostics["predictor_copy_conflicts"]["THTOTINC"] == 0


def test_a_predictor_copy_conflict_never_changes_the_household_exclusion():
    conflicted = _project(
        _row(pnum=101, THTOTINC="4200"), _row(pnum=102, THTOTINC="9900")
    )
    clean = _project(_row(pnum=101), _row(pnum=102))
    assert conflicted.predictors[0].status == "predictor_copy_conflict"
    assert clean.predictors[0].status == "predictor_complete"
    assert conflicted.households[0].exclusion is None
    assert (
        conflicted.diagnostics["exclusions_exclusive"]
        == clean.diagnostics["exclusions_exclusive"]
        == {"admitted": 1}
    )


def test_an_absent_predictor_column_is_distinct_from_a_blank_one():
    absent = _project(_bare_row()).predictors[0]
    blank = _project(_row(THTOTINC="", AHTOTINC="")).predictors[0]
    assert absent.household("THTOTINC").value is None
    assert blank.household("THTOTINC").value is None
    raw_absent = absent.household("THTOTINC").observations[0].raw_fields[0]
    raw_blank = blank.household("THTOTINC").observations[0].raw_fields[0]
    assert (raw_absent.present, raw_absent.input_type) == (False, "absent")
    assert (raw_blank.present, raw_blank.value) == (True, "")
    assert "THTOTINC:absent" in absent.household("THTOTINC").reasons
    assert "THTOTINC:blank" in blank.household("THTOTINC").reasons


# --- version 3 zero, universe, unknown and signed values --------------------


def test_zero_children_under_18_stays_an_observed_value():
    household = _project(_row(RHNUMU18="0")).predictors[0]
    measure = household.household("RHNUMU18")
    assert measure.value == 0
    assert hm.measurement_class(measure) == "observed"
    assert measure.observations[0].lexical_class == "numeric_zero"
    assert household.status == "predictor_complete"


def test_a_zero_household_size_is_outside_its_published_minimum_of_one():
    measure = _project(_row(RHNUMPER="0")).predictors[0].household("RHNUMPER")
    assert measure.value is None
    assert measure.observations[0].lexical_class == "invalid_domain"


def test_a_negative_household_income_is_preserved_as_a_signed_observation():
    measure = _project(_row(THTOTINC="-1750")).predictors[0].household("THTOTINC")
    assert measure.value == Decimal("-1750")
    assert hm.measurement_class(measure) == "observed"
    assert _project(_row(THTOTINC="-1750")).diagnostics["prospective_predictors"][
        "income_sign"
    ] == {"negative": 1}


def test_a_ten_million_dollar_loss_is_a_value_not_a_missing_sentinel():
    # The housing kernel's default out-of-domain token is -10,000,000, which is
    # a legitimate THTOTINC amount. A per-field sentinel keeps it a value.
    measure = _project(_row(THTOTINC="-10000000")).predictors[0].household("THTOTINC")
    assert measure.value == Decimal("-10000000")
    assert measure.observations[0].lexical_class == "numeric"


def test_an_income_below_the_published_minimum_is_out_of_domain():
    measure = (
        _project(_row(THTOTINC="-10000000001")).predictors[0].household("THTOTINC")
    )
    assert measure.value is None
    assert measure.observations[0].lexical_class == "invalid_domain"


def test_a_not_in_universe_predictor_flag_yields_no_value_and_its_own_status():
    result = _project(_row(RHNUMPER="", AHNUMPER="0"))
    household = result.predictors[0]
    measure = household.household("RHNUMPER")
    assert measure.value is None
    assert hm.measurement_class(measure) == "not_in_universe"
    assert household.status == "predictor_not_in_universe"
    assert result.diagnostics["prospective_predictors"][
        "predictor_status_exclusive"
    ] == {"predictor_not_in_universe": 1}


@pytest.mark.parametrize(
    ("flag", "quality"),
    [
        (2, "hot_deck_imputed"),
        (3, "logically_imputed"),
        (4, "model_imputed"),
        (5, "cold_deck_imputed"),
        (6, "range_imputed"),
        (7, "reported_and_imputed_components"),
        (8, "imputed_components"),
        (9, "component_derived_allocation_unresolved"),
    ],
)
def test_every_predictor_allocation_class_stays_distinct_from_reported(flag, quality):
    result = _project(_row(AHTOTINC=str(flag)))
    measure = result.predictors[0].household("THTOTINC")
    assert measure.value == Decimal("4200")
    assert hm.measurement_class(measure) == "allocated"
    assert hm.measurement_quality(measure) == quality == m._QUALITY[flag]
    assert result.diagnostics["prospective_predictors"]["predictor_field_quality"][
        "THTOTINC"
    ] == {quality: 1}
    assert result.diagnostics["prospective_predictors"]["predictor_field_classes"][
        "THTOTINC"
    ] == {"allocated": 1}


def test_a_suppressed_predictor_flag_is_neither_observed_nor_a_value():
    household = _project(_row(AEHC_ST="-99")).predictors[0]
    measure = household.household("TEHC_ST")
    assert measure.value is None
    assert hm.measurement_class(measure) == "suppressed"
    # Withheld evidence keeps its own state rather than joining plain unknowns.
    assert household.status == "predictor_suppressed"


def test_a_publisher_count_that_conflicts_is_kept_apart_from_an_unknown_one():
    conflict = _project(
        _row(pnum=101, RHNUMPER="2"), _row(pnum=102, RHNUMPER="3")
    ).predictors[0]
    unknown = _project(_row(RHNUMPER="", AHNUMPER="")).predictors[0]
    assert conflict.diagnostics["household_size_agreement"] == "publisher_conflict"
    assert unknown.diagnostics["household_size_agreement"] == "publisher_unknown"
    assert set(predictors.DISAGREEMENT_STATES) == {
        "agree",
        "disagree",
        "publisher_conflict",
        "publisher_unknown",
        "roster_coverage_incomplete",
    }


# --- version 3 age universe and coverage ------------------------------------


def test_the_maximum_age_uses_only_admissible_person_observations():
    result = _project(
        _row(pnum=101, TAGE_EHC="44", RHNUMPER="3", RHNUMU18="1"),
        _row(pnum=102, TAGE_EHC="71", RHNUMPER="3", RHNUMU18="1"),
        _row(pnum=103, TAGE_EHC="9", RHNUMPER="3", RHNUMU18="1"),
    )
    household = result.predictors[0]
    assert household.maximum_member_age.value == 71
    assert household.maximum_member_age.status == "household_maximum_member_age"
    assert household.diagnostics["age_coverage"] == "complete"
    assert household.diagnostics["age_covered_members"] == 3
    assert household.diagnostics["maximum_age_is_lower_bound"] is False
    assert household.diagnostics["roster_december_members_under_18"] == 1
    assert household.diagnostics["children_under_18_agreement"] == "agree"


def test_partial_age_coverage_yields_a_declared_lower_bound_not_a_maximum():
    result = _project(
        _row(pnum=101, TAGE_EHC="44", RHNUMPER="2"),
        _row(pnum=102, TAGE_EHC="", RHNUMPER="2"),
    )
    household = result.predictors[0]
    assert household.maximum_member_age.value == 44
    assert (
        household.maximum_member_age.status
        == "household_maximum_member_age_lower_bound"
    )
    assert "age_coverage:partial" in household.maximum_member_age.reasons
    assert household.diagnostics["age_coverage"] == "partial"
    assert household.diagnostics["maximum_age_is_lower_bound"] is True
    # The child roster counterpart is withheld, not guessed, and the household
    # cannot be prospectively complete.
    assert household.diagnostics["roster_december_members_under_18"] is None
    assert (
        household.diagnostics["children_under_18_agreement"]
        == "roster_coverage_incomplete"
    )
    assert household.status == "predictor_age_coverage_incomplete"
    assert hp.recipe_document()["prospective_predictors"]["age"][
        "partial_coverage_is_a_lower_bound_not_a_maximum"
    ]


def test_no_admissible_age_leaves_the_household_maximum_unknown():
    household = _project(_row(TAGE_EHC="")).predictors[0]
    assert household.maximum_member_age.value is None
    assert household.maximum_member_age.status == "unknown_household_maximum_member_age"
    assert household.diagnostics["age_coverage"] == "none"
    assert household.status == "predictor_age_coverage_incomplete"


def test_an_age_at_the_published_maximum_is_recorded_without_calling_it_a_topcode():
    result = _project(_row(TAGE_EHC="90"))
    household = result.predictors[0]
    assert household.maximum_member_age.value == 90
    assert household.diagnostics["maximum_age_at_domain_maximum"] is True
    diagnostics = result.diagnostics["prospective_predictors"]
    assert diagnostics["maximum_age_at_domain_maximum_households"] == 1
    declared = hp.recipe_document()["prospective_predictors"]["age"]
    assert declared["domain_maximum"] == 90
    assert declared["domain_maximum_called_a_topcode"] is False
    assert (
        hp.recipe_document()["non_claims"]["age_domain_maximum_called_a_topcode"]
        is False
    )
    # The topcode marker surface belongs to TRENTMORT and TUTILS only; age adds
    # nothing to it.
    assert set(hm.PERSON_TOPCODE_AMOUNTS) == {
        "TRNTMT_MED",
        "TRNTMT_STD",
        "TUTILS_MED",
        "TUTILS_STD",
    }
    assert "TAGE_EHC" not in hm.PERSON_TOPCODE_AMOUNTS


# --- version 3 publisher versus roster counts -------------------------------


def test_the_publisher_and_roster_household_sizes_are_both_retained():
    result = _project(
        _row(pnum=101, RHNUMPER="3"),
        _row(pnum=102, RHNUMPER="3"),
    )
    household = result.predictors[0]
    assert household.household("RHNUMPER").value == 3
    assert household.diagnostics["roster_december_member_count"] == 2
    assert household.diagnostics["household_size_agreement"] == "disagree"
    assert household.diagnostics["household_size_difference"] == 1
    # A disagreement is a diagnostic, never a correction and never an exclusion.
    assert household.status == "predictor_complete"
    assert result.households[0].exclusion is None
    diagnostics = result.diagnostics["prospective_predictors"]
    assert diagnostics["household_size_agreement"] == {"disagree": 1}
    assert diagnostics["household_size_difference_counts"] == {"1": 1}
    declared = hp.recipe_document()["prospective_predictors"]["household_size"]
    assert declared["both_retained"] is True
    assert declared["publisher_corrected_against_roster"] is False
    assert declared["roster_corrected_against_publisher"] is False
    assert declared["type_2_person_treatment_adjudicated"] is False


def test_an_unknown_publisher_size_is_reported_apart_from_a_disagreement():
    result = _project(_row(RHNUMPER="", AHNUMPER=""))
    assert result.predictors[0].diagnostics["household_size_agreement"] == (
        "publisher_unknown"
    )
    assert result.diagnostics["prospective_predictors"]["household_size_agreement"] == {
        "publisher_unknown": 1
    }


# --- version 3 separation from the measurement admission ladder --------------


def test_the_admission_ladder_is_identical_with_and_without_the_predictor_columns():
    def ladder(builder):
        rows = [
            builder(ssuid="0001", residence="100001"),
            builder(ssuid="0002", residence="100002", **OWNER),
            builder(ssuid="0003", residence="100003", ETENURE="3", EEHC_TEN="3"),
            builder(ssuid="0004", residence="100004", ERENTSUB="1"),
            builder(ssuid="0005", residence="100005", ARENTSUB="2"),
            builder(ssuid="0006", residence="100006", ERENTSUB="", ARENTSUB=""),
            builder(ssuid="0007", residence="100007", TRENTMORT=""),
            builder(ssuid="0008", residence="100008", weight="0"),
            builder(ssuid="0009", residence="100009", EVOUCHER="1", AVOUCHER="1"),
            builder(ssuid="0010", residence="100010", ETENURE="9"),
            builder(ssuid="0011", residence="100011", ERENTSUB="", ARENTSUB="0"),
            builder(ssuid="0012", residence="100012", EVOUCHER="9", AVOUCHER="0"),
        ]
        result = hp.project(rows, source=SOURCE, selection="all_ssuids")
        return (
            result.diagnostics["exclusions_exclusive"],
            result.diagnostics["admitted_unassisted_renter_households"],
            result.diagnostics["assistance_classes"],
            result.diagnostics["tenure_classes"],
            [h.exclusion for h in result.households],
        )

    assert ladder(_row) == ladder(_bare_row)


def test_the_predictor_status_ladder_is_exclusive_and_declared():
    rows = [
        _row(ssuid="0001", residence="100001"),
        _row(ssuid="0002", residence="100002", pnum=101, THTOTINC="4200"),
        _row(ssuid="0002", residence="100002", pnum=102, THTOTINC="9900"),
        _row(ssuid="0003", residence="100003", RHNUMPER="", AHNUMPER="0"),
        _row(ssuid="0004", residence="100004", TEHC_ST=""),
        _row(ssuid="0005", residence="100005", TAGE_EHC=""),
        _row(ssuid="0006", residence="100006", AHTOTINC="-99"),
    ]
    result = hp.project(rows, source=SOURCE, selection="all_ssuids")
    diagnostics = result.diagnostics["prospective_predictors"]
    assert diagnostics["predictor_status_exclusive"] == {
        "predictor_age_coverage_incomplete": 1,
        "predictor_complete": 1,
        "predictor_copy_conflict": 1,
        "predictor_not_in_universe": 1,
        "predictor_suppressed": 1,
        "predictor_unknown": 1,
    }
    assert set(diagnostics["predictor_status_exclusive"]) - {"predictor_complete"} == (
        set(predictors.PREDICTOR_STATUS_ORDER)
    )
    assert diagnostics["predictor_status_order_exclusive"] == list(
        predictors.PREDICTOR_STATUS_ORDER
    )
    # Not one of the six is excluded by the measurement ladder.
    assert result.diagnostics["exclusions_exclusive"] == {"admitted": 6}


def test_predictor_validity_is_reported_separately_for_admitted_households():
    rows = [
        _row(ssuid="0001", residence="100001"),
        _row(ssuid="0002", residence="100002", **OWNER),
        _row(ssuid="0003", residence="100003", TEHC_ST=""),
    ]
    result = hp.project(rows, source=SOURCE, selection="all_ssuids")
    diagnostics = result.diagnostics["prospective_predictors"]
    assert diagnostics["households"] == 3
    assert diagnostics["admitted_households"] == 2
    assert diagnostics["predictor_status_exclusive"] == {
        "predictor_complete": 2,
        "predictor_unknown": 1,
    }
    assert diagnostics["predictor_status_exclusive_admitted"] == {
        "predictor_complete": 1,
        "predictor_unknown": 1,
    }


def test_the_recipe_declares_that_no_predictor_state_joined_the_exclusion_order():
    declared = hp.recipe_document()["prospective_predictors"]
    assert declared["exclusion_order_unchanged"] is True
    assert declared["predictor_states_added_to_the_exclusion_order"] == []
    assert declared["exclusion_order_exclusive"] == list(hm.EXCLUSION_ORDER)
    assert declared["computed_after_admission"] is True
    assert not set(hm.EXCLUSION_ORDER) & set(predictors.PREDICTOR_STATUS_ORDER)
    separation = _project(_row()).diagnostics["prospective_predictors"][
        "separation_from_the_measurement_ladder"
    ]
    assert separation["predictor_values_read_by_the_exclusion_rule"] == []
    assert separation["predictors_computed_after_admission"] is True


def test_the_exclusion_rule_takes_no_predictor_argument():
    # A structural check, not a description: _exclusion's signature is the whole
    # set of things that can decide an admission.
    parameters = inspect.signature(hm._exclusion).parameters
    assert tuple(parameters) == ("tenure", "assistance", "rent", "weight")


# --- version 3 identity, digest and replay ----------------------------------


def test_the_recipe_moved_to_version_three_and_records_what_it_supersedes():
    assert hp.RECIPE_VERSION == "sipp-december-housing-projection/3"
    assert hp.RECEIPT_FORMAT == "sipp_december_housing_projection_v3"
    assert hp.SUPERSEDED_RECIPE_VERSIONS == (
        "sipp-december-housing-projection/1",
        "sipp-december-housing-projection/2",
    )
    document = hp.recipe_document()
    assert document["version"] == hp.RECIPE_VERSION
    assert document["superseded_versions"] == list(hp.SUPERSEDED_RECIPE_VERSIONS)
    assert hashlib.sha256(hp.recipe_bytes()).hexdigest() == hp.RECIPE_SHA256


def test_the_receipt_binds_the_predictor_module_and_carries_a_predictor_digest():
    result = _project(_row())
    receipt = json.loads(result.receipt_json)
    assert (
        "microcosm.build.sipp_housing_predictors"
        in receipt["implementation"]["module_sha256"]
    )
    assert (
        receipt["implementation"]["module_sha256"][
            "microcosm.build.sipp_housing_predictors"
        ]
        == hashlib.sha256(pathlib.Path(predictors.__file__).read_bytes()).hexdigest()
    )
    assert receipt["predictor_digest"] != receipt["measurement_digest"]
    # The recipe names what is bound; the digest itself stays out of the bytes
    # it would otherwise have to hash.
    recipe = hp.recipe_bytes().decode()
    assert "microcosm.build.sipp_housing_predictors" in recipe
    assert (
        receipt["implementation"]["module_sha256"][
            "microcosm.build.sipp_housing_predictors"
        ]
        not in recipe
    )
    assert receipt["predictor_digest"] not in recipe
    # The binding domain moved with the module set, so a version 2 binding over
    # the four-module set cannot be mistaken for this one.
    digests = hp.module_digests()
    assert receipt["implementation"]["binding"] == source._key(
        "sipp-december-housing-implementation-binding/2", digests
    )
    four = {k: v for k, v in digests.items() if "predictors" not in k}
    assert receipt["implementation"]["binding"] != source._key(
        "sipp-december-housing-implementation-binding/1", four
    )


def test_a_moved_predictor_value_moves_the_predictor_digest_and_the_receipt():
    first = json.loads(_project(_row(TAGE_EHC="44")).receipt_json)
    second = json.loads(_project(_row(TAGE_EHC="45")).receipt_json)
    assert first["predictor_digest"] != second["predictor_digest"]
    # The measurement surface the admission ladder read did not move with it.
    assert first["measurement_digest"] == second["measurement_digest"]
    assert first["membership_digest"] == second["membership_digest"]
    assert first["recipe_sha256"] == second["recipe_sha256"]


def test_the_predictor_pass_refuses_a_roster_it_did_not_measure():
    rows = [_row(ssuid="0001", residence="100001")]
    other = hm.december_housing(
        [_row(ssuid="0002", residence="100002")],
        source=SOURCE,
        complete_membership=True,
    )
    with pytest.raises(
        predictors.HousingPredictorError, match="PREDICTOR_ROSTER_DISAGREEMENT"
    ):
        predictors.december_predictors(
            rows, other, source=SOURCE, complete_membership=True
        )


def test_predictor_diagnostics_refuse_households_they_were_not_measured_against():
    rows = [_row(ssuid="0001", residence="100001")]
    measured = hm.december_housing(rows, source=SOURCE, complete_membership=True)
    computed = predictors.december_predictors(
        rows, measured, source=SOURCE, complete_membership=True
    )
    other = hm.december_housing(
        [_row(ssuid="0002", residence="100002")],
        source=SOURCE,
        complete_membership=True,
    )
    with pytest.raises(
        predictors.HousingPredictorError, match="PREDICTOR_HOUSEHOLD_ALIGNMENT"
    ):
        predictors.predictor_diagnostics(computed, other)


def test_the_receipt_is_byte_identical_cold_and_on_a_fresh_warm_replay():
    # Fresh-warm: the modules are already imported and have been used by every
    # test above, so a second projection over the same rows must reproduce the
    # first exactly.
    warm_first = _project(_row(pnum=101), _row(pnum=102, TAGE_EHC="9"))
    warm_second = _project(_row(pnum=101), _row(pnum=102, TAGE_EHC="9"))
    assert warm_first.receipt_json == warm_second.receipt_json

    # Cold: a process that has never imported these modules must land on the
    # same bytes, so nothing in the receipt depends on import order or on state
    # this session accumulated.
    script = """
import hashlib, json, sys
from microcosm.build import sipp_december_source as source
from microcosm.build import sipp_housing_projection as hp
from microcosm.build import sipp_measurements as m

payload = json.loads(sys.argv[1])
rows = [m.SippRow(values) for values in payload]
result = hp.project(
    rows,
    source=m.SippSource(
        source._PINS.sha256, source._PINS.metadata_sha256, 2023, 2022
    ),
    selection="first_32_ssuids",
)
print(hashlib.sha256(result.receipt_json).hexdigest())
"""
    rows = [
        dict(_row(pnum=101).values),
        dict(_row(pnum=102, TAGE_EHC="9").values),
    ]
    cold = subprocess.run(
        [sys.executable, "-c", script, json.dumps(rows)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert cold.stdout.strip() == warm_first.receipt_sha256


# --- version 3 non-claims ----------------------------------------------------


def test_no_bridge_fit_draw_or_model_binding_is_declared_or_performed():
    document = hp.recipe_document()
    assert document["prospective_predictors"]["not_performed"] == [
        "price_bridge",
        "annualization",
        "fit",
        "draw",
        "donor_mask_selection",
        "model_input_binding",
        "recipient_mapping",
        "policyengine_calculation",
    ]
    income = document["prospective_predictors"]["income"]
    assert income["period"] == "December_2022_monthly"
    assert income["annualization"] == "not_applied"
    assert income["price_rebasing"] == "not_applied"
    assert income["log_or_floor_transform"] == "not_applied"
    assert document["non_claims"]["monthly_income_annualized_or_rebased"] is False
    assert document["non_claims"]["predictor_validity_is_a_fit_admission"] is False
    assert document["non_claims"]["launch_readiness_declared"] is False
    assert document["period_and_basis_declarations"]["annualization"] == "not_applied"
    # The income observation itself is the published monthly amount, untouched.
    assert _project(_row()).predictors[0].household("THTOTINC").value == Decimal("4200")


def test_no_sub_state_geography_is_derived_and_the_non_state_codes_stay_visible():
    document = hp.recipe_document()
    geography = document["prospective_predictors"]["state_geography"]
    assert geography["sub_state_geography_published_on_this_column"] is False
    assert geography["sub_state_values_derived"] is False
    assert geography["geography_is_source_geography"] is True
    assert document["non_claims"]["sub_state_geography_derived"] is False
    result = _project(
        _row(ssuid="0001", residence="100001", TEHC_ST="24"),
        _row(ssuid="0002", residence="100002", TEHC_ST="60"),
        _row(ssuid="0003", residence="100003", TEHC_ST="61"),
    )
    diagnostics = result.diagnostics["prospective_predictors"]
    assert diagnostics["state_of_residence_codes"] == {"24": 1, "60": 1, "61": 1}
    assert diagnostics["state_published_non_state_area_households"] == 2
    assert diagnostics["state_codes_that_are_not_a_state_or_dc"] == [60, 61]


def test_the_state_arm_of_the_eehc_rentsub_universe_is_observed_without_a_reclass():
    with_state = _project(
        _row(ssuid="0001", residence="100001", TEHC_ST="24"),
        _row(ssuid="0002", residence="100002", TEHC_ST="61"),
    )
    without_state = _project(
        _bare_row(ssuid="0001", residence="100001"),
        _bare_row(ssuid="0002", residence="100002"),
    )
    diagnostics = with_state.diagnostics["prospective_predictors"]
    assert diagnostics["eehc_rentsub_state_arm"] == {
        "state_code_above_60": 1,
        "state_code_at_most_60": 1,
    }
    assert diagnostics["eehc_rentsub_state_arm_changes_no_classification"] is True
    for key in ("assistance_classes", "person_month_codes", "exclusions_exclusive"):
        assert with_state.diagnostics[key] == without_state.diagnostics[key]
