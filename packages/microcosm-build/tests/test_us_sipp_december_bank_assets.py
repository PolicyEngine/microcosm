"""Invented December bank-asset fixtures only; no source bytes, model or fit claim.

Proposed for packages/microcosm-build/tests/test_us_sipp_december_bank_assets.py.
Covers `december_bank_assets`: dictionary page 191 TVAL_BANK (0...99,999,999, missing
-999, status AVAL_BANK, publisher recode including CDs) with page 797 TAGE_EHC
(0...90, missing -999) required >= 15 at MONTHCODE 12. The SippSource below is a valid
fixture identity, never authenticated parent bytes. Reason spellings belong to the
implementation: these tests assert each unknown route keeps its own distinct reason.
"""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from microcosm.build.sipp_measurements import (
    SIPP_2023_MASTER_SHA256,
    MeasurementError,
    SippRow,
    SippSource,
    december_bank_assets,
)

SOURCE = SippSource("a" * 64, SIPP_2023_MASTER_SHA256, 2023, 2022)
ABSENT = object()


def row(person="101", month=12, ssuid="sample-A", **values):
    fields = {
        "SSUID": ssuid,
        "PNUM": person,
        "MONTHCODE": str(month),
        "TAGE_EHC": "39",
        "TVAL_BANK": "5000",
        "AVAL_BANK": "1",
        **values,
    }
    return SippRow({k: v for k, v in fields.items() if v is not ABSENT})


def one(**values):
    (measure,) = december_bank_assets([row(**values)], source=SOURCE)
    return measure


def stated(measure):
    """Reasons wherever the helper records them; placement is not part of this contract."""
    return frozenset(measure.reasons).union(
        r for o in measure.observations for r in o.reasons
    )


def test_known_bank_measurement_identity_and_december_person_evidence():
    measure = one()
    assert measure.value == Decimal(5000)
    assert measure.name == "sipp_december_bank_assets"
    assert measure.concept == "sipp_tval_bank"
    assert measure.rule_id == "sipp_december_bank_assets_v1"
    assert measure.basis == (
        "reference_year_end_financial_institution_assets_including_CDs"
    )
    assert measure.unit == "USD_nominal_2022"
    assert measure.source == SOURCE
    assert not stated(measure)
    assert len(measure.observations) == 1
    obs = measure.observations[0]
    assert (obs.person, obs.month, obs.field) == (("sample-A", 101), 12, "TVAL_BANK")
    assert obs.origin == "published_source_value"
    assert {"TVAL_BANK", "AVAL_BANK", "TAGE_EHC"} <= {f.name for f in obs.raw_fields}


def test_person_order_is_string_ssuid_then_integer_pnum_and_keeps_unknowns():
    rows = [
        row(ssuid="sample-9", person="102"),
        row(ssuid="sample-10", person="102", TVAL_BANK=""),
        row(ssuid="sample-9", person="101"),
    ]
    got = december_bank_assets(rows, source=SOURCE)
    assert [m.observations[0].person for m in got] == [
        ("sample-10", 102),
        ("sample-9", 101),
        ("sample-9", 102),
    ]
    assert [m.value for m in got] == [None, Decimal(5000), Decimal(5000)]


@pytest.mark.parametrize("age", ["15", "90", "14", "91", "-999", "", None, ABSENT])
def test_december_age_universe_boundaries_gate_every_known_value(age):
    known = age in ("15", "90")
    measure = one(TAGE_EHC=age)
    assert (measure.value == Decimal(5000)) is known
    assert bool(stated(measure)) is not known
    assert any(f.name == "TAGE_EHC" for f in measure.observations[0].raw_fields)


def test_below_universe_age_with_applicable_status_states_a_contradiction():
    measure = one(TAGE_EHC="14")
    obs = measure.observations[0]
    assert measure.value is None
    assert (obs.value, obs.flag, obs.lexical_class) == (Decimal(5000), 1, "numeric")
    assert stated(measure) not in (stated(one(TAGE_EHC="")), stated(one(TVAL_BANK="")))
    assert one(TAGE_EHC="14", TVAL_BANK="", AVAL_BANK="0").value is None


@pytest.mark.parametrize(
    "amount", ["0", "99999999", "-999", "", None, ABSENT, "100000000", "-1", "1.5"]
)
def test_amount_domain_sentinel_and_lexical_classes(amount):
    expected = Decimal(amount) if amount in ("0", "99999999") else None
    measure = one(TVAL_BANK=amount)
    assert measure.value == expected
    assert bool(stated(measure)) is (expected is None)
    assert any(f.name == "TVAL_BANK" for f in measure.observations[0].raw_fields)


def test_published_numeric_zero_is_not_an_invented_structural_zero():
    obs = one(TVAL_BANK="0").observations[0]
    assert (obs.value, obs.lexical_class) == (Decimal(0), "numeric_zero")
    assert obs.origin == "published_source_value"
    assert "structural" not in obs.origin


@pytest.mark.parametrize("flag", range(1, 10))
def test_applicable_status_including_nine_is_published_not_a_fit_admission(flag):
    measure = one(AVAL_BANK=str(flag))
    obs = measure.observations[0]
    assert measure.value == Decimal(5000)
    assert (obs.flag, obs.origin) == (flag, "published_source_value")
    assert (obs.quality == "reported") is (flag == 1)
    assert (obs.quality == "component_derived_allocation_unresolved") is (flag == 9)
    assert measure.rule_id == "sipp_december_bank_assets_v1"
    assert not any("fit" in name for name in vars(measure))


@pytest.mark.parametrize("flag", ["0", "-99", "10", "", None, ABSENT, "abc"])
def test_inapplicable_status_keeps_numeric_evidence_but_yields_no_value(flag):
    measure = one(AVAL_BANK=flag)
    obs = measure.observations[0]
    assert measure.value is None
    assert measure.reasons
    assert (obs.value, obs.lexical_class) == (Decimal(5000), "numeric")
    with pytest.raises(MeasurementError):
        measure.require_value()


def test_each_unknown_route_keeps_its_own_distinct_reason():
    cases = {
        "blank_amount": dict(TVAL_BANK=""),
        "null_amount": dict(TVAL_BANK=None),
        "absent_amount": dict(TVAL_BANK=ABSENT),
        "sentinel_amount": dict(TVAL_BANK="-999"),
        "unparsable_amount": dict(TVAL_BANK="abc"),
        "out_of_domain_amount": dict(TVAL_BANK="100000000"),
        "unresolved_age": dict(TAGE_EHC=""),
        "below_universe_age": dict(TAGE_EHC="14"),
        "not_in_universe_status": dict(AVAL_BANK="0"),
        "suppressed_status": dict(AVAL_BANK="-99"),
        "invalid_status": dict(AVAL_BANK="10"),
    }
    reasons = {name: stated(one(**kw)) for name, kw in cases.items()}
    assert all(reasons.values())
    assert len(set(reasons.values())) == len(cases)


def test_valid_non_december_rows_are_skipped_after_key_validation():
    rows = [row(person="102", month=6), row(person="101"), row(person="103", month=11)]
    (measure,) = december_bank_assets(rows, source=SOURCE)
    assert measure.observations[0].person == ("sample-A", 101)


def test_invalid_or_duplicate_supplied_person_months_refuse_before_any_skip():
    beside_december = (
        row(month=13),
        row(month=6, person="100"),
        row(month=6, person=None),
        row(month=6, ssuid=""),
        row(),
    )
    for other in beside_december:
        with pytest.raises(MeasurementError):
            december_bank_assets([row(), other], source=SOURCE)
    for pair in ((row(), row(person="0101")), (row(month=6), row(month="6.0"))):
        with pytest.raises(MeasurementError):
            december_bank_assets(list(pair), source=SOURCE)


def test_supplied_mapping_is_snapshotted_and_results_are_immutable():
    rows = [row()]
    rows[0].values["unrelated_wide_column"] = ["not retained"]
    (measure,) = december_bank_assets(rows, source=SOURCE)
    rows[0].values["TVAL_BANK"] = "99999999"
    rows[0].values["TAGE_EHC"] = "14"
    raw = {f.name: f for f in measure.observations[0].raw_fields}
    assert measure.value == Decimal(5000)
    assert (raw["TVAL_BANK"].value, raw["TVAL_BANK"].present) == ("5000", True)
    assert "unrelated_wide_column" not in raw
    with pytest.raises(FrozenInstanceError):
        measure.value = Decimal(0)


def test_caller_contradiction_refuses_an_otherwise_known_amount():
    supplied = replace(row(), contradictions=("bank_amount_conflicts_with_components",))
    (measure,) = december_bank_assets([supplied], source=SOURCE)
    assert measure.value is None
    assert "bank_amount_conflicts_with_components" in stated(measure)
    assert measure.observations[0].value == Decimal(5000)
    with pytest.raises(MeasurementError, match="INVALID_CONTRADICTION_EVIDENCE"):
        december_bank_assets([replace(row(), contradictions=["x"])], source=SOURCE)


def test_source_identity_and_year_bridge_refuse():
    calls = (
        dict(source=replace(SOURCE, reference_year=2024)),
        dict(source=replace(SOURCE, metadata_sha256="b" * 64)),
        dict(source=replace(SOURCE, data_sha256="bad")),
        dict(source=SOURCE, target_year=2024),
    )
    for call in calls:
        with pytest.raises(MeasurementError):
            december_bank_assets([row()], **call)
