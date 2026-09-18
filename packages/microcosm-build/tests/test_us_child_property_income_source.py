"""Pure invented source semantics: no donor fit, native reads or owner issuance."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_child_property_income_source as child
from microcosm.build.us_runtime import survey_population_domains as domains


def literal_inputs(ages=(15, 16, 17, 14), weights=(100, 200, 300, 400)):
    rows, homes = [], []
    for index, (age, weight) in enumerate(zip(ages, weights, strict=True), 1):
        key = domains.HouseholdKey(domains.Source.ASEC, 2024, 2025, f"{index:05d}")
        person = domains.AsecPerson(f"{index:022d}", "1", str(age), "2", "source", key)
        homes.append(
            domains.AsecHousehold(key, "1", "6", "1", "1", str(weight), (person,))
        )
        raw = {
            name: "0"
            for name in (*child.interest.READ_COLUMNS, *child.dividend.READ_COLUMNS)
        }
        raw.update(
            PERIDNUM=person.peridnum,
            PH_SEQ=str(index),
            A_LINENO="1",
            A_AGE=str(age),
            INT_YN="2" if age >= 15 else "0",
            DIV_YN="2" if age >= 15 else "0",
        )
        rows.append(raw)
    frame = pd.DataFrame(
        rows,
        index=pd.Index(
            np.arange(2**53 + 1, 2**53 + 1 + len(rows), dtype=np.int64),
            name="native_person_id",
        ),
    )
    return (
        frame[list(child.interest.READ_COLUMNS)].copy(),
        frame[list(child.dividend.READ_COLUMNS)].copy(),
        tuple(homes),
    )


def test_full_observed_pair_uses_ordinary_interest_without_retirement_or_rental():
    interest, dividend, homes = literal_inputs()
    interest.loc[interest.index[0], ["INT_YN", "INT_VAL", "TRDINT_VAL"]] = [
        "1",
        "200",
        "100",
    ]
    dividend.loc[dividend.index[0], ["DIV_YN", "DIV_VAL"]] = ["1", "50"]
    result = child.project_child_property_donors(interest, dividend, homes)
    assert len(result.donors) == 3
    assert result.donors.iloc[0][child.TARGETS[0]] == 100  # Combined INT_VAL is 200.
    assert result.donors.iloc[0][child.TARGETS[1]] == 50
    assert result.interest.RINT_VAL1_amount.isna().all()
    assert result.donors.original_household_design_weight.tolist() == [1, 2, 3]
    assert result.donors.donor_key.iloc[0] == (
        "asec",
        2024,
        2025,
        "00001",
        "0000000000000000000001",
        "1",
    )
    assert result.diagnostics.reason.iloc[-1] == "outside_donor_age_band"


def test_observed_ordinary_zero_and_allocation_provenance_are_retained():
    interest, dividend, homes = literal_inputs()
    interest.loc[interest.index[0], ["INT_YN", "INT_VAL", "I_INTYN", "TTRDINT_VAL"]] = [
        "1",
        "200",
        "1",
        "1",
    ]
    dividend.loc[dividend.index[0], ["DIV_YN", "DIV_VAL", "I_DIVYN", "TDIV_VAL"]] = [
        "1",
        "50",
        "2",
        "1",
    ]
    result = child.project_child_property_donors(interest, dividend, homes)
    assert (
        result.interest.TRDINT_VAL_reporting_status.iloc[0] == "observed_zero_component"
    )
    assert result.donors.iloc[0][child.TARGETS[0]] == 0
    assert result.donors.iloc[0][child.TARGETS[1]] == 50
    assert result.dividend.TDIV_VAL_code.iloc[0] == 1
    assert result.interest.TTRDINT_VAL_code.iloc[0] == 1
    assert result.dividend.I_DIVYN_code.iloc[0] == 2


def test_zero_weight_observed_donor_is_retained_as_zero_mass_diagnostic():
    interest, dividend, homes = literal_inputs(weights=(0, 200, 300, 400))
    result = child.project_child_property_donors(interest, dividend, homes)
    assert len(result.donors) == 3
    assert result.diagnostics.observed_donor.iloc[0]
    assert not result.diagnostics.positive_mass_donor.iloc[0]
    assert result.diagnostics.reason.iloc[0] == "zero_design_mass"
    assert result.diagnostics.observed_zero_pair.iloc[0]


@pytest.mark.parametrize(
    "change", ["age", "line", "household", "peridnum", "missing", "duplicate"]
)
def test_full_catalogue_coordinate_or_coverage_mismatch_refuses(change):
    interest, dividend, homes = literal_inputs()
    if change == "missing":
        homes = homes[:-1]
    elif change == "duplicate":
        homes = (*homes, homes[0])
    else:
        field, value = {
            "age": ("A_AGE", "16"),
            "line": ("A_LINENO", "2"),
            "household": ("PH_SEQ", "5"),
            "peridnum": ("PERIDNUM", "other"),
        }[change]
        interest.loc[interest.index[0], field] = value
        dividend.loc[dividend.index[0], field] = value
    with pytest.raises(ValueError, match="CHILD_PROPERTY_"):
        child.project_child_property_donors(interest, dividend, homes)


@pytest.mark.parametrize("token", ["15.5", " 15", "100", "-1", ""])
def test_imprecise_or_invalid_source_age_is_never_truncated(token):
    interest, dividend, homes = literal_inputs()
    interest.loc[interest.index[0], "A_AGE"] = token
    dividend.loc[dividend.index[0], "A_AGE"] = token
    with pytest.raises(ValueError, match="AGE"):
        child.project_child_property_donors(interest, dividend, homes)


@pytest.mark.parametrize(
    "token,status",
    [
        ("", "missing_amount"),
        ("bad", "invalid_amount_literal"),
        ("100", "contradictory_no_nonzero"),
    ],
)
def test_missing_and_source_error_donor_exclusions_remain_distinct(token, status):
    interest, dividend, homes = literal_inputs()
    interest.loc[interest.index[0], "TRDINT_VAL"] = token
    result = child.project_child_property_donors(interest, dividend, homes)
    assert result.interest.TRDINT_VAL_reporting_status.iloc[0] == status
    assert result.diagnostics.reason.iloc[0] == (
        "unresolved_observed_pair" if token == "" else "source_review_required"
    )
    assert interest.index[0] not in result.donors.index


def recipient_inputs():
    interest, dividend, homes = literal_inputs(ages=(14, 15, 16, 17))
    projected = child.project_child_property_donors(interest, dividend, homes)
    a = projected.interest.iloc[:1].copy()
    b = projected.dividend.iloc[:1].copy()
    native = int(a.index[0])
    for table in (a, b):
        table["native_person_id"] = np.array([native], dtype=np.int64)
        table.index = pd.Index([11], dtype="int64", name="person_id")
    acs = pd.DataFrame(
        dict(
            native_person_id=[22],
            AGEP=["9"],
            INTP=[""],
            property_income_status=["outside_universe_blank"],
            property_income_adjustment_known=[True],
        ),
        index=pd.Index([12], dtype="int64", name="person_id"),
    )
    columns = [
        "person_id",
        "source",
        "source_year",
        "survey_year",
        "raw_native_household_id",
        "raw_native_person_id",
        "native_line_numeric_original",
        "selected_receiving_person_id",
    ]
    origins = {
        "persons": {
            "columns": columns,
            "rows": [
                [
                    11,
                    "asec",
                    2024,
                    2025,
                    "00001",
                    "0000000000000000000001",
                    "1",
                    native,
                ],
                [12, "acs", 2024, 2024, "2024HU0000001", "1", "1", 22],
            ],
        }
    }
    return origins, a, b, acs


def test_only_explicitly_unmeasured_children_are_eligible():
    origins, a, b, acs = recipient_inputs()
    before = a.copy(deep=True)
    result = child.project_child_property_recipients(origins, a, b, acs)
    assert result.eligible.tolist() == [True, True]
    assert (
        not result.ordinary_source_known.any()
        and not result.dividend_source_known.any()
    )
    pd.testing.assert_frame_equal(a, before)


@pytest.mark.parametrize(
    "change",
    [
        "acs_adjustment",
        "acs_numeric_zero",
        "acs_negative",
        "asec_positive",
        "asec_malformed",
        "known_incumbent",
    ],
)
def test_child_source_errors_and_known_incumbents_are_never_imputation_eligibility(
    change,
):
    origins, a, b, acs = recipient_inputs()
    if change == "acs_adjustment":
        acs.loc[12, "property_income_adjustment_known"] = False
    elif change in ("acs_numeric_zero", "acs_negative"):
        acs.loc[12, "INTP"] = "0" if change == "acs_numeric_zero" else "-10"
        acs.loc[12, "property_income_status"] = "outside_universe_observation"
    elif change == "asec_positive":
        a.loc[11, "TRDINT_VAL_published_amount"] = 7
        a.loc[11, "TRDINT_VAL_reporting_status"] = (
            "contradictory_outside_reporting_universe"
        )
    elif change == "asec_malformed":
        a.loc[11, "TRDINT_VAL_literal_status"] = "malformed"
    else:
        a.loc[11, "TRDINT_VAL_amount_known"] = True
    result = child.project_child_property_recipients(origins, a, b, acs)
    target = 12 if change.startswith("acs") else 11
    assert not result.loc[target, "eligible"]


def test_complete_private_seal_checks_exact_tuple_coordinates():
    i, d, h = literal_inputs()
    donor = child.project_child_property_donors(i, d, h)
    origins, a, b, acs = recipient_inputs()
    value = child.QualifiedChildPropertySources(
        donor, child.project_child_property_recipients(origins, a, b, acs), {}
    )
    before = child.child_property_sources_seal(value)
    changed = donor.donors.copy(deep=True)
    changed.at[changed.index[0], "donor_key"] = (
        "asec",
        2024,
        2025,
        "00001",
        2**53 + 1,
        "1",
    )
    altered = replace(value, donor_projection=replace(donor, donors=changed))
    assert child.child_property_sources_seal(altered) != before


def test_recipient_axis_ceiling_refuses_at_a_patched_down_value(monkeypatch):
    """MAX_RECIPIENT_ROWS bounds the stacked-person axis the projection classifies.

    MAX_ROWS stays the ASEC source's own row bound. The recipient axis is the
    selection's stacked persons, 3,565,013 at full source, so it takes the
    row-count rule's 15,000,000. A patched-down value refuses RECIPIENT_AXIS
    before any frame is joined, and the value one row higher admits.
    """
    assert child.MAX_RECIPIENT_ROWS == 15_000_000
    assert child.MAX_RECIPIENT_ROWS == -(-4 * 3_565_013 // 1_000_000) * 1_000_000
    origins, a, b, acs = recipient_inputs()
    rows = len(origins["persons"]["rows"])
    monkeypatch.setattr(child, "MAX_RECIPIENT_ROWS", rows - 1)
    with pytest.raises(ValueError, match="RECIPIENT_AXIS"):
        child.project_child_property_recipients(origins, a, b, acs)
    monkeypatch.setattr(child, "MAX_RECIPIENT_ROWS", rows)
    result = child.project_child_property_recipients(origins, a, b, acs)
    assert len(result) == rows
