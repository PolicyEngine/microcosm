"""Pure participation controls on invented observations and complete tiny Frames."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.asec_housing_status import (
    DERIVED_COLUMNS,
    RAW_COLUMNS,
    _codebook,
    classify_synthetic_housing_status,
)
from microcosm.build.us_runtime.housing_participation import (
    HOUSING_PARTICIPATION_ASSUMPTIONS,
    HousingParticipationError,
    observed_participation,
    route_participation,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def observations():
    """Receipt, negative, unknown, NIU, conflict, allocated receipt and GQ."""
    rows = []
    for i, changes in enumerate(
        (
            {},
            {"HPUBLIC": 2, "HLORENT": 2},
            {"HPUBLIC": 0},
            {"HPUBLIC": 0, "H_TENURE": 1},
            {"HLORENT": 2},
            {"I_HPUBLI": 1},
            {"HRHTYPE": 9},
            {"HPUBLIC": 2, "HLORENT": 1},
        ),
        start=1,
    ):
        rows.append(
            {
                "household_id": i,
                "income_year": 2024,
                "survey_year": 2025,
                "H_SEQ": i,
                "H_TENURE": 2,
                "HRHTYPE": 1,
                "H_HHTYPE": 1,
                "HPUBLIC": 1,
                "HLORENT": 0,
                "I_HPUBLI": 0,
                "I_HLOREN": 0,
                **changes,
            }
        )
    classified = classify_synthetic_housing_status(
        pd.DataFrame(rows, columns=RAW_COLUMNS, dtype="int64")
    )
    return pd.DataFrame(
        {name: classified.array(name) for name in ("household_id", *DERIVED_COLUMNS)}
    )


def tiny_frame():
    # The first person is a nonhead in another SPM unit of assisted household 10.
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [10, 10, 10, 20],
            "person_spm_unit_id": [101, 100, 100, 200],
            "person_tax_unit_id": [101, 100, 100, 200],
            "person_family_id": [101, 100, 100, 200],
            "person_marital_unit_id": [101, 100, 100, 200],
            "is_household_head": [False, True, False, True],
        },
        index=[7, 7, 4, 2],
    )
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": [10, 20]}, index=[5, 5]),
        **{
            entity: pd.DataFrame({f"{entity}_id": [100, 101, 200]}, index=[9, 9, 3])
            for entity in ("spm_unit", "tax_unit", "family", "marital_unit")
        },
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([2.0, 3.0]), WeightKind.DESIGN)},
    )


def resolved_receipt():
    return pd.Series(
        [False, True], index=pd.Index([20, 10], name="household_id"), dtype="boolean"
    )


def test_observations_preserve_unknown_niu_conflict_quality_and_gq():
    source = observations()
    before = source.copy(deep=True)
    result = observed_participation(source)
    assert result.receipt.dtype == pd.BooleanDtype()
    assert result.receipt.iloc[[0, 1, 5, 6, 7]].tolist() == [
        True,
        False,
        True,
        True,
        True,
    ]
    assert result.receipt.iloc[2:5].isna().all()
    pd.testing.assert_frame_equal(result.evidence, source.set_index("household_id"))
    assert (
        result.evidence.loc[6, "public_quality"] == _codebook()["quality"]["allocated"]
    )
    assert (
        result.evidence.loc[7, "group_quarters"] == _codebook()["group_quarters"]["gq"]
    )
    pd.testing.assert_frame_equal(source, before)
    result.evidence.iloc[0, 0] = 99
    pd.testing.assert_frame_equal(source, before)


def test_graph_promoted_prefixed_observations_match_native_status():
    source = observations()
    prefixed = source.rename(
        columns={name: "asec_housing_observed_" + name for name in DERIVED_COLUMNS}
    ).astype("int64")
    pd.testing.assert_series_equal(
        observed_participation(prefixed).receipt, observed_participation(source).receipt
    )


def test_observed_receipt_uses_native_ids_not_duplicate_index_labels():
    source = observations().iloc[::-1].copy()
    source.index = [1] * len(source)
    expected = observed_participation(observations()).receipt.sort_index()
    pd.testing.assert_series_equal(
        observed_participation(source).receipt.sort_index(), expected
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("receipt_valid", 0),
        ("status", 99),
        ("receipt_valid", 2),
        ("conflicts", 1),
        ("route", 3),
        ("unknown_reasons", 1),
        ("group_quarters", 3),
    ],
)
def test_contradictory_or_out_of_domain_evidence_refuses(column, value):
    source = observations()
    source.loc[0, column] = value
    with pytest.raises(HousingParticipationError):
        observed_participation(source)


def test_invalid_missing_and_ambiguous_source_ids_or_aliases_refuse():
    for source in (
        observations().assign(household_id=1),
        observations().assign(household_id=np.nan),
        observations().assign(asec_housing_observed_status=1),
        observations().drop(columns="receipt_valid"),
    ):
        with pytest.raises(HousingParticipationError):
            observed_participation(source)


def test_assumptions_are_explicit_and_immutable():
    assert tuple(name for name, _ in HOUSING_PARTICIPATION_ASSUMPTIONS) == (
        "A1",
        "A2",
        "A3",
        "A4",
    )
    assert isinstance(HOUSING_PARTICIPATION_ASSUMPTIONS, tuple)
    assert all(isinstance(pair, tuple) for pair in HOUSING_PARTICIPATION_ASSUMPTIONS)


def test_head_spm_routes_once_and_preserves_all_inputs():
    frame = tiny_frame()
    original_strata = frame.strata
    before = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    receipt = resolved_receipt()
    receipt_before = receipt.copy(deep=True)
    result = route_participation(frame, receipt)
    assert result.index.tolist() == [100, 101, 200]
    assert result.dtypes.tolist() == [np.dtype("bool"), np.dtype("bool")]
    assert result.receives_housing_assistance.tolist() == [True, False, False]
    assert result.takes_up_housing_assistance_if_eligible.tolist() == [
        True,
        False,
        False,
    ]
    for entity, table in before.items():
        pd.testing.assert_frame_equal(frame.table(entity), table)
    pd.testing.assert_series_equal(receipt, receipt_before)
    assert frame.strata is original_strata


@pytest.mark.parametrize("cap", [None, 0, 1_000_000])
def test_route_is_invariant_to_cap_absence_or_value(cap):
    frame = tiny_frame()
    if cap is not None:
        frame.person["SPM_CAPHOUSESUB"] = cap
    assert route_participation(
        frame, resolved_receipt()
    ).receives_housing_assistance.tolist() == [True, False, False]


def test_person_permutation_and_duplicate_index_labels_do_not_select_first_person():
    frame = tiny_frame()
    frame.person.iloc[:] = frame.person.iloc[[3, 2, 0, 1]].to_numpy()
    pd.testing.assert_frame_equal(
        route_participation(frame, resolved_receipt()),
        route_participation(tiny_frame(), resolved_receipt()),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_household",
        "duplicate_person",
        "missing_id",
        "noninteger_id",
        "multiple_heads",
        "missing_head",
        "nonboolean_head",
        "cross_household_spm",
        "orphan_spm",
    ],
)
def test_invalid_native_identity_membership_or_head_refuses(mutation):
    frame = tiny_frame()
    if mutation == "duplicate_household":
        frame.table("household")["household_id"] = [10, 10]
    elif mutation == "duplicate_person":
        frame.person["person_id"] = [1, 1, 3, 4]
    elif mutation == "missing_id":
        frame.person["person_id"] = [1.0, 2.0, 3.0, np.nan]
    elif mutation == "noninteger_id":
        frame.person["person_id"] = [1.0, 2.0, 3.0, 4.5]
    elif mutation == "multiple_heads":
        frame.person["is_household_head"] = [True, True, False, True]
    elif mutation == "missing_head":
        frame.person["is_household_head"] = [False, False, False, True]
    elif mutation == "nonboolean_head":
        frame.person["is_household_head"] = [0, 1, 0, 1]
    elif mutation == "cross_household_spm":
        frame.person["person_household_id"] = [10, 10, 20, 20]
    else:
        frame.table("spm_unit").loc[42] = [999]
    with pytest.raises(HousingParticipationError):
        route_participation(frame, resolved_receipt())


@pytest.mark.parametrize(
    "receipt",
    [
        pd.Series([True, pd.NA], index=[10, 20], dtype="boolean"),
        pd.Series([1, 0], index=[10, 20]),
        pd.Series([True, False], index=[10, 10]),
        pd.Series([True], index=[10]),
        pd.Series([True, False], index=[10, 30]),
    ],
)
def test_unresolved_nonboolean_or_wrong_household_receipts_refuse(receipt):
    with pytest.raises(HousingParticipationError):
        route_participation(tiny_frame(), receipt)


def test_distinct_clone_households_preserve_receipt_without_double_awards():
    frame = tiny_frame()
    # Household 20 represents a separately identified clone; native head routing
    # must not collapse it with the first household's source identity.
    frame.table("household")["household_source_id"] = [77, 77]
    frame.table("household")["household_support_clone_index"] = [0, 1]
    result = route_participation(
        frame, pd.Series([True, True], index=[10, 20], dtype="boolean")
    )
    assert result.receives_housing_assistance.tolist() == [True, False, True]


def test_explicit_false_gq_exclusion_can_be_headless_without_changing_observations():
    frame = tiny_frame()
    frame.table("household")["housing_participation_universe"] = [
        "occupied_housing_unit",
        "group_quarters",
    ]
    frame.person["is_household_head"] = [False, True, False, False]
    result = route_participation(frame, resolved_receipt())
    assert result.receives_housing_assistance.tolist() == [True, False, False]
    # Source status remains a separate positive observation even for GQ.
    observed = observed_participation(observations())
    assert observed.receipt.loc[7]


@pytest.mark.parametrize(
    "universe,receipt", [("group_quarters", True), ("unknown", False), (None, False)]
)
def test_positive_gq_or_ambiguous_model_universe_refuses(universe, receipt):
    frame = tiny_frame()
    frame.table("household")["housing_participation_universe"] = [
        "occupied_housing_unit",
        universe,
    ]
    frame.person["is_household_head"] = [False, True, False, False]
    with pytest.raises(HousingParticipationError):
        route_participation(
            frame, pd.Series([True, receipt], index=[10, 20], dtype="boolean")
        )
