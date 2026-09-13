"""Invented donor values only; no source admission, model fitting or finalizer."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import full_puf_enrichment as full

PROFILES = (
    full.FULL65,
    full.PUF59,
    full.PUF55_SURVEY_SS,
    full.PUF55_SURVEY_SS_NO_TOTAL,
)


def _donor(profile):
    columns = (
        *profile.predictors,
        *profile.targets,
        "weight",
        *profile.donor_auxiliary_columns,
    )
    donor = pd.DataFrame(
        1.0,
        index=pd.Index([11, 33, 22], dtype="int64", name="RECID"),
        columns=columns,
    )
    donor[profile.predictors[1]] = 2.0
    donor["weight"] = [1.0, 0.0, 2.0]
    for column in profile.tax_unit_outputs:
        if column in full.support._PUF_TAX_DETAIL_DISCRETE_TAX_UNIT_OUTPUTS:
            donor[column] = 0.0
    return donor


@pytest.mark.parametrize("profile", PROFILES)
def test_existing_profiles_share_exact_selected_donor_check(profile):
    donor = _donor(profile)
    before = donor.copy(deep=True)
    selected = full._validated_model_donor(donor, profile=profile)
    pd.testing.assert_frame_equal(
        selected,
        donor.loc[:, [*profile.predictors, *profile.targets, "weight"]],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(donor, before, check_exact=True)
    assert tuple(selected.columns) == (*profile.predictors, *profile.targets, "weight")
    assert selected.index.tolist() == [11, 33, 22]
    assert selected.weight.tolist() == [1.0, 0.0, 2.0]


@pytest.mark.parametrize(
    "defect",
    (
        "missing",
        "numeric_string",
        "unknown",
        "nonfinite",
        "integer_range",
        "incidence_capacity",
        "boolean_count",
        "year",
    ),
)
def test_complete_historical_donor_domains_are_retained(defect):
    profile = full.FULL65 if defect == "year" else full.PUF55_SURVEY_SS
    donor = _donor(profile)
    value_column = profile.predictors[2]
    if defect == "missing":
        donor = donor.drop(columns="weight")
    elif defect in {"numeric_string", "unknown", "nonfinite", "integer_range"}:
        donor[value_column] = donor[value_column].astype(object)
        donor.loc[11, value_column] = {
            "numeric_string": "1",
            "unknown": np.nan,
            "nonfinite": np.inf,
            "integer_range": 2**53 + 1,
        }[defect]
    elif defect == "incidence_capacity":
        donor.loc[11, profile.donor_auxiliary_columns[0]] = 1.5
    elif defect == "boolean_count":
        column = next(
            c
            for c in profile.person_outputs
            if c in full.support._PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS
        )
        donor.loc[11, column] = 2.0  # Explicit incidence capacity is one, size two.
    else:
        column = next(
            c
            for c in profile.tax_unit_outputs
            if c in full.support._PUF_TAX_DETAIL_DISCRETE_TAX_UNIT_OUTPUTS
        )
        donor.loc[11, column] = 999.0
    with pytest.raises(
        ValueError,
        match="PUF_(FULL_DONOR_ROSTER|PHYSICAL_TYPE|UNKNOWN|NONFINITE|FLOAT64_INTEGER_RANGE|DONOR_MODEL_DOMAIN|BOOLEAN_COUNT_DOMAIN|YEAR_DOMAIN)",
    ):
        full._validated_model_donor(donor, profile=profile)


def test_donor_rejection_still_precedes_recipient_access():
    profile = full.PUF55_SURVEY_SS
    donor = _donor(profile).drop(columns="weight")
    # None is deliberately not a receiving Frame: this call must fail on the
    # earlier donor roster exactly as the original combined boundary did.
    with pytest.raises(ValueError, match="^PUF_FULL_DONOR_ROSTER$"):
        full.prepare_full_puf_inputs(None, donor, predictor_known=None, profile=profile)


def test_selected_values_are_detached_from_auxiliary_source_table():
    donor = _donor(full.PUF55_SURVEY_SS)
    before = donor.copy(deep=True)
    selected = full._validated_model_donor(donor, profile=full.PUF55_SURVEY_SS)
    selected.iloc[0, 2] = 900.0
    pd.testing.assert_frame_equal(donor, before, check_exact=True)


def test_excluded_ss_outputs_are_not_accepted_as_extra_donor_columns():
    donor = _donor(full.PUF55_SURVEY_SS)
    donor[full.SURVEY_SS_COMPONENTS[0]] = 0.0
    with pytest.raises(ValueError, match="^PUF_FULL_DONOR_ROSTER$"):
        full._validated_model_donor(donor, profile=full.PUF55_SURVEY_SS)


def test_eight_route_selection_keeps_targets_recids_weights_and_six_money_values():
    nine = _donor(full.PUF55_SURVEY_SS)
    eight = nine.drop(columns=full.SURVEY_SS_TOTAL_PREDICTOR)
    selected_nine = full._validated_model_donor(nine, profile=full.PUF55_SURVEY_SS)
    selected_eight = full._validated_model_donor(
        eight, profile=full.PUF55_SURVEY_SS_NO_TOTAL
    )
    pd.testing.assert_frame_equal(
        selected_eight,
        selected_nine.drop(columns=full.SURVEY_SS_TOTAL_PREDICTOR),
        check_exact=True,
    )
    assert len(full.PUF55_SURVEY_SS_NO_TOTAL.targets) == 55
