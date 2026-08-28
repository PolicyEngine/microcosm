from __future__ import annotations

import copy
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
)
from microcosm.build.uk_runtime.cgt_structure import (
    CGT_CLONE_MASS_CHANGE_REASON,
    DONOR_BAND_COUNT,
    DONOR_TOTAL,
    DONORS_PER_BAND,
    HOUSEHOLD_IS_CGT_BAND_DONOR,
    HOUSEHOLD_IS_CGT_CLONE,
    MIN_DONOR_BAND_LOWER,
    _assert_cgt_donor_stage_parameters,
    _assert_cgt_incidence_stage_parameters,
    _draw_banded_priors,
    clone_cgt_incidence,
    load_hmrc_cgt_size_bands,
    stack_cgt_band_donors,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind


def _distribution(*, negative: bool = False) -> dict[str, object]:
    knots = [-70.0, -60.0, -40.0, -20.0, -10.0, -5.0, -1.0]
    if not negative:
        knots = [-10.0, 0.0, 25.0, 50.0, 75.0, 100.0, 125.0]
    return {
        "rows": [
            {
                "minimum_total_income": 0,
                "percent_with_gains": 1.0,
                **dict(
                    zip(
                        ("p05", "p10", "p25", "p50", "p75", "p90", "p95"),
                        knots,
                        strict=True,
                    )
                ),
            }
        ]
    }


def _one_person_households(n: int, *, reverse_people: bool = False):
    ids = np.arange(1, n + 1, dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_benunit_id": ids,
            "person_household_id": ids,
            "age": np.full(n, 40),
            "capital_gains": np.zeros(n),
            "employment_income": np.linspace(10_000.0, 100_000.0, n),
        }
    )
    for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS:
        if column not in person:
            person[column] = 0.0
    if reverse_people:
        person = person.iloc[::-1].reset_index(drop=True)
    benunit = pd.DataFrame({"benunit_id": ids})
    household = pd.DataFrame(
        {
            "household_id": ids,
            "household_support_channel": np.where(ids % 2, "frs", "spi"),
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.linspace(1.0, 2.0, n),
        time_period="2024",
    )


def _adult_frame():
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "person_benunit_id": [1, 1, 1, 2, 2],
            "person_household_id": [1, 1, 1, 2, 2],
            "age": [45, 45, 12, 30, 50],
            "capital_gains": [1.0, 2.0, 3.0, 4.0, 5.0],
            "employment_income": [20_000.0] * 5,
        }
    )
    for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS:
        if column not in person:
            person[column] = 0.0
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame({"household_id": [1, 2]}),
        household_weights=[3.0, 7.0],
        time_period="2024",
    )


@lru_cache
def _stage(name: str):
    return load_country_spec("uk").sources.stage_map()[name]


def _drift(stage, operation_index: int, parameter: str):
    operations = list(stage.operations)
    operation = operations[operation_index]
    operations[operation_index] = SourceOperationSpec(
        kind=operation.kind,
        parameters={**operation.parameters, parameter: "__drift__"},
    )
    return replace(stage, operations=tuple(operations))


def test_clone_splits_exact_mass_and_uses_oldest_adult_carriers() -> None:
    result = clone_cgt_incidence(
        _adult_frame(), distribution=_distribution(negative=True)
    )
    household = result.frame.table("household")
    person = result.frame.table("person")

    assert result.original_mass == pytest.approx(5.0)
    assert result.clone_mass == pytest.approx(5.0)
    assert result.frame.weights_for("household").kind is WeightKind.IMPORTANCE
    assert result.frame.mass_log[-1].reason == CGT_CLONE_MASS_CHANGE_REASON
    clone_households = set(
        household.loc[household[HOUSEHOLD_IS_CGT_CLONE], "household_id"]
    )
    gainers = person.loc[person.capital_gains != 0]
    assert set(gainers.person_household_id) == clone_households
    # Household 1 has tied oldest adults: lower person_id carries the draw.
    assert (
        gainers.loc[
            gainers.person_household_id == min(clone_households)
        ].person_id.nunique()
        == 1
    )
    assert result.carrier_count == 2
    assert result.negative_prior_count == 1
    assert (gainers.capital_gains < 0.0).any()


def test_prior_spline_keeps_negative_values_and_extrapolates_linearly() -> None:
    draws = _draw_banded_priors(
        np.zeros(4),
        np.asarray([0.0, 0.05, 0.95, 1.0]),
        distribution=_distribution(),
    )

    assert draws[0] < 0.0
    assert draws[1] == pytest.approx(-10.0)
    assert draws[2] == pytest.approx(125.0)
    assert draws[3] > draws[2]


def test_band_donors_are_band_exact_positive_and_permutation_stable() -> None:
    first = stack_cgt_band_donors(
        _one_person_households(300),
        size_bands=load_hmrc_cgt_size_bands(),
        distribution=_distribution(),
    )
    second = stack_cgt_band_donors(
        _one_person_households(300, reverse_people=True),
        size_bands=load_hmrc_cgt_size_bands(),
        distribution=_distribution(),
    )
    donor_households = first.frame.table("household").loc[
        lambda table: table[HOUSEHOLD_IS_CGT_BAND_DONOR]
    ]

    assert len(donor_households) == DONOR_TOTAL == DONORS_PER_BAND * DONOR_BAND_COUNT
    assert (first.frame.weights_for("household").values[-DONOR_TOTAL:] > 0).all()
    assert [row["donor_count"] for row in first.band_rows] == [DONORS_PER_BAND] * 9
    assert [row["lower_limit"] for row in first.band_rows][0] == MIN_DONOR_BAND_LOWER
    assert [row["realized_min_gain"] for row in first.band_rows] == [
        row["mean_gain"] for row in first.band_rows
    ]
    assert [row["realized_max_gain"] for row in first.band_rows] == [
        row["mean_gain"] for row in first.band_rows
    ]
    assert [row["weighted_taxpayers"] for row in first.band_rows] == pytest.approx(
        [79_000, 74_000, 53_000, 37_000, 14_000, 8_000, 5_000, 3_000, 2_000]
    )
    second_donors = second.frame.table("household").loc[
        lambda table: table[HOUSEHOLD_IS_CGT_BAND_DONOR]
    ]
    assert set(donor_households.household_id) == set(second_donors.household_id)


def test_never_zero_band_weight_assertion_fires() -> None:
    resource = copy.deepcopy(load_hmrc_cgt_size_bands())
    retained = next(row for row in resource["rows"] if row["lower_limit"] == 12_300)
    retained["taxpayers_thousands"] = 0

    with pytest.raises(ValueError, match="zero initial weight"):
        _assert_cgt_donor_stage_parameters(
            _stage("cgt_band_donors"), size_bands=resource
        )


@pytest.mark.parametrize(
    "operation_index,parameter",
    [
        *[
            (0, name)
            for name in (
                "entity",
                "copies",
                "flag_column",
                "original_flag",
                "clone_flag",
                "mass_split",
                "weight_kind_out",
                "conservation",
                "id_remapping",
                "declared_factor",
                "reason",
            )
        ],
        *[
            (1, name)
            for name in (
                "resource",
                "income_proxy_components",
                "allowance_subtraction",
                "carrier",
                "adult_minimum_age",
                "quantile_points",
                "spline_degree",
                "extrapolation",
                "keep_negative_draws",
                "seed",
                "salt",
            )
        ],
    ],
)
def test_incidence_drift_assert_covers_every_reviewed_parameter(
    operation_index: int, parameter: str
) -> None:
    with pytest.raises(ValueError, match="drifted"):
        _assert_cgt_incidence_stage_parameters(
            _drift(_stage("cgt_incidence_clone"), operation_index, parameter)
        )


@pytest.mark.parametrize(
    "parameter",
    (
        "size_band_resource",
        "incidence_resource",
        "minimum_band_lower",
        "donors_per_band",
        "expected_band_count",
        "expected_donor_count",
        "candidate_order",
        "draw",
        "seed",
        "flag_column",
        "carrier",
        "initial_weight",
        "never_zero_weight",
        "weight_kind_out",
        "reason",
    ),
)
def test_donor_drift_assert_covers_every_reviewed_parameter(parameter: str) -> None:
    with pytest.raises(ValueError, match="drifted"):
        _assert_cgt_donor_stage_parameters(
            _drift(_stage("cgt_band_donors"), 0, parameter),
            size_bands=load_hmrc_cgt_size_bands(),
        )


def test_donor_drift_assert_rejects_propensity_and_extra_keys() -> None:
    """Closed-world equality: undeclared and extra parameters both fail."""
    for parameter in ("propensity", "undeclared_extra_key"):
        with pytest.raises(ValueError, match="drifted"):
            _assert_cgt_donor_stage_parameters(
                _drift(_stage("cgt_band_donors"), 0, parameter),
                size_bands=load_hmrc_cgt_size_bands(),
            )


def test_drift_asserts_reject_extra_operations() -> None:
    for name, check in (
        ("cgt_incidence_clone", _assert_cgt_incidence_stage_parameters),
        (
            "cgt_band_donors",
            lambda stage: _assert_cgt_donor_stage_parameters(
                stage, size_bands=load_hmrc_cgt_size_bands()
            ),
        ),
    ):
        stage = _stage(name)
        extra = replace(
            stage,
            operations=(*stage.operations, stage.operations[-1]),
        )
        with pytest.raises(ValueError, match="operation order drifted"):
            check(extra)
