from __future__ import annotations

import copy
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime.advani_summers import (
    advani_summers_knots,
    advani_summers_rows,
    exempt_range_quantiles,
    load_advani_summers_distribution,
)
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
    UKCGTPolicyParameters,
)
from microcosm.build.uk_runtime.cgt_structure import (
    CGT_ANCHOR_MASS_CHANGE_REASON,
    CGT_CLONE_MASS_CHANGE_REASON,
    CGT_INCIDENCE_ANCHOR_STAGE_NAME,
    DONOR_BAND_COUNT,
    DONOR_TOTAL,
    DONORS_PER_BAND,
    HOUSEHOLD_IS_CGT_BAND_DONOR,
    HOUSEHOLD_IS_CGT_CLONE,
    MIN_DONOR_BAND_LOWER,
    UKCGTIncidenceAnchorStageTransform,
    _assert_cgt_donor_stage_parameters,
    _assert_cgt_incidence_anchor_stage_parameters,
    _assert_cgt_incidence_stage_parameters,
    _draw_banded_priors,
    _solve_group_factors,
    anchor_cgt_incidence,
    cgt_incidence_anchor_operation_parameters,
    clone_cgt_incidence,
    load_hmrc_cgt_size_bands,
    pair_clone_households,
    reporter_composition_quantiles,
    stack_cgt_band_donors,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import MassChangeRecord, WeightKind


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
    # HMRC Table 2.1a 2024-25 individuals from GBP 12,300 (the vendored
    # conditioning facts), not the retired 2023-24 hand copy.
    assert [row["weighted_taxpayers"] for row in first.band_rows] == pytest.approx(
        [97_000, 98_000, 78_000, 61_000, 25_000, 16_000, 9_000, 5_000, 3_000]
    )
    second_donors = second.frame.table("household").loc[
        lambda table: table[HOUSEHOLD_IS_CGT_BAND_DONOR]
    ]
    assert set(donor_households.household_id) == set(second_donors.household_id)


def test_never_zero_band_weight_assertion_fires() -> None:
    resource = copy.deepcopy(load_hmrc_cgt_size_bands())
    retained = next(row for row in resource["rows"] if row["lower_limit"] == 12_300)
    retained["taxpayers"] = 0

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


# --- #970 incidence anchor ---------------------------------------------------

ANCHOR_KNOTS = [-10.0, 0.0, 25.0, 50.0, 75.0, 100.0, 125.0]
#: Per clone household: liable, sub-exempt, loss-making, zero-gain carriers.
ANCHOR_PATTERN = [60.0, 20.0, -5.0, 0.0] * 3


def _anchor_distribution() -> dict[str, object]:
    """Two bands: zero at q=0.1 in both, the AEA of 50 at q=0.5 and q=0.25."""

    columns = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")
    return {
        "rows": [
            {
                "minimum_total_income": 0,
                "percent_with_gains": 0.2,
                **dict(zip(columns, ANCHOR_KNOTS, strict=True)),
            },
            {
                "minimum_total_income": 50_000,
                "percent_with_gains": 0.5,
                **dict(
                    zip(columns, [2.0 * knot for knot in ANCHOR_KNOTS], strict=True)
                ),
            },
        ]
    }


def _parameters(annual_exempt_amount: float = 50.0) -> UKCGTPolicyParameters:
    return UKCGTPolicyParameters(
        personal_allowance=12_570.0,
        personal_allowance_taper_threshold=100_000.0,
        personal_allowance_taper_rate=0.5,
        annual_exempt_amount=annual_exempt_amount,
        instant="2024-06-01",
        source="test",
    )


def _anchor_input(
    pattern: list[float],
    *,
    weights: np.ndarray | None = None,
    donors: bool = True,
    permutation: np.ndarray | None = None,
):
    """Two-person households cloned at equal mass with carrier gains set.

    Each source household holds a 40-year-old carrier and a 30-year-old; the
    clone stage draws the prior on the carrier and the pattern overwrites it.
    Two band donors copied from the first clone are appended with the donor
    flag set and the inherited clone flag left true, as the donor stage
    leaves them.
    """

    n = len(pattern)
    household_ids = np.arange(1, n + 1, dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 2 * n + 1, dtype="int64"),
            "person_benunit_id": np.repeat(household_ids, 2),
            "person_household_id": np.repeat(household_ids, 2),
            "age": np.tile([40, 30], n),
            "capital_gains": 0.0,
            "employment_income": np.repeat(np.linspace(10_000.0, 120_000.0, n), 2),
        }
    )
    for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS:
        if column not in person:
            person[column] = 0.0
    frame = uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": household_ids}),
        household=pd.DataFrame({"household_id": household_ids}),
        household_weights=np.linspace(10.0, 20.0, n) if weights is None else weights,
        time_period="2024",
    )
    cloned = clone_cgt_incidence(frame, distribution=_anchor_distribution()).frame
    person = cloned.table("person").copy()
    benunit = cloned.table("benunit").copy()
    household = cloned.table("household").copy()
    household[HOUSEHOLD_IS_CGT_BAND_DONOR] = False
    clone_ids = household.loc[
        household[HOUSEHOLD_IS_CGT_CLONE], "household_id"
    ].tolist()
    carriers = person["age"].eq(40).to_numpy()
    for household_id, gain in zip(clone_ids, pattern, strict=True):
        person.loc[
            carriers & person["person_household_id"].eq(household_id).to_numpy(),
            "capital_gains",
        ] = gain
    household_weights = np.asarray(cloned.weights_for("household").values, dtype=float)
    mass_log = cloned.mass_log
    if donors:
        source = clone_ids[0]
        for offset in (1_000, 2_000):
            donor_person = person.loc[person.person_household_id == source].copy()
            for column in ("person_id", "person_benunit_id", "person_household_id"):
                donor_person[column] = donor_person[column] + offset
            donor_person["capital_gains"] = np.where(
                donor_person["age"] == 40, 500.0, 0.0
            )
            donor_benunit = benunit.loc[benunit.benunit_id == source].copy()
            donor_benunit["benunit_id"] = donor_benunit["benunit_id"] + offset
            donor_household = household.loc[household.household_id == source].copy()
            donor_household["household_id"] = donor_household["household_id"] + offset
            donor_household[HOUSEHOLD_IS_CGT_BAND_DONOR] = True
            person = pd.concat([person, donor_person], ignore_index=True)
            benunit = pd.concat([benunit, donor_benunit], ignore_index=True)
            household = pd.concat([household, donor_household], ignore_index=True)
            household_weights = np.r_[household_weights, 7.0]
        mass_log = (
            *mass_log,
            MassChangeRecord(
                entity="household",
                old_total=cloned.weights_for("household").total,
                new_total=float(household_weights.sum()),
                declared_factor=None,
                reason="test donors",
            ),
        )
    if permutation is not None:
        # Group ids must stay sorted; person order is free.
        person = person.iloc[permutation].reset_index(drop=True)
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=household_weights,
        mass_log=mass_log,
    )


def _weights_by_id(frame) -> pd.Series:
    return pd.Series(
        frame.weights_for("household").values,
        index=frame.table("household")["household_id"].to_numpy(),
    )


def test_pairing_is_a_bijection_and_ignores_donors() -> None:
    frame = _anchor_input(ANCHOR_PATTERN)
    person, benunit, household = (
        frame.table("person"),
        frame.table("benunit"),
        frame.table("household"),
    )
    clone_positions, original_positions, multiplier = pair_clone_households(
        person, benunit, household
    )

    assert multiplier == 100
    assert len(clone_positions) == len(ANCHOR_PATTERN)
    ids = household["household_id"].to_numpy()
    assert (ids[clone_positions] - multiplier == ids[original_positions]).all()
    donors = household[HOUSEHOLD_IS_CGT_BAND_DONOR].to_numpy()
    assert donors.sum() == 2
    assert household[HOUSEHOLD_IS_CGT_CLONE].to_numpy()[donors].all()
    assert not donors[clone_positions].any() and not donors[original_positions].any()

    orphaned = household.loc[household.household_id != 3]
    with pytest.raises(ValueError, match="without a paired original"):
        pair_clone_households(person, benunit, orphaned)
    extra = pd.concat(
        [household, household.iloc[[0]].assign(household_id=13)], ignore_index=True
    )
    with pytest.raises(ValueError, match="exactly one clone per original"):
        pair_clone_households(person, benunit, extra)
    with pytest.raises(ValueError, match=HOUSEHOLD_IS_CGT_BAND_DONOR):
        pair_clone_households(
            person, benunit, household.drop(columns=[HOUSEHOLD_IS_CGT_BAND_DONOR])
        )


def test_reporter_composition_crossings_match_the_published_first_band() -> None:
    distribution = load_advani_summers_distribution()
    rows = advani_summers_rows(distribution)
    lower, upper = reporter_composition_quantiles(
        np.asarray([0.0, 1e9]), distribution=distribution, annual_exempt_amount=3_000.0
    )
    # The <40 row: p10 = -4,400 and p25 = 3,800, so zero sits 4,400/8,200 of
    # the way along the 0.15-wide segment and 3,000 sits 7,400/8,200 along it.
    assert lower[0] == pytest.approx(0.10 + 0.15 * 4_400 / 8_200, abs=1e-12)
    assert upper[0] == pytest.approx(0.10 + 0.15 * 7_400 / 8_200, abs=1e-12)
    assert (lower[1], upper[1]) == pytest.approx(
        exempt_range_quantiles(advani_summers_knots(rows[-1]), 3_000.0)
    )
    synthetic_lower, synthetic_upper = reporter_composition_quantiles(
        np.asarray([10_000.0, 60_000.0]),
        distribution=_anchor_distribution(),
        annual_exempt_amount=50.0,
    )
    assert synthetic_lower == pytest.approx([0.1, 0.1])
    assert synthetic_upper == pytest.approx([0.5, 0.25])


def test_solve_group_factors_meets_the_target_and_caps_at_one() -> None:
    scale, factors = _solve_group_factors(np.ones(3), np.asarray([0.01, 0.5, 1.0]), 2.0)
    # One capped household leaves 1.0 for the other two: s = 1 / 0.51.
    assert scale == pytest.approx(1.0 / 0.51, rel=1e-12)
    assert factors == pytest.approx([0.01 / 0.51, 0.5 / 0.51, 1.0], rel=1e-12)
    assert factors.sum() == pytest.approx(2.0, rel=1e-12)
    uncapped_scale, uncapped = _solve_group_factors(
        np.asarray([2.0, 3.0]), np.asarray([0.2, 0.4]), 1.0
    )
    assert uncapped_scale == pytest.approx(1.0 / 1.6, rel=1e-12)
    assert (uncapped < 1.0).all()
    assert (np.asarray([2.0, 3.0]) * uncapped).sum() == pytest.approx(1.0, rel=1e-12)


def test_anchor_moves_non_liable_clone_mass_and_conserves_every_pair() -> None:
    frame = _anchor_input(ANCHOR_PATTERN)
    result = anchor_cgt_incidence(
        frame, distribution=_anchor_distribution(), parameters=_parameters()
    )
    person = frame.table("person")
    household = frame.table("household")
    before = np.asarray(frame.weights_for("household").values)
    after = np.asarray(result.frame.weights_for("household").values)
    clone_positions, original_positions, _ = pair_clone_households(
        person, frame.table("benunit"), household
    )
    evidence = result.evidence()

    position = pd.Series(np.arange(len(household)), index=household["household_id"])
    person_weight = before[position.reindex(person["person_household_id"]).to_numpy()]
    liable = person["capital_gains"].to_numpy() > 50.0
    assert result.liable_mass == pytest.approx(person_weight[liable].sum())
    assert result.liable_persons == 5  # three liable clones and two donors
    pattern = np.asarray(ANCHOR_PATTERN)
    for group, mask in (
        ("sub_exempt", (pattern >= 0.0) & (pattern <= 50.0)),
        ("loss", pattern < 0.0),
    ):
        assert result.before[group] == pytest.approx(
            before[clone_positions][mask].sum()
        )
        assert result.after[group] == pytest.approx(result.targets[group], rel=1e-9)
        assert result.after[group] < result.before[group]
        assert result.after[group] == pytest.approx(after[clone_positions][mask].sum())
    assert result.after["liable"] == result.before["liable"]
    liable_clones = clone_positions[pattern > 50.0]
    np.testing.assert_array_equal(after[liable_clones], before[liable_clones])
    donors = household[HOUSEHOLD_IS_CGT_BAND_DONOR].to_numpy()
    np.testing.assert_array_equal(after[donors], before[donors])
    np.testing.assert_allclose(
        after[clone_positions] + after[original_positions],
        before[clone_positions] + before[original_positions],
        rtol=1e-12,
    )
    assert (after[original_positions] >= before[original_positions]).all()
    assert (after > 0.0).all()
    assert result.frame.weights_for("household").total == pytest.approx(
        frame.weights_for("household").total, rel=1e-12
    )
    assert result.frame.weights_for("household").kind is WeightKind.IMPORTANCE
    for entity in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(
            result.frame.table(entity).reset_index(drop=True),
            frame.table(entity).reset_index(drop=True),
        )
    record = result.frame.mass_log[-1]
    assert record.reason == CGT_ANCHOR_MASS_CHANGE_REASON
    assert record.declared_factor == 1.0
    assert record.new_total == pytest.approx(record.old_total, rel=1e-12)
    assert record.new_total == result.frame.weights_for("household").total
    assert result.frame.mass_log[:-1] == frame.mass_log

    assert evidence["stage"] == CGT_INCIDENCE_ANCHOR_STAGE_NAME
    assert set(evidence) == {
        "stage",
        "annual_exempt_amount",
        "liable_mass",
        "liable_persons",
        "composition",
        "targets",
        "before",
        "after",
        "scale",
        "transferred_mass",
        "pair_count",
        "trimmed_households",
        "capped_households",
        "zero_gain_clone_households",
        "max_pair_relative_error",
        "mass_by_clone_flag",
        "donor_mass",
        "by_income_band",
    }
    composition = evidence["composition"]
    assert 0.0 < composition["zero_quantile"] < composition["exempt_quantile"] < 1.0
    assert composition["implied_reporter_mass"] == pytest.approx(
        result.liable_mass / (1.0 - composition["exempt_quantile"])
    )
    assert evidence["pair_count"] == len(ANCHOR_PATTERN)
    assert evidence["zero_gain_clone_households"] == 3
    assert evidence["trimmed_households"] == 9
    assert evidence["capped_households"] == 0
    assert evidence["transferred_mass"] == pytest.approx(
        (result.before["sub_exempt"] - result.after["sub_exempt"])
        + (result.before["loss"] - result.after["loss"])
    )
    assert evidence["max_pair_relative_error"] <= 1e-12
    assert (
        evidence["mass_by_clone_flag"]["true"] < evidence["mass_by_clone_flag"]["false"]
    )
    assert evidence["mass_by_clone_flag"]["true"] == pytest.approx(
        after[clone_positions].sum()
    )
    assert evidence["donor_mass"] == 14.0
    assert sum(row["sub_exempt_after"] for row in evidence["by_income_band"]) == (
        pytest.approx(result.after["sub_exempt"])
    )
    assert sum(row["clone_households"] for row in evidence["by_income_band"]) == (
        len(ANCHOR_PATTERN)
    )


def test_anchor_factor_rises_with_band_incidence() -> None:
    frame = _anchor_input(ANCHOR_PATTERN)
    result = anchor_cgt_incidence(
        frame, distribution=_anchor_distribution(), parameters=_parameters()
    )
    before = np.asarray(frame.weights_for("household").values)
    after = np.asarray(result.frame.weights_for("household").values)
    clone_positions, _, _ = pair_clone_households(
        frame.table("person"), frame.table("benunit"), frame.table("household")
    )
    factors = after[clone_positions] / before[clone_positions]
    income = np.linspace(10_000.0, 120_000.0, len(ANCHOR_PATTERN))
    pattern = np.asarray(ANCHOR_PATTERN)
    for mask in ((pattern >= 0.0) & (pattern <= 50.0), pattern < 0.0):
        low_band = factors[mask & (income < 50_000.0)]
        high_band = factors[mask & (income >= 50_000.0)]
        assert low_band.size and high_band.size
        # Incidence 0.5 against 0.2: the same scale, so factors sit 2.5x apart.
        assert np.unique(np.round(low_band, 12)).size == 1
        assert np.unique(np.round(high_band, 12)).size == 1
        assert high_band[0] / low_band[0] == pytest.approx(2.5, rel=1e-9)
        assert (factors[mask] < 1.0).all()
    assert (factors[pattern > 50.0] == 1.0).all()


def test_anchor_leaves_a_group_already_below_its_target_untouched() -> None:
    pattern = [60.0] * 10 + [20.0, -5.0]
    weights = np.full(len(pattern), 10.0)
    # A negligible sub-exempt clone sits under its target; a heavy loss-maker
    # sits far above its own.
    weights[10] = 1e-3
    weights[11] = 200.0
    frame = _anchor_input(pattern, weights=weights)
    result = anchor_cgt_incidence(
        frame, distribution=_anchor_distribution(), parameters=_parameters()
    )

    assert result.before["sub_exempt"] < result.targets["sub_exempt"]
    assert result.after["sub_exempt"] == result.before["sub_exempt"]
    assert result.scale["sub_exempt"] is None
    assert result.before["loss"] > result.targets["loss"]
    assert result.after["loss"] == pytest.approx(result.targets["loss"], rel=1e-9)
    assert result.scale["loss"] is not None
    assert result.trimmed_households == 1


def test_anchor_is_deterministic_under_person_permutation() -> None:
    permutation = np.random.default_rng(5).permutation(len(ANCHOR_PATTERN) * 4 + 4)
    straight = anchor_cgt_incidence(
        _anchor_input(ANCHOR_PATTERN),
        distribution=_anchor_distribution(),
        parameters=_parameters(),
    )
    permuted = anchor_cgt_incidence(
        _anchor_input(ANCHOR_PATTERN, permutation=permutation),
        distribution=_anchor_distribution(),
        parameters=_parameters(),
    )
    pd.testing.assert_series_equal(
        _weights_by_id(straight.frame).sort_index(),
        _weights_by_id(permuted.frame).sort_index(),
        rtol=0.0,
        atol=1e-12,
    )
    assert straight.after == pytest.approx(permuted.after, rel=1e-12)
    assert straight.evidence()["targets"] == pytest.approx(
        permuted.evidence()["targets"], rel=1e-12
    )


def test_anchor_refuses_missing_liable_mass_and_stray_gains() -> None:
    no_liable = _anchor_input([20.0, -5.0, 0.0, 20.0], donors=False)
    with pytest.raises(ValueError, match="positive liable mass"):
        anchor_cgt_incidence(
            no_liable, distribution=_anchor_distribution(), parameters=_parameters()
        )

    frame = _anchor_input(ANCHOR_PATTERN)
    person = frame.table("person").copy()
    household = frame.table("household")
    clone_id = household.loc[
        household[HOUSEHOLD_IS_CGT_CLONE] & ~household[HOUSEHOLD_IS_CGT_BAND_DONOR],
        "household_id",
    ].iloc[0]
    stray = person["person_household_id"].eq(clone_id) & person["age"].eq(30)
    person.loc[stray, "capital_gains"] = 1.0
    tainted = uk_national_frame(
        person=person,
        benunit=frame.table("benunit").copy(),
        household=household.copy(),
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    with pytest.raises(ValueError, match="only person with capital gains"):
        anchor_cgt_incidence(
            tainted, distribution=_anchor_distribution(), parameters=_parameters()
        )
    with pytest.raises(ValueError, match="positive annual exempt amount"):
        anchor_cgt_incidence(
            frame, distribution=_anchor_distribution(), parameters=_parameters(0.0)
        )


def test_anchor_transform_locks_the_manifest_and_reports_evidence() -> None:
    stage = _stage("cgt_incidence_anchor")
    _assert_cgt_incidence_anchor_stage_parameters(stage)
    operations = cgt_incidence_anchor_operation_parameters()
    assert [operation.kind for operation in stage.operations] == [
        kind for kind, _ in operations
    ]
    for index, (_, parameters) in enumerate(operations):
        for parameter in parameters:
            with pytest.raises(ValueError, match="drifted"):
                _assert_cgt_incidence_anchor_stage_parameters(
                    _drift(stage, index, parameter)
                )
    with pytest.raises(ValueError, match="drifted"):
        _assert_cgt_incidence_anchor_stage_parameters(
            replace(stage, operations=(*stage.operations, stage.operations[0]))
        )
    with pytest.raises(ValueError, match="no outputs"):
        _assert_cgt_incidence_anchor_stage_parameters(
            replace(stage, outputs=("capital_gains",))
        )
    with pytest.raises(ValueError, match="grain"):
        _assert_cgt_incidence_anchor_stage_parameters(replace(stage, grain="person"))
    assert stage.outputs == () and stage.rewrites == ()

    transform = UKCGTIncidenceAnchorStageTransform(
        stage=stage, distribution=_anchor_distribution(), parameters=_parameters()
    )
    assert transform.output_columns() == ()
    with pytest.raises(RuntimeError, match="completed stage run"):
        transform.checkpoint_metadata()
    frame = _anchor_input(ANCHOR_PATTERN)
    anchored = transform(frame)
    assert anchored.weights_for("household").total == pytest.approx(
        frame.weights_for("household").total, rel=1e-12
    )
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == CGT_INCIDENCE_ANCHOR_STAGE_NAME
    assert evidence["annual_exempt_amount"] == 50.0
