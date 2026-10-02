"""The CGT residential split: the residential flag carried as weight (microcosm#1063)."""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime import cgt_residential_split
from microcosm.build.uk_runtime.cgt_asset_type import (
    CGT_RESIDENTIAL_GAINS_COLUMN,
    CGT_STOCK_HOUSEHOLD_COLUMNS,
    CGT_STOCK_PERSON_COLUMNS,
    load_hmrc_cgt_asset_type_facts,
)
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
    UKCGTPolicyParameters,
)
from microcosm.build.uk_runtime.cgt_residential_split import (
    CGT_RESIDENTIAL_CLONE_INDEX_COLUMN,
    CGT_RESIDENTIAL_MAXIMUM_LIABLE_GAINERS,
    CGT_RESIDENTIAL_PROBABILITY_COLUMN,
    CGT_RESIDENTIAL_SPLIT_MASS_CHANGE_REASON,
    CGT_RESIDENTIAL_SPLIT_STAGE_NAME,
    HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
    UKCGTResidentialSplitStageTransform,
    _assert_cgt_residential_split_stage_parameters,
    cgt_residential_split_operation_parameters,
    residential_arm_factors,
    split_cgt_residential_households,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind

PARAMETERS = UKCGTPolicyParameters(
    personal_allowance=12_570.0,
    personal_allowance_taper_threshold=100_000.0,
    personal_allowance_taper_rate=0.5,
    annual_exempt_amount=3_000.0,
    instant="2024-06-01",
    source="test",
)


def _frame(gains, *, weights=None, people_per_household=1, stocks=None):
    """Households of ``people_per_household`` persons, one gain per person."""

    gains = np.asarray(gains, dtype=float)
    rows = gains.size
    households = rows // people_per_household
    assert households * people_per_household == rows
    household_of = np.repeat(np.arange(households, dtype="int64"), people_per_household)
    person = pd.DataFrame(
        {
            "person_id": np.arange(rows, dtype="int64"),
            "person_household_id": household_of,
            "person_benunit_id": household_of,
            "capital_gains": gains,
            "age": np.full(rows, 45, dtype="int64"),
        }
    )
    for column in (*UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS, *CGT_STOCK_PERSON_COLUMNS):
        person[column] = 0.0
    household = pd.DataFrame(
        {
            "household_id": np.arange(households, dtype="int64"),
            "household_weight": (
                np.full(households, 60.0)
                if weights is None
                else np.asarray(weights, dtype=float)
            ),
            "region": np.full(households, "LONDON", dtype=object),
        }
    )
    for column in CGT_STOCK_HOUSEHOLD_COLUMNS:
        household[column] = 0.0
    for column, values in (stocks or {}).items():
        table = person if column in CGT_STOCK_PERSON_COLUMNS else household
        table[column] = np.asarray(values, dtype=float)
    benunit = pd.DataFrame({"benunit_id": np.arange(households, dtype="int64")})
    return uk_national_frame(
        person=person, benunit=benunit, household=household, time_period="2024"
    )


def _facts(frame, *, share: float = 0.3):
    """Vendored facts with the Table 8 totals sized to the frame's liable mass."""

    person = frame.table("person")
    household = frame.table("household")
    weights = pd.Series(
        frame.weights_for("household").values, index=household["household_id"]
    )
    person_weight = person["person_household_id"].map(weights).to_numpy(dtype=float)
    gains = person["capital_gains"].to_numpy(dtype=float)
    liable = gains > PARAMETERS.annual_exempt_amount
    count = share * float(person_weight[liable].sum())
    total = share * float((person_weight[liable] * gains[liable]).sum())
    return dataclasses.replace(
        load_hmrc_cgt_asset_type_facts(),
        table8a_taxpayers_total=count,
        table8a_gains_total=total,
        table8b_individuals_taxpayers=count,
        table8b_individuals_gains=total,
        table8b_all_taxpayers=count,
        table8b_all_gains=total,
    )


def _gains(rows: int = 3_000, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gains = np.where(rng.random(rows) < 0.7, np.exp(rng.normal(10.5, 1.8, rows)), 0.0)
    gains[:20] = -500.0
    return gains


def test_arm_factors_are_the_subset_products_and_sum_to_one() -> None:
    assert residential_arm_factors(np.asarray([0.2])).tolist() == pytest.approx(
        [0.8, 0.2]
    )
    # Bit m of the arm index marks gainer m: arm 1 = first residential only,
    # arm 2 = second only, arm 3 = both.
    assert residential_arm_factors(np.asarray([0.2, 0.5])).tolist() == pytest.approx(
        [0.4, 0.1, 0.4, 0.1]
    )
    three = residential_arm_factors(np.asarray([0.1, 0.2, 0.3]))
    assert three.size == 8 and three.sum() == pytest.approx(1.0)
    assert three[7] == pytest.approx(0.1 * 0.2 * 0.3)
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        residential_arm_factors(np.asarray([1.2]))
    with pytest.raises(ValueError, match="at least one liable gainer"):
        residential_arm_factors(np.asarray([]))


def test_split_carries_table_8a_as_identities_and_conserves_mass_bitwise() -> None:
    gains = _gains()
    frame = _frame(gains)
    facts = _facts(frame)

    result = split_cgt_residential_households(frame, facts=facts, parameters=PARAMETERS)

    evidence = result.evidence()
    identities = evidence["identities"]
    # The solve puts the expectations on the targets; the arms realise the
    # expectations exactly, in every gain band.
    assert identities["count_solve_relative_error"] < 1e-6
    assert identities["gains_solve_relative_error"] < 1e-6
    assert identities["count_identity_relative_error"] < 1e-12
    assert identities["gains_identity_relative_error"] < 1e-12
    for band in evidence["bands"]:
        assert band["achieved_count"] == pytest.approx(
            band["expected_count"], rel=1e-12
        )
        assert band["achieved_gains"] == pytest.approx(
            band["expected_gains"], rel=1e-12
        )
    # Mass: bitwise total, one record at declared factor one, every arm the
    # product of its household's weight and the probabilities.
    assert float(result.frame.weights_for("household").total) == float(
        frame.weights_for("household").total
    )
    record = result.frame.mass_log[-1]
    assert record.reason == CGT_RESIDENTIAL_SPLIT_MASS_CHANGE_REASON
    assert record.declared_factor == 1.0 and record.old_total == record.new_total
    assert evidence["arm_weights_exact"] is True
    household = result.frame.table("household")
    person = result.frame.table("person")
    weights = pd.Series(
        result.frame.weights_for("household").values, index=household["household_id"]
    )
    arms = household[HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE].to_numpy(dtype=bool)
    liable = gains > PARAMETERS.annual_exempt_amount
    assert int(arms.sum()) == int(liable.sum()) == evidence["arms_created"]
    assert evidence["households_by_liable_gainers"] == {"1": int(liable.sum())}
    multiplier = cgt_residential_split.id_multiplier_for_values(
        frame.table("person")["person_id"],
        frame.table("person")["person_household_id"],
        frame.table("person")["person_benunit_id"],
        frame.table("benunit")["benunit_id"],
        frame.table("household")["household_id"],
    )
    probability = person.set_index("person_id")[CGT_RESIDENTIAL_PROBABILITY_COLUMN]
    for source in np.flatnonzero(liable):
        p = float(probability.loc[source])
        assert 0.0 < p < 1.0
        assert weights.loc[source] == 60.0 * (1.0 - p)
        assert weights.loc[source + multiplier] == 60.0 * p
        assert (
            household.set_index("household_id").loc[
                source + multiplier, CGT_RESIDENTIAL_CLONE_INDEX_COLUMN
            ]
            == 1
        )
    # The residential arm's gainer carries the whole gain; nobody else does.
    residential = person[CGT_RESIDENTIAL_GAINS_COLUMN].to_numpy()
    on_arm = person["person_id"].to_numpy() >= multiplier
    assert (residential[on_arm] == person["capital_gains"].to_numpy()[on_arm]).all()
    assert (residential[~on_arm] == 0.0).all()
    assert (probability.loc[np.flatnonzero(~liable)] == 0.0).all()


def test_households_with_several_liable_gainers_become_product_arms() -> None:
    # Three households of two persons: both liable, one liable, none liable.
    frame = _frame(
        [50_000.0, 80_000.0, 90_000.0, 0.0, 1_000.0, 0.0],
        weights=[10.0, 20.0, 30.0],
        people_per_household=2,
    )
    result = split_cgt_residential_households(
        frame, facts=_facts(frame), parameters=PARAMETERS
    )
    evidence = result.evidence()
    assert evidence["households_by_liable_gainers"] == {"1": 1, "2": 1}
    assert evidence["arms_created"] == 3 + 1
    household = result.frame.table("household").set_index("household_id")
    person = result.frame.table("person")
    weights = pd.Series(
        result.frame.weights_for("household").values, index=household.index
    )
    probability = person.set_index("person_id")[CGT_RESIDENTIAL_PROBABILITY_COLUMN]
    p0, p1 = float(probability.loc[0]), float(probability.loc[1])
    multiplier = 10
    assert weights.loc[0] == pytest.approx(10.0 * (1 - p0) * (1 - p1))
    assert weights.loc[0 + multiplier] == pytest.approx(10.0 * p0 * (1 - p1))
    assert weights.loc[0 + 2 * multiplier] == pytest.approx(10.0 * (1 - p0) * p1)
    assert weights.loc[0 + 3 * multiplier] == pytest.approx(10.0 * p0 * p1)
    assert household.loc[[10, 20, 30], CGT_RESIDENTIAL_CLONE_INDEX_COLUMN].tolist() == [
        1,
        2,
        3,
    ]
    by_person = person.set_index("person_id")[CGT_RESIDENTIAL_GAINS_COLUMN]
    # Arm 1: gainer 0 residential; arm 2: gainer 1; arm 3: both.
    assert by_person.loc[[0 + multiplier, 1 + multiplier]].tolist() == [50_000.0, 0.0]
    assert by_person.loc[[0 + 2 * multiplier, 1 + 2 * multiplier]].tolist() == [
        0.0,
        80_000.0,
    ]
    assert by_person.loc[[0 + 3 * multiplier, 1 + 3 * multiplier]].tolist() == [
        50_000.0,
        80_000.0,
    ]
    # The unsplit household is untouched and carries the exact-total correction.
    assert household.loc[2, HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE] is np.False_
    assert float(result.frame.weights_for("household").total) == 60.0
    assert weights.loc[2] == pytest.approx(30.0)


def test_split_refuses_too_many_gainers_an_already_split_or_classified_frame() -> None:
    gains = [10_000.0] * (CGT_RESIDENTIAL_MAXIMUM_LIABLE_GAINERS + 1)
    frame = _frame(gains, people_per_household=len(gains))
    with pytest.raises(ValueError, match="more than"):
        split_cgt_residential_households(
            frame, facts=_facts(frame), parameters=PARAMETERS
        )
    frame = _frame(_gains(500))
    facts = _facts(frame)
    once = split_cgt_residential_households(frame, facts=facts, parameters=PARAMETERS)
    with pytest.raises(ValueError, match="already ran"):
        split_cgt_residential_households(once.frame, facts=facts, parameters=PARAMETERS)
    person = frame.table("person").assign(**{CGT_RESIDENTIAL_GAINS_COLUMN: 0.0})
    classified = uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        time_period="2024",
        household_weights=frame.weights_for("household").values,
    )
    with pytest.raises(ValueError, match="already ran"):
        split_cgt_residential_households(classified, facts=facts, parameters=PARAMETERS)


def test_split_is_stable_under_person_order_and_follows_the_stock_signal() -> None:
    gains = _gains(1_500)
    frame = _frame(gains)
    facts = _facts(frame)
    first = split_cgt_residential_households(frame, facts=facts, parameters=PARAMETERS)
    reversed_person = frame.table("person").iloc[::-1].reset_index(drop=True)
    second = split_cgt_residential_households(
        uk_national_frame(
            person=reversed_person,
            benunit=frame.table("benunit"),
            household=frame.table("household"),
            time_period="2024",
            household_weights=frame.weights_for("household").values,
        ),
        facts=facts,
        parameters=PARAMETERS,
    )

    def arms(result):
        household = result.frame.table("household")
        return pd.Series(
            result.frame.weights_for("household").values,
            index=household["household_id"].to_numpy(),
        ).sort_index()

    pd.testing.assert_series_equal(arms(first), arms(second))
    # A gainer who shows the stock the flag implies is more likely residential.
    stocks = {"property_income": np.where(np.arange(gains.size) % 2 == 0, 1.0, 0.0)}
    shifted = split_cgt_residential_households(
        _frame(gains, stocks=stocks), facts=facts, parameters=PARAMETERS
    )
    evidence = shifted.evidence()
    assert (
        evidence["stock"]["expected_share_residential"]
        > evidence["stock"]["share_liable"]
    )
    assert evidence["identities"]["count_solve_relative_error"] < 1e-6


def test_evidence_is_json_plain_and_the_stage_transform_runs_from_the_seam() -> None:
    import json

    spec = load_country_spec("uk")
    assert spec.sources is not None
    stage = spec.sources.stage_map()[CGT_RESIDENTIAL_SPLIT_STAGE_NAME]
    assert stage.grain == "household"
    assert tuple(stage.outputs) == UKCGTResidentialSplitStageTransform.output_columns()
    assert stage.rewrites == ()
    frame = _frame(_gains(800))
    transform = UKCGTResidentialSplitStageTransform(
        stage=stage, facts=_facts(frame), parameters=PARAMETERS
    )
    result = transform(frame)
    assert result.weights_for("household").kind is WeightKind.IMPORTANCE
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == CGT_RESIDENTIAL_SPLIT_STAGE_NAME
    json.dumps(evidence)
    assert evidence["maximum_liable_gainers_per_household"] == (
        CGT_RESIDENTIAL_MAXIMUM_LIABLE_GAINERS
    )


def test_operation_dictionary_restates_the_reviewed_design() -> None:
    operations = dict(cgt_residential_split_operation_parameters())
    assert list(operations) == [
        "verify_vendored_fact_resource",
        "split_liable_gainers_by_residential_probability",
    ]
    split = operations["split_liable_gainers_by_residential_probability"]
    assert split["declared_factor"] == 1.0 and split["conservation"] == "exact_total"
    assert split["reason"] == CGT_RESIDENTIAL_SPLIT_MASS_CHANGE_REASON
    assert "no draw" in split["realization"]
    assert split["flag_column"] == HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE
    assert split["index_column"] == CGT_RESIDENTIAL_CLONE_INDEX_COLUMN
    assert split["probability_column"] == CGT_RESIDENTIAL_PROBABILITY_COLUMN
    assert split["output_column"] == CGT_RESIDENTIAL_GAINS_COLUMN


@pytest.mark.parametrize("parameter", ["realization", "declared_factor", "flag_column"])
def test_drift_assert_covers_every_reviewed_parameter(parameter: str) -> None:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    stage = spec.sources.stage_map()[CGT_RESIDENTIAL_SPLIT_STAGE_NAME]
    _assert_cgt_residential_split_stage_parameters(stage)
    operations = list(stage.operations)
    operation = operations[1]
    operations[1] = SourceOperationSpec(
        operation.kind, {**operation.parameters, parameter: "__drift__"}
    )
    with pytest.raises(ValueError):
        _assert_cgt_residential_split_stage_parameters(
            replace(stage, operations=tuple(operations))
        )
    with pytest.raises(ValueError, match="Expected stage"):
        _assert_cgt_residential_split_stage_parameters(
            replace(stage, stage="cgt_support_split")
        )
    with pytest.raises(ValueError, match="exactly its flag"):
        _assert_cgt_residential_split_stage_parameters(
            replace(stage, outputs=stage.outputs[:-1])
        )
