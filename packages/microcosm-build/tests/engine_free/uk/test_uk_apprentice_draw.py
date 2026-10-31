"""is_apprentice on the UK spine: employees aged 21 to 24, realized to a per-head rate."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime import frs_person_draws
from microcosm.build.uk_runtime.frs_person_draws import (
    APPRENTICE_OUTPUT,
    APPRENTICE_RATE_KEY,
    UKFRSPersonDrawsStageTransform,
    apprentice_relief_schedule,
    derive_is_apprentice,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from test_support.microcosm_build.uk_frs_stochastic import _Contract

_STATUSES = ("FT_EMPLOYED", "PT_EMPLOYED", "FT_SELF_EMPLOYED", "STUDENT")


def _people(n_per_cell: int = 2000) -> pd.DataFrame:
    """Every age from 18 to 30 crossed with four statuses, one household each."""

    ages = np.arange(18, 31)
    rows = [
        (age, status) for age in ages for status in _STATUSES for _ in range(n_per_cell)
    ]
    frame = pd.DataFrame(rows, columns=["age", "employment_status"])
    frame.insert(0, "person_id", np.arange(1, len(frame) + 1))
    frame.insert(1, "person_household_id", frame["person_id"])
    return frame


def _small(ages, statuses, *, households=None) -> pd.DataFrame:
    ids = np.arange(11, 11 + len(ages))
    return pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids if households is None else households,
            "age": ages,
            "employment_status": statuses,
        }
    )


def test_only_employees_aged_21_to_24_are_flagged() -> None:
    person = _people()

    flags = derive_is_apprentice(person, rate=0.06).flags

    age = person["age"].to_numpy()
    employee = person["employment_status"].isin(["FT_EMPLOYED", "PT_EMPLOYED"])
    eligible = (age >= 21) & (age <= 24) & employee.to_numpy()
    assert flags.dtype == bool
    assert not flags[~eligible].any()
    assert flags[eligible].any()


def test_flagged_weight_is_nearest_the_per_head_target() -> None:
    person = _people()
    rng = np.random.default_rng(0)
    weights = rng.uniform(100.0, 3000.0, len(person))
    rate = 0.06

    draw = derive_is_apprentice(person, rate=rate, person_weights=weights)

    age = person["age"].to_numpy()
    in_ages = (age >= 21) & (age <= 24)
    target = rate * weights[in_ages].sum()
    realized = weights[draw.flags].sum()
    # The walk stops at the cumulative weight nearest the target, so it misses
    # by at most half the last person's weight.
    assert abs(realized - target) <= weights.max() / 2
    receipt = draw.receipt
    assert receipt["rate_key"] == APPRENTICE_RATE_KEY
    assert receipt["population_mass"] == pytest.approx(weights[in_ages].sum())
    assert receipt["target"] == pytest.approx(target)
    assert receipt["realized_mass"] == pytest.approx(realized)
    assert receipt["realization_deviation"] == pytest.approx(realized - target)
    assert receipt["cap_bound"] is False
    assert receipt["eligible_support"] == "30+"
    assert receipt["flagged_support"] == "30+"


def test_people_are_flagged_in_the_order_of_their_identity_uniform() -> None:
    person = _small(
        [22, 22, 23, 24, 40, 23],
        [
            "FT_EMPLOYED",
            "STUDENT",
            "PT_EMPLOYED",
            "FT_EMPLOYED",
            "FT_EMPLOYED",
            "FT_EMPLOYED",
        ],
    )
    weights = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 1.0])
    # Five people aged 21 to 24 at 0.4 a head: a target of two people.
    draw = derive_is_apprentice(person, rate=0.4, person_weights=weights)

    draws = stable_identity_uniforms(
        person["person_id"].to_numpy(), seed=0, salt=APPRENTICE_OUTPUT
    )
    eligible = np.array([True, False, True, True, False, True])
    first_two = np.argsort(np.where(eligible, draws, np.inf))[:2]
    expected = np.zeros(len(person), dtype=bool)
    expected[first_two] = True
    np.testing.assert_array_equal(draw.flags, expected)
    reordered = derive_is_apprentice(
        person.iloc[::-1].reset_index(drop=True),
        rate=0.4,
        person_weights=weights[::-1],
    )
    np.testing.assert_array_equal(reordered.flags, expected[::-1])


def test_nobody_is_flagged_when_the_target_is_nearer_zero() -> None:
    person = _small([22, 30], ["FT_EMPLOYED", "STUDENT"])

    # A target of 0.1 people sits nearer zero than the one employee's weight.
    draw = derive_is_apprentice(person, rate=0.1, person_weights=np.ones(2))

    assert not draw.flags.any()
    assert draw.receipt["realized_mass"] == 0.0
    assert draw.receipt["cap_bound"] is False


def test_a_short_eligible_weight_flags_every_eligible_employee() -> None:
    person = _small([21, 22, 23], ["FT_EMPLOYED", "UNEMPLOYED", "UNEMPLOYED"])

    draw = derive_is_apprentice(person, rate=0.9)

    np.testing.assert_array_equal(draw.flags, [True, False, False])
    assert draw.receipt["cap_bound"] is True


def test_no_eligible_employee_flags_nobody() -> None:
    person = _small([22, 30], ["STUDENT", "FT_EMPLOYED"])

    draw = derive_is_apprentice(person, rate=0.06)

    assert not draw.flags.any()
    assert draw.receipt["eligible_support"] == 0
    assert draw.receipt["cap_bound"] is True


def test_receipt_withholds_a_flagged_weight_on_fewer_than_three_households() -> None:
    person = _small([22] * 20, ["FT_EMPLOYED"] * 20)

    draw = derive_is_apprentice(person, rate=0.1)

    assert int(draw.flags.sum()) == 2
    assert draw.receipt["flagged_support"] == "<3"
    assert draw.receipt["realized_mass"] == "<3"
    assert draw.receipt["realization_deviation"] == "<3"
    assert draw.receipt["population_mass"] == 20.0


def test_receipt_withholds_a_population_on_fewer_than_three_households() -> None:
    # Six people aged 21 to 24 in two households, with a one-person target.
    person = _small(
        [22, 23, 21, 24, 22, 23],
        ["FT_EMPLOYED"] * 6,
        households=[1, 1, 1, 2, 2, 2],
    )

    draw = derive_is_apprentice(person, rate=1 / 6)

    receipt = draw.receipt
    assert receipt["population_support"] == "<3"
    for key in (
        "population_mass",
        "eligible_mass",
        "target",
        "realized_mass",
        "realization_deviation",
    ):
        assert receipt[key] == "<3", key


def test_refuses_a_frame_without_employment_status() -> None:
    person = pd.DataFrame({"person_id": [1], "person_household_id": [1], "age": [22]})

    with pytest.raises(ValueError, match="employment_status"):
        derive_is_apprentice(person, rate=0.06)


def test_refuses_a_rate_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        derive_is_apprentice(_people(1), rate=1.5)


def _two_household_frame(**extra_person_columns):
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 4001),
            "person_benunit_id": np.arange(1, 4001),
            "person_household_id": np.repeat([1, 2], 2000),
            "age": np.tile([22, 23], 2000),
            "employment_status": np.repeat(["FT_EMPLOYED", "STUDENT"], 2000),
            **extra_person_columns,
        }
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": np.arange(1, 4001), "is_married": False}),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "region": ["LONDON", "WALES"],
                "household_weight": [1.0, 3.0],
            }
        ),
        time_period="2024",
    )


@pytest.fixture
def no_engine(monkeypatch):
    """Run the stage as the engine-free job does, whatever is installed."""

    monkeypatch.setattr(frs_person_draws, "_installed_uk_system", lambda: None)


def test_stage_realizes_the_target_at_the_frame_household_weights(no_engine) -> None:
    stage = UKFRSPersonDrawsStageTransform(contract=_Contract(), stage=None)

    stored = stage(_two_household_frame()).table("person")

    # Household 1 holds the employees and weighs 1, household 2 the students
    # and weighs 3: the target is 0.06 x (2000 + 6000) = 480 employees.
    flags = stored[APPRENTICE_OUTPUT].to_numpy()
    assert not flags[2000:].any()
    assert int(flags[:2000].sum()) == 480
    receipt = stage.checkpoint_metadata()["evidence"]
    assert receipt["stage"] == "frs_person_draws"
    apprentices = receipt["apprentice_receipt"]
    assert apprentices["build_year"] == 2024
    assert apprentices["rate"] == 0.06
    # Two households carry everyone, so the receipt withholds the weights.
    assert apprentices["population_support"] == "<3"
    assert apprentices["target"] == "<3"
    assert apprentices["flagged_support"] == "<3"


def test_stage_has_no_receipt_before_it_runs() -> None:
    stage = UKFRSPersonDrawsStageTransform(contract=_Contract(), stage=None)

    with pytest.raises(RuntimeError, match="completed stage run"):
        stage.checkpoint_metadata()


def test_stage_refuses_a_stored_flag(no_engine) -> None:
    frame = _two_household_frame(is_apprentice=False)

    with pytest.raises(ValueError, match="already stored"):
        UKFRSPersonDrawsStageTransform(contract=_Contract(), stage=None)(frame)


_SCHEDULE = {
    "period": 2024,
    "rate": 0.138,
    "secondary_threshold": 9_100.0,
    "apprentice_upper_secondary_threshold": 50_270.0,
}


def test_receipt_records_the_relief_the_flags_imply() -> None:
    pay = np.array([5_000.0, 20_000.0, 60_000.0] * 4)
    person = _small([22] * 12, ["FT_EMPLOYED"] * 12).assign(employment_income=pay)

    # Every employee is flagged: a rate of one a head is cap-bound.
    draw = derive_is_apprentice(person, rate=1.0, relief_schedule=_SCHEDULE)

    assert draw.flags.all()
    relieved = np.clip(np.minimum(pay, 50_270.0) - 9_100.0, 0.0, None)
    assert draw.receipt["implied_relief"] == pytest.approx(0.138 * relieved.sum())
    assert draw.receipt["relief_schedule"] == _SCHEDULE


def test_receipt_withholds_a_relief_on_fewer_than_three_households() -> None:
    person = _small([22] * 2, ["FT_EMPLOYED"] * 2).assign(employment_income=20_000.0)

    draw = derive_is_apprentice(person, rate=1.0, relief_schedule=_SCHEDULE)

    assert draw.receipt["implied_relief"] == "<3"


def test_receipt_records_no_relief_without_a_schedule() -> None:
    person = _small([22] * 12, ["FT_EMPLOYED"] * 12).assign(employment_income=1.0)

    assert "implied_relief" not in derive_is_apprentice(person, rate=0.5).receipt


def test_the_graph_node_reads_every_person_column_the_draw_needs() -> None:
    """A node receives its declared cells alone, so the declaration must name
    the pay the relief reads: the licensed spine at ab7c777e2 recorded no
    implied relief, and the gate blocked it, before the declaration followed
    (microcosm#1373)."""
    from microcosm.build.uk_runtime.graph import uk_spine_graph

    node = uk_spine_graph(source_mode="split").node("frs_person_draws")
    inputs = {
        (slice_.entity, column) for slice_ in node.inputs for column in slice_.columns
    }
    assert {
        ("person", "age"),
        ("person", "employment_status"),
        ("person", "employment_income"),
    } <= inputs


def _system(*, variables, apprentice_threshold=966.73):
    thresholds = SimpleNamespace()
    if apprentice_threshold is not None:
        thresholds.apprentice_upper_secondary_threshold = apprentice_threshold
    class_1 = SimpleNamespace(
        employer_scale=SimpleNamespace(
            components=[None, SimpleNamespace(rate=0.138, threshold=175.0)]
        ),
        thresholds=thresholds,
    )
    instants = []

    def parameters(instant):
        instants.append(instant)
        return SimpleNamespace(
            gov=SimpleNamespace(
                hmrc=SimpleNamespace(
                    national_insurance=SimpleNamespace(class_1=class_1)
                )
            )
        )

    return SimpleNamespace(
        variables=dict.fromkeys(variables), parameters=parameters
    ), instants


_SPLIT = ("qualifies_for_apprentice_minimum_wage", "is_in_first_year_of_apprenticeship")


def test_schedule_reads_the_engine_at_the_start_of_the_stage_year() -> None:
    system, instants = _system(variables=_SPLIT)

    schedule = apprentice_relief_schedule(system, "2024")

    assert instants == ["2024-01-01"]
    assert schedule == {
        "period": 2024,
        "rate": 0.138,
        "secondary_threshold": 175.0 * 52,
        "apprentice_upper_secondary_threshold": 966.73 * 52,
    }


def test_schedule_is_absent_without_an_engine() -> None:
    assert apprentice_relief_schedule(None, "2024") is None


def test_schedule_refuses_an_engine_without_the_minimum_wage_split() -> None:
    system, _ = _system(variables=())

    with pytest.raises(ValueError, match="2683"):
        apprentice_relief_schedule(system, "2024")


def test_schedule_refuses_an_engine_without_the_apprentice_threshold() -> None:
    system, _ = _system(variables=_SPLIT, apprentice_threshold=None)

    with pytest.raises(ValueError, match="apprentice_upper_secondary_threshold"):
        apprentice_relief_schedule(system, "2024")
