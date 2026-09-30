"""Tests split from packages/microcosm-build/tests/test_uk_frs_stochastic.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_frs_stochastic import *


def test_take_up_anchor_missing_source_column_fails_loud() -> None:
    frame = _frame()
    person = frame.table("person").drop(columns=["pension_credit_reported"])

    with pytest.raises(KeyError, match="pension_credit_reported"):
        aggregate_person_reported_to_benunit(person, frame.table("benunit"))


def test_take_up_anchors_or_over_persons_and_stage_writes_outputs() -> None:
    frame = _frame()
    engine = WorkingAgeStubEngine()
    transformed = UKFRSTakeUpStageTransform(
        contract=_Contract(), stage=_take_up_stage(), engine=engine
    )(frame)
    benunit = transformed.table("benunit")

    # The stage asks the engine once, for is_WA_adult at the frame's period.
    assert engine.calls == [(UK_TAKE_UP_ENGINE_PREDICTORS, "2024")]

    anchors = aggregate_person_reported_to_benunit(
        frame.table("person"), frame.table("benunit")
    )
    assert anchors["child_benefit_reported_anchor"].tolist() == [True, False, False]
    assert anchors["pension_credit_reported_anchor"].tolist() == [False, False, True]
    assert anchors["universal_credit_reported_anchor"].tolist() == [False, True, False]
    assert set(FRS_TAKE_UP_OUTPUT_COLUMNS) <= set(benunit.columns)
    assert bool(benunit.loc[0, "would_claim_child_benefit"])
    assert bool(benunit.loc[1, "would_claim_uc"])
    assert bool(benunit.loc[2, "would_claim_pc"])
    assert (
        (0 <= benunit["maximum_extended_childcare_hours_usage"])
        & (benunit["maximum_extended_childcare_hours_usage"] <= 30)
    ).all()


def test_person_draws_pin_scp_age_six_boundary_and_uniform_draw() -> None:
    person = _frame().table("person")

    derived = derive_frs_person_draws(person, contract=_Contract())

    draws = stable_identity_uniforms(
        person["person_id"].to_numpy(), seed=0, salt="would_claim_scp"
    )
    expected = draws < np.array([0.97, 0.85, 0.85, 0.85])
    np.testing.assert_array_equal(derived["would_claim_scp"].to_numpy(), expected)
    private_draws = derived["attends_private_school_random_draw"].to_numpy()
    assert ((0 <= private_draws) & (private_draws < 1)).all()
    assert (derived["tax_free_childcare_spend_routed_share"] == 0.593).all()


def test_person_and_household_stage_families_are_deterministic() -> None:
    frame = _frame()
    person_stage = UKFRSPersonDrawsStageTransform(contract=_Contract(), stage=None)
    household_stage = UKFRSHouseholdDrawsStageTransform(
        contract=_Contract(), stage=None
    )

    person_a = person_stage(frame).table("person")
    person_b = person_stage(frame).table("person")
    household_a = household_stage(frame).table("household")
    household_b = household_stage(frame).table("household")

    pd.testing.assert_frame_equal(
        person_a[list(FRS_PERSON_DRAW_OUTPUT_COLUMNS)],
        person_b[list(FRS_PERSON_DRAW_OUTPUT_COLUMNS)],
    )
    pd.testing.assert_frame_equal(
        household_a[list(FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS)],
        household_b[list(FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS)],
    )


def test_brma_cell_membership_and_household_collapse() -> None:
    frame = _frame()
    resource = {
        "cells": {
            "LONDON": {"A": {"LONDON_A": 4, "LONDON_B": 1}},
            "SCOTLAND": {"B": {"SCOTLAND_A": 1}},
        }
    }
    benunit = pd.DataFrame(
        {
            "benunit_id": [10, 20, 30],
            "region": ["LONDON", "LONDON", "SCOTLAND"],
            "LHA_category": ["A", "A", "B"],
        }
    )
    assigned = assign_brma_by_cell(benunit, count_resource=resource, seed=0)

    assert set(assigned[:2]) <= {"LONDON_A", "LONDON_B"}
    assert assigned[2] == "SCOTLAND_A"
    collapsed = collapse_benunit_brma_to_household(
        frame.table("person"),
        pd.DataFrame({"benunit_id": [10, 20, 30], "brma": assigned}),
        frame.table("household"),
        seed=0,
    )
    assert collapsed[0] in assigned[:2]
    assert collapsed[1] == "SCOTLAND_A"


def test_brma_stage_materializes_lha_category_and_writes_household_brma() -> None:
    frame = _frame()
    resource = {
        "cells": {
            "LONDON": {"A": {"LONDON_A": 1}},
            "SCOTLAND": {"B": {"SCOTLAND_A": 1}},
        }
    }
    stage = UKFRSBRMAStageTransform(
        stage=None,
        engine=_FakeEngine(["A", "A", "B"]),
        count_resource=resource,
    )

    transformed = stage(frame)

    assert transformed.table("household")["brma"].tolist() == [
        "LONDON_A",
        "SCOTLAND_A",
    ]
    assert "brma" not in transformed.table("benunit")


def test_brma_missing_cell_fails_closed() -> None:
    benunit = pd.DataFrame(
        {"benunit_id": [1], "region": ["LONDON"], "LHA_category": ["Z"]}
    )

    with pytest.raises(KeyError, match="missing BRMA"):
        assign_brma_by_cell(benunit, count_resource={"cells": {"LONDON": {}}}, seed=0)


def test_uc_take_up_population_excludes_units_without_a_working_age_adult() -> None:
    """A unit with no working-age adult is never drawn into UC."""

    frame = _frame()
    person, benunit = frame.table("person"), frame.table("benunit")

    eligible = uc_age_eligible_benunits(person, benunit, population_from_ages(person))
    assert eligible.tolist() == [False, True, False]  # children only / 40 / 70

    anchors = aggregate_person_reported_to_benunit(person, benunit)
    derived = derive_frs_take_up(
        benunit, anchors=anchors, contract=_Contract(), uc_age_eligible=eligible
    )
    assert derived["would_claim_uc"].tolist() == [False, True, False]

    # An anchor outside the population stays true: reported receipt is a fact.
    anchors.loc[2, "universal_credit_reported_anchor"] = True
    derived = derive_frs_take_up(
        benunit, anchors=anchors, contract=_Contract(), uc_age_eligible=eligible
    )
    assert derived["would_claim_uc"].tolist() == [False, True, True]

    with pytest.raises(ValueError, match="uc_age_eligible must align"):
        derive_frs_take_up(
            benunit, anchors=anchors, contract=_Contract(), uc_age_eligible=eligible[:2]
        )
    misaligned = UKTakeUpPopulation(
        working_age_adult=np.array([True, False]), period="2024", source="test"
    )
    with pytest.raises(ValueError, match="must align with the person table"):
        uc_age_eligible_benunits(person, benunit, misaligned)


def test_uc_take_up_population_is_the_engines_per_person_status() -> None:
    """Units follow each member's is_WA_adult, not an age bound.

    From 2026-27 State Pension age follows the date of birth, so of two
    66-year-olds one can be under it and the other over (policyengine-uk#1899).
    The stage takes the engine's per-person answer as it is: the unit whose
    66-year-old is under State Pension age joins the draw's population, the
    unit whose 66-year-old is over it does not.
    """

    person = pd.DataFrame(
        {
            "person_id": [101, 201, 301, 401],
            "person_benunit_id": [10, 20, 30, 40],
            "person_household_id": [1, 2, 3, 4],
            "age": [66, 66, 67, 17],
            "gender": ["MALE", "FEMALE", "MALE", "FEMALE"],
            "child_benefit_reported": [0, 0, 0, 0],
            "pension_credit_reported": [0, 0, 0, 0],
            "universal_credit_reported": [0, 0, 0, 0],
        }
    )
    frame = uk_national_frame(
        person=person,
        benunit=pd.DataFrame(
            {"benunit_id": [10, 20, 30, 40], "is_married": [False] * 4}
        ),
        household=pd.DataFrame(
            {"household_id": [1, 2, 3, 4], "household_weight": [1.0] * 4}
        ),
        time_period="2026",
    )
    engine = WorkingAgeStubEngine(working_age_adult=[True, False, False, False])

    population = uk_take_up_population(frame, engine)

    assert engine.calls == [(UK_TAKE_UP_ENGINE_PREDICTORS, "2026")]
    assert population.working_age_adult.tolist() == [True, False, False, False]
    assert population.period == "2026"
    assert population.evidence()["working_age_adults"] == 1
    eligible = uc_age_eligible_benunits(person, frame.table("benunit"), population)
    assert eligible.tolist() == [True, False, False, False]

    # The engine's answer must be one boolean per person row.
    with pytest.raises(ValueError, match="one boolean per person row"):
        uk_take_up_population(
            frame, WorkingAgeStubEngine(working_age_adult=[1, 0, 0, 0])
        )
    with pytest.raises(ValueError, match="has shape"):
        uk_take_up_population(frame, WorkingAgeStubEngine(working_age_adult=[True]))


def test_take_up_stage_refuses_a_manifest_that_drops_the_population_declaration() -> (
    None
):
    """The manifest's population declaration is read, not decorative."""

    stage = _take_up_stage()
    assert_take_up_stage_population_declaration(stage)

    stripped = [
        SourceOperationSpec(op.kind, {**op.parameters, "population": "everyone"})
        if op.parameters.get("output") == "would_claim_uc"
        else op
        for op in stage.operations
    ]
    broken = replace(stage, operations=tuple(stripped))
    with pytest.raises(ValueError, match="population='uc_age_eligible'"):
        assert_take_up_stage_population_declaration(broken)
    engine = WorkingAgeStubEngine()
    with pytest.raises(ValueError, match="population='uc_age_eligible'"):
        UKFRSTakeUpStageTransform(contract=_Contract(), stage=broken, engine=engine)(
            _frame()
        )
    # The refusal comes before the engine is asked anything.
    assert engine.calls == []

    # A manifest that stops declaring the engine read, or declares the
    # aggregate over something else, is refused too.
    for mutate in (
        lambda op: (
            SourceOperationSpec(op.kind, {**op.parameters, "predictors": ["age"]})
            if op.kind == "materialize_rules_engine_predictors"
            else op
        ),
        lambda op: (
            SourceOperationSpec(
                op.kind, {**op.parameters, "aggregates": {"uc_age_eligible": "age"}}
            )
            if op.parameters.get("method") == "any"
            else op
        ),
    ):
        mutated = replace(
            stage, operations=tuple(mutate(op) for op in stage.operations)
        )
        with pytest.raises(ValueError, match="consumed engine read of"):
            assert_take_up_stage_population_declaration(mutated)


def test_uc_childcare_take_up_is_drawn_by_family_type() -> None:
    """The couple rate applies where is_married is set; zero means never drawn."""

    frame = _frame()
    person, benunit = frame.table("person"), frame.table("benunit")
    anchors = aggregate_person_reported_to_benunit(person, benunit)
    derived = derive_frs_take_up(
        benunit,
        anchors=anchors,
        contract=_Contract(),
        uc_age_eligible=uc_age_eligible_benunits(
            person, benunit, population_from_ages(person)
        ),
    )
    # Benefit unit 20 is the couple; its rate is 0.0 in the fixture contract.
    assert not derived.loc[1, "would_claim_uc_childcare"]
    assert derived["would_claim_uc_childcare"].dtype == bool
    with pytest.raises(KeyError, match="benunit.is_married is missing"):
        derive_frs_take_up(
            benunit.drop(columns=["is_married"]),
            anchors=anchors,
            contract=_Contract(),
            uc_age_eligible=uc_age_eligible_benunits(
                person, benunit, population_from_ages(person)
            ),
        )
