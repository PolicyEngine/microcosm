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
    transformed = UKFRSTakeUpStageTransform(
        contract=_Contract(), stage=_take_up_stage(), population_policy=_POLICY
    )(frame)
    benunit = transformed.table("benunit")

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
    """A unit with no UC claimant or partner of working age is never drawn."""

    frame = _frame()
    person, benunit = frame.table("person"), frame.table("benunit")

    eligible = uc_age_eligible_benunits(person, benunit, _POLICY)
    assert eligible.tolist() == [False, True, False]  # children only / 40 / 70
    # The age bounds: 17 is out, 18 in, and 66, Pension Credit qualifying age
    # for the 2025 cohort, out.
    assert _POLICY.working_age(np.array([17, 18, 65, 66])).tolist() == [
        False,
        True,
        True,
        False,
    ]

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
    with pytest.raises(KeyError, match=r"\['age'\] are missing"):
        uc_age_eligible_benunits(person.drop(columns=["age"]), benunit, _POLICY)
    with pytest.raises(KeyError, match=r"\['is_uc_claimant'\] are missing"):
        uc_age_eligible_benunits(
            person.drop(columns=["is_uc_claimant"]), benunit, _POLICY
        )


def test_uc_take_up_population_needs_a_claimant_or_partner_of_working_age() -> None:
    """A working-age dependant does not bring a unit into the UC population.

    policyengine-uk 2.122.2's is_uc_eligible reads the age of the unit's UC
    claimant or partner (is_uc_claimant), not of any adult: a pensioner unit
    with an 18-year-old dependant cannot claim Universal Credit.
    """

    frame = _frame()
    person, benunit = frame.table("person").copy(), frame.table("benunit")
    # Unit 30 (a 70-year-old) gains an 18-year-old dependant; unit 20's
    # 40-year-old stops being a claimant or partner.
    dependant = person.iloc[[3]].assign(person_id=302, age=18, is_uc_claimant=False)
    person = pd.concat([person, dependant], ignore_index=True)
    person.loc[person["person_id"] == 201, "is_uc_claimant"] = False

    eligible = uc_age_eligible_benunits(person, benunit, _POLICY)

    assert eligible.tolist() == [False, False, False]
    person.loc[person["person_id"] == 302, "is_uc_claimant"] = True
    assert uc_age_eligible_benunits(person, benunit, _POLICY).tolist() == [
        False,
        False,
        True,
    ]


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
    with pytest.raises(ValueError, match="population='uc_age_eligible'"):
        UKFRSTakeUpStageTransform(
            contract=_Contract(), stage=broken, population_policy=_POLICY
        )(_frame())


def test_uc_childcare_take_up_is_drawn_by_family_type() -> None:
    """The couple rate applies where is_married is set; zero means never drawn."""

    frame = _frame()
    person, benunit = frame.table("person"), frame.table("benunit")
    anchors = aggregate_person_reported_to_benunit(person, benunit)
    derived = derive_frs_take_up(
        benunit,
        anchors=anchors,
        contract=_Contract(),
        uc_age_eligible=uc_age_eligible_benunits(person, benunit, _POLICY),
    )
    # Benefit unit 20 is the couple; its rate is 0.0 in the fixture contract.
    assert not derived.loc[1, "would_claim_uc_childcare"]
    assert derived["would_claim_uc_childcare"].dtype == bool
    with pytest.raises(KeyError, match="benunit.is_married is missing"):
        derive_frs_take_up(
            benunit.drop(columns=["is_married"]),
            anchors=anchors,
            contract=_Contract(),
            uc_age_eligible=uc_age_eligible_benunits(person, benunit, _POLICY),
        )
