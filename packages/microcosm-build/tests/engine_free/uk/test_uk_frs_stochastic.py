"""Tests split from packages/microcosm-build/tests/test_uk_frs_stochastic.py."""

# ruff: noqa: F403, F405
from microcosm.build.stochastic_assignment import sample_categorical_from_counts
from microcosm.build.uk_runtime.frs_brma import (
    UK_BRMA_BENUNIT_SALT,
    UK_BRMA_HOUSEHOLD_PICK_SALT,
    _benunit_regions,
    add_frs_brma,
    draw_household_brmas,
    load_brma_count_resource,
)
from microcosm.build.uk_runtime.receipt_counts import own_source_households
from test_support.microcosm_build.uk_frs_stochastic import *


def test_take_up_anchor_missing_source_column_fails_loud() -> None:
    frame = _frame()
    person = frame.table("person").drop(columns=["pension_credit_reported"])

    with pytest.raises(KeyError, match="pension_credit_reported"):
        aggregate_person_reported_to_benunit(person, frame.table("benunit"))


def test_take_up_anchors_or_over_persons_and_stage_writes_outputs() -> None:
    from microcosm.build.uk_runtime.frs_take_up import take_up_signal_outputs

    frame = _frame()
    stage = UKFRSTakeUpStageTransform(
        contract=_Contract(), stage=_take_up_stage(), population_policy=_POLICY
    )
    transformed = stage(frame)
    benunit = transformed.table("benunit")
    # The receipt carries the take-up flags the stage draws, for its stage-time
    # signal gate (microcosm#1095).
    evidence = stage.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == "frs_take_up"
    assert [row["output"] for row in evidence["take_up_signal"]["flags"]] == [
        f"{entity}.{output}"
        for entity, output, _ in take_up_signal_outputs(
            UKFRSTakeUpStageTransform.output_columns()
        )
    ]

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
    # No child under 2 here: every person keeps the engine's 30 hours.
    person = transformed.table("person")
    assert (person["max_free_entitlement_hours_used"] == 30.0).all()


def test_under_two_usage_draws_children_in_claiming_units_only() -> None:
    from microcosm.build.uk_runtime.frs_take_up import (
        UK_ENGINE_FREE_HOURS_DEFAULT,
        UK_UNDER_TWO_USAGE_OUTPUT,
        derive_under_two_childcare_usage,
    )

    # The draw keys on the DfE age (31 December): a whole-year age plus one for
    # a birthday from 7 October, more than about 9.19 months before 6 October.
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 6001),
            "age": np.tile([0, 0, 1, 1, 2, 35], 1000),
            "months_since_last_birthday": np.tile(
                [3.0, 11.0, 3.0, 11.0, 3.0, 3.0], 1000
            ),
        }
    )
    claiming = np.repeat([True, False], 3000)
    usage = derive_under_two_childcare_usage(
        person, claiming=claiming, contract=_Contract()
    )
    draws = stable_identity_uniforms(
        person["person_id"].to_numpy(), seed=0, salt=UK_UNDER_TWO_USAGE_OUTPUT
    )
    # DfE ages 0, 1, 1, 2, 2, 35: a child of whole age 1 born from 7 October is
    # a 2-year-old to DfE and is not drawn.
    under_two = np.tile([True, True, True, False, False, False], 1000)
    expected = np.where(
        under_two & claiming & (draws >= 0.5), 0.0, UK_ENGINE_FREE_HOURS_DEFAULT
    )
    np.testing.assert_array_equal(usage, expected)
    # Only claiming children under 2 can lose their hours, at about the rate.
    assert (usage[~(under_two & claiming)] == 30.0).all()
    share = float((usage[under_two & claiming] > 0).mean())
    assert abs(share - 0.5) < 0.05


def test_person_draws_pin_scp_age_six_boundary_and_uniform_draw() -> None:
    person = _frame().table("person")

    derived = derive_frs_person_draws(
        person,
        contract=_Contract(),
        person_weights=np.ones(len(person)),
        person_sources=own_source_households(person),
    )

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
    # Each stage's receipt carries the take-up flags it draws, for its
    # stage-time signal gate (microcosm#1095).
    person_evidence = person_stage.checkpoint_metadata()["evidence"]
    assert "apprentice_receipt" in person_evidence
    assert [row["output"] for row in person_evidence["take_up_signal"]["flags"]] == [
        "person.would_claim_marriage_allowance",
        "person.would_claim_scp",
    ]
    household_evidence = household_stage.checkpoint_metadata()["evidence"]
    assert [row["output"] for row in household_evidence["take_up_signal"]["flags"]] == [
        f"household.{column}" for column in FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS
    ]


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


_BRMA_REGIONS = ("LONDON", "NORTHERN_IRELAND", "SCOTLAND")
_LHA_CATEGORIES = ("A", "B", "C", "D", "E")


def _brma_frame() -> object:
    """Twelve invented households in three regions, with one to three
    benefit units each and one person per benefit unit."""

    households = np.arange(1, 13)
    units = [
        (household, household * 100 + unit)
        for household in households.tolist()
        for unit in range(1, 2 + household % 3)
    ]
    benunit_ids = [unit for _, unit in units]
    person = pd.DataFrame(
        {
            "person_id": [unit * 10 + 1 for unit in benunit_ids],
            "person_benunit_id": benunit_ids,
            "person_household_id": [household for household, _ in units],
            "age": [40] * len(units),
            "is_uc_claimant": [True] * len(units),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": households,
            "region": [_BRMA_REGIONS[h % 3] for h in households.tolist()],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": benunit_ids}),
        household=household,
        household_weights=np.full(len(households), 10.0),
        time_period="2024",
    )


def _brma_lha(benunit: pd.DataFrame) -> list[str]:
    return [_LHA_CATEGORIES[int(unit) % 5] for unit in benunit["benunit_id"]]


def test_add_frs_brma_draws_what_the_cell_draw_and_household_pick_give() -> None:
    """The stage's values are today's: each benefit unit's cell draw under
    salt "brma", then each household's pick under "brma:household_pick",
    both keyed on ids with seed 0; the benefit-unit table is left as it was."""

    frame = _brma_frame()
    resource = load_brma_count_resource()
    person, benunit = frame.table("person"), frame.table("benunit")
    household = frame.table("household")
    assert (UK_BRMA_BENUNIT_SALT, UK_BRMA_HOUSEHOLD_PICK_SALT) == (
        "brma",
        "brma:household_pick",
    )
    cells = benunit.assign(
        LHA_category=_brma_lha(benunit),
        region=_benunit_regions(person, household, benunit),
    )
    drawn = assign_brma_by_cell(cells, count_resource=resource, seed=0)
    expected = collapse_benunit_brma_to_household(
        person, cells[["benunit_id"]].assign(brma=drawn), household, seed=0
    )
    result = add_frs_brma(
        frame, lha_category=_brma_lha(benunit), count_resource=resource
    )
    assert result.table("household")["brma"].tolist() == expected.tolist()
    assert result.table("benunit").equals(benunit)
    # The cell draw, written out for one benefit unit.
    first = cells.iloc[0]
    uniform = stable_identity_uniforms([first["benunit_id"]], seed=0, salt="brma")
    counts = resource["cells"][first["region"]][first["LHA_category"]]
    assert drawn[0] == sample_categorical_from_counts(uniform, counts=counts)[0]


def test_add_frs_brma_picks_each_households_brma_as_written_out() -> None:
    """The household pick, which sets frs_brma's live values, written out
    without the module's pick: the uniform keyed on the household's id under
    salt "brma:household_pick" and seed 0 picks among its benefit units'
    cell draws in ascending benefit-unit id."""

    frame = _brma_frame()
    resource = load_brma_count_resource()
    person, benunit = frame.table("person"), frame.table("benunit")
    household = frame.table("household")
    lha = _brma_lha(benunit)
    cells = benunit.assign(
        LHA_category=lha, region=_benunit_regions(person, household, benunit)
    )
    drawn = pd.Series(
        assign_brma_by_cell(cells, count_resource=resource, seed=0),
        index=benunit["benunit_id"].to_numpy(),
    )
    owner = person.drop_duplicates("person_benunit_id").set_index("person_benunit_id")[
        "person_household_id"
    ]
    result = add_frs_brma(frame, lha_category=lha, count_resource=resource)
    picked = result.table("household")["brma"].tolist()
    choices = 0
    for row, household_id in enumerate(household["household_id"].tolist()):
        units = sorted(int(unit) for unit in owner.index[owner == household_id])
        options = [str(drawn[unit]) for unit in units]
        (uniform,) = stable_identity_uniforms(
            [household_id], seed=0, salt="brma:household_pick"
        )
        expected = options[min(int(uniform * len(options)), len(options) - 1)]
        assert picked[row] == expected
        choices += len(set(options)) > 1
    # The invented frame holds households whose benefit units drew BRMAs
    # that differ, so the pick is exercised.
    assert choices


def test_brma_draw_of_a_household_subset_takes_the_whole_frames_draws() -> None:
    """Every uniform is keyed on its own id alone, so a subset of the
    households, given every benefit unit or only its own, takes the whole
    frame's BRMAs; benefit units outside the subset are never read."""

    frame = _brma_frame()
    resource = load_brma_count_resource()
    person, benunit = frame.table("person"), frame.table("benunit")
    household = frame.table("household")
    lha = _brma_lha(benunit)
    whole = add_frs_brma(frame, lha_category=lha, count_resource=resource)
    whole_brma = whole.table("household")["brma"].to_numpy()
    for region in _BRMA_REGIONS:
        rows = (household["region"] == region).to_numpy()
        subset = household.loc[rows]
        drawn = draw_household_brmas(
            person, benunit, subset, lha_category=lha, count_resource=resource
        )
        assert drawn.tolist() == whole_brma[rows].tolist()
        members = person.loc[person["person_household_id"].isin(subset["household_id"])]
        own = benunit["benunit_id"].isin(members["person_benunit_id"]).to_numpy()
        alone = draw_household_brmas(
            members,
            benunit.loc[own],
            subset,
            lha_category=[value for value, keep in zip(lha, own, strict=True) if keep],
            count_resource=resource,
        )
        assert alone.tolist() == whole_brma[rows].tolist()
        # A category no cell holds, on the other households' benefit units.
        foreign = [value if keep else "Z" for value, keep in zip(lha, own, strict=True)]
        assert (
            draw_household_brmas(
                person, benunit, subset, lha_category=foreign, count_resource=resource
            ).tolist()
            == whole_brma[rows].tolist()
        )


def test_brma_refusals_name_no_household_or_benefit_unit_id() -> None:
    frame = _brma_frame()
    resource = load_brma_count_resource()
    person, benunit = frame.table("person"), frame.table("benunit")
    household = frame.table("household")
    lha = _brma_lha(benunit)
    # A benefit unit whose members sit in two households.
    split = person.copy()
    split.loc[split.index[0], "person_household_id"] = int(
        household["household_id"].iloc[-1]
    )
    split = pd.concat([split, person.iloc[[0]].assign(person_id=99_999)])
    with pytest.raises(ValueError, match="splits a benefit unit") as refused:
        draw_household_brmas(
            split, benunit, household, lha_category=lha, count_resource=resource
        )
    assert not any(character.isdigit() for character in str(refused.value))
    # A household without a benefit unit drawn for it.
    with pytest.raises(ValueError, match="has no benefit unit") as refused:
        collapse_benunit_brma_to_household(
            person,
            pd.DataFrame({"benunit_id": benunit["benunit_id"].iloc[:1], "brma": ["X"]}),
            household,
            seed=0,
        )
    assert not any(character.isdigit() for character in str(refused.value))
    with pytest.raises(ValueError, match="must align"):
        draw_household_brmas(
            person, benunit, household, lha_category=lha[:-1], count_resource=resource
        )


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

    policyengine-uk 2.174.1's is_uc_eligible reads the age of the unit's UC
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
