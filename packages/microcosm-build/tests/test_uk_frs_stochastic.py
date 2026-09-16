from __future__ import annotations

import json
from dataclasses import replace
from importlib import metadata, resources
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import (
    SourceManifest,
    SourceOperationSpec,
    SourceStageSpec,
)
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_brma import (
    UKFRSBRMAStageTransform,
    assign_brma_by_cell,
    collapse_benunit_brma_to_household,
)
from microcosm.build.uk_runtime.frs_household_draws import (
    FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS,
    UKFRSHouseholdDrawsStageTransform,
)
from microcosm.build.uk_runtime.frs_person_draws import (
    FRS_PERSON_DRAW_OUTPUT_COLUMNS,
    UKFRSPersonDrawsStageTransform,
    derive_frs_person_draws,
)
from microcosm.build.uk_runtime.frs_take_up import (
    FRS_TAKE_UP_OUTPUT_COLUMNS,
    UKFRSTakeUpStageTransform,
    UKTakeUpPopulationPolicy,
    aggregate_person_reported_to_benunit,
    assert_take_up_stage_population_declaration,
    derive_frs_take_up,
    uc_age_eligible_benunits,
    uk_take_up_population_policy,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame

# The engine's 2025 working-age bounds (is_adult at 18, State Pension age 66),
# injected so the hermetic tests need no engine; the lockstep test below checks
# the reader returns exactly this.
_POLICY = UKTakeUpPopulationPolicy(
    adult_age=18, state_pension_age=66, instant="2025-01-01", source="test"
)


def _take_up_stage() -> SourceStageSpec:
    manifest = SourceManifest.from_mapping(
        json.loads(
            resources.files("microcosm.build.uk")
            .joinpath("source_stages.json")
            .read_text(encoding="utf-8")
        )
    )
    return next(stage for stage in manifest.stages if stage.stage == "frs_take_up")


class _Contract:
    rates = {
        "child_benefit": 0.5,
        "child_benefit_opts_out_rate": 0.0,
        "pension_credit": 0.5,
        "universal_credit": 0.5,
        "tax_free_childcare": 0.5,
        "extended_childcare": 0.5,
        "universal_childcare": 0.5,
        "targeted_childcare": 0.5,
        "uc_childcare_single": 0.5,
        "uc_childcare_couple": 0.0,
        "marriage_allowance": 0.5,
        "scp_under_6": 0.97,
        "scp_6_plus": 0.85,
        "tv_ownership_rate": 0.5,
        "tv_licence_evasion_rate": 0.5,
        "first_time_buyer_rate": 0.5,
        "property_purchase_rate": 0.5,
        "tax_free_childcare_spend_routed_share": 0.593,
    }

    def rate(self, key: str, build_year: int | None = None) -> float:
        return self.rates[key]

    def continuous_entry(self, key: str):
        assert key == "maximum_extended_childcare_hours_usage"
        return {"mean": 15.019, "sd": 4.972, "lower": 0, "upper": 30}

    def entry(self, key: str):
        assert key == "tax_free_childcare_spend_routed_share"
        return SimpleNamespace(raw={"entity": "person"})

    build_year = 2024


class _FakeEngine:
    country = "uk"

    def __init__(self, lha_category):
        self.lha_category = lha_category

    def materialize(self, frame, variables, period):
        assert tuple(variables) == ("LHA_category",)
        assert period == "2024"
        return {"LHA_category": self.lha_category}


def _frame() -> object:
    person = pd.DataFrame(
        {
            "person_id": [101, 102, 201, 301],
            "person_benunit_id": [10, 10, 20, 30],
            "person_household_id": [1, 1, 1, 2],
            "age": [5, 6, 40, 70],
            "child_benefit_reported": [0, 10, 0, 0],
            "pension_credit_reported": [0, 0, 0, 5],
            "universal_credit_reported": [0, 0, 20, 0],
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": [10, 20, 30], "is_married": [False, True, False]}
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "region": ["LONDON", "SCOTLAND"],
            "household_weight": [2.0, 3.0],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
    )


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
    """A unit with no adult under State Pension age is never drawn into UC."""

    frame = _frame()
    person, benunit = frame.table("person"), frame.table("benunit")

    eligible = uc_age_eligible_benunits(person, benunit, _POLICY)
    assert eligible.tolist() == [False, True, False]  # children only / 40 / 70
    # The bounds are the engine's is_WA_adult edges: 17 is out, 18 in, 66 out.
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
    with pytest.raises(KeyError, match="person.age is missing"):
        uc_age_eligible_benunits(person.drop(columns=["age"]), benunit, _POLICY)


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


@pytest.mark.requires_uk
def test_uc_take_up_population_policy_matches_the_engine_working_age_test() -> None:
    """The policy read from the engine reproduces is_WA_adult for every age."""

    policyengine_uk = pytest.importorskip("policyengine_uk")

    policy = uk_take_up_population_policy(2025)
    assert policy == UKTakeUpPopulationPolicy(
        adult_age=18,
        state_pension_age=66,
        instant="2025-01-01",
        source="policyengine-uk parameters " + metadata.version("policyengine-uk"),
    )

    ages = list(range(0, 101))
    people = {f"p{age}": {"age": {2025: age}} for age in ages}
    sim = policyengine_uk.Simulation(
        situation={
            "people": people,
            "benunits": {f"b{age}": {"members": [f"p{age}"]} for age in ages},
            "households": {f"h{age}": {"members": [f"p{age}"]} for age in ages},
        }
    )
    is_wa_adult = np.asarray(sim.calculate("is_WA_adult", 2025), dtype=bool)

    assert is_wa_adult.tolist() == policy.working_age(np.array(ages)).tolist()


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
