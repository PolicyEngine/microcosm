"""The UC limited capability stage (microcosm#1095, Y1 of the November event-data plan; PolicyEngine/microcosm#1663)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_spine import FRS_UC_LCW_STATUS_COLUMN
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.uc_limited_capability import (
    UC_LCW,
    UC_LCWRA,
    UC_LIMITED_CAPABILITY_AGE,
    UC_LIMITED_CAPABILITY_CARER,
    UC_LIMITED_CAPABILITY_CLAIMANT,
    UC_LIMITED_CAPABILITY_CONDITION,
    UC_LIMITED_CAPABILITY_CONDITION_CELL,
    UC_LIMITED_CAPABILITY_ENGINE_VARIABLES,
    UC_LIMITED_CAPABILITY_ESA,
    UC_LIMITED_CAPABILITY_MASS_CHANGE_REASON,
    UC_LIMITED_CAPABILITY_SALT,
    UC_LIMITED_CAPABILITY_SEED,
    UC_LIMITED_CAPABILITY_STAGE_NAME,
    UKUCLimitedCapabilityStageTransform,
    assign_uc_limited_capability,
)
from microcosm.frame import WeightKind

# One row per person: benefit unit, household, carrier, SPI household, and the
# engine's answers (WRA, LCW, assessed claimant, carer).
_PEOPLE = [
    # 1. A linked NONE single the proxy flags (PIP): both flags go.
    (1, 1, "NONE", False, True, True, True, False),
    # 2. A linked LCWRA single the proxy flags: kept.
    (2, 2, "LCWRA", False, True, True, True, False),
    # 3. A linked LCWRA single the proxy misses: the claimant takes both.
    (3, 3, "LCWRA", False, False, False, True, False),
    # 4-5. A linked LCWRA couple the proxy misses, the UC row's person cares:
    # the partner takes it.
    (4, 4, "LCWRA", False, False, False, True, True),
    (4, 4, "NOT_LINKED", False, False, False, True, False),
    # 6-7. The same couple shape, nobody cares: the UC row's person takes it.
    (5, 5, "LCWRA", False, False, False, True, False),
    (5, 5, "NOT_LINKED", False, False, False, True, False),
    # 8. A linked LCW single the proxy gives WRA: WRA goes, LCW stays.
    (6, 6, "LCW", False, True, True, True, False),
    # 9. An unlinked reporter the proxy flags: an imputed status, at the rate
    # of the linked claims the proxy flags at its age.
    (7, 7, "NOT_LINKED", False, True, True, True, False),
    # 10. An SPI twin with its donor's carrier: ignored, the proxy stands.
    (8, 8, "NONE", True, True, True, True, False),
    # 11. A child in the first claim's unit: the proxy stands.
    (1, 1, "NOT_LINKED", False, True, True, False, False),
]


def _frame(
    people=_PEOPLE,
    *,
    spi_column: bool = True,
    carrier: bool = True,
    ages=None,
    age_column: bool = True,
    conditions=None,
    condition_column: bool = True,
):
    count = len(people)
    person = pd.DataFrame(
        {
            "person_id": [100 + i for i in range(count)],
            "person_benunit_id": [row[0] for row in people],
            "person_household_id": [row[1] for row in people],
        }
    )
    if age_column:
        person[UC_LIMITED_CAPABILITY_AGE] = ages if ages is not None else [30] * count
    if carrier:
        person[FRS_UC_LCW_STATUS_COLUMN] = [row[2] for row in people]
    if condition_column:
        person[UC_LIMITED_CAPABILITY_CONDITION] = (
            conditions if conditions is not None else ["NO_CONDITION"] * count
        )
    households = sorted({row[1] for row in people})
    spi = {row[1]: row[3] for row in people}
    household = pd.DataFrame(
        {
            "household_id": households,
            "region": ["WALES"] * len(households),
            "household_source_id": households,
            # The true source each floor counts (microcosm#931).
            "source_household_id": households,
        }
    )
    if spi_column:
        household["household_is_spi_synthetic"] = [spi[h] for h in households]
    benunits = sorted({row[0] for row in people})
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": benunits}),
        household=household,
        household_weights=np.full(len(households), 1000.0),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def _answers(people=_PEOPLE, *, esa=None) -> dict[str, np.ndarray]:
    return {
        UC_LCWRA: np.asarray([row[4] for row in people]),
        UC_LCW: np.asarray([row[5] for row in people]),
        UC_LIMITED_CAPABILITY_CLAIMANT: np.asarray([row[6] for row in people]),
        UC_LIMITED_CAPABILITY_CARER: np.asarray([row[7] for row in people]),
        UC_LIMITED_CAPABILITY_ESA: np.asarray(
            esa if esa is not None else [False] * len(people)
        ),
    }


def _uniform(benunit: int) -> float:
    return float(
        stable_identity_uniforms(
            [benunit], seed=UC_LIMITED_CAPABILITY_SEED, salt=UC_LIMITED_CAPABILITY_SALT
        )[0]
    )


def test_linked_claims_take_dwps_status_and_the_rest_an_imputed_one() -> None:
    result = assign_uc_limited_capability(_frame(), engine_answers=_answers())
    person = result.frame.table("person")
    # Every claimant is 30: of the three linked claims the proxy flags, one is
    # LCWRA (a third), so the unlinked reporter the proxy flags keeps its flags
    # when its uniform falls below a third and loses both otherwise.
    imputed = _uniform(7) < 1 / 3

    assert person[UC_LCWRA].tolist() == [
        False,  # NONE over the proxy
        True,  # LCWRA kept
        True,  # LCWRA given to the single claimant
        False,  # the carer keeps the carer element...
        True,  # ...and the partner takes LCWRA
        True,  # the UC row's person takes it
        False,
        False,  # LCW: nobody in the claim has WRA
        imputed,  # unlinked: the imputed status
        True,  # SPI twin: the proxy
        True,  # the child: the proxy
    ]
    assert person[UC_LCW].tolist() == [
        False,
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        imputed,
        True,
        True,
    ]
    imputation = result.evidence()["imputation"]
    assert imputation["statuses"] == (
        {"LCWRA": "<3", "NONE": 0} if imputed else {"LCWRA": 0, "NONE": "<3"}
    )
    assert imputation["kept_the_proxy"] == {
        "own_esa_award": 0,
        "spi": "<3",
        "no_rate": 0,
    }
    branches = result.evidence()["branches"]
    assert branches["none_over_a_proxy_flag"] == "<3"
    assert branches["lcwra_given_to_the_partner_of_a_carer"] == "<3"
    assert branches["lcwra_given_to_the_uc_rows_person"] == "<3"
    assert branches["lcw_kept_from_the_proxy"] == "<3"
    assert branches["none_where_the_proxy_agrees"] == 0


def test_the_receipt_withholds_cells_under_three_households() -> None:
    receipt = assign_uc_limited_capability(
        _frame(), engine_answers=_answers()
    ).evidence()

    assert receipt["stage"] == UC_LIMITED_CAPABILITY_STAGE_NAME
    cells = receipt["status_against_the_proxy"]
    # LCWRA where the proxy misses rests on three households, but its partner
    # cell (the proxy agrees) on one, so both are withheld: no total gives the
    # small one back.
    assert cells["LCWRA"]["proxy_does_not"]["support"] == "3-9"
    assert cells["LCWRA"]["proxy_flags"]["support"] == "<3"
    for sides in cells.values():
        for entry in sides.values():
            assert entry["weighted"] == "withheld"
    assert receipt["linked_claims"] == "3-9"


def test_weights_and_mass_pass_through() -> None:
    frame = _frame()
    result = assign_uc_limited_capability(frame, engine_answers=_answers()).frame

    assert result.weights_for("household").total == frame.weights_for("household").total
    assert result.mass_log[-1].reason == UC_LIMITED_CAPABILITY_MASS_CHANGE_REASON


def test_the_stage_refuses_what_it_cannot_read() -> None:
    with pytest.raises(KeyError, match=FRS_UC_LCW_STATUS_COLUMN):
        assign_uc_limited_capability(_frame(carrier=False), engine_answers=_answers())
    with pytest.raises(KeyError, match="household_is_spi_synthetic"):
        assign_uc_limited_capability(
            _frame(spi_column=False), engine_answers=_answers()
        )
    stored = assign_uc_limited_capability(_frame(), engine_answers=_answers()).frame
    with pytest.raises(ValueError, match="only writer"):
        assign_uc_limited_capability(stored, engine_answers=_answers())
    odd = [(*_PEOPLE[0][:2], "LCFWRA", *_PEOPLE[0][3:]), *_PEOPLE[1:]]
    with pytest.raises(ValueError, match="outside the codes"):
        assign_uc_limited_capability(_frame(odd), engine_answers=_answers(odd))
    twice = [
        *_PEOPLE[:5],
        (4, 4, "LCWRA", False, False, False, True, False),
        *_PEOPLE[6:],
    ]
    with pytest.raises(ValueError, match="two people"):
        assign_uc_limited_capability(_frame(twice), engine_answers=_answers(twice))
    short = {name: values[:-1] for name, values in _answers().items()}
    with pytest.raises(ValueError, match="shape"):
        assign_uc_limited_capability(_frame(), engine_answers=short)
    with pytest.raises(KeyError, match=UC_LIMITED_CAPABILITY_AGE):
        assign_uc_limited_capability(
            _frame(age_column=False), engine_answers=_answers()
        )
    with pytest.raises(KeyError, match=UC_LIMITED_CAPABILITY_CONDITION):
        assign_uc_limited_capability(
            _frame(condition_column=False), engine_answers=_answers()
        )
    with pytest.raises(ValueError, match="outside the labels"):
        assign_uc_limited_capability(
            _frame(conditions=["SOMETIMES"] + ["NO_CONDITION"] * (len(_PEOPLE) - 1)),
            engine_answers=_answers(),
        )
    with pytest.raises(ValueError, match="missing values"):
        assign_uc_limited_capability(
            _frame(ages=[30] * (len(_PEOPLE) - 1) + [None]), engine_answers=_answers()
        )


class _Engine:
    def __init__(self, answers: dict[str, np.ndarray]) -> None:
        self.answers = answers
        self.calls: list[tuple[tuple[str, ...], object]] = []

    def materialize(self, frame, variables, period):
        self.calls.append((tuple(variables), period))
        return {name: self.answers[name] for name in variables}


def test_the_transform_asks_the_engine_for_the_declared_variables() -> None:
    stage = load_country_spec("uk").sources.stage_map()[
        UC_LIMITED_CAPABILITY_STAGE_NAME
    ]
    engine = _Engine(_answers())
    transform = UKUCLimitedCapabilityStageTransform(stage=stage, engine=engine)

    out = transform(_frame())

    assert engine.calls == [(UC_LIMITED_CAPABILITY_ENGINE_VARIABLES, "2024")]
    assert transform.output_columns() == (UC_LCWRA, UC_LCW)
    assert set(transform.output_columns()) <= set(out.table("person").columns)
    assert transform.checkpoint_metadata()["evidence"]["stage"] == (
        UC_LIMITED_CAPABILITY_STAGE_NAME
    )


def test_the_committed_stage_binds_its_operations_closed_world() -> None:
    stage = load_country_spec("uk").sources.stage_map()[
        UC_LIMITED_CAPABILITY_STAGE_NAME
    ]
    operations = list(stage.operations)
    assert [operation.kind for operation in operations] == [
        "materialize_rules_engine_predictors",
        "assign_uc_limited_capability_from_admin_status",
        "record_mass_conservation_receipt",
    ]
    drifted = replace(
        operations[1],
        parameters={**dict(operations[1].parameters), "carrier": "uc_status"},
    )
    with pytest.raises(ValueError, match="parameters drifted"):
        UKUCLimitedCapabilityStageTransform(
            stage=replace(stage, operations=(operations[0], drifted, operations[2])),
            engine=_Engine(_answers()),
        )


def _single(unit: int, carrier: str, proxy: bool) -> tuple:
    return (unit, unit, carrier, False, proxy, proxy, True, False)


def test_unlinked_claims_draw_their_cells_linked_rate() -> None:
    people = [
        # Young linked claims the proxy flags: half LCWRA.
        *[_single(unit, "LCWRA", True) for unit in (1, 2, 3)],
        *[_single(unit, "NONE", True) for unit in (4, 5, 6)],
        # Young linked claims the proxy misses: a third LCWRA.
        _single(7, "LCWRA", False),
        *[_single(unit, "NONE", False) for unit in (8, 9)],
        # Linked claims aged 25 to 49 the proxy flags: none LCWRA.
        *[_single(unit, "NONE", True) for unit in (10, 11, 12)],
        *[_single(unit, "NOT_LINKED", True) for unit in range(100, 120)],
        *[_single(unit, "NOT_LINKED", False) for unit in range(200, 220)],
        *[_single(unit, "NOT_LINKED", True) for unit in range(300, 305)],
    ]
    ages = [20] * 9 + [40] * 3 + [20] * 40 + [40] * 5
    result = assign_uc_limited_capability(
        _frame(people, ages=ages), engine_answers=_answers(people)
    )
    person = result.frame.table("person").set_index("person_benunit_id")

    for unit in range(100, 120):
        assert person.loc[unit, UC_LCWRA] == (_uniform(unit) < 0.5)
        assert person.loc[unit, UC_LCW] == (_uniform(unit) < 0.5)
    for unit in range(200, 220):
        assert person.loc[unit, UC_LCWRA] == (_uniform(unit) < 1 / 3)
        assert person.loc[unit, UC_LCW] == (_uniform(unit) < 1 / 3)
    for unit in range(300, 305):
        assert not person.loc[unit, UC_LCWRA]
        assert not person.loc[unit, UC_LCW]
    imputation = result.evidence()["imputation"]
    cells = imputation["cells"]
    assert cells["16_to_24"]["proxy_flags"]["no_limit_a_lot"] == {
        "linked_claims": "3-9",
        "support": "3-9",
        "rate": 0.5,
        "source": "cell",
        "own_esa_award_claims": {"linked_claims": 0, "support": 0, "rate": "withheld"},
    }
    # One linked claim is LCWRA: the rate is drawn at but never printed.
    assert cells["16_to_24"]["proxy_does_not"]["no_limit_a_lot"]["rate"] == "withheld"
    assert cells["16_to_24"]["proxy_does_not"]["no_limit_a_lot"]["source"] == "cell"
    # No linked claim is 50 or over: both sides take their condition group's
    # rate over every age, and no linked claim reports a condition that
    # limits activities a lot, so those cells take their proxy group's.
    for side in ("proxy_flags", "proxy_does_not"):
        assert cells["50_and_over"][side]["no_limit_a_lot"]["source"] == (
            "pooled_by_condition"
        )
        for band in ("16_to_24", "25_to_49", "50_and_over"):
            assert cells[band][side]["limits_a_lot"]["source"] == "pooled"
    # 45 imputed claims, rounded to the nearest ten as every receipt count is.
    assert imputation["claims"] == 50
    assert imputation["kept_the_proxy"] == {"own_esa_award": 0, "spi": 0, "no_rate": 0}


def test_linked_claims_with_their_own_esa_award_stay_out_of_the_rates() -> None:
    people = [
        # Linked claims aged 25 to 49 the proxy flags. With their own ESA award:
        # half LCWRA.
        *[_single(unit, "LCWRA", True) for unit in (1, 2, 3)],
        *[_single(unit, "NONE", True) for unit in (4, 5, 6)],
        # Without one: a third LCWRA.
        *[_single(unit, "LCWRA", True) for unit in (7, 8, 9)],
        *[_single(unit, "NONE", True) for unit in range(10, 16)],
        # Unlinked claims the proxy flags, with no award of their own.
        *[_single(unit, "NOT_LINKED", True) for unit in range(100, 140)],
    ]
    esa = [True] * 6 + [False] * (len(people) - 6)
    unlinked = range(100, 140)
    # Only the claims without an award set the rate, a third: a claim with one
    # keeps the proxy rather than drawing. Every linked claim would give 0.4,
    # and some draw falls between the two, so the test tells them apart.
    assert any(1 / 3 <= _uniform(unit) < 0.4 for unit in unlinked)
    result = assign_uc_limited_capability(
        _frame(people, ages=[40] * len(people)),
        engine_answers=_answers(people, esa=esa),
    )
    person = result.frame.table("person").set_index("person_benunit_id")

    for unit in unlinked:
        assert person.loc[unit, UC_LCWRA] == (_uniform(unit) < 1 / 3)
        assert person.loc[unit, UC_LCW] == (_uniform(unit) < 1 / 3)
    # The linked claims with their own award still take DWP's status.
    assert person.loc[[1, 2, 3], UC_LCWRA].all()
    assert not person.loc[[4, 5, 6], UC_LCWRA].any()
    assert result.evidence()["imputation"]["cells"]["25_to_49"]["proxy_flags"][
        "no_limit_a_lot"
    ] == {
        "linked_claims": "3-9",
        "support": "3-9",
        "rate": 0.333,
        "source": "cell",
        "own_esa_award_claims": {"linked_claims": "3-9", "support": "3-9", "rate": 0.5},
    }


def test_a_cell_under_three_households_draws_at_its_groups_rate_over_every_age() -> (
    None
):
    people = [
        # Linked claims the proxy flags: young, half LCWRA; aged 25 to 49, none.
        *[_single(unit, "LCWRA", True) for unit in (1, 2, 3)],
        *[_single(unit, "NONE", True) for unit in (4, 5, 6)],
        *[_single(unit, "NONE", True) for unit in (7, 8, 9)],
        # One linked LCWRA claim aged 50 or over: its cell rests on one household.
        _single(10, "LCWRA", True),
        # Unlinked claims aged 50 or over the proxy flags.
        *[_single(unit, "NOT_LINKED", True) for unit in range(400, 430)],
    ]
    ages = [20] * 6 + [40] * 3 + [60] * 31
    unlinked = range(400, 430)
    # Over every age, four of the ten linked claims the proxy flags are LCWRA.
    # The draws reach both sides of 0.4 and fall between it and the young
    # cell's 0.5, so the test tells the pooled rate from the cell rates.
    assert any(_uniform(unit) < 0.4 for unit in unlinked)
    assert any(0.4 <= _uniform(unit) < 0.5 for unit in unlinked)
    result = assign_uc_limited_capability(
        _frame(people, ages=ages), engine_answers=_answers(people)
    )
    person = result.frame.table("person").set_index("person_benunit_id")

    for unit in unlinked:
        assert person.loc[unit, UC_LCWRA] == (_uniform(unit) < 0.4)
        assert person.loc[unit, UC_LCW] == (_uniform(unit) < 0.4)
    imputation = result.evidence()["imputation"]
    over_fifty = imputation["cells"]["50_and_over"]["proxy_flags"]["no_limit_a_lot"]
    assert over_fifty["source"] == "pooled_by_condition"
    assert over_fifty["support"] == "<3"
    assert over_fifty["rate"] == "withheld"
    # The draws take 0.4, but the receipt withholds it: beside the young cell's
    # printed 0.5 it would give back the one LCWRA household among the cells it
    # withholds, the 50-and-over claim (Vahid's review of
    # PolicyEngine/microcosm#1663).
    assert imputation["pooled"]["proxy_flags"] == {
        "linked_claims": 10,
        "support": "10-29",
        "rate": "withheld",
        "own_esa_award_claims": {"linked_claims": 0, "support": 0, "rate": "withheld"},
    }
    # Every linked claim is on the no-condition side, so its group over every
    # age gives the same rate.
    assert (
        imputation["pooled_by_condition"]["proxy_flags"]["no_limit_a_lot"]
        == (imputation["pooled"]["proxy_flags"])
    )


def test_claims_with_their_own_esa_award_keep_the_proxy() -> None:
    people = [
        *[_single(unit, "NONE", True) for unit in (1, 2, 3)],
        _single(10, "NOT_LINKED", True),
        _single(11, "NOT_LINKED", False),
    ]
    esa = [False, False, False, True, True]
    result = assign_uc_limited_capability(
        _frame(people), engine_answers=_answers(people, esa=esa)
    )

    # Without the award, the flagged claim would take the linked rate, zero.
    assert result.frame.table("person")[UC_LCWRA].tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert result.evidence()["imputation"]["kept_the_proxy"]["own_esa_award"] == "<3"


def test_without_linked_claims_every_claim_keeps_the_proxy() -> None:
    people = [_single(10, "NOT_LINKED", True), _single(11, "NOT_LINKED", False)]
    result = assign_uc_limited_capability(
        _frame(people), engine_answers=_answers(people)
    )

    assert result.frame.table("person")[UC_LCWRA].tolist() == [True, False]
    imputation = result.evidence()["imputation"]
    assert imputation["kept_the_proxy"]["no_rate"] == "<3"
    assert {
        entry["source"]
        for band in imputation["cells"].values()
        for side in band.values()
        for entry in side.values()
    } == {"proxy_kept"}


def test_imputed_lcwra_goes_to_the_first_claimant_or_a_carers_partner() -> None:
    people = [
        # Every linked claim the proxy misses is LCWRA: the rate is one.
        *[_single(unit, "LCWRA", False) for unit in (1, 2, 3)],
        # An unlinked couple nobody flags: the first claimant takes both flags.
        (10, 10, "NOT_LINKED", False, False, False, True, False),
        (10, 10, "NOT_LINKED", False, False, False, True, False),
        # The same couple, but the first claimant cares: the partner takes them.
        (11, 11, "NOT_LINKED", False, False, False, True, True),
        (11, 11, "NOT_LINKED", False, False, False, True, False),
    ]
    result = assign_uc_limited_capability(
        _frame(people), engine_answers=_answers(people)
    )

    person = result.frame.table("person")
    assert person[UC_LCWRA].tolist() == [True, True, True, True, False, False, True]
    assert person[UC_LCW].tolist() == [True, True, True, True, False, False, True]
    branches = result.evidence()["imputation"]["branches"]
    assert branches["lcwra_given_to_the_first_claimant"] == "<3"
    assert branches["lcwra_given_to_the_partner_of_a_carer"] == "<3"


def test_the_cells_split_on_a_condition_that_limits_activities_a_lot() -> None:
    a_lot = UC_LIMITED_CAPABILITY_CONDITION_CELL
    people = [
        # Young linked claims the proxy misses. With a condition that limits
        # activities a lot: two of three LCWRA.
        *[_single(unit, "LCWRA", False) for unit in (1, 2)],
        _single(3, "NONE", False),
        # Without one: none of six.
        *[_single(unit, "NONE", False) for unit in range(4, 10)],
        # Unlinked young claims the proxy misses, on either side.
        *[_single(unit, "NOT_LINKED", False) for unit in range(100, 130)],
        *[_single(unit, "NOT_LINKED", False) for unit in range(200, 230)],
    ]
    conditions = (
        [a_lot] * 3
        + ["NOT_AT_ALL", "A_LITTLE", "NO_CONDITION"] * 2
        + [a_lot] * 30
        + ["NO_CONDITION"] * 30
    )
    result = assign_uc_limited_capability(
        _frame(people, ages=[20] * len(people), conditions=conditions),
        engine_answers=_answers(people),
    )
    person = result.frame.table("person").set_index("person_benunit_id")

    # Pooled over the condition the rate would be two of nine; split, a claim
    # with the condition draws at two thirds and one without it never draws.
    assert any(2 / 9 <= _uniform(unit) < 2 / 3 for unit in range(100, 130))
    for unit in range(100, 130):
        assert person.loc[unit, UC_LCWRA] == (_uniform(unit) < 2 / 3)
    assert not person.loc[list(range(200, 230)), UC_LCWRA].any()
    cells = result.evidence()["imputation"]["cells"]["16_to_24"]["proxy_does_not"]
    assert cells["limits_a_lot"]["source"] == "cell"
    assert cells["limits_a_lot"]["rate"] == "withheld"  # one NONE household
    assert cells["no_limit_a_lot"]["source"] == "cell"
    assert cells["no_limit_a_lot"]["rate"] == "withheld"  # no LCWRA household


def test_a_cell_under_three_households_takes_its_condition_groups_rate() -> None:
    a_lot = UC_LIMITED_CAPABILITY_CONDITION_CELL
    people = [
        # Linked claims aged 25 to 49 the proxy flags, with the condition: a
        # third LCWRA; without it, all NONE.
        _single(1, "LCWRA", True),
        *[_single(unit, "NONE", True) for unit in (2, 3)],
        *[_single(unit, "NONE", True) for unit in (4, 5, 6)],
        # One young linked claim the proxy flags, with the condition: its cell
        # rests on one household.
        _single(7, "NONE", True),
        # Unlinked young claims the proxy flags, with the condition.
        *[_single(unit, "NOT_LINKED", True) for unit in range(100, 140)],
    ]
    ages = [40] * 6 + [20] + [20] * 40
    conditions = [a_lot] * 3 + ["NO_CONDITION"] * 3 + [a_lot] + [a_lot] * 40
    unlinked = range(100, 140)
    # Over every age, one of the four linked claims with the condition is
    # LCWRA; over the whole proxy group, one of seven.
    assert any(1 / 7 <= _uniform(unit) < 1 / 4 for unit in unlinked)
    result = assign_uc_limited_capability(
        _frame(people, ages=ages, conditions=conditions),
        engine_answers=_answers(people),
    )
    person = result.frame.table("person").set_index("person_benunit_id")

    for unit in unlinked:
        assert person.loc[unit, UC_LCWRA] == (_uniform(unit) < 1 / 4)
    imputation = result.evidence()["imputation"]
    young = imputation["cells"]["16_to_24"]["proxy_flags"]["limits_a_lot"]
    assert young["source"] == "pooled_by_condition"
    assert young["support"] == "<3"
    assert young["rate"] == "withheld"
    assert (
        imputation["pooled_by_condition"]["proxy_flags"]["limits_a_lot"]["support"]
        == "3-9"
    )


def test_a_group_rate_beside_a_withheld_cell_it_would_give_back_is_withheld() -> None:
    # Vahid's example (review of PolicyEngine/microcosm#1663): two cells print
    # 0.4 and a one-household cell is withheld, so a group rate above or below
    # 0.4 would say which way that household went.
    a_lot = UC_LIMITED_CAPABILITY_CONDITION_CELL
    people = [
        # Linked claims the proxy flags, with the condition, aged 25 to 49 and
        # 50 or over: four of ten LCWRA in each.
        *[_single(unit, "LCWRA", True) for unit in range(1, 5)],
        *[_single(unit, "NONE", True) for unit in range(5, 11)],
        *[_single(unit, "LCWRA", True) for unit in range(11, 15)],
        *[_single(unit, "NONE", True) for unit in range(15, 21)],
        # One young linked LCWRA claim with the condition.
        _single(21, "LCWRA", True),
    ]
    ages = [40] * 10 + [60] * 10 + [20]
    result = assign_uc_limited_capability(
        _frame(people, ages=ages, conditions=[a_lot] * len(people)),
        engine_answers=_answers(people),
    )
    imputation = result.evidence()["imputation"]
    cells = imputation["cells"]
    assert cells["25_to_49"]["proxy_flags"]["limits_a_lot"]["rate"] == 0.4
    assert cells["50_and_over"]["proxy_flags"]["limits_a_lot"]["rate"] == 0.4
    assert cells["16_to_24"]["proxy_flags"]["limits_a_lot"]["rate"] == "withheld"
    group = imputation["pooled_by_condition"]["proxy_flags"]["limits_a_lot"]
    assert group["support"] == "10-29"
    assert group["rate"] == "withheld"
    # The proxy group over every age is the same claims, so it is withheld too.
    assert imputation["pooled"]["proxy_flags"]["rate"] == "withheld"


def test_a_group_rate_prints_where_its_withheld_cells_together_pass_the_floor() -> None:
    people = [
        # Linked claims the proxy flags, without the condition. Young: two of
        # three LCWRA; aged 25 to 49: one of three. Each cell is withheld, but
        # together they hold three LCWRA households and three others.
        *[_single(unit, "LCWRA", True) for unit in (1, 2)],
        _single(3, "NONE", True),
        _single(4, "LCWRA", True),
        *[_single(unit, "NONE", True) for unit in (5, 6)],
        # Aged 50 or over: half of six.
        *[_single(unit, "LCWRA", True) for unit in (7, 8, 9)],
        *[_single(unit, "NONE", True) for unit in (10, 11, 12)],
    ]
    ages = [20] * 3 + [40] * 3 + [60] * 6
    result = assign_uc_limited_capability(
        _frame(people, ages=ages), engine_answers=_answers(people)
    )
    imputation = result.evidence()["imputation"]
    cells = imputation["cells"]
    assert cells["16_to_24"]["proxy_flags"]["no_limit_a_lot"]["rate"] == "withheld"
    assert cells["25_to_49"]["proxy_flags"]["no_limit_a_lot"]["rate"] == "withheld"
    assert cells["50_and_over"]["proxy_flags"]["no_limit_a_lot"]["rate"] == 0.5
    assert (
        imputation["pooled_by_condition"]["proxy_flags"]["no_limit_a_lot"]["rate"]
        == 0.5
    )
    assert imputation["pooled"]["proxy_flags"]["rate"] == 0.5


def test_an_own_esa_group_rate_beside_a_withheld_own_esa_cell_is_withheld() -> None:
    people = [
        # Linked claims the proxy flags with their own ESA award: aged 25 to 49,
        # half of six LCWRA; one young LCWRA claim.
        *[_single(unit, "LCWRA", True) for unit in (1, 2, 3)],
        *[_single(unit, "NONE", True) for unit in (4, 5, 6)],
        _single(7, "LCWRA", True),
        # Without an award: aged 25 to 49, half of six.
        *[_single(unit, "LCWRA", True) for unit in (8, 9, 10)],
        *[_single(unit, "NONE", True) for unit in (11, 12, 13)],
    ]
    ages = [40] * 6 + [20] + [40] * 6
    esa = [True] * 7 + [False] * 6
    result = assign_uc_limited_capability(
        _frame(people, ages=ages), engine_answers=_answers(people, esa=esa)
    )
    imputation = result.evidence()["imputation"]
    own = "own_esa_award_claims"
    assert (
        imputation["cells"]["25_to_49"]["proxy_flags"]["no_limit_a_lot"][own]["rate"]
        == 0.5
    )
    assert (
        imputation["cells"]["16_to_24"]["proxy_flags"]["no_limit_a_lot"][own]["rate"]
        == "withheld"
    )
    assert (
        imputation["pooled_by_condition"]["proxy_flags"]["no_limit_a_lot"][own]["rate"]
        == "withheld"
    )
    assert imputation["pooled"]["proxy_flags"][own]["rate"] == "withheld"
    # The claims without an award print where nothing inside them is withheld.
    assert imputation["pooled"]["proxy_flags"]["rate"] == 0.5


def test_a_couple_takes_the_condition_side_when_either_claimant_reports_it() -> None:
    # Vahid's review of PolicyEngine/microcosm#1663: the cell's condition side
    # reads every claimant, not only the first.
    a_lot = UC_LIMITED_CAPABILITY_CONDITION_CELL
    couples = []
    conditions = []
    for unit, carrier in (
        (1, "LCWRA"),
        (2, "LCWRA"),
        (3, "LCWRA"),
        (4, "NONE"),
        (5, "NONE"),
        (6, "NONE"),
    ):
        couples += [
            (unit, unit, carrier, False, False, False, True, False),
            (unit, unit, "NOT_LINKED", False, False, False, True, False),
        ]
        # Only the partner reports the condition.
        conditions += ["NO_CONDITION", a_lot]
    result = assign_uc_limited_capability(
        _frame(couples, ages=[40] * len(couples), conditions=conditions),
        engine_answers=_answers(couples),
    )
    cells = result.evidence()["imputation"]["cells"]["25_to_49"]["proxy_does_not"]
    assert cells["limits_a_lot"]["linked_claims"] == "3-9"
    assert cells["limits_a_lot"]["rate"] == 0.5
    assert cells["no_limit_a_lot"]["linked_claims"] == 0
