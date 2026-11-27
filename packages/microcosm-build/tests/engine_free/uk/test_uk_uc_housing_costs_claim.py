"""The UC housing costs element claim stage (B1 of microcosm#1710)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.frs_spine import FRS_UC_HOUSING_ELEMENT_COLUMN
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.uc_housing_costs_claim import (
    UC_HOUSING_COSTS_CLAIM,
    UC_HOUSING_COSTS_CLAIM_ENGINE_VARIABLES,
    UC_HOUSING_COSTS_CLAIM_MASS_CHANGE_REASON,
    UC_HOUSING_COSTS_CLAIM_STAGE_NAME,
    UC_HOUSING_COSTS_ELEMENT,
    UKUCHousingCostsClaimStageTransform,
    assign_uc_housing_costs_claim,
)
from microcosm.frame import WeightKind

SOCIAL, PRIVATE = "RENT_FROM_COUNCIL", "RENT_PRIVATELY"

# One benefit unit per household: its carrier on the UC row's person, whether
# the household is an SPI copy, its tenure and the engine's element.
_UNITS = [
    # 1-6. Six linked social claims the engine pays: three with no element in
    # the statement (false), three paid (true).
    (1, "NONE", False, SOCIAL, 4_000.0),
    (2, "NONE", False, SOCIAL, 4_000.0),
    (3, "NONE", False, SOCIAL, 4_000.0),
    (4, "PAID", False, SOCIAL, 4_000.0),
    (5, "PAID", False, SOCIAL, 4_000.0),
    (6, "PAID", False, SOCIAL, 4_000.0),
    # 7. A linked claim the engine pays nothing (an owner): true.
    (7, "NONE", False, "OWNED_OUTRIGHT", 0.0),
    # 8. An SPI copy carrying its donor's NONE: it keeps the default.
    (8, "NONE", True, SOCIAL, 4_000.0),
    # 9-12. Unlinked units the engine pays: they keep the default.
    (9, "NOT_LINKED", False, SOCIAL, 4_000.0),
    (10, "NOT_LINKED", False, SOCIAL, 4_000.0),
    (11, "NOT_LINKED", False, PRIVATE, 6_000.0),
    (12, "NOT_LINKED", False, PRIVATE, 6_000.0),
    # 13. An unlinked unit the engine pays nothing: it keeps the default.
    (13, "NOT_LINKED", False, PRIVATE, 0.0),
    # 14 and 19. SPI copies carrying their donors' PAID and NONE, and 15, one
    # the engine pays nothing: whatever the copied statement says, they keep
    # the default.
    (14, "PAID", True, SOCIAL, 4_000.0),
    (15, "NONE", True, SOCIAL, 0.0),
    (19, "NONE", True, SOCIAL, 4_000.0),
]


def _frame(units=_UNITS, *, carrier: bool = True, spi_column: bool = True):
    count = len(units)
    person = pd.DataFrame(
        {
            "person_id": [100 + row[0] for row in units],
            "person_benunit_id": [row[0] for row in units],
            "person_household_id": [row[0] for row in units],
        }
    )
    if carrier:
        person[FRS_UC_HOUSING_ELEMENT_COLUMN] = [row[1] for row in units]
    household = pd.DataFrame(
        {
            "household_id": [row[0] for row in units],
            "region": ["WALES"] * count,
            "tenure_type": [row[3] for row in units],
            "household_source_id": [row[0] for row in units],
            # The true source each floor counts (microcosm#931).
            "source_household_id": [row[0] for row in units],
        }
    )
    if spi_column:
        household["household_is_spi_synthetic"] = [row[2] for row in units]
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": [row[0] for row in units]}),
        household=household,
        household_weights=np.full(count, 1000.0),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def _answers(units=_UNITS) -> dict[str, np.ndarray]:
    return {UC_HOUSING_COSTS_ELEMENT: np.asarray([row[4] for row in units])}


def _flags(result) -> dict[int, bool]:
    benunit = result.frame.table("benunit")
    return dict(
        zip(benunit["benunit_id"], benunit[UC_HOUSING_COSTS_CLAIM], strict=True)
    )


def test_linked_claims_follow_the_statement_and_the_rest_keep_the_default() -> None:
    result = assign_uc_housing_costs_claim(_frame(), engine_answers=_answers())
    flags = _flags(result)

    # The statement decides linked claims the engine pays; an owner the engine
    # pays nothing keeps the default whatever the statement says.
    assert [flags[unit] for unit in (1, 2, 3, 4, 5, 6, 7)] == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    # Nothing is imputed: unlinked units and SPI copies keep the default,
    # whatever the engine pays them and whatever a copied statement says.
    for unit in (8, 9, 10, 11, 12, 13, 14, 15, 19):
        assert bool(flags[unit]) is True, unit
    receipt = result.receipt
    assert receipt["kept_the_default_over_a_positive_engine_element"] == {
        "unlinked": "3-9",
        "spi_channel": "3-9",
    }
    assert receipt["spi_channel_benefit_units"] == "3-9"
    assert "imputation" not in receipt


def test_the_receipt_bands_its_counts() -> None:
    receipt = assign_uc_housing_costs_claim(_frame(), engine_answers=_answers()).receipt
    # Counts are banded, never exact below ten.
    assert receipt["linked_claims"] == "3-9"
    assert receipt["linked"]["no_element_over_a_positive_engine_element"] == "3-9"
    assert receipt["linked"]["no_element_where_the_engine_pays_none"] == "<3"
    assert receipt["linked"]["element_paid"] == "3-9"
    # Three source households without the element: their weight prints.
    assert receipt["without_the_element"] == {
        "benefit_units": "3-9",
        "weighted": 3000.0,
        "support": "3-9",
    }


def test_the_receipt_withholds_a_weight_under_three_households() -> None:
    units = [
        (1, "NONE", False, SOCIAL, 4_000.0),
        (2, "NONE", False, SOCIAL, 4_000.0),
        (3, "PAID", False, SOCIAL, 4_000.0),
    ]
    result = assign_uc_housing_costs_claim(
        _frame(units), engine_answers=_answers(units)
    )
    assert [bool(flag) for flag in _flags(result).values()] == [False, False, True]
    without = result.receipt["without_the_element"]
    assert without["weighted"] == "withheld"
    assert without["benefit_units"] == "<3"
    assert without["support"] == "<3"


def test_weights_and_mass_pass_through() -> None:
    frame = _frame()
    result = assign_uc_housing_costs_claim(frame, engine_answers=_answers())
    assert result.frame.weights_for("household").values.tolist() == (
        frame.weights_for("household").values.tolist()
    )
    record = result.frame.mass_log[-1]
    assert record.reason == UC_HOUSING_COSTS_CLAIM_MASS_CHANGE_REASON
    assert record.old_total == record.new_total


def test_without_linked_claims_every_unit_keeps_the_default() -> None:
    units = [(unit, "NOT_LINKED", False, SOCIAL, 4_000.0) for unit in range(1, 6)]
    result = assign_uc_housing_costs_claim(
        _frame(units), engine_answers=_answers(units)
    )
    assert all(bool(flag) for flag in _flags(result).values())
    assert result.receipt["kept_the_default_over_a_positive_engine_element"] == {
        "unlinked": "3-9",
        "spi_channel": 0,
    }


def test_the_stage_refuses_what_it_cannot_read() -> None:
    with pytest.raises(KeyError, match=FRS_UC_HOUSING_ELEMENT_COLUMN):
        assign_uc_housing_costs_claim(_frame(carrier=False), engine_answers=_answers())
    with pytest.raises(KeyError, match="household_is_spi_synthetic"):
        assign_uc_housing_costs_claim(
            _frame(spi_column=False), engine_answers=_answers()
        )
    with pytest.raises(ValueError, match="shape"):
        assign_uc_housing_costs_claim(
            _frame(), engine_answers={UC_HOUSING_COSTS_ELEMENT: np.zeros(2)}
        )
    stored = _frame()
    benunit = stored.table("benunit").assign(**{UC_HOUSING_COSTS_CLAIM: True})
    with pytest.raises(ValueError, match="only writer"):
        assign_uc_housing_costs_claim(
            uk_national_frame(
                person=stored.table("person"),
                benunit=benunit,
                household=stored.table("household"),
                household_weights=stored.weights_for("household").values,
                weight_kind=WeightKind.IMPORTANCE,
                time_period="2024",
            ),
            engine_answers=_answers(),
        )
    odd = [(1, "SOMETIMES", False, SOCIAL, 4_000.0)]
    with pytest.raises(ValueError, match="outside the codes"):
        assign_uc_housing_costs_claim(_frame(odd), engine_answers=_answers(odd))


class _Engine:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def materialize(self, frame, variables, period):
        self.calls.append((tuple(variables), period))
        return {name: self.answers[name] for name in variables}


def test_the_transform_asks_the_engine_for_the_declared_variables() -> None:
    stage = load_country_spec("uk").sources.stage_map()[
        UC_HOUSING_COSTS_CLAIM_STAGE_NAME
    ]
    engine = _Engine(_answers())
    transform = UKUCHousingCostsClaimStageTransform(stage=stage, engine=engine)

    out = transform(_frame())

    assert engine.calls == [(UC_HOUSING_COSTS_CLAIM_ENGINE_VARIABLES, "2024")]
    assert transform.output_columns() == (UC_HOUSING_COSTS_CLAIM,)
    assert UC_HOUSING_COSTS_CLAIM in out.table("benunit").columns
    assert transform.checkpoint_metadata()["evidence"]["stage"] == (
        UC_HOUSING_COSTS_CLAIM_STAGE_NAME
    )


def test_the_committed_stage_binds_its_operations_closed_world() -> None:
    spec = load_country_spec("uk").sources
    stage = spec.stage_map()[UC_HOUSING_COSTS_CLAIM_STAGE_NAME]
    operations = list(stage.operations)
    assert [operation.kind for operation in operations] == [
        "materialize_rules_engine_predictors",
        "assign_uc_housing_claim_from_admin_status",
        "record_mass_conservation_receipt",
    ]
    # Nothing is drawn, so the stage carries no seed, salt or cells.
    assert not {"seed", "salt", "tenure", "imputation_rule"} & set(
        operations[1].parameters
    )
    # It runs right after the limited capability flags and before the
    # Universal Credit redraw reads entitlement.
    names = [entry.stage for entry in spec.stages]
    position = names.index(UC_HOUSING_COSTS_CLAIM_STAGE_NAME)
    assert names[position - 1 : position + 2] == [
        "uc_limited_capability",
        UC_HOUSING_COSTS_CLAIM_STAGE_NAME,
        "uc_take_up",
    ]
    drifted = replace(
        operations[1],
        parameters={**dict(operations[1].parameters), "carrier": "uc_status"},
    )
    with pytest.raises(ValueError, match="parameters drifted"):
        UKUCHousingCostsClaimStageTransform(
            stage=replace(stage, operations=(operations[0], drifted, operations[2])),
            engine=_Engine(_answers()),
        )
