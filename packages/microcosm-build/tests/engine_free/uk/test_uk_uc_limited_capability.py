"""The UC limited capability stage (microcosm#1095, Y1 of the November event-data plan)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.frs_spine import FRS_UC_LCW_STATUS_COLUMN
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.uc_limited_capability import (
    UC_LCW,
    UC_LCWRA,
    UC_LIMITED_CAPABILITY_CARER,
    UC_LIMITED_CAPABILITY_CLAIMANT,
    UC_LIMITED_CAPABILITY_ENGINE_VARIABLES,
    UC_LIMITED_CAPABILITY_MASS_CHANGE_REASON,
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
    # 9. An unlinked reporter: the proxy stands.
    (7, 7, "NOT_LINKED", False, True, True, True, False),
    # 10. An SPI twin with its donor's carrier: ignored, the proxy stands.
    (8, 8, "NONE", True, True, True, True, False),
    # 11. A child in the first claim's unit: the proxy stands.
    (1, 1, "NOT_LINKED", False, True, True, False, False),
]


def _frame(people=_PEOPLE, *, spi_column: bool = True, carrier: bool = True):
    count = len(people)
    person = pd.DataFrame(
        {
            "person_id": [100 + i for i in range(count)],
            "person_benunit_id": [row[0] for row in people],
            "person_household_id": [row[1] for row in people],
        }
    )
    if carrier:
        person[FRS_UC_LCW_STATUS_COLUMN] = [row[2] for row in people]
    households = sorted({row[1] for row in people})
    spi = {row[1]: row[3] for row in people}
    household = pd.DataFrame(
        {
            "household_id": households,
            "region": ["WALES"] * len(households),
            "household_source_id": households,
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


def _answers(people=_PEOPLE) -> dict[str, np.ndarray]:
    return {
        UC_LCWRA: np.asarray([row[4] for row in people]),
        UC_LCW: np.asarray([row[5] for row in people]),
        UC_LIMITED_CAPABILITY_CLAIMANT: np.asarray([row[6] for row in people]),
        UC_LIMITED_CAPABILITY_CARER: np.asarray([row[7] for row in people]),
    }


def test_linked_claims_take_dwps_status_and_everyone_else_the_proxy() -> None:
    result = assign_uc_limited_capability(_frame(), engine_answers=_answers())
    person = result.frame.table("person")

    assert person[UC_LCWRA].tolist() == [
        False,  # NONE over the proxy
        True,  # LCWRA kept
        True,  # LCWRA given to the single claimant
        False,  # the carer keeps the carer element...
        True,  # ...and the partner takes LCWRA
        True,  # the UC row's person takes it
        False,
        False,  # LCW: nobody in the claim has WRA
        True,  # unlinked: the proxy
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
        True,
        True,
        True,
    ]
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
