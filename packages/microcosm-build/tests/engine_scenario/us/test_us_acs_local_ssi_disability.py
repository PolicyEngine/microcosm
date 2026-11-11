"""End to end: SSI for ACS persons the SSI disability-criteria stage completes.

A working-age ACS person is taken through
:func:`~microcosm.build.us_runtime.acs_local_ssi_disability.with_acs_local_ssi_disability_criteria`
with a small real QRF, and the completed inputs go through policyengine-us
(microcosm#1022, review of PR #1058). The criteria-positive person receives
SSI; an ineligible control (not aged, not blind, criteria False) does not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_local_ssi_disability import (
    ACS_LOCAL_SSI_DISABILITY_COLUMN,
    acs_local_ssi_disability_signal_gate,
    with_acs_local_ssi_disability_criteria,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_release_predictors import ACS_DIFFICULTY_TO_CPS
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.ssi_disability_criteria import (
    SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
    SIPP_SSI_DISABILITY_MODEL_PREDICTORS,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TAG = spine_column("person")
_OUTPUT = ACS_LOCAL_SSI_DISABILITY_COLUMN
_WALKING = "difficulty_walking_or_climbing_stairs"
_PERIOD = "2024"
#: The completed person inputs the engine reads for SSI income and resources.
_ENGINE_INPUTS = (
    "age",
    "is_female",
    "employment_income_before_lsr",
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "rental_income",
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
    "social_security_disability",
    "disability_benefits",
    _OUTPUT,
)
_NO_DIFFICULTY = dict.fromkeys(ACS_DIFFICULTY_TO_CPS, 2)
# Working-age ACS persons: a walking difficulty and no income (the model
# path), the ineligible control, and an under-65 SSIP reporter (the anchor).
_ACS_PEOPLE: dict[str, dict[str, Any]] = {
    "walking_difficulty": {"age": 40, "items": {**_NO_DIFFICULTY, "DPHY": 1}},
    "control": {"age": 40, "items": _NO_DIFFICULTY},
    "ssi_reporter": {"age": 30, "items": _NO_DIFFICULTY, "ssi": 9_000.0},
}


def _sipp_training_frame(n: int = 80) -> pd.DataFrame:
    """A small SIPP training frame where only a walking difficulty varies.

    Every other predictor is constant, so each tree splits on the difficulty
    alone and its leaves are pure: the criteria follow it exactly.
    """

    donor = pd.DataFrame(
        dict.fromkeys(SIPP_SSI_DISABILITY_MODEL_PREDICTORS, 0.0), index=range(n)
    )
    donor["age"] = 40.0
    donor[_WALKING] = (np.arange(n) % 2).astype(np.float64)
    donor[_OUTPUT] = donor[_WALKING].eq(1.0)
    donor["household_weight"] = 1.0
    donor.attrs["source_audit"] = {"training_rows": n, "pinned_transform": False}
    return donor


def _pooled() -> Frame:
    """Two donor persons and the three ACS persons, one household each.

    ACS criteria are missing (as staging leaves them), incomes and assets are
    zero, and the transferred amounts the stage reads are complete.
    """

    spines = [ASEC_PUF_DONOR_SPINE] * 2 + [ACS_2024_1YR_SPINE] * len(_ACS_PEOPLE)
    people = [{"age": 45}, {"age": 50}, *_ACS_PEOPLE.values()]
    n = len(people)
    ids = np.arange(1, n + 1)
    person = pd.DataFrame(
        {
            "person_id": ids,
            TAG: spines,
            "age": [float(p["age"]) for p in people],
            "is_female": [False] * n,
            "A_MARITL": [7] * n,
            "employment_income_before_lsr": [0.0] * n,
            "ssi_reported": [p.get("ssi", 0.0) for p in people],
            "SPORDER": [1] * n,
            _OUTPUT: [True, False, *([None] * len(_ACS_PEOPLE))],
        }
    )
    for entity in US_SCHEMA.entities:
        if entity != "person":
            person[f"person_{entity}_id"] = ids
    for column in _ENGINE_INPUTS[3:-1]:
        person[column] = 0.0
    for item in ACS_DIFFICULTY_TO_CPS:
        person[item] = [None, None, *(p["items"][item] for p in _ACS_PEOPLE.values())]
    household = pd.DataFrame(
        {
            "household_id": ids,
            "SERIALNO": [None, None, *(f"2024HU{i:07d}" for i in ids[2:])],
        }
    )
    return Frame(
        {
            "person": person,
            "household": household,
            **{
                entity: pd.DataFrame({f"{entity}_id": ids})
                for entity in US_SCHEMA.entities
                if entity not in {"person", "household"}
            },
        },
        US_SCHEMA,
        {"household": Weights(np.full(n, 100.0), WeightKind.DESIGN)},
    )


def _situation(acs: pd.DataFrame) -> dict[str, object]:
    """One single-person household per completed ACS person row."""

    def value(cell: Any) -> Any:
        return cell.item() if isinstance(cell, np.generic) else cell

    people: dict[str, object] = {}
    groups: dict[str, dict[str, object]] = {
        plural: {}
        for plural in (
            "tax_units",
            "families",
            "spm_units",
            "marital_units",
            "households",
        )
    }
    for name, (_, row) in zip(_ACS_PEOPLE, acs.iterrows(), strict=True):
        people[name] = {
            **{column: {_PERIOD: value(row[column])} for column in _ENGINE_INPUTS},
            "is_blind": {_PERIOD: False},
            "takes_up_ssi_if_eligible": {_PERIOD: True},
        }
        for plural, units in groups.items():
            units[f"{name}_{plural}"] = {"members": [name]}
        groups["households"][f"{name}_households"]["state_code"] = {_PERIOD: "CA"}
    return {"people": people, **groups}


def test_a_criteria_positive_acs_person_receives_ssi_and_the_control_does_not():
    from policyengine_us import Simulation

    frame = _pooled()
    completed, receipt = with_acs_local_ssi_disability_criteria(
        frame,
        sipp_donor=_sipp_training_frame(),
        seed=11,
        donor_identity={"sha256": SIPP_2023_SSI_DISABILITY_DONOR_SHA256},
        n_estimators=10,
    )

    # The stage itself makes the criteria: before it, every ACS cell is missing.
    before = frame.table("person")
    assert before.loc[before[TAG].eq(ACS_2024_1YR_SPINE), _OUTPUT].isna().all()
    person = completed.table("person")
    acs = person.loc[person[TAG].eq(ACS_2024_1YR_SPINE)].reset_index(drop=True)
    criteria = dict(zip(_ACS_PEOPLE, acs[_OUTPUT].astype(bool), strict=True))
    assert criteria == {
        "walking_difficulty": True,
        "control": False,
        "ssi_reporter": True,
    }
    assert receipt["model"]["fitted"] is True
    assert receipt["filled_rows"] == len(_ACS_PEOPLE)
    assert receipt["outcome"]["model_positive_rows"] == 1
    assert receipt["outcome"]["reporter_anchor_rows"] == 1
    assert receipt["outcome"]["filled_true_rows"] == 2
    # The donor rows keep their stored values.
    assert person.loc[person[TAG].eq(ASEC_PUF_DONOR_SPINE), _OUTPUT].tolist() == [
        True,
        False,
    ]
    assert acs_local_ssi_disability_signal_gate(completed, receipt=receipt).passed

    # The completed inputs, through policyengine-us: working age, not blind,
    # no income, no assets, SSI take-up True.
    simulation = Simulation(situation=_situation(acs))
    monthly = simulation.calculate("ssi", "2024-01")
    ssi = dict(zip(_ACS_PEOPLE, (float(amount) for amount in monthly), strict=True))
    # The 2024 federal benefit rate: $943 a month for an individual.
    assert ssi["walking_difficulty"] == pytest.approx(943.0)
    assert ssi["ssi_reporter"] == pytest.approx(943.0)
    assert ssi["control"] == 0.0
