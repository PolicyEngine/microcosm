# ruff: noqa: F401
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import EvidenceContext
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_support import (
    support_channel_column,
    support_clone_index_column,
)
from microcosm.build.uk_runtime.uc_capital_coherence import cohere_uc_capital
from microcosm.build.uk_runtime.uc_reporter_redraw import (
    UC_REPORTER_REDRAW_OUTPUT,
    UC_REPORTER_REDRAW_SEED,
    UC_REPORTER_SCREEN_VARIABLES,
    UKUCReporterRedrawStageTransform,
    _assert_stage_parameters,
    _benefit_unit_reporter_amounts,
    _materialize_screen_inputs,
    _positive_pre_takeup_award_screen,
    redraw_spi_reported_uc,
)
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine


def _stage():
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()["uc_reporter_redraw"]


class _StubEngine:
    country = "uk"

    def __init__(self, *, fail_all_spi: bool = False) -> None:
        self.calls: list[tuple[object, tuple[str, ...], str]] = []
        self.fail_all_spi = fail_all_spi

    def materialize(self, frame, variables, period):
        self.calls.append((frame, tuple(variables), str(period)))
        benunit = frame.table("benunit")
        person = frame.table("person")
        # Benefit unit 202 fails the hard award screen. All others pass —
        # including a child-only unit, because uc_maximum_amount is
        # mechanical and pays it.
        maximum = np.full(len(benunit), 100.0)
        reduction = np.zeros(len(benunit))
        reduction[benunit["benunit_id"].eq(202).to_numpy()] = 100.0
        if self.fail_all_spi:
            reduction[
                benunit[support_channel_column("benunit")].eq("spi").to_numpy()
            ] = 100.0
        return {
            "uc_maximum_amount": maximum,
            "uc_income_reduction": reduction,
            "is_child_or_qualifying_young_person_for_universal_credit": (
                person["is_uc_child"].to_numpy(dtype=bool)
            ),
            "is_SP_age": person["age"].to_numpy(dtype=float) >= 66.0,
        }


class _StubFittedQRF:
    def __init__(self, record: dict[str, object]) -> None:
        self.record = record

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        self.record["predict_index"] = table.index.tolist()
        draws = np.zeros(len(table), dtype=float)
        if len(draws):
            draws[0] = 250.0
        return pd.DataFrame(
            {"universal_credit_reported_amount": draws},
            index=table.index,
        )


class _StubQRF:
    def __init__(self, record: dict[str, object], *, seed: int) -> None:
        record["seed"] = seed
        self.record = record

    def fit(self, table, predictors, targets, *, weights):
        self.record["fit_table"] = table.copy()
        self.record["fit_index"] = table.index.tolist()
        self.record["fit_predictors"] = list(predictors)
        self.record["fit_targets"] = list(targets)
        self.record["fit_weights"] = np.asarray(weights).copy()
        return _StubFittedQRF(self.record)


class _StubQRFFactory:
    def __init__(self) -> None:
        self.record: dict[str, object] = {}

    def __call__(self, *, seed: int):
        return _StubQRF(self.record, seed=seed)


def _frame(*, child_only: bool = False):
    benunit_rows = [
        # Four canonical, screened FRS training benefit units. The first is a
        # couple-with-child reporter donor needed by uc_capital_coherence.
        (101, 1, "frs", 0, False, True, 1, 100.0, 2.0),
        (102, 2, "frs", 0, False, False, 0, 0.0, 3.0),
        # Both are screened but excluded from training by the exact mask.
        (103, 3, "frs", 1, False, False, 0, 70.0, 6.0),
        (104, 4, "frs", 0, True, False, 0, 60.0, 7.0),
        (105, 5, "frs", 0, False, False, 0, 80.0, 4.0),
        (106, 6, "frs", 0, False, True, 1, 0.0, 5.0),
        # SPI: 201 passes and receives a draw; 202 fails and is zeroed; 203
        # passes but the stub model returns zero.
        (201, 7, "spi", 1, False, True, 1, 10.0, 1.0),
        (202, 8, "spi", 1, False, False, 0, 50.0, 1.0),
        (203, 9, "spi", 1, False, False, 0, 0.0, 1.0),
    ]
    if child_only:
        # A 17-year-old qualifying young person heading their own SPI unit,
        # carrying a stage-2 chain fill of 300. Its only member is a child.
        benunit_rows.append((204, 10, "spi", 1, False, False, 0, 300.0, 1.0))
    people: list[dict[str, object]] = []
    for (
        benunit_id,
        household_id,
        _channel,
        _clone,
        _,
        married,
        children,
        reported,
        _,
    ) in benunit_rows:
        if married:
            ages = (45, 45)
        elif benunit_id == 204:
            ages = (17,)
        else:
            ages = (70,) if benunit_id == 203 else (40,)
        for member, age in enumerate(ages):
            person_id = benunit_id * 10 + member + 1
            # Reverse the IDs for BU 201 so lowest-id, not input order, proves
            # the tie-break. The first table row has the larger ID.
            if benunit_id == 201:
                person_id = 2012 - member
            people.append(
                {
                    "person_id": person_id,
                    "person_benunit_id": benunit_id,
                    "person_household_id": household_id,
                    "age": age,
                    "is_benunit_head": member == 0,
                    "is_parent": children > 0,
                    "is_uc_child": benunit_id == 204,
                    "employment_income": 10_000.0 + benunit_id + member,
                    "self_employment_income": 100.0 * member,
                    "savings_interest_income": 10.0,
                    "dividend_income": 20.0,
                    "property_income": 30.0,
                    "other_investment_income": 40.0,
                    UC_REPORTER_REDRAW_OUTPUT: reported if member == 0 else 0.0,
                }
            )
        if children:
            people.append(
                {
                    "person_id": benunit_id * 10 + 9,
                    "person_benunit_id": benunit_id,
                    "person_household_id": household_id,
                    "age": 10,
                    "is_benunit_head": False,
                    "is_parent": False,
                    "is_uc_child": True,
                    "employment_income": 0.0,
                    "self_employment_income": 0.0,
                    "savings_interest_income": 0.0,
                    "dividend_income": 0.0,
                    "property_income": 0.0,
                    "other_investment_income": 0.0,
                    UC_REPORTER_REDRAW_OUTPUT: 0.0,
                }
            )
    person = pd.DataFrame(people)
    benunit = pd.DataFrame(
        {
            "benunit_id": [row[0] for row in benunit_rows],
            support_channel_column("benunit"): [row[2] for row in benunit_rows],
            support_clone_index_column("benunit"): [row[3] for row in benunit_rows],
            "frs_benunit_capital": [1_000.0 + row[0] for row in benunit_rows],
            "is_married": [row[5] for row in benunit_rows],
            "dependent_children": [row[6] for row in benunit_rows],
            "would_claim_uc": False,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [row[1] for row in benunit_rows],
            "region": ["LONDON" if row[1] % 2 else "SCOTLAND" for row in benunit_rows],
            "household_is_capital_gains_clone": [row[4] for row in benunit_rows],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.asarray([row[8] for row in benunit_rows]),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def _stub_run():
    engine = _StubEngine()
    factory = _StubQRFFactory()
    result = redraw_spi_reported_uc(_frame(), engine=engine, qrf_factory=factory)
    return result, engine, factory.record


__all__ = [name for name in globals() if not name.startswith("__")]
