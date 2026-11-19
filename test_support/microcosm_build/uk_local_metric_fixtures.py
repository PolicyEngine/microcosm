"""The measure resolver over invented engine values, for the UK local metrics.

The local metrics are their declarations read through ``UKMeasureResolver``
(microcosm#1334). Tests drive the real resolver over a stub of the engine
surface it reads (variable metadata, ``calculate`` with ``map_to``, and
``map_result``) and a frame carrying the stored columns its frame routes read:
the FRS benefit-unit roles of the UC child count, and the stored home values
and tenure of the high-value homes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime import measure_simulation
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver

#: Each engine variable the local declarations read, with its entity and the
#: value type policyengine-uk declares for it.
ENGINE_VARIABLES: Mapping[str, tuple[str, type]] = {
    "self_employment_income": ("person", float),
    "loss_relief": ("person", float),
    # Declared for the loss-relief measures, which know every component.
    "total_income": ("person", float),
    "employment_income": ("person", float),
    # The company car and car fuel benefits HMRC's employment income counts
    # (PolicyEngine/uk-autumn-budget-dashboard-2026#316).
    "benefits_in_kind": ("person", float),
    # Private and State Pension together (microcosm#1418).
    "total_pension_income": ("person", float),
    "income_tax": ("person", float),
    # The HMRC rows' taxpayer test reads the tax before the Winter Fuel
    # Payment charge (microcosm#1095).
    "income_tax_before_winter_fuel_payment_charge": ("person", float),
    "tax_band": ("person", str),
    "age": ("person", float),
    "is_child_or_qualifying_young_person_for_universal_credit": ("person", bool),
    "universal_credit": ("benunit", float),
    "benunit_rent": ("benunit", float),
    "equiv_hbai_household_net_income": ("household", float),
    "equiv_hbai_household_net_income_ahc": ("household", float),
    "tenure_type": ("household", str),
    "council_tax_band": ("household", str),
    # The single adult discount cells (microcosm#1095).
    "country": ("household", str),
    "council_tax_discount_rate": ("household", float),
    "electricity_consumption": ("household", float),
    "gas_consumption": ("household", float),
    "region": ("household", str),
}


class StubResult:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)


class StubUKEngine:
    """The engine surface the measure resolver reads, over fixed values.

    ``values`` holds each variable at its own entity; ``calculate`` records
    the periods each variable is read at.
    """

    def __init__(
        self,
        values: dict[str, Sequence[Any]],
        *,
        person_benunit: Sequence[int],
        person_household: Sequence[int],
        benunit_household: Sequence[int],
    ) -> None:
        self.values = values
        self.person_benunit = np.asarray(person_benunit)
        self.person_household = np.asarray(person_household)
        self.benunit_household = np.asarray(benunit_household)
        self.households = int(self.benunit_household.max()) + 1
        self.periods: dict[str, set[Any]] = {}
        self.tax_benefit_system = SimpleNamespace(
            variables={
                name: SimpleNamespace(
                    entity=SimpleNamespace(key=entity), value_type=value_type
                )
                for name, (entity, value_type) in ENGINE_VARIABLES.items()
            }
        )

    def calculate(self, variable: str, period: Any = None, map_to: Any = None):
        self.periods.setdefault(variable, set()).add(period)
        values = np.asarray(self.values[variable])
        native = ENGINE_VARIABLES[variable][0]
        if map_to is None or map_to == native:
            return StubResult(values)
        return StubResult(self.map_result(values, native, map_to))

    def map_result(self, values: Any, from_entity: str, to_entity: str) -> np.ndarray:
        links = {
            ("person", "benunit"): (self.person_benunit, len(self.benunit_household)),
            ("person", "household"): (self.person_household, self.households),
            ("benunit", "household"): (self.benunit_household, self.households),
        }
        index, size = links[(from_entity, to_entity)]
        return np.bincount(
            index, weights=np.asarray(values, dtype=float), minlength=size
        )


class StubFrame:
    def __init__(self, tables: dict[str, pd.DataFrame], *, time_period: str) -> None:
        self._tables = tables
        self.metadata = {"time_period": time_period}

    def table(self, entity: str) -> pd.DataFrame:
        return self._tables[entity]


def local_metric_resolver(
    engine: StubUKEngine,
    *,
    household_ids: Sequence[Any],
    is_benunit_head: Sequence[bool],
    is_parent: Sequence[bool],
    dependent_children: Sequence[int],
    main_residence_value: Sequence[float],
    scratch_dir: Path,
    monkeypatch: Any,
    time_period: str = "2024",
    year: int = 2025,
) -> UKMeasureResolver:
    """The real resolver over ``engine`` and a frame of its rows.

    The frame stores the engine's ages and tenure, the FRS benefit-unit roles
    and the home values; at the frame's own period it also stores the
    self-employment profit and the losses brought forward, which the
    resolver then reads from it. It carries no annual income descriptors, so
    the annual measures the local ``hmrc`` rows read are its current values
    (the engine test of the annual measures covers the annual route).
    """

    monkeypatch.setattr(
        measure_simulation,
        "_policyengine_uk_module",
        lambda: SimpleNamespace(__version__="test"),
    )
    people = len(engine.person_benunit)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, people + 1),
            "person_benunit_id": engine.person_benunit + 1,
            "person_household_id": np.asarray(household_ids)[engine.person_household],
            "age": np.asarray(engine.values["age"], dtype=float),
            "is_benunit_head": np.asarray(is_benunit_head, dtype=bool),
            "is_parent": np.asarray(is_parent, dtype=bool),
        }
    )
    if time_period == str(year):
        for column in ("self_employment_income", "loss_relief"):
            person[column] = np.asarray(engine.values[column], dtype=float)
    benunit = pd.DataFrame(
        {
            "benunit_id": np.arange(1, len(engine.benunit_household) + 1),
            "dependent_children": np.asarray(dependent_children),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray(household_ids),
            "main_residence_value": np.asarray(main_residence_value, dtype=float),
            "tenure_type": np.asarray(engine.values["tenure_type"]),
        }
    )
    resolver = UKMeasureResolver(
        simulation_source=scratch_dir / "engine.h5",
        scratch_dir=scratch_dir,
        year=year,
        frame=StubFrame(
            {"person": person, "benunit": benunit, "household": household},
            time_period=time_period,
        ),
        microsimulation_factory=lambda **_: engine,
    )
    resolver.use_current_income_for_annual_measures(
        reason="a stub frame without annual income descriptors"
    )
    return resolver


def three_household_engine(**values: Sequence[Any]) -> StubUKEngine:
    """Three households of one benefit unit each, the local metric tests' rows.

    A child and an adult; a pensioner; a lone parent and a 12-year-old the
    FRS records as a dependant. ``values`` replaces any variable's values.
    """

    return StubUKEngine(
        {
            "self_employment_income": [0.0, 100.0, 0.0, 50.0, 0.0],
            # Losses brought forward the profit is counted after (microcosm#1095).
            "loss_relief": [0.0, 0.0, 0.0, 0.0, 0.0],
            "employment_income": [10.0, 20.0, 0.0, 30.0, 0.0],
            # Nobody has a company car.
            "benefits_in_kind": [0.0, 0.0, 0.0, 0.0, 0.0],
            # The pensioner's private and State Pension (microcosm#1418).
            "total_pension_income": [0.0, 0.0, 90.0, 0.0, 0.0],
            "income_tax": [1.0, 0.0, 2.0, 3.0, 0.0],
            "income_tax_before_winter_fuel_payment_charge": [1.0, 0.0, 2.0, 3.0, 0.0],
            "tax_band": ["HIGHER", "HIGHER", "BASIC", "HIGHER", "NONE"],
            "age": [5, 35, 72, 12, 45],
            "is_child_or_qualifying_young_person_for_universal_credit": [
                True,
                False,
                False,
                True,
                False,
            ],
            "universal_credit": [0.0, 100.0, 50.0],
            "benunit_rent": [0.0, 10.0, 20.0],
            "equiv_hbai_household_net_income": [100.0, 200.0, 300.0],
            "equiv_hbai_household_net_income_ahc": [80.0, 150.0, 260.0],
            "tenure_type": ["OWNED_OUTRIGHT", "RENT_PRIVATELY", "RENT_FROM_COUNCIL"],
            "council_tax_band": ["A", "H", "D"],
            # microcosm#1095: each household has one adult, so each takes the
            # single adult discount; only the first is in England.
            "country": ["ENGLAND", "WALES", "SCOTLAND"],
            "council_tax_discount_rate": [0.25, 0.25, 0.25],
            # microcosm#1207: the lcfs stage's stored energy spend; the second
            # household pays for no gas.
            "electricity_consumption": [900.0, 1100.0, 700.0],
            "gas_consumption": [800.0, 0.0, 950.0],
            "region": ["LONDON", "WALES", "SCOTLAND"],
            **values,
        },
        person_benunit=[0, 0, 1, 2, 2],
        person_household=[0, 0, 1, 2, 2],
        benunit_household=[0, 1, 2],
    )


def three_household_resolver(
    scratch_dir: Path,
    monkeypatch: Any,
    *,
    engine: StubUKEngine | None = None,
    time_period: str = "2024",
    year: int = 2025,
) -> UKMeasureResolver:
    """The resolver over :func:`three_household_engine`'s rows."""

    return local_metric_resolver(
        three_household_engine() if engine is None else engine,
        household_ids=[101, 102, 103],
        is_benunit_head=[False, True, True, False, True],
        is_parent=[False, True, False, False, True],
        dependent_children=[1, 0, 1],
        # microcosm#1240: stored home values at the dataset period; only the
        # owner-occupied first household clears the deflated thresholds.
        main_residence_value=[2_500_000.0, 1_800_000.0, 0.0],
        scratch_dir=scratch_dir,
        monkeypatch=monkeypatch,
        time_period=time_period,
        year=year,
    )
