"""Shared fixtures for the County Files household-measures and manifest tools.

``tools/us_household_measures.py`` and ``tools/us_release_schema_manifest.py``
are loaded by path (they live in ``tools/``). The engine-free tests drive the
measures tool through a fake ``Microsimulation`` whose household values are
known functions of the input tables, so batching, alignment and the checks
can be tested for every generated population without policyengine-us.

This module must never import ``policyengine_us`` or
``microcosm.build.us_runtime``: importing the latter builds the whole
policyengine-us tax-benefit system wherever the engine is installed.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from test_support.paths import paths_for

_REPOSITORY = paths_for("microcosm-build").repository

US_ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
GROUP_ENTITIES = US_ENTITIES[2:]


def load_tool(module_name: str):
    """Import ``tools/<module_name>.py``, registered in ``sys.modules`` first.

    The manifest tool imports the measures tool by name, so both register
    under their file names and load once per process.
    """

    if module_name in sys.modules:
        return sys.modules[module_name]
    path = _REPOSITORY / "tools" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def population_tables(
    household_sizes: Sequence[int],
    *,
    seed: int = 0,
    tax_unit_splits: Sequence[bool] | None = None,
    household_order: Sequence[int] | None = None,
    person_order: Sequence[int] | None = None,
    id_base: int = 100,
) -> dict[str, pd.DataFrame]:
    """Nested US entity tables: every group unit sits inside one household.

    Household ``k`` (``0 <= k < len(household_sizes)``) has id ``id_base + 7k``
    (sparse, not positions) and ``household_sizes[k]`` persons. A household
    with two or more persons splits into two tax units when
    ``tax_unit_splits[k]``; it has one SPM unit and one family; every person
    has a marital unit of their own except the first two, who share one.
    ``household_order`` and ``person_order`` permute the table rows, so a
    test can check that nothing depends on row order.
    """

    rng = np.random.default_rng(seed)
    splits = list(tax_unit_splits or [False] * len(household_sizes))
    persons, households = [], []
    person_id = 0
    for position, size in enumerate(household_sizes):
        household_id = id_base + 7 * position
        households.append(
            {
                "household_id": household_id,
                "state_fips": int(rng.choice([6, 36, 48])),
                "county_fips": "06001",
                "household_weight": float(rng.integers(1, 500)),
            }
        )
        for index in range(size):
            person_id += 1
            second_tax_unit = splits[position] and size >= 2 and index >= size // 2
            persons.append(
                {
                    "person_id": person_id,
                    "person_household_id": household_id,
                    "person_tax_unit_id": household_id * 10 + int(second_tax_unit),
                    "person_spm_unit_id": household_id * 100,
                    "person_family_id": household_id * 1000,
                    "person_marital_unit_id": household_id * 10_000
                    + (0 if index < 2 else index),
                    "age": float(rng.integers(0, 90)),
                    "employment_income_before_lsr": float(rng.integers(0, 120_000)),
                    "self_employment_income_before_lsr": float(
                        rng.integers(-20_000, 40_000)
                    ),
                    "social_security": float(rng.integers(0, 30_000)),
                    "is_disabled": bool(rng.random() < 0.1),
                }
            )
    person = pd.DataFrame(persons)
    household = pd.DataFrame(households)
    tables = {"person": person, "household": household}
    for group in GROUP_ENTITIES:
        ids = pd.unique(person[f"person_{group}_id"])
        tables[group] = pd.DataFrame({f"{group}_id": ids})
    tables["spm_unit"]["spm_unit_net_income_reported"] = np.linspace(
        0.0, 1.0, len(tables["spm_unit"])
    )
    if household_order is not None:
        tables["household"] = household.iloc[list(household_order)].reset_index(
            drop=True
        )
    if person_order is not None:
        tables["person"] = person.iloc[list(person_order)].reset_index(drop=True)
    return tables


def write_population_h5(
    tables: Mapping[str, pd.DataFrame], path: Path, *, year: int
) -> Path:
    """Write the tables as ``USSingleYearDataset.save`` does, without the engine."""

    path = Path(path)
    path.unlink(missing_ok=True)
    with pd.HDFStore(str(path), mode="w") as store:
        for entity in US_ENTITIES:
            if len(tables[entity]):
                store.put(entity, tables[entity], format="table", data_columns=True)
        store.put("_time_period", pd.Series([int(year)]), format="table")
    return path


# ---------------------------------------------------------------------------
# A fake engine with known household values
# ---------------------------------------------------------------------------


def expected_household_values(
    tables: Mapping[str, pd.DataFrame], year: int
) -> pd.DataFrame:
    """What :class:`FakeMicrosimulation` gives every household, by household id.

    Person variables are functions of the person's own columns and the year,
    summed over the household; household variables are functions of the
    household's members only. So any batch holding the whole household gives
    the same values as the whole file.
    """

    person = tables["person"]
    per_person = pd.DataFrame(
        {
            "household_id": person["person_household_id"].to_numpy(),
            **{
                name: fake_person_values(name, person, year)
                for name in FAKE_PERSON_VARIABLES
            },
        }
    )
    sums = per_person.groupby("household_id").sum()
    household_ids = tables["household"]["household_id"].to_numpy()
    sums = sums.reindex(household_ids, fill_value=0.0)
    out = pd.DataFrame(index=pd.Index(household_ids, name="household_id"))
    for name in FAKE_PERSON_VARIABLES:
        out[name] = sums[name].to_numpy(dtype=np.float64)
    market = (
        out["employment_income"]
        + out["self_employment_income"]
        + 0.5 * out["farm_operations_income"]
    )
    benefits = sums["social_security"].to_numpy(dtype=np.float64)
    credits = 0.01 * market.clip(lower=0.0)
    tax = 0.2 * market.clip(lower=0.0)
    out["household_market_income"] = market
    out["household_benefits"] = benefits
    out["household_refundable_tax_credits"] = credits
    out["household_tax_before_refundable_credits"] = tax
    out["household_health_costs"] = 0.0
    out["household_net_income"] = (market + benefits + credits) - (tax + 0.0)
    return out


FAKE_PERSON_VARIABLES = (
    "employment_income",
    "self_employment_income",
    "sstb_self_employment_income",
    "farm_operations_income",
    "partnership_self_employment_net_earnings",
    "social_security",
)


def fake_person_values(name: str, person: pd.DataFrame, year: int) -> np.ndarray:
    uprate = 1.0 + (int(year) - 2024) * 0.03
    wages = person["employment_income_before_lsr"].to_numpy(dtype=np.float64)
    self_employment = person["self_employment_income_before_lsr"].to_numpy(
        dtype=np.float64
    )
    ids = person["person_id"].to_numpy(dtype=np.float64)
    return {
        "employment_income": wages * uprate,
        "self_employment_income": self_employment * uprate,
        "sstb_self_employment_income": (ids % 3) * 101.25,
        "farm_operations_income": (ids % 5 - 2) * 33.5,
        "partnership_self_employment_net_earnings": (ids % 2) * 7.75,
        "social_security": person["social_security"].to_numpy(dtype=np.float64),
    }[name]


class _Holder:
    def __init__(self, periods: set[str]) -> None:
        self._periods = periods

    def get_known_periods(self) -> list[str]:
        return sorted(self._periods)


class FakeDataset:
    """Stands in for ``USSingleYearDataset(**tables, time_period=year)``."""

    def __init__(self, *, time_period: int, **tables: pd.DataFrame) -> None:
        assert set(tables) == set(US_ENTITIES)
        self.tables = tables
        self.time_period = str(time_period)


def fake_system(
    *,
    net_income_adds: Sequence[str] | None = None,
    entities: Mapping[str, str] | None = None,
    drop: Sequence[str] = (),
) -> SimpleNamespace:
    """A tax-benefit system stand-in exposing what the definition check reads."""

    tool = load_tool("us_household_measures")
    entity_of = {name: "person" for name in tool.EARNINGS_VARIABLES}
    entity_of.update(
        {
            name: "household"
            for name in (
                *tool.COMPONENT_VARIABLES,
                tool.NET_INCOME_VARIABLE,
                "household_id",
                *tool.POPULATION_AGGREGATE_VARIABLES,
            )
        }
    )
    entity_of.update(entities or {})
    variables = {}
    for name, entity in entity_of.items():
        if name in drop:
            continue
        variables[name] = SimpleNamespace(entity=SimpleNamespace(key=entity))
    if tool.NET_INCOME_VARIABLE in variables:
        net = variables[tool.NET_INCOME_VARIABLE]
        definition = tool.ENGINE_DEFINITIONS[tool.NET_INCOME_VARIABLE]
        net.adds = list(net_income_adds or definition["adds"])
        net.subtracts = list(definition["subtracts"])
    if "household_market_income" in variables:
        variables["household_market_income"].adds = tool.MARKET_INCOME_PARAMETER

    def parameters(instant: str) -> SimpleNamespace:
        return SimpleNamespace(
            gov=SimpleNamespace(
                household=SimpleNamespace(
                    market_income_sources=["employment_income", "dividend_income"],
                    household_benefits=["social_security", "snap"],
                ),
                simulation=SimpleNamespace(include_health_benefits_in_net_income=False),
                hud=SimpleNamespace(abolition=False),
            )
        )

    return SimpleNamespace(variables=variables, parameters=parameters)


class FakeMicrosimulation:
    """Stands in for ``policyengine_us.Microsimulation`` over one batch.

    ``calculate(name, year, map_to="household")`` returns the household
    values of :func:`expected_household_values` in the batch's household-table
    order, reversed when ``reverse_households`` is set (an engine need not
    keep the input order; the tool realigns by id). ``aggregate_on`` names
    population-aggregate variables the engine pretends it computed while
    scoring ``household_net_income``.
    """

    instances: list[FakeMicrosimulation] = []
    reverse_households = False
    aggregate_on: tuple[str, ...] = ()
    drop_household: int | None = None

    def __init__(self, *, dataset: FakeDataset, spm: Mapping[str, object]) -> None:
        self.dataset = dataset
        self.spm = dict(spm)
        self.year = int(dataset.time_period)
        self.tax_benefit_system = fake_system()
        self._known: dict[str, set[str]] = {}
        expected = expected_household_values(dataset.tables, self.year)
        if self.reverse_households:
            expected = expected.iloc[::-1]
        if self.drop_household is not None:
            expected = expected.drop(index=self.drop_household, errors="ignore")
        self._values = expected
        type(self).instances.append(self)

    def get_holder(self, name: str) -> _Holder:
        return _Holder(self._known.get(name, set()))

    def calculate(self, name: str, period: int, map_to: str | None = None):
        assert map_to == "household"
        assert int(period) == self.year
        if name == "household_id":
            return self._values.index.to_numpy()
        if name == "household_net_income":
            for aggregate in self.aggregate_on:
                self._known.setdefault(aggregate, set()).add(str(period))
        return self._values[name].to_numpy()


def fake_microsimulation_class(**attributes) -> type[FakeMicrosimulation]:
    """A fresh :class:`FakeMicrosimulation` subclass with its own settings."""

    return type(
        "ConfiguredFakeMicrosimulation",
        (FakeMicrosimulation,),
        {"instances": [], **attributes},
    )


def fake_calculator(tool, *, multi_batch: bool, **attributes):
    """The tool's engine seam over a configured fake engine."""

    cls = fake_microsimulation_class(**attributes)
    run = tool.EngineRun()
    calculate = tool.engine_batch_calculator(
        microsimulation_cls=cls,
        dataset_cls=FakeDataset,
        multi_batch=multi_batch,
        run=run,
    )
    return calculate, cls, run
