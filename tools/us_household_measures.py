#!/usr/bin/env python3
"""Materialize household earnings, gross income and net income beside a US H5.

The County Files (microcosm#922) promise three household measures that a
reader can use without running policyengine-us: earnings, gross income and
net income, keyed by ``household_id``. The export gate refuses formula-owned
columns in the entity tables (``PolicyEngineUSEngine.write_dataset``, "Export
contract violated"), because a stored formula output becomes an input and
masks reforms. So the measures ship as a **sidecar** in the release directory,
never inside the H5:

- ``<h5 stem>.household_measures.csv.gz``: one row per household, the columns
  of :data:`SIDECAR_COLUMNS`;
- ``<h5 stem>.household_measures.json``: the receipt that binds it to the H5's
  sha256 and year, records the engine and its resolved definitions, and the
  coverage and identity checks.

Why a sidecar and not a non-input key inside the H5. policyengine-us 2.2.1
would ignore an extra key (``USSingleYearDataset`` reads only the six entity
tables and ``_time_period``), so both keep the file runnable. The sidecar is
chosen because it leaves the certified H5 byte for byte unchanged, so every
sha256 a release manifest or receipt records for it stays true. It needs no
in-place HDF5 write to a multi-GB release file. It cannot collide with a
loader that enumerates the H5's keys. And a gzip CSV reads with
``pandas.read_csv`` and no optional dependency (no PyTables, no pyarrow), as
well as from R, Stata or a spreadsheet.

Definitions (each a policyengine-us 2.2.1 variable, computed with
``Microsimulation`` at the H5's own ``_time_period``):

- **earnings** (labor income): the household sum of ``employment_income``
  (wages and salaries) and the self-employment sources policyengine-us's
  ``taxable_self_employment_income`` gathers under 26 USC 1402(a), before
  its employer-share deduction: ``self_employment_income``,
  ``sstb_self_employment_income``, ``farm_operations_income`` and
  ``partnership_self_employment_net_earnings``. A loss makes it negative.
- **gross_income** (pre-tax market and benefit income):
  ``household_market_income + household_benefits``.
- **net_income** (after taxes and transfers): ``household_net_income``, which
  the engine defines as ``household_market_income + household_benefits +
  household_refundable_tax_credits - household_tax_before_refundable_credits
  - household_health_costs``.

The sidecar stores those five components too, so a reader can check
``net_income`` against them; the tool checks that identity itself. It refuses
to run when the installed engine defines ``household_net_income``'s adds and
subtracts, or ``household_market_income``'s source, differently from what
this module documents, and it records the benefit and market-income lists
and the net-income switches the engine resolved for the year.

policyengine-core stores float variables as float32, so every amount carries
about seven significant digits, and the engine's own float32 sum behind
``household_net_income`` can differ from the components' float64 sum by a
few float32 roundings of their magnitude; the identity check allows for that.

Scoring runs in batches of whole households, each a fresh
``Microsimulation`` with the release's SPM selection
(``reform_validation.US_RELEASE_SPM_SELECTION``). Every measure is a
per-household quantity, and policyengine-core forbids formula-time
randomness, so a batch gives each household the value a whole-file engine
would when all references stay within households. A multi-batch run refuses
positive ``medicaid_claiming_tax_unit_id`` references outside the person's
own household, including unresolved references. It also refuses once any
batch engine or its branches computes a whole-population formula
(``POPULATION_AGGREGATE_VARIABLES``, the deciles and the Medicaid state sums;
microcosm#956).

Usage::

    uv run python tools/us_household_measures.py compute --h5 PATH \\
        [--output-dir DIR] [--expect-year 2026] [--batch-households 5000]
    uv run python tools/us_household_measures.py verify --h5 PATH \\
        [--sidecar PATH] [--receipt PATH]

``verify`` needs pandas and PyTables only: it re-checks coverage, the year,
the identities and both digests against the receipt.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import io
import json
import os
import platform
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "BENEFIT_PARAMETER",
    "COMPONENT_VARIABLES",
    "EARNINGS_VARIABLES",
    "ENGINE_DEFINITIONS",
    "MARKET_INCOME_PARAMETER",
    "MEASURE_DEFINITIONS",
    "POPULATION_AGGREGATE_VARIABLES",
    "SIDECAR_COLUMNS",
    "SIDECAR_KIND",
    "SIDECAR_SCHEMA_VERSION",
    "US_ENTITIES",
    "US_RELEASE_SPM_SELECTION",
    "HouseholdMeasuresError",
    "assemble_household_measures",
    "check_household_measures",
    "compute_household_measures",
    "engine_batch_calculator",
    "household_batches",
    "main",
    "read_h5_tables",
    "read_sidecar",
    "sidecar_paths",
    "verify_sidecar",
    "write_sidecar",
]

SIDECAR_SCHEMA_VERSION = 1
SIDECAR_KIND = "microcosm_us_household_measures"
SIDECAR_SUFFIX = ".household_measures.csv.gz"
RECEIPT_SUFFIX = ".household_measures.json"

US_ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
GROUP_ENTITIES = US_ENTITIES[1:]
TIME_PERIOD_KEY = "_time_period"
HOUSEHOLD_ID = "household_id"
YEAR = "year"

#: Wages plus the self-employment sources of 26 USC 1402(a), as
#: policyengine-us's ``taxable_self_employment_income`` lists them. All are
#: person variables; each is summed to the household.
EARNINGS_VARIABLES = (
    "employment_income",
    "self_employment_income",
    "sstb_self_employment_income",
    "farm_operations_income",
    "partnership_self_employment_net_earnings",
)
#: Household variables stored beside the measures, in sidecar order.
COMPONENT_VARIABLES = (
    "household_market_income",
    "household_benefits",
    "household_refundable_tax_credits",
    "household_tax_before_refundable_credits",
    "household_health_costs",
)
NET_INCOME_VARIABLE = "household_net_income"
MEASURES = ("earnings", "gross_income", "net_income")

#: The sidecar's columns, in order, with the dtype pandas reads them as.
SIDECAR_COLUMNS: tuple[tuple[str, str], ...] = (
    (HOUSEHOLD_ID, "int64"),
    (YEAR, "int64"),
    *((name, "float64") for name in MEASURES),
    *((name, "float64") for name in COMPONENT_VARIABLES),
)

#: How the installed engine must define the variables this module composes;
#: anything else refuses before a batch runs.
ENGINE_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    NET_INCOME_VARIABLE: {
        "entity": "household",
        "adds": (
            "household_market_income",
            "household_benefits",
            "household_refundable_tax_credits",
        ),
        "subtracts": (
            "household_tax_before_refundable_credits",
            "household_health_costs",
        ),
    },
    "household_market_income": {
        "entity": "household",
        "adds": "gov.household.market_income_sources",
        "subtracts": None,
    },
}
MARKET_INCOME_PARAMETER = "gov.household.market_income_sources"
BENEFIT_PARAMETER = "gov.household.household_benefits"
HEALTH_BENEFITS_PARAMETER = "gov.simulation.include_health_benefits_in_net_income"
#: household_benefits drops housing_assistance when this reform switch is on.
HUD_ABOLITION_PARAMETER = "gov.hud.abolition"
#: household_head_start_benefits is zero unless this switch is on.
HEAD_START_PARAMETER = "gov.simulation.include_head_start_benefits_in_net_income"

#: Human-readable definitions, written into the receipt and the schema doc.
MEASURE_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "earnings": {
        "concept": "labor income",
        "formula": " + ".join(EARNINGS_VARIABLES),
        "engine_variables": list(EARNINGS_VARIABLES),
        "entity": "person, summed over the household's members",
        "note": (
            "Wages and salaries plus self-employment income: the sources "
            "policyengine-us's taxable_self_employment_income gathers under "
            "26 USC 1402(a), before its employer-share deduction. Negative "
            "when self-employment losses exceed wages."
        ),
    },
    "gross_income": {
        "concept": "pre-tax market and benefit income",
        "formula": "household_market_income + household_benefits",
        "engine_variables": ["household_market_income", "household_benefits"],
        "entity": "household",
        "note": (
            "Market income sums the sources the parameter "
            f"{MARKET_INCOME_PARAMETER} lists; benefits sum the programs "
            f"{BENEFIT_PARAMETER} lists. Refundable tax credits are not part "
            "of it; they enter net income."
        ),
    },
    "net_income": {
        "concept": "income after taxes and transfers",
        "formula": NET_INCOME_VARIABLE,
        "engine_variables": [NET_INCOME_VARIABLE],
        "entity": "household",
        "note": (
            "household_market_income + household_benefits + "
            "household_refundable_tax_credits - "
            "household_tax_before_refundable_credits - household_health_costs."
        ),
    },
}

#: Copied from ``microcosm.build.us_runtime.reform_validation`` (a test holds
#: them equal): importing that package builds the policyengine-us system.
US_RELEASE_SPM_SELECTION: Mapping[str, object] = {"geography_kind": "county"}
#: Copied from ``microcosm.data.compatibility_recertification`` (a test holds
#: them equal); the fiscal-refresh builder aliases this list.
POPULATION_AGGREGATE_VARIABLES = (
    "household_income_decile",
    "medicaid_slcsp_state_average_cost_index",
    "medicaid_slcsp_state_denominator",
    "spm_unit_income_decile",
)

DEFAULT_BATCH_HOUSEHOLDS = 5_000
#: net_income against its components: |difference| <= ATOL + RTOL * scale.
#: ``scale`` is the sum of the components' absolute values. The engine adds
#: them in float32: four roundings of at most 6e-8 each give a bound of
#: about 2.4e-7 * scale. The tolerance allows 1e-6 * scale plus one cent.
IDENTITY_ATOL = 0.01
IDENTITY_RTOL = 1e-6


class HouseholdMeasuresError(ValueError):
    """A sidecar, its source H5 or the engine breaks this module's contract."""


def _household_ids(values: Any) -> np.ndarray:
    """IDs must convert to int64 without losing their value."""

    ids = np.asarray(values)
    if ids.dtype.kind in "iu":
        valid = (ids >= np.iinfo(np.int64).min) & (ids <= np.iinfo(np.int64).max)
    elif ids.dtype.kind == "f":
        valid = (
            np.isfinite(ids)
            & (ids == np.floor(ids))
            & (ids >= -(2.0**63))
            & (ids < 2.0**63)
        )
    else:
        valid = np.zeros(ids.shape, dtype=bool)
    if not valid.all():
        raise HouseholdMeasuresError("household_id values must be integral int64 IDs.")
    return ids.astype(np.int64)


def _validate_batch_households(batch_households: int) -> None:
    if type(batch_households) is not int or batch_households <= 0:
        raise HouseholdMeasuresError("batch_households must be a positive integer.")


# ---------------------------------------------------------------------------
# Reading the H5 (pandas and PyTables only)
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_time_period(store: pd.HDFStore, path: Path) -> int:
    if f"/{TIME_PERIOD_KEY}" not in store.keys():
        raise HouseholdMeasuresError(
            f"{path} has no {TIME_PERIOD_KEY} key, so it states no year to "
            "compute the measures at."
        )
    values = np.asarray(store[TIME_PERIOD_KEY])
    if values.shape != (1,):
        raise HouseholdMeasuresError(
            f"{path}'s {TIME_PERIOD_KEY} must hold one value; got {values.shape}."
        )
    year = int(values[0])
    if year != values[0]:
        raise HouseholdMeasuresError(
            f"{path}'s {TIME_PERIOD_KEY} {values[0]!r} is not a whole year."
        )
    return year


def read_h5_household_ids(path: Path) -> tuple[np.ndarray, int]:
    """The household table's ``household_id`` column and the file's year."""

    with pd.HDFStore(str(path), mode="r") as store:
        year = read_time_period(store, path)
        if "/household" not in store.keys():
            raise HouseholdMeasuresError(f"{path} has no household table.")
        ids = store.select("household", columns=None)[HOUSEHOLD_ID]
    return _household_ids(ids), year


def read_h5_tables(path: Path) -> tuple[dict[str, pd.DataFrame], int]:
    """Every entity table, read as ``USSingleYearDataset`` reads it."""

    with pd.HDFStore(str(path), mode="r") as store:
        year = read_time_period(store, path)
        keys = {key.strip("/") for key in store.keys()}
        missing = [entity for entity in US_ENTITIES if entity not in keys]
        if missing:
            raise HouseholdMeasuresError(f"{path} lacks entity table(s) {missing}.")
        tables = {
            entity: store[entity].reset_index(drop=True) for entity in US_ENTITIES
        }
    return tables, year


# ---------------------------------------------------------------------------
# Whole-household batches
# ---------------------------------------------------------------------------


def _unique_index(ids: np.ndarray, label: str) -> pd.Index:
    index = pd.Index(np.asarray(ids))
    if not index.is_unique:
        repeated = index[index.duplicated()][0]
        raise HouseholdMeasuresError(f"{label} {repeated} appears more than once.")
    return index


def _positions(index: pd.Index, values: np.ndarray, message: str) -> np.ndarray:
    """Each value's position in ``index``; ``message`` names the first absent one."""

    positions = index.get_indexer(np.asarray(values))
    absent = positions < 0
    if absent.any():
        raise HouseholdMeasuresError(message.format(np.asarray(values)[absent][0]))
    return positions.astype(np.int64)


def household_batches(
    tables: Mapping[str, pd.DataFrame], batch_households: int
) -> Iterator[dict[str, pd.DataFrame]]:
    """Split the tables into batches of whole households, in household order.

    Batch ``k`` holds households ``k * batch_households`` up to the next
    multiple (household-table order), every person whose household is among
    them and every group unit those persons belong to. Every group unit must
    nest in one household and every person and group row must be reachable,
    or this refuses before yielding anything: a batch must hold the whole
    units the engine sums over.
    """

    _validate_batch_households(batch_households)
    household = tables["household"]
    person = tables["person"]
    household_ids = _household_ids(household[HOUSEHOLD_ID])
    household_index = _unique_index(household_ids, "household_id")
    person_house_pos = _positions(
        household_index,
        person["person_household_id"].to_numpy(),
        "person_household_id {} names no household row.",
    )
    memberless = np.ones(len(household_ids), dtype=bool)
    memberless[person_house_pos] = False
    if memberless.any():
        # policyengine-core maps persons to groups by sorted id position, so
        # a group row no person names shifts every later group's members.
        raise HouseholdMeasuresError(
            f"{int(memberless.sum())} household(s) have no member person "
            f"(first: household_id {household_ids[memberless][0]}); the engine "
            "would attach later households' members to the wrong rows."
        )
    household_batch = np.arange(len(household_ids), dtype=np.int64) // batch_households
    person_batch = household_batch[person_house_pos]
    n_batches = int(household_batch[-1]) + 1 if len(household_ids) else 0
    if n_batches > 1:
        _assert_medicaid_claiming_tax_units_local(person)

    row_batches: dict[str, np.ndarray] = {
        "household": household_batch,
        "person": person_batch,
    }
    for group in GROUP_ENTITIES[1:]:
        frame = tables[group]
        id_column = f"{group}_id"
        membership = person[f"person_{group}_id"].to_numpy()
        # Each group id's households, from its member persons.
        pairs = pd.DataFrame({"group": membership, "household": person_house_pos})
        spans = pairs.groupby("group", sort=False)["household"].nunique()
        spanning = spans[spans > 1]
        if len(spanning):
            raise HouseholdMeasuresError(
                f"{len(spanning)} {group} unit(s) span more than one household "
                f"(first: {group}_id {spanning.index[0]}); a household batch "
                "cannot hold them whole."
            )
        group_household = pairs.drop_duplicates("group").set_index("group")["household"]
        group_ids = frame[id_column].to_numpy()
        _unique_index(group_ids, id_column)
        located = group_household.reindex(group_ids)
        if located.isna().any():
            orphan = group_ids[located.isna().to_numpy()][0]
            raise HouseholdMeasuresError(
                f"{group}_id {orphan} has no member person, so no household "
                "batch can hold it."
            )
        unreferenced = set(group_household.index.tolist()) - set(group_ids.tolist())
        if unreferenced:
            raise HouseholdMeasuresError(
                f"person_{group}_id {sorted(unreferenced)[0]} names no {group} row."
            )
        row_batches[group] = household_batch[located.to_numpy(dtype=np.int64)]

    order = {
        entity: np.argsort(row_batches[entity], kind="stable") for entity in US_ENTITIES
    }
    bounds = {
        entity: np.searchsorted(
            row_batches[entity][order[entity]], np.arange(n_batches + 1)
        )
        for entity in US_ENTITIES
    }
    for index in range(n_batches):
        yield {
            entity: tables[entity]
            .iloc[order[entity][bounds[entity][index] : bounds[entity][index + 1]]]
            .reset_index(drop=True)
            for entity in US_ENTITIES
        }


def _assert_medicaid_claiming_tax_units_local(person: pd.DataFrame) -> None:
    """Keep positive Medicaid claiming references within their household.

    As in the fiscal-refresh builder, unresolved positive IDs refuse too:
    their household might have been omitted by a caller slicing the input.
    """

    column = "medicaid_claiming_tax_unit_id"
    if column not in person:
        return
    claims = person.loc[person[column] > 0]
    if claims.empty:
        return
    local_units = pd.MultiIndex.from_frame(
        person[["person_household_id", "person_tax_unit_id"]]
    )
    claimed_units = pd.MultiIndex.from_frame(claims[["person_household_id", column]])
    if not claimed_units.isin(local_units).all():
        raise HouseholdMeasuresError(
            "A positive medicaid_claiming_tax_unit_id does not resolve within "
            "the person's own household; batched measures would differ from "
            "a whole-file simulation. Rerun with --batch-households at least "
            "the household count."
        )


# ---------------------------------------------------------------------------
# The engine seam
# ---------------------------------------------------------------------------

#: ``calculate(batch_tables, year) -> {variable: household values}``, the
#: values aligned with ``batch_tables["household"]`` rows.
BatchCalculator = Callable[[Mapping[str, pd.DataFrame], int], Mapping[str, np.ndarray]]


def _parameter_node(parameters: Any, dotted: str) -> Any:
    node = parameters
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


def _as_tuple(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def check_engine_definitions(system: Any) -> None:
    """Refuse an engine whose definitions differ from :data:`ENGINE_DEFINITIONS`."""

    variables = system.variables
    wanted = {
        *EARNINGS_VARIABLES,
        *COMPONENT_VARIABLES,
        NET_INCOME_VARIABLE,
        HOUSEHOLD_ID,
        *POPULATION_AGGREGATE_VARIABLES,
    }
    missing = sorted(name for name in wanted if name not in variables)
    if missing:
        raise HouseholdMeasuresError(
            f"The installed policyengine-us defines no variable(s) {missing}."
        )
    for name in EARNINGS_VARIABLES:
        entity = variables[name].entity.key
        if entity != "person":
            raise HouseholdMeasuresError(
                f"{name} is a {entity} variable; earnings sums person variables."
            )
    for name in (*COMPONENT_VARIABLES, NET_INCOME_VARIABLE):
        entity = variables[name].entity.key
        if entity != "household":
            raise HouseholdMeasuresError(
                f"{name} is a {entity} variable; the sidecar stores household ones."
            )
    for name, definition in ENGINE_DEFINITIONS.items():
        variable = variables[name]
        for attribute in ("adds", "subtracts"):
            if attribute not in definition:
                continue
            found = _as_tuple(getattr(variable, attribute, None))
            if found != definition[attribute]:
                raise HouseholdMeasuresError(
                    f"The installed engine's {name}.{attribute} is {found!r}, "
                    f"not {definition[attribute]!r}; the documented measure "
                    "definitions no longer describe this engine."
                )


def resolved_engine_parameters(system: Any, year: int) -> dict[str, Any]:
    parameters = system.parameters(f"{year}-01-01")
    return {
        MARKET_INCOME_PARAMETER: list(
            _parameter_node(parameters, MARKET_INCOME_PARAMETER)
        ),
        BENEFIT_PARAMETER: list(_parameter_node(parameters, BENEFIT_PARAMETER)),
        HEALTH_BENEFITS_PARAMETER: bool(
            _parameter_node(parameters, HEALTH_BENEFITS_PARAMETER)
        ),
        HUD_ABOLITION_PARAMETER: bool(
            _parameter_node(parameters, HUD_ABOLITION_PARAMETER)
        ),
        HEAD_START_PARAMETER: bool(_parameter_node(parameters, HEAD_START_PARAMETER)),
    }


@dataclass
class EngineRun:
    """What the engine seam reports beside the values."""

    batches: int = 0
    aggregate_variables_computed: tuple[str, ...] = ()


def _known(simulation: Any, names: Sequence[str]) -> set[tuple[str, str]]:
    from microcosm.data.compatibility_recertification import engine_known_periods

    return engine_known_periods(simulation, names)


def engine_batch_calculator(
    *,
    microsimulation_cls: Any,
    dataset_cls: Any,
    spm: Mapping[str, object] = US_RELEASE_SPM_SELECTION,
    multi_batch: bool,
    run: EngineRun | None = None,
) -> BatchCalculator:
    """A :data:`BatchCalculator` over one fresh ``Microsimulation`` per batch.

    Each batch engine reports the household ids it computed; values are
    realigned to the batch's household rows by id, never by position. In a
    multi-batch run a batch engine that computed any of
    :data:`POPULATION_AGGREGATE_VARIABLES` refuses (microcosm#956).
    """

    run = run if run is not None else EngineRun()
    variables = (*EARNINGS_VARIABLES, *COMPONENT_VARIABLES, NET_INCOME_VARIABLE)

    def calculate(
        batch: Mapping[str, pd.DataFrame], year: int
    ) -> Mapping[str, np.ndarray]:
        dataset = dataset_cls(**{e: batch[e] for e in US_ENTITIES}, time_period=year)
        simulation = microsimulation_cls(dataset=dataset, spm=dict(spm))
        try:
            before = _known(simulation, POPULATION_AGGREGATE_VARIABLES)
            engine_ids = _household_ids(
                simulation.calculate(HOUSEHOLD_ID, year, map_to="household")
            )
            values = {
                name: np.asarray(
                    simulation.calculate(name, year, map_to="household"),
                    dtype=np.float64,
                )
                for name in variables
            }
            computed = sorted(
                {
                    name
                    for name, _ in _known(simulation, POPULATION_AGGREGATE_VARIABLES)
                    - before
                }
            )
        finally:
            del simulation
            gc.collect()
        if computed:
            run.aggregate_variables_computed = tuple(
                sorted({*run.aggregate_variables_computed, *computed})
            )
            if multi_batch:
                raise HouseholdMeasuresError(
                    f"A household batch engine computed {computed}, which "
                    "aggregate over the whole simulated population; batched "
                    "measures would hold batch-local values (microcosm#956). "
                    "Rerun with --batch-households at least the household count."
                )
        expected = _household_ids(batch["household"][HOUSEHOLD_ID])
        if engine_ids.shape != expected.shape:
            raise HouseholdMeasuresError(
                f"A batch engine returned {engine_ids.shape[0]} households for "
                f"{expected.shape[0]} household rows."
            )
        take = _positions(
            _unique_index(engine_ids, "engine household_id"),
            expected,
            "The batch engine computed no household {}.",
        )
        run.batches += 1
        return {name: array[take] for name, array in values.items()}

    return calculate


# ---------------------------------------------------------------------------
# Assembly and checks (engine-free)
# ---------------------------------------------------------------------------


def assemble_household_measures(
    household_ids: np.ndarray,
    year: int,
    values: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """The sidecar table from per-household engine values, in household order."""

    household_ids = _household_ids(household_ids)
    n = len(household_ids)
    needed = (*EARNINGS_VARIABLES, *COMPONENT_VARIABLES, NET_INCOME_VARIABLE)
    missing = [name for name in needed if name not in values]
    if missing:
        raise HouseholdMeasuresError(f"No engine values for {missing}.")
    arrays: dict[str, np.ndarray] = {}
    for name in needed:
        array = np.asarray(values[name], dtype=np.float64)
        if array.shape != (n,):
            raise HouseholdMeasuresError(
                f"{name} has shape {array.shape}; expected ({n},)."
            )
        if not np.isfinite(array).all():
            raise HouseholdMeasuresError(f"{name} holds a non-finite value.")
        arrays[name] = array
    earnings = np.zeros(n, dtype=np.float64)
    for name in EARNINGS_VARIABLES:
        earnings = earnings + arrays[name]
    table = pd.DataFrame(
        {
            HOUSEHOLD_ID: household_ids,
            YEAR: np.full(n, int(year), dtype=np.int64),
            "earnings": earnings,
            "gross_income": arrays["household_market_income"]
            + arrays["household_benefits"],
            "net_income": arrays[NET_INCOME_VARIABLE],
            **{name: arrays[name] for name in COMPONENT_VARIABLES},
        }
    )
    return table[[name for name, _ in SIDECAR_COLUMNS]]


def _net_income_residual(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``net_income`` less its rebuilt components, and the components' scale."""

    rebuilt = (
        table["household_market_income"]
        + table["household_benefits"]
        + table["household_refundable_tax_credits"]
    ) - (
        table["household_tax_before_refundable_credits"]
        + table["household_health_costs"]
    )
    scale = sum(table[name].abs() for name in COMPONENT_VARIABLES)
    return (
        (table["net_income"] - rebuilt).to_numpy(dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
    )


def check_household_measures(
    table: pd.DataFrame,
    household_ids: np.ndarray,
    year: int,
) -> dict[str, Any]:
    """Refuse a sidecar that does not cover the H5's households exactly once.

    Checks, each over every row: the columns and their dtypes are
    :data:`SIDECAR_COLUMNS`; every ``household_id`` of the H5 appears exactly
    once and no other does; every ``year`` equals the H5's ``_time_period``;
    every value is finite; ``gross_income`` equals market income plus
    benefits; and ``net_income`` equals its components within
    ``IDENTITY_ATOL + IDENTITY_RTOL * scale``, ``scale`` being the sum of
    the components' absolute values. Returns the summary the receipt
    records.
    """

    failures: list[str] = []
    expected_columns = [name for name, _ in SIDECAR_COLUMNS]
    if list(table.columns) != expected_columns:
        raise HouseholdMeasuresError(
            f"Sidecar columns are {list(table.columns)}, not {expected_columns}."
        )
    for name, dtype in SIDECAR_COLUMNS:
        if str(table[name].dtype) != dtype:
            failures.append(f"{name} has dtype {table[name].dtype}, not {dtype}")
    if failures:
        raise HouseholdMeasuresError("; ".join(failures) + ".")

    source = _household_ids(household_ids)
    if len(np.unique(source)) != len(source):
        raise HouseholdMeasuresError("The H5's household table repeats a household_id.")
    ids = table[HOUSEHOLD_ID].to_numpy(dtype=np.int64)
    unique, counts = np.unique(ids, return_counts=True)
    repeated = unique[counts > 1]
    if len(repeated):
        failures.append(
            f"{len(repeated)} household_id(s) appear more than once "
            f"(first: {int(repeated[0])})"
        )
    missing = np.setdiff1d(source, unique, assume_unique=True)
    if len(missing):
        failures.append(
            f"{len(missing)} H5 household(s) have no row (first: {int(missing[0])})"
        )
    extra = np.setdiff1d(unique, source, assume_unique=True)
    if len(extra):
        failures.append(
            f"{len(extra)} row(s) name a household the H5 lacks "
            f"(first: {int(extra[0])})"
        )
    years = table[YEAR].to_numpy(dtype=np.int64)
    wrong_year = years != int(year)
    if wrong_year.any():
        failures.append(
            f"{int(wrong_year.sum())} row(s) carry year "
            f"{int(years[wrong_year][0])}, not the H5's {int(year)}"
        )
    numeric = table[[name for name, dtype in SIDECAR_COLUMNS if dtype == "float64"]]
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        failures.append("a measure holds a non-finite value")
    gross = (table["household_market_income"] + table["household_benefits"]).to_numpy()
    gross_off = table["gross_income"].to_numpy() != gross
    if gross_off.any():
        failures.append(
            f"{int(gross_off.sum())} row(s) have gross_income != "
            "household_market_income + household_benefits"
        )
    residual, scale = _net_income_residual(table)
    identity_off = ~(np.abs(residual) <= IDENTITY_ATOL + IDENTITY_RTOL * scale)
    if identity_off.any():
        failures.append(
            f"{int(identity_off.sum())} row(s) have net_income off its "
            f"components by more than {IDENTITY_ATOL} + {IDENTITY_RTOL} times "
            f"their scale (largest {float(np.abs(residual).max()):.6g})"
        )
    if failures:
        raise HouseholdMeasuresError("Sidecar refused: " + "; ".join(failures) + ".")
    return {
        "households": int(len(source)),
        "rows": int(len(table)),
        "every_household_exactly_once": True,
        "year_matches_h5": True,
        "gross_income_identity": "exact",
        "net_income_identity_max_abs_residual": float(np.abs(residual).max())
        if len(residual)
        else 0.0,
        "net_income_identity_tolerance": {
            "atol": IDENTITY_ATOL,
            "rtol": IDENTITY_RTOL,
            "scale": "sum of the five components' absolute values",
        },
    }


# ---------------------------------------------------------------------------
# Computing
# ---------------------------------------------------------------------------


def compute_household_measures(
    tables: Mapping[str, pd.DataFrame],
    year: int,
    *,
    calculate: BatchCalculator,
    batch_households: int = DEFAULT_BATCH_HOUSEHOLDS,
    progress: Callable[[int, int, float], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every household through ``calculate``, batch by batch."""

    _validate_batch_households(batch_households)
    household_ids = _household_ids(tables["household"][HOUSEHOLD_ID])
    n = len(household_ids)
    needed = (*EARNINGS_VARIABLES, *COMPONENT_VARIABLES, NET_INCOME_VARIABLE)
    collected: dict[str, list[np.ndarray]] = {name: [] for name in needed}
    collected_ids: list[np.ndarray] = []
    batches = 0
    total = -(-n // batch_households) if n else 0
    batch_seconds: list[float] = []
    for batch in household_batches(tables, batch_households):
        started = time.perf_counter()
        values = calculate(batch, year)
        batch_seconds.append(time.perf_counter() - started)
        batch_ids = batch["household"][HOUSEHOLD_ID].to_numpy(dtype=np.int64)
        for name in needed:
            array = np.asarray(values[name], dtype=np.float64)
            if array.shape != batch_ids.shape:
                raise HouseholdMeasuresError(
                    f"{name} has {array.shape[0]} values for "
                    f"{batch_ids.shape[0]} households in batch {batches}."
                )
            collected[name].append(array)
        collected_ids.append(batch_ids)
        batches += 1
        if progress is not None:
            progress(batches, total, batch_seconds[-1])
    scored_ids = (
        np.concatenate(collected_ids) if collected_ids else np.zeros(0, np.int64)
    )
    if not np.array_equal(scored_ids, household_ids):
        raise HouseholdMeasuresError(
            "The batches did not score the household table in its own order."
        )
    values = {
        name: np.concatenate(parts) if parts else np.zeros(0, np.float64)
        for name, parts in collected.items()
    }
    table = assemble_household_measures(household_ids, year, values)
    checks = check_household_measures(table, household_ids, year)
    return table, {
        "batch_households": batch_households,
        "batches": batches,
        "batch_seconds": _batch_timing(batch_seconds),
        "checks": checks,
    }


def _batch_timing(seconds: Sequence[float]) -> dict[str, float | None]:
    """The first batch (it builds the engine's caches) apart from the rest."""

    rest = list(seconds[1:])
    return {
        "first": round(seconds[0], 3) if seconds else None,
        "mean_after_first": round(sum(rest) / len(rest), 3) if rest else None,
        "max": round(max(seconds), 3) if seconds else None,
        "total": round(sum(seconds), 3),
    }


# ---------------------------------------------------------------------------
# Writing and reading the sidecar
# ---------------------------------------------------------------------------


def sidecar_paths(h5_path: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    h5_path = Path(h5_path)
    directory = Path(output_dir) if output_dir is not None else h5_path.parent
    return (
        directory / f"{h5_path.stem}{SIDECAR_SUFFIX}",
        directory / f"{h5_path.stem}{RECEIPT_SUFFIX}",
    )


def _write_exclusive(path: Path, write: Callable[[Path], None]) -> None:
    """Write through a temporary sibling and link it in without overwriting."""

    if os.path.lexists(path):
        raise FileExistsError(f"{path} exists; refusing to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    handle = os.open(temporary_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    os.close(handle)
    try:
        write(temporary_path)
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_sidecar(table: pd.DataFrame, path: Path) -> None:
    """Gzip CSV at full float precision, reproducible with the same zlib build.

    The gzip header carries no file name and a zero timestamp, so the same
    table gives the same bytes with the same zlib build (the temporary name
    never leaks).
    """

    def write(target: Path) -> None:
        with (
            target.open("wb") as raw,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
            ) as compressed,
            io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
        ):
            table.to_csv(text, index=False, lineterminator="\n")

    _write_exclusive(Path(path), write)


def read_sidecar(path: Path) -> pd.DataFrame:
    """Read a sidecar with pandas alone, in the dtypes it was written with."""

    return pd.read_csv(
        path,
        dtype={name: dtype for name, dtype in SIDECAR_COLUMNS},
        float_precision="round_trip",
    )


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    def write(target: Path) -> None:
        target.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    _write_exclusive(Path(path), write)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def build_receipt(
    *,
    h5_path: Path,
    h5_sha256: str,
    year: int,
    sidecar_path: Path,
    table: pd.DataFrame,
    run_info: Mapping[str, Any],
    engine: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "kind": SIDECAR_KIND,
        "source_h5": {
            "name": Path(h5_path).name,
            "sha256": h5_sha256,
            "size_bytes": Path(h5_path).stat().st_size,
            "time_period": int(year),
        },
        "sidecar": {
            "name": Path(sidecar_path).name,
            "sha256": sha256_file(sidecar_path),
            "format": "csv, gzip-compressed, header row, comma separated",
            "rows": int(len(table)),
            "columns": [
                {"name": name, "dtype": dtype} for name, dtype in SIDECAR_COLUMNS
            ],
            "key": HOUSEHOLD_ID,
        },
        "year": int(year),
        "measures": {name: dict(value) for name, value in MEASURE_DEFINITIONS.items()},
        "engine": dict(engine),
        "batching": {
            "method": "whole-household batches, one Microsimulation each",
            "batch_households": run_info["batch_households"],
            "batches": run_info["batches"],
            "batch_seconds": dict(run_info["batch_seconds"]),
            "population_aggregate_guard": list(POPULATION_AGGREGATE_VARIABLES),
        },
        "checks": dict(run_info["checks"]),
        "timing": dict(timing),
    }


def verify_sidecar(
    h5_path: Path,
    sidecar_path: Path | None = None,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Re-check a written sidecar against its H5 with pandas and PyTables only."""

    default_sidecar, default_receipt = sidecar_paths(h5_path)
    sidecar_path = Path(sidecar_path or default_sidecar)
    receipt_path = Path(receipt_path or default_receipt)
    household_ids, year = read_h5_household_ids(Path(h5_path))
    table = read_sidecar(sidecar_path)
    checks = check_household_measures(table, household_ids, year)
    receipt = json.loads(receipt_path.read_text())
    problems: list[str] = []
    if receipt.get("kind") != SIDECAR_KIND:
        problems.append(f"receipt kind is {receipt.get('kind')!r}")
    if receipt.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        problems.append(f"receipt schema_version is {receipt.get('schema_version')!r}")
    if receipt.get("year") != year:
        problems.append(f"receipt year {receipt.get('year')!r} is not the H5's {year}")
    h5_sha256 = sha256_file(Path(h5_path))
    if (receipt.get("source_h5") or {}).get("sha256") != h5_sha256:
        problems.append("the H5's sha256 differs from the receipt's")
    sidecar_sha256 = sha256_file(sidecar_path)
    if (receipt.get("sidecar") or {}).get("sha256") != sidecar_sha256:
        problems.append("the sidecar's sha256 differs from the receipt's")
    if problems:
        raise HouseholdMeasuresError("Sidecar receipt refused: " + "; ".join(problems))
    return {
        "h5": str(h5_path),
        "h5_sha256": h5_sha256,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sidecar_sha256,
        "year": year,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _installed_engine() -> tuple[Any, Any, Any]:
    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset, extend_single_year_dataset
    from policyengine_us.system import system

    def dataset_through_file_year(**kwargs: Any) -> Any:
        dataset = USSingleYearDataset(**kwargs)
        # A USMultiYearDataset bypasses Microsimulation's default extension
        # through the CPI-U horizon. No measure reads after the file's year.
        return extend_single_year_dataset(dataset, end_year=int(kwargs["time_period"]))

    return Microsimulation, dataset_through_file_year, system


def run_compute(
    h5_path: Path,
    *,
    output_dir: Path | None = None,
    expect_year: int | None = None,
    batch_households: int = DEFAULT_BATCH_HOUSEHOLDS,
    engine_factory: Callable[[], tuple[Any, Any, Any]] = _installed_engine,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Compute, check and write the sidecar and receipt for ``h5_path``."""

    _validate_batch_households(batch_households)
    started = time.perf_counter()
    h5_path = Path(h5_path)
    sidecar_path, receipt_path = sidecar_paths(h5_path, output_dir)
    for path in (sidecar_path, receipt_path):
        if os.path.lexists(path):
            raise FileExistsError(f"{path} exists; refusing to overwrite it.")
    h5_sha256 = sha256_file(h5_path)
    tables, year = read_h5_tables(h5_path)
    if expect_year is not None and year != expect_year:
        raise HouseholdMeasuresError(
            f"{h5_path} states year {year}, not the expected {expect_year}."
        )
    n_households = len(tables["household"])
    log(f"{h5_path.name}: year {year}, {n_households:,} households")
    microsimulation_cls, dataset_cls, system = engine_factory()
    check_engine_definitions(system)
    parameters = resolved_engine_parameters(system, year)
    run = EngineRun()
    calculate = engine_batch_calculator(
        microsimulation_cls=microsimulation_cls,
        dataset_cls=dataset_cls,
        multi_batch=n_households > batch_households,
        run=run,
    )
    table, run_info = compute_household_measures(
        tables,
        year,
        calculate=calculate,
        batch_households=batch_households,
        progress=lambda done, total, seconds: log(
            f"batch {done}/{total} in {seconds:.1f} s"
        ),
    )
    del tables
    if sha256_file(h5_path) != h5_sha256:
        raise HouseholdMeasuresError(f"{h5_path} changed while it was scored.")
    write_sidecar(table, sidecar_path)
    try:
        reread = read_sidecar(sidecar_path)
        if not reread.equals(table):
            raise HouseholdMeasuresError(
                "The written sidecar does not read back exactly."
            )
        engine = {
            "policyengine_us": _package_version("policyengine-us"),
            "policyengine_core": _package_version("policyengine-core"),
            "python": platform.python_version(),
            "spm_selection": dict(US_RELEASE_SPM_SELECTION),
            "period": int(year),
            "resolved_parameters": parameters,
            "aggregate_variables_computed": list(run.aggregate_variables_computed),
        }
        timing = {
            "wall_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        receipt = build_receipt(
            h5_path=h5_path,
            h5_sha256=h5_sha256,
            year=year,
            sidecar_path=sidecar_path,
            table=table,
            run_info=run_info,
            engine=engine,
            timing=timing,
        )
        _write_json(receipt_path, receipt)
    except BaseException:
        sidecar_path.unlink(missing_ok=True)
        raise
    log(f"wrote {sidecar_path} and {receipt_path}")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    compute = commands.add_parser("compute", help="compute and write the sidecar")
    compute.add_argument("--h5", type=Path, required=True)
    compute.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the sidecar and receipt (default: beside the H5)",
    )
    compute.add_argument(
        "--expect-year",
        type=int,
        default=None,
        help="refuse unless the H5's _time_period is this year",
    )
    compute.add_argument(
        "--batch-households", type=int, default=DEFAULT_BATCH_HOUSEHOLDS
    )
    verify = commands.add_parser("verify", help="re-check a written sidecar")
    verify.add_argument("--h5", type=Path, required=True)
    verify.add_argument("--sidecar", type=Path, default=None)
    verify.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "compute":
        run_compute(
            args.h5,
            output_dir=args.output_dir,
            expect_year=args.expect_year,
            batch_households=args.batch_households,
        )
    else:
        print(json.dumps(verify_sidecar(args.h5, args.sidecar, args.receipt), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
