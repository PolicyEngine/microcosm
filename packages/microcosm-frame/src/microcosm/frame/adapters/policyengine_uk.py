"""PolicyEngine-UK adapter for the RulesEngine protocol.

The PolicyEngine-UK import is deferred until a method needs engine metadata or
simulation. Importing this module therefore keeps ``microcosm-frame`` usable
without the UK extra.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from microcosm.frame.adapters.policyengine_uk_concepts import (
    POLICYENGINE_UK_CONCEPT_MAPPING,
)
from microcosm.frame.bundle import Frame
from microcosm.frame.concept_mapping import ConceptMapping
from microcosm.frame.materialize import engine_tables
from microcosm.frame.rules import ExportContract, RulesEngine
from microcosm.frame.schema import EntitySchema, VariableMetadata

__all__ = [
    "POLICYENGINE_UK_CONCEPT_MAPPING",
    "PolicyEngineUKEngine",
    "ProjectingRulesEngine",
    "SUPPLIED_ROLE_INPUTS",
    "UK_DEFAULT_SAVINGS_INCOME_PROJECTION",
    "UK_ISA_INCOME_CONTRACT",
    "UK_ISA_INCOME_CONTRACT_ATTR",
    "UK_LEGACY_ISA_INCOME_CONTRACT",
    "UK_SAVINGS_INCOME_PROJECTION",
    "UK_SAVINGS_INCOME_PROJECTION_ATTR",
    "UK_SCHEMA",
    "frame_isa_income_contract",
    "frame_savings_income_projection",
    "uk_dataset_through",
    "uk_isa_income_contract_from",
    "uk_rebalancing_years_through",
    "uk_savings_income_projection_from",
    "uk_whole_rebalancing_years",
    "validate_uc_claimant_input",
]

UK_SCHEMA = EntitySchema(group_entities=("benunit", "household"))
#: How the producer's savings interest holds ISA interest, declared to
#: policyengine-uk (its ``isa_income_contract``, from 2.162.2): non-ISA interest
#: in ``savings_interest_income`` and ISA interest in
#: ``individual_savings_account_interest_income``, each projected by its own
#: index (microcosm#1360). ``tax_free_savings_income`` is the engine's sum of
#: the ISA parts; a stored copy would override it and never be uprated, so the
#: producer never stores it.
UK_ISA_INCOME_CONTRACT: Mapping[str, object] = MappingProxyType(
    {"version": 1, "representation": "additive", "projection": "additive"}
)
#: How policyengine-uk reads a dataset that declares no contract: gross
#: ``savings_interest_income`` with the ISA part in ``tax_free_savings_income``,
#: the enhanced FRS's layout and Microcosm's before microcosm#1360.
UK_LEGACY_ISA_INCOME_CONTRACT: Mapping[str, object] = MappingProxyType(
    {"version": 1, "representation": "legacy_gross", "projection": "legacy_gross"}
)
#: The H5 root attribute policyengine-uk reads the contract from, and the frame
#: metadata key that carries it.
UK_ISA_INCOME_CONTRACT_ATTR = "policyengine_uk.isa_income_contract"
#: The (representation, projection) pairs policyengine-uk accepts at version 1.
_ISA_INCOME_CONTRACT_PAIRS = frozenset(
    {
        ("legacy_gross", "legacy_gross"),
        ("additive", "legacy_gross"),
        ("additive", "additive"),
    }
)


def uk_isa_income_contract_from(value: object) -> dict[str, object]:
    """Read a declared ISA income contract, a mapping or its JSON.

    Anything but one of policyengine-uk's version-1 pairs fails, so a frame's
    reading of its savings columns is never inferred from their amounts.
    """

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"ISA income contract {value!r} is not valid JSON."
            ) from exc
    if not isinstance(value, Mapping):
        raise ValueError(
            f"ISA income contract must be a mapping, got {type(value).__name__}."
        )
    contract = dict(value)
    version = contract.get("version")
    pair = (contract.get("representation"), contract.get("projection"))
    if (
        set(contract) != {"version", "representation", "projection"}
        or type(version) is not int
        or version != 1
        or pair not in _ISA_INCOME_CONTRACT_PAIRS
    ):
        raise ValueError(f"Unsupported ISA income contract {contract!r}.")
    return {"version": 1, "representation": pair[0], "projection": pair[1]}


def frame_isa_income_contract(bundle: Frame) -> dict[str, object]:
    """The contract a frame's savings columns follow.

    A frame loaded from an H5 carries the file's declaration (one declaring
    nothing reads as legacy gross); a frame assembled in process holds the
    producer's columns and declares the producer's contract.
    """

    metadata = getattr(bundle, "metadata", None) or {}
    declared = metadata.get(UK_ISA_INCOME_CONTRACT_ATTR)
    if declared is None:
        return dict(UK_ISA_INCOME_CONTRACT)
    return uk_isa_income_contract_from(declared)


#: The path the producer's savings income was calibrated to grow by, declared
#: to policyengine-uk (its ``savings_income_projection``, from 2.165.2): the
#: OBR's savings income per person (October 2026 EFO Table 4.5) in the fiscal
#: years 2022 to 2025, and GDP per head in other years. The SPI income stage
#: carries the donors' interest to the frame's year, and the SPI Table 3.7
#: interest targets reach the calibration year, on the OBR series (María's
#: ruling of 2026-10-29, microcosm#1418), so the engine's projection follows it
#: (declared on her ruling of 2026-11-04, microcosm#1095).
UK_SAVINGS_INCOME_PROJECTION = "obr_savings_income_2022_2025"
#: How policyengine-uk projects a dataset that declares no rule: GDP per head
#: in every year.
UK_DEFAULT_SAVINGS_INCOME_PROJECTION = "gdp_per_head"
#: The H5 root attribute policyengine-uk reads the rule from, and the frame
#: metadata key that carries a rule other than the producer's.
UK_SAVINGS_INCOME_PROJECTION_ATTR = "policyengine_uk.savings_income_projection"
_SAVINGS_INCOME_PROJECTIONS = frozenset(
    {UK_SAVINGS_INCOME_PROJECTION, UK_DEFAULT_SAVINGS_INCOME_PROJECTION}
)


def uk_savings_income_projection_from(value: object) -> str:
    """Read a declared savings income projection rule, a name or its bytes.

    A rule policyengine-uk does not name fails, so a frame's savings growth is
    never guessed.
    """

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or value not in _SAVINGS_INCOME_PROJECTIONS:
        raise ValueError(f"Unsupported savings income projection {value!r}.")
    return value


def frame_savings_income_projection(bundle: Frame) -> str:
    """The rule a frame's savings income was calibrated to grow by.

    A frame loaded from an H5 that declares another rule, or none (GDP per
    head), carries it; a frame assembled in process holds the producer's
    savings income and declares the producer's rule.
    """

    metadata = getattr(bundle, "metadata", None) or {}
    declared = metadata.get(UK_SAVINGS_INCOME_PROJECTION_ATTR)
    if declared is None:
        return UK_SAVINGS_INCOME_PROJECTION
    return uk_savings_income_projection_from(declared)


def uk_dataset_through(dataset: Any, end_year: int | str, parameters: Any) -> Any:
    """A one-year ``UKSingleYearDataset`` projected only to ``end_year``.

    policyengine-uk's ``Simulation`` projects a one-year dataset (an object,
    or a path to a one-year H5) with its own uprating to
    ``PROJECTION_END_YEAR`` (2032) and holds every year, whatever years the
    caller reads. ``extend_single_year_dataset`` with ``end_year`` returns
    that projection from the dataset's year through ``end_year`` alone, as a
    ``UKMultiYearDataset``, which ``Simulation`` takes as given.
    policyengine-uk builds each year only from earlier years, so the kept
    years are the ones the full projection holds. ``parameters`` is the
    parameter tree the uprating reads: current law in a build.
    """

    from policyengine_uk.data.economic_assumptions import extend_single_year_dataset

    end_year = int(end_year)
    data_year = int(str(dataset.time_period)[:4])
    if end_year < data_year:
        raise ValueError(
            f"end_year {end_year} is before the dataset's year {data_year}; "
            "policyengine-uk projects a dataset forward only."
        )
    return extend_single_year_dataset(dataset, parameters, end_year=end_year)


#: One cut of policyengine-uk's rebalancing loop at a time. The cut patches the
#: engine's module, which every thread reads, so a second thread waits here
#: until the first has restored the installed range.
_REBALANCING_LOCK = threading.RLock()
#: The cut in force, empty when there is none: the range the engine had
#: installed, the end year, the cut range, and how many open blocks share it.
_REBALANCING_CUT: dict[str, Any] = {}


@contextmanager
def uk_rebalancing_years_through(end_year: int | str) -> Iterator[range]:
    """Run policyengine-uk's Universal Credit rebalancing loop only to ``end_year``.

    Every policyengine-uk constructor runs ``add_universal_credit_reform``
    (``scenarios/uc_reform.py``). For each year of the module's
    ``REBALANCING_YEARS``, 2026 to ``PROJECTION_END_YEAR``, it computes
    ``uc_LCWRA_element`` and the arrays behind it, whatever years the dataset
    holds. Inside the block that range stops at ``end_year`` (it is empty for
    2024 and 2025), and it never gains a year. On exit, on success or on error,
    the installed range is restored. The range is built from
    ``PROJECTION_END_YEAR`` when the module is imported, so a patch of that
    constant would not reach the loop.

    The patch is process-wide, so one cut is in force at a time. A block on
    another thread waits until this one has restored the range. A block
    opened inside this one on the same thread shares the cut when it names
    the same year, and the range is restored when the last of them exits,
    whatever the order; one that names another year is refused, since its
    simulation would get this block's loop. A range that something else
    changed inside the block is refused on exit, after the installed one is
    restored: the simulation may not have run the loop it asked for.
    """

    from policyengine_uk.scenarios import uc_reform

    end_year = int(end_year)
    with _REBALANCING_LOCK:
        cut = _REBALANCING_CUT
        if not cut:
            installed = uc_reform.REBALANCING_YEARS
            if not isinstance(installed, range) or installed.step != 1:
                raise TypeError(
                    "policyengine-uk's uc_reform.REBALANCING_YEARS is no longer a "
                    f"range of consecutive years ({installed!r}); review the "
                    "rebalancing cut against the installed engine."
                )
            years = range(installed.start, min(installed.stop, end_year + 1))
            uc_reform.REBALANCING_YEARS = years
            cut.update(installed=installed, end_year=end_year, years=years, blocks=0)
        elif cut["end_year"] != end_year:
            raise RuntimeError(
                "policyengine-uk's rebalancing loop is cut at "
                f"{cut['end_year']} for a simulation that is being built; a "
                f"simulation through {end_year} cannot be built inside that "
                "block, because the cut is process-wide."
            )
        years = cut["years"]
        cut["blocks"] += 1
        found: Any = years
        try:
            yield years
        finally:
            cut["blocks"] -= 1
            if not cut["blocks"]:
                found = uc_reform.REBALANCING_YEARS
                uc_reform.REBALANCING_YEARS = cut["installed"]
                cut.clear()
        if not (isinstance(found, range) and found == years):
            raise RuntimeError(
                "policyengine-uk's uc_reform.REBALANCING_YEARS changed inside the "
                f"rebalancing cut at {end_year}: something else set it to "
                f"{found!r}. The installed range is restored, but the simulation "
                "built in the block may not have run the loop it asked for."
            )


@contextmanager
def uk_whole_rebalancing_years() -> Iterator[None]:
    """Build a simulation that needs policyengine-uk's whole rebalancing loop.

    A simulation built outside :func:`uk_rebalancing_years_through` runs the
    Universal Credit rebalancing loop over the range the engine's module holds
    at that moment. A release is scored as users load it, on every year of
    that range, so its simulation is built inside this block, where no cut is
    in force: a cut on another thread is waited for and none starts before the
    block exits, and a cut open on this thread is refused. A loaded engine
    module whose range is not the one policyengine-uk builds from its own
    ``FIRST_REBALANCING_YEAR`` and ``PROJECTION_END_YEAR`` is refused too, so
    a cut that anything left in the process cannot shorten the loop without an
    error. An engine module that was never imported still has its own range.
    """

    with _REBALANCING_LOCK:
        if _REBALANCING_CUT:
            raise RuntimeError(
                "policyengine-uk's rebalancing loop is cut at "
                f"{_REBALANCING_CUT['end_year']} for a simulation that is being "
                "built; a simulation that needs the whole loop cannot be built "
                "inside that block."
            )
        uc_reform = sys.modules.get("policyengine_uk.scenarios.uc_reform")
        if uc_reform is not None:
            installed = getattr(uc_reform, "REBALANCING_YEARS", None)
            first = getattr(uc_reform, "FIRST_REBALANCING_YEAR", None)
            last = getattr(uc_reform, "PROJECTION_END_YEAR", None)
            if (
                not isinstance(installed, range)
                or type(first) is not int
                or type(last) is not int
            ):
                raise TypeError(
                    "policyengine-uk's uc_reform no longer builds REBALANCING_YEARS "
                    "as a range from FIRST_REBALANCING_YEAR to PROJECTION_END_YEAR "
                    f"({installed!r}, {first!r}, {last!r}); review the rebalancing "
                    "cut against the installed engine."
                )
            if installed != range(first, last + 1):
                raise RuntimeError(
                    "policyengine-uk's uc_reform.REBALANCING_YEARS is "
                    f"{installed!r}, not the engine's own "
                    f"{range(first, last + 1)!r}: a cut was left in this process. "
                    "A simulation built now would skip the Universal Credit "
                    "rebalancing loop in the missing years."
                )
        yield


_PERSON_TABLE = "person"
_GROUP_TABLES = ("benunit", "household")
_DTYPE_KIND_BY_VALUE_TYPE: dict[type, str] = {
    float: "float",
    int: "int",
    bool: "bool",
    str: "str",
}
_PERIOD_BY_DEFINITION: dict[str, str] = {"year": "year", "month": "month"}


class ProjectingRulesEngine(RulesEngine, Protocol):
    """A rules engine that can read a bundle from an earlier year.

    ``RulesEngine.materialize`` reads the bundle's values as ``period``
    inputs. An engine that projects a dataset forward itself accepts
    ``dataset_period``: the year the bundle's values belong to. It stamps the
    dataset with that year and computes ``variables`` at ``period`` on its own
    projection, as it does for a released file. A stage that needs this names
    the protocol. It is declared beside the adapter that implements it:
    ``microcosm.frame.rules`` is inside pinned kernel hashes, and a
    typing-only addition there would move them.
    """

    def materialize(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str,
        *,
        dataset_period: int | str | None = None,
    ) -> Mapping[str, np.ndarray]:
        """Compute ``variables`` for ``period`` on a ``dataset_period`` bundle."""
        ...


class PolicyEngineUKEngine:
    """RulesEngine adapter backed by ``policyengine_uk``."""

    country = "uk"

    def __init__(self) -> None:
        self._system: Any = None

    def variable_metadata(self, name: str) -> VariableMetadata:
        """Return entity, dtype kind, and period semantics for a UK variable."""

        variable = self._variable(name)
        return VariableMetadata(
            name=name,
            entity=variable.entity.key,
            dtype=_DTYPE_KIND_BY_VALUE_TYPE.get(variable.value_type, "str"),
            period=_PERIOD_BY_DEFINITION.get(
                getattr(variable, "definition_period", "year"), "point"
            ),
        )

    def variables(self) -> list[str]:
        """Return the UK engine's input variable names."""

        return sorted(
            name
            for name, variable in self._tax_benefit_system().variables.items()
            if not _is_engine_computed(variable)
        )

    def entity_schema(self) -> EntitySchema:
        """Return the UK national entity schema."""

        return UK_SCHEMA

    def concept_mapping(self) -> ConceptMapping:
        """Return the concept-to-input mapping (no engine import required)."""

        return POLICYENGINE_UK_CONCEPT_MAPPING

    def materialize(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str,
        *,
        dataset_period: int | str | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute ``variables`` for ``period`` with a UK Microsimulation.

        The bundle's values are read as ``period`` inputs. Pass
        ``dataset_period`` where they belong to an earlier year: the dataset is
        then stamped with that year and policyengine-uk projects it forward to
        ``period`` as it does a released file (``extend_single_year_dataset``).

        The simulation reads only ``period``, so it holds no later year
        (:meth:`_read_year_simulation`).
        """

        microsimulation_class = self._import_policyengine_uk().Microsimulation
        if dataset_period is not None and int(dataset_period) > int(period):
            raise ValueError(
                f"dataset_period {dataset_period} is after period {period}; "
                "policyengine-uk projects a dataset forward only."
            )
        dataset = self._build_dataset(
            bundle, period if dataset_period is None else dataset_period
        )
        simulation = self._read_year_simulation(microsimulation_class, dataset, period)
        validate_uc_claimant_input(simulation, bundle.table("person"), period)
        results: dict[str, np.ndarray] = {}
        for name in variables:
            entity = self.variable_metadata(name).entity
            values = np.asarray(simulation.calculate(name, period=period))
            expected = bundle.n(entity)
            if values.shape != (expected,):
                raise ValueError(
                    f"Materialized variable {name!r} has shape {values.shape} "
                    f"but entity {entity!r} has {expected} row(s)."
                )
            results[name] = values
        return results

    def export_contract(self) -> ExportContract:
        """UK export contracts are owned by microcosm-build for this release."""

        raise NotImplementedError(
            "PolicyEngine-UK export contracts are not implemented in the frame "
            "adapter yet; use the UK national build writer."
        )

    def write_dataset(
        self,
        bundle: Frame,
        path: str | Path,
        period: int | str,
    ) -> None:
        """UK dataset export remains on the national-frame writer in E1."""

        raise NotImplementedError(
            "PolicyEngine-UK dataset export is not implemented in the frame "
            "adapter yet; use microcosm.build.uk_runtime.national_frame."
            "write_uk_national_frame."
        )

    def _read_year_simulation(
        self, microsimulation_class: Any, dataset: Any, period: int | str
    ) -> Any:
        """A simulation of a one-year ``dataset`` that holds no year after ``period``.

        The dataset's year through ``period`` is projected by policyengine-uk's
        own uprating on the adapter's current-law parameters
        (:func:`uk_dataset_through`), and the Universal Credit rebalancing
        loop stops at ``period`` while the simulation is built
        (:func:`uk_rebalancing_years_through`). A later year would change no
        value at ``period``: policyengine-uk builds each year only from
        earlier ones.
        """

        projected = uk_dataset_through(
            dataset, period, self._tax_benefit_system().parameters
        )
        with uk_rebalancing_years_through(period):
            return microsimulation_class(dataset=projected)

    def _import_policyengine_uk(self) -> Any:
        try:
            import policyengine_uk
        except ImportError as exc:
            raise ImportError(
                "The PolicyEngine-UK adapter requires the 'policyengine-uk' "
                "package. Install it with 'microcosm-frame[uk]'."
            ) from exc
        return policyengine_uk

    def _tax_benefit_system(self) -> Any:
        if self._system is None:
            self._system = self._import_policyengine_uk().CountryTaxBenefitSystem()
        return self._system

    def _variable(self, name: str) -> Any:
        variables = self._tax_benefit_system().variables
        if name not in variables:
            raise ValueError(f"Unknown PolicyEngine-UK variable {name!r}.")
        return variables[name]

    def enum_domain(self, name: str) -> Any:
        """The Enum an engine variable's stored values must belong to (gates)."""

        return getattr(self._variable(name), "possible_values", None)

    def _build_dataset(self, bundle: Frame, period: int | str) -> Any:
        from policyengine_uk.data import UKSingleYearDataset

        expected = (_PERSON_TABLE, *_GROUP_TABLES)
        if set(bundle.entities) != set(expected):
            raise ValueError(
                f"PolicyEngine-UK adapter requires the UK entities "
                f"{list(expected)}; bundle has {list(bundle.entities)}."
            )
        tables = engine_tables(bundle, weighted_entities=("household",))
        return UKSingleYearDataset(
            person=tables["person"].copy(),
            benunit=tables["benunit"].copy(),
            household=tables["household"].copy(),
            fiscal_year=int(period),
            isa_income_contract=frame_isa_income_contract(bundle),
            savings_income_projection=frame_savings_income_projection(bundle),
        )


def _is_engine_computed(variable: Any) -> bool:
    if getattr(variable, "adds", None) or getattr(variable, "subtracts", None):
        return True
    if getattr(variable, "formula", None) is not None:
        return True
    return bool(getattr(variable, "formulas", None))


#: Supplied benefit-unit roles and the first policyengine-uk release that
#: reads each. ``is_claimant_or_partner`` (2.107.0) is read across the
#: model's programmes, and ``is_uc_claimant`` derives from it from that
#: release. ``is_hbai_dependent_child`` (2.107.0) sets the HBAI person
#: types, which the BHC and AHC equivalisation reads.
SUPPLIED_ROLE_INPUTS: tuple[tuple[str, str], ...] = (
    ("is_uc_claimant", "2.98.0"),
    ("is_claimant_or_partner", "2.107.0"),
    ("is_hbai_dependent_child", "2.107.0"),
)


def validate_uc_claimant_input(simulation: Any, person: Any, period: int | str) -> None:
    """Require supplied benefit-unit roles to survive the model's dataset loader.

    The UK loader ignores unknown columns. FRS roles must therefore remain
    explicit inputs; recalculating the model's relationship fallback can give
    a different answer for young partners or older nonqualifying dependants.
    Frames without recorded roles retain ordinary calculator behavior. Every
    role in ``SUPPLIED_ROLE_INPUTS`` that the frame carries is checked.
    """
    for name, floor in SUPPLIED_ROLE_INPUTS:
        if name in person:
            _validate_supplied_role(simulation, person, period, name, floor)


def _validate_supplied_role(
    simulation: Any, person: Any, period: int | str, name: str, floor: str
) -> None:
    try:
        installed = metadata.version("policyengine-uk")
    except metadata.PackageNotFoundError:
        installed = "unavailable"
    guidance = (
        f"Supplied {name} requires policyengine-uk>={floor} with explicit "
        f"person/Boolean/year input support (installed {installed})."
    )
    definition = simulation.tax_benefit_system.variables.get(name)
    if (
        definition is None
        or getattr(getattr(definition, "entity", None), "key", None) != "person"
        or getattr(definition, "value_type", None) is not bool
        or getattr(definition, "definition_period", None) != "year"
    ):
        raise ValueError(guidance)
    expected = person[name].to_numpy()
    if expected.ndim != 1 or expected.dtype.kind != "b":
        raise ValueError(
            f"{guidance} Source {name} must contain complete Boolean roles."
        )
    if name not in getattr(simulation, "input_variables", ()):
        raise ValueError(f"{guidance} The loader did not retain {name} as an input.")
    raw = simulation.calculate(name, period)
    actual = np.asarray(raw.values if hasattr(raw, "values") else raw)
    if actual.dtype.kind != "b" or not np.array_equal(actual, expected):
        raise ValueError(
            f"{guidance} Loaded {name} differs from the supplied source roles."
        )
