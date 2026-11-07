"""PolicyEngine-UK adapter for the RulesEngine protocol.

The PolicyEngine-UK import is deferred until a method needs engine metadata or
simulation. Importing this module therefore keeps ``microcosm-frame`` usable
without the UK extra.

A frame whose households carry more than one clone of the dense spine's
geography (``UK_SPINE_ENGINE_BATCH_COLUMN``) is computed clone by clone, one
simulation per clone, through the shared engine-block machinery
(microcosm#931); every other frame takes one simulation, as it always has.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from microcosm.frame.adapters.engine_population import (
    SPINE_BATCHED_PASS,
    HouseholdBatch,
    household_batches,
    materialize_in_batches,
)
from microcosm.frame.adapters.policyengine_uk_concepts import (
    POLICYENGINE_UK_CONCEPT_MAPPING,
)
from microcosm.frame.adapters.policyengine_uk_population import (
    UK_ENGINE_POPULATION,
    UK_SPINE_ENGINE_BATCH_COLUMN,
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
    "uk_isa_income_contract_from",
    "uk_savings_income_projection_from",
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
    """RulesEngine adapter backed by ``policyengine_uk``.

    Whether a call runs in batches is decided from the frame, never set
    here: one whose household table carries ``UK_SPINE_ENGINE_BATCH_COLUMN``
    with more than one clone is computed clone by clone
    (:meth:`simulation_batches`), since one simulation over a dense spine's
    pool does not fit in memory (microcosm#931). ``clones``, the spine's
    clone count when known, only refuses: a frame carrying the column with
    any other number of clones is part of a pool, or not this spine's, and
    an engine over it would read its policyengine-uk shares over the wrong
    population. A frame without the column still takes one simulation over
    the rows it holds, as a stage before the clones exist must; that no
    stage after them drops the column is the clone stage's integration
    check, not this adapter's.
    """

    country = "uk"

    def __init__(self, *, clones: int | None = None) -> None:
        if clones is not None:
            # Any whole number (a numpy integer too), but not a boolean.
            try:
                count = None if isinstance(clones, bool) else operator.index(clones)
            except TypeError:
                count = None
            if count is None or count < 1:
                raise ValueError(
                    "clones is a positive whole number of clones, or None."
                )
            clones = count
        self._system: Any = None
        self._clones = clones

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

        A bundle carrying more than one clone (:meth:`simulation_batches`) is
        computed clone by clone (:func:`materialize_in_batches`): each clone's
        simulation must hold the parameters the UK declaration's
        restatements read as :meth:`parameters` does, reads the pool's
        shares (``c_b = S_b / S`` at the frame's own weights), the
        declaration's guard judges what it ran under the spine's pass, which
        raises on anything only the pool defines, and the values come back
        as one simulation over the bundle returns them.
        """

        microsimulation_class = self._import_policyengine_uk().Microsimulation
        if dataset_period is not None and int(dataset_period) > int(period):
            raise ValueError(
                f"dataset_period {dataset_period} is after period {period}; "
                "policyengine-uk projects a dataset forward only."
            )
        batches = self._clone_batches(bundle)
        if batches is not None:
            return self._materialize_in_batches(
                bundle,
                variables,
                period,
                dataset_period=dataset_period,
                batches=batches,
                microsimulation_class=microsimulation_class,
            )
        dataset = self._build_dataset(
            bundle, period if dataset_period is None else dataset_period
        )
        simulation = microsimulation_class(dataset=dataset)
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

    def simulation_batches(self, bundle: Frame) -> int:
        """How many simulations :meth:`materialize` runs for ``bundle``.

        One, or one per clone where the bundle's households carry more than
        one; the same decision, and the same refusals, as materialize's. No
        engine is built.
        """

        batches = self._clone_batches(bundle)
        return 1 if batches is None else len(batches)

    def parameters(self, period: int | str) -> Any:
        """The engine's parameters at ``period``, from this adapter's own
        fresh ``CountryTaxBenefitSystem``.

        No simulation reads this tree: each builds its own and edits it
        while it is constructed. A call that runs in batches holds every
        batch's tree equal to this one at the call's period wherever the UK
        declaration's restatements read it (``UK_ENGINE_POPULATION.restated``:
        the private-school attendance rates and population factor), and
        raises otherwise, so a stage that restates the attendance formula
        over the pool from these parameters reads what each batch's engine
        holds (microcosm#931).
        """

        return self._tax_benefit_system().parameters(period)

    def batch_record(self, bundle: Frame) -> dict[str, object] | None:
        """How a call over ``bundle`` runs in batches, or None when it takes
        one simulation.

        The column the batches follow, the number of batches (one per
        clone), and the pass and the declaration each batch is judged under
        (a violation in any raises). Read from the frame alone, by the same
        decision :meth:`materialize` makes, never from the calls this
        process ran, so a spine resumed from its checkpoints records what a
        fresh run of it does. Names and the clone count only, never a
        record's value or a count of records.
        """

        batches = self.simulation_batches(bundle)
        if batches == 1:
            return None
        return {
            "by": UK_SPINE_ENGINE_BATCH_COLUMN,
            "batches": batches,
            "pass": SPINE_BATCHED_PASS.name,
            "declaration_sha256": UK_ENGINE_POPULATION.sha256,
        }

    def _clone_batches(self, bundle: Frame) -> tuple[HouseholdBatch, ...] | None:
        """The clone batches a call over ``bundle`` runs in, or None for one
        simulation: a bundle without the clone column, or with one clone,
        runs as it always has."""

        entities = getattr(bundle, "entities", None)
        if entities is not None and "household" not in entities:
            return None
        household = bundle.table("household")
        column = UK_SPINE_ENGINE_BATCH_COLUMN
        if column not in household.columns:
            return None
        values = household[column]
        if values.isna().any():
            raise ValueError(
                f"{column!r} is missing for some households, so the clone each "
                "belongs to is not known."
            )
        clones = int(values.nunique())
        if self._clones is not None and clones != self._clones:
            raise ValueError(
                f"The frame carries {clones} clone(s) of the spine's {self._clones}. "
                "A call over part of the pool would read policyengine-uk's "
                "shares over that part, so no engine call runs per clone or "
                "over a sub-pool."
            )
        if clones < 2:
            return None
        return household_batches(bundle, column)

    def _materialize_in_batches(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str,
        *,
        dataset_period: int | str | None,
        batches: tuple[HouseholdBatch, ...],
        microsimulation_class: Any,
    ) -> dict[str, np.ndarray]:
        dataset_year = int(period if dataset_period is None else dataset_period)

        def build(frame: Frame) -> Any:
            simulation = microsimulation_class(
                dataset=self._build_dataset(frame, dataset_year)
            )
            validate_uc_claimant_input(simulation, frame.table("person"), period)
            return simulation

        return materialize_in_batches(
            bundle,
            variables,
            int(period),
            batches=batches,
            declaration=UK_ENGINE_POPULATION,
            policy=SPINE_BATCHED_PASS,
            dataset_year=dataset_year,
            build=build,
            entity_of=lambda name: self.variable_metadata(name).entity,
            reference_parameters=self.parameters,
        )

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
