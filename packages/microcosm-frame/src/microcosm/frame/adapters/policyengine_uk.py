"""PolicyEngine-UK adapter for the RulesEngine protocol.

The PolicyEngine-UK import is deferred until a method needs engine metadata or
simulation. Importing this module therefore keeps ``microcosm-frame`` usable
without the UK extra.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
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
    "UK_SCHEMA",
    "validate_uc_claimant_input",
]

UK_SCHEMA = EntitySchema(group_entities=("benunit", "household"))
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
        )


def _is_engine_computed(variable: Any) -> bool:
    if getattr(variable, "adds", None) or getattr(variable, "subtracts", None):
        return True
    if getattr(variable, "formula", None) is not None:
        return True
    return bool(getattr(variable, "formulas", None))


#: Supplied benefit-unit roles and the first policyengine-uk release that
#: reads each. ``is_claimant_or_partner`` and ``is_hbai_dependent_child``
#: (2.107.0) are read across the model's programmes, and ``is_uc_claimant``
#: derives from ``is_claimant_or_partner`` from that release.
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
