"""PolicyEngine-UK adapter for the RulesEngine protocol.

The PolicyEngine-UK import is deferred until a method needs engine metadata or
simulation. Importing this module therefore keeps ``microcosm-frame`` usable
without the UK extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.frame.bundle import Frame
from microcosm.frame.materialize import engine_tables
from microcosm.frame.rules import ExportContract
from microcosm.frame.schema import EntitySchema, VariableMetadata

__all__ = ["PolicyEngineUKEngine", "UK_SCHEMA"]

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

    def materialize(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str,
    ) -> dict[str, np.ndarray]:
        """Compute ``variables`` for ``period`` with a UK Microsimulation."""

        microsimulation_class = self._import_policyengine_uk().Microsimulation
        dataset = self._build_dataset(bundle, period)
        simulation = microsimulation_class(dataset=dataset)
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
        """UK dataset export remains on the national-build writer in E1."""

        raise NotImplementedError(
            "PolicyEngine-UK dataset export is not implemented in the frame "
            "adapter yet; use microcosm.build.uk_runtime.national_build."
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
            time_period=str(period),
        )


def _is_engine_computed(variable: Any) -> bool:
    if getattr(variable, "adds", None) or getattr(variable, "subtracts", None):
        return True
    if getattr(variable, "formula", None) is not None:
        return True
    return bool(getattr(variable, "formulas", None))
