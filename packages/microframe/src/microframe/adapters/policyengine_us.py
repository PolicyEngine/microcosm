"""PolicyEngine-US adapter for the :class:`~microframe.rules.RulesEngine` protocol.

``policyengine_us`` is imported lazily inside methods: this module (and
microframe itself) imports without it, and every entry point that does need
it raises a clear ``ImportError`` naming the ``microframe[policyengine]``
extra when it is absent.

Layout contract (load-bearing for the engine)
---------------------------------------------
``USSingleYearDataset`` flattens every entity table into a single
``{column: array}`` dict; ``policyengine-core`` then reconstructs the entity
graph from PolicyEngine's id/membership conventions — exactly the bundle
invariants (``person_id``, ``person_{group}_id`` on the person table,
``{group}_id`` on each group table, globally unique column names). The
adapter therefore never fabricates id or membership columns: the bundle
already guarantees them. The one thing it adds is the ``household_weight``
column, materialized from the bundle's typed household weights.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microframe.bundle import WeightedBundle
from microframe.rules import ExportContract
from microframe.schema import EntitySchema
from microframe.units import US_SCHEMA

__all__ = ["PolicyEngineUSEngine"]

_PERSON_TABLE = "person"
_GROUP_TABLES: tuple[str, ...] = (
    "household",
    "tax_unit",
    "spm_unit",
    "family",
    "marital_unit",
)
_HOUSEHOLD_WEIGHT_COLUMN = "household_weight"


class PolicyEngineUSEngine:
    """RulesEngine adapter backed by ``policyengine_us``.

    Args:
        contract: Column-parity contract for :meth:`write_dataset` exports.
            ``None`` means an empty contract (no required/forbidden checks).
        defaults: Scalar defaults broadcast onto the owning entity table for
            contract-required columns no bundle table provides.

    The PolicyEngine tax-benefit system is instantiated lazily and cached on
    first metadata lookup, so constructing the adapter never imports
    ``policyengine_us``.
    """

    def __init__(
        self,
        contract: ExportContract | None = None,
        defaults: Mapping[str, object] | None = None,
    ) -> None:
        self._contract = contract if contract is not None else ExportContract.empty()
        self._defaults = dict(defaults or {})
        self._system: Any = None

    # ------------------------------------------------------------------
    # Variable metadata
    # ------------------------------------------------------------------

    def variable_entity(self, name: str) -> str:
        """Return the PolicyEngine entity key a variable lives on.

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
            ValueError: If the variable is unknown to the tax-benefit system.
        """
        return self._variable(name).entity.key

    def variable_dtype(self, name: str) -> type:
        """Return the PolicyEngine value type of a variable (e.g. ``float``).

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
            ValueError: If the variable is unknown to the tax-benefit system.
        """
        return self._variable(name).value_type

    def entity_schema(self) -> EntitySchema:
        """Return the US entity schema (no engine import required)."""
        return US_SCHEMA

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def materialize(
        self,
        bundle: WeightedBundle,
        variables: Sequence[str],
        period: int | str,
    ) -> Mapping[str, np.ndarray]:
        """Compute ``variables`` for ``period`` with a Microsimulation.

        Builds a ``USSingleYearDataset`` from the bundle's entity tables
        (with the bundle's household weights as ``household_weight``), runs a
        ``Microsimulation`` over it, and calculates each variable.

        Args:
            bundle: A US-schema bundle.
            variables: PolicyEngine variable names to compute.
            period: Period to compute for (e.g. ``2026``).

        Returns:
            One array per variable, row-aligned to the variable's entity
            table in the bundle.

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
            ValueError: If a computed array's length does not match its
                entity table (a structural mismatch the kernel refuses to
                pass through).
        """
        microsimulation_class = self._import_policyengine_us().Microsimulation
        tables = self._engine_tables(bundle)
        dataset = self._build_dataset(tables, period)
        simulation = microsimulation_class(dataset=dataset)
        results: dict[str, np.ndarray] = {}
        for name in variables:
            entity = self.variable_entity(name)
            values = np.asarray(simulation.calculate(name, period=period))
            expected = bundle.n(entity)
            if values.shape != (expected,):
                raise ValueError(
                    f"Materialized variable {name!r} has shape {values.shape} "
                    f"but entity {entity!r} has {expected} row(s)."
                )
            results[name] = values
        return results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_contract(self) -> ExportContract:
        """Return the column-parity contract exports are gated against."""
        return self._contract

    def write_dataset(
        self,
        bundle: WeightedBundle,
        path: str | Path,
        period: int | str,
    ) -> None:
        """Write the bundle as a ``USSingleYearDataset`` HDF5 file.

        Applies the export gate: forbidden and formula-owned columns are
        dropped, defaults are broadcast onto the owning entity table for
        required columns no table provides, and a dataset with missing
        required columns is never written. After writing, the dataset is
        reloaded and every persisted column verified (round-trip check).

        Args:
            bundle: A US-schema bundle.
            path: Destination ``.h5`` path.
            period: Dataset time period (e.g. ``2026``).

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
            ValueError: If ``path`` does not end in ``.h5``, the contract is
                violated (the message lists the missing/forbidden columns),
                or the round-trip verification fails.
        """
        output_path = Path(path)
        if output_path.suffix != ".h5":
            raise ValueError(
                f"path must end with '.h5', got {output_path.name!r}."
            )
        contract = self._contract
        tables = self._engine_tables(bundle)

        forbidden = set(contract.forbidden)
        drop_on_sight = forbidden | set(contract.formula_owned_excluded)
        dropped: set[str] = set()
        forbidden_present: set[str] = set()
        for name, frame in tables.items():
            present_drops = drop_on_sight.intersection(frame.columns)
            if present_drops:
                tables[name] = frame.drop(columns=sorted(present_drops))
            dropped.update(present_drops)
            forbidden_present.update(forbidden.intersection(present_drops))

        present_columns: set[str] = set()
        for frame in tables.values():
            present_columns.update(frame.columns)

        defaulted: set[str] = set()
        missing_required: list[str] = []
        for column in contract.required:
            if column in present_columns:
                continue
            if column in self._defaults:
                target = self._default_entity(column)
                if target in tables:
                    tables[target][column] = self._defaults[column]
                    present_columns.add(column)
                    defaulted.add(column)
                    continue
            missing_required.append(column)

        if forbidden_present or missing_required:
            raise ValueError(
                "Export contract violated; nothing was written. Missing "
                f"required column(s): {sorted(missing_required)}; forbidden "
                f"column(s) present: {sorted(forbidden_present)}."
            )

        self._write_and_verify(tables, period=int(period), output_path=output_path)

    # ------------------------------------------------------------------
    # Lazy engine plumbing
    # ------------------------------------------------------------------

    def _import_policyengine_us(self) -> Any:
        try:
            import policyengine_us
        except ImportError as exc:
            raise ImportError(
                "The PolicyEngine-US adapter requires the 'policyengine-us' "
                "package. Install it with 'microframe[policyengine]'."
            ) from exc
        return policyengine_us

    def _tax_benefit_system(self) -> Any:
        if self._system is None:
            self._system = self._import_policyengine_us().CountryTaxBenefitSystem()
        return self._system

    def _variable(self, name: str) -> Any:
        variables = self._tax_benefit_system().variables
        if name not in variables:
            raise ValueError(f"Unknown PolicyEngine-US variable {name!r}.")
        return variables[name]

    def _engine_tables(self, bundle: WeightedBundle) -> dict[str, pd.DataFrame]:
        """Copy the bundle's tables and materialize the household weights.

        The bundle owns the typed weights; the engine wants them as the
        ``household_weight`` column on the household table.
        """
        expected = (_PERSON_TABLE, *_GROUP_TABLES)
        if set(bundle.entities) != set(expected):
            raise ValueError(
                f"PolicyEngine-US adapter requires the US entities "
                f"{list(expected)}; bundle has {list(bundle.entities)}."
            )
        tables = {name: bundle.table(name).copy() for name in expected}
        if _HOUSEHOLD_WEIGHT_COLUMN not in tables["household"].columns:
            tables["household"][_HOUSEHOLD_WEIGHT_COLUMN] = bundle.weights_for(
                "household"
            ).values
        return tables

    def _build_dataset(
        self, tables: Mapping[str, pd.DataFrame], period: int | str
    ) -> Any:
        from policyengine_us.data import USSingleYearDataset

        return USSingleYearDataset(
            person=tables[_PERSON_TABLE].copy(),
            household=tables["household"].copy(),
            tax_unit=tables["tax_unit"].copy(),
            spm_unit=tables["spm_unit"].copy(),
            family=tables["family"].copy(),
            marital_unit=tables["marital_unit"].copy(),
            time_period=int(period),
        )

    def _default_entity(self, column: str) -> str:
        """Owning table for a defaulted column, from PolicyEngine metadata.

        A column unknown to the tax-benefit system defaults to the person
        table.
        """
        variables = self._tax_benefit_system().variables
        if column in variables:
            return variables[column].entity.key
        return _PERSON_TABLE

    def _write_and_verify(
        self,
        tables: Mapping[str, pd.DataFrame],
        *,
        period: int,
        output_path: Path,
    ) -> None:
        """Persist tables as a ``USSingleYearDataset`` and verify the round-trip.

        Saves the dataset, reloads it, and asserts every column from a
        non-empty table survived (``.save`` only writes tables with rows).

        Raises:
            ValueError: If a column expected after reload is missing.
        """
        from policyengine_us.data import USSingleYearDataset

        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = self._build_dataset(tables, period)
        dataset.save(str(output_path))

        expected_columns: set[str] = set()
        for frame in tables.values():
            if len(frame) > 0:
                expected_columns.update(frame.columns)

        reloaded = USSingleYearDataset(file_path=str(output_path))
        persisted_columns: set[str] = set()
        for name in (_PERSON_TABLE, *_GROUP_TABLES):
            persisted_columns.update(getattr(reloaded, name).columns)

        missing = expected_columns - persisted_columns
        if missing:
            raise ValueError(
                "Export round-trip verification failed; columns absent after "
                f"reload: {sorted(missing)}."
            )
