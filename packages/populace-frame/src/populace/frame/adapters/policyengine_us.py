"""PolicyEngine-US adapter for the :class:`~populace.frame.rules.RulesEngine` protocol.

``policyengine_us`` is imported lazily inside methods: this module (and
populace-frame itself) imports without it, and every entry point that does
need it raises a clear ``ImportError`` naming the
``populace-frame[policyengine]`` extra when it is absent.

Layout contract (load-bearing for the engine)
---------------------------------------------
``USSingleYearDataset`` flattens every entity table into a single
``{column: array}`` dict; ``policyengine-core`` then reconstructs the entity
graph from PolicyEngine's id/membership conventions — exactly the frame
invariants (``person_id``, ``person_{group}_id`` on the person table,
``{group}_id`` on each group table, globally unique column names). The
adapter therefore never fabricates id or membership columns: the frame
already guarantees them. The one thing it adds is the ``household_weight``
column, materialized from the frame's typed household weights.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.frame.bundle import Frame
from populace.frame.rules import ExportContract
from populace.frame.schema import EntitySchema, VariableMetadata
from populace.frame.units import US_SCHEMA

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

# PolicyEngine ``value_type`` (a Python type) → kernel dtype kind. Enum value
# types are not listed and fall back to ``"str"`` at the call site.
_DTYPE_KIND_BY_VALUE_TYPE: dict[type, str] = {
    float: "float",
    int: "int",
    bool: "bool",
    str: "str",
}
# PolicyEngine ``definition_period`` → kernel period semantics. Anything else
# (``"eternity"``, ``"day"``) is point-in-time state.
_PERIOD_BY_DEFINITION: dict[str, str] = {"year": "year", "month": "month"}


def _has_formula(variable: Any) -> bool:
    """Return whether a PolicyEngine variable is computed by a formula.

    Input variables (read from data, what a pool must produce) have no
    formula; formula-owned outputs do. PolicyEngine exposes formulas either as
    a ``formula`` attribute or a ``formulas`` mapping keyed by start date.
    """
    if getattr(variable, "formula", None) is not None:
        return True
    formulas = getattr(variable, "formulas", None)
    return bool(formulas)


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

    def variable_metadata(self, name: str) -> VariableMetadata:
        """Return entity, dtype kind, and period semantics for a variable.

        Maps the PolicyEngine variable's ``value_type`` to a kernel dtype kind
        (``float``/``int``/``bool``/``str``; enums are reported as ``str``) and
        its ``definition_period`` to period semantics (``year``/``month``, with
        ``eternity``/``day`` reported as ``point``).

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
            ValueError: If the variable is unknown to the tax-benefit system.
        """
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
        """Return the engine's input variable names (those without a formula).

        Computed/formula-owned variables are excluded — a pool produces inputs,
        not outputs.

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
        """
        system_variables = self._tax_benefit_system().variables
        return sorted(
            name
            for name, variable in system_variables.items()
            if not _has_formula(variable)
        )

    def _entity_of(self, name: str) -> str:
        """Return the entity key a variable lives on (internal use)."""
        return self._variable(name).entity.key

    def entity_schema(self) -> EntitySchema:
        """Return the US entity schema (no engine import required)."""
        return US_SCHEMA

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def materialize(
        self,
        bundle: Frame,
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
            entity = self._entity_of(name)
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
        bundle: Frame,
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
            raise ValueError(f"path must end with '.h5', got {output_path.name!r}.")
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
                "package. Install it with 'populace-frame[policyengine]'."
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

    def _engine_tables(self, bundle: Frame) -> dict[str, pd.DataFrame]:
        """Copy the bundle's tables and materialize the household weights.

        The bundle owns the typed weights; the engine wants them as the
        ``household_weight`` column on the household table. The typed weights
        are always authoritative: any ``household_weight`` column already on
        the table is overwritten (never trusted), so a stale or leftover
        column can never override calibrated weights on export.
        """
        expected = (_PERSON_TABLE, *_GROUP_TABLES)
        if set(bundle.entities) != set(expected):
            raise ValueError(
                f"PolicyEngine-US adapter requires the US entities "
                f"{list(expected)}; bundle has {list(bundle.entities)}."
            )
        tables = {name: bundle.table(name).copy() for name in expected}
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
        dtype_mismatches: list[str] = []
        for name in (_PERSON_TABLE, *_GROUP_TABLES):
            reloaded_table = getattr(reloaded, name)
            persisted_columns.update(reloaded_table.columns)
            source_table = tables.get(name)
            if source_table is None or len(source_table) == 0:
                continue
            for column in source_table.columns:
                if column not in reloaded_table.columns:
                    continue
                source_kind = source_table[column].dtype.kind
                reloaded_kind = reloaded_table[column].dtype.kind
                # Treat the numeric kinds (int/uint/float) as compatible; a
                # round-trip that turns a number into a string (or drops a
                # column's values) is the failure this guards against.
                numeric = {"i", "u", "f"}
                same = source_kind == reloaded_kind or (
                    source_kind in numeric and reloaded_kind in numeric
                )
                if not same:
                    dtype_mismatches.append(
                        f"{name}.{column}: {source_kind!r}->{reloaded_kind!r}"
                    )

        missing = expected_columns - persisted_columns
        if missing:
            raise ValueError(
                "Export round-trip verification failed; columns absent after "
                f"reload: {sorted(missing)}."
            )
        if dtype_mismatches:
            raise ValueError(
                "Export round-trip verification failed; dtype changed on "
                f"reload: {sorted(dtype_mismatches)}."
            )
