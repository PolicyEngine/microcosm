"""Axiom adapter for the :class:`~populace.frame.rules.RulesEngine` protocol.

The first non-PolicyEngine adapter: it wraps the Axiom rules engine's dense
(vectorized) surface — ``axiom_rules_engine.CompiledDenseProgram`` — for a
RuleSpec country module such as rulespec-be. Belgium is the pilot: there is
no policyengine-be package, so this adapter also owns the engine-native
dataset format (entity-table HDF5 mirroring the US/UK single-year layout,
read/written by :class:`AxiomEntityTableDataset`).

``axiom_rules_engine`` is imported lazily inside methods: this module (and
populace-frame itself) imports without it, and every entry point that needs
the engine raises a clear ``ImportError`` describing installation when it is
absent. The engine is not on PyPI yet, so the ``populace-frame[axiom]``
extra carries only the adapter's resolvable dependencies (pytables for the
HDF5 dataset); the engine installs from an axiom-rules-engine checkout::

    pip install <axiom-rules-engine>/python
    maturin build --release --manifest-path <axiom-rules-engine>/python-ext/Cargo.toml
    pip install <built wheel>

Entity mapping
--------------
RuleSpec scopes rules to engine entities (``Person``, ``Household``, ...);
the frame declares kernel entities (``person``, ``household``). The adapter
maps between them via ``entity_names`` (frame name -> engine name, default:
capitalize). Engine entities outside the mapping (rulespec-be also defines
``Child``, ``Vehicle``, ...) are invisible to the kernel: their variables
resolve and materialize only once a frame entity is mapped to them.

Inputs are declared by usage, not typed
---------------------------------------
The dense surface enumerates input *names* per entity but carries no input
dtypes (a RuleSpec input is any referenced-but-underived name). The frame's
column dtypes are therefore authoritative: ``materialize`` builds each batch
column from the entity table's pandas dtype (bool -> Bool, integer ->
Integer, float -> Decimal/f64). A truthiness-context input fed from a
non-bool column fails inside the engine — loudly, per the charter — and the
fix is to store the column as bool. ``variable_metadata`` resolves computed
(derived) variables only and refuses input names rather than fabricating a
dtype; exposing typed input specs is named follow-up work on the engine side
(TheAxiomFoundation/axiom-rules-engine#62).

Reform materialization (decision for populace#260)
--------------------------------------------------
The engine has no parameter-overlay API: a counterfactual is a *different
compiled module*. One adapter instance therefore wraps one parameter world,
and reform runs construct a second adapter over the reform module — no
``materialize(..., reform=...)`` protocol extension is needed for the BE
validation oracles (populace#264), which sequence behind reform modules
compiled upstream, not behind a protocol change.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.frame.bundle import Frame
from populace.frame.materialize import engine_tables
from populace.frame.rules import ExportContract
from populace.frame.schema import EntitySchema, VariableMetadata

__all__ = ["AxiomEngine", "AxiomEntityTableDataset", "BE_SCHEMA"]

#: The Belgian frame schema for the populace-be pilot: persons in households.
#: Belgian PIT is individual with household-level elements (joint assessment,
#: quotient conjugal); benefits use household/family units. Fiscal/benefit
#: units beyond the household enter as group entities when the encoded slice
#: needs them, mapped via ``entity_names``.
BE_SCHEMA = EntitySchema(group_entities=("household",))

#: Engine dtype vocabulary -> kernel dtype kind. ``judgment`` is tri-state
#: (holds / not holds / undetermined) and materializes as int8 codes
#: ``1 / -1 / 0``, so it reports as ``int``, not ``bool``.
_DTYPE_KIND_BY_ENGINE: dict[str, str] = {
    "bool": "bool",
    "integer": "int",
    "decimal": "float",
    "text": "str",
    "date": "str",
    "judgment": "int",
}

#: Engine period vocabulary (authored ``period:`` on derived rules) ->
#: kernel period semantics. Anything else (``Day``, ``Instant``, absent) is
#: point-in-time state.
_PERIOD_BY_ENGINE: dict[str, str] = {"year": "year", "month": "month"}

_WEIGHT_COLUMN_SUFFIX = "_weight"


class AxiomEngine:
    """RulesEngine adapter backed by the Axiom dense vectorized surface.

    Args:
        module: Path to the RuleSpec module to compile (e.g.
            ``rulespec-be/be/statutes/income_tax/individual/rate_scale.yaml``).
            Imports resolve through the country-monorepo layout the file
            lives in; compilation happens in-process on first engine use.
        schema: The frame-side entity structure (:data:`BE_SCHEMA` for the
            Belgian pilot).
        contract: Column-parity contract for :meth:`write_dataset` exports.
            ``None`` means an empty contract (no required/forbidden/closed
            surface checks).
        defaults: Scalar defaults broadcast onto the owning entity table for
            contract-required columns no bundle table provides.
        entity_names: Frame entity -> engine entity mapping. Defaults to
            capitalizing the frame name (``person`` -> ``Person``).
        arithmetic: ``"decimal"`` (exact, canonical) or ``"f64"`` (faster,
            floating-point rounding) — which dense execution mode
            :meth:`materialize` uses.

    The compiled dense programs (one per engine entity) and the module's
    variable metadata are loaded lazily and cached; constructing the adapter
    never imports ``axiom_rules_engine``.
    """

    def __init__(
        self,
        module: str | Path,
        schema: EntitySchema = BE_SCHEMA,
        *,
        contract: ExportContract | None = None,
        defaults: Mapping[str, object] | None = None,
        entity_names: Mapping[str, str] | None = None,
        arithmetic: str = "decimal",
    ) -> None:
        if arithmetic not in ("decimal", "f64"):
            raise ValueError(
                f"arithmetic must be 'decimal' or 'f64', got {arithmetic!r}."
            )
        self._module = Path(module)
        self._schema = schema
        self._contract = contract if contract is not None else ExportContract.empty()
        self._defaults = dict(defaults or {})
        self._entity_names = dict(entity_names) if entity_names is not None else {
            entity: entity.capitalize() for entity in schema.entities
        }
        unknown = sorted(set(self._entity_names) - set(schema.entities))
        if unknown:
            raise ValueError(
                f"entity_names maps undeclared frame entit(ies) {unknown}; "
                f"schema declares {list(schema.entities)}."
            )
        self._arithmetic = arithmetic
        self._frame_entity_by_engine = {
            engine: frame for frame, engine in self._entity_names.items()
        }
        self._programs: dict[str, Any] = {}
        self._metadata: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Variable metadata
    # ------------------------------------------------------------------

    def variable_metadata(self, name: str) -> VariableMetadata:
        """Return entity, dtype kind, and period semantics for a variable.

        Resolves computed (derived) variables from the compiled module's
        authoring metadata. Input variables are declared by usage and carry
        no dtype/period on the engine surface, so resolving one raises
        instead of fabricating metadata — the frame's own column dtypes are
        authoritative for inputs.

        Raises:
            ImportError: If ``axiom_rules_engine`` is not installed.
            ValueError: If the variable is unknown, is an input variable, or
                lives on an engine entity no frame entity is mapped to.
        """
        derived = self._derived_metadata()
        if name not in derived:
            if name in set(self.variables()):
                raise ValueError(
                    f"{name!r} is an input variable: RuleSpec inputs are "
                    "declared by usage and carry no dtype/period metadata; "
                    "the frame's column dtype is authoritative. (Typed input "
                    "specs are follow-up work on the engine surface.)"
                )
            raise ValueError(f"Unknown Axiom variable {name!r}.")
        item = derived[name]
        frame_entity = self._frame_entity_by_engine.get(item.entity)
        if frame_entity is None:
            raise ValueError(
                f"Variable {name!r} lives on engine entity {item.entity!r}, "
                f"which no frame entity is mapped to; entity_names covers "
                f"{sorted(self._frame_entity_by_engine)}."
            )
        period = (item.period or "").lower()
        return VariableMetadata(
            name=name,
            entity=frame_entity,
            dtype=_DTYPE_KIND_BY_ENGINE.get(item.dtype, "str"),
            period=_PERIOD_BY_ENGINE.get(period, "point"),
        )

    def variables(self) -> list[str]:
        """Return the input variables the engine accepts on a dataset.

        The union of the dense root inputs across every mapped engine
        entity, sorted. Computed (derived) outputs are not included.

        Raises:
            ImportError: If ``axiom_rules_engine`` is not installed.
        """
        names: set[str] = set()
        for frame_entity in self._schema.entities:
            program = self._program(frame_entity, missing_ok=True)
            if program is not None:
                names.update(program.root_inputs)
        return sorted(names)

    def entity_schema(self) -> EntitySchema:
        """Return the frame entity schema (no engine import required)."""
        return self._schema

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def materialize(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str,
    ) -> Mapping[str, np.ndarray]:
        """Compute ``variables`` for ``period`` over the bundle's tables.

        Groups the requested variables by owning entity, builds one dense
        columnar batch per entity from that entity's table (column dtypes
        decide the engine column types), and executes the compiled module.

        Args:
            bundle: A bundle whose entities match the adapter's schema.
            variables: Computed (derived) variable names.
            period: ``2025`` / ``"2025"`` for a calendar year, ``"2025-01"``
                for a month.

        Returns:
            One array per variable, row-aligned to the variable's entity
            table. Judgment variables come back as int8 codes (``1`` holds,
            ``-1`` not holds, ``0`` undetermined).

        Raises:
            ImportError: If ``axiom_rules_engine`` is not installed.
            ValueError: If the bundle's entities do not match the schema, a
                requested variable is unknown or an input, or a computed
                array's length does not match its entity table.
            NotImplementedError: If the compiled module declares relations
                (cross-entity aggregation batches; not yet wired — the BE
                pilot slice declares none).
        """
        self._require_schema(bundle)
        start, end, period_kind = _period_bounds(period)

        by_entity: dict[str, list[str]] = {}
        for name in variables:
            metadata = self.variable_metadata(name)
            by_entity.setdefault(metadata.entity, []).append(name)

        results: dict[str, np.ndarray] = {}
        for frame_entity, names in by_entity.items():
            program = self._program(frame_entity)
            if program.relations:
                raise NotImplementedError(
                    f"Module {self._module.name!r} declares dense relations "
                    f"{[item.name for item in program.relations]}; relation "
                    "batches from frame membership are not wired yet."
                )
            table = bundle.table(frame_entity)
            inputs = _batch_from_table(table, program.root_inputs)
            execute = (
                program.execute_f64 if self._arithmetic == "f64" else program.execute
            )
            outputs = execute(
                period_kind=period_kind,
                start=start,
                end=end,
                inputs=inputs,
                outputs=list(names),
            )["outputs"]
            expected = bundle.n(frame_entity)
            for name in names:
                values = np.asarray(outputs[name])
                if values.shape != (expected,):
                    raise ValueError(
                        f"Materialized variable {name!r} has shape "
                        f"{values.shape} but entity {frame_entity!r} has "
                        f"{expected} row(s)."
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
        """Write the bundle as an entity-table HDF5 dataset at ``path``.

        The format mirrors the US/UK single-year layout (one table per
        entity plus ``_time_period``) and is read back by
        :class:`AxiomEntityTableDataset`. Applies the export gate strictly:
        missing required columns (after defaults), forbidden columns,
        formula-owned columns (any stored column matching a derived rule of
        the compiled module — a persisted engine output would mask reforms),
        and — under a closed contract — unexpected non-structural columns
        all block the export; nothing is written on violation. After
        writing, the dataset is reloaded and every persisted column verified
        (round-trip check).

        Args:
            bundle: A bundle whose entities match the adapter's schema.
            path: Destination ``.h5`` path.
            period: Dataset time period (e.g. ``2025``).

        Raises:
            ImportError: If ``axiom_rules_engine`` is not installed (the
                formula-owned check reads the compiled module's metadata).
            ValueError: If ``path`` does not end in ``.h5``, the contract is
                violated (the message lists missing/forbidden/formula-owned/
                unexpected columns), or the round-trip verification fails.
        """
        output_path = Path(path)
        if output_path.suffix != ".h5":
            raise ValueError(f"path must end with '.h5', got {output_path.name!r}.")
        contract = self._contract
        tables = self._engine_tables(bundle)

        present_columns: set[str] = set()
        for frame in tables.values():
            present_columns.update(frame.columns)

        missing_required: list[str] = []
        for column in contract.required:
            if column in present_columns:
                continue
            if column in self._defaults:
                target = self._default_entity(column)
                if target in tables:
                    tables[target][column] = self._defaults[column]
                    present_columns.add(column)
                    continue
            missing_required.append(column)

        structural = self._structural_columns()
        weight_columns = {
            f"{entity}{_WEIGHT_COLUMN_SUFFIX}" for entity in bundle.weighted_entities
        }
        forbidden_present = set(contract.forbidden).intersection(present_columns)
        formula_owned_present = {
            column
            for column in present_columns - structural - weight_columns
            if column in self._derived_metadata()
        } | set(contract.formula_owned_excluded).intersection(present_columns)
        unexpected: set[str] = set()
        if contract.closed:
            allowed = (
                set(contract.required)
                | set(contract.optional)
                | structural
                | weight_columns
            )
            unexpected = present_columns - allowed

        if forbidden_present or missing_required or formula_owned_present or unexpected:
            raise ValueError(
                "Export contract violated; nothing was written. Missing "
                f"required column(s): {sorted(missing_required)}; forbidden "
                f"column(s) present: {sorted(forbidden_present)}; formula-owned "
                f"column(s) present: {sorted(formula_owned_present)}; unexpected "
                f"column(s) present: {sorted(unexpected)}."
            )

        dataset = AxiomEntityTableDataset(tables=tables, time_period=int(period))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save(output_path)
        self._verify_round_trip(tables, output_path)

    # ------------------------------------------------------------------
    # Lazy engine plumbing
    # ------------------------------------------------------------------

    def _import_engine(self) -> Any:
        try:
            import axiom_rules_engine
        except ImportError as exc:
            raise ImportError(
                "The Axiom adapter requires the 'axiom-rules-engine' Python "
                "package and its axiom_rules_engine_dense native extension. "
                "It is not on PyPI yet; install from an axiom-rules-engine "
                "checkout (pip install <checkout>/python, then build "
                "python-ext with maturin)."
            ) from exc
        return axiom_rules_engine

    def _program(self, frame_entity: str, *, missing_ok: bool = False) -> Any:
        """The compiled dense program rooted at ``frame_entity``'s engine entity.

        A module compiles per root entity; an entity with no derived rules in
        the module has no program (``missing_ok`` returns ``None`` for it —
        ``variables()`` unions across entities and must tolerate, e.g., a
        household-less PIT module).
        """
        if frame_entity not in self._schema.entities:
            raise ValueError(
                f"Unknown frame entity {frame_entity!r}; schema declares "
                f"{list(self._schema.entities)}."
            )
        engine_entity = self._entity_names[frame_entity]
        if frame_entity in self._programs:
            program = self._programs[frame_entity]
            if program is None and not missing_ok:
                raise ValueError(
                    f"Module {self._module.name!r} has no derived rules on "
                    f"engine entity {engine_entity!r}."
                )
            return program
        engine = self._import_engine()
        try:
            program = engine.CompiledDenseProgram.from_file(
                self._module, entity=engine_entity
            )
        except ValueError:
            self._programs[frame_entity] = None
            if missing_ok:
                return None
            raise ValueError(
                f"Module {self._module.name!r} has no derived rules on "
                f"engine entity {engine_entity!r}."
            ) from None
        self._programs[frame_entity] = program
        if self._metadata is None:
            self._metadata = {
                item.name: item for item in program.derived_metadata
            }
        return program

    def _derived_metadata(self) -> dict[str, Any]:
        """Name -> authoring metadata for every derived rule in the module."""
        if self._metadata is None:
            for frame_entity in self._schema.entities:
                if self._program(frame_entity, missing_ok=True) is not None:
                    break
            if self._metadata is None:
                raise ValueError(
                    f"Module {self._module.name!r} has no derived rules on "
                    f"any mapped engine entity "
                    f"({sorted(self._entity_names.values())})."
                )
        return self._metadata

    def _require_schema(self, bundle: Frame) -> None:
        if set(bundle.entities) != set(self._schema.entities):
            raise ValueError(
                f"Axiom adapter requires the schema entities "
                f"{list(self._schema.entities)}; bundle has "
                f"{list(bundle.entities)}."
            )

    def _engine_tables(self, bundle: Frame) -> dict[str, pd.DataFrame]:
        """Copy the bundle's tables and materialize typed weights as columns.

        Delegates to the shared :func:`populace.frame.materialize.engine_tables`
        (typed weights authoritative, any existing ``{entity}_weight`` column
        overwritten, never trusted), keyed to this adapter's schema order.
        """
        self._require_schema(bundle)
        tables = engine_tables(bundle)
        return {name: tables[name] for name in self._schema.entities}

    def _default_entity(self, column: str) -> str:
        """Owning table for a defaulted column, from the module's metadata.

        A column unknown to the module (or on an unmapped engine entity)
        defaults to the person table.
        """
        item = self._derived_metadata().get(column)
        if item is not None:
            frame_entity = self._frame_entity_by_engine.get(item.entity)
            if frame_entity is not None:
                return frame_entity
        return self._schema.person_entity

    def _structural_columns(self) -> set[str]:
        """Entity ids and memberships required to reconstruct the frame."""
        schema = self._schema
        return {schema.person_id_column} | {
            column
            for group in schema.group_entities
            for column in (schema.id_column(group), schema.membership_column(group))
        }

    def _verify_round_trip(
        self, tables: Mapping[str, pd.DataFrame], output_path: Path
    ) -> None:
        """Reload the written dataset and assert every column survived.

        Raises:
            ValueError: If a column is missing after reload or a numeric
                column came back non-numeric (or vice versa).
        """
        reloaded = AxiomEntityTableDataset(file_path=output_path)
        numeric = {"i", "u", "f", "b"}
        dtype_mismatches: list[str] = []
        missing: list[str] = []
        for name, source_table in tables.items():
            if len(source_table) == 0:
                continue
            reloaded_table = reloaded.table(name)
            for column in source_table.columns:
                if column not in reloaded_table.columns:
                    missing.append(f"{name}.{column}")
                    continue
                source_kind = source_table[column].dtype.kind
                reloaded_kind = reloaded_table[column].dtype.kind
                same = source_kind == reloaded_kind or (
                    source_kind in numeric and reloaded_kind in numeric
                )
                if not same:
                    dtype_mismatches.append(
                        f"{name}.{column}: {source_kind!r}->{reloaded_kind!r}"
                    )
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


class AxiomEntityTableDataset:
    """Entity-table HDF5 dataset for Axiom-engine countries.

    The on-disk layout mirrors the US/UK single-year datasets — one
    ``pandas`` table per entity plus a ``_time_period`` series — so
    ``populace.data`` loaders generalize: a registry entry points its
    ``engine_class`` here and ``load(...)`` returns this object.

    Construct from tables (to write) or from a file (to read):

        >>> AxiomEntityTableDataset(tables={"person": ..., "household": ...},
        ...                         time_period=2025).save("populace_be_2025.h5")
        >>> dataset = AxiomEntityTableDataset(file_path="populace_be_2025.h5")
        >>> dataset.person, dataset.time_period

    Attributes:
        tables: Entity name -> table.
        time_period: The dataset's period (e.g. ``2025``).
    """

    def __init__(
        self,
        *,
        tables: Mapping[str, pd.DataFrame] | None = None,
        time_period: int | None = None,
        file_path: str | Path | None = None,
    ) -> None:
        if file_path is not None:
            if tables is not None or time_period is not None:
                raise ValueError(
                    "Pass either file_path or (tables, time_period), not both."
                )
            self.tables, self.time_period = self._read(Path(file_path))
            return
        if tables is None or time_period is None:
            raise ValueError(
                "AxiomEntityTableDataset needs tables and time_period (or "
                "file_path)."
            )
        self.tables = {name: table.copy() for name, table in tables.items()}
        self.time_period = int(time_period)

    _TIME_PERIOD_KEY = "_time_period"

    def table(self, entity: str) -> pd.DataFrame:
        """Return the ``entity`` table.

        Raises:
            KeyError: If the dataset has no such table.
        """
        return self.tables[entity]

    def __getattr__(self, name: str) -> pd.DataFrame:
        tables = self.__dict__.get("tables", {})
        if name in tables:
            return tables[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}."
        )

    def save(self, file_path: str | Path) -> None:
        """Write the entity tables and time period to ``file_path``.

        Tables with zero rows are skipped (matching the US writer); the
        time period is stored under ``_time_period``.
        """
        path = Path(file_path)
        path.unlink(missing_ok=True)
        with pd.HDFStore(str(path)) as store:
            for name, table in self.tables.items():
                if len(table) > 0:
                    store.put(name, table, format="table", data_columns=True)
            store.put(
                self._TIME_PERIOD_KEY,
                pd.Series([int(self.time_period)]),
                format="table",
            )

    @classmethod
    def _read(cls, path: Path) -> tuple[dict[str, pd.DataFrame], int]:
        if not path.exists():
            raise FileNotFoundError(f"No dataset at {path}.")
        tables: dict[str, pd.DataFrame] = {}
        time_period: int | None = None
        with pd.HDFStore(str(path), mode="r") as store:
            for key in store.keys():
                name = key.lstrip("/")
                if name == cls._TIME_PERIOD_KEY:
                    time_period = int(store[key].iloc[0])
                    continue
                tables[name] = store[key]
        if time_period is None:
            raise ValueError(f"Dataset at {path} carries no {cls._TIME_PERIOD_KEY}.")
        return tables, time_period


def _period_bounds(period: int | str) -> tuple[str, str, str]:
    """Map a kernel period to dense-execution (start, end, period_kind).

    ``2025`` / ``"2025"`` -> the calendar year (rulespec-be's own test
    convention names annual periods ``calendar_year``); ``"2025-01"`` -> the
    month.
    """
    text = str(period)
    if len(text) == 4 and text.isdigit():
        return f"{text}-01-01", f"{text}-12-31", "calendar_year"
    if (
        len(text) == 7
        and text[4] == "-"
        and text[:4].isdigit()
        and text[5:].isdigit()
    ):
        year, month = int(text[:4]), int(text[5:])
        if not 1 <= month <= 12:
            raise ValueError(f"Invalid month in period {period!r}.")
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthEnd(0)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "month"
    raise ValueError(
        f"Unsupported period {period!r}; pass a year (2025, '2025') or a "
        "month ('2025-01')."
    )


def _batch_from_table(
    table: pd.DataFrame, root_inputs: Sequence[str]
) -> dict[str, np.ndarray]:
    """Build the dense input batch from an entity table.

    The table's column dtypes are authoritative: bool columns become Bool
    engine columns (truthiness-context inputs require them), integers become
    Integer, floats become the numeric column of the active arithmetic.
    Inputs the table does not carry are omitted — the engine defaults
    declared-optional inputs and errors on required ones, naming the input.

    Raises:
        ValueError: If a needed column's dtype is not bool/integer/float
            (object/string columns cannot become dense columns).
    """
    batch: dict[str, np.ndarray] = {}
    for name in root_inputs:
        if name not in table.columns:
            continue
        column = table[name]
        kind = column.dtype.kind
        if kind == "b":
            batch[name] = column.to_numpy(dtype=bool)
        elif kind in ("i", "u"):
            batch[name] = column.to_numpy(dtype=np.int64)
        elif kind == "f":
            batch[name] = column.to_numpy(dtype=np.float64)
        else:
            raise ValueError(
                f"Column {name!r} has dtype kind {kind!r}; dense inputs must "
                "be bool, integer, or float columns."
            )
    return batch
