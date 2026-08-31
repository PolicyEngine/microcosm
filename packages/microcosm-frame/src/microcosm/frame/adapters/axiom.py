"""Axiom adapter for the :class:`~microcosm.frame.rules.RulesEngine` protocol.

The first non-PolicyEngine adapter: it wraps the Axiom rules engine's dense
(vectorized) surface — ``axiom_rules_engine.CompiledDenseProgram`` — for a
RuleSpec country module such as rulespec-be. Belgium is the pilot: there is
no policyengine-be package, so this adapter also owns the engine-native
dataset format (entity-table HDF5 mirroring the US/UK single-year layout,
read/written by :class:`AxiomEntityTableDataset`).

``axiom_rules_engine`` is imported lazily inside methods: this module (and
microcosm-frame itself) imports without it, and every entry point that needs
the engine raises a clear ``ImportError`` describing installation when it is
absent. The engine is not on PyPI yet, so the ``microcosm-frame[axiom]``
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

RuleSpec authority roots
------------------------
Filesystem compilation requires a non-empty, explicit sequence of canonical
``rulespec-<country>`` roots. The adapter forwards exactly the caller-supplied
roots to Axiom; it never searches the working directory, environment, module
ancestors, or sibling checkouts. Axiom remains the authority for validating
that each root is absolute, canonical, and structurally valid when the module
is compiled lazily.

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

Reform materialization (decision for microcosm#260)
--------------------------------------------------
The engine has no parameter-overlay API: a counterfactual is a *different
compiled module*. One adapter instance therefore wraps one parameter world,
and reform runs construct a second adapter over the reform module — no
``materialize(..., reform=...)`` protocol extension is needed for the BE
validation oracles (microcosm#264), which sequence behind reform modules
compiled upstream, not behind a protocol change.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame.bundle import Frame
from microcosm.frame.materialize import engine_tables, put_frame_table, read_frame_table
from microcosm.frame.rules import ExportContract
from microcosm.frame.schema import EntitySchema, VariableMetadata

__all__ = [
    "AxiomEngine",
    "AxiomEntityTableDataset",
    "AxiomRelationBinding",
    "BE_SCHEMA",
    "verify_axiom_materialization_receipt",
]

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


@dataclass(frozen=True)
class AxiomRelationBinding:
    """Explicit frame-side edge projection for one Axiom dense relation.

    Axiom's dense schema exposes a relation key and its runtime slots, but it
    deliberately does not claim which Microcosm entity tables those slots
    represent. Callers therefore bind both sides and the exact edge columns.
    The adapter never infers orientation from a relation name such as
    ``member_of_household``.

    ``edge_table`` may be either an entity table or a provided link table. One
    row names a current id and a related id. This represents both common group
    directions without duplicating entities: household aggregation uses the
    person table with ``person_household_id -> person_id``; person lookup of a
    household value uses the same table with ``person_id ->
    person_household_id``. Repeated related ids and current ids with no edge
    are valid dense-relation shapes.

    Attributes:
        current_entity: Frame entity on which the dense program executes.
        related_entity: Frame entity supplying the relation's rows and inputs.
        edge_table: Entity or link table carrying the relation edges.
        edge_current_id_column: Edge column containing ``current_entity`` ids.
        edge_related_id_column: Edge column containing ``related_entity`` ids.
    """

    current_entity: str
    related_entity: str
    edge_table: str
    edge_current_id_column: str
    edge_related_id_column: str

    def __post_init__(self) -> None:
        for name in (
            "current_entity",
            "related_entity",
            "edge_table",
            "edge_current_id_column",
            "edge_related_id_column",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"AxiomRelationBinding.{name} must be non-empty.")


class AxiomEngine:
    """RulesEngine adapter backed by the Axiom dense vectorized surface.

    Args:
        module: Path to the RuleSpec module to compile (e.g.
            ``rulespec-be/be/statutes/income_tax/individual/rate_scale.yaml``).
            Compilation happens in-process on first engine use.
        rulespec_roots: Non-empty explicit sequence of canonical
            ``rulespec-<country>`` checkout roots. These are forwarded exactly
            to Axiom as its filesystem authority boundary; the adapter never
            infers roots from the module, environment, or working directory.
        schema: The frame-side entity structure (:data:`BE_SCHEMA` for the
            Belgian pilot).
        contract: Column-parity contract for :meth:`write_dataset` exports.
            ``None`` means an empty contract (no required/forbidden/closed
            surface checks).
        defaults: Scalar defaults broadcast onto the owning entity table for
            contract-required columns no bundle table provides.
        entity_names: Frame entity -> engine entity mapping. Defaults to
            capitalizing the frame name (``person`` -> ``Person``).
        relation_bindings: Explicit frame orientation keyed by the exact Axiom
            dense relation key. Each value names the current and related frame
            entities plus an explicit edge table and its current/related id
            columns. The set must match every executed program's distinct
            relation-batch keys.
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
        rulespec_roots: Sequence[str | Path],
        contract: ExportContract | None = None,
        defaults: Mapping[str, object] | None = None,
        entity_names: Mapping[str, str] | None = None,
        relation_bindings: Mapping[str, AxiomRelationBinding] | None = None,
        arithmetic: str = "decimal",
    ) -> None:
        if arithmetic not in ("decimal", "f64"):
            raise ValueError(
                f"arithmetic must be 'decimal' or 'f64', got {arithmetic!r}."
            )
        if isinstance(rulespec_roots, (str, Path)):
            raise TypeError(
                "rulespec_roots must be a non-empty sequence of explicit "
                "rulespec-<country> root paths, not a scalar path."
            )
        roots = tuple(rulespec_roots)
        if not roots:
            raise ValueError(
                "at least one explicit rulespec-<country> root is required"
            )
        if not all(isinstance(root, (str, Path)) for root in roots):
            raise TypeError(
                "rulespec_roots entries must each be a str or pathlib.Path."
            )
        self._module = Path(module)
        self._rulespec_roots = tuple(Path(root) for root in roots)
        self._schema = schema
        self._contract = contract if contract is not None else ExportContract.empty()
        self._defaults = dict(defaults or {})
        self._entity_names = (
            dict(entity_names)
            if entity_names is not None
            else {entity: entity.capitalize() for entity in schema.entities}
        )
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
        if relation_bindings is not None and not isinstance(relation_bindings, Mapping):
            raise TypeError(
                "relation_bindings must map exact Axiom dense relation keys to "
                "AxiomRelationBinding values."
            )
        self._relation_bindings: dict[str, AxiomRelationBinding] = {}
        for key, binding in (relation_bindings or {}).items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("relation_bindings keys must be non-empty strings.")
            if not isinstance(binding, AxiomRelationBinding):
                raise TypeError(
                    f"relation_bindings[{key!r}] must be an AxiomRelationBinding."
                )
            for side in (binding.current_entity, binding.related_entity):
                if side not in schema.entities:
                    raise ValueError(
                        f"Relation binding {key!r} names undeclared frame entity "
                        f"{side!r}; schema declares {list(schema.entities)}."
                    )
            declared_edge_tables = set(schema.entities) | {
                link.name for link in schema.links
            }
            if binding.edge_table not in declared_edge_tables:
                raise ValueError(
                    f"Relation binding {key!r} names undeclared edge table "
                    f"{binding.edge_table!r}; schema declares entity tables "
                    f"{list(schema.entities)} and link tables "
                    f"{[link.name for link in schema.links]}."
                )
            self._relation_bindings[key] = binding
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

        The union of dense root inputs and relation-side inputs across every
        mapped engine entity, sorted. Computed (derived) outputs are not
        included.

        Raises:
            ImportError: If ``axiom_rules_engine`` is not installed.
        """
        names: set[str] = set()
        for frame_entity in self._schema.entities:
            program = self._program(frame_entity, missing_ok=True)
            if program is not None:
                names.update(program.root_inputs)
                for relation in program.relations:
                    names.update(relation.related_inputs)
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
                array's length does not match its entity table, or a declared
                dense relation lacks an exact explicit frame-side binding.
        """
        results, _ = self._materialize_with_receipt(bundle, variables, period)
        return results

    def materialize_with_receipt(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str,
    ) -> tuple[Mapping[str, np.ndarray], Mapping[str, object]]:
        """Materialize values and return the exact dense execution receipt.

        The ordinary :meth:`materialize` protocol remains unchanged. Policy
        outputs used as calibration measures call this surface so an outer
        signed build manifest can authenticate every supplied root input,
        relation edge and ordered related input, requested output, entity-row
        identity, period, arithmetic mode, and explicit frame/engine mapping.

        The receipt hashes detect drift; authenticity belongs to the signed
        outer build manifest that carries the complete receipt.
        """
        return self._materialize_with_receipt(bundle, variables, period)

    def _materialize_with_receipt(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        bundle.revalidate()
        self._require_schema(bundle)
        start, end, period_kind = _period_bounds(period)
        requested = tuple(variables)
        if not requested:
            raise ValueError("Axiom materialization requires at least one variable.")
        if len(set(requested)) != len(requested):
            raise ValueError("Axiom materialization variables must be unique.")

        by_entity: dict[str, list[str]] = {}
        metadata_by_name: dict[str, VariableMetadata] = {}
        for name in requested:
            metadata = self.variable_metadata(name)
            metadata_by_name[name] = metadata
            by_entity.setdefault(metadata.entity, []).append(name)

        results: dict[str, np.ndarray] = {}
        entity_receipts: dict[str, object] = {}
        for frame_entity in sorted(by_entity):
            names = sorted(by_entity[frame_entity])
            program = self._program(frame_entity)
            expected_root = self._entity_names[frame_entity]
            if program.root_entity != expected_root:
                raise ValueError(
                    f"Dense program root {program.root_entity!r} does not match "
                    f"frame entity {frame_entity!r}'s explicit engine mapping "
                    f"{expected_root!r}."
                )
            table = bundle.table(frame_entity)
            current_id_column = self._schema.entity_id_column(frame_entity)
            current_ids = _integer_id_vector(
                table[current_id_column],
                f"{frame_entity}.{current_id_column}",
                unique=True,
            )
            declared_root_inputs = tuple(program.root_inputs)
            if len(set(declared_root_inputs)) != len(declared_root_inputs):
                raise ValueError(
                    f"Dense program for {frame_entity!r} repeats root input names."
                )
            inputs = _batch_from_table(table, declared_root_inputs)
            relations, relation_receipts = self._relation_batches(
                bundle,
                frame_entity=frame_entity,
                program=program,
                current_ids=current_ids,
            )
            execute = (
                program.execute_f64 if self._arithmetic == "f64" else program.execute
            )
            outputs = execute(
                period_kind=period_kind,
                start=start,
                end=end,
                inputs=inputs,
                relations=relations or None,
                outputs=list(names),
            )["outputs"]
            expected = bundle.n(frame_entity)
            output_receipts: dict[str, object] = {}
            for name in names:
                values = np.asarray(outputs[name])
                if values.shape != (expected,):
                    raise ValueError(
                        f"Materialized variable {name!r} has shape "
                        f"{values.shape} but entity {frame_entity!r} has "
                        f"{expected} row(s)."
                    )
                results[name] = values
                metadata = metadata_by_name[name]
                authored = (self._metadata or {}).get(name)
                output_receipts[name] = {
                    "declared_engine_dtype": (
                        authored.dtype if authored is not None else metadata.dtype
                    ),
                    "declared_kernel_dtype": metadata.dtype,
                    "declared_period": metadata.period,
                    "values": _typed_array_identity(values),
                }

            entity_receipts[frame_entity] = {
                "frame_entity": frame_entity,
                "engine_entity": program.root_entity,
                "current_id_column": current_id_column,
                "current_ids": _array_identity(current_ids),
                "declared_root_inputs": list(declared_root_inputs),
                "provided_root_inputs": {
                    name: _array_identity(inputs[name]) for name in sorted(inputs)
                },
                "relations": relation_receipts,
                "requested_outputs": output_receipts,
            }

        period_receipt = {"kind": period_kind, "start": start, "end": end}
        input_projection = _input_projection_receipt(
            period=period_receipt,
            arithmetic=self._arithmetic,
            entities=entity_receipts,
        )
        receipt: dict[str, object] = {
            "schema_version": 2,
            "receipt_kind": "axiom_dense_materialization",
            "period": period_receipt,
            "arithmetic": self._arithmetic,
            "entities": entity_receipts,
            "input_frame_sha256": _canonical_digest(input_projection),
        }
        return results, {
            **receipt,
            "receipt_sha256": _canonical_digest(receipt),
        }

    def _relation_batches(
        self,
        bundle: Frame,
        *,
        frame_entity: str,
        program: Any,
        current_ids: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, object]]:
        """Build exact dense relation batches and their drift receipt."""
        declared = _group_relation_declarations(program.relations)
        configured = {
            key: binding
            for key, binding in self._relation_bindings.items()
            if binding.current_entity == frame_entity
        }
        missing = sorted(set(declared) - set(configured))
        extra = sorted(set(configured) - set(declared))
        if missing or extra:
            raise ValueError(
                f"Dense relations for frame entity {frame_entity!r} require an "
                "exact explicit binding set; "
                f"missing={missing}, extra={extra}."
            )
        if not declared:
            return {}, {}

        engine = self._import_engine()
        batches: dict[str, Any] = {}
        receipts: dict[str, object] = {}

        for key in sorted(declared):
            binding = configured[key]
            offsets, related_inputs, relation_receipt = _relation_projection(
                bundle,
                relation_key=key,
                declarations=declared[key],
                binding=binding,
                current_ids=current_ids,
            )

            batches[key] = engine.DenseRelationBatch(
                offsets=offsets,
                inputs=related_inputs,
            )
            receipts[key] = {
                **relation_receipt,
                "receipt_sha256": _canonical_digest(relation_receipt),
            }
        return batches, receipts

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
                self._module,
                rulespec_roots=self._rulespec_roots,
                entity=engine_entity,
            )
        except ValueError as exc:
            missing_entity = (
                "dense compilation could not find derived outputs for entity "
                f"`{engine_entity}`"
            )
            if str(exc) != missing_entity:
                # The native surface currently reports both an entity with no
                # derived outputs and authority-root/module validation failures
                # as ValueError. Only the exact former condition is optional;
                # root or module failures must propagate instead of becoming a
                # silently absent program under missing_ok=True.
                raise
            self._programs[frame_entity] = None
            if missing_ok:
                return None
            raise ValueError(
                f"Module {self._module.name!r} has no derived rules on "
                f"engine entity {engine_entity!r}."
            ) from None
        self._programs[frame_entity] = program
        if self._metadata is None:
            self._metadata = {item.name: item for item in program.derived_metadata}
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

        Delegates to the shared :func:`microcosm.frame.materialize.engine_tables`
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
    ``microcosm.data`` loaders generalize: a registry entry points its
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
                "AxiomEntityTableDataset needs tables and time_period (or file_path)."
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
                    put_frame_table(
                        store,
                        name,
                        table,
                        preferred_format="table",
                        data_columns=True,
                    )
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
                tables[name] = read_frame_table(store, key)
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
    if len(text) == 7 and text[4] == "-" and text[:4].isdigit() and text[5:].isdigit():
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


def _group_relation_declarations(
    relations: Sequence[Any],
) -> dict[str, list[dict[str, object]]]:
    """Group Axiom relation schemas by their shared runtime batch key.

    Filtered or composed derived relations legitimately produce more than one
    schema declaration backed by the same raw dense-relation batch. The batch
    must be supplied once with the union of all declaration inputs.
    """

    grouped: dict[str, list[dict[str, object]]] = {}
    for relation in relations:
        key = relation.key
        if not isinstance(key, str) or not key:
            raise ValueError("Dense relation keys must be non-empty strings.")
        related_inputs = tuple(relation.related_inputs)
        if len(set(related_inputs)) != len(related_inputs) or any(
            not isinstance(name, str) or not name for name in related_inputs
        ):
            raise ValueError(
                f"Dense relation {key!r} has invalid/repeated related inputs."
            )
        grouped.setdefault(key, []).append(
            {
                "relation_key": key,
                "relation_name": relation.name,
                "current_slot": relation.current_slot,
                "related_slot": relation.related_slot,
                "related_inputs": sorted(related_inputs),
            }
        )
    for declarations in grouped.values():
        declarations.sort(key=_canonical_json)
    return grouped


def _edge_table(bundle: Frame, binding: AxiomRelationBinding) -> pd.DataFrame:
    if binding.edge_table in bundle.entities:
        return bundle.table(binding.edge_table)
    if binding.edge_table in bundle.links:
        return bundle.link(binding.edge_table)
    raise ValueError(
        f"Relation edge table {binding.edge_table!r} is not present on the frame; "
        f"entity tables={list(bundle.entities)}, provided links={list(bundle.links)}."
    )


def _relation_projection(
    bundle: Frame,
    *,
    relation_key: str,
    declarations: Sequence[Mapping[str, object]],
    binding: AxiomRelationBinding,
    current_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Project one explicit edge table into an exact Axiom dense batch."""

    if binding.current_entity not in bundle.entities:
        raise ValueError(
            f"Relation binding {relation_key!r} current entity "
            f"{binding.current_entity!r} is absent from the frame."
        )
    related_table = bundle.table(binding.related_entity)
    related_id_column = bundle.schema.entity_id_column(binding.related_entity)
    related_entity_ids = _integer_id_vector(
        related_table[related_id_column],
        f"{binding.related_entity}.{related_id_column}",
        unique=True,
    )
    edge_table = _edge_table(bundle, binding)
    for column in (
        binding.edge_current_id_column,
        binding.edge_related_id_column,
    ):
        if column not in edge_table.columns:
            raise ValueError(
                f"Relation binding {relation_key!r} requires edge column "
                f"{column!r} on table {binding.edge_table!r}."
            )
    edge_current_ids = _integer_id_vector(
        edge_table[binding.edge_current_id_column],
        f"{binding.edge_table}.{binding.edge_current_id_column}",
        unique=False,
    )
    edge_related_ids = _integer_id_vector(
        edge_table[binding.edge_related_id_column],
        f"{binding.edge_table}.{binding.edge_related_id_column}",
        unique=False,
    )
    if edge_current_ids.shape != edge_related_ids.shape:
        raise ValueError(
            f"Relation binding {relation_key!r} edge id columns do not align."
        )

    current_positions = {int(value): i for i, value in enumerate(current_ids)}
    related_positions = {int(value): i for i, value in enumerate(related_entity_ids)}
    unknown_current = sorted(
        set(int(value) for value in edge_current_ids) - current_positions.keys()
    )
    if unknown_current:
        raise ValueError(
            f"Relation binding {relation_key!r} edge references current ids absent "
            f"from {binding.current_entity!r}: {unknown_current[:5]}."
        )
    unknown_related = sorted(
        set(int(value) for value in edge_related_ids) - related_positions.keys()
    )
    if unknown_related:
        raise ValueError(
            f"Relation binding {relation_key!r} edge references related ids absent "
            f"from {binding.related_entity!r}: {unknown_related[:5]}."
        )

    positions = np.fromiter(
        (current_positions[int(value)] for value in edge_current_ids),
        dtype=np.int64,
        count=len(edge_current_ids),
    )
    counts = np.bincount(positions, minlength=len(current_ids))
    order = np.argsort(positions, kind="stable")
    offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    ordered_edge_current_ids = edge_current_ids[order]
    ordered_edge_related_ids = edge_related_ids[order]
    related_row_order = np.fromiter(
        (related_positions[int(value)] for value in ordered_edge_related_ids),
        dtype=np.int64,
        count=len(ordered_edge_related_ids),
    )
    ordered_related_table = related_table.iloc[related_row_order]

    declared_related_inputs = sorted(
        {
            name
            for declaration in declarations
            for name in _declaration_inputs(declaration, relation_key)
        }
    )
    related_inputs = _batch_from_table(ordered_related_table, declared_related_inputs)
    for name, values in related_inputs.items():
        if np.asarray(values).shape != (len(ordered_edge_related_ids),):
            raise ValueError(
                f"Relation binding {relation_key!r} input {name!r} is not "
                "row-aligned to its ordered edge rows."
            )

    normalized_declarations = [dict(item) for item in declarations]
    normalized_declarations.sort(key=_canonical_json)
    relation_receipt: dict[str, object] = {
        "schema_version": 2,
        "relation_key": relation_key,
        "declarations": normalized_declarations,
        "binding": {
            "current_entity": binding.current_entity,
            "related_entity": binding.related_entity,
            "edge_table": binding.edge_table,
            "edge_current_id_column": binding.edge_current_id_column,
            "edge_related_id_column": binding.edge_related_id_column,
        },
        "related_id_column": related_id_column,
        "source_related_entity_ids": _array_identity(related_entity_ids),
        "source_edge_current_ids": _array_identity(edge_current_ids),
        "source_edge_related_ids": _array_identity(edge_related_ids),
        "ordered_edge_current_ids": _array_identity(ordered_edge_current_ids),
        "ordered_edge_related_ids": _array_identity(ordered_edge_related_ids),
        "offsets": _array_identity(offsets),
        "declared_related_inputs": declared_related_inputs,
        "provided_related_inputs": {
            name: _array_identity(related_inputs[name])
            for name in sorted(related_inputs)
        },
    }
    return offsets, related_inputs, relation_receipt


def _declaration_inputs(
    declaration: Mapping[str, object], relation_key: str
) -> list[str]:
    value = declaration.get("related_inputs")
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(
            f"Dense relation declaration {relation_key!r} has invalid inputs."
        )
    if value != sorted(set(value)):
        raise ValueError(
            f"Dense relation declaration {relation_key!r} inputs must be "
            "unique and sorted."
        )
    return value


def _input_projection_receipt(
    *,
    period: Mapping[str, object],
    arithmetic: str,
    entities: Mapping[str, object],
) -> dict[str, object]:
    projected_entities: dict[str, object] = {}
    for entity, raw in entities.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"Materialization entity {entity!r} must be an object.")
        projected_entities[entity] = {
            key: value for key, value in raw.items() if key != "requested_outputs"
        }
    return {
        "schema_version": 2,
        "receipt_kind": "axiom_dense_input_projection",
        "period": dict(period),
        "arithmetic": arithmetic,
        "entities": projected_entities,
    }


def _integer_id_vector(
    values: pd.Series,
    label: str,
    *,
    unique: bool,
) -> np.ndarray:
    """Return a canonical signed-int64 id vector for a relation receipt."""
    if values.isna().any():
        raise ValueError(f"{label} must not contain missing ids.")
    kind = values.dtype.kind
    if kind not in ("i", "u"):
        raise ValueError(
            f"{label} must use an integer dtype for Axiom relation binding, "
            f"got dtype kind {kind!r}."
        )
    raw = values.to_numpy()
    if kind == "u" and raw.size and raw.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{label} contains an id outside signed int64 range.")
    result = raw.astype("<i8", copy=False)
    if unique and len(np.unique(result)) != len(result):
        raise ValueError(f"{label} must contain unique ids.")
    return result


def _array_identity(values: np.ndarray) -> dict[str, object]:
    """Canonical identity for a numeric Axiom input or structural vector."""
    return _typed_array_identity(values)


def _typed_array_identity(values: object) -> dict[str, object]:
    """Hash a one-dimensional native value vector without object pointers.

    Numeric values use explicit little-endian bytes. Text/date values use
    canonical JSON UTF-8, including when NumPy represents them as ``object``.
    The semantic dtype name is recorded separately from the canonical storage
    encoding so receipts are stable across native byte order.
    """

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            f"Axiom receipt vectors must be one-dimensional, got {array.shape}."
        )
    kind = array.dtype.kind
    if kind in ("b", "i", "u", "f"):
        dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
        encoding = "little_endian_raw_v1"
        semantic_dtype = array.dtype.name
        payload = canonical.tobytes()
        storage_dtype = canonical.dtype.str
    elif kind in ("U", "S", "O"):
        items = array.tolist()
        if any(not isinstance(item, str) for item in items):
            raise ValueError(
                "Axiom text/date receipt vectors must contain only strings."
            )
        encoding = "canonical_json_utf8_v1"
        semantic_dtype = "string"
        storage_dtype = "utf8"
        payload = json.dumps(
            items,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    else:
        raise ValueError(
            f"Axiom receipt vectors cannot canonicalize dtype {array.dtype!s}."
        )
    header = {
        "dtype": semantic_dtype,
        "storage_dtype": storage_dtype,
        "encoding": encoding,
        "shape": list(array.shape),
    }
    return {
        **header,
        "sha256": hashlib.sha256(
            _canonical_json(header).encode("utf-8") + b"\n" + payload
        ).hexdigest(),
    }


def verify_axiom_materialization_receipt(
    frame: Frame,
    receipt: Mapping[str, object],
) -> None:
    """Verify a schema-v2 Axiom receipt against the live input frame.

    This verifier does not import or execute Axiom. It revalidates the Frame,
    authenticates the closed receipt/hash structure, and reconstructs every
    exact root-input and relation-edge projection from the live tables. Output
    identities are structurally and cryptographically bound by the receipt;
    the signed outer manifest supplies authenticity for those hashes and the
    RuleSpec/runtime parameter world.
    """

    frame.revalidate()
    top = _exact_mapping(
        receipt,
        {
            "schema_version",
            "receipt_kind",
            "period",
            "arithmetic",
            "entities",
            "input_frame_sha256",
            "receipt_sha256",
        },
        "Axiom materialization receipt",
    )
    if top["schema_version"] != 2:
        raise ValueError("Unsupported Axiom materialization receipt version.")
    if top["receipt_kind"] != "axiom_dense_materialization":
        raise ValueError("Invalid Axiom materialization receipt kind.")
    _require_sha256(top["input_frame_sha256"], "input_frame_sha256")
    _require_sha256(top["receipt_sha256"], "receipt_sha256")
    unsigned = {key: value for key, value in top.items() if key != "receipt_sha256"}
    if _canonical_digest(unsigned) != top["receipt_sha256"]:
        raise ValueError("Axiom materialization receipt digest differs.")
    arithmetic = top["arithmetic"]
    if arithmetic not in ("decimal", "f64"):
        raise ValueError("Invalid Axiom materialization arithmetic mode.")
    period = _verify_period_receipt(top["period"])

    raw_entities = top["entities"]
    if not isinstance(raw_entities, Mapping) or not raw_entities:
        raise ValueError("Axiom materialization receipt needs executed entities.")
    recomputed_entities: dict[str, object] = {}
    for entity_key in sorted(raw_entities):
        if not isinstance(entity_key, str) or not entity_key:
            raise ValueError("Axiom receipt entity keys must be non-empty strings.")
        entity = _verify_entity_receipt(
            frame,
            frame_entity=entity_key,
            value=raw_entities[entity_key],
        )
        recomputed_entities[entity_key] = entity

    input_projection = _input_projection_receipt(
        period=period,
        arithmetic=arithmetic,
        entities=recomputed_entities,
    )
    if _canonical_digest(input_projection) != top["input_frame_sha256"]:
        raise ValueError("Axiom materialization input-frame projection differs.")


def _verify_entity_receipt(
    frame: Frame,
    *,
    frame_entity: str,
    value: object,
) -> dict[str, object]:
    entity = _exact_mapping(
        value,
        {
            "frame_entity",
            "engine_entity",
            "current_id_column",
            "current_ids",
            "declared_root_inputs",
            "provided_root_inputs",
            "relations",
            "requested_outputs",
        },
        f"Axiom entity receipt {frame_entity!r}",
    )
    if entity["frame_entity"] != frame_entity or frame_entity not in frame.entities:
        raise ValueError(f"Axiom receipt frame entity {frame_entity!r} differs.")
    if not isinstance(entity["engine_entity"], str) or not entity["engine_entity"]:
        raise ValueError(
            f"Axiom receipt engine entity for {frame_entity!r} is invalid."
        )
    expected_id_column = frame.schema.entity_id_column(frame_entity)
    if entity["current_id_column"] != expected_id_column:
        raise ValueError(f"Axiom receipt id column for {frame_entity!r} differs.")
    current_ids = _integer_id_vector(
        frame.table(frame_entity)[expected_id_column],
        f"{frame_entity}.{expected_id_column}",
        unique=True,
    )
    expected_current_identity = _array_identity(current_ids)
    _verify_array_identity(entity["current_ids"], f"{frame_entity} current ids")
    if entity["current_ids"] != expected_current_identity:
        raise ValueError(f"Axiom receipt current ids for {frame_entity!r} differ.")

    declared = entity["declared_root_inputs"]
    if not isinstance(declared, list) or any(
        not isinstance(name, str) or not name for name in declared
    ):
        raise ValueError(f"Axiom declared inputs for {frame_entity!r} are invalid.")
    if len(set(declared)) != len(declared):
        raise ValueError(f"Axiom declared inputs for {frame_entity!r} repeat.")
    provided = entity["provided_root_inputs"]
    if not isinstance(provided, Mapping):
        raise ValueError(f"Axiom provided inputs for {frame_entity!r} are invalid.")
    live_inputs = _batch_from_table(frame.table(frame_entity), declared)
    expected_inputs = {
        name: _array_identity(live_inputs[name]) for name in sorted(live_inputs)
    }
    for name, identity in provided.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Axiom provided input names must be non-empty strings.")
        _verify_array_identity(identity, f"root input {name!r}")
    if dict(provided) != expected_inputs:
        raise ValueError(f"Axiom provided inputs for {frame_entity!r} differ.")

    raw_relations = entity["relations"]
    if not isinstance(raw_relations, Mapping):
        raise ValueError(f"Axiom relations for {frame_entity!r} are invalid.")
    relations: dict[str, object] = {}
    for key in sorted(raw_relations):
        relation = _verify_relation_receipt(
            frame,
            frame_entity=frame_entity,
            relation_key=key,
            current_ids=current_ids,
            value=raw_relations[key],
        )
        relations[key] = relation

    outputs = entity["requested_outputs"]
    if not isinstance(outputs, Mapping) or not outputs:
        raise ValueError(f"Axiom outputs for {frame_entity!r} are invalid.")
    normalized_outputs: dict[str, object] = {}
    for name in sorted(outputs):
        if not isinstance(name, str) or not name:
            raise ValueError("Axiom output names must be non-empty strings.")
        output = _exact_mapping(
            outputs[name],
            {
                "declared_engine_dtype",
                "declared_kernel_dtype",
                "declared_period",
                "values",
            },
            f"Axiom output {name!r}",
        )
        for metadata_key in (
            "declared_engine_dtype",
            "declared_kernel_dtype",
            "declared_period",
        ):
            if not isinstance(output[metadata_key], str) or not output[metadata_key]:
                raise ValueError(f"Axiom output {name!r} metadata is invalid.")
        _verify_array_identity(output["values"], f"output {name!r}")
        normalized_outputs[name] = output

    return {
        "frame_entity": frame_entity,
        "engine_entity": entity["engine_entity"],
        "current_id_column": expected_id_column,
        "current_ids": expected_current_identity,
        "declared_root_inputs": declared,
        "provided_root_inputs": expected_inputs,
        "relations": relations,
        "requested_outputs": normalized_outputs,
    }


def _verify_relation_receipt(
    frame: Frame,
    *,
    frame_entity: str,
    relation_key: object,
    current_ids: np.ndarray,
    value: object,
) -> dict[str, object]:
    if not isinstance(relation_key, str) or not relation_key:
        raise ValueError("Axiom relation receipt keys must be non-empty strings.")
    relation = _exact_mapping(
        value,
        {
            "schema_version",
            "relation_key",
            "declarations",
            "binding",
            "related_id_column",
            "source_related_entity_ids",
            "source_edge_current_ids",
            "source_edge_related_ids",
            "ordered_edge_current_ids",
            "ordered_edge_related_ids",
            "offsets",
            "declared_related_inputs",
            "provided_related_inputs",
            "receipt_sha256",
        },
        f"Axiom relation receipt {relation_key!r}",
    )
    if relation["schema_version"] != 2 or relation["relation_key"] != relation_key:
        raise ValueError(f"Axiom relation receipt {relation_key!r} identity differs.")
    _require_sha256(relation["receipt_sha256"], "relation receipt_sha256")
    unsigned = {key: item for key, item in relation.items() if key != "receipt_sha256"}
    if _canonical_digest(unsigned) != relation["receipt_sha256"]:
        raise ValueError(f"Axiom relation receipt {relation_key!r} digest differs.")
    raw_declarations = relation["declarations"]
    if not isinstance(raw_declarations, list) or not raw_declarations:
        raise ValueError(f"Axiom relation {relation_key!r} needs declarations.")
    declarations: list[dict[str, object]] = []
    for raw in raw_declarations:
        declaration = _exact_mapping(
            raw,
            {
                "relation_key",
                "relation_name",
                "current_slot",
                "related_slot",
                "related_inputs",
            },
            f"Axiom relation declaration {relation_key!r}",
        )
        if declaration["relation_key"] != relation_key:
            raise ValueError(f"Axiom relation declaration {relation_key!r} differs.")
        if (
            not isinstance(declaration["relation_name"], str)
            or not declaration["relation_name"]
        ):
            raise ValueError(f"Axiom relation {relation_key!r} name is invalid.")
        for slot in ("current_slot", "related_slot"):
            if type(declaration[slot]) is not int or declaration[slot] < 0:
                raise ValueError(f"Axiom relation {relation_key!r} slot is invalid.")
        _declaration_inputs(declaration, relation_key)
        declarations.append(declaration)
    declarations.sort(key=_canonical_json)
    if declarations != raw_declarations:
        raise ValueError(f"Axiom relation {relation_key!r} declarations are unsorted.")

    binding_data = _exact_mapping(
        relation["binding"],
        {
            "current_entity",
            "related_entity",
            "edge_table",
            "edge_current_id_column",
            "edge_related_id_column",
        },
        f"Axiom relation binding {relation_key!r}",
    )
    binding = AxiomRelationBinding(**binding_data)
    if binding.current_entity != frame_entity:
        raise ValueError(f"Axiom relation {relation_key!r} current entity differs.")
    _, _, projected = _relation_projection(
        frame,
        relation_key=relation_key,
        declarations=declarations,
        binding=binding,
        current_ids=current_ids,
    )
    expected = {
        **projected,
        "receipt_sha256": _canonical_digest(projected),
    }
    for identity_key in (
        "source_related_entity_ids",
        "source_edge_current_ids",
        "source_edge_related_ids",
        "ordered_edge_current_ids",
        "ordered_edge_related_ids",
        "offsets",
    ):
        _verify_array_identity(relation[identity_key], identity_key)
    provided = relation["provided_related_inputs"]
    if not isinstance(provided, Mapping):
        raise ValueError(f"Axiom relation {relation_key!r} inputs are invalid.")
    for name, identity in provided.items():
        _verify_array_identity(identity, f"related input {name!r}")
    if relation != expected:
        raise ValueError(f"Axiom relation receipt {relation_key!r} differs live.")
    return expected


def _verify_period_receipt(value: object) -> dict[str, object]:
    period = _exact_mapping(value, {"kind", "start", "end"}, "Axiom period")
    kind, start, end = period["kind"], period["start"], period["end"]
    if not all(isinstance(item, str) and item for item in (kind, start, end)):
        raise ValueError("Axiom period fields must be non-empty strings.")
    if kind == "calendar_year" and len(start) == 10:
        year = start[:4]
        expected = _period_bounds(year)
    elif kind == "month" and len(start) == 10:
        expected = _period_bounds(start[:7])
    else:
        raise ValueError("Axiom receipt period kind is invalid.")
    if (start, end, kind) != expected:
        raise ValueError("Axiom receipt period bounds differ.")
    return period


def _verify_array_identity(value: object, label: str) -> None:
    identity = _exact_mapping(
        value,
        {"dtype", "storage_dtype", "encoding", "shape", "sha256"},
        f"Axiom array identity {label}",
    )
    for key in ("dtype", "storage_dtype", "encoding"):
        if not isinstance(identity[key], str) or not identity[key]:
            raise ValueError(f"Axiom array identity {label} {key} is invalid.")
    shape = identity["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or type(shape[0]) is not int
        or shape[0] < 0
    ):
        raise ValueError(f"Axiom array identity {label} shape is invalid.")
    _require_sha256(identity["sha256"], f"array identity {label}")


def _exact_mapping(
    value: object,
    keys: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    result = dict(value)
    if set(result) != keys:
        raise ValueError(
            f"{label} keys differ: expected {sorted(keys)}, got {sorted(result)}."
        )
    return result


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 identity.")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _canonical_digest(value: Mapping[str, object]) -> str:
    """SHA-256 of canonical JSON receipt data."""
    payload = _canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
