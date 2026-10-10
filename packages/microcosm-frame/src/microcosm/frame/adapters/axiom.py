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

Explicit periods
----------------
A year label maps to the calendar year and ``YYYY-MM`` to the month unless
the adapter is given a ``periods`` mapping. With one, every period label must
appear in it: ``"2026-27"`` resolves to the :class:`AxiomPeriod` bounds the
caller declared (for example a New Zealand ``tax_year`` from 1 April), and an
unmapped label fails instead of falling back to a calendar year.

Graph output types
------------------
By default :meth:`AxiomEngine.materialize` returns each dense output as a
numpy array (``np.asarray`` of what the surface returns), with no cast:
judgments as int8 codes, and text and dates (which the surface returns as
Python string lists) as numpy string arrays. A graph node may own only a
closed set of dtypes, so
``output_dtypes="graph"`` casts each output to one of them without loss —
judgments to int64 codes, integers to int64, decimals to float64, booleans
to bool — and refuses text and date outputs before the engine runs.

Engine references
-----------------
``simulate.rules@1`` binds a node to its adapter through the node's
``engine_ref`` parameter, and its implementation hash covers adapter source,
not RuleSpec bytes or the native engine build. :func:`axiom_engine_ref` builds
that reference from the pins that decide the adapter's outputs: the engine
commit and wheel digest, the RuleSpec commit, the module path and digest, the
content digest of the whole RuleSpec root, and the adapter's configuration.
Any edit to a RuleSpec byte therefore moves the reference, and with it every
node key that names it. The reference also pins the adapter: from then on it
compiles only from a root whose digest still matches and that holds no
symbolic link, and only when its module path still resolves to the file the
reference named, so a node keyed by the reference is never computed from
other bytes.
"""

import hashlib
import json
import os
import re
import subprocess
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from functools import cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame.bundle import Frame
from microcosm.frame.concept_mapping import ConceptMapping
from microcosm.frame.materialize import engine_tables, put_frame_table, read_frame_table
from microcosm.frame.rules import ExportContract
from microcosm.frame.schema import EntitySchema, VariableMetadata
from microcosm.graph.canonical import canonical_json

__all__ = [
    "AxiomEngine",
    "AxiomEntityTableDataset",
    "AxiomPeriod",
    "BE_SCHEMA",
    "NZ_NESTING",
    "NZ_SCHEMA",
    "assert_no_relations",
    "axiom_concept_mapping",
    "axiom_engine_ref",
    "rulespec_tree_digest",
]

#: The Belgian frame schema for the populace-be pilot: persons in households.
#: Belgian PIT is individual with household-level elements (joint assessment,
#: quotient conjugal); benefits use household/family units. Fiscal/benefit
#: units beyond the household enter as group entities when the encoded slice
#: needs them, mapped via ``entity_names``.
BE_SCHEMA = EntitySchema(group_entities=("household",))

#: The New Zealand transport schema: persons in households and in benefit
#: units (``family``, the engine's ``Family`` entity). Only households carry
#: explicit weights; a family inherits its household's weight through
#: membership (:meth:`Frame.resolve_weights`).
NZ_SCHEMA = EntitySchema(group_entities=("household", "family"))

#: Group nesting the New Zealand schema requires: every family lies inside
#: exactly one household. Pass it as ``AxiomEngine(nesting=NZ_NESTING)``.
NZ_NESTING: Mapping[str, str] = MappingProxyType({"family": "household"})

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

# A rulespec country tree: ``nz``, or a subnational ``be-bru`` under ``be``.
_COUNTRY_TREE = re.compile(r"^([a-z]{2})(?:-[a-z0-9]+)*$")
_MAPPINGS_DIRECTORY = "axiom_concept_mappings"

#: ``native`` returns the dense surface's arrays unchanged; ``graph`` casts
#: them to graph-ownable dtypes (see "Graph output types" above).
_OUTPUT_DTYPE_MODES: tuple[str, ...] = ("native", "graph")

#: Engine dtype -> the owned-column dtype token of its graph-typed output.
#: Text and date outputs reach Python as string lists and have no entry.
_GRAPH_DTYPE_BY_ENGINE: dict[str, str] = {
    "bool": "bool",
    "integer": "int64",
    "decimal": "float64",
    "judgment": "int64",
}
_GRAPH_OUTPUT_DTYPES: frozenset[str] = frozenset(_GRAPH_DTYPE_BY_ENGINE)

#: The tri-state judgment codes the dense surface emits.
_JUDGMENT_CODES = np.asarray((-1, 0, 1), dtype=np.int64)

#: Versioned schema of :func:`axiom_engine_ref` documents.
_ENGINE_REF_SCHEMA = "microcosm.frame.axiom-engine-ref/1"

# A full git object name (SHA-1 or SHA-256 repositories), lowercase.
_GIT_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AxiomPeriod:
    """Explicit dense-execution bounds for one policy period.

    Attributes:
        start: First day, ISO ``YYYY-MM-DD``.
        end: Last day, ISO ``YYYY-MM-DD``, not before ``start``.
        kind: The engine's period identifier (for example ``tax_year``), not
            a display label. No fiscal-year convention is inferred from a
            year; the caller states the bounds.

    Raises:
        ValueError: If ``kind`` is empty or padded with whitespace, a date is
            not a valid ISO ``YYYY-MM-DD`` string, or ``start`` follows
            ``end``.
    """

    start: str
    end: str
    kind: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, str)
            or not self.kind.strip()
            or self.kind != self.kind.strip()
        ):
            raise ValueError(
                "Axiom period kind must be a non-empty string without "
                "surrounding whitespace."
            )
        try:
            start = date.fromisoformat(self.start)
            end = date.fromisoformat(self.end)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Axiom period dates must be valid ISO YYYY-MM-DD dates."
            ) from exc
        if start.isoformat() != self.start or end.isoformat() != self.end:
            raise ValueError("Axiom period dates must use ISO YYYY-MM-DD format.")
        if start > end:
            raise ValueError("Axiom period start must not follow end.")

    def bounds(self) -> tuple[str, str, str]:
        """Return the dense executor's ``(start, end, period_kind)`` tuple."""
        return self.start, self.end, self.kind


@cache
def axiom_concept_mapping(country: str) -> ConceptMapping:
    """The committed concept mapping for rulespec ``country``.

    Axiom inputs are module-scoped and untyped (usage-inferred;
    TheAxiomFoundation/axiom-rules-engine#62), so each binding names its
    RuleSpec module and the engine's canonical request name, and the mapping
    records ``input_declaration = usage_inferred``. The household concept
    entity corresponds to the engine's ``Household`` entity; rulespec-nz has
    none, so New Zealand household concepts reach it only through ``Family``
    bindings with group rules. The mapping is data,
    committed beside this module as ``axiom_concept_mappings/<country>.json``
    and validated like every other mapping when loaded.

    Raises:
        FileNotFoundError: If no mapping is committed for ``country``.
    """

    resource = resources.files(__package__).joinpath(
        _MAPPINGS_DIRECTORY, f"{country}.json"
    )
    if not resource.is_file():
        raise FileNotFoundError(f"No Axiom concept mapping for {country!r}.")
    return ConceptMapping.from_dict(json.loads(resource.read_text("utf-8")))


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
        arithmetic: ``"decimal"`` (exact, canonical) or ``"f64"`` (faster,
            floating-point rounding) — which dense execution mode
            :meth:`materialize` uses.
        periods: Optional period label -> :class:`AxiomPeriod` mapping. When
            given, :meth:`materialize` resolves every label through it and an
            unmapped label fails, rather than becoming a calendar year or
            month. Labels compare as strings, so ``2026`` and ``"2026"`` are
            the same label and may not both appear. :meth:`materialize` also
            accepts an :class:`AxiomPeriod` directly.
        nesting: Optional child group -> parent group mapping (for example
            :data:`NZ_NESTING`). Each declared child group must lie inside
            exactly one parent group, which :meth:`materialize` and
            :meth:`write_dataset` check from the person memberships alone.
        output_dtypes: ``"native"`` (default) returns the dense surface's
            arrays unchanged; ``"graph"`` casts them to graph-ownable dtypes
            and refuses text and date outputs (see the module docstring).

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
        arithmetic: str = "decimal",
        periods: Mapping[int | str, AxiomPeriod] | None = None,
        nesting: Mapping[str, str] | None = None,
        output_dtypes: str = "native",
    ) -> None:
        if arithmetic not in ("decimal", "f64"):
            raise ValueError(
                f"arithmetic must be 'decimal' or 'f64', got {arithmetic!r}."
            )
        if output_dtypes not in _OUTPUT_DTYPE_MODES:
            raise ValueError(
                f"output_dtypes must be one of {list(_OUTPUT_DTYPE_MODES)}, "
                f"got {output_dtypes!r}."
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
        self._output_dtypes = output_dtypes
        self._periods = _validated_periods(periods)
        self._nesting = _validated_nesting(nesting, schema)
        self._frame_entity_by_engine = {
            engine: frame for frame, engine in self._entity_names.items()
        }
        self._programs: dict[str, Any] = {}
        self._metadata: dict[str, Any] | None = None
        # Set by axiom_engine_ref: the RuleSpec tree digest this adapter's
        # reference names, and the file its module path resolved to. Every
        # later compile re-checks both.
        self._pinned_tree_sha256: str | None = None
        self._pinned_module: Path | None = None

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

    def graph_dtype(self, name: str) -> str:
        """The owned-column dtype token of ``name``'s graph-typed output.

        What a graph node declares in ``Owned(entity, name, dtype)`` when it
        materializes ``name`` through an ``output_dtypes="graph"`` adapter:
        ``bool``, ``int64`` (integers and judgment codes), or ``float64``
        (decimals).

        Raises:
            ImportError: If ``axiom_rules_engine`` is not installed.
            ValueError: If the adapter is not ``output_dtypes="graph"`` (its
                native arrays, int8 judgment codes among them, are not graph
                dtypes), or the variable is unknown, an input, on an unmapped
                engine entity, or a text or date output.
        """
        if self._output_dtypes != "graph":
            raise ValueError(
                "graph_dtype describes output_dtypes='graph' adapters only; this "
                "adapter returns native engine arrays."
            )
        self.variable_metadata(name)
        engine_dtype = self._derived_metadata()[name].dtype
        token = _GRAPH_DTYPE_BY_ENGINE.get(engine_dtype)
        if token is None:
            raise ValueError(
                f"{name!r} is a {engine_dtype!r} output; no graph column holds "
                "text or date values."
            )
        return token

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

    def concept_mapping(self) -> ConceptMapping:
        """Return the concept mapping for this adapter's rulespec country.

        The country is read from the module's path under its root (see
        :meth:`rulespec_country`), so no engine import is needed. The
        mapping is country-wide; each binding names the module whose
        program takes it.

        Raises:
            ValueError: If no root contains the module, or it is not under a
                country tree.
            FileNotFoundError: If no mapping is committed for that country.
        """
        return axiom_concept_mapping(self.rulespec_country())

    def rulespec_country(self) -> str:
        """The rulespec country whose tree holds the module.

        Read from the module's first path component under its root
        (``nz/statutes/...`` is New Zealand; ``be-bru/...`` is Belgium), so
        the root directory's own name (a worktree, a symlink) never matters.

        Raises:
            ValueError: If no root contains the module.
        """
        module = self._module.resolve()
        for root in self._rulespec_roots:
            resolved = root.resolve()
            if module.is_relative_to(resolved):
                top = module.relative_to(resolved).parts[0]
                match = _COUNTRY_TREE.match(top)
                if match is None:
                    raise ValueError(
                        f"Module {self._module} is not under a country tree "
                        f"(found {top!r})."
                    )
                return match.group(1)
        raise ValueError(f"No RuleSpec root contains module {self._module}.")

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def materialize(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str | AxiomPeriod,
    ) -> Mapping[str, np.ndarray]:
        """Compute ``variables`` for ``period`` over the bundle's tables.

        Groups the requested variables by owning entity, builds one dense
        columnar batch per entity from that entity's table (column dtypes
        decide the engine column types), and executes the compiled module.

        Args:
            bundle: A bundle whose entities match the adapter's schema.
            variables: Computed (derived) variable names.
            period: ``2025`` / ``"2025"`` for a calendar year, ``"2025-01"``
                for a month, or explicit :class:`AxiomPeriod` bounds. When
                the adapter has a ``periods`` mapping, a label resolves only
                through it.

        Returns:
            One array per variable, row-aligned to the variable's entity
            table. By default judgment variables come back as int8 codes
            (``1`` holds, ``-1`` not holds, ``0`` undetermined); with
            ``output_dtypes="graph"`` they come back as int64 codes and every
            array holds a graph-ownable dtype.

        Raises:
            ImportError: If ``axiom_rules_engine`` is not installed.
            ValueError: If the bundle's entities do not match the schema, a
                declared nesting or relation column is violated, the period
                label is unmapped, a requested variable is unknown or an
                input, a graph-typed output is text, date, or not losslessly
                castable, or a computed array's length does not match its
                entity table.
            NotImplementedError: If the compiled module declares relations
                (cross-entity aggregation batches; not yet wired — the BE
                pilot slice declares none).
        """
        self._require_schema(bundle)
        start, end, period_kind = self._materialization_period(period)

        by_entity: dict[str, list[str]] = {}
        for name in variables:
            metadata = self.variable_metadata(name)
            by_entity.setdefault(metadata.entity, []).append(name)
        engine_dtypes: dict[str, str] = {}
        if self._output_dtypes == "graph":
            derived = self._derived_metadata()
            engine_dtypes = {name: derived[name].dtype for name in variables}
            unsupported = sorted(
                f"{name} ({dtype})"
                for name, dtype in engine_dtypes.items()
                if dtype not in _GRAPH_OUTPUT_DTYPES
            )
            if unsupported:
                raise ValueError(
                    "output_dtypes='graph' cannot type text or date outputs as "
                    f"graph columns; refused: {unsupported}."
                )

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
                if self._output_dtypes == "graph":
                    values = _graph_output(name, engine_dtypes[name], values)
                results[name] = values
        return results

    def _materialization_period(
        self, period: int | str | AxiomPeriod
    ) -> tuple[str, str, str]:
        """Resolve a period to dense ``(start, end, period_kind)`` bounds.

        Raises:
            ValueError: If the adapter has a ``periods`` mapping and the label
                is not in it, or (without one) the label is neither a year
                nor a month.
        """
        if isinstance(period, AxiomPeriod):
            return period.bounds()
        if self._periods is not None:
            bounds = self._periods.get(str(period))
            if bounds is None:
                raise ValueError(
                    f"No explicit Axiom period bounds for {period!r}; the "
                    f"adapter maps only {sorted(self._periods)}."
                )
            return bounds.bounds()
        return _period_bounds(period)

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

    def _module_label(self) -> str:
        """The module's path under its RuleSpec root, for error messages.

        Many modules share a file name (``core.yaml``), so a message names the
        relative path; a module under no root falls back to its given path.
        """
        module = self._module.resolve()
        for root in self._rulespec_roots:
            resolved = root.resolve()
            if module.is_relative_to(resolved):
                return module.relative_to(resolved).as_posix()
        return str(self._module)

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
        if self._pinned_tree_sha256 is not None:
            self._require_pinned_tree()
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

    def _require_pinned_tree(self) -> None:
        """Refuse to compile when the root no longer holds the referenced bytes.

        Once :func:`axiom_engine_ref` has named this adapter's RuleSpec tree,
        every compile requires the module path to resolve to the file the
        reference named (a module given through a link outside the root could
        be repointed at another file under it without moving the digest),
        refuses a symbolic link under the root (the digest does not descend
        into a linked directory, so a directory or dangling link added after
        the reference would not move it) and re-hashes the root, so a node
        keyed by that reference is never computed from other bytes.

        Raises:
            ValueError: If the module path resolves to another file, the root
                holds a symbolic link, or its digest moved since the
                reference.
        """
        module = self._module.resolve()
        if module != self._pinned_module:
            raise ValueError(
                f"Module {self._module} resolves to {module}, not the file "
                f"{self._pinned_module} this adapter's engine_ref named; "
                "construct a new adapter and reference."
            )
        _refuse_symlinks(self._rulespec_roots[0].resolve())
        current, _ = rulespec_tree_digest(self._rulespec_roots[0].resolve())
        if current != self._pinned_tree_sha256:
            raise ValueError(
                f"RuleSpec root {self._rulespec_roots[0]} changed after this "
                "adapter's engine_ref named it (tree digest "
                f"{self._pinned_tree_sha256} -> {current}); construct a new "
                "adapter and reference."
            )

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

    def _require_schema(self, bundle: Frame, *, export: bool = False) -> None:
        """Refuse a bundle whose entities or group nesting the adapter cannot run.

        Beyond matching the schema's entities, two nesting checks run, both
        against the person memberships. Weight agreement alone never proves
        nesting: a family split across two equally weighted households still
        resolves a family weight.

        * Every child -> parent pair in the adapter's ``nesting`` must nest:
          all members of one child group share one parent group.
        * An explicit group-to-group id column ``{group}_{parent}_id`` (for
          example ``family_household_id``) must sit on ``group``'s table and
          agree with the members' ``parent`` membership. When ``export`` is
          true (the :meth:`write_dataset` path), a relation column the export
          contract requires must be present; a graph node materializing
          through the adapter need not slice it.

        Raises:
            ValueError: On an entity mismatch or a violated nesting check.
        """
        if set(bundle.entities) != set(self._schema.entities):
            raise ValueError(
                f"Axiom adapter requires the schema entities "
                f"{list(self._schema.entities)}; bundle has "
                f"{list(bundle.entities)}."
            )
        schema = self._schema
        person = bundle.table(schema.person_entity)
        for child, parent in self._nesting.items():
            pairs = pd.DataFrame(
                {
                    "child": person[schema.membership_column(child)].to_numpy(),
                    "parent": person[schema.membership_column(parent)].to_numpy(),
                }
            ).drop_duplicates()
            spanning = pairs.loc[pairs["child"].duplicated(keep=False), "child"]
            if not spanning.empty:
                ids = sorted(spanning.unique().tolist())
                raise ValueError(
                    f"Every {child!r} must nest in exactly one {parent!r}; "
                    f"{child} id(s) {ids[:5]} have members in more than one "
                    f"{parent}."
                )
        for group in schema.group_entities:
            for parent in schema.group_entities:
                if group == parent:
                    continue
                column = f"{group}_{parent}_id"
                try:
                    owner = bundle.column_entity(column)
                except ValueError:
                    if export and column in self._contract.required:
                        raise ValueError(
                            f"Required relation column {column!r} is missing "
                            f"from entity {group!r}."
                        ) from None
                    continue
                if owner != group:
                    raise ValueError(
                        f"{column!r} must be on entity {group!r}, not {owner!r}."
                    )
                broadcast = bundle.place(
                    column, schema.person_entity, how="broadcast"
                ).table(schema.person_entity)[column]
                membership = person[schema.membership_column(parent)]
                if not np.array_equal(broadcast.to_numpy(), membership.to_numpy()):
                    raise ValueError(
                        f"{column!r} disagrees with person membership; every "
                        f"{group!r} must be nested in its declared {parent!r}."
                    )

    def _engine_tables(self, bundle: Frame) -> dict[str, pd.DataFrame]:
        """Copy the bundle's tables and materialize typed weights as columns.

        Delegates to the shared :func:`microcosm.frame.materialize.engine_tables`
        (typed weights authoritative, any existing ``{entity}_weight`` column
        overwritten, never trusted), keyed to this adapter's schema order.
        """
        self._require_schema(bundle, export=True)
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


def rulespec_tree_digest(root: str | Path) -> tuple[str, int]:
    """Content digest and payload size of a RuleSpec root directory.

    The formula is the graph's directory source identity
    (``microcosm.graph.keys``): SHA-256 over a fixed prefix and then, for every
    regular file in sorted path order, its length-prefixed relative POSIX path
    and its length-prefixed bytes. The digest therefore equals the content hash
    :func:`microcosm.graph.keys.source_content_key` binds when the same
    directory is declared as a graph source. Renaming the root is inert;
    adding, removing, renaming, or editing any file inside it changes the
    digest. A ``.git`` directory, if present, is hashed like any other.

    Args:
        root: The RuleSpec root directory (for example a ``git archive``
            export of rulespec-nz).

    Returns:
        ``(sha256 hex digest, total bytes of the hashed files)``.

    Raises:
        ValueError: If ``root`` is not a directory.
    """
    path = Path(root)
    if not path.is_dir():
        raise ValueError(f"RuleSpec root {path} is not a directory.")
    digest = hashlib.sha256(b"microcosm-graph/source-directory/1\0")
    size = 0
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        content = candidate.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
        size += len(content)
    return digest.hexdigest(), size


def axiom_engine_ref(
    engine: AxiomEngine,
    *,
    engine_commit: str,
    wheel_sha256: str,
    rulespec_root: str | Path,
    rulespec_commit: str,
) -> str:
    """The ``engine_ref`` that names one Axiom adapter and the bytes behind it.

    A graph node binds its rules engine through ``params["engine_ref"]``, and
    that parameter enters the node key. The reference this function returns is
    canonical JSON of everything that decides the adapter's outputs:

    * the engine (``axiom-rules-engine``) and the commit and wheel SHA-256
      the caller declares for it;
    * the RuleSpec commit, the module's path relative to the root, the
      module's SHA-256, and :func:`rulespec_tree_digest` of the whole root
      (imports resolve anywhere under it);
    * the adapter's arithmetic, output dtypes, entity schema, entity names,
      period map, and declared nesting.

    For a ``git archive`` export (the normal root, with no ``.git``) no
    absolute path enters the reference: the same tree exported to two places
    gives the same reference, and editing any byte under the root gives a
    different one. The engine commit and wheel SHA-256 are pins the caller
    declares, not values read from the installed engine: nothing here
    inspects ``axiom_rules_engine``, so a different engine installed under the
    same pins gives the same reference. The RuleSpec commit is also a
    declaration, checked only for a git checkout root (below). The directory
    digest is the authority.

    A git checkout root is accepted only when it is the top level of its own
    repository, clean, at ``rulespec_commit``, contains no submodules, has no
    tracked file flagged skip-worktree or assume-unchanged (``git status``
    does not report edits to those), and holds outside ``.git`` exactly the
    commit's files, each byte for byte the blob the commit records and with
    its executable bit. A recorded name matches a file whose name is the
    same or, when that one file opens under both, canonically equivalent
    (git on macOS with ``core.precomposeunicode=true`` records precomposed
    names, usually NFC, that the file system may list in NFD); one directory
    entry never answers for two recorded names. Git runs
    without the caller's repository-selecting variables (``GIT_DIR``,
    ``GIT_WORK_TREE`` and the rest of ``git rev-parse --local-env-vars``) and
    without replace refs, so neither can point the check at other objects.
    The commit's file list and blob names are read through git from the
    repository's object database, whose commit and tree objects are not
    rehashed, so the check assumes that database is intact: it does not
    detect an object rewritten under another object's name (a substituted
    nested tree, say), though the directory digest still pins the bytes
    present.
    Its ``.git`` is then hashed
    like any other file, as the graph's source key hashes it. That makes the
    reference specific to one clone: two clones or worktrees of one commit
    (a worktree's ``.git`` file names an absolute path) and an export of that
    commit all give different references, and any later git operation in the
    checkout moves it. Prefer the export.

    The first call pins the adapter to the tree digest it names and to the
    file its module path resolves to: it refuses an adapter that has already
    compiled (those programs were read from bytes no reference names), and
    from then on every compile, and every later reference to the same
    adapter, refuses a root whose digest has moved or that holds a symbolic
    link, and a module path that now resolves to another file.

    Args:
        engine: The adapter the reference names.
        engine_commit: Full lowercase git commit of axiom-rules-engine, as
            the caller declares it. It is recorded as given; the installed
            engine is not checked against it.
        wheel_sha256: Lowercase SHA-256 of the engine wheel, as the caller
            declares it. It is recorded as given; no installed wheel or
            native library is hashed to confirm it.
        rulespec_root: The adapter's only RuleSpec root.
        rulespec_commit: Full lowercase git commit the root was exported from.

    Returns:
        The reference, a canonical JSON string.

    Raises:
        TypeError: If ``engine`` is not an :class:`AxiomEngine`.
        ValueError: If a pin is malformed; ``rulespec_root`` is not the
            adapter's only root; the module is not a file under it; the root
            contains a symbolic link (the digest would not cover what it
            points at); a git checkout root is not its repository's top
            level, is dirty, contains a submodule, has a file flagged
            skip-worktree or assume-unchanged, holds bytes its commit does
            not, or is at another commit; or the adapter compiled before its
            first reference, or its root or module moved after it.
    """
    if not isinstance(engine, AxiomEngine):
        raise TypeError(f"{engine!r} is not an AxiomEngine.")
    pins = (
        ("engine_commit", engine_commit, _GIT_COMMIT, "a full lowercase git commit"),
        (
            "rulespec_commit",
            rulespec_commit,
            _GIT_COMMIT,
            "a full lowercase git commit",
        ),
        ("wheel_sha256", wheel_sha256, _SHA256, "a lowercase SHA-256 hex digest"),
    )
    for label, value, pattern, shape in pins:
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise ValueError(f"{label} must be {shape}, got {value!r}.")
    root = Path(rulespec_root)
    if not root.is_dir():
        raise ValueError(f"rulespec_root {root} is not a directory.")
    resolved_root = root.resolve()
    if tuple(item.resolve() for item in engine._rulespec_roots) != (resolved_root,):
        raise ValueError(
            "axiom_engine_ref digests one RuleSpec root, which must be the "
            "adapter's only root; adapter roots "
            f"{[str(item) for item in engine._rulespec_roots]}, rulespec_root "
            f"{root}."
        )
    module = engine._module.resolve()
    if not module.is_file() or not module.is_relative_to(resolved_root):
        raise ValueError(
            f"Module {engine._module} is not a file under rulespec_root {root}."
        )
    _refuse_symlinks(resolved_root)
    if (resolved_root / ".git").exists():
        _require_clean_checkout(resolved_root, rulespec_commit)
    tree_sha256, tree_bytes = rulespec_tree_digest(resolved_root)
    if engine._pinned_tree_sha256 is None:
        if engine._programs:
            raise ValueError(
                "axiom_engine_ref must name the adapter's RuleSpec tree before "
                "the adapter compiles: it has already compiled programs from "
                "bytes no reference names. Compute the reference on a fresh "
                "adapter."
            )
        engine._pinned_tree_sha256 = tree_sha256
        engine._pinned_module = module
    elif engine._pinned_tree_sha256 != tree_sha256:
        raise ValueError(
            f"RuleSpec root {root} changed after this adapter's engine_ref named "
            f"it (tree digest {engine._pinned_tree_sha256} -> {tree_sha256}); "
            "construct a new adapter and reference."
        )
    elif engine._pinned_module != module:
        raise ValueError(
            f"Module {engine._module} resolves to {module}, not the file "
            f"{engine._pinned_module} this adapter's engine_ref named; "
            "construct a new adapter and reference."
        )
    schema = engine._schema
    periods = (
        None
        if engine._periods is None
        else {
            label: {"start": item.start, "end": item.end, "kind": item.kind}
            for label, item in sorted(engine._periods.items())
        }
    )
    document = {
        "format": _ENGINE_REF_SCHEMA,
        "engine": "axiom-rules-engine",
        "engine_commit": engine_commit,
        "engine_wheel_sha256": wheel_sha256,
        "rulespec_commit": rulespec_commit,
        "rulespec_tree_sha256": tree_sha256,
        "rulespec_tree_bytes": tree_bytes,
        "module": module.relative_to(resolved_root).as_posix(),
        "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "arithmetic": engine._arithmetic,
        "output_dtypes": engine._output_dtypes,
        "person_entity": schema.person_entity,
        "group_entities": list(schema.group_entities),
        "entity_names": dict(sorted(engine._entity_names.items())),
        "periods": periods,
        "nesting": dict(sorted(engine._nesting.items())),
    }
    return canonical_json(document).decode("utf-8")


def assert_no_relations(engine: AxiomEngine, entity: str) -> None:
    """Refuse a module whose dense program on ``entity`` declares relations.

    The adapter builds no relation batches from frame membership, so a module
    with dense relations cannot run on the graph. Callers check this when
    they compose a node, before any data reaches the engine.

    Args:
        engine: The adapter to check.
        entity: The frame entity the node materializes (``"family"``).

    Raises:
        TypeError: If ``engine`` is not an :class:`AxiomEngine`.
        ImportError: If ``axiom_rules_engine`` is not installed.
        ValueError: If the module has no derived rules on ``entity``.
        NotImplementedError: If the program declares relations; the message
            names the module and the relations.
    """
    if not isinstance(engine, AxiomEngine):
        raise TypeError(f"{engine!r} is not an AxiomEngine.")
    relations = list(engine._program(entity).relations)
    if relations:
        raise NotImplementedError(
            f"Module {engine._module_label()} declares dense relations "
            f"{[item.name for item in relations]} on entity {entity!r}; "
            "relation batches from frame membership are not wired, so it "
            "cannot run on the graph."
        )


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


def _validated_periods(
    periods: Mapping[int | str, AxiomPeriod] | None,
) -> dict[str, AxiomPeriod] | None:
    """Normalize a ``periods`` mapping to string labels, refusing ambiguity.

    Raises:
        TypeError: If ``periods`` is not a mapping, a label is not an int or
            a non-empty string, or a value is not an :class:`AxiomPeriod`.
        ValueError: If the mapping is empty, or two labels share a string
            form (``2026`` and ``"2026"``).
    """
    if periods is None:
        return None
    if not isinstance(periods, Mapping):
        raise TypeError("periods must map period labels to AxiomPeriod bounds.")
    if not periods:
        raise ValueError(
            "periods must map at least one label; omit it to use calendar years "
            "and months."
        )
    resolved: dict[str, AxiomPeriod] = {}
    for label, bounds in periods.items():
        if (
            isinstance(label, bool)
            or not isinstance(label, int | str)
            or (isinstance(label, str) and not label)
        ):
            raise TypeError(
                f"periods labels must be ints or non-empty strings, got {label!r}."
            )
        if not isinstance(bounds, AxiomPeriod):
            raise TypeError("periods values must be AxiomPeriod instances.")
        key = str(label)
        if key in resolved:
            raise ValueError(f"Duplicate explicit Axiom period label {key!r}.")
        resolved[key] = bounds
    return resolved


def _validated_nesting(
    nesting: Mapping[str, str] | None, schema: EntitySchema
) -> dict[str, str]:
    """Validate a child group -> parent group mapping against ``schema``.

    Raises:
        TypeError: If ``nesting`` is not a mapping.
        ValueError: If a side is not a group entity of ``schema``, a group
            maps to itself, or the declared parents form a cycle.
    """
    if nesting is None:
        return {}
    if not isinstance(nesting, Mapping):
        raise TypeError("nesting must map child group entities to parent groups.")
    groups = set(schema.group_entities)
    resolved: dict[str, str] = {}
    for child, parent in nesting.items():
        for side in (child, parent):
            if not isinstance(side, str) or side not in groups:
                raise ValueError(
                    f"nesting names {side!r}, which is not a group entity of "
                    f"the schema {list(schema.group_entities)}."
                )
        if child == parent:
            raise ValueError(f"nesting maps group {child!r} to itself.")
        resolved[child] = parent
    for start in resolved:
        seen = {start}
        current = resolved[start]
        while current in resolved:
            if current in seen:
                raise ValueError(f"nesting is cyclic through group {current!r}.")
            seen.add(current)
            current = resolved[current]
    return resolved


def _graph_output(name: str, engine_dtype: str, values: np.ndarray) -> np.ndarray:
    """Cast one dense output array to its graph-ownable dtype without loss.

    Bool stays bool; integers and judgment codes become int64; decimals
    become float64. Judgment codes must be ``-1``, ``0`` or ``1``. The input
    must already hold the engine's numpy kind for that dtype: nothing is
    parsed, rounded, or truncated.

    Raises:
        ValueError: If ``engine_dtype`` has no graph type, the array's numpy
            dtype does not match it, or a value would not survive the cast.
    """
    if engine_dtype not in _GRAPH_DTYPE_BY_ENGINE:
        raise ValueError(
            f"Output {name!r} has engine dtype {engine_dtype!r}, which no graph "
            "column holds."
        )
    kind = values.dtype.kind
    if engine_dtype == "bool" and kind == "b":
        return values.astype(np.bool_)
    if engine_dtype == "decimal" and kind == "f":
        return values.astype(np.float64)
    if engine_dtype in ("integer", "judgment") and kind in ("i", "u"):
        if kind == "u" and values.size and values.max() > np.iinfo(np.int64).max:
            raise ValueError(
                f"Output {name!r} holds unsigned values beyond the int64 range."
            )
        cast = values.astype(np.int64)
        if engine_dtype == "judgment" and not np.isin(cast, _JUDGMENT_CODES).all():
            invalid = sorted(set(np.unique(cast).tolist()) - {-1, 0, 1})
            raise ValueError(
                f"Judgment output {name!r} holds codes outside -1/0/1: {invalid[:5]}."
            )
        return cast
    raise ValueError(
        f"Output {name!r} ({engine_dtype}) arrived as numpy dtype {values.dtype}, "
        "which does not cast to a graph column without loss."
    )


def _refuse_symlinks(root: Path) -> None:
    """Refuse a RuleSpec root holding symbolic links.

    The tree digest walks the root without following directory links, so a
    linked directory's files would reach the engine without entering the
    reference; a linked file would bind its target's bytes under the link's
    name. A ``git archive`` export of a RuleSpec repository has none.

    Raises:
        ValueError: Naming the first link found.
    """
    for directory, directories, files in root.walk(follow_symlinks=False):
        for name in sorted((*directories, *files)):
            candidate = directory / name
            if candidate.is_symlink():
                raise ValueError(
                    f"RuleSpec root {root} contains the symbolic link "
                    f"{candidate.relative_to(root).as_posix()}; export the tree "
                    "without links."
                )


def _require_clean_checkout(root: Path, commit: str) -> None:
    """Require a git-checkout RuleSpec root to hold exactly ``commit``.

    Git examines the root's own repository: it runs without the caller's
    repository-selecting variables (:func:`_git_environment`) and without
    replace refs (a ``refs/replace`` entry makes one commit read as another),
    and the root must be the repository's top level (``core.worktree`` can
    point a repository at another directory). ``--no-optional-locks`` stops
    ``git status`` refreshing the index, which would otherwise rewrite
    ``.git/index`` and move the digest taken next.

    The index may hold no submodule: the commit records a submodule's
    commit, not its files, and ``git status`` does not report every change
    inside one. No tracked file may be flagged skip-worktree or
    assume-unchanged, even unedited, because ``git status`` does not report
    edits to it. Ignored files count as changes: the digest hashes them, but
    the commit does not hold them. Last, the files outside the top-level
    ``.git`` must be exactly the commit's, each byte for byte the regular-file
    blob the commit records at its path, with its recorded executable bit.
    That refuses a change ``git status``
    does not report: an edit hidden from a stat cache trusted under
    ``core.trustctime=false``, by a clean filter, or by a file system monitor;
    a nested ``.git`` entry, which git's untracked scan skips; or a committed
    symbolic link held as a plain file under ``core.symlinks=false``; or an
    executable-bit change hidden by ``core.filemode=false``. Recorded paths
    pair with files one to one as :func:`_match_checkout_paths` describes,
    so an NFC name git records matches the one file the file system lists
    under an equivalent NFD name. The paths and blob names come from
    ``git ls-tree``, which reads the object database; the commit and tree
    objects are not rehashed, so an intact object database is assumed.

    Raises:
        ValueError: If git fails or times out; the root is not its
            repository's top level; the index holds a submodule or a file
            flagged skip-worktree or assume-unchanged; the checkout has
            modified, untracked, or ignored files; ``HEAD`` is not
            ``commit``; or the files outside ``.git`` are not the commit's.
    """

    environment = _git_environment()

    def git(*args: str) -> str:
        try:
            completed = subprocess.run(
                [
                    "git",
                    "--no-optional-locks",
                    "--no-replace-objects",
                    "-C",
                    str(root),
                    *args,
                ],
                capture_output=True,
                encoding="utf-8",
                errors="surrogateescape",
                check=False,
                timeout=120,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                f"git is required to verify the RuleSpec checkout {root}."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                f"git {' '.join(args)} timed out in RuleSpec checkout {root}."
            ) from exc
        if completed.returncode != 0:
            raise ValueError(
                f"git {' '.join(args)} failed in RuleSpec checkout {root}: "
                f"{completed.stderr.strip()}"
            )
        return completed.stdout

    toplevel = Path(git("rev-parse", "--show-toplevel").removesuffix("\n"))
    try:
        same = toplevel.samefile(root)
    except OSError:
        same = False
    if not same:
        raise ValueError(
            f"RuleSpec checkout {root} is not the top level of its git "
            f"repository, whose work tree is {toplevel}; reference a git archive "
            "export instead."
        )
    submodules: list[str] = []
    flagged: list[str] = []
    for entry in git("ls-files", "--stage", "-v", "-z").split("\0"):
        if not entry:
            continue
        # "<tag> <mode> <object> <stage>\t<path>". A submodule has mode
        # 160000; -v tags a skip-worktree file "S" and an assume-unchanged
        # file in lower case.
        fields, path = entry.split("\t", 1)
        tag, mode = fields.split(" ")[:2]
        if mode == "160000":
            submodules.append(path)
        if tag == "S" or tag.islower():
            flagged.append(path)
    if submodules:
        raise ValueError(
            f"RuleSpec checkout {root} contains the submodule {submodules[0]} "
            f"({len(submodules)} submodule(s)); commit {commit} records a "
            "submodule's commit, not its files, and git status does not report "
            "every change inside one. Reference a git archive export instead."
        )
    if flagged:
        raise ValueError(
            f"RuleSpec checkout {root}: {flagged[0]} is flagged skip-worktree "
            f"or assume-unchanged ({len(flagged)} flagged file(s)), so git "
            "status does not report edits to it; clear the flags or reference "
            "a git archive export instead."
        )
    if git("status", "--porcelain=v1", "--untracked-files=all", "--ignored").strip():
        raise ValueError(
            f"RuleSpec checkout {root} has modified, untracked, or ignored "
            f"files that commit {commit} does not hold; reference a git "
            "archive export instead."
        )
    head = git("rev-parse", "--verify", "HEAD^{commit}").strip()
    if head != commit:
        raise ValueError(
            f"RuleSpec checkout {root} is at {head}, not the declared "
            f"rulespec_commit {commit}."
        )
    algorithm = git("rev-parse", "--show-object-format").strip()
    if algorithm not in ("sha1", "sha256"):
        raise ValueError(
            f"RuleSpec checkout {root} uses the git object format {algorithm!r}, "
            "which this check cannot hash."
        )
    recorded: dict[str, tuple[str, str]] = {}
    for entry in git("ls-tree", "-r", "-z", "--full-tree", commit).split("\0"):
        if not entry:
            continue
        # "<mode> <type> <object>\t<path>"
        fields, path = entry.split("\t", 1)
        mode, _, name = fields.split(" ")
        recorded[path] = (mode, name)
    files, lacking, extra = _match_checkout_paths(root, recorded, _checkout_files(root))
    unexpected = sorted((*lacking, *extra))
    if unexpected:
        first = unexpected[0]
        state = "lacks" if first in lacking else "holds"
        raise ValueError(
            f"RuleSpec checkout {root} {state} {first}, unlike commit {commit} "
            f"({len(unexpected)} path(s) differ), though git status reports no "
            "change; reference a git archive export instead."
        )
    for path, (mode, name) in sorted(recorded.items()):
        # A symbolic link checked out under core.symlinks=false is a plain
        # file holding the link's blob, and git status reports no change.
        if mode not in ("100644", "100755"):
            raise ValueError(
                f"RuleSpec checkout {root}: commit {commit} records {path} with "
                f"mode {mode}, not as a regular file, though the checkout holds "
                "a plain file there; reference a git archive export instead."
            )
        file = files[path]
        if not file.is_file() or _git_blob_id(file.read_bytes(), algorithm) != name:
            raise ValueError(
                f"RuleSpec checkout {root}: {path} is not the blob {name} commit "
                f"{commit} records for it, though git status reports no change; "
                "reference a git archive export instead."
            )
        # Git records the owner's executable bit; core.filemode=false can
        # hide a change to it from git status.
        actual_mode = "100755" if file.stat().st_mode & 0o100 else "100644"
        if actual_mode != mode:
            raise ValueError(
                f"RuleSpec checkout {root}: {path} has mode {actual_mode}, not "
                f"the mode {mode} commit {commit} records for it, though git "
                "status reports no change; reference a git archive export "
                "instead."
            )


def _git_environment() -> dict[str, str]:
    """This process's environment without the variables that choose git's repository.

    The names removed are the installed git's own list of variables local to
    one repository (``git rev-parse --local-env-vars``): ``GIT_DIR``,
    ``GIT_WORK_TREE``, ``GIT_INDEX_FILE``, ``GIT_OBJECT_DIRECTORY``,
    ``GIT_COMMON_DIR``, the ``git -c`` carriers ``GIT_CONFIG_PARAMETERS`` and
    ``GIT_CONFIG_COUNT``, and the rest. Each names the repository, index,
    object store, or work tree git reads, or carries configuration into it.

    Raises:
        ValueError: If git is missing, fails, or times out.
    """

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--local-env-vars"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise ValueError("git is required to verify a RuleSpec checkout.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("git rev-parse --local-env-vars timed out.") from exc
    if completed.returncode != 0:
        raise ValueError(
            f"git rev-parse --local-env-vars failed: {completed.stderr.strip()}"
        )
    local = frozenset(completed.stdout.split())
    return {name: value for name, value in os.environ.items() if name not in local}


def _checkout_files(root: Path) -> dict[str, Path]:
    """Every non-directory entry under ``root`` except the top-level ``.git``.

    Keyed by POSIX path relative to ``root``, the form ``git ls-tree`` prints.
    """

    present: dict[str, Path] = {}
    for directory, directories, files in root.walk():
        if directory == root:
            directories[:] = [name for name in directories if name != ".git"]
            files = [name for name in files if name != ".git"]
        for name in files:
            path = directory / name
            present[path.relative_to(root).as_posix()] = path
    return present


def _match_checkout_paths(
    root: Path, recorded: Collection[str], present: Mapping[str, Path]
) -> tuple[dict[str, Path], list[str], list[str]]:
    """Pair each path a commit records with the checkout entry holding it, one to one.

    Git and the file system can spell one file's name differently: on macOS
    with ``core.precomposeunicode=true``, git records a name in Unicode NFC
    while the file system lists it as it was created, for instance in NFD,
    and both spellings open the same file. A recorded path pairs with the
    entry of the identical name when there is one. Otherwise it pairs with
    the one remaining entry whose name is canonically equivalent (the same
    NFD form), and only when ``root / path`` opens that entry's file
    (:meth:`Path.samefile`). Each entry pairs at most once; when a form has
    more than one unpaired path or entry, none of them pairs. Nothing is
    normalized wholesale: where the file system keeps an NFC and an NFD name
    apart they are two entries, neither standing in for the other, and one
    file cannot stand in for two recorded spellings. Names that differ in
    case are not equivalent.

    Args:
        root: The checkout's top level.
        recorded: Paths the commit records, POSIX and relative to ``root``.
        present: :func:`_checkout_files` of ``root``.

    Returns:
        The entry each paired recorded path names, the recorded paths left
        unpaired, and the entries left unpaired (as the file system spells
        them); both lists sorted.
    """

    paired = {path: present[path] for path in recorded if path in present}
    lacking: dict[str, list[str]] = {}
    for path in recorded:
        if path not in paired:
            lacking.setdefault(unicodedata.normalize("NFD", path), []).append(path)
    extra: dict[str, list[str]] = {}
    for path in present:
        if path not in paired:
            extra.setdefault(unicodedata.normalize("NFD", path), []).append(path)
    unpaired: list[str] = []
    for form, paths in lacking.items():
        entries = extra.get(form, [])
        if len(paths) == 1 and len(entries) == 1:
            try:
                same = (root / paths[0]).samefile(present[entries[0]])
            except OSError:
                same = False
            if same:
                paired[paths[0]] = present[entries.pop()]
                continue
        unpaired.extend(paths)
    leftover = sorted(path for entries in extra.values() for path in entries)
    return paired, sorted(unpaired), leftover


def _git_blob_id(content: bytes, algorithm: str) -> str:
    """The object name git gives ``content`` as a blob (no filters applied).

    Args:
        content: The blob's bytes.
        algorithm: The repository's object format, ``"sha1"`` or ``"sha256"``.
    """

    digest = hashlib.new(algorithm)
    digest.update(b"blob %d\0" % len(content))
    digest.update(content)
    return digest.hexdigest()
