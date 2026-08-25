"""PolicyEngine-US adapter for the :class:`~microcosm.frame.rules.RulesEngine` protocol.

``policyengine_us`` is imported lazily inside methods: this module (and
microcosm-frame itself) imports without it, and every entry point that does
need it raises a clear ``ImportError`` naming the
``microcosm-frame[policyengine]`` extra when it is absent.

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

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame.adapters._policyengine_us_source_index import (
    ConsumerReceipt,
    _PolicyEngineUSSourceIndex,
    _SourceVariableDefinition,
)
from microcosm.frame.adapters._policyengine_us_source_index import (
    _index_policyengine_us_sources as _build_policyengine_us_source_index,
)
from microcosm.frame.bundle import Frame
from microcosm.frame.materialize import (
    engine_tables,
    materialize_nullable_booleans_for_pytables,
    put_frame_table,
    read_frame_table,
)
from microcosm.frame.rules import ExportContract
from microcosm.frame.schema import EntitySchema, VariableMetadata
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ConsumerReceipt",
    "PolicyEngineUSEngine",
    "PolicyEngineUSVariableMetadataIndex",
    "VariableDependencyClosure",
]

_PERSON_TABLE = "person"
_GROUP_TABLES: tuple[str, ...] = (
    "household",
    "tax_unit",
    "spm_unit",
    "family",
    "marital_unit",
)
_HOUSEHOLD_WEIGHT_COLUMN = "household_weight"
_FORMULA_OWNED_COMPAT_COLUMNS = frozenset(
    {
        # PolicyEngine-US PR #8614 made this an aggregate of the two source
        # leaves. Some published wheels can still report it as an input, but
        # Microcosm must not persist it as a final dataset input.
        "partnership_s_corp_income",
        # Dividends are sourced as qualified and non-qualified leaves. Persisting
        # either total can make the stored inputs internally inconsistent.
        "dividend_income",
        "ordinary_dividend_income",
        # Social Security is sourced and targeted through benefit-type leaves.
        # Persisting the aggregate can disagree with those leaves and mask the
        # engine-owned total.
        "social_security",
    }
)


@dataclass(frozen=True)
class VariableDependencyClosure:
    """Static transitive variable graph for one PolicyEngine output.

    Edges are ordered ``(consumer, dependency)`` pairs and are deduplicated
    across source-reference sites.  The digest binds the installed engine
    version and the complete normalized graph, so a checked-in downstream
    input manifest can fail closed on either source or dependency drift
    without executing a microsimulation.
    """

    engine_version: str
    root: str
    input_leaves: tuple[str, ...]
    formula_nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    sha256: str


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

# PolicyEngine-US 1.819.0 creates 110 default-system variables outside ordinary
# top-level ``class ...(Variable)`` declarations. Keep the compact metadata
# snapshot tied to every source/activation surface that produced it: a changed
# wheel must fail closed until this audit is refreshed, never silently omit a
# newly generated formula-owned output.
_GENERATED_SOURCE_VERSION = "1.820.2"
_GENERATED_SOURCE_SHA256: dict[str, str] = {
    "model_api.py": "d7edb7436b84733f179fe223376fb588bb7a3ad6817d119703faeb599d4bb9c7",
    "variables/household/demographic/geographic/state/in_state.py": (
        "a3792c642387b652752461c85c03e5a9cb39fab55b4e038270374dd7e7d8aa60"
    ),
    "variables/gov/puf.py": (
        "17545c43549ecf34016107bc8ed2dce25a53610afb802431a1b0ea6724215e7b"
    ),
    "variables/gov/states/tax/income/_generate_state_mfs_variables.py": (
        "a0c9decd81b6eb76ac7edcddfc913d89ee86e0f18d8c51702bcb2a015dc2fabe"
    ),
    "reforms/states/mi/surtax.py": (
        "e1d0c0207c46243d3509b22b15fbdc07aa02b4df9461f7b93bec872dc7124ea9"
    ),
    "reforms/reforms.py": (
        "9846915c2b03e776dc37cdd6d92566de117f6eebafbf5508179e196fce28d474"
    ),
    "system.py": "820dadeb7d22ef14d9cb2c34607f2afe68ba5fae616e05e634c2319e61eb457d",
}
_GENERATED_VARIABLE_GROUPS: tuple[tuple[tuple[str, ...], str, str, bool], ...] = (
    (
        tuple(
            "AL AK AZ AR CA CO CT DC DE FL GA HI ID IL IN IA KS KY LA ME MD MA "
            "MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX "
            "UT VT VA WA WV WI WY PR VI".split()
        ),
        "household",
        "bool",
        True,
    ),
    (
        tuple(
            "e02000 e26270 e19200 e18500 e19800 e20400 e20100 e00700 e03270 "
            "e24515 e03300 e07300 e62900 e32800 e87530 e03240 e01100 e01200 "
            "e24518 e09900 e27200 e03290 e58990 e03230 e11200 e07260 e07240 "
            "e03220 p08000 e03400 e09800 e09700 e03500 e87521".split()
        ),
        "person",
        "float",
        False,
    ),
    (
        tuple(
            "ar_standard_deduction ar_itemized_deductions ar_taxable_income ar_agi "
            "dc_taxable_income de_standard_deduction de_itemized_deductions "
            "de_taxable_income de_agi ia_standard_deduction ia_itemized_deductions "
            "ia_taxable_income ia_agi ky_standard_deduction ky_itemized_deductions "
            "ky_taxable_income ms_standard_deduction ms_itemized_deductions "
            "ms_taxable_income mt_standard_deduction mt_itemized_deductions "
            "mt_taxable_income".split()
        ),
        "tax_unit",
        "float",
        True,
    ),
    (("mi_surtax",), "tax_unit", "float", True),
)


def _index_policyengine_us_sources(
    variables_root: Path,
    *,
    parameters_root: Path | None = None,
) -> _PolicyEngineUSSourceIndex:
    """Thin adapter hook over the single combined source-index implementation."""

    return _build_policyengine_us_source_index(
        variables_root,
        parameters_root=parameters_root,
    )


def _index_policyengine_us_variable_sources(
    variables_root: Path,
) -> Mapping[str, _SourceVariableDefinition]:
    """Compatibility metadata view over the combined source index."""

    return _index_policyengine_us_sources(variables_root).definitions


def _index_policyengine_us_generated_variable_sources(
    package_root: Path,
    *,
    version: str,
) -> Mapping[str, _SourceVariableDefinition]:
    """Return the audited generated-variable snapshot or fail closed."""

    if version != _GENERATED_SOURCE_VERSION:
        raise RuntimeError(
            "PolicyEngine-US generated-variable metadata has not been audited for "
            f"installed version {version!r}; expected {_GENERATED_SOURCE_VERSION!r}."
        )
    for relative_path, expected_digest in _GENERATED_SOURCE_SHA256.items():
        source_path = package_root / relative_path
        try:
            actual_digest = sha256(source_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(
                f"Required PolicyEngine-US generated-variable source is unavailable: "
                f"{source_path}."
            ) from exc
        if actual_digest != expected_digest:
            raise RuntimeError(
                "PolicyEngine-US generated-variable source changed without a "
                f"metadata audit: {source_path}."
            )

    definitions: dict[str, _SourceVariableDefinition] = {}
    for names, entity, dtype, formula_owned in _GENERATED_VARIABLE_GROUPS:
        for name in names:
            if name in definitions:
                raise RuntimeError(
                    f"Duplicate audited PolicyEngine-US generated variable {name!r}."
                )
            definitions[name] = _SourceVariableDefinition(
                metadata=VariableMetadata(
                    name=name,
                    entity=entity,
                    dtype=dtype,
                    period="year",
                ),
                always_computed=formula_owned,
                formula_starts=(),
            )
    return MappingProxyType(definitions)


@lru_cache(maxsize=1)
def _installed_policyengine_us_variable_sources() -> _PolicyEngineUSSourceIndex:
    try:
        package = distribution("policyengine-us")
    except PackageNotFoundError as exc:
        raise ImportError(
            "The PolicyEngine-US metadata index requires the 'policyengine-us' "
            "package. Install it with 'microcosm-frame[policyengine]'."
        ) from exc
    package_root = Path(package.locate_file("policyengine_us"))
    variables_root = package_root / "variables"
    if not variables_root.is_dir():
        raise RuntimeError(
            "The installed PolicyEngine-US variable source tree is unavailable "
            f"at {variables_root}."
        )
    generated = _index_policyengine_us_generated_variable_sources(
        package_root,
        version=package.version,
    )
    source_index = _index_policyengine_us_sources(
        variables_root,
        parameters_root=package_root / "parameters",
    )
    definitions = dict(source_index.definitions)
    duplicates = sorted(set(definitions) & set(generated))
    if duplicates:
        raise RuntimeError(
            "PolicyEngine-US generated-variable audit overlaps ordinary source "
            f"classes: {duplicates}."
        )
    definitions.update(generated)
    return _PolicyEngineUSSourceIndex(
        definitions=MappingProxyType(definitions),
        consumers=source_index.consumers,
    )


class PolicyEngineUSVariableMetadataIndex:
    """Import-free PolicyEngine-US variable metadata read from installed source.

    Importing :mod:`policyengine_us` constructs a complete tax-benefit system,
    and constructing an adapter system registers thousands more variable
    modules. Ownership and physical-dtype guards need only the variable class
    declarations, so this index parses those declarations once and retains only
    compact metadata. The default system's generated variables come from a
    compact audited snapshot whose source and activation files are fingerprinted;
    an unreviewed wheel version or source change fails closed.
    """

    def __init__(self) -> None:
        source_index = _installed_policyengine_us_variable_sources()
        self._definitions = source_index.definitions
        self._consumers = source_index.consumers
        self._engine_version = distribution("policyengine-us").version

    def variable_metadata(self, name: str) -> VariableMetadata:
        definition = self._definitions.get(name)
        if definition is None:
            raise ValueError(f"Unknown PolicyEngine-US source variable {name!r}.")
        return definition.metadata

    def variables(self) -> list[str]:
        return sorted(
            name
            for name, definition in self._definitions.items()
            if name not in _FORMULA_OWNED_COMPAT_COLUMNS
            and not definition.formula_owned
        )

    def consumer_receipts(self, name: str) -> tuple[ConsumerReceipt, ...]:
        """Return immutable external reference sites for an engine variable."""

        if name not in self._definitions:
            raise ValueError(f"Unknown PolicyEngine-US source variable {name!r}.")
        return self._consumers.get(name, ())

    def variable_dependency_closure(self, name: str) -> VariableDependencyClosure:
        """Return the statically authenticated transitive graph for ``name``.

        The source index records references in the target-to-consumer
        direction.  This method inverts those receipts, walks outward from the
        requested output, and classifies each reachable definition exactly as
        :meth:`variables` does.  Multiple source sites for the same reference
        collapse to one semantic edge.
        """

        if name not in self._definitions:
            raise ValueError(f"Unknown PolicyEngine-US source variable {name!r}.")

        dependencies: dict[str, set[str]] = {}
        for target, receipts in self._consumers.items():
            for receipt in receipts:
                if receipt.consumer in self._definitions:
                    dependencies.setdefault(receipt.consumer, set()).add(target)

        reachable: set[str] = set()
        edges: set[tuple[str, str]] = set()
        pending = [name]
        while pending:
            consumer = pending.pop()
            if consumer in reachable:
                continue
            reachable.add(consumer)
            for target in dependencies.get(consumer, ()):
                edges.add((consumer, target))
                if target not in reachable:
                    pending.append(target)

        inputs = set(self.variables())
        input_leaves = tuple(sorted(reachable & inputs))
        formula_nodes = tuple(sorted(reachable - inputs))
        ordered_edges = tuple(sorted(edges))
        payload = {
            "engine_version": self._engine_version,
            "root": name,
            "input_leaves": list(input_leaves),
            "formula_nodes": list(formula_nodes),
            "edges": [list(edge) for edge in ordered_edges],
        }
        digest = sha256(
            json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return VariableDependencyClosure(
            engine_version=self._engine_version,
            root=name,
            input_leaves=input_leaves,
            formula_nodes=formula_nodes,
            edges=ordered_edges,
            sha256=digest,
        )

    def formula_owned_outputs(self, names: Iterable[str]) -> set[str]:
        requested = set(names)
        return set(requested & _FORMULA_OWNED_COMPAT_COLUMNS) | {
            name
            for name in requested
            if (definition := self._definitions.get(name)) is not None
            and definition.formula_owned
        }

    def _engine_computed_columns(
        self,
        tables: Mapping[str, pd.DataFrame],
        *,
        period: int | str,
    ) -> set[str]:
        present = {column for frame in tables.values() for column in frame.columns}
        structural = {US_SCHEMA.person_id_column} | {
            column
            for group in US_SCHEMA.group_entities
            for column in (
                US_SCHEMA.id_column(group),
                US_SCHEMA.membership_column(group),
            )
        }
        return set(present & _FORMULA_OWNED_COMPAT_COLUMNS) | {
            name
            for name in present
            if name not in structural
            and (definition := self._definitions.get(name)) is not None
            and definition.computed_at(period)
        }


def _is_engine_computed(variable: Any, period: int | str | None = None) -> bool:
    """Return whether a PolicyEngine variable is computed by a formula.

    Input variables (read from data, what a pool must produce) are plain source
    variables. Formula-owned outputs may be backed by a direct formula or a
    formula mapping keyed by start date.
    """
    if getattr(variable, "adds", None) or getattr(variable, "subtracts", None):
        return True
    if period is not None:
        return variable.get_formula(str(period)) is not None
    if getattr(variable, "formula", None) is not None:
        return True
    formulas = getattr(variable, "formulas", None)
    return bool(formulas)


def _references_variable(consumer: Any, target: str) -> bool:
    """Whether ``consumer`` reads ``target`` in a way that makes it load-bearing.

    A take-up flag matters to the model only if some other variable consumes
    it: through an ``adds``/``subtracts`` aggregation (e.g. ``*_enrolled``
    variables that add a take-up flag), through a ``defined_for`` gate, or by
    naming it inside a formula body. A flag no variable reads is dead — the
    engine's own default never reaches an output, so seeding it in the dataset
    changes nothing.
    """
    import inspect

    for attr in ("adds", "subtracts"):
        value = getattr(consumer, attr, None)
        if isinstance(value, (list, tuple)) and target in value:
            return True
        if isinstance(value, str) and value == target:
            return True
    if getattr(consumer, "defined_for", None) == target:
        return True
    for attribute in dir(consumer):
        if attribute != "formula" and not attribute.startswith("formula_"):
            continue
        formula = getattr(consumer, attribute, None)
        if not callable(formula):
            continue
        try:
            source = inspect.getsource(formula)
        except (OSError, TypeError):
            continue
        if target in source:
            return True
    return False


def _enum_domain(variable: Any) -> tuple[str, ...]:
    possible_values = getattr(variable, "possible_values", None)
    members = getattr(possible_values, "__members__", None)
    if isinstance(members, Mapping):
        return tuple(str(name) for name in members)
    return ()


def _stored_enum_name(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, bytes):
        return value.decode()
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


class PolicyEngineUSEngine:
    """RulesEngine adapter backed by ``policyengine_us``.

    Args:
        contract: Column-parity contract for :meth:`write_dataset` exports.
            ``None`` means an empty contract (no required/forbidden/closed
            surface checks).
        defaults: Scalar defaults broadcast onto the owning entity table for
            contract-required columns no bundle table provides.

    The PolicyEngine tax-benefit system is instantiated lazily and cached on
    first metadata lookup, so constructing the adapter never imports
    ``policyengine_us``.
    """

    country = "us"

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
            if name not in _FORMULA_OWNED_COMPAT_COLUMNS
            and not _is_engine_computed(variable)
        )

    def formula_owned_outputs(self, names: Iterable[str]) -> set[str]:
        """Return which of ``names`` are engine formula-owned, not input leaves.

        A name is formula-owned when the tax-benefit system computes it —
        directly or through an ``adds``/``subtracts`` aggregation or a
        start-date formula mapping — or when it is one of the
        compatibility-blocked aggregates the adapter refuses to persist even
        though some published wheels still report them as inputs
        (:data:`_FORMULA_OWNED_COMPAT_COLUMNS`). This is the complement of
        :meth:`variables` restricted to ``names``: persisting a formula-owned
        variable as a dataset input pins its baseline and masks reforms, so
        callers deriving an imputation/export surface reject exactly this set
        instead of maintaining a hand-written blocklist that goes stale as
        PolicyEngine-US adds variables (microcosm issue #301).

        Names unknown to the tax-benefit system are not flagged: they cannot be
        classified as formula-owned here, and the export/enum guards own
        unknown columns.

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
        """
        variables = self._tax_benefit_system().variables
        flagged: set[str] = set()
        for name in names:
            if name in _FORMULA_OWNED_COMPAT_COLUMNS:
                flagged.add(name)
                continue
            variable = variables.get(name)
            if variable is not None and _is_engine_computed(variable):
                flagged.add(name)
        return flagged

    def take_up_variables(self) -> list[str]:
        """Return the engine's take-up-flag variable names, sorted.

        A take-up flag is a boolean variable whose name begins ``takes_up`` or
        carries a ``take_up_seed`` marker (the model-side draw seed some
        programs migrate to). The set is discovered from engine metadata rather
        than hard-coded, so a take-up variable PolicyEngine-US adds is picked up
        automatically (microcosm issue #312).

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
        """
        system_variables = self._tax_benefit_system().variables
        return sorted(
            name
            for name in system_variables
            if name.startswith("takes_up") or "take_up_seed" in name
        )

    def take_up_contract(self) -> dict[str, dict[str, object]]:
        """Classify every take-up variable against the installed engine.

        For each name from :meth:`take_up_variables`, report the engine facts
        that decide whether the dataset must seed the flag:

        - ``entity`` — the entity the flag lives on.
        - ``value_type`` — ``"bool"``, ``"int"``, ... (the Python type name).
        - ``default`` — the engine default (``True`` means "everyone eligible
          takes up unless the dataset says otherwise").
        - ``engine_computed`` — whether PolicyEngine-US computes the flag with
          a formula, ``adds``/``subtracts`` aggregation, or a start-date
          formula mapping. ``True`` means the model draws take-up itself and
          the dataset must NOT seed the flag (it would fight the draw).
        - ``consumers`` — the variables that read the flag (empty means dead:
          seeding it changes no output).
        - ``engine_class`` — the derived class:
            ``"model_simulated"`` if ``engine_computed``;
            ``"dead"`` if no consumer reads it;
            ``"data_seeded"`` otherwise (an input leaf defaulting to universal
            take-up that the dataset must populate or ship known-wrong
            participation).

        This is the engine-derived half of the take-up contract inventory: a
        checked-in table records the intended per-program treatment and a test
        asserts it against this method, so the classification tracks the pinned
        engine version instead of a remembered snapshot (same
        metadata-derivation doctrine as :meth:`formula_owned_outputs`, microcosm
        issue #312).

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
        """
        system_variables = self._tax_benefit_system().variables
        names = self.take_up_variables()
        name_set = set(names)
        consumers: dict[str, list[str]] = {name: [] for name in names}
        for consumer_name, consumer in system_variables.items():
            if consumer_name in name_set:
                continue
            for target in names:
                if _references_variable(consumer, target):
                    consumers[target].append(consumer_name)

        contract: dict[str, dict[str, object]] = {}
        for name in names:
            variable = system_variables[name]
            engine_computed = _is_engine_computed(variable)
            reads = sorted(consumers[name])
            if engine_computed:
                engine_class = "model_simulated"
            elif not reads:
                engine_class = "dead"
            else:
                engine_class = "data_seeded"
            default = getattr(variable, "default_value", None)
            if variable.value_type in _DTYPE_KIND_BY_VALUE_TYPE:
                default_value: object = default
                value_type = variable.value_type.__name__
            else:
                default_value = _stored_enum_name(default)
                value_type = "enum"
            contract[name] = {
                "entity": variable.entity.key,
                "value_type": value_type,
                "default": default_value,
                "engine_computed": engine_computed,
                "consumers": reads,
                "engine_class": engine_class,
            }
        return contract

    def default_values(self, names: Sequence[str]) -> dict[str, object]:
        """Return engine default values for the given input variable names.

        Only names the tax-benefit system knows as non-formula input
        variables with a declared default are returned; unknown names and
        formula-owned variables are silently omitted, so callers can pass a
        whole export surface. Enum defaults are normalized to their stored
        member name (the representation datasets persist).

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
        """
        variables = self._tax_benefit_system().variables
        defaults: dict[str, object] = {}
        for name in names:
            variable = variables.get(name)
            if variable is None or _is_engine_computed(variable):
                continue
            default = getattr(variable, "default_value", None)
            if default is None:
                continue
            if variable.value_type not in _DTYPE_KIND_BY_VALUE_TYPE:
                stored = _stored_enum_name(default)
                if stored is None:
                    continue
                defaults[name] = stored
            else:
                defaults[name] = default
        return defaults

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

        Applies the export gate: forbidden and formula-owned columns block the
        export, defaults are broadcast onto the owning entity table for
        required columns no table provides, closed contracts reject unexpected
        non-structural columns, and a dataset with violations is never
        written. After writing, the dataset is reloaded and every persisted
        column verified (round-trip check).

        Args:
            bundle: A US-schema bundle.
            path: Destination ``.h5`` path.
            period: Dataset time period (e.g. ``2026``).

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
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

        forbidden_present = set(contract.forbidden).intersection(present_columns)
        formula_owned_present = self._engine_computed_columns(
            tables, period=period
        ) | set(contract.formula_owned_excluded).intersection(present_columns)
        unexpected: set[str] = set()
        if contract.closed:
            allowed = (
                set(contract.required)
                | set(contract.optional)
                | self._structural_columns()
                | {_HOUSEHOLD_WEIGHT_COLUMN}
            )
            unexpected = present_columns - allowed

        enum_domain_failures = self._enum_domain_failures(tables)

        if (
            forbidden_present
            or missing_required
            or formula_owned_present
            or unexpected
            or enum_domain_failures
        ):
            raise ValueError(
                "Export contract violated; nothing was written. Missing "
                f"required column(s): {sorted(missing_required)}; forbidden "
                f"column(s) present: {sorted(forbidden_present)}; formula-owned "
                f"column(s) present: {sorted(formula_owned_present)}; unexpected column(s) "
                f"present: {sorted(unexpected)}; enum-domain violation(s): "
                f"{enum_domain_failures}."
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
                "package. Install it with 'microcosm-frame[policyengine]'."
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
        # The export contract materializes household weights only, so the
        # entity set is pinned rather than inherited from the bundle's
        # weighted entities.
        tables = engine_tables(bundle, weighted_entities=("household",))
        return {name: tables[name] for name in expected}

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

    def _engine_computed_columns(
        self,
        tables: Mapping[str, pd.DataFrame],
        *,
        period: int | str,
    ) -> set[str]:
        """PolicyEngine-computed columns present in the pending export.

        Formula-owned columns cannot be allowed through implicitly: if a
        source table carries a PolicyEngine output name such as ``ssi``,
        keeping it in the HDF5 file turns that formula output into an input
        and masks reforms. Such columns must be removed upstream before the
        writer is called, after checking aggregate deltas.
        """
        variables = self._tax_benefit_system().variables
        present = {column for frame in tables.values() for column in frame.columns}
        structural = self._structural_columns()
        return set(present & _FORMULA_OWNED_COMPAT_COLUMNS) | {
            column
            for column in present
            if column not in structural
            and column in variables
            and _is_engine_computed(variables[column], period=period)
        }

    def _enum_domain_failures(
        self,
        tables: Mapping[str, pd.DataFrame],
    ) -> list[str]:
        """Return enum input columns carrying values outside engine domains."""
        variables = self._tax_benefit_system().variables
        structural = self._structural_columns()
        failures: list[str] = []
        for entity, frame in tables.items():
            for column in frame.columns:
                if column in structural or column not in variables:
                    continue
                allowed = set(_enum_domain(variables[column]))
                if not allowed:
                    continue
                invalid: list[str] = []
                for value in frame[column].to_numpy(dtype=object):
                    name = _stored_enum_name(value)
                    if name not in allowed:
                        invalid.append("<missing>" if name is None else name)
                if invalid:
                    failures.append(
                        f"{entity}.{column}: {len(invalid)}/{len(frame)} value(s) "
                        "outside enum domain; invalid examples "
                        f"{sorted(set(invalid))[:8]}; allowed values "
                        f"{sorted(allowed)[:8]}"
                    )
        return failures

    def _structural_columns(self) -> set[str]:
        """Entity ids and memberships required to reconstruct the frame."""
        schema = self.entity_schema()
        return {schema.person_id_column} | {
            column
            for group in schema.group_entities
            for column in (schema.id_column(group), schema.membership_column(group))
        }

    def _write_and_verify(
        self,
        tables: Mapping[str, pd.DataFrame],
        *,
        period: int,
        output_path: Path,
    ) -> None:
        """Persist PolicyEngine-US tables and verify its dataset round-trip.

        This owns the same entity-table HDF layout as ``USSingleYearDataset``
        while routing every Frame table through Microcosm's nullable-boolean
        boundary. It then reloads through ``USSingleYearDataset`` and asserts
        every column from a non-empty table survived.

        Raises:
            ValueError: If a column expected after reload is missing.
        """
        from policyengine_us.data import USSingleYearDataset

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        materialized_tables = {
            name: materialize_nullable_booleans_for_pytables(table).table
            for name, table in tables.items()
        }
        with pd.HDFStore(str(output_path), mode="w") as store:
            for name in (_PERSON_TABLE, *_GROUP_TABLES):
                table = tables[name]
                if len(table) > 0:
                    put_frame_table(
                        store,
                        name,
                        table,
                        preferred_format="table",
                        data_columns=True,
                    )
            store.put(
                "_time_period",
                pd.Series([int(period)]),
                format="table",
            )

        expected_columns: set[str] = set()
        for frame in tables.values():
            if len(frame) > 0:
                expected_columns.update(frame.columns)

        reloaded = USSingleYearDataset(file_path=str(output_path))
        with pd.HDFStore(str(output_path), mode="r") as store:
            logical_tables = {
                name: read_frame_table(store, name)
                for name in (_PERSON_TABLE, *_GROUP_TABLES)
                if len(tables[name]) > 0
            }
        persisted_columns: set[str] = set()
        dtype_mismatches: list[str] = []
        for name in (_PERSON_TABLE, *_GROUP_TABLES):
            external_table = getattr(reloaded, name)
            reloaded_table = logical_tables.get(name, external_table)
            persisted_columns.update(external_table.columns)
            source_table = materialized_tables.get(name)
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
