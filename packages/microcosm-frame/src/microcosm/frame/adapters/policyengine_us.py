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

import ast
import importlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
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
from microcosm.frame.adapters._policyengine_us_source_index import (
    _index_policyengine_us_variable_sources as _build_policyengine_us_variable_index,
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

# --- BEGIN GENERATED VARIABLE AUDIT ---
# Regenerate with:
#   uv run python tools/refresh_us_generated_variable_audit.py
# The block below is generated from the installed wheels; never edit a
# digest or a name by hand. 119 default-system variables are
# created outside ordinary top-level ``class ...(Variable)``
# declarations, so the snapshot is tied to every source and activation
# surface that produced it: a changed wheel fails closed until this
# audit is refreshed, and never silently omits a newly generated
# formula-owned output.
_GENERATED_SOURCE_VERSION = "2.2.1"
_GENERATED_SOURCE_SHA256: dict[str, str] = {
    "model_api.py": "d7edb7436b84733f179fe223376fb588bb7a3ad6817d119703faeb599d4bb9c7",
    "reforms/reforms.py": (
        "b4077ecee0342080f9a738422f2275479f1b8923366700aece79ef92c64baa61"
    ),
    "reforms/states/mi/surtax.py": (
        "e1d0c0207c46243d3509b22b15fbdc07aa02b4df9461f7b93bec872dc7124ea9"
    ),
    "system.py": "1c8539dcb8aeba4973823887895f5cd42bb9a2ee1270b0a947c9e7c185571302",
    "variables/gov/puf.py": (
        "17545c43549ecf34016107bc8ed2dce25a53610afb802431a1b0ea6724215e7b"
    ),
    "variables/gov/states/tax/income/_generate_state_mfs_variables.py": (
        "a0c9decd81b6eb76ac7edcddfc913d89ee86e0f18d8c51702bcb2a015dc2fabe"
    ),
    "variables/household/demographic/geographic/state/in_state.py": (
        "a3792c642387b652752461c85c03e5a9cb39fab55b4e038270374dd7e7d8aa60"
    ),
}
_GENERATED_SPM_SOURCE_VERSION = "1.0.0"
_GENERATED_SPM_SOURCE_SHA256: dict[str, str] = {
    "policyengine_adapter.py": (
        "aa5c20cd94abd3287bec097e6f3544f1c822d708cf63cf34c9ce294c3a1e85f7"
    ),
}
_GENERATED_VARIABLE_GROUPS: tuple[tuple[tuple[str, ...], str, str, str, bool], ...] = (
    (
        ("mi_surtax",),
        "tax_unit",
        "float",
        "year",
        True,
    ),
    (
        tuple(
            "e00700 e01100 e01200 e02000 e03220 e03230 e03240 e03270 "
            "e03290 e03300 e03400 e03500 e07240 e07260 e07300 e09700 "
            "e09800 e09900 e11200 e18500 e19200 e19800 e20100 e20400 "
            "e24515 e24518 e26270 e27200 e32800 e58990 e62900 e87521 "
            "e87530 p08000".split()
        ),
        "person",
        "float",
        "year",
        False,
    ),
    (
        tuple(
            "ar_agi ar_itemized_deductions ar_standard_deduction "
            "ar_taxable_income dc_taxable_income de_agi "
            "de_itemized_deductions de_standard_deduction "
            "de_taxable_income ia_agi ia_itemized_deductions "
            "ia_standard_deduction ia_taxable_income "
            "ky_itemized_deductions ky_standard_deduction "
            "ky_taxable_income ms_itemized_deductions "
            "ms_standard_deduction ms_taxable_income "
            "mt_itemized_deductions mt_standard_deduction "
            "mt_taxable_income".split()
        ),
        "tax_unit",
        "float",
        "year",
        True,
    ),
    (
        tuple(
            "AK AL AR AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY LA MA "
            "MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA PR "
            "RI SC SD TN TX UT VA VI VT WA WI WV WY".split()
        ),
        "household",
        "bool",
        "year",
        True,
    ),
    (
        ("is_household_spouse",),
        "person",
        "bool",
        "point",
        False,
    ),
    (
        ("is_spm_independent_minor_role",),
        "person",
        "bool",
        "point",
        True,
    ),
    (
        tuple(
            "spm_unit_geographic_adjustment "
            "spm_unit_reference_spm_threshold spm_unit_spm_threshold "
            "spm_unit_spm_threshold_housing_portion "
            "spm_unit_unadjusted_spm_threshold".split()
        ),
        "spm_unit",
        "float",
        "year",
        True,
    ),
    (
        tuple("spm_measurement_adults spm_measurement_children".split()),
        "spm_unit",
        "int",
        "year",
        True,
    ),
)
# --- END GENERATED VARIABLE AUDIT ---


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
    """Shared declaration parser without parameter-dependent consumer work."""

    return _build_policyengine_us_variable_index(variables_root)


def _audit_pinned_sources(
    label: str,
    package_root: Path,
    version: str,
    *,
    expected_version: str,
    digests: Mapping[str, str],
) -> None:
    """Fail closed unless the installed distribution is the audited one."""

    if version != expected_version:
        raise RuntimeError(
            "PolicyEngine-US generated-variable metadata has not been audited "
            f"for installed {label} version {version!r}; expected "
            f"{expected_version!r}."
        )
    for relative_path, expected_digest in digests.items():
        source_path = package_root / relative_path
        try:
            actual_digest = sha256(source_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(
                f"Required PolicyEngine-US generated-variable source is "
                f"unavailable: {source_path}."
            ) from exc
        if actual_digest != expected_digest:
            raise RuntimeError(
                "PolicyEngine-US generated-variable source changed without a "
                f"metadata audit: {source_path}."
            )


def _index_policyengine_us_generated_variable_sources(
    package_root: Path,
    *,
    version: str,
    spm_package_root: Path,
    spm_version: str,
) -> Mapping[str, _SourceVariableDefinition]:
    """Return the audited generated-variable snapshot or fail closed.

    Two distributions produce the default system's generated variables:
    policyengine-us itself (the 50-state flags, the PUF leaves, the state MFS
    factory and the Michigan surtax reform) and spm-calculator, whose
    ``build_policyengine_variables`` ``system.py`` calls to install the SPM
    measurement thresholds and independence roles. Both are pinned, because a
    new spm-calculator alone can add a formula-owned output.
    """

    _audit_pinned_sources(
        "policyengine-us",
        package_root,
        version,
        expected_version=_GENERATED_SOURCE_VERSION,
        digests=_GENERATED_SOURCE_SHA256,
    )
    _audit_pinned_sources(
        "spm-calculator",
        spm_package_root,
        spm_version,
        expected_version=_GENERATED_SPM_SOURCE_VERSION,
        digests=_GENERATED_SPM_SOURCE_SHA256,
    )

    definitions: dict[str, _SourceVariableDefinition] = {}
    for names, entity, dtype, period, formula_owned in _GENERATED_VARIABLE_GROUPS:
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
                    period=period,
                ),
                always_computed=formula_owned,
                formula_starts=(),
            )
    return MappingProxyType(definitions)


def _installed_policyengine_us_source_parts():
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
    try:
        spm_package = distribution("spm-calculator")
    except PackageNotFoundError as exc:
        raise ImportError(
            "The PolicyEngine-US metadata index requires the 'spm-calculator' "
            "package, which policyengine-us 2.x installs to generate the SPM "
            "measurement variables. Install it with "
            "'microcosm-frame[policyengine]'."
        ) from exc
    generated = _index_policyengine_us_generated_variable_sources(
        package_root,
        version=package.version,
        spm_package_root=Path(spm_package.locate_file("spm_calculator")),
        spm_version=spm_package.version,
    )
    source_inputs = _source_dataset_source_inputs(
        package_root, Path(spm_package.locate_file("spm_calculator"))
    )
    return package_root, variables_root, generated, source_inputs


def _merge_installed_variable_definitions(ordinary, generated, source_inputs):
    definitions = dict(ordinary)
    duplicates = sorted(set(definitions) & set(generated))
    if duplicates:
        raise RuntimeError(
            "PolicyEngine-US generated-variable audit overlaps ordinary source "
            f"classes: {duplicates}."
        )
    definitions.update(generated)
    _validate_source_input_names(source_inputs, definitions)
    # Preserve the audited formula snapshot. Dataset ownership is a separate
    # country declaration; both ordinary and generated fallbacks obey it.
    for name in source_inputs:
        definitions[name] = replace(
            definitions[name], always_computed=False, formula_starts=()
        )
    return MappingProxyType(definitions)


@lru_cache(maxsize=1)
def _installed_policyengine_us_variable_definitions():
    """Installed ordinary/generated ownership metadata, with the same audits."""
    _, variables_root, generated, source_inputs = (
        _installed_policyengine_us_source_parts()
    )
    return _merge_installed_variable_definitions(
        _index_policyengine_us_variable_sources(variables_root),
        generated,
        source_inputs,
    )


@lru_cache(maxsize=1)
def _installed_policyengine_us_variable_sources() -> _PolicyEngineUSSourceIndex:
    package_root, variables_root, generated, source_inputs = (
        _installed_policyengine_us_source_parts()
    )
    source_index = _index_policyengine_us_sources(
        variables_root,
        parameters_root=package_root / "parameters",
    )
    return _PolicyEngineUSSourceIndex(
        definitions=_merge_installed_variable_definitions(
            source_index.definitions, generated, source_inputs
        ),
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

    def __init__(self, *, include_consumers: bool = True) -> None:
        if type(include_consumers) is not bool:
            raise TypeError("include_consumers must be an explicit boolean.")
        if include_consumers:
            source_index = _installed_policyengine_us_variable_sources()
            self._definitions = source_index.definitions
            self._consumers = source_index.consumers
        else:
            self._definitions = _installed_policyengine_us_variable_definitions()
            self._consumers = None
        self._engine_version = distribution("policyengine-us").version

    def _consumer_index(self):
        if self._consumers is None:
            source_index = _installed_policyengine_us_variable_sources()
            if source_index.definitions != self._definitions:
                raise RuntimeError(
                    "PolicyEngine-US definitions changed before consumer indexing."
                )
            self._consumers = source_index.consumers
        return self._consumers

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
        return self._consumer_index().get(name, ())

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
        for target, receipts in self._consumer_index().items():
            for receipt in receipts:
                if receipt.consumer in self._definitions:
                    dependencies.setdefault(receipt.consumer, set()).add(target)

        reachable: set[str] = set()
        edges: set[tuple[str, str]] = set()
        inputs = set(self.variables())
        pending = [name]
        while pending:
            consumer = pending.pop()
            if consumer in reachable:
                continue
            reachable.add(consumer)
            if consumer in inputs:
                # A declared source observation replaces its household fallback;
                # the fallback's own dependencies are not dataset requirements.
                continue
            for target in dependencies.get(consumer, ()):
                edges.add((consumer, target))
                if target not in reachable:
                    pending.append(target)

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


def _validate_dataset_source_inputs(
    declared: object, rejected: object
) -> frozenset[str]:
    """Validate the consuming country's explicit source/derived boundary."""
    for name, value in (
        ("DATASET_SOURCE_INPUTS", declared),
        ("REJECTED_DATASET_INPUTS", rejected),
    ):
        if type(value) is not frozenset or not all(
            type(item) is str and item.isidentifier() for item in value
        ):
            raise RuntimeError(f"{name} must be a frozenset of variable names.")
    if not declared:
        raise RuntimeError("DATASET_SOURCE_INPUTS must not be empty.")
    overlap = declared & (rejected | _FORMULA_OWNED_COMPAT_COLUMNS)
    if overlap:
        raise RuntimeError(
            "Country dataset source inputs overlap rejected/formula-owned "
            f"inputs: {sorted(overlap)}."
        )
    return declared


def _validate_source_input_names(source_inputs, variables):
    unknown = source_inputs - variables.keys()
    if unknown:
        raise RuntimeError(f"Unknown country dataset source inputs: {sorted(unknown)}.")


def _engine_dataset_source_inputs(policyengine_us: Any) -> frozenset[str]:
    """Read source ownership from the consuming engine, never a local name list."""
    return _engine_dataset_input_declarations(policyengine_us)[0]


def _engine_dataset_input_declarations(
    policyengine_us: Any,
) -> tuple[frozenset[str], frozenset[str]]:
    spm = importlib.import_module(f"{policyengine_us.__name__}.spm")
    rejected = getattr(spm, "REJECTED_DATASET_INPUTS", None)
    declared = _validate_dataset_source_inputs(
        getattr(spm, "DATASET_SOURCE_INPUTS", None), rejected
    )
    return declared, rejected


def _source_dataset_source_inputs(
    package_root: Path, spm_package_root: Path
) -> frozenset[str]:
    """Read the same declarations statically, without importing either model.

    Only literal containers inside frozenset, immutable constant references/unions and
    the calculator's FORMULA_OWNED_INPUTS import are supported. Unknown or
    rebound expressions refuse instead of executing country initialization.
    Generated-variable version/source audits remain independent and unchanged.
    """
    country_path = package_root / "spm.py"
    calculator_path = spm_package_root / "policyengine_adapter.py"
    trees = {}
    visiting = set()

    def binds_name(statement, name):
        """Inspect explicit import-time bindings, excluding function bodies."""
        pending = [statement]
        while pending:
            node = pending.pop()
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.id == name
            ):
                return True
            if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or alias.name.split(".")[0]) == name
                for alias in node.names
            ):
                return True
            if (
                isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.ExceptHandler,
                        ast.MatchAs,
                        ast.MatchStar,
                    ),
                )
                and node.name == name
            ):
                return True
            if isinstance(node, ast.MatchMapping) and node.rest == name:
                return True
            for field, child in ast.iter_fields(node):
                if field == "body" and isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                if isinstance(child, ast.AST):
                    pending.append(child)
                elif isinstance(child, list):
                    pending.extend(item for item in child if isinstance(item, ast.AST))
        return False

    def declaration(path, name):
        key = (path, name)
        if key in visiting:
            raise RuntimeError(f"Cyclic dataset ownership declaration: {name}.")
        if path not in trees:
            try:
                trees[path] = ast.parse(path.read_text(), filename=str(path))
            except (OSError, SyntaxError) as exc:
                raise RuntimeError(
                    f"Dataset ownership source unavailable: {path}."
                ) from exc
            if any(
                isinstance(node, ast.ImportFrom)
                and any(alias.name == "*" for alias in node.names)
                for node in ast.walk(trees[path])
            ):
                raise RuntimeError(
                    "Wildcard imports cannot establish static dataset ownership."
                )
        bindings = []
        for statement in trees[path].body:
            if isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                if statement.name == name:
                    raise RuntimeError(
                        f"Nonconstant dataset ownership declaration: {name}."
                    )
                if binds_name(statement, name):
                    # Defaults/decorators/bases and class bodies execute while
                    # importing; only function bodies are deferred.
                    raise RuntimeError(
                        f"Dynamic dataset ownership declaration: {name}."
                    )
                continue
            if isinstance(statement, ast.ImportFrom):
                for alias in statement.names:
                    if (alias.asname or alias.name) == name:
                        bindings.append((statement, alias))
            elif isinstance(statement, ast.Import):
                if any(
                    (a.asname or a.name.split(".")[0]) == name for a in statement.names
                ):
                    raise RuntimeError(f"Unsupported dataset ownership import: {name}.")
            elif binds_name(statement, name):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == name
                ):
                    bindings.append((statement.value, None))
                elif (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id == name
                    and statement.value is not None
                ):
                    bindings.append((statement.value, None))
                else:
                    raise RuntimeError(
                        f"Dynamic dataset ownership declaration: {name}."
                    )
        if len(bindings) != 1:
            raise RuntimeError(
                f"Missing or rebound dataset ownership declaration: {name}."
            )
        node, alias = bindings[0]
        visiting.add(key)
        try:
            if alias is not None:
                if (
                    path != country_path
                    or node.level
                    or node.module != "spm_calculator.policyengine_adapter"
                    or alias.name != "FORMULA_OWNED_INPUTS"
                ):
                    raise RuntimeError(f"Unsupported dataset ownership import: {name}.")
                return declaration(calculator_path, alias.name)
            resolved = value(path, node)
            if type(resolved) is not frozenset:
                # A list/set alias may be mutated by a later call without any
                # assignment node. Never infer its final runtime contents.
                raise RuntimeError(
                    f"Dataset ownership constant {name} must be an immutable frozenset."
                )
            return resolved
        finally:
            visiting.remove(key)

    def value(path, node):
        if isinstance(node, ast.Name):
            return declaration(path, node.id)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "frozenset"
            and len(node.args) <= 1
            and not node.keywords
        ):
            # A rebound constructor cannot be interpreted as the builtin.
            shadowed = any(
                binds_name(statement, "frozenset") for statement in trees[path].body
            )
            if shadowed:
                raise RuntimeError(
                    "Rebound frozenset in dataset ownership declaration."
                )
            return frozenset(value(path, node.args[0])) if node.args else frozenset()
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left, right = value(path, node.left), value(path, node.right)
            if type(left) in (set, frozenset) and type(right) in (set, frozenset):
                return left | right
        if isinstance(node, (ast.Set, ast.List, ast.Tuple, ast.Constant)):
            return ast.literal_eval(node)
        raise RuntimeError("Unsupported dataset ownership expression.")

    try:
        return _validate_dataset_source_inputs(
            declaration(country_path, "DATASET_SOURCE_INPUTS"),
            declaration(country_path, "REJECTED_DATASET_INPUTS"),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Malformed dataset ownership declaration.") from exc


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


def _input_representation_reason(series, variable, enum_type) -> str | None:
    """Return a fixed reason only; never include source values in diagnostics."""
    if series.isna().any():
        return "NULL"
    try:
        # No default dtype: np.dtype(None) would silently choose float64.
        if getattr(variable, "dtype", None) is None:
            return "DTYPE"
        dtype = np.dtype(variable.dtype)
    except (TypeError, ValueError):
        return "DTYPE"
    values = np.asarray(series.values)
    value_type = getattr(variable, "value_type", None)
    if value_type is enum_type:
        allowed = _enum_domain(variable)
        if not allowed or dtype.kind not in "iu":
            return "ENUM"
        # Core handles a homogeneous array of actual members, or named inputs.
        # An arbitrary object with a .name (or mixed members/strings) is not
        # equivalent to either input path, even when its name matches.
        named = values.dtype.kind in "OUS" and all(
            isinstance(value, (str, bytes)) for value in values
        )
        possible_values = variable.possible_values
        members = (
            values.dtype.kind == "O"
            and isinstance(possible_values, type)
            and all(isinstance(value, possible_values) for value in values)
        )
        if not named and not members:
            return "ENUM"
        if named:
            # Match Core Enum.encode's array-level normalization, including
            # the different encoding behavior of byte and object arrays.
            if values.dtype.kind == "S":
                values = np.char.decode(values, "utf-8")
            elif values.dtype.kind == "O":
                values = values.astype(str)
        if any(_stored_enum_name(value) not in allowed for value in values):
            return "ENUM"
    elif value_type is int:
        if values.dtype.kind not in "iu" or dtype.kind not in "iu":
            return "INTEGER_DTYPE"
        bounds = np.iinfo(dtype)
        if len(values) and (
            int(values.min()) < bounds.min or int(values.max()) > bounds.max
        ):
            return "INTEGER_RANGE"
        converted = values.astype(dtype)
        if not np.array_equal(converted.astype(values.dtype), values):
            return "INTEGER_LOSS"
    elif value_type is bool:
        if values.dtype.kind != "b" or dtype.kind != "b":
            return "BOOL"
    elif value_type is float:
        if values.dtype.kind not in "iuf" or dtype.kind != "f":
            return "FLOAT"
        with np.errstate(over="ignore", invalid="ignore"):
            converted = values.astype(dtype)
        if (
            np.isnan(converted).any()
            or (np.isfinite(values) & ~np.isfinite(converted)).any()
        ):
            return "FLOAT_RANGE"
    elif value_type is str:
        if dtype.kind not in "OUS" or not all(
            isinstance(value, str) for value in values
        ):
            return "STRING"
        converted = values.astype(dtype)
        if dtype.kind == "S":
            converted = np.char.decode(converted, "ascii")
        if not np.array_equal(converted, values):
            return "STRING_LOSS"
    else:
        return "TYPE"
    return None


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
        spm: Mapping[str, object] | None = None,
    ) -> None:
        self._contract = contract if contract is not None else ExportContract.empty()
        self._defaults = dict(defaults or {})
        # Explicit SPM measurement selection, forwarded verbatim to the engine
        # as ``Microsimulation(spm=...)``.  PolicyEngine-US 2.0.0 stopped
        # inferring SPM geography from an absent county: an SPM-dependent
        # variable now raises ``SPMInputError(SPM_GEOGRAPHY_REQUIRED)`` unless
        # the caller supplies five-digit string county FIPS or selects
        # ``{"geography_kind": "national"}`` (or a fixed ``"metro"`` area with
        # its ``geography_id``).  ``None`` keeps the engine default, which is
        # county measurement, so a Frame that already carries ``county_fips``
        # is measured on its own counties exactly as before.  A stage that runs
        # before geography assignment must pass the national selection rather
        # than let the default raise.
        self._spm = None if spm is None else dict(spm)
        self._system: Any = None

    # ------------------------------------------------------------------
    # Variable metadata
    # ------------------------------------------------------------------

    def validate_input_representation(
        self, bundle: Frame, *, period: int | str
    ) -> None:
        """Check inputs against this consumer's effective variable metadata.

        Copies entity tables and materializes the typed household weight; never
        applies defaults, mutates source cells, constructs a dataset or calculates.
        Success establishes only the supported representation profile, not source
        authority, runtime admission, scientific validity or simulation readiness.
        In particular, ordinary float rounding and representable infinities do not
        become scientific qualifications. Unsupported null inputs are not filled.
        """
        from policyengine_core.enums import Enum
        from policyengine_core.periods import period as parse_period

        try:
            if type(period) not in (int, str) or (
                isinstance(period, str)
                and (not period.isascii() or not period.isdecimal())
            ):
                raise ValueError
            selected = parse_period(period)
            year = int(period)
            if (
                selected.unit != "year"
                or selected.size != 1
                or selected.start.date.isoformat() != f"{year:04d}-01-01"
            ):
                raise ValueError
        except (TypeError, ValueError, AttributeError, OverflowError):
            raise ValueError("INPUT_REPRESENTATION_PERIOD") from None

        variables = self._tax_benefit_system().variables
        declared, rejected = _engine_dataset_input_declarations(
            self._import_policyengine_us()
        )
        _validate_source_input_names(declared, variables)
        tables = self._engine_tables(bundle)
        computed = self._engine_computed_columns(tables, period=period)
        for entity, table in tables.items():
            for column in table:
                variable = variables.get(column)
                reason = None
                if variable is None:
                    reason = "UNKNOWN"
                elif getattr(getattr(variable, "entity", None), "key", None) != entity:
                    reason = "ENTITY"
                elif column in rejected or column in computed:
                    reason = "OWNERSHIP"
                elif getattr(variable, "definition_period", None) not in (
                    "year",
                    "eternity",
                ):
                    reason = "PERIOD"
                elif getattr(variable, "is_neutralized", False):
                    reason = "NEUTRALIZED"
                else:
                    try:
                        end = getattr(variable, "end", None)
                        if end is not None and selected.start.date > end:
                            reason = "EXPIRED"
                        else:
                            reason = _input_representation_reason(
                                table[column], variable, Enum
                            )
                    except (TypeError, ValueError, AttributeError, OverflowError):
                        # Conversion/encoding failures can contain row examples.
                        reason = "UNSUPPORTED"
                if reason:
                    raise ValueError(
                        f"INPUT_REPRESENTATION_{reason}: {entity}.{column}"
                    ) from None

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
        """Return input leaves, including the country's declared source inputs.

        Computed/formula-owned variables are excluded — a pool produces inputs,
        not outputs.

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
        """
        system_variables = self._tax_benefit_system().variables
        source_inputs = self._dataset_source_inputs()
        return sorted(
            name
            for name, variable in system_variables.items()
            if name not in _FORMULA_OWNED_COMPAT_COLUMNS
            and (name in source_inputs or not _is_engine_computed(variable))
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
        source_inputs = self._dataset_source_inputs()
        flagged: set[str] = set()
        for name in names:
            if name in _FORMULA_OWNED_COMPAT_COLUMNS:
                flagged.add(name)
                continue
            if name in source_inputs:
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
        whole export surface. Declared source inputs are also omitted: accepting
        an observation does not authorize inventing a missing one.
        Enum defaults are normalized to their stored
        member name (the representation datasets persist).

        Raises:
            ImportError: If ``policyengine_us`` is not installed.
        """
        variables = self._tax_benefit_system().variables
        source_inputs = self._dataset_source_inputs()
        defaults: dict[str, object] = {}
        for name in names:
            variable = variables.get(name)
            if (
                variable is None
                or name in source_inputs
                or _is_engine_computed(variable)
            ):
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
        simulation = microsimulation_class(
            dataset=dataset,
            **({"spm": dict(self._spm)} if self._spm is not None else {}),
        )
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

    def _dataset_source_inputs(self) -> frozenset[str]:
        """Source ownership is independent of a household fallback formula."""
        declared = _engine_dataset_source_inputs(self._import_policyengine_us())
        _validate_source_input_names(declared, self._tax_benefit_system().variables)
        return declared

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
            self._system = self._import_policyengine_us().CountryTaxBenefitSystem(
                **({"spm": dict(self._spm)} if self._spm is not None else {})
            )
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
        source_inputs = self._dataset_source_inputs()
        present = {column for frame in tables.values() for column in frame.columns}
        structural = self._structural_columns()
        return set(present & _FORMULA_OWNED_COMPAT_COLUMNS) | {
            column
            for column in present
            if column not in structural
            and column not in source_inputs
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
