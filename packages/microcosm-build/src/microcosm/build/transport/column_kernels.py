"""Column kernels for transported concepts, units, receipts and scenarios.

Structured parameters are canonical JSON, each beside a ``*_sha256`` naming
the resource it came from. The kernels check only that digest's format, so
it moves the node key and authenticates nothing. The kernels never load a
country package. ``unit_rule`` is the resolved :class:`BenefitUnitRule`
representation, rather than an unresolved country proposal. ``mapping`` is a
:class:`ConceptMapping` representation; ``rulespec_paths`` selects its
bindings and the closure's defaults.

The encode and scenario kernels derive no take-up draws: they refuse
``take_up_rates``, so the encoder refuses any selected take-up-threshold
binding it would execute, for want of a rate. ``takeup.assign@1`` owns the
keyed draws.

Scenario overrides re-run the same group encoder with the closure's knobs.
This preserves the encoder's allocation and state-column semantics, including
household rent conservation, instead of implementing another set of transforms.
Only columns declared as encoded inputs or closure defaults may be owned.

Receipt assignment reads a ``receipt_contract`` with ``seed_column``, ordered
``programs`` and ``exclusion_groups``. Each program names a ``program`` draw
key, ``output``, ``judgment_column`` and either ``rate`` or ``target`` (a
compiled person-count target in the declared ``surface`` artifact). A program
may require ``payment_column`` to be positive and may name existing boolean
``exclude_columns``. Program order is an explicit conflict priority. No law,
priority or take-up rate is inferred from names. A target sets the expected
take-up rate, target / eligible weighted mass; the realized weighted count is
recorded separately. All judgments must be determined.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

from microcosm.calibrate import hierarchy as hierarchy_module
from microcosm.calibrate import registry as registry_module
from microcosm.frame import concept_mapping as mapping_module
from microcosm.frame import concepts as concepts_module
from microcosm.frame import input_closure as closure_module
from microcosm.frame import unit_construction as units_module
from microcosm.frame.concept_mapping import ConceptMapping
from microcosm.frame.concepts import derive_take_up_draws
from microcosm.frame.input_closure import (
    InputClosure,
    UndeterminedAction,
    apply_defaults,
    resolve_judgments,
)
from microcosm.frame.unit_construction import (
    BenefitUnitRule,
    benefit_unit_attributes,
    benefit_unit_membership,
    units_per_household,
)
from microcosm.graph import (
    ROWS_ALL,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
    source_hash,
)
from microcosm.graph.population import dtype_for_token

from . import graph_inputs
from .artifact_types import TARGET_SURFACE_TYPE
from .graph_inputs import (
    canonical_document_param,
    require_params,
    sha256_param,
    string_param,
)
from .target_kernels import decode_target_surface

__all__ = [
    "CONCEPTS_ENCODE",
    "CONCEPTS_ENCODE_GROUPS",
    "TAKEUP_ASSIGN",
    "UNIT_ATTRIBUTES",
    "SCENARIO_OVERRIDE",
    "TRANSPORT_SCENARIO_OVERRIDE",
    "TRANSPORT_UNIT_ATTRIBUTES",
    "ConceptsEncodeGroupsKernel",
    "ConceptsEncodeKernel",
    "ScenarioOverrideKernel",
    "TakeupAssignKernel",
    "UnitAttributesKernel",
    "assign_receipts",
    "register_column_kernels",
]


def _whole_tables(
    context: KernelContext,
    ref: str,
    entities: tuple[str, ...],
    *,
    artifact_aliases: frozenset[str] = frozenset(),
):
    if any(
        item.rows != ROWS_ALL for item in (*context.node.inputs, *context.node.outputs)
    ):
        raise ValueError(f"{ref} requires all-row slices.")
    missing = set(entities) - set(context.tables)
    if missing:
        raise ValueError(
            f"{ref} needs {sorted(missing)} tables; declare a data-column slice "
            "on each entity."
        )
    if context.node.sources or context.node.artifact_outputs:
        raise ValueError(f"{ref} declares no sources or artifact outputs.")
    if set(context.artifacts) - artifact_aliases:
        raise ValueError(f"{ref} received undeclared artifact aliases.")
    return {entity: context.tables[entity].copy(deep=True) for entity in entities}


def _document(context: KernelContext, ref: str, name: str):
    sha256_param(context, ref, f"{name}_sha256")
    return canonical_document_param(context, ref, name)


def _rule(context: KernelContext, ref: str) -> BenefitUnitRule:
    return BenefitUnitRule.from_dict(_document(context, ref, "unit_rule"))


def _paths(context: KernelContext, ref: str) -> tuple[str, ...] | None:
    value = context.params.get("rulespec_paths")
    if value is None:
        return None
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(path, str) or not path for path in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{ref} rulespec_paths must be distinct non-empty paths.")
    return value


def _full_mapping(context: KernelContext, ref: str) -> ConceptMapping:
    """The whole declared mapping, after checking its selected paths exist."""
    mapping = ConceptMapping.from_dict(_document(context, ref, "mapping"))
    paths = _paths(context, ref)
    if paths is not None:
        unknown = set(paths) - {binding.module for binding in mapping.bindings}
        if unknown:
            raise ValueError(f"{ref} has unknown mapping paths {sorted(unknown)}.")
    return mapping


def _mapping(context: KernelContext, ref: str) -> ConceptMapping:
    """The mapping restricted to the selected paths, for person encoding."""
    mapping = _full_mapping(context, ref)
    paths = _paths(context, ref)
    if paths is None:
        return mapping
    bindings = tuple(binding for binding in mapping.bindings if binding.module in paths)
    bound = {concept for binding in bindings for concept in binding.reads}
    unmapped = {
        item.id: mapping.unmapped.get(item.id, "Outside the selected RuleSpec paths.")
        for item in concepts_module.CONCEPTS
        if item.id not in bound
    }
    return replace(mapping, bindings=bindings, unmapped=unmapped)


def _closure(context: KernelContext, ref: str) -> InputClosure | None:
    if "closure" not in context.params:
        if "closure_sha256" in context.params:
            raise ValueError(f"{ref} closure_sha256 needs a closure document.")
        return None
    closure = InputClosure.from_dict(_document(context, ref, "closure"))
    mapping = ConceptMapping.from_dict(_document(context, ref, "mapping"))
    if (closure.mapping_engine, closure.mapping_engine_version) != (
        mapping.engine,
        mapping.engine_version,
    ):
        raise ValueError(f"{ref} closure and mapping identities differ.")
    paths = _paths(context, ref)
    if paths is None:
        raise ValueError(f"{ref} needs rulespec_paths when a closure is supplied.")
    unknown = set(paths) - set(closure.modules)
    if unknown:
        raise ValueError(f"{ref} closure has no paths {sorted(unknown)}.")
    return closure


def _defaults(
    context: KernelContext,
    ref: str,
    tables: Mapping[str, pd.DataFrame],
    closure: InputClosure | None,
    engine_entities: Mapping[str, str],
) -> dict[str, pd.DataFrame]:
    out = dict(tables)
    if closure is None:
        return out
    for frame_entity, engine_entity in engine_entities.items():
        defaults: dict[str, bool | int | float] = {}
        for path in _paths(context, ref):
            for name, value in closure.defaults(path, engine_entity).items():
                if name in defaults and (
                    type(defaults[name]) is not type(value) or defaults[name] != value
                ):
                    raise ValueError(f"{ref} has conflicting defaults for {name!r}.")
                defaults[name] = value
        out[frame_entity] = apply_defaults(out[frame_entity], defaults)
    return out


def _equal_scalars(left: pd.Series, right: pd.Series) -> bool:
    # Python scalar int/float equality preserves the integer's precision;
    # numpy array comparison would promote an int64 to an inexact float64.
    for before, after in zip(left.tolist(), right.tolist(), strict=True):
        before_null, after_null = pd.isna(before), pd.isna(after)
        if before_null or after_null:
            if not (before_null and after_null):
                return False
        elif before != after:
            return False
    return True


def _columns(context: KernelContext, ref: str, tables: Mapping[str, pd.DataFrame]):
    """Project computed values by id, casting only explicit graph dtypes."""
    columns = {}
    if not context.node.outputs:
        raise ValueError(f"{ref} must own at least one computed input column.")
    for owned in context.node.outputs:
        table = tables.get(owned.entity)
        if table is None or owned.column not in table:
            raise ValueError(
                f"{ref} did not encode declared column {owned.entity}.{owned.column}."
            )
        id_column = f"{owned.entity}_id"
        if (
            owned.column == id_column
            or owned.column.startswith("person_")
            and owned.column.endswith("_id")
        ):
            raise ValueError(f"{ref} cannot return structural columns.")
        values = table[owned.column].copy(deep=True)
        # The graph's dtype for the token; its "string" pins Python storage,
        # where a bare "string" would store pyarrow strings when installed.
        dtype = None if owned.dtype is None else dtype_for_token(owned.dtype)
        if dtype is not None and values.dtype != dtype:
            # The group encoder uses numpy object for labels; the graph has
            # an explicit string dtype. Integer casts must never truncate.
            if (
                owned.dtype == "string"
                and not values.dropna().map(lambda value: isinstance(value, str)).all()
            ):
                raise ValueError(f"{ref} refuses non-text values in {owned.column}.")
            if (
                owned.dtype in {"bool", "boolean"}
                and not values.dropna()
                .map(lambda value: isinstance(value, bool | np.bool_))
                .all()
            ):
                raise ValueError(f"{ref} refuses non-boolean values in {owned.column}.")
            cast = values.astype(dtype)
            if owned.dtype != "string" and not _equal_scalars(values, cast):
                raise ValueError(f"{ref} would lose values casting {owned.column}.")
            values = cast
        values.index = pd.Index(table[id_column].to_numpy(copy=True))
        columns[owned.entity, owned.column] = values
    return columns


class _ColumnKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas"),
    )
    _modules = ()

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            *self._modules,
            dependencies=self.capabilities.dependencies,
        )


class UnitAttributesKernel(_ColumnKernel):
    """Attributes of the units CREATE already constructed; never new ids."""

    ref = "transport.unit_attributes@1"
    _modules = (units_module, mapping_module, concepts_module)

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context, self.ref, required=frozenset({"unit_rule", "unit_rule_sha256"})
        )
        rule = _rule(context, self.ref)
        tables = _whole_tables(context, self.ref, ("person", "household", rule.entity))
        person, household, units = (
            tables["person"],
            tables["household"],
            tables[rule.entity],
        )
        attributes = benefit_unit_attributes(
            person, units, person[rule.membership_column], rule
        )
        # These are aliases used by input-bridge declarations, with the same
        # adult composition as the wrapped function's is_couple attribute.
        attributes["is_partnered"] = attributes["is_couple"]
        attributes["is_single"] = ~attributes["is_couple"]
        household_attributes = household.loc[:, ["household_id"]].copy()
        household_attributes[f"n_{rule.entity}_units"] = units_per_household(
            household, units, rule
        )
        return KernelResult(
            columns=_columns(
                context,
                self.ref,
                {rule.entity: attributes, "household": household_attributes},
            ),
            receipt={"entity": rule.entity, "n_units": len(units)},
        )


_ENCODE_REQUIRED = frozenset({"mapping", "mapping_sha256"})
_ENCODE_OPTIONAL = frozenset(
    {
        "rulespec_paths",
        "shares",
        "shares_sha256",
        "closure",
        "closure_sha256",
    }
)
_TAKE_UP_PARAMS = frozenset({"take_up_rates", "take_up_rates_sha256"})


def _refuse_take_up(context: KernelContext, ref: str) -> None:
    """Keep keyed take-up draws in the receipt layer (``SeedSource.NONE``)."""
    if _TAKE_UP_PARAMS & set(context.params):
        raise ValueError(
            f"{ref} derives no take-up draws; takeup.assign@1 owns them, so "
            "take_up_rates are refused."
        )


def _optional_document(context: KernelContext, ref: str, name: str):
    return (
        canonical_document_param(context, ref, name) if name in context.params else {}
    )


def _optional_resource_document(context: KernelContext, ref: str, name: str):
    if name not in context.params:
        if f"{name}_sha256" in context.params:
            raise ValueError(f"{ref} {name}_sha256 needs a {name} document.")
        return {}
    return _document(context, ref, name)


class ConceptsEncodeKernel(_ColumnKernel):
    """Encode person and household bindings plus declared closure defaults."""

    ref = "concepts.encode@1"
    _modules = (mapping_module, concepts_module, closure_module)

    def run(self, context: KernelContext) -> KernelResult:
        _refuse_take_up(context, self.ref)
        require_params(
            context, self.ref, required=_ENCODE_REQUIRED, optional=_ENCODE_OPTIONAL
        )
        tables = _whole_tables(context, self.ref, ("person", "household"))
        mapping = _mapping(context, self.ref)
        encoded = mapping.encode(
            tables,
            shares=_optional_resource_document(context, self.ref, "shares"),
        )
        outputs = _defaults(
            context,
            self.ref,
            encoded.tables,
            _closure(context, self.ref),
            mapping.entity_correspondence,
        )
        return KernelResult(
            columns=_columns(context, self.ref, outputs),
            receipt={"engine": mapping.engine, "n_deferred": len(encoded.deferred)},
        )


_GROUP_REQUIRED = _ENCODE_REQUIRED | frozenset(
    {"unit_rule", "unit_rule_sha256", "engine_entity"}
)
_GROUP_OPTIONAL = _ENCODE_OPTIONAL | frozenset({"knobs", "scenario_sha256"})


def _group_inputs(context: KernelContext, ref: str):
    rule = _rule(context, ref)
    tables = _whole_tables(context, ref, ("person", "household", rule.entity))
    # The whole mapping: encode_groups selects modules itself and checks
    # state bindings against every concept binding's input name.
    mapping = _full_mapping(context, ref)
    closure = _closure(context, ref)
    knobs = _optional_document(context, ref, "knobs")
    if knobs and closure is None:
        raise ValueError(f"{ref} requires a closure for scenario knobs.")
    if knobs:
        sha256_param(context, ref, "scenario_sha256")
    elif "scenario_sha256" in context.params:
        sha256_param(context, ref, "scenario_sha256")
    entity = string_param(context, ref, "engine_entity")
    membership = benefit_unit_membership(
        tables["person"],
        tables[rule.entity],
        tables["person"][rule.membership_column],
        rule,
    )
    encoded = mapping.encode_groups(
        tables,
        {entity: membership},
        modules=_paths(context, ref),
        knobs=None if closure is None else closure.group_knobs(knobs),
        state_bindings=() if closure is None else closure.state_bindings(),
        shares=_optional_resource_document(context, ref, "shares"),
    )
    outputs = _defaults(context, ref, encoded.tables, closure, {rule.entity: entity})
    return mapping, encoded, outputs


class ConceptsEncodeGroupsKernel(_ColumnKernel):
    """Group bindings execute through G4's encoder and built memberships."""

    ref = "concepts.encode_groups@1"
    _modules = (mapping_module, units_module, concepts_module, closure_module)

    def run(self, context: KernelContext) -> KernelResult:
        _refuse_take_up(context, self.ref)
        require_params(
            context, self.ref, required=_GROUP_REQUIRED, optional=_GROUP_OPTIONAL
        )
        mapping, encoded, outputs = _group_inputs(context, self.ref)
        return KernelResult(
            columns=_columns(context, self.ref, outputs),
            receipt={"engine": mapping.engine, "n_deferred": len(encoded.deferred)},
        )


class ScenarioOverrideKernel(ConceptsEncodeGroupsKernel):
    """Re-encode declared group inputs under the closure's scenario knobs."""

    ref = "transport.scenario_override@1"

    def run(self, context: KernelContext) -> KernelResult:
        _refuse_take_up(context, self.ref)
        require_params(
            context,
            self.ref,
            required=_GROUP_REQUIRED
            | frozenset({"closure", "closure_sha256", "knobs", "scenario_sha256"}),
            optional=_GROUP_OPTIONAL,
        )
        if any(not owned.rewrite for owned in context.node.outputs):
            raise ValueError(f"{self.ref} owns rewrites of encoded inputs only.")
        mapping, encoded, outputs = _group_inputs(context, self.ref)
        return KernelResult(
            columns=_columns(context, self.ref, outputs),
            receipt={"engine": mapping.engine, "n_deferred": len(encoded.deferred)},
        )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Receipt {name} must be non-empty text.")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Receipt {name} must be a number.")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Receipt {name} must be finite and non-negative.")
    return value


def assign_receipts(
    person: pd.DataFrame,
    contract: Mapping[str, object],
    *,
    weights: np.ndarray | None = None,
    targets: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply explicit receipt priorities to determined eligible program draws.

    This shared pure operation contains receipt orchestration only. G4's
    judgment decoder and the content layer's stable program draws supply its
    semantics. Existing exclusions are read as booleans and never rewritten.
    """
    if set(contract) != {"seed_column", "programs", "exclusion_groups"}:
        raise ValueError(
            "Receipt contract declares seed_column, programs and exclusion_groups."
        )
    seed_column = _text(contract["seed_column"], "seed_column")
    programs = contract["programs"]
    groups = contract["exclusion_groups"]
    if not isinstance(programs, list | tuple) or not programs:
        raise ValueError("Receipt programs must be a non-empty ordered list.")
    if not isinstance(groups, list | tuple):
        raise ValueError("Receipt exclusion_groups must be a list.")
    parsed = []
    for row in programs:
        if not isinstance(row, Mapping):
            raise ValueError("A receipt program must be an object.")
        required = {"program", "output", "judgment_column"}
        optional = {
            "rate",
            "target",
            "target_measure",
            "payment_column",
            "exclude_columns",
        }
        if not required <= set(row) or set(row) - required - optional:
            raise ValueError("A receipt program has missing or unknown fields.")
        if ("rate" in row) == ("target" in row):
            raise ValueError("A receipt program declares exactly one rate or target.")
        if "target_measure" in row and "target" not in row:
            raise ValueError("A target_measure requires a target reference.")
        parsed.append({**row, **{key: _text(row[key], key) for key in required}})
    outputs = [row["output"] for row in parsed]
    keys = [row["program"] for row in parsed]
    if len(set(outputs)) != len(outputs) or len(set(keys)) != len(keys):
        raise ValueError("Receipt program keys and output names must be distinct.")
    for group in groups:
        if (
            not isinstance(group, list | tuple)
            or len(group) < 2
            or len(set(group)) != len(group)
            or not set(group) <= set(outputs)
        ):
            raise ValueError("Each receipt exclusion group names distinct outputs.")
    out = person.loc[:, ["person_id"]].copy()
    audit = {}
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if (
            weights.shape != (len(person),)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError(
                "Receipt weights must be aligned, finite and non-negative."
            )
    for row in parsed:
        output = row["output"]
        eligible = resolve_judgments(
            person[row["judgment_column"]].to_numpy(), UndeterminedAction.REFUSE
        )
        payment_column = row.get("payment_column")
        if payment_column is not None:
            payment = person[_text(payment_column, "payment_column")].to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(payment).all():
                raise ValueError("Receipt payment values must be finite.")
            eligible &= payment > 0
        excludes = row.get("exclude_columns", [])
        if not isinstance(excludes, list | tuple) or len(set(excludes)) != len(
            excludes
        ):
            raise ValueError("Receipt exclude_columns must be distinct columns.")
        for column in excludes:
            values = person[_text(column, "exclude column")]
            if str(values.dtype) not in {"bool", "boolean"} or values.isna().any():
                raise ValueError(
                    "Existing receipt exclusions must be non-null booleans."
                )
            eligible &= ~values.to_numpy(dtype=bool)
        for group in groups:
            if output in group:
                for previous in group:
                    if previous in out:
                        eligible &= ~out[previous].to_numpy()
        target = None
        mass = None if weights is None else math.fsum(weights[eligible].tolist())
        if "target" in row:
            name = _text(row["target"], "target")
            if targets is None or name not in targets or weights is None:
                raise ValueError(
                    f"Receipt target {name!r} needs a surface and weights."
                )
            target = _finite_nonnegative(targets[name], "target")
            if target > mass:
                raise ValueError(
                    f"Receipt target {name!r} exceeds eligible weighted mass."
                )
            # target <= mass, so a zero eligible mass has a zero target.
            rate = target / mass if mass else target
        else:
            rate = _finite_nonnegative(row["rate"], "rate")
            if rate > 1:
                raise ValueError("Receipt rate must lie on [0, 1].")
        draws = derive_take_up_draws(person[seed_column].to_numpy(), row["program"])
        assigned = eligible & (draws < rate)
        out[output] = assigned
        audit[output] = {
            "program": row["program"],
            "rate": rate,
            "target": target,
            "eligible_mass": mass,
            "n_receipts": int(assigned.sum()),
            "realized_mass": None
            if weights is None
            else math.fsum(weights[assigned].tolist()),
        }
    return out, audit


class TakeupAssignKernel(_ColumnKernel):
    """Receipt assignment over deterministic stable program-key draws."""

    ref = "takeup.assign@1"
    _modules = (concepts_module, closure_module)
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.KEYED,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        from . import target_kernels

        return source_hash(
            type(self),
            graph_inputs,
            concepts_module,
            closure_module,
            target_kernels,
            registry_module,
            hierarchy_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context,
            self.ref,
            required=frozenset({"receipt_contract", "receipt_contract_sha256"}),
        )
        tables = _whole_tables(
            context, self.ref, ("person",), artifact_aliases=frozenset({"surface"})
        )
        contract = _document(context, self.ref, "receipt_contract")
        targets = None
        if context.artifacts:
            if (
                set(context.artifacts) != {"surface"}
                or context.artifacts["surface"].type != TARGET_SURFACE_TYPE
            ):
                raise ValueError(f"{self.ref} reads only the typed surface artifact.")
            surface = decode_target_surface(context.artifacts["surface"].payload)
            targets = {}
            programs = contract.get("programs", ())
            for row in programs:
                if not isinstance(row, Mapping) or "target" not in row:
                    continue
                name = _text(row["target"], "target")
                expected_measure = _text(
                    row.get("target_measure", row.get("output")), "target_measure"
                )
                matches = [
                    target for target in surface.registry.specs if target.name == name
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Receipt target {name!r} must identify one surface row."
                    )
                (target,) = matches
                if (
                    target.entity != "person"
                    or target.filter is not None
                    or target.measure != expected_measure
                    or target.metadata.get("ledger_measure_unit") != "count"
                ):
                    raise ValueError(
                        f"Receipt target {name!r} must be an unfiltered person count of {expected_measure!r}."
                    )
                targets[name] = target.value
        table, audit = assign_receipts(
            tables["person"],
            contract,
            weights=None
            if "person" not in context.weights
            else context.weights["person"].values,
            targets=targets,
        )
        expected = set(table.columns) - {"person_id"}
        owned = {(item.entity, item.column) for item in context.node.outputs}
        if owned != {("person", column) for column in expected}:
            raise ValueError(
                f"{self.ref} must own exactly the contract's receipt flags."
            )
        return KernelResult(
            columns=_columns(context, self.ref, {"person": table}),
            receipt={"programs": audit},
        )


TRANSPORT_UNIT_ATTRIBUTES = UnitAttributesKernel()
CONCEPTS_ENCODE = ConceptsEncodeKernel()
CONCEPTS_ENCODE_GROUPS = ConceptsEncodeGroupsKernel()
TAKEUP_ASSIGN = TakeupAssignKernel()
TRANSPORT_SCENARIO_OVERRIDE = ScenarioOverrideKernel()
UNIT_ATTRIBUTES = TRANSPORT_UNIT_ATTRIBUTES
SCENARIO_OVERRIDE = TRANSPORT_SCENARIO_OVERRIDE


def register_column_kernels(registry: KernelRegistry) -> KernelRegistry:
    """Register one instance of each column kernel in the run's registry."""
    for kernel in (
        TRANSPORT_UNIT_ATTRIBUTES,
        CONCEPTS_ENCODE,
        CONCEPTS_ENCODE_GROUPS,
        TAKEUP_ASSIGN,
        TRANSPORT_SCENARIO_OVERRIDE,
    ):
        registry.register(kernel)
    return registry
