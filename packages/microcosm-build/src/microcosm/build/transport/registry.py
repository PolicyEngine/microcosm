"""Explicit transport registrations and checked rules-engine bindings.

Registries are local to a run. All rules bindings share one routing kernel;
adding a binding against the same RuleSpec tree and pins leaves existing
references unchanged. Engine commit and wheel SHA-256 pins, and the RuleSpec
commit for exported trees, are format-checked caller declarations. Module
bytes and tree contents are hashed; installed engine provenance is not checked.
Kernel implementation identity binds adapter classes,
never spec data, so adding a binding of an existing class leaves it unchanged.
Counterfactual and zero-cutout bridges share the same prepared engine mapping.
Gap and scenario-band kernels are registered with the other transport kernels.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from microcosm.build.graph_atomic_geography import register_atomic_geography_kernels
from microcosm.calibrate.ordered_kernels import CALIBRATE_ORDERED_ADAM
from microcosm.frame import EntitySchema
from microcosm.frame.adapters.axiom import (
    AxiomEngine,
    AxiomPeriod,
    assert_no_relations,
    axiom_engine_ref,
    rulespec_tree_digest,
)
from microcosm.frame.rules import RulesEngine
from microcosm.frame.rules_kernels import (
    SimulateCounterfactualKernel,
    SimulateRulesByRefKernel,
    SimulateSolveZeroKernel,
)
from microcosm.frame.unit_construction import BenefitUnitRule
from microcosm.graph import KernelRegistry
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import SourceCodecRegistry

from .codecs import register_transport_codecs
from .column_kernels import register_column_kernels
from .gate_bindings import TRANSPORT_GATE_REGISTRY
from .gate_kernels import register_gate_kernel
from .geography_kernels import register_geography_kernels
from .population_kernels import register_population_kernels
from .takeup_kernels import register_takeup_kernels
from .target_kernels import register_target_kernels
from .terminal_kernels import register_terminal_kernels

__all__ = [
    "PYTHON_ENGINE_REF_FORMAT",
    "TransportRegistry",
    "build_transport_registry",
    "register_transport_population_kernels",
    "transport_entity_schema",
]

#: Public format identity for an injected pure-Python rules-engine reference.
PYTHON_ENGINE_REF_FORMAT = "microcosm.transport.python-rules-engine-ref/1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


@dataclass(frozen=True)
class TransportRegistry:
    """One run's codecs, kernels, adapters and binding-to-reference lookup.

    ``engines`` is the same mapping bridge kernels must receive when added.
    ``engine_refs`` lets composition resolve a spec binding without depending
    on the adapter's internal state. Both mappings are immutable snapshots.
    """

    kernels: KernelRegistry
    codecs: SourceCodecRegistry
    engines: Mapping[str, RulesEngine]
    engine_refs: Mapping[str, str]
    schema: EntitySchema
    nesting: Mapping[str, str]


def transport_entity_schema(
    rule: BenefitUnitRule,
) -> tuple[EntitySchema, Mapping[str, str]]:
    """The persons, their households and the rule's benefit units.

    This is the schema ``transport.create@1`` gives its frame. Each unit is
    built inside one household (``build_benefit_units`` returns
    ``<unit>_household_id``), so the unit nests in the household.
    """
    if not isinstance(rule, BenefitUnitRule):
        raise TypeError("transport_entity_schema requires a BenefitUnitRule.")
    schema = EntitySchema(group_entities=("household", rule.entity))
    return schema, MappingProxyType({rule.entity: "household"})


def _mapping(value: object, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _pin(value: object, label: str, pattern: re.Pattern) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} is missing or is not a lowercase content pin.")
    return value


def _binding_rows(document: Mapping) -> tuple[Mapping, ...]:
    rows = document.get("bindings")
    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("axiom_rules_bindings.bindings must be a non-empty list.")
    result = tuple(_mapping(row, "rules binding") for row in rows)
    ids = [_text(row.get("id"), "rules binding.id") for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("axiom_rules_bindings.bindings repeats a binding id.")
    return result


def _binding_semantics(
    row: Mapping,
    root: Path,
    periods: Mapping,
    entity_names: Mapping,
    schema: EntitySchema,
) -> dict:
    label = f"rules binding {row['id']!r}"
    relative = _text(row.get("rulespec_path"), f"{label}.rulespec_path")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or str(path) != relative:
        raise ValueError(f"{label}.rulespec_path must be a canonical relative path.")
    module = (root / relative).resolve()
    if not module.is_file() or not module.is_relative_to(root):
        raise ValueError(
            f"{label}.rulespec_path is not a file under the RuleSpec tree."
        )
    digest = _pin(row.get("sha256"), f"{label}.sha256", _SHA256)
    if hashlib.sha256(module.read_bytes()).hexdigest() != digest:
        raise ValueError(f"{label}.sha256 does not match rulespec_path {relative!r}.")
    entity = _text(row.get("entity"), f"{label}.entity")
    if entity not in schema.entities:
        raise ValueError(f"{label}.entity {entity!r} is absent from the schema.")
    engine_entity = _text(row.get("engine_entity"), f"{label}.engine_entity")
    if engine_entity != entity_names[entity]:
        raise ValueError(f"{label}.engine_entity disagrees with entity_names.")
    period = _text(row.get("period"), f"{label}.period")
    if period not in periods:
        raise ValueError(f"{label}.period {period!r} has no explicit period bounds.")
    variables = row.get("variables")
    if isinstance(variables, str | bytes) or not isinstance(variables, Sequence):
        raise TypeError(f"{label}.variables must be a list of strings.")
    variables = tuple(_text(item, f"{label}.variables") for item in variables)
    if not variables or len(variables) != len(set(variables)):
        raise ValueError(f"{label}.variables must be non-empty and distinct.")
    # Metadata prose and declarations do not decide engine outputs.
    return {
        "id": row["id"],
        "rulespec_path": relative,
        "sha256": digest,
        "entity": entity,
        "engine_entity": engine_entity,
        "period": period,
        "variables": variables,
    }


def _check_engine(
    engine: RulesEngine,
    semantics: Mapping,
    root: Path,
    periods: Mapping[str, AxiomPeriod],
    entity_names: Mapping[str, str],
    schema: EntitySchema,
    nesting: Mapping[str, str],
) -> None:
    label = f"rules binding {semantics['id']!r}"
    if not isinstance(engine, RulesEngine):
        raise TypeError(f"{label} does not bind a RulesEngine.")
    if engine.entity_schema() != schema:
        raise ValueError(f"{label} engine schema does not match the transport schema.")
    if isinstance(engine, AxiomEngine):
        roots = tuple(path.resolve() for path in engine._rulespec_roots)
        if roots != (root,):
            raise ValueError(
                f"{label} engine root must equal the resolved RuleSpec tree."
            )
        if engine._module.resolve() != (root / semantics["rulespec_path"]).resolve():
            raise ValueError(f"{label} engine module does not match rulespec_path.")
        if (
            engine._periods != periods
            or engine._entity_names != entity_names
            or engine._nesting != dict(nesting)
            or engine._output_dtypes != "graph"
        ):
            raise ValueError(
                f"{label} engine must use declared periods, unit nesting and graph dtypes."
            )
    else:
        declared_root = getattr(engine, "rulespec_root", None)
        if (
            not isinstance(declared_root, str | Path)
            or Path(declared_root).resolve() != root
        ):
            raise ValueError(
                f"{label} engine root must equal the resolved RuleSpec tree."
            )
        if not callable(getattr(engine, "assert_no_relations", None)):
            raise TypeError(
                f"{label} injected engine must expose assert_no_relations(entity)."
            )
        if not callable(getattr(engine, "cache_identity", None)):
            raise TypeError(f"{label} injected engine must expose cache_identity().")


def build_transport_registry(
    rules_bindings: Mapping,
    rulespec_root: str | Path,
    *,
    unit_rule: BenefitUnitRule,
    engines_by_binding: Mapping[str, RulesEngine] | None = None,
    dependencies: tuple[str, ...] = (),
) -> TransportRegistry:
    """Build a run registry with one rules router and the available kernels.

    Real adapters use the schema and unit nesting of
    :func:`transport_entity_schema` for ``unit_rule`` (the frame CREATE
    builds), explicit periods, graph output dtypes, checked module bytes and
    :func:`axiom_engine_ref`. Engine commit and wheel SHA-256 pins, and the
    RuleSpec commit for exported trees, are format-checked caller declarations.
    The module hash and tree digest bind the actual RuleSpec bytes. References
    are pinned before any
    relation check compiles a program. An injected adapter must report the
    same schema.

    Engine-free fixtures may inject pure-Python adapters by binding id. Each
    must implement ``RulesEngine``, declare ``rulespec_root`` and expose an
    ``assert_no_relations(entity)`` check and ``cache_identity()`` mapping.
    That mapping must describe every output-affecting adapter configuration
    value; the caller is responsible for its completeness and for the adapter
    implementing the pinned fixture tree. Output-affecting configuration must
    remain fixed after preparation. Their references explicitly name a
    Python adapter and bind its declared identity and spec bytes, without
    claiming an Axiom execution. Injection must cover exactly the bindings.

    This registers the kernels implemented in this package; it does not imply
    readiness of a graph that also needs AS bridge or gap kernels.
    """
    document = _mapping(rules_bindings, "axiom_rules_bindings")
    schema, nesting = transport_entity_schema(unit_rule)
    root = Path(rulespec_root).resolve()
    if not root.is_dir():
        raise ValueError("The RuleSpec tree source must resolve to a directory.")
    # A symlink can point at bytes outside the tree the source claims to name.
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("The RuleSpec tree source must not contain symbolic links.")
    engine_pins = _mapping(document.get("engine"), "axiom_rules_bindings.engine")
    rulespec_pins = _mapping(document.get("rulespec"), "axiom_rules_bindings.rulespec")
    pins = {
        "engine_commit": _pin(
            engine_pins.get("commit"), "axiom_rules_bindings.engine.commit", _COMMIT
        ),
        "wheel_sha256": _pin(
            engine_pins.get("wheel_sha256"),
            "axiom_rules_bindings.engine.wheel_sha256",
            _SHA256,
        ),
        "rulespec_commit": _pin(
            rulespec_pins.get("commit"), "axiom_rules_bindings.rulespec.commit", _COMMIT
        ),
    }
    raw_periods = _mapping(document.get("periods"), "axiom_rules_bindings.periods")
    if not raw_periods:
        raise ValueError("axiom_rules_bindings.periods must supply explicit bounds.")
    periods = {}
    for label, bounds in raw_periods.items():
        _text(label, "period label")
        values = _mapping(bounds, f"period {label!r}")
        if set(values) != {"start", "end", "kind"}:
            raise ValueError(
                f"period {label!r} must supply exactly start, end and kind."
            )
        periods[label] = AxiomPeriod(**values)
    entity_names = dict(
        _mapping(document.get("entity_names"), "axiom_rules_bindings.entity_names")
    )
    if set(entity_names) != set(schema.entities):
        raise ValueError(
            "axiom_rules_bindings.entity_names must cover the transport schema."
        )
    for entity, name in entity_names.items():
        _text(name, f"entity_names.{entity}")
    if len(set(entity_names.values())) != len(entity_names):
        raise ValueError("axiom_rules_bindings.entity_names must be distinct.")
    rows = _binding_rows(document)
    if engines_by_binding is not None:
        injected = _mapping(engines_by_binding, "engines_by_binding")
        if set(injected) != {row["id"] for row in rows}:
            raise ValueError(
                "engines_by_binding must cover exactly the declared binding ids."
            )
    engines = {}
    refs = {}
    tree_sha256, tree_bytes = rulespec_tree_digest(root)
    for row in rows:
        semantics = _binding_semantics(row, root, periods, entity_names, schema)
        binding_periods = {semantics["period"]: periods[semantics["period"]]}
        engine = (
            AxiomEngine(
                root / semantics["rulespec_path"],
                schema=schema,
                rulespec_roots=(root,),
                periods=binding_periods,
                entity_names=entity_names,
                nesting=nesting,
                output_dtypes="graph",
            )
            if engines_by_binding is None
            else engines_by_binding[row["id"]]
        )
        _check_engine(
            engine, semantics, root, binding_periods, entity_names, schema, nesting
        )
        if isinstance(engine, AxiomEngine):
            reference = axiom_engine_ref(engine, rulespec_root=root, **pins)
            assert_no_relations(engine, semantics["entity"])
        else:
            identity = _mapping(
                engine.cache_identity(),
                f"rules binding {row['id']!r} adapter cache identity",
            )
            reference = canonical_json(
                {
                    "format": PYTHON_ENGINE_REF_FORMAT,
                    "engine": "python-rules-engine",
                    "adapter": f"{type(engine).__module__}.{type(engine).__qualname__}",
                    "adapter_identity": identity,
                    "binding": semantics,
                    "pins": pins,
                    "rulespec_tree_sha256": tree_sha256,
                    "rulespec_tree_bytes": tree_bytes,
                    "entity_names": entity_names,
                    "periods": {semantics["period"]: raw_periods[semantics["period"]]},
                }
            ).decode("utf-8")
            engine.assert_no_relations(semantics["entity"])
        for variable in semantics["variables"]:
            if engine.variable_metadata(variable).entity != semantics["entity"]:
                raise ValueError(
                    f"rules binding {row['id']!r} variable {variable!r} belongs to another entity."
                )
            if isinstance(engine, AxiomEngine):
                engine.graph_dtype(variable)
        if reference in engines:
            raise ValueError(
                f"rules binding {row['id']!r} repeats an engine reference."
            )
        refs[row["id"]] = reference
        engines[reference] = engine
    engine_mapping = MappingProxyType(dict(engines))
    kernels = KernelRegistry()
    codecs = SourceCodecRegistry()
    register_transport_codecs(codecs)
    register_transport_population_kernels(kernels)
    register_atomic_geography_kernels(kernels)
    register_target_kernels(kernels)
    kernels.register(CALIBRATE_ORDERED_ADAM)
    register_gate_kernel(kernels, TRANSPORT_GATE_REGISTRY)
    register_terminal_kernels(kernels, codecs=codecs)
    kernels.register(SimulateRulesByRefKernel(engine_mapping, dependencies))
    kernels.register(SimulateCounterfactualKernel(engine_mapping, dependencies))
    kernels.register(SimulateSolveZeroKernel(engine_mapping, dependencies))
    register_takeup_kernels(kernels)
    return TransportRegistry(
        kernels, codecs, engine_mapping, MappingProxyType(refs), schema, nesting
    )


def register_transport_population_kernels(registry: KernelRegistry) -> KernelRegistry:
    """Install the ten population kernels without import-time side effects.

    Source codecs are separately installed with ``register_transport_codecs``;
    target, solver and terminal kernels keep their own registration functions.
    Repeating registration with the same kernel objects is idempotent.
    """
    register_population_kernels(registry)
    register_column_kernels(registry)
    register_geography_kernels(registry)
    return registry
