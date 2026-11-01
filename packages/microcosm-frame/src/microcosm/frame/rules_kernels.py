"""Graph kernel routing each node to one of several bound rules engines.

``simulate.rules@1`` (:class:`~microcosm.frame.kernels.SimulateRulesKernel`)
binds one adapter per kernel instance and refuses any other ``engine_ref``,
and a :class:`~microcosm.graph.KernelRegistry` holds one kernel per ref. A run
can therefore reach only one rules engine through ``simulate.rules@1``. An
Axiom adapter wraps one RuleSpec module, so a graph that runs several modules
needs ``simulate.rules_by_ref@1``: it holds an ``engine_ref -> adapter``
mapping and runs each node exactly as ``simulate.rules@1`` does, through the
adapter the node's ``params["engine_ref"]`` names.

Node identity is unchanged by the routing: the node's ``engine_ref`` parameter
already enters its key, and the kernel's implementation hash binds code, not
the set of bound references, so adding a binding of an adapter class already
bound re-keys no existing node.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from decimal import InvalidOperation
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pandas as pd

import microcosm.frame.bundle as frame_bundle_module
import microcosm.frame.exact_money as frame_exact_money_module
import microcosm.frame.rules as frame_rules_module
import microcosm.frame.schema as frame_schema_module
from microcosm.frame.bundle import Frame
from microcosm.frame.exact_money import (
    _decimal_text,
    _exact_value,
    _half_even,
    _shortest_decimal,
)
from microcosm.frame.kernels import SimulateRulesKernel
from microcosm.frame.rules import RulesEngine
from microcosm.graph import (
    ROWS_ALL,
    KernelBase,
    KernelContext,
    KernelResult,
    Ownership,
    source_hash,
)

__all__ = [
    "SimulateRulesByRefKernel",
    "SimulateCounterfactualKernel",
    "SimulateSolveZeroKernel",
]

_PARAMS = frozenset({"engine_ref", "period", "variables"})


class SimulateRulesByRefKernel(KernelBase):
    """Materialize declared variables through the adapter a node's ref names.

    Args:
        engines: Non-empty mapping from ``engine_ref`` to the adapter it names.
            Each reference must uniquely identify its adapter and that
            adapter's configuration (for an Axiom adapter, use
            :func:`~microcosm.frame.adapters.axiom.axiom_engine_ref`).
        dependencies: Distribution names whose versions affect every bound
            engine's behavior, as for ``simulate.rules@1``.

    Each binding is a :class:`~microcosm.frame.kernels.SimulateRulesKernel`
    held by composition, so a node gets that kernel's parameter contract
    (exactly ``engine_ref``, ``period`` and ``variables``), its declared-
    outputs check, its all-rows rule, and its receipt, unchanged.

    The implementation hash binds this module, ``microcosm.frame.kernels``,
    the frame modules ``simulate.rules@1`` binds, and the source of each
    distinct adapter *class*. It does not bind the references: a second
    binding of a class already bound leaves every node key where it was. A
    binding of a new adapter class adds that class's source to the hash and so
    re-keys every node of this kernel, as replacing an adapter class does for
    ``simulate.rules@1``.

    Raises:
        TypeError: If ``engines`` is not a mapping, or a binding is not a
            :class:`~microcosm.frame.rules.RulesEngine`.
        ValueError: If ``engines`` is empty, or a reference is empty.
    """

    ref = "simulate.rules_by_ref@1"

    def __init__(
        self,
        engines: Mapping[str, RulesEngine],
        dependencies: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(engines, Mapping):
            raise TypeError(
                "engines must be a mapping from engine_ref to a RulesEngine."
            )
        if not engines:
            raise ValueError("engines must bind at least one engine_ref.")
        delegates = {
            engine_ref: SimulateRulesKernel(engine_ref, engine, dependencies)
            for engine_ref, engine in engines.items()
        }
        self._delegates = MappingProxyType(dict(sorted(delegates.items())))
        self._adapter_classes = tuple(
            sorted(
                {type(engine) for engine in engines.values()},
                key=lambda cls: (cls.__module__, cls.__qualname__),
            )
        )
        # Every delegate declares the same contract; take it from one so the
        # two kernels cannot drift apart.
        self.capabilities = next(iter(self._delegates.values())).capabilities

    def engine_refs(self) -> tuple[str, ...]:
        """The bound references, sorted."""

        return tuple(self._delegates)

    def implementation_hash(self) -> str:
        """Hash the routing and delegate kernels plus every bound adapter class."""

        return source_hash(
            type(self),
            SimulateRulesKernel,
            *self._adapter_classes,
            frame_bundle_module,
            frame_rules_module,
            frame_schema_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        """Run the node through the delegate its ``engine_ref`` names.

        The parameter set and the reference's type are checked before
        routing, in the order and with the exception types
        ``simulate.rules@1`` uses, so a malformed node fails the same way
        under either kernel.
        """

        actual = set(context.params)
        if actual != _PARAMS:
            raise ValueError(
                "simulate.rules_by_ref parameters must be exactly "
                f"{sorted(_PARAMS)!r}; got {sorted(actual)!r}."
            )
        engine_ref = context.params["engine_ref"]
        if not isinstance(engine_ref, str) or not engine_ref:
            raise TypeError(
                "simulate.rules_by_ref engine_ref must be a non-empty string."
            )
        delegate = self._delegates.get(engine_ref)
        if delegate is None:
            bound = ", ".join(_describe(ref) for ref in self._delegates)
            raise ValueError(
                "simulate.rules_by_ref has no engine bound to engine_ref "
                f"{_describe(engine_ref)}; bound: {bound}."
            )
        return delegate.run(context)


def _describe(engine_ref: str) -> str:
    """Name a reference briefly: its SHA-256 prefix, and its module if any.

    References are often long canonical JSON whose leading keys coincide
    (every Axiom reference starts with the same arithmetic and engine pins), so
    a truncated prefix would not tell two apart.
    """

    digest = hashlib.sha256(engine_ref.encode("utf-8")).hexdigest()[:16]
    try:
        document = json.loads(engine_ref)
    except ValueError:
        document = None
    module = document.get("module") if isinstance(document, dict) else None
    if isinstance(module, str):
        return f"sha256:{digest} (module {module!r})"
    preview = engine_ref if len(engine_ref) <= 60 else engine_ref[:60] + "..."
    return f"sha256:{digest} ({preview!r})"


class SimulateCounterfactualKernel(SimulateRulesByRefKernel):
    """Combine declared engine outputs after declared input overrides.

    The primary component uses ``engine_ref``, ``period`` and ``variables``.
    ``input_overrides`` is a JSON list of records with ``entity``, ``column``
    and either ``value`` or ``source_entity``/``source_column``. A source group
    column broadcasts to persons through structural membership. An optional
    ``mask`` record (``entity``, ``column``, ``equals``) limits an override.
    Override columns must be inputs of that component's engine.

    ``components`` is an optional JSON list of additional components, each
    declaring those four primary parameters (variables and overrides as JSON
    arrays). Components evaluate independently on copies of the original
    frame, allowing modules with distinct input worlds to share a bridge.

    ``output_coefficients`` is a JSON list of terms. Each declares an engine
    ``variable``, numeric ``coefficient``, and target ``entity``/``column``.
    ``engine_ref`` defaults to the primary reference. A person output mapped
    to a group must declare ``aggregation`` as ``sum``, ``max`` or ``first``;
    group outputs may broadcast to persons. ``first`` selects the member with
    the lowest structural person id. Terms may declare the same mask shape,
    evaluated on the original frame, to select receipt or eligibility flags.
    Terms targeting the same column add in their declared order.

    Every owned output is a produced float64 column on all rows. Parameters
    contain all policy coefficients and overrides; ``resource_sha256`` may
    bind the declaring spec resource. Neither engine bindings nor spec
    fingerprints enter the implementation hash.

    NumPy and pandas versions enter the implementation hash because bridge
    arithmetic and person/group reductions use their numerical operations.
    Caller-supplied dependencies also bind the engines' behavior.
    """

    ref = "simulate.counterfactual@1"

    def __init__(
        self,
        engines: Mapping[str, RulesEngine],
        dependencies: tuple[str, ...] = (),
    ) -> None:
        super().__init__(engines, dependencies)
        self.capabilities = replace(
            self.capabilities,
            dependencies=tuple(
                dict.fromkeys((*self.capabilities.dependencies, "numpy", "pandas"))
            ),
        )

    def run(self, context: KernelContext) -> KernelResult:
        components, terms, frame = self._prepare(context)
        values = self._evaluate(frame, components, terms)
        return self._result(context, frame, values)

    def _prepare(self, context: KernelContext, *, solver: bool = False):
        required = {
            "engine_ref",
            "period",
            "variables",
            "input_overrides",
            "output_coefficients",
        }
        if solver:
            required |= {"solve_input", "bracket", "tolerance", "iterations"}
        allowed = required | {"components", "resource_sha256"}
        if solver:
            quantization = {
                "quantum",
                "tie_rule",
                "quantized_residual_bound",
                "extra_evaluation_budget",
                "certificate_margin",
            }
            allowed |= quantization
            if set(context.params) & quantization and not quantization <= set(
                context.params
            ):
                raise ValueError(
                    "Quantization requires quantum, tie_rule, quantized_residual_bound, extra_evaluation_budget and certificate_margin together."
                )
        if not required <= set(context.params) or set(context.params) - allowed:
            raise ValueError(
                f"{self.ref} parameters require {sorted(required)!r} and allow "
                f"only {sorted(allowed)!r}."
            )
        if "resource_sha256" in context.params:
            digest = context.params["resource_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError("resource_sha256 must be a lowercase SHA-256.")
        primary = {
            key: context.params[key] for key in ("engine_ref", "period", "variables")
        }
        primary["input_overrides"] = _json_list(
            context.params["input_overrides"], "input_overrides"
        )
        additional = _json_list(context.params.get("components", "[]"), "components")
        components = [primary, *additional]
        references = set()
        schema = None
        for component in components:
            if set(component) != {
                "engine_ref",
                "period",
                "variables",
                "input_overrides",
            }:
                raise ValueError(
                    "Each component requires engine_ref, period, variables and input_overrides."
                )
            ref = component["engine_ref"]
            if not isinstance(ref, str) or ref not in self._delegates:
                raise ValueError(
                    f"{self.ref} has no engine bound to engine_ref {ref!r}."
                )
            if ref in references:
                raise ValueError("Components must not repeat engine_ref.")
            references.add(ref)
            period = component["period"]
            if (
                isinstance(period, bool)
                or not isinstance(period, int | str)
                or period == ""
            ):
                raise TypeError("Component period must be an int or non-empty string.")
            variables = component["variables"]
            if (
                not isinstance(variables, list | tuple)
                or not variables
                or any(not isinstance(item, str) or not item for item in variables)
                or len(set(variables)) != len(variables)
            ):
                raise ValueError(
                    "Component variables must be distinct non-empty names."
                )
            if not isinstance(component["input_overrides"], list):
                raise TypeError("Component input_overrides must be a list.")
            engine_schema = self._delegates[ref]._engine.entity_schema()
            if schema is not None and schema != engine_schema:
                raise ValueError("Bridge components must share an entity schema.")
            schema = engine_schema
        tables = SimulateRulesKernel._tables(context, schema)
        frame = Frame(
            tables=tables,
            schema=schema,
            weights=dict(context.weights),
            strata=context.strata,
        )
        if solver and "quantum" in context.params:
            person = schema.person_entity
            for group in schema.group_entities:
                membership = schema.membership_column(group)
                positions = pd.Index(
                    frame.table(group)[schema.id_column(group)]
                ).get_indexer(frame.table(person)[membership])
                SimulateSolveZeroKernel._refuse_rows(
                    frame,
                    (person, membership),
                    positions < 0,
                    "membership identifiers must match group ids",
                )
        terms = _json_list(context.params["output_coefficients"], "output_coefficients")
        if not terms:
            raise ValueError("output_coefficients must not be empty.")
        outputs = set()
        for term in terms:
            if solver and "trial" in term:
                _fields(
                    term,
                    {"trial", "coefficient", "entity", "column"},
                    set(),
                    "trial coefficient",
                )
                if term["trial"] is not True:
                    raise ValueError("A trial coefficient requires trial: true.")
            elif solver and {"source_entity", "source_column"} & set(term):
                _fields(
                    term,
                    {
                        "source_entity",
                        "source_column",
                        "coefficient",
                        "entity",
                        "column",
                    },
                    set(),
                    "source coefficient",
                )
                source_entity, source_column = (
                    term["source_entity"],
                    term["source_column"],
                )
                frame.schema.entity_id_column(source_entity)
                if not isinstance(
                    source_column, str
                ) or source_column not in frame.table(source_entity):
                    raise ValueError(
                        "Source coefficient must name an existing source column."
                    )
                if source_column in _structural_columns(frame.schema, source_entity):
                    raise ValueError(
                        "Source coefficient cannot read a structural column."
                    )
                _finite_source(
                    frame,
                    source_entity,
                    source_column,
                    exact="quantum" in context.params,
                )
            else:
                _fields(
                    term,
                    {"variable", "coefficient", "entity", "column"},
                    {"engine_ref", "aggregation", "mask"},
                    "output coefficient",
                )
                ref = term.get("engine_ref", primary["engine_ref"])
                component = next(
                    (item for item in components if item["engine_ref"] == ref), None
                )
                if component is None or term["variable"] not in component["variables"]:
                    raise ValueError(
                        "Output coefficient must name a requested component variable."
                    )
            _number(term["coefficient"], "coefficient")
            frame.schema.entity_id_column(term["entity"])
            if not isinstance(term["column"], str) or not term["column"]:
                raise ValueError("Output column must be a non-empty string.")
            outputs.add((term["entity"], term["column"]))
        declared = {(owned.entity, owned.column) for owned in context.node.outputs}
        if declared != outputs:
            raise ValueError("Declared outputs do not match output_coefficients.")
        if any(
            owned.rows != ROWS_ALL
            or owned.ownership is not Ownership.PRODUCED
            or owned.dtype != "float64"
            for owned in context.node.outputs
        ):
            raise ValueError(
                "Bridge outputs must be produced float64 columns on all rows."
            )
        return components, terms, frame

    def _evaluate(
        self,
        frame,
        components,
        terms,
        *,
        trial=None,
        solve_input=None,
        exact_affine=False,
    ):
        materialized = {}
        for component in components:
            ref = component["engine_ref"]
            engine = self._delegates[ref]._engine
            tables = {entity: frame.table(entity).copy() for entity in frame.entities}
            inputs = set(engine.variables())
            overridden = set()
            for override in component["input_overrides"]:
                _fields(
                    override,
                    {"entity", "column"},
                    {"value", "source_entity", "source_column", "mask"},
                    "input override",
                )
                entity, column = override["entity"], override["column"]
                frame.schema.entity_id_column(entity)
                if column not in inputs or (entity, column) in overridden:
                    raise ValueError(
                        "Overrides must name distinct engine input columns."
                    )
                overridden.add((entity, column))
                if column in _structural_columns(frame.schema, entity):
                    raise ValueError(
                        "Input overrides cannot change structural columns."
                    )
                if "value" in override:
                    if "source_entity" in override or "source_column" in override:
                        raise ValueError(
                            "An override must declare either value or source column."
                        )
                    value = override["value"]
                    if not isinstance(value, bool | int | float | str) or (
                        isinstance(value, float) and not math.isfinite(value)
                    ):
                        raise ValueError("Override values must be finite scalar data.")
                    values = np.full(len(tables[entity]), value)
                else:
                    if not {"source_entity", "source_column"} <= set(override):
                        raise ValueError(
                            "A column override needs source_entity and source_column."
                        )
                    source_entity = override["source_entity"]
                    source = frame.table(source_entity)[
                        override["source_column"]
                    ].to_numpy()
                    values = _map_values(
                        frame, source, source_entity, entity, aggregation=None
                    )
                if "mask" in override:
                    if column not in tables[entity]:
                        raise ValueError(
                            "A masked override requires the existing input column."
                        )
                    mask = _mask(
                        frame, override["mask"], entity, refuse_missing=exact_affine
                    )
                    values = np.where(mask, values, tables[entity][column].to_numpy())
                tables[entity][column] = values
            if trial is not None and solve_input["engine_ref"] == ref:
                entity, column = solve_input["entity"], solve_input["column"]
                if column not in inputs:
                    raise ValueError("solve_input must name an engine input column.")
                tables[entity][column] = _map_values(
                    frame, trial[1], trial[0], entity, aggregation=None
                )
            counterfactual = Frame(
                tables,
                frame.schema,
                {
                    entity: frame.weights_for(entity)
                    for entity in frame.weighted_entities
                },
                strata=frame.strata,
            )
            variables = tuple(component["variables"])
            raw = engine.materialize(counterfactual, variables, component["period"])
            for variable in variables:
                metadata = engine.variable_metadata(variable)
                if variable not in raw:
                    raise ValueError(
                        f"Engine did not return requested variable {variable!r}."
                    )
                values = np.asarray(
                    raw[variable], dtype=object if exact_affine else np.float64
                )
                if values.shape != (len(frame.table(metadata.entity)),):
                    raise ValueError(f"Engine output {variable!r} has the wrong shape.")
                if exact_affine:
                    invalid = np.zeros(len(values), dtype=bool)
                    exact_values = []
                    for row, value in enumerate(values):
                        try:
                            exact_values.append(_exact_value(value))
                        except (TypeError, ValueError, OverflowError):
                            invalid[row] = True
                            exact_values.append(Fraction(0))
                    for term in terms:
                        if (
                            term.get("variable") == variable
                            and term.get("engine_ref", components[0]["engine_ref"])
                            == ref
                        ):
                            target = term["entity"]
                            mask = _map_values(
                                frame,
                                invalid,
                                metadata.entity,
                                target,
                                aggregation=(
                                    "max"
                                    if metadata.entity == frame.schema.person_entity
                                    and target in frame.schema.group_entities
                                    else None
                                ),
                            )
                            self._refuse_rows(
                                frame,
                                (target, term["column"]),
                                mask,
                                f"engine output {variable!r} must contain finite exact numbers",
                            )
                    self._refuse_rows(
                        frame,
                        (metadata.entity, variable),
                        invalid,
                        f"engine output {variable!r} must contain finite exact numbers",
                    )
                    values = np.asarray(exact_values, dtype=object)
                elif not np.all(np.isfinite(values)):
                    raise ValueError(f"Engine output {variable!r} must be finite.")
                materialized[(ref, variable)] = metadata.entity, values
        output = {}
        exact_output = {}
        for term in terms:
            if "trial" in term:
                entity, values = trial
            elif "source_entity" in term:
                entity = term["source_entity"]
                values = _finite_source(
                    frame, entity, term["source_column"], exact=exact_affine
                )
            else:
                ref = term.get("engine_ref", components[0]["engine_ref"])
                entity, values = materialized[(ref, term["variable"])]
            if "mask" in term:
                values = np.where(
                    _mask(frame, term["mask"], entity, refuse_missing=exact_affine),
                    values,
                    0,
                )
            mapped = _map_values(
                frame,
                values,
                entity,
                term["entity"],
                aggregation=term.get("aggregation"),
            )
            key = term["entity"], term["column"]
            if exact_affine:
                coefficient = _exact_value(term["coefficient"])
                totals = exact_output.setdefault(
                    key, [Fraction(0) for _ in range(len(mapped))]
                )
                for row, value in enumerate(mapped):
                    totals[row] += _exact_value(value) * coefficient
                continue
            contribution = mapped * _number(term["coefficient"], "coefficient")
            output[key] = (
                output.get(key, np.zeros(len(mapped), dtype=np.float64)) + contribution
            )
            if not np.all(np.isfinite(output[key])):
                raise ValueError("Bridge coefficient sum must be finite.")
        for key, totals in exact_output.items():
            converted = np.zeros(len(totals), dtype=np.float64)
            for row, value in enumerate(totals):
                try:
                    converted[row] = float(value)
                except OverflowError:
                    converted[row] = np.inf
            self._refuse_rows(
                frame,
                key,
                ~np.isfinite(converted),
                "bridge coefficient sum must be finite",
            )
            self._refuse_rows(
                frame,
                key,
                np.asarray(
                    [
                        total != 0 and value == 0
                        for total, value in zip(totals, converted, strict=True)
                    ]
                ),
                "bridge exact coefficient sum underflows float64",
            )
            output[key] = np.asarray(totals, dtype=object)
        return output

    def _result(self, context, frame, values, **receipt):
        columns = {}
        for (entity, column), data in values.items():
            id_column = frame.schema.entity_id_column(entity)
            columns[(entity, column)] = pd.Series(
                data,
                index=pd.Index(frame.table(entity)[id_column], name=id_column),
                name=column,
                copy=False,
            )
        return KernelResult(
            columns=columns,
            receipt={
                "engine_ref": context.params["engine_ref"],
                "period": context.params["period"],
                "variables": context.params["variables"],
                "components": context.params.get("components", "[]"),
                **receipt,
            },
        )


class SimulateSolveZeroKernel(SimulateCounterfactualKernel):
    """Find the first nonpositive engine residual by fixed vector bisection.

    The counterfactual contract describes exactly one residual/output column.
    ``solve_input`` is a JSON record naming ``entity`` and input ``column``;
    its optional ``engine_ref`` defaults to the primary component. Trial values
    on the output entity broadcast to that input entity. ``bracket`` is a
    finite increasing pair, ``tolerance`` bounds the final income interval
    and positive residual at the returned answer, and ``iterations`` is a
    fixed positive integer sufficient for the interval bound. Each output
    row's residual must be independent of other rows' trial inputs and must
    be non-increasing in its own trial input on the bracket (the caller's rules
    contract). Coupled rows are unsupported. A positive upper endpoint is
    refused, and an already nonpositive lower endpoint returns that endpoint.
    All rows perform every iteration, without value-dependent early stopping.
    Returned answers are
    evaluated once more; a positive residual beyond ``tolerance`` is refused.
    Successful runs perform exactly ``iterations + 3`` residual evaluations,
    each evaluating every component once. Outputs are the upper endpoints,
    so a benefit that floors at zero yields its cutout. An exact-zero residual
    does not replace a nonzero answer with zero.

    Only this kernel also accepts affine source-column terms (source_entity,
    source_column, coefficient, entity, column) and trial terms (trial: true,
    coefficient, entity, column). They never request engine output metadata.

    Opt-in quantization requires decimal-text ``quantum`` and
    ``quantized_residual_bound``, ``tie_rule: half_even`` and an odd
    ``extra_evaluation_budget`` B >= 3, and decimal-text ``certificate_margin``
    m >= 0. The caller contracts that, at every evaluated point, the computed
    residual differs from the exact residual of the declared terms by at most
    m. Every row executes (B - 1) / 2 pairs
    of half-quantum boundary evaluations, including already-certified rows,
    followed by one evaluation at the quantized answer. Thus a successful
    quantized run performs exactly ``iterations + 3 + B`` residual evaluations
    and ``components * (iterations + 3 + B)`` engine calls, independent of row
    values. Residuals above m and below -m certify boundary signs. A needed
    boundary with absolute residual at most m refuses and names its rows and
    largest absolute residual. Only m = 0 permits an exact-zero boundary to
    select the even adjacent integer. Uncertified rows refuse.
    Quantization requires an isolated zero and a non-increasing residual on
    the bracket and evaluated boundaries. These remain caller contracts: the
    sampled signs do not prove global monotonicity or uniqueness. A plateau
    with both half-boundary residuals zero cannot certify a rounding bin.

    Integer quantum indices are authoritative. Outputs retain the bridge's
    produced float64 contract, but must round-trip through their shortest
    decimal to the exact index times quantum. A downstream money kernel can
    therefore recover the integer units exactly, without a second rounding.
    Half-quantum boundary coordinates must pass the same exact decimal
    round-trip check before their residuals can certify an answer.
    In quantized runs, coefficient products and their residual sum use exact
    Fractions: integers enter as themselves, floats enter through their
    shortest decimals, and exact rational values retain their values. The
    residual bound requires abs(F(k * quantum)) + m <= bound, compared as
    Fractions. Quantized masks refuse missing or incomparable scalar values
    with their source-row ids. Quantized solves refuse memberships that miss
    group ids in pandas lookup, naming the person rows. Unquantized
    declarations that use only the earlier term forms keep their outputs,
    receipts and refusals, except the solve-zero permitted-fields diagnostic,
    which lists the five quantization names; counterfactual keeps its list.
    """

    ref = "simulate.solve_zero@1"

    def implementation_hash(self) -> str:
        """Bind the exact helpers as imports are not followed by source_hash."""
        return source_hash(
            type(self),
            SimulateRulesKernel,
            *self._adapter_classes,
            frame_bundle_module,
            frame_rules_module,
            frame_schema_module,
            frame_exact_money_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        components, terms, frame = self._prepare(context, solver=True)
        output_keys = {(term["entity"], term["column"]) for term in terms}
        if len(output_keys) != 1:
            raise ValueError("solve_zero requires exactly one output column.")
        key = next(iter(output_keys))
        solve_input = _json_record(context.params["solve_input"], "solve_input")
        _fields(solve_input, {"entity", "column"}, {"engine_ref"}, "solve_input")
        solve_input = {"engine_ref": components[0]["engine_ref"], **solve_input}
        if solve_input["engine_ref"] not in {item["engine_ref"] for item in components}:
            raise ValueError("solve_input must name a declared component.")
        frame.schema.entity_id_column(solve_input["entity"])
        if solve_input["column"] in _structural_columns(
            frame.schema, solve_input["entity"]
        ):
            raise ValueError("solve_input cannot change structural columns.")
        bracket = context.params["bracket"]
        if not isinstance(bracket, tuple) or len(bracket) != 2:
            raise ValueError("bracket must be an increasing finite pair.")
        lower, upper = (_number(value, "bracket endpoint") for value in bracket)
        tolerance = _number(context.params["tolerance"], "tolerance")
        iterations = context.params["iterations"]
        if (
            lower >= upper
            or tolerance <= 0
            or not math.isfinite(upper - lower)
            or isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations <= 0
        ):
            raise ValueError(
                "solve_zero requires an increasing bracket, positive tolerance and iterations."
            )
        if math.ldexp(upper - lower, -iterations) > tolerance:
            raise ValueError(
                "iterations do not achieve the declared bracket tolerance."
            )
        lo = np.full(len(frame.table(key[0])), lower, dtype=np.float64)
        hi = np.full(len(lo), upper, dtype=np.float64)
        margin = (
            _decimal_text(
                context.params["certificate_margin"],
                "certificate_margin",
                allow_zero=True,
            )
            if "quantum" in context.params
            else None
        )

        def residual(trial):
            return self._evaluate(
                frame,
                components,
                terms,
                trial=(key[0], trial),
                solve_input=solve_input,
                exact_affine="quantum" in context.params,
            )[key]

        lower_values = residual(lo)
        upper_values = residual(hi)
        if margin is not None:
            self._refuse_rows(
                frame,
                key,
                upper_values > margin,
                "bracket upper endpoint has a positive residual",
            )
        elif np.any(upper_values > 0):
            raise ValueError(
                "solve_zero bracket upper endpoint has a positive residual."
            )
        already_zero = lower_values <= 0
        for _ in range(iterations):
            mid = lo + (hi - lo) / 2
            positive = residual(mid) > 0
            lo = np.where(positive, mid, lo)
            hi = np.where(positive, hi, mid)
        if margin is not None:
            self._refuse_rows(
                frame,
                key,
                (hi - lo) > tolerance,
                "float64 bisection cannot meet the declared tolerance",
            )
        elif np.any((hi - lo) > tolerance):
            raise ValueError("Float64 bisection cannot meet the declared tolerance.")
        answer = np.where(already_zero, lower, hi)
        answer_values = residual(answer)
        if margin is not None:
            self._refuse_rows(
                frame,
                key,
                answer_values > _exact_value(tolerance) + margin,
                "final answer has a positive residual beyond the declared tolerance; output rows must be independent",
            )
        elif np.any(answer_values > tolerance):
            raise ValueError(
                "solve_zero final answer has a positive residual beyond the "
                "declared tolerance; output rows must be independent."
            )
        if "quantum" in context.params:
            answer, quantized_receipt = self._quantize(
                context, frame, key, lo, hi, residual
            )
            return self._result(
                context,
                frame,
                {key: answer},
                bracket=bracket,
                tolerance=tolerance,
                iterations=iterations,
                evaluations=iterations + 3 + context.params["extra_evaluation_budget"],
                **quantized_receipt,
            )
        return self._result(
            context,
            frame,
            {key: answer},
            bracket=bracket,
            tolerance=tolerance,
            iterations=iterations,
            evaluations=iterations + 3,
        )

    def _quantize(self, context, frame, key, lo, hi, residual):
        quantum = _decimal_text(context.params["quantum"], "quantum")
        bound = _decimal_text(
            context.params["quantized_residual_bound"],
            "quantized_residual_bound",
            allow_zero=True,
        )
        margin = _decimal_text(
            context.params["certificate_margin"], "certificate_margin", allow_zero=True
        )
        if context.params["tie_rule"] != "half_even":
            raise ValueError("Quantization tie_rule must be half_even.")
        budget = context.params["extra_evaluation_budget"]
        if (
            isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget < 3
            or budget % 2 != 1
        ):
            raise ValueError(
                "extra_evaluation_budget must be an odd integer at least 3."
            )
        count = len(lo)
        indices = [0] * count
        certified = np.zeros(count, dtype=bool)
        uncertain_boundaries = [{} for _ in range(count)]
        for _ in range((budget - 1) // 2):
            candidates = [
                indices[row]
                if certified[row]
                else _half_even(
                    (_shortest_decimal(lo[row]) + _shortest_decimal(hi[row]))
                    / (2 * quantum)
                )
                for row in range(count)
            ]
            left_exact = [(Fraction(k) - Fraction(1, 2)) * quantum for k in candidates]
            right_exact = [(Fraction(k) + Fraction(1, 2)) * quantum for k in candidates]
            left, right = np.zeros(count), np.zeros(count)
            overflow = np.zeros(count, dtype=bool)
            for row, (a, b) in enumerate(zip(left_exact, right_exact, strict=True)):
                try:
                    left[row], right[row] = float(a), float(b)
                except OverflowError:
                    overflow[row] = True
            self._refuse_rows(
                frame, key, overflow, "quantization boundaries exceed float64"
            )
            self._refuse_rows(
                frame,
                key,
                ~np.isfinite(left) | ~np.isfinite(right) | (left >= right),
                "quantization boundaries must be distinct finite float64 values",
            )
            self._refuse_rows(
                frame,
                key,
                np.asarray(
                    [
                        _shortest_decimal(a) != a_exact
                        or _shortest_decimal(b) != b_exact
                        for a, b, a_exact, b_exact in zip(
                            left, right, left_exact, right_exact, strict=True
                        )
                    ]
                ),
                "quantization boundaries must round-trip exactly through float64 decimal text",
            )
            left_values, right_values = residual(left), residual(right)
            ambiguous = np.zeros(count, dtype=bool)
            ambiguous_values = []
            for row in range(count):
                for boundary, value in (
                    (left_exact[row], left_values[row]),
                    (right_exact[row], right_values[row]),
                ):
                    if abs(value) <= margin:
                        uncertain_boundaries[row][boundary] = max(
                            uncertain_boundaries[row].get(boundary, Fraction(0)),
                            abs(value),
                        )
                        if margin > 0 and not certified[row]:
                            ambiguous[row] = True
                            ambiguous_values.append(abs(value))
                if len(uncertain_boundaries[row]) > 1:
                    ambiguous[row] = True
                    ambiguous_values.extend(uncertain_boundaries[row].values())
            if np.any(ambiguous):
                largest = max(ambiguous_values, default=Fraction(0))
                self._refuse_rows(
                    frame,
                    key,
                    ambiguous,
                    f"ambiguous quantization certificate within margin; largest |F(b)| {largest}",
                )
            for row, k in enumerate(candidates):
                a, b = left_values[row], right_values[row]
                if a + 2 * margin < b:
                    self._refuse_rows(
                        frame,
                        key,
                        np.arange(count) == row,
                        "non-increasing quantization residual",
                    )
                if certified[row]:
                    continue
                if a > margin and b < -margin:
                    indices[row], certified[row] = k, True
                elif margin == 0 and a == 0 and b < 0:
                    indices[row], certified[row] = (k if k % 2 == 0 else k - 1), True
                elif margin == 0 and a > 0 and b == 0:
                    indices[row], certified[row] = (k if k % 2 == 0 else k + 1), True
                elif a > margin and b > margin:
                    lo[row] = max(lo[row], right[row])
                elif a < -margin and b < -margin:
                    hi[row] = min(hi[row], left[row])
        self._refuse_rows(
            frame, key, ~certified, "quantization certification budget exhausted"
        )
        minimum, maximum = np.iinfo(np.int64).min, np.iinfo(np.int64).max
        self._refuse_rows(
            frame,
            key,
            np.asarray([k < minimum or k > maximum for k in indices]),
            "quantization integer units exceed int64",
        )
        exact = [k * quantum for k in indices]
        answer = np.zeros(count, dtype=np.float64)
        unrepresentable = np.zeros(count, dtype=bool)
        for row, target in enumerate(exact):
            try:
                answer[row] = float(target)
                unrepresentable[row] = (
                    not math.isfinite(answer[row])
                    or _shortest_decimal(answer[row]) != target
                )
            except OverflowError:
                unrepresentable[row] = True
        self._refuse_rows(
            frame,
            key,
            unrepresentable,
            "quantized values must round-trip exactly through float64 decimal text",
        )
        final_residual = residual(answer)
        excessive = np.asarray(
            [abs(value) + margin > bound for value in final_residual]
        )
        self._refuse_rows(
            frame, key, excessive, "quantized residual exceeds declared bound"
        )
        return answer, {
            "quantum": context.params["quantum"],
            "tie_rule": context.params["tie_rule"],
            "quantized_residual_bound": context.params["quantized_residual_bound"],
            "extra_evaluation_budget": budget,
            "extra_evaluations": budget,
            "certificate_margin": context.params["certificate_margin"],
            "certificate_contract": (
                "At every evaluated point, the computed residual differs from "
                "the exact residual of the declared terms by at most certificate_margin."
            ),
            "evaluation_count_formula": "iterations + 3 + extra_evaluation_budget",
            "max_quantized_residual": float(np.max(np.abs(final_residual), initial=0)),
        }

    @staticmethod
    def _refuse_rows(frame, key, mask, message):
        if np.any(mask):
            ids = (
                frame.table(key[0])[frame.schema.entity_id_column(key[0])]
                .to_numpy()[mask]
                .tolist()
            )
            raise ValueError(f"solve_zero {message}; rows {ids!r}.")


def _json_list(value, name):
    payload = _json_value(value, name)
    if not isinstance(payload, list) or any(
        not isinstance(item, dict) for item in payload
    ):
        raise TypeError(f"{name} must be a JSON list of records.")
    return payload


def _json_record(value, name):
    payload = _json_value(value, name)
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be a JSON record.")
    return payload


def _json_value(value, name):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a JSON string.")
    try:
        return json.loads(value, parse_constant=lambda item: _invalid_constant(item))
    except ValueError as error:
        raise ValueError(f"{name} is not valid finite JSON.") from error


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant {value!r}.")


def _fields(record, required, optional, name):
    if (
        not isinstance(record, dict)
        or not required <= set(record)
        or set(record) - required - optional
    ):
        raise ValueError(
            f"Invalid {name} fields; require {sorted(required)!r}, allow {sorted(optional)!r}."
        )


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _finite_source(frame, entity, column, *, exact=False):
    source = frame.table(entity)[column]
    values = np.zeros(len(source), dtype=object if exact else np.float64)
    invalid = np.zeros(len(source), dtype=bool)
    for row, value in enumerate(source):
        try:
            values[row] = _exact_value(value) if exact else float(value)
            if not exact and not math.isfinite(values[row]):
                invalid[row] = True
        except (TypeError, ValueError, OverflowError):
            invalid[row] = True
    SimulateSolveZeroKernel._refuse_rows(
        frame,
        (entity, column),
        invalid,
        "source coefficient columns must contain finite numbers",
    )
    return values


def _structural_columns(schema, entity):
    columns = {schema.entity_id_column(entity)}
    if entity == schema.person_entity:
        columns.update(
            schema.membership_column(group) for group in schema.group_entities
        )
    return columns


def _mask(frame, record, entity, *, refuse_missing=False):
    _fields(record, {"entity", "column", "equals"}, set(), "mask")
    match = record["equals"]
    if not isinstance(match, bool | int | float | str) or (
        isinstance(match, float) and not math.isfinite(match)
    ):
        raise ValueError("Mask equals must be finite scalar data.")
    source_entity = record["entity"]
    source = frame.table(source_entity)[record["column"]]
    if refuse_missing:
        values = np.zeros(len(source), dtype=bool)
        missing = np.zeros(len(source), dtype=bool)
        invalid = np.zeros(len(source), dtype=bool)
        for row, value in enumerate(source):
            try:
                missing_value = pd.isna(value)
                if not isinstance(missing_value, bool | np.bool_):
                    invalid[row] = True
                elif missing_value:
                    missing[row] = True
                else:
                    comparison = value == match
                    if isinstance(comparison, bool | np.bool_):
                        values[row] = comparison
                    else:
                        invalid[row] = True
            except (TypeError, ValueError, OverflowError, InvalidOperation):
                invalid[row] = True
        SimulateSolveZeroKernel._refuse_rows(
            frame,
            (source_entity, record["column"]),
            missing,
            f"mask source {source_entity}.{record['column']} must not contain missing values",
        )
        SimulateSolveZeroKernel._refuse_rows(
            frame,
            (source_entity, record["column"]),
            invalid,
            f"mask source {source_entity}.{record['column']} must contain scalar comparable values",
        )
    else:
        values = source.to_numpy() == record["equals"]
    return _map_values(frame, values, source_entity, entity, aggregation=None)


def _map_values(frame, values, source, target, *, aggregation):
    """Map arrays through declared partitions, retaining the target id order."""
    schema = frame.schema
    schema.entity_id_column(source)
    schema.entity_id_column(target)
    if source == target:
        if aggregation is not None:
            raise ValueError("Same-entity terms cannot declare aggregation.")
        return values
    person = schema.person_entity
    if source in schema.group_entities and target == person:
        if aggregation is not None:
            raise ValueError("Broadcast terms cannot declare aggregation.")
        index = pd.Index(frame.table(source)[schema.id_column(source)])
        positions = index.get_indexer(
            frame.table(person)[schema.membership_column(source)]
        )
        return values[positions]
    if source == person and target in schema.group_entities:
        if aggregation not in {"sum", "max", "first"}:
            raise ValueError(
                "Person-to-group terms require sum, max or first aggregation."
            )
        persons = frame.table(person)
        membership = schema.membership_column(target)
        series = pd.Series(values, index=pd.Index(persons[membership]))
        if aggregation == "first":
            order = np.argsort(
                persons[schema.person_id_column].to_numpy(), kind="stable"
            )
            series = series.iloc[order]
        grouped = series.groupby(level=0, sort=False).agg(aggregation)
        return grouped.reindex(frame.table(target)[schema.id_column(target)]).to_numpy()
    raise ValueError(
        "Bridge mapping requires the same entity or a person/group partition."
    )
