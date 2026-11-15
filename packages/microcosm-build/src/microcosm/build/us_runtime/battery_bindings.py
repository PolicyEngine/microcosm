"""US gate bindings for the shared battery (microcosm#1585 item 11, PR 2 of 4).

Every US gate was written before the battery. It takes its evidence as
ordinary arguments, keeps its thresholds as module constants or keyword
defaults, and mints its own result name. A binding here adapts one such gate
to the battery's ``(context, parameters)`` call:

- the evidence comes from named context artifacts, and from the phase frame
  where the gate reads one;
- a threshold the gate already takes as a keyword is routed from the entry
  parameter of the same name, so the battery evaluates with the declared
  value;
- a threshold the gate reads from a module constant is *pinned*: the entry
  declares it, and the binding fails the entry closed unless the declared
  value equals today's constant. The gate functions stay as they are in this
  pull request, which adds code and removes nothing (about 25 of their modules
  are attested by the US seed protocol, so editing them would move its pins).
  The flip (PR 3) removes the constants and turns each pinned threshold into
  a routed keyword;
- the gate's own result name is re-minted under the declared one, as
  :class:`~microcosm.build.gate_battery.EvaluatorBinding` does.

Most US gates are declared as ``stage_health`` entries whose ``check``
parameter names the gate, the UK's pattern for stage-time checks, so that no
name enters the shared vocabulary (``country_spec.ALLOWED_GATE_FUNCTIONS``)
before Max's Step 0 answer. One :class:`USCheckBinding` serves every such
entry, dispatching on ``check``.

A binding calls the gate function the gate's site calls, looked up where the
site looks it up and when it runs, so a test that patches a tool's global
reaches both verdicts. A reference is ``"tool:<name>"`` for a name in the
build tool's own module (the package never imports ``tools/``: the tool
passes its live ``globals()`` when it builds its registry, which works however
the tool was loaded), or ``"<module>:<name>"`` for a package module. Each registry call returns a new mapping and resolves every
reference once up front, so a misspelt name is refused before a gate runs and
no shared registry is mutated.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType, ModuleType
from typing import Any

from microcosm.build.gate_battery import (
    DEFAULT_REGISTRY,
    EvidenceContext,
    GateBinding,
)
from microcosm.build.gates import GateResult

__all__ = [
    "FRAME",
    "ToolNamespace",
    "USCheckBinding",
    "USGateSpec",
    "declared_value",
    "resolve_reference",
    "us_check_binding",
]

#: The argument source that means "the phase's frame", not an artifact.
FRAME = "frame"

_TOOL = "tool"


def declared_value(value: object) -> object:
    """``value`` as a ``gates.json`` entry declares it (JSON's data model)."""

    if isinstance(value, Mapping):
        return {str(key): declared_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [declared_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(declared_value(item) for item in value)
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (bool, int, float, str)):
        return declared_value(item())
    return value


def _routed(value: object) -> object:
    """A declared value as the gate's keyword takes it: JSON lists as tuples.

    Gates format their bounds into failure lines, so a band must reach them
    as the tuple today's default is, or the two verdicts' lines would differ.
    """

    if isinstance(value, list):
        return tuple(_routed(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _routed(item) for key, item in value.items()}
    return value


def _same(left: object, right: object) -> bool:
    return json.dumps(declared_value(left), sort_keys=True) == json.dumps(
        declared_value(right), sort_keys=True
    )


def _split_reference(reference: str) -> tuple[str, str]:
    module, separator, name = reference.partition(":")
    if not separator or not module or not name:
        raise ValueError(
            f"reference {reference!r} must be 'tool:<name>' or '<module>:<name>'."
        )
    return module, name


#: A build tool's namespace: its module, or its live ``globals()``.
ToolNamespace = ModuleType | Mapping[str, Any]


def resolve_reference(reference: str, *, tool: ToolNamespace | None) -> object:
    """The object ``reference`` names, looked up now."""

    module, name = _split_reference(reference)
    if module == _TOOL:
        if tool is None:
            raise ValueError(
                f"{reference!r} names a tool global, but no tool was given."
            )
        if isinstance(tool, Mapping):
            try:
                return tool[name]
            except KeyError:
                raise AttributeError(f"the tool has no global {name!r}") from None
        return getattr(tool, name)
    return getattr(importlib.import_module(module), name)


@dataclass(frozen=True)
class USGateSpec:
    """How the battery calls one US gate.

    Attributes:
        check: The value of the entry's ``check`` parameter.
        legacy_name: The result name the gate mints today; re-minted under
            the declared gate name.
        gate: Reference to the gate function the site calls.
        positional: Argument sources passed positionally, in order: artifact
            keys, or :data:`FRAME`.
        arguments: Keyword argument to its source (artifact key or
            :data:`FRAME`).
        parameters: Threshold keywords the gate takes, each routed from the
            entry parameter of the same name.
        pinned: Entry parameter to the reference of the module constant the
            gate reads it from; the declared value must equal the constant.
        returns_failures: The gate returns failure lines rather than a
            :class:`~microcosm.build.gates.GateResult`; the binding wraps
            them (an empty list passes).
        adapt: Optional final step turning the gate's return value into its
            :class:`~microcosm.build.gates.GateResult`.
    """

    check: str
    legacy_name: str
    gate: str
    positional: tuple[str, ...] = ()
    arguments: Mapping[str, str] = field(default_factory=dict)
    parameters: frozenset[str] = frozenset()
    pinned: Mapping[str, str] = field(default_factory=dict)
    returns_failures: bool = False
    adapt: Callable[[Any], GateResult] | None = None

    def __post_init__(self) -> None:
        if not self.check or not self.legacy_name:
            raise ValueError("a US gate spec needs its check and its legacy name.")
        _split_reference(self.gate)
        for reference in self.pinned.values():
            _split_reference(reference)
        keys = set(self.parameters) | set(self.pinned)
        if "check" in keys:
            raise ValueError(
                f"{self.check}: 'check' is the dispatch key, not a threshold."
            )
        if set(self.parameters) & set(self.pinned):
            raise ValueError(
                f"{self.check}: {sorted(set(self.parameters) & set(self.pinned))} "
                "are both routed and pinned."
            )
        overlap = keys & set(self.arguments)
        if overlap:
            raise ValueError(
                f"{self.check}: {sorted(overlap)} are both evidence and thresholds."
            )
        if self.returns_failures and self.adapt is not None:
            raise ValueError(f"{self.check}: returns_failures and adapt both wrap.")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        object.__setattr__(self, "pinned", MappingProxyType(dict(self.pinned)))

    @property
    def parameter_keys(self) -> frozenset[str]:
        return frozenset(self.parameters) | frozenset(self.pinned)

    @property
    def sources(self) -> tuple[str, ...]:
        return (*self.positional, *self.arguments.values())

    @property
    def artifact_keys(self) -> frozenset[str]:
        return frozenset(source for source in self.sources if source != FRAME)

    @property
    def needs_frame(self) -> bool:
        return FRAME in self.sources

    @property
    def references(self) -> tuple[str, ...]:
        return (self.gate, *self.pinned.values())


def _as_gate_result(value: object, legacy_name: str) -> object:
    """A verdict object that is not a GateResult, read as today's code reads it.

    Today's sites read only ``passed``, ``failures`` and ``details``, and the
    builders' tests stand such objects in for gates; anything else is
    returned as is, and the executor fails it closed.
    """

    passed = getattr(value, "passed", None)
    failures = getattr(value, "failures", None)
    if not isinstance(passed, bool) or failures is None or isinstance(failures, str):
        return value
    return GateResult(
        name=str(getattr(value, "name", legacy_name)),
        passed=passed,
        failures=tuple(str(line) for line in failures),
        details=dict(getattr(value, "details", None) or {}),
    )


def _source(context: EvidenceContext, source: str) -> object:
    return context.frame if source == FRAME else context.artifacts[source]


class USCheckBinding:
    """One declared gate name serving many US gates, chosen by ``check``.

    Implements :class:`~microcosm.build.gate_battery.GateBinding`. Its
    ``parameter_keys`` is ``check`` plus every threshold any of its gates
    takes or pins, so ``validate_gate_parameters`` refuses an unknown key
    before a gate runs. An entry that declares fewer keys than its gate pins,
    or a key its own gate does not take, fails closed when it evaluates.

    Args:
        name: The declared gate name.
        specs: One per check.
        tool: The build tool's module, for ``"tool:"`` references.
    """

    def __init__(
        self,
        name: str,
        specs: Sequence[USGateSpec],
        *,
        tool: ToolNamespace | None = None,
    ) -> None:
        by_check: dict[str, USGateSpec] = {}
        for spec in specs:
            if spec.check in by_check:
                raise ValueError(f"{name}: check {spec.check!r} is declared twice.")
            by_check[spec.check] = spec
        unresolved = []
        for spec in by_check.values():
            for reference in spec.references:
                try:
                    target = resolve_reference(reference, tool=tool)
                except (AttributeError, ImportError, ValueError) as exc:
                    unresolved.append(f"{spec.check}: {reference} ({exc})")
                    continue
                if reference == spec.gate and not callable(target):
                    unresolved.append(f"{spec.check}: {reference} is not callable")
        if unresolved:
            raise ValueError(f"{name}: unresolved references: {unresolved}.")
        self.name = name
        self._tool = tool
        self._specs: Mapping[str, USGateSpec] = MappingProxyType(by_check)
        self.parameter_keys = frozenset(
            {"check"}.union(*(spec.parameter_keys for spec in specs))
        )

    @property
    def checks(self) -> Mapping[str, USGateSpec]:
        return self._specs

    @property
    def tool(self) -> ToolNamespace | None:
        return self._tool

    def _lookup(self, parameters: Mapping[str, Any]) -> USGateSpec | None:
        check = parameters.get("check")
        return self._specs.get(check) if isinstance(check, str) else None

    def _spec(self, parameters: Mapping[str, Any]) -> USGateSpec:
        spec = self._lookup(parameters)
        if spec is None:
            raise ValueError(
                f"{self.name}: no US gate is bound to check "
                f"{parameters.get('check')!r}; bound: {sorted(self._specs)}."
            )
        return spec

    def required_artifacts(self, parameters: Mapping[str, Any]) -> frozenset[str]:
        spec = self._lookup(parameters)
        return spec.artifact_keys if spec is not None else frozenset()

    def requires_frame(self, parameters: Mapping[str, Any]) -> bool:
        spec = self._lookup(parameters)
        return spec.needs_frame if spec is not None else False

    def pinned_values(self, check: str) -> dict[str, object]:
        """Today's value of each threshold ``check`` pins, as JSON data."""

        spec = self._specs[check]
        return {
            key: declared_value(resolve_reference(reference, tool=self._tool))
            for key, reference in spec.pinned.items()
        }

    def evaluate(
        self, context: EvidenceContext, parameters: Mapping[str, Any]
    ) -> GateResult:
        spec = self._spec(parameters)
        declared = {key: value for key, value in parameters.items() if key != "check"}
        unknown = sorted(set(declared) - spec.parameter_keys)
        if unknown:
            raise ValueError(
                f"check {spec.check!r} does not take parameter(s) {unknown}; it "
                f"takes {sorted(spec.parameter_keys)}."
            )
        undeclared = sorted(set(spec.pinned) - set(declared))
        if undeclared:
            raise ValueError(
                f"check {spec.check!r} pins {undeclared}, which the entry does "
                "not declare."
            )
        for key, reference in spec.pinned.items():
            today = resolve_reference(reference, tool=self._tool)
            if not _same(declared[key], today):
                raise ValueError(
                    f"check {spec.check!r}: declared {key}={declared_value(declared[key])!r} "
                    f"differs from today's {reference} = {declared_value(today)!r}; "
                    "until the flip the spec must equal today's constant."
                )
        gate = resolve_reference(spec.gate, tool=self._tool)
        value = gate(
            *(_source(context, source) for source in spec.positional),
            **{
                argument: _source(context, source)
                for argument, source in spec.arguments.items()
            },
            **{
                key: _routed(declared[key])
                for key in spec.parameters
                if key in declared
            },
        )
        if spec.returns_failures:
            if isinstance(value, str):
                raise TypeError(f"check {spec.check!r} returned one string, not lines.")
            lines = tuple(str(line) for line in value)
            value = GateResult(name=spec.legacy_name, passed=not lines, failures=lines)
        elif spec.adapt is not None:
            value = spec.adapt(value)
        elif not isinstance(value, GateResult):
            value = _as_gate_result(value, spec.legacy_name)
        if isinstance(value, GateResult) and value.name == spec.legacy_name:
            value = replace(value, name=self.name)
        return value

    def evidence_payload(
        self, context: EvidenceContext, parameters: Mapping[str, Any]
    ) -> object | None:
        return None


def us_check_binding(
    name: str,
    specs: Sequence[USGateSpec],
    *,
    tool: ToolNamespace | None = None,
    extra: Mapping[str, GateBinding] | None = None,
) -> dict[str, GateBinding]:
    """A fresh registry: the shared bindings, ``extra``, and ``name`` over ``specs``."""

    return {
        **DEFAULT_REGISTRY,
        **dict(extra or {}),
        name: USCheckBinding(name, specs, tool=tool),
    }


# ---------------------------------------------------------------------------
# The four scopes' gates
# ---------------------------------------------------------------------------

_US = "microcosm.build.us_runtime"


def _pins(module: str, *constants: str) -> dict[str, str]:
    """Pinned parameters named after today's constants.

    ``_RENT_SHARE_BAND`` in ``us_runtime.housing_inputs`` is declared as
    ``rent_share_band``.
    """

    return {
        constant.lstrip("_").lower(): f"{_US}.{module}:{constant}"
        for constant in constants
    }


def _frame_check(
    check: str,
    legacy_name: str,
    gate: str,
    module: str,
    *constants: str,
) -> USGateSpec:
    """A signal gate over the stage frame, called as the tool calls it."""

    return USGateSpec(
        check=check,
        legacy_name=legacy_name,
        gate=f"tool:{gate}",
        positional=(FRAME,),
        pinned=_pins(module, *constants),
    )


#: The frame signal gates both the base builder and the national release
#: run, keyed by check: (legacy name, gate function, module, pinned
#: constants). Each tool calls its own global of the same name.
_SCOPE_SPECS: Mapping[str, Callable[[], tuple[USGateSpec, ...]]] = MappingProxyType({})


def us_gate_specs(scope: str) -> tuple[USGateSpec, ...]:
    """The gates a scope's ``stage_health`` entries name by ``check``."""

    try:
        build = _SCOPE_SPECS[scope]
    except KeyError:
        raise ValueError(
            f"no US gate specs for scope {scope!r}; declared: {sorted(_SCOPE_SPECS)}."
        ) from None
    return build()


def us_gate_registry(scope: str, *, tool: ToolNamespace) -> dict[str, GateBinding]:
    """A fresh registry for one tool's scope.

    ``tool`` is the build tool's live ``globals()``, whose names the bindings
    call, so the gate a site calls is the gate the battery calls. Resolves
    every reference now: a name the tool does not have is refused before a
    gate runs.
    """

    return us_check_binding("stage_health", us_gate_specs(scope), tool=tool)
