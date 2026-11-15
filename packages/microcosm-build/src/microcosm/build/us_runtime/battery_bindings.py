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
    "RETURNS",
    "us_flag_overrides",
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
        returns: How the gate hands back its verdict (:data:`RETURNS`):
            ``result``, a :class:`~microcosm.build.gates.GateResult` (or an
            object with ``passed`` and ``failures``); ``failures``, failure
            lines (none is a pass); ``failures_and_details``, a pair of those
            lines and a details mapping; ``result_and_details``, a pair whose
            first item is the result; ``failure_or_none``, one failure line or
            ``None``; ``mapping``, a dict carrying ``passed`` and
            ``failures``; or ``receipt``, a refusal-style check whose
            returning at all is a pass and whose ``refusals`` are its
            failures.
        refusals: Exception types a refusal-style check raises to fail; the
            binding reads one as a failure whose line is its message, as
            the shadow reads today's.
        flags: Routed parameter to the tool's argparse destination that
            sets it per run. The entry declares the parser's default; a
            build that passes another value is a runtime override, which
            the shadow records and explains.
    """

    check: str
    legacy_name: str
    gate: str
    positional: tuple[str, ...] = ()
    arguments: Mapping[str, str] = field(default_factory=dict)
    parameters: frozenset[str] = frozenset()
    pinned: Mapping[str, str] = field(default_factory=dict)
    returns: str = "result"
    refusals: tuple[type[BaseException], ...] = ()
    flags: Mapping[str, str] = field(default_factory=dict)

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
        if self.returns not in RETURNS:
            raise ValueError(
                f"{self.check}: returns {self.returns!r} is not one of {sorted(RETURNS)}."
            )
        if self.returns == "receipt" and not self.refusals:
            raise ValueError(f"{self.check}: a receipt check fails only by a refusal.")
        stray_flags = sorted(set(self.flags) - set(self.parameters))
        if stray_flags:
            raise ValueError(
                f"{self.check}: flags {stray_flags} name parameters it does not route."
            )
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        object.__setattr__(self, "pinned", MappingProxyType(dict(self.pinned)))
        object.__setattr__(self, "flags", MappingProxyType(dict(self.flags)))

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
        try:
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
        except spec.refusals as refusal:
            return GateResult(name=self.name, passed=False, failures=(str(refusal),))
        value = _verdict_of(spec, value)
        if isinstance(value, GateResult) and value.name == spec.legacy_name:
            value = replace(value, name=self.name)
        return value

    def evidence_payload(
        self, context: EvidenceContext, parameters: Mapping[str, Any]
    ) -> object | None:
        return None


#: How a gate hands back its verdict (:attr:`USGateSpec.returns`).
RETURNS = frozenset(
    {
        "result",
        "failures",
        "failures_and_details",
        "result_and_details",
        "failure_or_none",
        "mapping",
        "receipt",
    }
)


def _lines(value: object, check: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"check {check!r} returned one string, not lines.")
    return tuple(str(line) for line in value)


def _verdict_of(spec: USGateSpec, value: Any) -> object:
    """The gate's return value as its result, by :attr:`USGateSpec.returns`."""

    name = spec.legacy_name
    if spec.returns == "result":
        return value if isinstance(value, GateResult) else _as_gate_result(value, name)
    if spec.returns == "result_and_details":
        first = value[0]
        return first if isinstance(first, GateResult) else _as_gate_result(first, name)
    if spec.returns == "receipt":
        return GateResult(name=name, passed=True)
    if spec.returns == "mapping":
        failures = _lines(value.get("failures") or (), spec.check)
        details = {key: item for key, item in value.items() if key != "failures"}
        return GateResult(
            name=name,
            passed=bool(value.get("passed")),
            failures=failures,
            details=details,
        )
    if spec.returns == "failure_or_none":
        if value is None:
            return GateResult(name=name, passed=True)
        return GateResult(name=name, passed=False, failures=(str(value),))
    details: dict[str, object] = {}
    if spec.returns == "failures_and_details":
        value, raw_details = value
        details = dict(raw_details or {})
    failures = _lines(value, spec.check)
    return GateResult(
        name=name, passed=not failures, failures=failures, details=details
    )


def us_flag_overrides(
    gates: Any, specs: Sequence[USGateSpec], args: Any
) -> dict[str, dict[str, dict[str, object]]]:
    """The declared thresholds this build's flags overrode, by entry id.

    For each ``stage_health`` entry whose gate routes a flag-set threshold,
    a parsed value that differs from the declared one is an override.
    """

    by_check = {spec.check: spec for spec in specs}
    overrides: dict[str, dict[str, dict[str, object]]] = {}
    for entry in gates.gates:
        if entry.gate != "stage_health":
            continue
        spec = by_check.get(entry.parameters.get("check"))
        if spec is None:
            continue
        for key, destination in spec.flags.items():
            if not hasattr(args, destination):
                continue
            runtime = getattr(args, destination)
            declared = entry.parameters.get(key)
            if not _same(declared, runtime):
                overrides.setdefault(entry.id, {})[key] = {
                    "declared": declared_value(declared),
                    "runtime": declared_value(runtime),
                }
    return overrides


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
_SHARED_FRAME_SIGNALS: Mapping[str, tuple[str, str, str, tuple[str, ...]]] = (
    MappingProxyType(
        {
            "relationship_inputs_signal": (
                "relationship_inputs_signal",
                "us_relationship_inputs_signal_gate",
                "relationship_inputs",
                (
                    "_HOUSEHOLD_HEAD_SHARE_BAND",
                    "_SEPARATED_SHARE_BAND",
                    "_SURVIVING_SPOUSE_SHARE_BAND",
                ),
            ),
            "spm_independence_role_signal": (
                "spm_independence_role_signal",
                "us_spm_independence_role_signal_gate",
                "spm_independence_role",
                ("_ROLE_SHARE_BAND", "_MINOR_ROLE_SHARE_BAND"),
            ),
            "medicare_take_up_input_signal": (
                "medicare_take_up_input_signal",
                "us_medicare_take_up_signal_gate",
                "medicare_take_up",
                ("_WEIGHTED_ENROLLED_SHARE_BAND", "_CHANNEL_ENROLLED_SHARE_BAND"),
            ),
            "housing_inputs_signal": (
                "housing_inputs_signal",
                "us_housing_inputs_signal_gate",
                "housing_inputs",
                ("_RENT_SHARE_BAND", "_HOUSING_ASSISTANCE_SHARE_BAND"),
            ),
            "eligibility_inputs_signal": (
                "eligibility_inputs_signal",
                "us_eligibility_inputs_signal_gate",
                "eligibility_inputs",
                (
                    "_DISABLED_SHARE_BAND",
                    "_FULL_TIME_STUDENT_SHARE_BAND",
                    "_PARENT_SHARE_BAND",
                    "_VETERANS_BENEFITS_SHARE_BAND",
                ),
            ),
            "pregnancy_signal": (
                "pregnancy_signal",
                "us_pregnancy_signal_gate",
                "pregnancy",
                ("_PREGNANT_SHARE_BAND",),
            ),
            "wic_claim_input_signal": (
                "wic_claim_input_signal",
                "us_wic_claim_signal_gate",
                "wic_claim",
                ("_WEIGHTED_CLAIM_SHARE_BAND", "_CATEGORY_WEIGHTED_CLAIM_SHARE_BANDS"),
            ),
            "esi_premiums_signal": (
                "esi_premiums_signal",
                "us_esi_premiums_signal_gate",
                "esi_premiums",
                ("_POSITIVE_SHARE_BAND", "_RAW_MEAN_BAND"),
            ),
            "prior_year_income_signal": (
                "prior_year_income_signal",
                "us_prior_year_income_signal_gate",
                "prior_year_income",
                (
                    "_PREVIOUS_YEAR_AVAILABLE_SHARE_BAND",
                    "_SELF_EMPLOYMENT_NONZERO_SHARE_BAND",
                ),
            ),
            "prior_year_income_source_reconciliation": (
                "prior_year_income_source_reconciliation",
                "us_prior_year_income_source_reconciliation_gate",
                "prior_year_income",
                (),
            ),
            "qbi_inputs_signal": (
                "qbi_inputs_signal",
                "us_qbi_inputs_signal_gate",
                "qbi_inputs",
                (
                    "_BOOLEAN_SHARE_BANDS",
                    "_NUMERIC_NONZERO_SHARE_BANDS",
                    "_INVARIANT_ATOL",
                ),
            ),
            "farm_business_income_signal": (
                "farm_business_income_signal",
                "us_farm_business_income_signal_gate",
                "farm_business_income",
                ("_NONZERO_SHARE_BANDS",),
            ),
            "domestic_production_ald_signal": (
                "domestic_production_ald_signal",
                "us_domestic_production_ald_signal_gate",
                "domestic_production",
                ("_DOMESTIC_PRODUCTION_ALD_POSITIVE_SHARE_BAND",),
            ),
            "child_support_inputs_signal": (
                "child_support_inputs_signal",
                "us_child_support_signal_gate",
                "child_support",
                ("_NONZERO_SHARE_BANDS", "_CHANNEL_NONZERO_SHARE_BAND"),
            ),
            "disability_benefits_signal": (
                "disability_benefits_signal",
                "us_disability_benefits_signal_gate",
                "disability_benefits",
                ("_NONZERO_SHARE_BAND", "_CHANNEL_NONZERO_SHARE_BANDS"),
            ),
            "workers_compensation_signal": (
                "workers_compensation_signal",
                "us_workers_compensation_signal_gate",
                "workers_compensation",
                ("_NONZERO_SHARE_BAND", "_CHANNEL_NONZERO_SHARE_BANDS"),
            ),
            "weeks_unemployed_signal": (
                "weeks_unemployed_signal",
                "us_weeks_unemployed_signal_gate",
                "weeks_unemployed",
                (
                    "_CHANNEL_POSITIVE_SHARE_BANDS",
                    "_CHANNEL_WEIGHTED_MEAN_WEEKS_BANDS",
                ),
            ),
            "educator_expense_signal": (
                "educator_expense_signal",
                "us_educator_expense_signal_gate",
                "educator_expenses",
                ("_OVERALL_POSITIVE_SHARE_BAND", "_PUF_POSITIVE_SHARE_BAND"),
            ),
            "form_4952_election_signal": (
                "form_4952_election_signal",
                "us_form_4952_election_signal_gate",
                "form_4952",
                ("_OVERALL_POSITIVE_SHARE_BAND", "_PUF_POSITIVE_SHARE_BAND"),
            ),
            "salt_refund_income_signal": (
                "salt_refund_income_signal",
                "us_salt_refund_income_signal_gate",
                "salt_refund_income",
                ("_OVERALL_POSITIVE_SHARE_BAND", "_PUF_POSITIVE_SHARE_BAND"),
            ),
            "capital_gain_details_signal": (
                "capital_gain_details_signal",
                "us_capital_gain_details_signal_gate",
                "capital_gain_details",
                ("_SIGNAL_BANDS",),
            ),
            "childcare_inputs_signal": (
                "childcare_inputs_signal",
                "us_childcare_signal_gate",
                "childcare",
                ("_NONZERO_SHARE_BAND",),
            ),
            "adult_care_inputs_signal": (
                "adult_care_inputs_signal",
                "us_adult_care_signal_gate",
                "adult_care",
                ("_FLAG_SHARE_BAND", "_EXPENSE_PLAUSIBILITY_CEILING"),
            ),
            "energy_subsidy_signal": (
                "energy_subsidy_signal",
                "us_energy_subsidy_signal_gate",
                "energy_subsidy",
                ("_OVERALL_POSITIVE_SHARE_BAND", "_CHANNEL_POSITIVE_SHARE_BANDS"),
            ),
            "alimony_inputs_signal": (
                "alimony_inputs_signal",
                "us_alimony_signal_gate",
                "alimony",
                ("_NONZERO_SHARE_BAND",),
            ),
            "casualty_loss_signal": (
                "casualty_loss_signal",
                "us_casualty_loss_signal_gate",
                "casualty_losses",
                ("_CASUALTY_LOSS_NONZERO_SHARE_BAND",),
            ),
            "misc_itemized_signal": (
                "misc_itemized_signal",
                "us_misc_itemized_signal_gate",
                "misc_itemized",
                ("_MISC_ITEMIZED_NONZERO_SHARE_BAND",),
            ),
            "retirement_contributions_signal": (
                "retirement_contributions_signal",
                "us_retirement_contributions_signal_gate",
                "retirement_contributions",
                ("_NONZERO_SHARE_BAND",),
            ),
            "retirement_distributions_signal": (
                "retirement_distributions_signal",
                "us_retirement_distributions_signal_gate",
                "retirement_distributions",
                ("_NONZERO_SHARE_BANDS",),
            ),
            "education_inputs_signal": (
                "education_inputs_signal",
                "us_education_inputs_signal_gate",
                "education_inputs",
                ("_TUITION_SHARE_BAND", "_ASSISTANCE_SHARE_BAND"),
            ),
        }
    )
)


def _shared_frame_checks(*checks: str) -> tuple[USGateSpec, ...]:
    return tuple(
        _frame_check(check, legacy, gate, module, *constants)
        for check in checks
        for legacy, gate, module, constants in (_SHARED_FRAME_SIGNALS[check],)
    )


#: Thresholds the geography ladder gate takes as keywords today.
_GEOGRAPHY_LADDER_THRESHOLDS = frozenset(
    {
        "nyc_share_bounds",
        "nyc_share_of_new_york_bounds",
        "place_share_bounds",
        "cbsa_share_bounds",
        "sld_min_share",
    }
)


def _us_base_specs() -> tuple[USGateSpec, ...]:
    return (
        *_shared_frame_checks(
            "relationship_inputs_signal",
            "spm_independence_role_signal",
            "medicare_take_up_input_signal",
            "housing_inputs_signal",
            "eligibility_inputs_signal",
            "pregnancy_signal",
            "wic_claim_input_signal",
            "esi_premiums_signal",
            "prior_year_income_signal",
            "prior_year_income_source_reconciliation",
            "qbi_inputs_signal",
            "farm_business_income_signal",
            "domestic_production_ald_signal",
            "child_support_inputs_signal",
            "disability_benefits_signal",
            "workers_compensation_signal",
            "weeks_unemployed_signal",
            "educator_expense_signal",
            "form_4952_election_signal",
            "salt_refund_income_signal",
            "capital_gain_details_signal",
            "childcare_inputs_signal",
            "adult_care_inputs_signal",
            "energy_subsidy_signal",
            "alimony_inputs_signal",
            "casualty_loss_signal",
            "misc_itemized_signal",
            "retirement_contributions_signal",
            "retirement_distributions_signal",
            "education_inputs_signal",
        ),
        USGateSpec(
            check="geography_ladder",
            legacy_name="us_geography_ladder",
            gate="tool:us_geography_ladder_gate",
            positional=("household", "household_weights"),
            parameters=_GEOGRAPHY_LADDER_THRESHOLDS,
        ),
        USGateSpec(
            check="block_location",
            legacy_name="us_block_location",
            gate="tool:us_block_location_gate",
            positional=("household", "location_ladder"),
        ),
    )


#: Configuration and thresholds the PUMA ladder gate takes as keywords today.
_PUMA_LADDER_KEYWORDS = frozenset(
    {"assign_tract", "nyc_share_bounds", "nyc_share_of_new_york_bounds"}
)


def _us_pool_specs() -> tuple[USGateSpec, ...]:
    return (
        USGateSpec(
            check="puma_ladder",
            legacy_name="us_puma_ladder",
            gate="tool:us_puma_ladder_gate",
            positional=("household", "household_weights"),
            parameters=_PUMA_LADDER_KEYWORDS,
        ),
        USGateSpec(
            check="block_location",
            legacy_name="us_block_location",
            gate="tool:us_block_location_gate",
            positional=("household", "location_ladder"),
        ),
        USGateSpec(
            # Evaluated inside the pool runtime's source preparation, not in
            # the tool, so the binding calls the runtime's own global.
            check="hours_worked_signal",
            legacy_name="hours_worked_signal",
            gate=f"{_US}.multispine_pool:us_hours_worked_signal_gate",
            positional=(FRAME,),
            pinned=_pins(
                "hours_worked", "_WORKED_SHARE_BAND", "_MEAN_WEEKLY_HOURS_BAND"
            ),
        ),
        USGateSpec(
            check="stacked_completeness",
            legacy_name="us_stacked_completeness",
            gate="tool:stacked_completeness_gate",
            positional=(FRAME,),
            arguments={"tail_manifest": "tail_manifest"},
        ),
        USGateSpec(
            check="by_origin_battery",
            legacy_name="us_by_origin_battery",
            gate="tool:by_origin_battery",
            positional=(FRAME,),
            arguments={"tail_manifest": "tail_manifest"},
            pinned=_pins(
                "stacked_spine",
                "_BATTERY_INCIDENCE_RATIO_BOUNDS",
                "_BATTERY_QUANTILES",
                "_BATTERY_QUANTILE_ENVELOPE_TOLERANCE",
                "_BATTERY_CATEGORICAL_TVD_TOLERANCE",
                "_MASS_RTOL",
            ),
        ),
    )


def _tool_pins(*constants: str) -> dict[str, str]:
    """Pinned parameters named after the tool's own module constants."""

    return {constant.lstrip("_").lower(): f"tool:{constant}" for constant in constants}


#: Frame signal gates only the national release runs: (check, legacy name,
#: gate, module, pinned constants).
_NATIONAL_FRAME_SIGNALS: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    (
        "immigration_composition",
        "immigration_composition",
        "us_immigration_composition_gate",
        "immigration",
        ("_NON_CITIZEN_SHARE_BAND", "_UNDOCUMENTED_ANCHOR_RELATIVE_BAND"),
    ),
    (
        "take_up_signal",
        "take_up_signal",
        "us_take_up_signal_gate",
        "take_up",
        ("US_TAKE_UP_SHARE_BAND",),
    ),
    (
        "hours_worked_signal",
        "hours_worked_signal",
        "us_hours_worked_signal_gate",
        "hours_worked",
        ("_WORKED_SHARE_BAND", "_MEAN_WEEKLY_HOURS_BAND"),
    ),
    (
        "snap_take_up_signal",
        "snap_take_up_signal",
        "us_snap_take_up_signal_gate",
        "snap_take_up",
        ("_TAKE_UP_SHARE_BAND",),
    ),
    (
        "snap_discretionary_exemption_signal",
        "snap_discretionary_exemption_signal",
        "us_snap_discretionary_exemption_signal_gate",
        "snap_discretionary_exemption",
        ("_EXEMPT_SHARE_BAND",),
    ),
    (
        "ssi_disability_criteria_signal",
        "ssi_disability_criteria_signal",
        "us_ssi_disability_criteria_signal_gate",
        "ssi_disability_criteria",
        ("_TRUE_SHARE_BAND",),
    ),
    (
        "sipp_head_start_signal",
        "sipp_head_start_signal",
        "us_sipp_head_start_signal_gate",
        "sipp_head_start",
        ("_ELIGIBLE_TAKE_UP_SHARE_BAND",),
    ),
    (
        "scf_auto_loans_signal",
        "scf_auto_loans_signal",
        "us_scf_auto_loans_signal_gate",
        "scf_auto_loans",
        ("_AUTO_LOAN_NONZERO_SHARE_BAND",),
    ),
    (
        "scf_education_debt_signal",
        "scf_education_debt_signal",
        "us_scf_education_debt_signal_gate",
        "scf_education_debt",
        ("_BORROWER_HOUSEHOLD_SHARE_BAND",),
    ),
    (
        "sipp_vehicles_signal",
        "sipp_vehicles_signal",
        "us_sipp_vehicles_signal_gate",
        "sipp_vehicles",
        ("_OWNED_NONZERO_SHARE_BAND", "_VALUE_NONZERO_SHARE_BAND"),
    ),
    (
        "voluntary_filing_signal",
        "voluntary_filing_signal",
        "us_voluntary_filing_signal_gate",
        "voluntary_filing",
        ("_TRUE_SHARE_BAND",),
    ),
    (
        "sipp_tips_signal",
        "sipp_tips_signal",
        "us_sipp_tips_signal_gate",
        "sipp_tips",
        ("_TIPPED_OCCUPATION_SHARE_BAND", "_TIP_INCOME_NONZERO_SHARE_BAND"),
    ),
    (
        "org_wages_signal",
        "org_wages_signal",
        "us_org_wages_signal_gate",
        "org_wages",
        ("_SHARE_BANDS",),
    ),
    (
        "other_health_insurance_premiums_signal",
        "other_health_insurance_premiums_signal",
        "us_other_health_insurance_signal_gate",
        "other_health_insurance",
        ("_POSITIVE_SHARE_BAND",),
    ),
)


def _us_national_specs() -> tuple[USGateSpec, ...]:
    return (
        # Preflight: the registers and the target surface, before any stage.
        USGateSpec(
            check="validation_input_coverage",
            legacy_name="us_validation_input_coverage",
            gate="tool:us_validation_input_coverage_gate",
        ),
        USGateSpec(
            check="register_consistency",
            legacy_name="us_register_consistency",
            gate="tool:us_register_consistency_gate",
            arguments={
                "degenerate_reviewed_exclusions": "degenerate_reviewed_exclusions",
                "documented_absent_inputs": "documented_absent_inputs",
                "nonconstant_required_columns": "nonconstant_required_columns",
            },
        ),
        USGateSpec(
            check="esi_anchor_feed",
            legacy_name="esi_anchor_feed",
            gate="tool:us_esi_premiums_anchor_feed_mismatches",
            positional=("ledger_facts",),
            returns="failures",
        ),
        USGateSpec(
            check="target_parity",
            legacy_name="us_release_target_parity",
            gate="tool:us_release_target_parity_gate",
            positional=("target_parity_registry",),
        ),
        USGateSpec(
            check="target_profile_coverage",
            legacy_name="target_profile_coverage",
            gate="tool:target_profile_coverage_gate",
            positional=("target_specs", "target_coverage_requirements"),
        ),
        # The staged stages' signal gates.
        USGateSpec(
            check="capital_gains_tail_survives_selection",
            legacy_name="capital_gains_tail_survives_selection",
            gate="tool:assert_puf_capital_gains_tail_survives_selection",
            positional=("tail_candidate_frame", FRAME),
            arguments={"require_present": "tail_require_present"},
            refusals=(ValueError,),
            returns="receipt",
        ),
        *_shared_frame_checks(
            "weeks_unemployed_signal",
            "qbi_inputs_signal",
            "farm_business_income_signal",
            "domestic_production_ald_signal",
            "child_support_inputs_signal",
            "disability_benefits_signal",
            "workers_compensation_signal",
            "educator_expense_signal",
            "form_4952_election_signal",
            "salt_refund_income_signal",
            "capital_gain_details_signal",
            "childcare_inputs_signal",
            "energy_subsidy_signal",
            "alimony_inputs_signal",
            "casualty_loss_signal",
            "misc_itemized_signal",
            "retirement_contributions_signal",
            "relationship_inputs_signal",
            "spm_independence_role_signal",
            "medicare_take_up_input_signal",
            "prior_year_income_signal",
            "housing_inputs_signal",
            "retirement_distributions_signal",
            "eligibility_inputs_signal",
            "education_inputs_signal",
            "pregnancy_signal",
            "wic_claim_input_signal",
            "esi_premiums_signal",
        ),
        *(
            _frame_check(check, legacy, gate, module, *constants)
            for check, legacy, gate, module, constants in _NATIONAL_FRAME_SIGNALS
        ),
        USGateSpec(
            check="base_population_scale",
            legacy_name="base_population_scale",
            gate="tool:_base_population_scale_gate",
            positional=(FRAME,),
            arguments={"mass_repair": "base_population_mass_repair"},
            pinned=_tool_pins("US_BASE_PERSON_POPULATION_MAX_ABS_RELATIVE_ERROR"),
        ),
        USGateSpec(
            check="reported_coverage_vintage_signal",
            legacy_name="reported_coverage_vintage_signal",
            gate="tool:us_reported_coverage_vintage_signal_gate",
            positional=(FRAME,),
            pinned=_pins("cps_carried", "US_REPORTED_COVERAGE_VINTAGE_GATE_MIN_ROWS"),
        ),
        USGateSpec(
            check="scf_wealth_signal",
            legacy_name="scf_wealth_signal",
            gate="tool:us_scf_wealth_signal_gate",
            positional=(FRAME,),
            arguments={"require_sipp_blend": "scf_require_sipp_blend"},
            pinned=_pins(
                "scf_wealth",
                "_BANK_NONZERO_SHARE_BAND",
                "_STOCK_NONZERO_SHARE_BAND",
                "_BOND_NONZERO_SHARE_BAND",
                "_NET_WORTH_NONZERO_SHARE_BAND",
                "_NET_WORTH_POSITIVE_SHARE_BAND",
                "_NET_WORTH_NEGATIVE_SHARE_BAND",
                "_SCF_SOURCE_SHARE_BAND",
                "_BANK_LOW_TAIL_THRESHOLD",
            ),
        ),
        USGateSpec(
            check="ssi_take_up",
            legacy_name="ssi_take_up",
            gate="tool:us_ssi_take_up_gate",
            positional=("ssi_take_up_diagnostics",),
            arguments={"targets": "ssi_band_targets"},
        ),
        USGateSpec(
            check="health_input_signal",
            legacy_name="health_input_signal",
            gate="tool:_health_input_signal_gate",
            positional=(FRAME,),
        ),
        USGateSpec(
            check="medicaid_take_up",
            legacy_name="medicaid_take_up",
            gate="tool:us_medicaid_take_up_gate",
            positional=("medicaid_take_up_diagnostics",),
            parameters=frozenset({"tolerance"}),
        ),
        USGateSpec(
            check="snap_state_take_up",
            legacy_name="snap_state_take_up",
            gate="tool:us_snap_state_take_up_gate",
            positional=("snap_state_take_up_diagnostics",),
            parameters=frozenset({"tolerance"}),
        ),
        USGateSpec(
            check="input_mass_parity",
            legacy_name="input_mass_parity",
            gate="tool:us_input_mass_parity_gate",
            positional=("candidate_input_mass_totals", "reference_input_mass_totals"),
            arguments={
                "carriers": "input_mass_carriers",
                "candidate_name": "input_mass_candidate_name",
                "reference_name": "input_mass_reference_name",
            },
            parameters=frozenset({"relative_tolerance", "minimum_reference_total"}),
            flags={
                "relative_tolerance": "input_mass_relative_tolerance",
                "minimum_reference_total": "input_mass_minimum_reference_total",
            },
        ),
        USGateSpec(
            check="degenerate_input_signal",
            legacy_name="degenerate_input_signal",
            gate="tool:_degenerate_input_signal_gate",
            positional=(FRAME, "policyengine_engine"),
        ),
        USGateSpec(
            check="ecps_parity",
            legacy_name="parity",
            gate="tool:_ecps_parity_gate",
            positional=(FRAME,),
        ),
        # After the solve: the terminal batch and the export checks.
        USGateSpec(
            check="exact_k_puf_capital_gains_tail",
            legacy_name="exact_k_puf_capital_gains_tail",
            gate="tool:_exact_k_puf_tail_support_gate",
            positional=(FRAME, "exact_k_support"),
        ),
        USGateSpec(
            check="ssi_take_up_delivery",
            legacy_name="ssi_take_up_delivery",
            gate="tool:us_ssi_take_up_delivery_gate",
            positional=("ssi_take_up_diagnostics",),
            arguments={
                "targets": "ssi_band_targets",
                "enforcement_fences": "ssi_enforcement_fences",
            },
            pinned=_pins(
                "ssi_take_up", "US_SSI_TAKE_UP_BAND_DELIVERY_RELATIVE_TOLERANCE"
            ),
        ),
        USGateSpec(
            check="exact_k_frozen_register_fit",
            legacy_name="exact_k_frozen_register_fit",
            gate="tool:_exact_k_frozen_register_fit_gate",
            positional=("calibration_result", "incumbent_diagnostics"),
            arguments={
                "target_registry": "target_registry",
                "target_loss_weights": "target_loss_weights",
                "configured_loss_basis": "configured_loss_basis",
                "incumbent_loss_basis": "incumbent_loss_basis",
            },
            pinned=_tool_pins("US_FISCAL_TARGET_LOSS_CAP"),
        ),
        USGateSpec(
            check="release_gate_failures",
            legacy_name="release_gate_failures",
            gate="tool:_release_gate_failures",
            positional=("calibration_result", "target_compilation"),
            arguments={
                name: f"release_gates_{name}"
                for name in (
                    "target_profile_gate",
                    "health_input_gate",
                    "base_population_gate",
                    "incumbent_diagnostics",
                    "immigration_gate",
                    "input_mass_reference_gate",
                    "degenerate_input_gate",
                    "ecps_parity_gate",
                    "hours_worked_gate",
                    "snap_take_up_gate",
                    "eligibility_inputs_gate",
                    "pregnancy_gate",
                    "reported_coverage_vintage_gate",
                    "snap_discretionary_exemption_gate",
                    "target_registry",
                )
            },
            returns="failures",
        ),
        USGateSpec(
            check="spm_composition",
            legacy_name="spm_composition",
            gate="tool:_spm_composition_gate_failures",
            positional=(FRAME,),
            arguments={"stage": "spm_composition_stage"},
            returns="failures_and_details",
        ),
        USGateSpec(
            check="stored_inputs",
            legacy_name="stored_inputs",
            gate="tool:_stored_input_gate_failures",
            positional=(FRAME,),
            arguments={"stage": "stored_inputs_stage"},
            returns="failures_and_details",
        ),
        USGateSpec(
            check="release_input_coverage",
            legacy_name="us_release_input_coverage",
            gate="tool:us_release_input_coverage_gate",
            positional=(FRAME, "release_engine"),
        ),
        USGateSpec(
            check="esi_premiums_anchor",
            legacy_name="esi_premiums_anchor",
            gate="tool:us_esi_premiums_anchor_gate",
            positional=(FRAME,),
            arguments={"time_period": "time_period"},
            pinned=_pins(
                "esi_premiums",
                "ANCHOR_RELATIVE_TOLERANCE",
                "_EMPLOYED_SHARE_BAND",
                "_RAW_MEAN_BAND",
            ),
        ),
        USGateSpec(
            check="export_input_mass_parity",
            legacy_name="input_mass_parity",
            gate="tool:_export_input_mass_gate",
            positional=(FRAME, "base_frame"),
            arguments={
                "reference_frame": "export_input_mass_reference_frame",
                "reference_name": "export_input_mass_reference_name",
                "reviewed_exclusions": "export_input_mass_reviewed_exclusions",
            },
            parameters=frozenset({"relative_tolerance", "minimum_reference_total"}),
            flags={
                "relative_tolerance": "input_mass_relative_tolerance",
                "minimum_reference_total": "input_mass_minimum_reference_total",
            },
        ),
        USGateSpec(
            check="qrf_tail_concentration",
            legacy_name="tail_concentration",
            gate="tool:_qrf_tail_concentration_gate",
            positional=(FRAME,),
            arguments={"reviewed_exclusions": "qrf_tail_reviewed_exclusions"},
            returns="result_and_details",
            pinned=_tool_pins(
                "US_QRF_TAIL_CONCENTRATION_TOP_K",
                "US_QRF_TAIL_CONCENTRATION_MAX_TOP_SHARE",
                "US_QRF_TAIL_CONCENTRATION_MIN_NONZERO_RECORDS",
                "US_QRF_SPARSE_NONZERO_SHARE_MAX",
            ),
        ),
        USGateSpec(
            check="stored_inputs_written",
            legacy_name="stored_inputs_written",
            gate="tool:_written_stored_input_verdict_mismatch",
            positional=("dataset_path", "pre_export_stored_inputs"),
            returns="failure_or_none",
        ),
    )


_SCOPE_SPECS: Mapping[str, Callable[[], tuple[USGateSpec, ...]]] = MappingProxyType(
    {
        "us_base": _us_base_specs,
        "us_pool": _us_pool_specs,
        "us_national": _us_national_specs,
    }
)


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
