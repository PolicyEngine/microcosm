"""UK bindings for the country-agnostic gate battery (microcosm#611).

The UK's gates are declared in ``uk/gates.json`` under the shared
country-neutral vocabulary; this module supplies the other half of the
contract — the :class:`~microcosm.build.gate_battery.GateBinding`
implementations that know where each gate's evidence lives in an
:class:`~microcosm.build.gate_battery.EvidenceContext` and how declared
spec parameters become gate arguments. The gate *comparisons* stay in
:mod:`microcosm.build.uk_runtime.terminal_gates` and
:mod:`microcosm.build.uk_runtime.weighted_integrity`, unchanged: a
binding adapts evidence, it never re-implements a verdict, so the
migration onto the battery executor is behaviour-preserving by
construction (the differential test pins it).

Two legacy evaluators mint UK-flavored result names
(``uk_release_input_coverage``, ``qrf_tail_concentration``); their
bindings re-mint the verdict under the declared neutral name, because the
battery executor fails a gate closed when the returned name disagrees
with the declared one. Any *other* unexpected name passes through
untouched so that check keeps biting.

The evidence surface handed to the legacy gate modules (the three entity
tables plus period, weight-kind, and mass-log metadata) lives in
:mod:`microcosm.build.uk_runtime.national_frame` as the single copy shared
with the legacy report evaluator; the Frame-typed gates (input-mass parity,
QRF tail concentration) read the frame directly and skip it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any

from microcosm.build.gate_battery import (
    DEFAULT_REGISTRY,
    EvidenceContext,
    GateBinding,
)
from microcosm.build.gates import (
    GateResult,
    nonnegative_columns_gate,
    weights_audit_gate,
)
from microcosm.build.uk_runtime.national_frame import _uk_gate_surface
from microcosm.build.uk_runtime.release_input_coverage import (
    assert_uk_release_input_coverage_build_stages,
    assert_uk_release_input_coverage_manifest_current,
    uk_release_input_coverage_gate,
)
from microcosm.build.uk_runtime.source_runtime import UK_NONNEGATIVE_OUTPUTS_BY_STAGE
from microcosm.build.uk_runtime.terminal_gates import (
    UKZeroWeightStratumDeclaration,
    _household_weights,
    _missing_fit_weight_evidence_gate,
    uk_default_degenerate_reviewed_exclusions,
    uk_degenerate_release_surface_gate,
    uk_export_surface_gate,
    uk_target_fit_gate,
    uk_target_surface_gate,
    uk_weight_ess_gate,
    uk_weight_ratio_gate,
    uk_zero_weight_strata_gate,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE,
    UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
    UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256,
    UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE,
    UKReviewedExclusion,
    _input_mass_reference_evidence_sha256,
    coerce_reviewed_exclusions,
    uk_input_mass_parity_gate,
    uk_input_mass_totals,
    uk_qrf_tail_concentration_columns,
    uk_qrf_tail_concentration_gate,
)

__all__ = [
    "UK_GATE_REGISTRY",
    "UKGateBinding",
]


@dataclass(frozen=True)
class UKGateBinding:
    """A :class:`GateBinding` over a UK runtime evaluator.

    Attributes:
        name: The declared country-neutral gate name the battery expects
            back.
        evaluator: Runs the legacy UK gate over ``(context, parameters)``.
            Declared parameters it does not consume pass through to the
            gate call, so an unknown parameter raises and fails closed —
            the same discipline as the battery's ``FunctionBinding``.
        parameter_keys: The spec parameter keys the evaluator can route —
            checked against every declared entry before any gate runs
            (``validate_gate_parameters``), so a typo'd parameter is
            refused at arm time instead of surfacing as a mid-battery
            evaluator error. Fail-closed empty by default.
        artifact_keys: Context artifact keys the gate needs; missing keys
            resolve to ``evidence_absent`` before evaluation.
        needs_frame: Whether the gate reads the phase frame.
        frame_predicate: Optional parameter-dependent override of
            ``needs_frame`` for bindings that serve more than one declared
            entry (the coverage gate runs frameless at preflight).
        legacy_name: The UK-flavored name the legacy evaluator mints, when
            it differs from ``name``; the verdict is re-minted under the
            declared name. Any other unexpected name passes through so the
            executor's name check fails it closed.
        evidence: Optional hook producing the entry's ``evidence_sha256``
            payload; ``None`` records no evidence hash.
    """

    name: str
    evaluator: Callable[[EvidenceContext, Mapping[str, Any]], GateResult]
    parameter_keys: frozenset[str] = frozenset()
    artifact_keys: frozenset[str] = frozenset()
    needs_frame: bool = True
    frame_predicate: Callable[[Mapping[str, Any]], bool] | None = None
    legacy_name: str | None = None
    evidence: Callable[[EvidenceContext, Mapping[str, Any]], object] | None = None

    def required_artifacts(self, parameters: Mapping[str, Any]) -> frozenset[str]:
        return self.artifact_keys

    def requires_frame(self, parameters: Mapping[str, Any]) -> bool:
        if self.frame_predicate is not None:
            return self.frame_predicate(parameters)
        return self.needs_frame

    def evaluate(
        self, context: EvidenceContext, parameters: Mapping[str, Any]
    ) -> GateResult:
        result = self.evaluator(context, parameters)
        if self.legacy_name is not None and result.name == self.legacy_name:
            result = replace(result, name=self.name)
        return result

    def evidence_payload(
        self, context: EvidenceContext, parameters: Mapping[str, Any]
    ) -> object | None:
        if self.evidence is None:
            return None
        return self.evidence(context, parameters)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def _evaluate_release_input_coverage(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    engine = context.artifacts["coverage_engine"]
    manifest = context.artifacts.get("coverage_manifest")
    check = parameters.get("check")
    if check == "manifest_current":
        assert_uk_release_input_coverage_manifest_current(
            engine=engine, manifest=manifest
        )
        return GateResult(
            name="release_input_coverage",
            passed=True,
            details={"check": "manifest_current"},
        )
    if check is not None:
        raise ValueError(
            f"unknown release_input_coverage check {check!r}; the declared "
            "preflight mode is 'manifest_current'."
        )
    return uk_release_input_coverage_gate(
        _uk_gate_surface(context.frame), engine, manifest=manifest
    )


def _coverage_requires_frame(parameters: Mapping[str, Any]) -> bool:
    return parameters.get("check") != "manifest_current"


def _evaluate_source_coverage(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    stage_names = tuple(str(name) for name in context.artifacts["build_stage_names"])
    assert_uk_release_input_coverage_build_stages(
        stage_names, manifest=context.artifacts.get("coverage_manifest")
    )
    return GateResult(
        name="source_coverage",
        passed=True,
        details={"stage_names": stage_names},
    )


def _evaluate_nonnegative_columns(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    if parameters:
        raise ValueError(
            "uk_nonnegative_columns takes no parameters; the checked columns "
            "come from source_stages.json nonnegative_outputs."
        )
    # The required set is every nonnegative output declared by a stage the
    # build actually scheduled — absence of a scheduled stage's declared
    # column is a failure (the shared gate's missing-column path), while
    # unscheduled stages' columns are not demanded. Never pre-filter to
    # present columns: that would let a frame missing every declared output
    # pass with columns_required=0.
    stage_names = tuple(context.artifacts["build_stage_names"])
    required = tuple(
        dict.fromkeys(
            column
            for stage in stage_names
            for column in UK_NONNEGATIVE_OUTPUTS_BY_STAGE.get(str(stage), ())
        )
    )
    column_values: dict[str, Any] = {}
    for entity in context.frame.entities:
        table = context.frame.table(entity)
        for column in table.columns:
            column_values.setdefault(str(column), table[column])
    return nonnegative_columns_gate(
        column_values,
        required,
    )


def _stage_names_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    return {
        "stage_names": [str(name) for name in context.artifacts["build_stage_names"]]
    }


def _exclusion_clock(context: EvidenceContext) -> date:
    """The one expiry clock every exclusion-consuming gate shares.

    Exclusion receipts carry approval and expiry dates, and the release
    contract requires every gate in one report to evaluate them on the same
    date. A per-gate default could straddle midnight, so the clock is a
    required artifact and anything but a plain ``date`` is refused.
    """

    clock = context.artifacts["exclusions_evaluated_on"]
    if isinstance(clock, datetime) or not isinstance(clock, date):
        raise ValueError(
            "exclusions_evaluated_on must be a datetime.date, got "
            f"{type(clock).__name__}; expiry must be evaluated on one "
            "shared clock."
        )
    return clock


def _resolve_degenerate_exclusions(
    context: EvidenceContext,
) -> tuple[Mapping[str, UKReviewedExclusion], str]:
    """The exclusion records the degenerate gate runs, and their source.

    The committed register is the reviewed policy of record (#630/#610);
    a supplied ``reviewed_degenerate_exclusions`` artifact is the loud
    review-time override — the evidence hook digests whichever resolved,
    so an overridden run self-describes in the signed report. The label
    follows the *content*, not the artifact's presence: records identical
    to the committed register are the committed policy whichever route
    delivered them, so an override cannot masquerade as a deviation (or a
    caller re-supplying the register as a false one).
    """

    committed = uk_default_degenerate_reviewed_exclusions()
    override = context.artifacts.get("reviewed_degenerate_exclusions")
    if override is None:
        return committed, "committed"
    resolved = coerce_reviewed_exclusions(
        override, label="UK degenerate-surface policy"
    )
    committed_payload = {
        name: record.policy_payload() for name, record in committed.items()
    }
    resolved_payload = {
        name: record.policy_payload() for name, record in resolved.items()
    }
    if resolved_payload == committed_payload:
        return committed, "committed"
    return resolved, "override"


def _evaluate_degenerate_release_surface(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    kwargs = dict(parameters)
    register = kwargs.pop("reviewed_exclusions_resource", None)
    if register != UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE:
        raise ValueError(
            f"uk/gates.json names exclusion register {register!r} but the "
            f"runtime loads {UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE!r}."
        )
    resolved, _source = _resolve_degenerate_exclusions(context)
    return uk_degenerate_release_surface_gate(
        _uk_gate_surface(context.frame),
        reviewed_exclusions=resolved,
        now=_exclusion_clock(context),
        **kwargs,
    )


def _degenerate_exclusions_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    resolved, source = _resolve_degenerate_exclusions(context)
    # ``exclusions_policy`` answers "which register content governed this
    # run" — deliberately distinct from the build record's
    # ``degenerate_exclusions_override_supplied``, which answers "did the
    # operator invoke the override path". The two can honestly disagree
    # (a review file byte-identical to the committed register), so they
    # carry different names.
    return {
        "exclusions_policy": source,
        "reviewed_exclusions": {
            name: record.policy_payload() for name, record in sorted(resolved.items())
        },
    }


def _evaluate_zero_weight_strata(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    kwargs = dict(parameters)
    declared = kwargs.pop("declarations", None)
    if declared is None:
        raise ValueError(
            "zero_weight_strata requires declared strata; an undeclared "
            "zero-weight surface is not a reviewed one."
        )
    declaration_keys = {"name", "selector", "maximum_zero_weight_rows", "reason"}
    for entry in declared:
        unknown = sorted(set(entry) - declaration_keys)
        if unknown:
            raise ValueError(
                f"zero_weight_strata declaration has unknown keys {unknown}; "
                f"allowed: {sorted(declaration_keys)}. A silently ignored key "
                "would sit in the policy hash while binding nothing."
            )
    declarations = tuple(
        UKZeroWeightStratumDeclaration(
            name=entry["name"],
            selector=entry["selector"],
            maximum_zero_weight_rows=entry["maximum_zero_weight_rows"],
            reason=entry["reason"],
        )
        for entry in declared
    )
    return uk_zero_weight_strata_gate(
        _uk_gate_surface(context.frame).household,
        declarations=declarations,
        **kwargs,
    )


def _evaluate_weight_ess(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    weights = _household_weights(_uk_gate_surface(context.frame).household)
    return uk_weight_ess_gate(weights, **dict(parameters))


def _evaluate_weight_ratio(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    weights = _household_weights(_uk_gate_surface(context.frame).household)
    return uk_weight_ratio_gate(weights, **dict(parameters))


def _evaluate_weights_audit(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    # The shared binding would pass a present-but-empty record tuple to the
    # gate, which audits it vacuously; the legacy battery's guard is that a
    # fit stage which emitted nothing is a failed audit, not a passing one.
    records = tuple(context.artifacts["fit_weight_records"])
    if not records:
        return _missing_fit_weight_evidence_gate()
    return weights_audit_gate(records, **dict(parameters))


def _evaluate_export_surface(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    parity = context.artifacts["parity_evidence"]
    return uk_export_surface_gate(
        parity.candidate_columns,
        parity.reference_columns,
        **dict(parameters),
    )


def _evaluate_target_surface(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    parity = context.artifacts["parity_evidence"]
    return uk_target_surface_gate(
        parity.candidate_targets,
        parity.reference_targets,
        **dict(parameters),
    )


def _evaluate_target_fit(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    parity = context.artifacts["parity_evidence"]
    return uk_target_fit_gate(parity.target_relative_errors, **dict(parameters))


def _evaluate_input_mass_parity(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    kwargs = dict(parameters)
    declared_sha256 = kwargs.pop("reference_sha256", None)
    # Inert at runtime: the declared identity is held equal to the module
    # constant by the spec-pin tests, and the canonical digest covers the
    # reference's actual identity + totals.
    kwargs.pop("reference_identity", None)
    register = kwargs.pop("reviewed_exclusions_resource", None)
    if declared_sha256 != UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256:
        raise ValueError(
            "uk/gates.json declares input-mass reference digest "
            f"{declared_sha256!r} but the runtime enforces "
            f"{UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256!r}; the declared pin "
            "and the enforced pin must move together."
        )
    if register != UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE:
        raise ValueError(
            f"uk/gates.json names exclusion register {register!r} but the "
            f"runtime loads {UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE!r}."
        )
    return uk_input_mass_parity_gate(
        uk_input_mass_totals(context.frame),
        context.artifacts["input_mass_reference"],
        policy=context.artifacts["input_mass_policy"],
        now=_exclusion_clock(context),
        **kwargs,
    )


def _input_mass_reference_evidence(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> object:
    reference = context.artifacts["input_mass_reference"]
    return {
        "reference_evidence_sha256": _input_mass_reference_evidence_sha256(reference)
    }


def _evaluate_tail_concentration(
    context: EvidenceContext, parameters: Mapping[str, Any]
) -> GateResult:
    kwargs = dict(parameters)
    register = kwargs.pop("reviewed_exclusions_resource", None)
    if register != UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE:
        raise ValueError(
            f"uk/gates.json names exclusion register {register!r} but the "
            f"runtime loads {UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE!r}."
        )
    values, weights, surface = uk_qrf_tail_concentration_columns(context.frame)
    return uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=context.artifacts["qrf_tail_policy"],
        surface=surface,
        now=_exclusion_clock(context),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

#: The UK consumer registry: the shared defaults plus a binding for every
#: gate ``uk/gates.json`` declares. ``weights_audit`` overrides the shared
#: binding to keep the legacy guard the shared one lacks — a fit stage
#: that emitted an empty record set is a failed audit, not a vacuous pass.
#: ``input_mass_parity`` and ``tail_concentration`` override the generic
#: data-in-artifacts bindings with UK ones that derive candidate evidence
#: from the frame and hold the supplied reference to the spec-declared pin
#: (the microcosm#327 rule: a parity gate's reference is a declared input,
#: never implicit module state).
UK_GATE_REGISTRY: Mapping[str, GateBinding] = {
    **DEFAULT_REGISTRY,
    "weights_audit": UKGateBinding(
        name="weights_audit",
        evaluator=_evaluate_weights_audit,
        parameter_keys=frozenset({"allowed_unweighted"}),
        artifact_keys=frozenset({"fit_weight_records"}),
        needs_frame=False,
    ),
    "release_input_coverage": UKGateBinding(
        name="release_input_coverage",
        evaluator=_evaluate_release_input_coverage,
        parameter_keys=frozenset({"check"}),
        artifact_keys=frozenset({"coverage_engine"}),
        frame_predicate=_coverage_requires_frame,
        legacy_name="uk_release_input_coverage",
    ),
    "source_coverage": UKGateBinding(
        name="source_coverage",
        evaluator=_evaluate_source_coverage,
        artifact_keys=frozenset({"build_stage_names"}),
        needs_frame=False,
        evidence=_stage_names_evidence,
    ),
    "nonnegative_columns": UKGateBinding(
        name="nonnegative_columns",
        evaluator=_evaluate_nonnegative_columns,
        artifact_keys=frozenset({"build_stage_names"}),
    ),
    "degenerate_release_surface": UKGateBinding(
        name="degenerate_release_surface",
        evaluator=_evaluate_degenerate_release_surface,
        parameter_keys=frozenset({"reviewed_exclusions_resource"}),
        artifact_keys=frozenset({"exclusions_evaluated_on"}),
        evidence=_degenerate_exclusions_evidence,
    ),
    "zero_weight_strata": UKGateBinding(
        name="zero_weight_strata",
        evaluator=_evaluate_zero_weight_strata,
        parameter_keys=frozenset({"declarations"}),
    ),
    "weight_ess": UKGateBinding(
        name="weight_ess",
        evaluator=_evaluate_weight_ess,
        parameter_keys=frozenset({"minimum_ess_fraction"}),
    ),
    "weight_ratio": UKGateBinding(
        name="weight_ratio",
        evaluator=_evaluate_weight_ratio,
        parameter_keys=frozenset({"maximum_max_to_median_ratio"}),
    ),
    "export_surface": UKGateBinding(
        name="export_surface",
        evaluator=_evaluate_export_surface,
        parameter_keys=frozenset({"allowed_extra_columns", "reviewed_exclusions"}),
        artifact_keys=frozenset({"parity_evidence"}),
        needs_frame=False,
    ),
    "target_surface": UKGateBinding(
        name="target_surface",
        evaluator=_evaluate_target_surface,
        artifact_keys=frozenset({"parity_evidence"}),
        needs_frame=False,
    ),
    "target_fit": UKGateBinding(
        name="target_fit",
        evaluator=_evaluate_target_fit,
        parameter_keys=frozenset({"max_abs_relative_error", "reviewed_exclusions"}),
        artifact_keys=frozenset({"parity_evidence"}),
        needs_frame=False,
    ),
    "input_mass_parity": UKGateBinding(
        name="input_mass_parity",
        evaluator=_evaluate_input_mass_parity,
        parameter_keys=frozenset(
            {
                "reference_sha256",
                "reference_identity",
                "reviewed_exclusions_resource",
                "candidate_name",
            }
        ),
        artifact_keys=frozenset(
            {"input_mass_reference", "input_mass_policy", "exclusions_evaluated_on"}
        ),
        evidence=_input_mass_reference_evidence,
    ),
    "tail_concentration": UKGateBinding(
        name="tail_concentration",
        evaluator=_evaluate_tail_concentration,
        parameter_keys=frozenset({"reviewed_exclusions_resource"}),
        artifact_keys=frozenset({"qrf_tail_policy", "exclusions_evaluated_on"}),
        legacy_name="qrf_tail_concentration",
    ),
}
