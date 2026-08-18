"""Aggregate-only HMRC replay classification and reporting.

The published HMRC surface contains 208 facts: eight income components, two
measures, and thirteen total-income bands.  A fact is comparable only when both
its component measure and its band assignment are source-faithful.  The
reviewed 2023-24 FRS audit cannot reconstruct the complete HMRC total-income
measure, so the current contract classifies every published fact as a fenced
exclusion.  In particular, a component-level subset does not create a safe
directional fact inside non-overlapping bands: omitted income can move a person
between bands.

This module deliberately contains no donor reader, calibration, simulation, or
row-level serialization.  Callers supply JSON-safe aggregate evidence from
those stages.  The report model nevertheless supports future exact and
directional outcomes, with invariants that prevent an excluded fact from
carrying a biased estimate or delta.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from microcosm.build.uk_runtime.hmrc_income import (
    HMRC_SPI_BUILD_PERIOD,
    HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    HMRC_SPI_INCOME_COMPONENTS,
    HMRC_SPI_TARGET_RECORD_COUNT,
    HMRCIncomeBandTargetRecord,
    HMRCIncomeTargetSet,
)

__all__ = [
    "CANONICAL_HMRC_FACT_FENCES",
    "FULL_FRS_TI_BAND_FENCE_ID",
    "HMRCFactFence",
    "HMRCReplayDiagnosticAggregate",
    "HMRCReplayFact",
    "HMRCReplayReport",
    "build_conservative_hmrc_replay_report",
    "classify_hmrc_replay_targets",
    "write_hmrc_replay_report",
]

HMRCFactClassification = Literal["exact", "directional", "excluded"]
HMRCFactOutcome = Literal[
    "exact_pass",
    "exact_fail",
    "directional_pass",
    "directional_fail",
    "excluded_with_fence",
]

_CLASSIFICATION_OUTCOMES: dict[str, frozenset[str]] = {
    "exact": frozenset(("exact_pass", "exact_fail")),
    "directional": frozenset(("directional_pass", "directional_fail")),
    "excluded": frozenset(("excluded_with_fence",)),
}
_SUMMARY_OUTCOMES = (
    "exact_pass",
    "exact_fail",
    "directional_pass",
    "directional_fail",
    "excluded_with_fence",
)
_UPPER_BOUND_BY_LOWER = dict(
    zip(
        HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
        (*HMRC_SPI_INCOME_BAND_LOWER_BOUNDS[1:], None),
        strict=True,
    )
)
_UNIT_BY_MEASURE = {"count": "people", "amount": "GBP"}
_FORBIDDEN_ROW_DATA_KEYS = frozenset(
    {
        "person_id",
        "person_ids",
        "person_household_id",
        "person_benunit_id",
        "household_id",
        "household_ids",
        "benunit_id",
        "benunit_ids",
        "row_data",
        "sample_rows",
        "records",
        "draws",
        "predictions",
    }
)


def _require_nonempty_string(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")


def _require_string_tuple(
    values: tuple[str, ...],
    label: str,
    *,
    require_nonempty: bool = False,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple of strings.")
    if require_nonempty and not values:
        raise ValueError(f"{label} must not be empty.")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{label} must contain only non-empty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates.")


def _require_finite_positive(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric.")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{label} must be finite and strictly positive.")


@dataclass(frozen=True)
class HMRCFactFence:
    """Chesterton's-fence evidence for one unavailable or partial concept."""

    fence_id: str
    constituents: tuple[str, ...]
    raw_sources_searched: tuple[str, ...]
    finding: str
    mass_implication: str
    rationale: str
    dependent_fence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.fence_id, "fence_id")
        _require_string_tuple(self.constituents, "constituents", require_nonempty=True)
        _require_string_tuple(self.raw_sources_searched, "raw_sources_searched")
        _require_nonempty_string(self.finding, f"{self.fence_id}.finding")
        _require_nonempty_string(
            self.mass_implication,
            f"{self.fence_id}.mass_implication",
        )
        _require_nonempty_string(self.rationale, f"{self.fence_id}.rationale")
        _require_string_tuple(
            self.dependent_fence_ids,
            f"{self.fence_id}.dependent_fence_ids",
        )
        if self.fence_id in self.dependent_fence_ids:
            raise ValueError(f"Fence {self.fence_id!r} cannot depend on itself.")

    def to_payload(self) -> dict[str, object]:
        return {
            "constituents": list(self.constituents),
            "raw_sources_searched": list(self.raw_sources_searched),
            "finding": self.finding,
            "mass_implication": self.mass_implication,
            "rationale": self.rationale,
            "dependent_fence_ids": list(self.dependent_fence_ids),
        }


_EPB_FENCE_ID = "frs_epb_source_absent"
_EXPS_FENCE_ID = "frs_exps_source_absent"
_TAXTERM_FENCE_ID = "frs_taxterm_source_absent"
_MOTHINC_FENCE_ID = "frs_mothinc_source_absent"
_OTHERINC_FENCE_ID = "frs_otherinc_source_absent"
_OSSBEN_SUBSET_FENCE_ID = "frs_ossben_identifiable_subset"
_SRP_SUBSET_FENCE_ID = "frs_srp_regular_code5_subset"
FULL_FRS_TI_BAND_FENCE_ID = "full_frs_tei_band_unavailable"

_CONSTITUENT_FENCE_IDS = (
    _EPB_FENCE_ID,
    _EXPS_FENCE_ID,
    _TAXTERM_FENCE_ID,
    _MOTHINC_FENCE_ID,
    _OTHERINC_FENCE_ID,
    _OSSBEN_SUBSET_FENCE_ID,
    _SRP_SUBSET_FENCE_ID,
)

CANONICAL_HMRC_FACT_FENCES = (
    HMRCFactFence(
        fence_id=_EPB_FENCE_ID,
        constituents=("EPB",),
        raw_sources_searched=(
            "JOB.EXPBEN01-EXPBEN13",
            "JOB.CARVAL",
            "JOB.CARAMT",
            "JOB.FUELAMT",
            "JOB.VCHAMT",
            "JOB.CHVAMT",
        ),
        finding=(
            "Missing. EXPBEN* are receipt flags, and the amount fields cover only "
            "selected benefits; they cannot produce complete taxable expenses "
            "payments and benefits."
        ),
        mass_implication=(
            "12.9485464% of certified-candidate FRS effective person mass has at "
            "least one receipt flag, but this is not monetary support."
        ),
        rationale=(
            "Receipt flags and selected benefit amounts cannot be promoted to the "
            "SPI EPB monetary concept without an imputation or proxy."
        ),
    ),
    HMRCFactFence(
        fence_id=_EXPS_FENCE_ID,
        constituents=("EXPS",),
        raw_sources_searched=(
            "JOB.EXPBEN04/EXPBEN05",
            "JOB.MILEAMT/JOB.MOTAMT",
            "JOB.UMILEAMT/JOB.UMOTAMT",
            "JOB.DEDUC1-DEDUC9",
            "JOB.UDEDUC1-UDEDUC9",
        ),
        finding=(
            "Missing. These fields describe reimbursements or payroll deductions, "
            "not the complete tax-deductible employment-expense amount required by "
            "SPI."
        ),
        mass_implication=(
            "5.1302528% of certified-candidate FRS effective person mass has an "
            "adjacent reimbursement flag; the true EXPS mass is not estimable."
        ),
        rationale=(
            "The nearby fields do not measure the required deductible amount, and "
            "EXPS enters the employed-income identity with a negative sign."
        ),
    ),
    HMRCFactFence(
        fence_id=_TAXTERM_FENCE_ID,
        constituents=("TAXTERM",),
        raw_sources_searched=(
            "ADULT.REDAMT",
            "ADULT and JOB taxable-termination split search",
        ),
        finding=(
            "Missing. REDAMT is gross redundancy pay and has neither the taxable "
            "amount nor non-redundancy termination pay."
        ),
        mass_implication=(
            "0.3746084% of certified-candidate FRS effective person mass has "
            "positive gross redundancy pay; taxable mass is unknown."
        ),
        rationale=(
            "Gross redundancy pay cannot be relabeled as taxable termination pay."
        ),
    ),
    HMRCFactFence(
        fence_id=_MOTHINC_FENCE_ID,
        constituents=("MOTHINC",),
        raw_sources_searched=(
            "ODDJOB.OJAMT/ODDJOB.OJNOW",
            "ADULT.ALLPAY2",
            "ADULT.ROYYR2-ROYYR4",
            "JOB.OWNOTHER",
        ),
        finding=(
            "Missing. The fields are heterogeneous and belong to distinct income "
            "concepts; assigning their union to SPI miscellaneous employment "
            "income would be a proxy."
        ),
        mass_implication=(
            "Odd-job-only effective person mass is 0.1724207%; the broader "
            "unresolved miscellaneous pool is 1.4650566%."
        ),
        rationale=(
            "The FRS instrument cannot separate the SPI miscellaneous-employment "
            "concept source-faithfully."
        ),
    ),
    HMRCFactFence(
        fence_id=_OTHERINC_FENCE_ID,
        constituents=("OTHERINC",),
        raw_sources_searched=(
            "ADULT, ODDJOB, and JOB miscellaneous fields",
            "PENSION",
            "ACCOUNTS",
            "ASSETS",
            "BENEFITS",
        ),
        finding=(
            "Missing. No person-level raw FRS variable has SPI OTHERINC semantics, "
            "and the miscellaneous pool cannot be split between MOTHINC and "
            "OTHERINC from source evidence."
        ),
        mass_implication=(
            "No separable mass estimate exists; the unresolved miscellaneous pool "
            "is 1.4650566% of certified-candidate FRS effective person mass."
        ),
        rationale=(
            "A union of heterogeneous residual fields would be a new proxy, not a "
            "retained source constituent."
        ),
    ),
    HMRCFactFence(
        fence_id=_OSSBEN_SUBSET_FENCE_ID,
        constituents=("OSSBEN", "ossben_identifiable_subset"),
        raw_sources_searched=(
            "BENEFITS.BENAMT",
            "BENEFITS.BENEFIT",
            "BENEFITS.VAR2",
            "BENEFITS codes 13, 16, 6, and 30",
        ),
        finding=(
            "Incomplete. Carer's Allowance and contribution-based ESA form an "
            "identifiable subset, but code 6 mixes tax treatments and code 30 is an "
            "undifferentiated catch-all, so the complete taxable family cannot be "
            "emitted."
        ),
        mass_implication=(
            "1.8045088% of certified-candidate FRS effective person mass carries "
            "the identifiable lower-bound subset; it is not full OSSBEN support."
        ),
        rationale=(
            "The retained column must remain explicitly named as a subset and "
            "cannot satisfy the full SPI concept."
        ),
    ),
    HMRCFactFence(
        fence_id=_SRP_SUBSET_FENCE_ID,
        constituents=("SRP", "srp_regular_code5"),
        raw_sources_searched=(
            "BENEFITS.BENAMT where BENEFIT == 5",
            "BENEFITS codes 6 and 9",
        ),
        finding=(
            "Incomplete. Code 5 supplies regular State Pension, but the FRS source "
            "does not identify the full SPI combination of State Pension lump sums "
            "and widow's pension; code 6 mixes benefits and code 9 is tax-free War "
            "Widow's Pension."
        ),
        mass_implication=(
            "18.1567916% of certified-candidate FRS effective person mass carries "
            "regular code-5 State Pension; it is not complete SRP support."
        ),
        rationale=(
            "The retained column must remain explicitly named as a subset and "
            "cannot be reported as the full published state-pension measure."
        ),
    ),
    HMRCFactFence(
        fence_id=FULL_FRS_TI_BAND_FENCE_ID,
        constituents=(
            "EPB",
            "EXPS",
            "TAXTERM",
            "MOTHINC",
            "OTHERINC",
            "OSSBEN",
            "SRP",
        ),
        raw_sources_searched=(),
        finding=(
            "The complete FRS TEI measure cannot be materialized from retained "
            "source constituents, so exact HMRC total-income band assignment is "
            "unavailable on the FRS channel."
        ),
        mass_implication=(
            "Every one of the 208 published facts is banded by total income and "
            "therefore depends on this unavailable like-for-like measure."
        ),
        rationale=(
            "A component-level subset does not imply a per-band lower bound: "
            "omitted income can move a taxpayer into or out of any non-overlapping "
            "published band. Biased partial bands are not emitted as estimates."
        ),
        dependent_fence_ids=_CONSTITUENT_FENCE_IDS,
    ),
)


@dataclass(frozen=True)
class HMRCReplayFact:
    """One classified published fact, with comparison fields when admissible."""

    target_name: str
    component: str
    measure: str
    unit: str
    period: str
    total_income_lower_bound: int
    total_income_upper_bound: int | None
    published_value: float
    classification: HMRCFactClassification
    outcome: HMRCFactOutcome
    estimate: float | None = None
    delta: float | None = None
    relative_delta: float | None = None
    operator: str | None = None
    comparison_limit: float | None = None
    fence_ids: tuple[str, ...] = ()
    blocked_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.target_name, "target_name")
        _require_nonempty_string(self.component, f"{self.target_name}.component")
        _require_nonempty_string(self.measure, f"{self.target_name}.measure")
        _require_nonempty_string(self.unit, f"{self.target_name}.unit")
        _require_nonempty_string(self.period, f"{self.target_name}.period")
        if isinstance(self.total_income_lower_bound, bool) or not isinstance(
            self.total_income_lower_bound,
            int,
        ):
            raise TypeError("total_income_lower_bound must be an integer.")
        if self.total_income_upper_bound is not None and (
            isinstance(self.total_income_upper_bound, bool)
            or not isinstance(self.total_income_upper_bound, int)
            or self.total_income_upper_bound <= self.total_income_lower_bound
        ):
            raise ValueError(
                "total_income_upper_bound must be None or an integer above the "
                "lower bound."
            )
        _require_finite_positive(self.published_value, "published_value")
        if self.classification not in _CLASSIFICATION_OUTCOMES:
            raise ValueError(
                f"Unknown HMRC fact classification {self.classification!r}."
            )
        if self.outcome not in _CLASSIFICATION_OUTCOMES[self.classification]:
            raise ValueError(
                f"Outcome {self.outcome!r} is invalid for classification "
                f"{self.classification!r}."
            )
        _require_string_tuple(self.fence_ids, f"{self.target_name}.fence_ids")
        _require_string_tuple(
            self.blocked_dependencies,
            f"{self.target_name}.blocked_dependencies",
        )

        comparison = (self.estimate, self.delta, self.relative_delta)
        if self.classification == "excluded":
            if (
                any(value is not None for value in comparison)
                or self.operator is not None
                or self.comparison_limit is not None
            ):
                raise ValueError(
                    "Excluded HMRC facts must not carry an estimate, delta, "
                    "relative_delta, comparison operator, or comparison limit."
                )
            if not self.fence_ids or not self.blocked_dependencies:
                raise ValueError(
                    "Excluded HMRC facts require both a fence and a blocked dependency."
                )
            return

        if any(value is None for value in comparison):
            raise ValueError(
                "Exact and directional HMRC facts require estimate, delta, and "
                "relative_delta values."
            )
        if not isinstance(self.operator, str) or not self.operator.strip():
            raise ValueError(
                "Exact and directional HMRC facts require a comparison operator."
            )
        estimate, delta, relative_delta = (float(value) for value in comparison)
        if not all(math.isfinite(value) for value in (estimate, delta, relative_delta)):
            raise ValueError("HMRC fact comparison values must be finite.")
        if estimate < 0.0:
            raise ValueError("HMRC fact estimates must be non-negative.")
        expected_delta = estimate - float(self.published_value)
        expected_relative = expected_delta / float(self.published_value)
        if not math.isclose(delta, expected_delta, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("HMRC fact delta must equal estimate - published_value.")
        if not math.isclose(
            relative_delta,
            expected_relative,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "HMRC fact relative_delta must equal delta / published_value."
            )
        if self.classification == "exact" and (
            self.fence_ids or self.blocked_dependencies
        ):
            raise ValueError("Exact HMRC facts cannot carry exclusion fences.")
        if self.classification == "exact":
            if self.operator != "absolute_relative_error_less_than_or_equal":
                raise ValueError(
                    "Exact HMRC facts must use "
                    "operator='absolute_relative_error_less_than_or_equal'."
                )
            if (
                self.comparison_limit is None
                or not math.isfinite(float(self.comparison_limit))
                or float(self.comparison_limit) < 0.0
            ):
                raise ValueError(
                    "Exact HMRC facts require a finite, non-negative comparison limit."
                )
            expected_outcome = (
                "exact_pass"
                if abs(relative_delta) <= float(self.comparison_limit)
                else "exact_fail"
            )
            if self.outcome != expected_outcome:
                raise ValueError(
                    "Exact HMRC fact outcome disagrees with its relative error "
                    "and comparison limit."
                )
        if self.classification == "directional":
            if self.operator != "less_than_or_equal":
                raise ValueError(
                    "Directional HMRC facts must use operator='less_than_or_equal'."
                )
            if not self.fence_ids:
                raise ValueError(
                    "Directional HMRC facts require a fence establishing the bound."
                )
            if self.comparison_limit is not None:
                raise ValueError(
                    "Directional HMRC facts do not use a comparison limit."
                )
            expected_outcome = (
                "directional_pass"
                if estimate <= float(self.published_value)
                else "directional_fail"
            )
            if self.outcome != expected_outcome:
                raise ValueError(
                    "Directional HMRC fact outcome disagrees with its "
                    "less-than-or-equal assertion."
                )

    @classmethod
    def excluded_from_target(
        cls,
        target: HMRCIncomeBandTargetRecord,
        *,
        fence_ids: tuple[str, ...],
        blocked_dependencies: tuple[str, ...],
    ) -> HMRCReplayFact:
        return cls(
            target_name=target.name,
            component=target.component,
            measure=target.measure,
            unit=target.unit,
            period=target.period,
            total_income_lower_bound=target.total_income_lower_bound,
            total_income_upper_bound=target.total_income_upper_bound,
            published_value=float(target.value),
            classification="excluded",
            outcome="excluded_with_fence",
            fence_ids=fence_ids,
            blocked_dependencies=blocked_dependencies,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "target_name": self.target_name,
            "component": self.component,
            "measure": self.measure,
            "unit": self.unit,
            "period": self.period,
            "total_income_lower_bound": self.total_income_lower_bound,
            "total_income_upper_bound": self.total_income_upper_bound,
            "published_value": self.published_value,
            "classification": self.classification,
            "outcome": self.outcome,
            "estimate": self.estimate,
            "delta": self.delta,
            "relative_delta": self.relative_delta,
            "operator": self.operator,
            "comparison_limit": self.comparison_limit,
            "fence_ids": list(self.fence_ids),
            "blocked_dependencies": list(self.blocked_dependencies),
        }


@dataclass(frozen=True)
class HMRCReplayDiagnosticAggregate:
    """A scalar replay diagnostic that is explicitly not an HMRC fact."""

    name: str
    scope: str
    metric: str
    value: int | float
    unit: str
    non_comparability_reason: str
    comparable_to_hmrc: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, "diagnostic.name")
        _require_nonempty_string(self.scope, f"{self.name}.scope")
        _require_nonempty_string(self.metric, f"{self.name}.metric")
        _require_nonempty_string(self.unit, f"{self.name}.unit")
        _require_nonempty_string(
            self.non_comparability_reason,
            f"{self.name}.non_comparability_reason",
        )
        if self.comparable_to_hmrc is not False:
            raise ValueError(
                "Replay diagnostic aggregates must be explicitly non-comparable "
                "to the published HMRC facts."
            )
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise TypeError("Replay diagnostic aggregate values must be numeric.")
        if not math.isfinite(float(self.value)):
            raise ValueError("Replay diagnostic aggregate values must be finite.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError(f"{self.name}.metadata must be an aggregate mapping.")
        object.__setattr__(
            self,
            "metadata",
            _freeze_json(self.metadata, path=f"{self.name}.metadata"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scope": self.scope,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "comparable_to_hmrc": False,
            "non_comparability_reason": self.non_comparability_reason,
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True)
class HMRCReplayReport:
    """Validated, JSON-safe aggregate replay report."""

    facts: tuple[HMRCReplayFact, ...]
    fences: tuple[HMRCFactFence, ...]
    source_evidence: Mapping[str, object]
    build_evidence: Mapping[str, object]
    qrf_evidence: Mapping[str, object]
    effective_mass_evidence: Mapping[str, object]
    diagnostic_aggregates: tuple[HMRCReplayDiagnosticAggregate, ...] = ()
    schema_version: int = 1
    report_kind: str = "uk_hmrc_income_208_fact_replay"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("HMRC replay report schema_version must be 1.")
        _require_nonempty_string(self.report_kind, "report_kind")
        object.__setattr__(self, "facts", tuple(self.facts))
        object.__setattr__(self, "fences", tuple(self.fences))
        object.__setattr__(
            self,
            "diagnostic_aggregates",
            tuple(self.diagnostic_aggregates),
        )
        if any(not isinstance(fact, HMRCReplayFact) for fact in self.facts):
            raise TypeError("HMRC replay facts must be HMRCReplayFact instances.")
        if any(not isinstance(fence, HMRCFactFence) for fence in self.fences):
            raise TypeError("HMRC replay fences must be HMRCFactFence instances.")
        if any(
            not isinstance(item, HMRCReplayDiagnosticAggregate)
            for item in self.diagnostic_aggregates
        ):
            raise TypeError(
                "HMRC replay diagnostics must be HMRCReplayDiagnosticAggregate "
                "instances."
            )
        _validate_complete_fact_surface(self.facts)
        _validate_report_fences(self.fences, self.facts)
        diagnostic_names = [item.name for item in self.diagnostic_aggregates]
        if len(set(diagnostic_names)) != len(diagnostic_names):
            raise ValueError("HMRC replay diagnostic aggregate names must be unique.")
        for name in (
            "source_evidence",
            "build_evidence",
            "qrf_evidence",
            "effective_mass_evidence",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_evidence_mapping(getattr(self, name), label=name),
            )

    @property
    def summary(self) -> dict[str, object]:
        counts = Counter(fact.outcome for fact in self.facts)
        outcome_counts = {name: int(counts.get(name, 0)) for name in _SUMMARY_OUTCOMES}
        compared = sum(
            outcome_counts[name]
            for name in (
                "exact_pass",
                "exact_fail",
                "directional_pass",
                "directional_fail",
            )
        )
        failures = outcome_counts["exact_fail"] + outcome_counts["directional_fail"]
        exclusions = outcome_counts["excluded_with_fence"]
        if exclusions == len(self.facts):
            status = "reviewed_exclusions_only"
        elif failures:
            status = "comparisons_failed"
        elif exclusions:
            status = "comparisons_passed_with_reviewed_exclusions"
        else:
            status = "comparisons_passed"
        return {
            "status": status,
            "total_facts": len(self.facts),
            **outcome_counts,
            "comparison_coverage_count": compared,
            "comparison_coverage_share": compared / len(self.facts),
            "release_blocking_comparison_failures": failures,
            "all_facts_adjudicated": len(self.facts) == HMRC_SPI_TARGET_RECORD_COUNT,
            "all_exclusions_fenced": all(
                fact.outcome != "excluded_with_fence" or bool(fact.fence_ids)
                for fact in self.facts
            ),
        }

    def to_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "report_kind": self.report_kind,
            "summary": self.summary,
            "source_evidence": _thaw_json(self.source_evidence),
            "build_evidence": _thaw_json(self.build_evidence),
            "qrf_evidence": _thaw_json(self.qrf_evidence),
            "effective_mass_evidence": _thaw_json(self.effective_mass_evidence),
            "fences": {
                fence.fence_id: fence.to_payload()
                for fence in sorted(self.fences, key=lambda item: item.fence_id)
            },
            "diagnostic_aggregates": [
                item.to_payload() for item in self.diagnostic_aggregates
            ],
            "facts": [fact.to_payload() for fact in self.facts],
        }
        # This is both a final JSON-safety assertion and a guard against NaN or
        # Infinity leaking from caller-supplied aggregate evidence.
        json.dumps(payload, allow_nan=False, sort_keys=True)
        return payload


def classify_hmrc_replay_targets(
    source_targets: HMRCIncomeTargetSet,
) -> tuple[HMRCReplayFact, ...]:
    """Conservatively fence the complete current 208-fact source surface."""

    if not isinstance(source_targets, HMRCIncomeTargetSet):
        raise TypeError("source_targets must be an HMRCIncomeTargetSet.")
    if source_targets.source.build_period != HMRC_SPI_BUILD_PERIOD:
        raise ValueError(
            "HMRC replay source period must match the reviewed build period "
            f"{HMRC_SPI_BUILD_PERIOD!r}."
        )
    _validate_source_target_surface(source_targets.targets)
    return tuple(
        HMRCReplayFact.excluded_from_target(
            target,
            fence_ids=(FULL_FRS_TI_BAND_FENCE_ID,),
            blocked_dependencies=("hmrc_spi_assessable_income",),
        )
        for target in source_targets.targets
    )


def build_conservative_hmrc_replay_report(
    source_targets: HMRCIncomeTargetSet,
    *,
    source_evidence: Mapping[str, object],
    build_evidence: Mapping[str, object],
    qrf_evidence: Mapping[str, object],
    effective_mass_evidence: Mapping[str, object],
    diagnostic_aggregates: Sequence[HMRCReplayDiagnosticAggregate] = (),
    report_kind: str = "uk_hmrc_income_208_fact_replay",
) -> HMRCReplayReport:
    """Build the reviewed 0 exact / 0 directional / 208 excluded report."""

    return HMRCReplayReport(
        facts=classify_hmrc_replay_targets(source_targets),
        fences=CANONICAL_HMRC_FACT_FENCES,
        source_evidence=source_evidence,
        build_evidence=build_evidence,
        qrf_evidence=qrf_evidence,
        effective_mass_evidence=effective_mass_evidence,
        diagnostic_aggregates=tuple(diagnostic_aggregates),
        report_kind=report_kind,
    )


def write_hmrc_replay_report(
    report: HMRCReplayReport,
    path: str | Path,
) -> Path:
    """Atomically write an aggregate replay report to a caller-chosen JSON path."""

    if not isinstance(report, HMRCReplayReport):
        raise TypeError("report must be an HMRCReplayReport.")
    requested = Path(path).expanduser()
    if requested.suffix.lower() != ".json":
        raise ValueError("HMRC replay report path must end with '.json'.")
    if requested.is_symlink():
        raise ValueError("HMRC replay report path must not be a symbolic link.")
    parent = requested.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / requested.name
    if output.exists() and not output.is_file():
        raise ValueError("HMRC replay report path must name a regular file.")

    payload = report.to_payload()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _validate_source_target_surface(
    targets: Sequence[HMRCIncomeBandTargetRecord],
) -> None:
    if len(targets) != HMRC_SPI_TARGET_RECORD_COUNT:
        raise ValueError(
            "HMRC replay classification requires the complete 208-fact source "
            f"surface; got {len(targets)}."
        )
    if any(not isinstance(target, HMRCIncomeBandTargetRecord) for target in targets):
        raise TypeError("HMRC replay source targets must be target records.")
    expected = {
        (lower_bound, component, measure)
        for lower_bound in HMRC_SPI_INCOME_BAND_LOWER_BOUNDS
        for component in HMRC_SPI_INCOME_COMPONENTS
        for measure in ("count", "amount")
    }
    actual = {
        (target.total_income_lower_bound, target.component, target.measure)
        for target in targets
    }
    if actual != expected:
        raise ValueError(
            "HMRC replay source surface keys are incomplete or unexpected; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )
    for target in targets:
        expected_upper = _UPPER_BOUND_BY_LOWER[target.total_income_lower_bound]
        if target.total_income_upper_bound != expected_upper:
            raise ValueError(
                f"HMRC replay source target {target.name!r} has upper bound "
                f"{target.total_income_upper_bound!r}, expected {expected_upper!r}."
            )
        if target.unit != _UNIT_BY_MEASURE[target.measure]:
            raise ValueError(
                f"HMRC replay source target {target.name!r} has unit "
                f"{target.unit!r}, expected {_UNIT_BY_MEASURE[target.measure]!r}."
            )
        if target.period != HMRC_SPI_BUILD_PERIOD:
            raise ValueError(
                f"HMRC replay source target {target.name!r} has period "
                f"{target.period!r}, expected {HMRC_SPI_BUILD_PERIOD!r}."
            )
    names = [target.name for target in targets]
    if len(set(names)) != len(names):
        raise ValueError("HMRC replay source target names must be unique.")


def _validate_complete_fact_surface(facts: tuple[HMRCReplayFact, ...]) -> None:
    if len(facts) != HMRC_SPI_TARGET_RECORD_COUNT:
        raise ValueError(
            "HMRC replay report must contain exactly "
            f"{HMRC_SPI_TARGET_RECORD_COUNT} facts; got {len(facts)}."
        )
    expected = {
        (lower_bound, component, measure)
        for lower_bound in HMRC_SPI_INCOME_BAND_LOWER_BOUNDS
        for component in HMRC_SPI_INCOME_COMPONENTS
        for measure in ("count", "amount")
    }
    actual = {
        (fact.total_income_lower_bound, fact.component, fact.measure) for fact in facts
    }
    if actual != expected:
        raise ValueError(
            "HMRC replay report fact keys are incomplete or unexpected; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )
    for fact in facts:
        expected_upper = _UPPER_BOUND_BY_LOWER[fact.total_income_lower_bound]
        if fact.total_income_upper_bound != expected_upper:
            raise ValueError(
                f"HMRC replay fact {fact.target_name!r} has upper bound "
                f"{fact.total_income_upper_bound!r}, expected {expected_upper!r}."
            )
        if fact.unit != _UNIT_BY_MEASURE[fact.measure]:
            raise ValueError(
                f"HMRC replay fact {fact.target_name!r} has unit {fact.unit!r}, "
                f"expected {_UNIT_BY_MEASURE[fact.measure]!r}."
            )
        if fact.period != HMRC_SPI_BUILD_PERIOD:
            raise ValueError(
                f"HMRC replay fact {fact.target_name!r} has period "
                f"{fact.period!r}, expected {HMRC_SPI_BUILD_PERIOD!r}."
            )
    names = [fact.target_name for fact in facts]
    if len(set(names)) != len(names):
        raise ValueError("HMRC replay report target names must be unique.")


def _validate_report_fences(
    fences: tuple[HMRCFactFence, ...],
    facts: tuple[HMRCReplayFact, ...],
) -> None:
    by_id = {fence.fence_id: fence for fence in fences}
    if len(by_id) != len(fences):
        raise ValueError("HMRC replay fence IDs must be unique.")
    expected_ids = {fence.fence_id for fence in CANONICAL_HMRC_FACT_FENCES}
    if set(by_id) != expected_ids:
        raise ValueError(
            "HMRC replay report must carry the complete canonical fence set; "
            f"missing={sorted(expected_ids - set(by_id))}, "
            f"unexpected={sorted(set(by_id) - expected_ids)}."
        )
    canonical_by_id = {fence.fence_id: fence for fence in CANONICAL_HMRC_FACT_FENCES}
    drifted = sorted(
        fence_id
        for fence_id, fence in by_id.items()
        if fence != canonical_by_id[fence_id]
    )
    if drifted:
        raise ValueError(
            "HMRC replay report fence evidence differs from the canonical "
            f"reviewed findings: {drifted}."
        )
    for fence in fences:
        missing_dependencies = sorted(set(fence.dependent_fence_ids) - set(by_id))
        if missing_dependencies:
            raise ValueError(
                f"Fence {fence.fence_id!r} references missing dependent fence(s) "
                f"{missing_dependencies}."
            )
    missing_fact_fences = sorted(
        {
            fence_id
            for fact in facts
            for fence_id in fact.fence_ids
            if fence_id not in by_id
        }
    )
    if missing_fact_fences:
        raise ValueError(
            f"HMRC replay facts reference missing fence(s) {missing_fact_fences}."
        )


def _freeze_evidence_mapping(
    value: Mapping[str, object],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty aggregate evidence mapping.")
    frozen = _freeze_json(value, path=label)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
        raise TypeError(f"{label} must be a mapping.")
    return frozen


def _freeze_json(value: object, *, path: str) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise TypeError(f"{path} keys must be non-empty strings.")
            if key.casefold() in _FORBIDDEN_ROW_DATA_KEYS:
                raise ValueError(
                    f"{path}.{key} is row-level data and cannot enter an aggregate "
                    "HMRC replay report."
                )
            result[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        if any(isinstance(item, Mapping) for item in value):
            raise ValueError(
                f"{path} contains a sequence of records; aggregate evidence must "
                "use named scalar mappings instead of row-like records."
            )
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or Infinity.")
        return value
    raise TypeError(
        f"{path} contains non-JSON aggregate evidence of type {type(value).__name__}."
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
