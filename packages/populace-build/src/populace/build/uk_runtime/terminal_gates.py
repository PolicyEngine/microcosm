"""Batched terminal acceptance gates for UK release candidates.

The national builder evaluates this battery once, after every source stage and
immediately before writing its staging H5.  Every evaluator runs even when an
earlier evaluator fails, so one expensive build produces one complete named
failure report.  Evidence that does not exist yet is omitted: in particular,
future weighted-integrity and delivered-take-up gates are not represented by
placeholder passes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import (
    FitWeightRecord,
    GateReport,
    GateResult,
    export_surface_gate,
    weights_audit_gate,
)
from populace.build.gates import (
    target_surface_gate as _target_surface_gate,
)
from populace.build.uk_runtime.diagnostics import uk_weight_summary
from populace.build.uk_runtime.release_input_coverage import (
    uk_release_input_coverage_gate,
)

__all__ = [
    "UK_ALLOWED_EXTRA_EXPORT_COLUMNS",
    "UK_CANDIDATE_DATASET_NAME",
    "UK_DEFAULT_ZERO_WEIGHT_STRATA",
    "UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS",
    "UK_MAX_TARGET_ABS_RELATIVE_ERROR",
    "UK_MAX_TO_MEDIAN_WEIGHT_RATIO",
    "UK_MIN_ESS_FRACTION",
    "UK_TERMINAL_GATE_ATTESTATION_SCHEMA_VERSION",
    "UK_TERMINAL_GATE_POLICY_SHA256",
    "UK_TERMINAL_GATE_PRODUCER",
    "UK_TERMINAL_GATE_SIGNATURE_ALGORITHM",
    "UK_TERMINAL_GATE_SIGNING_KEY_ENV",
    "UK_REFERENCE_DATASET_NAME",
    "UK_REVIEWED_EXPORT_EXCLUSIONS",
    "UK_TERMINAL_GATE_SCHEMA_VERSION",
    "UKReleaseParityEvidence",
    "UKZeroWeightStratumDeclaration",
    "uk_degenerate_release_surface_gate",
    "uk_export_surface_gate",
    "uk_target_fit_gate",
    "uk_target_surface_gate",
    "uk_terminal_gate_report",
    "uk_weight_ess_gate",
    "uk_weight_ratio_gate",
    "uk_zero_weight_strata_gate",
    "write_uk_terminal_gate_report",
]

UK_TERMINAL_GATE_SCHEMA_VERSION = 2
UK_TERMINAL_GATE_ATTESTATION_SCHEMA_VERSION = 3
UK_TERMINAL_GATE_PRODUCER = (
    "populace.build.uk_runtime.terminal_gates.uk_terminal_gate_report"
)
UK_TERMINAL_GATE_SIGNATURE_ALGORITHM = "hmac-sha256"
UK_TERMINAL_GATE_SIGNING_KEY_ENV = "POPULACE_UK_TERMINAL_GATE_SIGNING_KEY"
UK_CANDIDATE_DATASET_NAME = "populace_uk_2023"
UK_REFERENCE_DATASET_NAME = "enhanced_frs_2023_24_recalibrated"
UK_MAX_TARGET_ABS_RELATIVE_ERROR = 0.25

# The certified June artifact is at 0.039953 ESS fraction.  Its measured
# max/positive-median ratio is exactly 1,151.2542195939373 (maximum weight
# 18,652.802734375 / positive median 16.202157974243164).  Under the #578
# acceptance rule, a candidate must not regress the incumbent on battery
# observables: this boundary has no discretionary headroom.  Raising it
# requires an explicit future adjudication.  These are acceptance fences, not
# solver knobs.
UK_MIN_ESS_FRACTION = 0.01
UK_MAX_TO_MEDIAN_WEIGHT_RATIO = 1_151.2542195939373

_UK_ALWAYS_APPLICABLE_GATE_NAMES = (
    "uk_release_input_coverage",
    "degenerate_release_surface",
    "zero_weight_strata",
    "weight_ess",
    "weight_ratio",
)
_UK_HMRC_GATE_NAMES = ("weights_audit",)
_UK_PARITY_GATE_NAMES = ("export_surface", "target_surface", "target_fit")
_UK_WEIGHT_SUMMARY_FIELDS = (
    "n_records",
    "positive_weight_records",
    "zero_weight_records",
    "total_weight",
    "effective_sample_size",
    "ess_fraction",
    "median_positive_weight",
    "max_weight",
    "max_to_median_positive_weight",
    "top_1pct_weight_share",
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _terminal_gate_signing_key() -> bytes:
    """Read the 256-bit release-attestation key from the build environment."""

    encoded = os.environ.get(UK_TERMINAL_GATE_SIGNING_KEY_ENV)
    if not encoded:
        raise RuntimeError(
            f"{UK_TERMINAL_GATE_SIGNING_KEY_ENV} must contain a base64-encoded "
            "32-byte key before writing a UK terminal gate report."
        )
    try:
        key = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(
            f"{UK_TERMINAL_GATE_SIGNING_KEY_ENV} must be valid base64."
        ) from exc
    if len(key) != 32:
        raise RuntimeError(
            f"{UK_TERMINAL_GATE_SIGNING_KEY_ENV} must decode to exactly 32 bytes."
        )
    return key


def _terminal_gate_signature(key: bytes, payload: object) -> str:
    return hmac.new(key, _canonical_json_bytes(payload), hashlib.sha256).hexdigest()


_SPI_FLAG = "household_is_spi_synthetic"
_CAPITAL_GAINS_FLAG = "household_is_capital_gains_clone"
_WEIGHT_COLUMN = "household_weight"


@dataclass(frozen=True)
class UKZeroWeightStratumDeclaration:
    """One reviewed zero-weight household stratum and its maximum size."""

    name: str
    selector: Mapping[str, object]
    maximum_zero_weight_rows: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("UK zero-weight stratum name must be non-empty.")
        if not isinstance(self.selector, Mapping) or not self.selector:
            raise ValueError(
                f"UK zero-weight stratum {self.name!r} needs a non-empty selector."
            )
        normalized: dict[str, object] = {}
        for raw_column, value in self.selector.items():
            column = str(raw_column)
            if not column:
                raise ValueError(
                    f"UK zero-weight stratum {self.name!r} has an empty selector "
                    "column."
                )
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"UK zero-weight stratum {self.name!r} selector {column!r} "
                    "must be finite."
                )
            if value is not None and not isinstance(value, str | bool | int | float):
                raise TypeError(
                    f"UK zero-weight stratum {self.name!r} selector {column!r} "
                    f"has unsupported value type {type(value).__name__}."
                )
            normalized[column] = value
        maximum = self.maximum_zero_weight_rows
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            raise ValueError(
                f"UK zero-weight stratum {self.name!r} maximum must be a "
                "non-negative integer."
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                f"UK zero-weight stratum {self.name!r} needs a reviewed reason."
            )
        object.__setattr__(self, "selector", dict(sorted(normalized.items())))


UK_DEFAULT_ZERO_WEIGHT_STRATA: tuple[UKZeroWeightStratumDeclaration, ...] = (
    UKZeroWeightStratumDeclaration(
        name="june_spi_synthetic_base",
        selector={_SPI_FLAG: True, _CAPITAL_GAINS_FLAG: False},
        maximum_zero_weight_rows=100_000,
        reason=(
            "The certified June FRS-derived artifact ships 100,000 zero-weight "
            "SPI-synthetic non-capital-gains rows."
        ),
    ),
    UKZeroWeightStratumDeclaration(
        name="june_spi_synthetic_capital_gains",
        selector={_SPI_FLAG: True, _CAPITAL_GAINS_FLAG: True},
        maximum_zero_weight_rows=100_000,
        reason=(
            "The certified June FRS-derived artifact ships 100,000 zero-weight "
            "SPI-synthetic capital-gains-clone rows."
        ),
    ),
)


# Reviewed candidate-only fields from the June UK prototype.  They are source
# provenance or genuine additional model inputs, not incumbent-surface losses.
UK_ALLOWED_EXTRA_EXPORT_COLUMNS: tuple[str, ...] = (
    "benunit.child_benefit_opts_out",
    "household.bus_fare_spending",
    "household.bus_subsidy_spending",
    "household.clone_index",
    "household.constituency_code_oa",
    "household.consumer_debt",
    "household.electricity_consumption",
    "household.gas_consumption",
    "household.has_fuel_consumption",
    "household.household_is_capital_gains_clone",
    "household.household_is_spi_synthetic",
    "household.la_code_oa",
    "household.lsoa_code",
    "household.mortgage_debt",
    "household.msoa_code",
    "household.num_vehicles",
    "household.oa_code",
    "household.property_purchased",
    "household.rail_usage",
    "household.region_code_oa",
    "person.aa_category",
    "person.age_started_or_accepted_current_education_or_training",
    "person.attends_private_school_random_draw",
    "person.charitable_investment_gifts",
    "person.dla_m_category",
    "person.dla_sc_category",
    "person.esa_health_condition_proxy",
    "person.esa_support_group_proxy",
    "person.employment_sector",
    "person.gift_aid",
    "person.higher_earner_tie_break",
    "person.highest_education",
    "person.is_before_universal_credit_qualifying_young_person_terminal_date",
    "person.is_in_non_advanced_education",
    "person.is_parent",
    "person.legacy_jobseeker_proxy",
    "person.pension_contributions_via_salary_sacrifice",
    "person.pip_dl_category",
    "person.pip_m_category",
    "person.receives_benefits_in_own_right",
    "person.salary_sacrifice_asked",
    "person.salary_sacrifice_reported",
    "person.sic_industry_division",
    "person.student_loan_balance",
    "person.student_loan_plan",
    "person.would_claim_marriage_allowance",
    "person.would_claim_scp",
)

UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS: tuple[str, ...] = (
    "person.attends_private_school",
    "person.is_higher_earner",
)

UK_REVIEWED_EXPORT_EXCLUSIONS: Mapping[str, str] = {
    "person.incapacity_benefit_reported": (
        "The enhanced FRS stores this legacy reported-benefit input as an "
        "all-zero layer; the candidate must drop dead zero layers."
    ),
}

_STRUCTURAL_COLUMNS: Mapping[str, frozenset[str]] = {
    "person": frozenset({"person_id", "person_household_id", "person_benunit_id"}),
    "benunit": frozenset({"benunit_id"}),
    "household": frozenset({"household_id", _WEIGHT_COLUMN}),
}


def _policy_mapping(value: object) -> object:
    if not isinstance(value, Mapping):
        return {"invalid_type": f"{type(value).__module__}.{type(value).__qualname__}"}
    return {
        str(key): str(item)
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }


def _terminal_gate_policy_payload(
    *,
    builtin_coverage_evaluator: bool,
    reviewed_degenerate_exclusions: object,
    zero_weight_declarations: Sequence[object],
    minimum_ess_fraction: object,
    maximum_max_to_median_ratio: object,
) -> dict[str, object]:
    declarations: list[dict[str, object]] = []
    for declaration in zero_weight_declarations:
        if isinstance(declaration, UKZeroWeightStratumDeclaration):
            declarations.append(
                {
                    "name": declaration.name,
                    "selector": dict(declaration.selector),
                    "maximum_zero_weight_rows": declaration.maximum_zero_weight_rows,
                    "reason": declaration.reason,
                }
            )
        else:
            declarations.append(
                {
                    "invalid_type": (
                        f"{type(declaration).__module__}."
                        f"{type(declaration).__qualname__}"
                    )
                }
            )
    try:
        ess = float(minimum_ess_fraction)
    except (TypeError, ValueError):
        ess = repr(minimum_ess_fraction)
    try:
        ratio = float(maximum_max_to_median_ratio)
    except (TypeError, ValueError):
        ratio = repr(maximum_max_to_median_ratio)
    return {
        "coverage_evaluator": "builtin" if builtin_coverage_evaluator else "injected",
        "reviewed_degenerate_exclusions": (
            {}
            if reviewed_degenerate_exclusions is None
            else _policy_mapping(reviewed_degenerate_exclusions)
        ),
        "zero_weight_declarations": declarations,
        "minimum_ess_fraction": ess,
        "maximum_max_to_median_ratio": ratio,
        "maximum_target_abs_relative_error": UK_MAX_TARGET_ABS_RELATIVE_ERROR,
        "allowed_extra_export_columns": list(UK_ALLOWED_EXTRA_EXPORT_COLUMNS),
        "known_missing_reference_export_columns": list(
            UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS
        ),
        "reviewed_export_exclusions": dict(
            sorted(UK_REVIEWED_EXPORT_EXCLUSIONS.items())
        ),
    }


UK_TERMINAL_GATE_POLICY_SHA256 = _canonical_sha256(
    _terminal_gate_policy_payload(
        builtin_coverage_evaluator=True,
        reviewed_degenerate_exclusions=None,
        zero_weight_declarations=UK_DEFAULT_ZERO_WEIGHT_STRATA,
        minimum_ess_fraction=UK_MIN_ESS_FRACTION,
        maximum_max_to_median_ratio=UK_MAX_TO_MEDIAN_WEIGHT_RATIO,
    )
)


@dataclass(frozen=True)
class UKReleaseParityEvidence:
    """Complete real evidence needed to run the June parity-gate trio."""

    candidate_columns: Iterable[str]
    reference_columns: Iterable[str]
    candidate_targets: Iterable[str]
    reference_targets: Iterable[str]
    target_relative_errors: Mapping[str, float]

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_columns",
            "reference_columns",
            "candidate_targets",
            "reference_targets",
        ):
            raw = getattr(self, field_name)
            materialized = tuple(sorted({str(value) for value in raw}))
            if not materialized or any(not value for value in materialized):
                raise ValueError(f"UK parity evidence {field_name} must be non-empty.")
            object.__setattr__(self, field_name, materialized)
        errors = {
            str(name): float(error)
            for name, error in self.target_relative_errors.items()
        }
        if not errors or any(not math.isfinite(error) for error in errors.values()):
            raise ValueError(
                "UK parity evidence target_relative_errors must be non-empty and "
                "finite."
            )
        if set(errors) != set(self.candidate_targets):
            raise ValueError(
                "UK parity evidence target_relative_errors must exactly cover "
                "candidate_targets."
            )
        object.__setattr__(self, "target_relative_errors", dict(sorted(errors.items())))


def _gate_results_payload(results: Sequence[GateResult]) -> dict[str, object]:
    return {
        result.name: {
            "passed": result.passed,
            "failures": list(result.failures),
            "details": dict(result.details),
        }
        for result in results
    }


def _unsigned_terminal_gate_attestation(
    results: Sequence[GateResult],
    *,
    policy_sha256: str,
    evidence_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Return the provenance fields covered by the release signature."""

    gates = _gate_results_payload(results)
    return {
        "schema_version": UK_TERMINAL_GATE_ATTESTATION_SCHEMA_VERSION,
        "producer": UK_TERMINAL_GATE_PRODUCER,
        "policy_sha256": policy_sha256,
        "evaluated_gates": [result.name for result in results],
        "evidence_sha256": dict(evidence_sha256),
        "gate_results_sha256": _canonical_sha256(gates),
    }


def _terminal_gate_report_payload(
    results: Sequence[GateResult],
    attestation: Mapping[str, object],
    *,
    signature_available: bool,
) -> dict[str, object]:
    return {
        "schema_version": UK_TERMINAL_GATE_SCHEMA_VERSION,
        "enforced": True,
        "passed": all(result.passed for result in results) and signature_available,
        "gates": _gate_results_payload(results),
        "attestation": dict(attestation),
    }


def _release_dataset_evidence_payload(
    results: Sequence[GateResult],
) -> dict[str, object]:
    """Bind the attestation to the evaluated release's weight observables.

    The ratio gate receives the final shipped household-weight vector and
    records the same ten-field summary written to ``uk_diagnostics.weights``.
    Projecting that summary gives the publication contract an independently
    reconstructible release-data digest.  Erroring gates deliberately project
    missing fields as null so a failed terminal report can still be written.
    """

    ratio = next((result for result in results if result.name == "weight_ratio"), None)
    details = ratio.details if ratio is not None else {}
    return {
        "weights": {
            field: _json_scalar(details.get(field))
            for field in _UK_WEIGHT_SUMMARY_FIELDS
        }
    }


@dataclass(frozen=True)
class _AttestedUKTerminalGateReport(GateReport):
    """Aggregator-signed report sealed against post-evaluation mutation."""

    policy_sha256: str
    evidence_sha256: Mapping[str, str]
    attestation: Mapping[str, object]
    _signing_error: RuntimeError | None = field(repr=False, compare=False)
    _sealed_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = tuple(result.name for result in self.results)
        if len(names) != len(set(names)):
            raise ValueError("Attested UK terminal gate names must be unique.")
        if len(self.policy_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.policy_sha256
        ):
            raise ValueError("UK terminal policy digest must be a lowercase sha256.")
        evidence = dict(sorted(self.evidence_sha256.items()))
        if not evidence or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for name, digest in evidence.items()
        ):
            raise ValueError("UK terminal evidence digests must be named sha256s.")
        evidence_names = set(evidence)
        if "release_dataset" not in evidence_names:
            raise ValueError(
                "UK terminal evidence must include the release_dataset digest."
            )
        unknown_evidence = sorted(
            evidence_names - {"release_dataset", "hmrc_spi_income", "release_parity"}
        )
        if unknown_evidence:
            raise ValueError(
                f"UK terminal evidence has unknown stages: {unknown_evidence}."
            )
        expected_names = list(_UK_ALWAYS_APPLICABLE_GATE_NAMES)
        if "hmrc_spi_income" in evidence_names:
            expected_names.extend(_UK_HMRC_GATE_NAMES)
        if "release_parity" in evidence_names:
            expected_names.extend(_UK_PARITY_GATE_NAMES)
        if names != tuple(expected_names):
            raise ValueError(
                "UK terminal gate membership must follow the attested evidence "
                f"stages; expected {expected_names}, got {list(names)}."
            )
        object.__setattr__(self, "evidence_sha256", MappingProxyType(evidence))
        expected_unsigned = _unsigned_terminal_gate_attestation(
            self.results,
            policy_sha256=self.policy_sha256,
            evidence_sha256=evidence,
        )
        attestation = dict(self.attestation)
        expected_fields = {
            *expected_unsigned,
            "signature_algorithm",
            "signing_key_sha256",
            "signature",
        }
        if set(attestation) != expected_fields:
            raise ValueError(
                "UK terminal attestation must contain the complete signed schema."
            )
        if any(
            attestation.get(name) != value for name, value in expected_unsigned.items()
        ):
            raise ValueError(
                "UK terminal attestation must bind the evaluated gates and evidence."
            )
        if (
            attestation.get("signature_algorithm")
            != UK_TERMINAL_GATE_SIGNATURE_ALGORITHM
        ):
            raise ValueError("UK terminal attestation signature algorithm is invalid.")
        signature_values = (
            attestation.get("signing_key_sha256"),
            attestation.get("signature"),
        )
        if self._signing_error is None:
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in signature_values
            ):
                raise ValueError(
                    "Signed UK terminal attestations require lowercase sha256 values."
                )
        elif signature_values != (None, None):
            raise ValueError(
                "A failed UK terminal signature must not claim signature values."
            )
        object.__setattr__(self, "attestation", MappingProxyType(attestation))
        object.__setattr__(self, "_sealed_sha256", self._current_attestation_sha256())

    @property
    def evaluated_gates(self) -> tuple[str, ...]:
        return tuple(result.name for result in self.results)

    def _current_attestation_sha256(self) -> str:
        return _canonical_sha256(
            _terminal_gate_report_payload(
                self.results,
                self.attestation,
                signature_available=self._signing_error is None,
            )
        )

    def report_payload(self) -> dict[str, object]:
        """Return the sealed aggregator output without granting signing power."""

        if self._current_attestation_sha256() != self._sealed_sha256:
            raise ValueError("UK terminal gate attestation changed after evaluation.")
        return _terminal_gate_report_payload(
            self.results,
            self.attestation,
            signature_available=self._signing_error is None,
        )


def _fit_evidence_payload(
    records: tuple[object, ...] | None,
    *,
    required: bool,
    materialization_error: Exception | None,
) -> dict[str, object]:
    payload: dict[str, object] = {"required": required}
    if materialization_error is not None:
        payload["materialization_error"] = {
            "type": type(materialization_error).__name__,
            "message": str(materialization_error),
        }
        return payload
    payload["fit_weight_records"] = [
        (
            {"fit_name": record.fit_name, "weight_kind": record.weight_kind}
            if isinstance(record, FitWeightRecord)
            else {
                "invalid_type": (
                    f"{type(record).__module__}.{type(record).__qualname__}"
                )
            }
        )
        for record in (records or ())
    ]
    return payload


def _parity_evidence_payload(evidence: object) -> dict[str, object]:
    if not isinstance(evidence, UKReleaseParityEvidence):
        return {
            "invalid_type": f"{type(evidence).__module__}.{type(evidence).__qualname__}"
        }
    return {
        "candidate_columns": list(evidence.candidate_columns),
        "reference_columns": list(evidence.reference_columns),
        "candidate_targets": list(evidence.candidate_targets),
        "reference_targets": list(evidence.reference_targets),
        "target_relative_errors": dict(evidence.target_relative_errors),
    }


def _entity_tables(dataset: Any) -> tuple[tuple[str, pd.DataFrame], ...]:
    if isinstance(dataset, Mapping):
        raw = tuple((entity, dataset.get(entity)) for entity in _STRUCTURAL_COLUMNS)
    else:
        raw = tuple(
            (entity, getattr(dataset, entity, None)) for entity in _STRUCTURAL_COLUMNS
        )
    if any(not isinstance(table, pd.DataFrame) for _entity, table in raw):
        raise TypeError(
            "UK terminal gates require person, benunit, and household DataFrames."
        )
    return tuple(
        (entity, table) for entity, table in raw if isinstance(table, pd.DataFrame)
    )


def _reviewed_reasons(values: Mapping[str, str] | None) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("UK reviewed exclusions must be a mapping.")
    normalized = {str(name): str(reason) for name, reason in values.items()}
    missing = sorted(name for name, reason in normalized.items() if not reason.strip())
    if missing:
        raise ValueError(f"UK reviewed exclusions need reasons: {missing}.")
    return normalized


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return repr(value)


def _degenerate_kind(series: pd.Series) -> tuple[str, object | None] | None:
    missing = series.isna()
    if bool(missing.all()):
        return ("all_null", None)
    observed = series.loc[~missing]
    if (
        pd.api.types.is_numeric_dtype(observed.dtype)
        or pd.api.types.is_bool_dtype(observed.dtype)
    ) and bool((observed == 0).all()):
        return ("all_zero", 0)
    try:
        unique = pd.unique(observed)
    except (TypeError, ValueError):
        unique = np.asarray(list(dict.fromkeys(map(repr, observed))), dtype=object)
    if len(unique) == 1:
        return ("constant", _json_scalar(unique[0]))
    return None


def uk_degenerate_release_surface_gate(
    dataset: Any,
    *,
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """Reject every all-null, all-zero, or constant nonstructural column."""

    exclusions = _reviewed_reasons(reviewed_exclusions)
    present: set[str] = set()
    live: dict[str, dict[str, object]] = {}
    excluded: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    checked = 0
    for entity, table in _entity_tables(dataset):
        structural = _STRUCTURAL_COLUMNS[entity]
        for column in table.columns:
            if column in structural:
                continue
            checked += 1
            name = f"{entity}.{column}"
            present.add(name)
            finding = _degenerate_kind(table[column])
            if finding is None:
                continue
            kind, value = finding
            detail = {"kind": kind, "value": value}
            if name in exclusions:
                excluded[name] = {**detail, "reason": exclusions[name]}
                continue
            live[name] = detail
            failures.append(
                f"{name}: persisted release column is {kind.replace('_', '-')}"
                + (f" at {value!r}" if kind == "constant" else "")
                + "; populate it with signal, drop it, or record a reviewed "
                "exclusion."
            )

    stale = sorted(
        name for name in exclusions if name in present and name not in excluded
    )
    dormant = sorted(set(exclusions) - present)
    if stale:
        failures.append(
            "Stale reviewed degenerate-column exclusions now carry signal; remove "
            f"them: {stale}."
        )
    by_kind = {
        kind: sorted(name for name, detail in live.items() if detail["kind"] == kind)
        for kind in ("all_null", "all_zero", "constant")
    }
    return GateResult(
        name="degenerate_release_surface",
        passed=not failures,
        failures=tuple(failures),
        details={
            "columns_checked": checked,
            "findings": dict(sorted(live.items())),
            "all_null_columns": by_kind["all_null"],
            "all_zero_columns": by_kind["all_zero"],
            "constant_columns": by_kind["constant"],
            "reviewed_exclusions": dict(sorted(excluded.items())),
            "stale_exclusions": stale,
            "dormant_exclusions": dormant,
        },
    )


def _household_weights(household: pd.DataFrame) -> np.ndarray:
    if _WEIGHT_COLUMN not in household:
        raise ValueError(f"UK household table is missing {_WEIGHT_COLUMN!r}.")
    weights = pd.to_numeric(household[_WEIGHT_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("UK household weights must be a non-empty vector.")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("UK household weights must be finite and non-negative.")
    return weights


def _selector_mask(
    household: pd.DataFrame,
    selector: Mapping[str, object],
) -> tuple[np.ndarray, list[str]]:
    missing = sorted(set(selector) - set(household.columns))
    if missing:
        return np.zeros(len(household), dtype=bool), missing
    mask = np.ones(len(household), dtype=bool)
    for column, expected in selector.items():
        values = household[column]
        matched = (
            values.isna() if expected is None else values.eq(expected).fillna(False)
        )
        mask &= matched.to_numpy(dtype=bool)
    return mask, []


def uk_zero_weight_strata_gate(
    household: pd.DataFrame,
    *,
    declarations: Sequence[UKZeroWeightStratumDeclaration] = (
        UK_DEFAULT_ZERO_WEIGHT_STRATA
    ),
) -> GateResult:
    """Reject zero-weight rows outside or beyond reviewed declarations."""

    if not isinstance(household, pd.DataFrame):
        raise TypeError("UK zero-weight strata gate requires a household DataFrame.")
    weights = _household_weights(household)
    materialized = tuple(declarations)
    if any(
        not isinstance(item, UKZeroWeightStratumDeclaration) for item in materialized
    ):
        raise TypeError(
            "UK zero-weight declarations must be UKZeroWeightStratumDeclaration "
            "instances."
        )
    names = [item.name for item in materialized]
    if len(names) != len(set(names)):
        raise ValueError("UK zero-weight stratum declaration names must be unique.")

    zero = weights == 0.0
    matches = np.zeros(len(household), dtype=np.int64)
    details: list[dict[str, object]] = []
    failures: list[str] = []
    for declaration in materialized:
        mask, missing = _selector_mask(household, declaration.selector)
        selected = zero & mask
        matches += selected.astype(np.int64)
        count = int(selected.sum())
        details.append(
            {
                "name": declaration.name,
                "selector": dict(declaration.selector),
                "maximum_zero_weight_rows": declaration.maximum_zero_weight_rows,
                "zero_weight_rows": count,
                "missing_selector_columns": missing,
                "reason": declaration.reason,
            }
        )
        if missing:
            failures.append(
                f"{declaration.name}: selector column(s) are missing from the "
                f"household release surface: {missing}."
            )
        if count > declaration.maximum_zero_weight_rows:
            failures.append(
                f"{declaration.name}: {count} zero-weight rows exceed the declared "
                f"maximum {declaration.maximum_zero_weight_rows}."
            )

    unmatched_positions = np.flatnonzero(zero & (matches == 0))
    ambiguous_positions = np.flatnonzero(zero & (matches > 1))
    if unmatched_positions.size:
        failures.append(
            f"{unmatched_positions.size} zero-weight household row(s) match no "
            "declared stratum."
        )
    if ambiguous_positions.size:
        failures.append(
            f"{ambiguous_positions.size} zero-weight household row(s) match more "
            "than one declared stratum."
        )
    id_values = (
        household["household_id"].tolist()
        if "household_id" in household
        else list(household.index)
    )
    return GateResult(
        name="zero_weight_strata",
        passed=not failures,
        failures=tuple(failures),
        details={
            "household_rows": len(household),
            "zero_weight_rows": int(zero.sum()),
            "declared_strata": details,
            "unmatched_zero_weight_rows": int(unmatched_positions.size),
            "unmatched_household_examples": [
                _json_scalar(id_values[index]) for index in unmatched_positions[:20]
            ],
            "ambiguous_zero_weight_rows": int(ambiguous_positions.size),
            "ambiguous_household_examples": [
                _json_scalar(id_values[index]) for index in ambiguous_positions[:20]
            ],
        },
    )


def uk_weight_ess_gate(
    weights: Sequence[float] | np.ndarray,
    *,
    minimum_ess_fraction: float = UK_MIN_ESS_FRACTION,
) -> GateResult:
    """Require the shipped household weights to retain effective support."""

    minimum = float(minimum_ess_fraction)
    if not math.isfinite(minimum) or not 0.0 < minimum <= 1.0:
        raise ValueError("minimum_ess_fraction must be finite and in (0, 1].")
    summary = uk_weight_summary(weights)
    fraction = float(summary["ess_fraction"])
    if fraction < minimum:
        failures = (
            f"ESS fraction {fraction:.6g} is below the reviewed minimum {minimum:.6g}.",
        )
    else:
        failures = ()
    return GateResult(
        name="weight_ess",
        passed=not failures,
        failures=failures,
        details={**summary, "minimum_ess_fraction": minimum},
    )


def uk_weight_ratio_gate(
    weights: Sequence[float] | np.ndarray,
    *,
    maximum_max_to_median_ratio: float = UK_MAX_TO_MEDIAN_WEIGHT_RATIO,
) -> GateResult:
    """Backstop a shipped-weight max/positive-median concentration blowout."""

    maximum = float(maximum_max_to_median_ratio)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(
            "maximum_max_to_median_ratio must be finite and strictly positive."
        )
    summary = uk_weight_summary(weights)
    raw_ratio = summary["max_to_median_positive_weight"]
    failures: tuple[str, ...]
    if raw_ratio is None:
        failures = (
            "Max/positive-median weight ratio is undefined because the release "
            "has no positive median weight.",
        )
    else:
        ratio = float(raw_ratio)
        failures = (
            (
                f"Max/positive-median weight ratio {ratio:.6g} exceeds the "
                f"reviewed maximum {maximum:.6g}.",
            )
            if ratio > maximum
            else ()
        )
    return GateResult(
        name="weight_ratio",
        passed=not failures,
        failures=failures,
        details={**summary, "maximum_max_to_median_ratio": maximum},
    )


def _reviewed_export_exclusions(
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    exclusions = dict(UK_REVIEWED_EXPORT_EXCLUSIONS)
    if overrides:
        exclusions.update(_reviewed_reasons(overrides))
    hard = sorted(set(exclusions) & set(UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS))
    if hard:
        raise ValueError(
            "UK export-surface reviewed exclusions cannot waive hard-required "
            f"reference columns: {hard}."
        )
    return exclusions


def uk_export_surface_gate(
    candidate_columns: Iterable[str],
    reference_columns: Iterable[str],
    *,
    allowed_extra_columns: Iterable[str] = UK_ALLOWED_EXTRA_EXPORT_COLUMNS,
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """Run the incumbent-compatible UK export-surface gate."""

    candidate = {str(name) for name in candidate_columns}
    reference = {str(name) for name in reference_columns}
    exclusions = _reviewed_export_exclusions(reviewed_exclusions)
    result = export_surface_gate(
        candidate,
        reference,
        candidate_name=UK_CANDIDATE_DATASET_NAME,
        reference_name=UK_REFERENCE_DATASET_NAME,
        allowed_extra_columns=allowed_extra_columns,
        reviewed_exclusions=exclusions,
    )
    forbidden = sorted(candidate & set(exclusions))
    failures = [*result.failures]
    if not candidate:
        failures.append("UK candidate export-surface evidence is empty.")
    if not reference:
        failures.append("Enhanced-FRS reference export-surface evidence is empty.")
    if forbidden:
        failures.append(
            f"{UK_CANDIDATE_DATASET_NAME}: exports {len(forbidden)} reviewed "
            f"reference-only column(s) that must be dropped: {forbidden[:20]}."
        )
    return GateResult(
        name=result.name,
        passed=not failures,
        failures=tuple(failures),
        details={**dict(result.details), "forbidden_candidate_columns": forbidden},
    )


def uk_target_surface_gate(
    candidate_targets: Iterable[str],
    reference_targets: Iterable[str],
) -> GateResult:
    """Require the UK candidate target surface to cover enhanced FRS."""

    candidate = {str(name) for name in candidate_targets}
    reference = {str(name) for name in reference_targets}
    result = _target_surface_gate(
        candidate,
        reference,
        candidate_name=UK_CANDIDATE_DATASET_NAME,
        reference_name=UK_REFERENCE_DATASET_NAME,
    )
    failures = [*result.failures]
    if not candidate:
        failures.append("UK candidate target-surface evidence is empty.")
    if not reference:
        failures.append("Enhanced-FRS reference target-surface evidence is empty.")
    return GateResult(
        name=result.name,
        passed=not failures,
        failures=tuple(failures),
        details=dict(result.details),
    )


def uk_target_fit_gate(
    target_relative_errors: Mapping[str, float],
    *,
    max_abs_relative_error: float = UK_MAX_TARGET_ABS_RELATIVE_ERROR,
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """Fail a UK artifact with severe shipped-weight target errors."""

    maximum = float(max_abs_relative_error)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("max_abs_relative_error must be finite and non-negative.")
    exclusions = _reviewed_reasons(reviewed_exclusions)
    errors = {str(name): float(error) for name, error in target_relative_errors.items()}
    nonfinite = sorted(
        name for name, error in errors.items() if not math.isfinite(error)
    )
    if nonfinite:
        raise ValueError(f"UK target relative errors must be finite: {nonfinite}.")
    failing = {
        name: error
        for name, error in errors.items()
        if abs(error) > maximum and name not in exclusions
    }
    worst = sorted(failing, key=lambda name: abs(failing[name]), reverse=True)
    failures = [
        f"{UK_CANDIDATE_DATASET_NAME}: {name} relative error "
        f"{failing[name]:+.1%} exceeds {maximum:.0%}."
        for name in worst[:20]
    ]
    if not errors:
        failures.append("UK target-fit evidence is empty.")
    return GateResult(
        name="target_fit",
        passed=not failures,
        failures=tuple(failures),
        details={
            "candidate_name": UK_CANDIDATE_DATASET_NAME,
            "targets_checked": len(errors),
            "max_abs_relative_error": maximum,
            "reviewed_exclusions": exclusions,
            "failing_targets": {name: failing[name] for name in worst},
        },
    )


def _missing_fit_weight_evidence_gate() -> GateResult:
    return GateResult(
        name="weights_audit",
        passed=False,
        failures=(
            "A production fit stage ran but emitted no FitWeightRecord evidence; "
            "an absent audit is not a passing audit.",
        ),
        details={"fits_checked": 0, "evidence_missing": True},
    )


def _evaluate_gate(name: str, evaluator: Callable[[], GateResult]) -> GateResult:
    try:
        result = evaluator()
    except Exception as exc:  # noqa: BLE001 - terminal batch must keep evaluating
        return GateResult(
            name=name,
            passed=False,
            failures=(
                f"Gate evaluation failed closed with {type(exc).__name__}: {exc}",
            ),
            details={
                "evaluation_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            },
        )
    if not isinstance(result, GateResult):
        return GateResult(
            name=name,
            passed=False,
            failures=(
                "Gate evaluation failed closed because the evaluator did not "
                "return GateResult.",
            ),
            details={"returned_type": type(result).__name__},
        )
    if result.name != name:
        return GateResult(
            name=name,
            passed=False,
            failures=(
                f"Gate evaluator returned name {result.name!r}, expected {name!r}.",
            ),
            details={"returned_gate": result.name},
        )
    return result


def uk_terminal_gate_report(
    dataset: Any,
    coverage_engine: Any,
    *,
    input_coverage_evaluator: Callable[[], GateResult] | None = None,
    reviewed_degenerate_exclusions: Mapping[str, str] | None = None,
    zero_weight_declarations: Sequence[UKZeroWeightStratumDeclaration] = (
        UK_DEFAULT_ZERO_WEIGHT_STRATA
    ),
    minimum_ess_fraction: float = UK_MIN_ESS_FRACTION,
    maximum_max_to_median_ratio: float = UK_MAX_TO_MEDIAN_WEIGHT_RATIO,
    fit_weight_records: Iterable[FitWeightRecord] | None = None,
    require_fit_weight_records: bool = False,
    parity_evidence: UKReleaseParityEvidence | None = None,
) -> GateReport:
    """Evaluate every evidenced UK terminal gate and seal its provenance."""

    builtin_coverage_evaluator = input_coverage_evaluator is None
    coverage = input_coverage_evaluator or (
        lambda: uk_release_input_coverage_gate(dataset, coverage_engine)
    )
    fit_stage_present = fit_weight_records is not None or require_fit_weight_records
    materialized_fit_records: tuple[object, ...] | None = None
    fit_materialization_error: Exception | None = None
    if fit_weight_records is not None:
        try:
            materialized_fit_records = tuple(fit_weight_records)
        except Exception as exc:  # noqa: BLE001 - gate records the failed evidence
            materialized_fit_records = ()
            fit_materialization_error = exc
    evaluators: list[tuple[str, Callable[[], GateResult]]] = [
        ("uk_release_input_coverage", coverage),
        (
            "degenerate_release_surface",
            lambda: uk_degenerate_release_surface_gate(
                dataset,
                reviewed_exclusions=reviewed_degenerate_exclusions,
            ),
        ),
        (
            "zero_weight_strata",
            lambda: uk_zero_weight_strata_gate(
                dict(_entity_tables(dataset))["household"],
                declarations=zero_weight_declarations,
            ),
        ),
        (
            "weight_ess",
            lambda: uk_weight_ess_gate(
                _household_weights(dict(_entity_tables(dataset))["household"]),
                minimum_ess_fraction=minimum_ess_fraction,
            ),
        ),
        (
            "weight_ratio",
            lambda: uk_weight_ratio_gate(
                _household_weights(dict(_entity_tables(dataset))["household"]),
                maximum_max_to_median_ratio=maximum_max_to_median_ratio,
            ),
        ),
    ]

    if fit_stage_present:

        def fit_weight_evaluator() -> GateResult:
            if fit_materialization_error is not None:
                raise fit_materialization_error
            if fit_weight_records is None:
                return _missing_fit_weight_evidence_gate()
            if not materialized_fit_records:
                return _missing_fit_weight_evidence_gate()
            return weights_audit_gate(materialized_fit_records)

        evaluators.append(
            (
                "weights_audit",
                fit_weight_evaluator,
            )
        )

    if parity_evidence is not None:

        def checked_parity_evidence() -> UKReleaseParityEvidence:
            if not isinstance(parity_evidence, UKReleaseParityEvidence):
                raise TypeError("parity_evidence must be UKReleaseParityEvidence.")
            return parity_evidence

        evaluators.extend(
            (
                (
                    "export_surface",
                    lambda: uk_export_surface_gate(
                        checked_parity_evidence().candidate_columns,
                        checked_parity_evidence().reference_columns,
                    ),
                ),
                (
                    "target_surface",
                    lambda: uk_target_surface_gate(
                        checked_parity_evidence().candidate_targets,
                        checked_parity_evidence().reference_targets,
                    ),
                ),
                (
                    "target_fit",
                    lambda: uk_target_fit_gate(
                        checked_parity_evidence().target_relative_errors,
                    ),
                ),
            )
        )

    names = [name for name, _evaluator in evaluators]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"UK terminal gate names must be unique: {duplicates}.")
    results = tuple(_evaluate_gate(name, evaluator) for name, evaluator in evaluators)
    evidence_sha256 = {
        "release_dataset": _canonical_sha256(_release_dataset_evidence_payload(results))
    }
    if fit_stage_present:
        evidence_sha256["hmrc_spi_income"] = _canonical_sha256(
            _fit_evidence_payload(
                materialized_fit_records,
                required=require_fit_weight_records,
                materialization_error=fit_materialization_error,
            )
        )
    if parity_evidence is not None:
        evidence_sha256["release_parity"] = _canonical_sha256(
            _parity_evidence_payload(parity_evidence)
        )
    policy_sha256 = _canonical_sha256(
        _terminal_gate_policy_payload(
            builtin_coverage_evaluator=builtin_coverage_evaluator,
            reviewed_degenerate_exclusions=reviewed_degenerate_exclusions,
            zero_weight_declarations=zero_weight_declarations,
            minimum_ess_fraction=minimum_ess_fraction,
            maximum_max_to_median_ratio=maximum_max_to_median_ratio,
        )
    )
    unsigned_attestation = _unsigned_terminal_gate_attestation(
        results,
        policy_sha256=policy_sha256,
        evidence_sha256=evidence_sha256,
    )
    signing_error: RuntimeError | None = None
    try:
        signing_key = _terminal_gate_signing_key()
    except RuntimeError as exc:
        signing_key = None
        signing_error = exc
    attestation = {
        **unsigned_attestation,
        "signature_algorithm": UK_TERMINAL_GATE_SIGNATURE_ALGORITHM,
        "signing_key_sha256": (
            hashlib.sha256(signing_key).hexdigest() if signing_key is not None else None
        ),
    }
    if signing_key is None:
        attestation["signature"] = None
    else:
        unsigned_report = _terminal_gate_report_payload(
            results,
            attestation,
            signature_available=True,
        )
        attestation["signature"] = _terminal_gate_signature(
            signing_key,
            unsigned_report,
        )
    return _AttestedUKTerminalGateReport(
        results,
        policy_sha256=policy_sha256,
        evidence_sha256=evidence_sha256,
        attestation=attestation,
        _signing_error=signing_error,
    )


def write_uk_terminal_gate_report(
    report: GateReport,
    path: str | Path,
) -> Path:
    """Atomically persist an aggregator-signed report without signing input."""

    if type(report) is not _AttestedUKTerminalGateReport:
        raise TypeError(
            "UK terminal gate report writer requires the attested report "
            "returned by uk_terminal_gate_report()."
        )
    output = Path(path)
    payload = report.report_payload()
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    if report._signing_error is not None:
        raise RuntimeError(
            f"{report._signing_error} Unsigned failed report was written to {output}."
        ) from report._signing_error
    return output
