"""Declared household-to-SPM housing receipt interpretation and quantity separation.

This consumer reads only the status, validity, quality, zero-origin and period
evidence the frozen ``asec_housing_status`` artifact already carries. It derives
no new source semantics, opens no source file, and never rewrites the artifact.

Five quantities stay separate here and are never fused:

1. the interview-time receipt status observed on the ASEC household,
2. the counterfactual full contract charge an unassisted comparable unit would
   command (not available in this slice),
3. the reported contract rent an occupant states they pay,
4. the modeled HUD housing assistance payment the engine computes, and
5. the capped SPM housing-subsidy valuation.

The consumer refuses to emit a complete engine boolean while any SPM unit's
assignment is unjustified: those units are named and counted, never silently
dropped and never coerced to observed ``False``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .asec_housing_status import (
    DERIVED_COLUMNS,
    _HousingStatus,
)
from .asec_housing_status_source import ATTACHED_COLUMNS

CONSUMER_VERSION = 1
CONSUMER_KIND = "microcosm.asec_housing_receipt_interpretation.v1"

#: Carried status codes reused verbatim; this consumer renames, never rederives.
RECEIPT_VIEWS = (
    "unknown",
    "observed_receipt",
    "observed_nonreceipt",
    "not_in_universe",
    "conflict",
)

#: SPM-grain assignment outcomes. Only the ``assigned_`` members may reach an
#: engine input; every ``withheld_`` member is preserved with its own count.
ASSIGNMENTS = (
    "assigned_receipt_sole_spm_unit",
    "assigned_nonreceipt_sole_spm_unit",
    "assigned_nonreceipt_household_negative",
    "withheld_scope_ambiguous_receipt",
    "withheld_scope_ambiguous_nonreceipt",
    "withheld_unknown",
    "withheld_conflict",
    "withheld_not_in_universe",
)
ASSIGNED = tuple(name for name in ASSIGNMENTS if name.startswith("assigned_"))

#: ``receives_housing_assistance`` projection codes. ``withheld`` is a third
#: state on purpose: an int8 code cannot be mistaken for a Boolean column.
RECEIPT_INPUT_CODES = {"false": 0, "true": 1, "withheld": 2}

#: The only supported carried period kind. Interview status is not annual
#: income-year participation, and this module derives no bridge to one.
SUPPORTED_PERIOD_KIND = "interview_status"

MULTI_UNIT_NEGATIVE_POLICIES = ("refuse", "distribute")

TAKE_UP_NOTE = (
    "policyengine-us takes_up_housing_assistance_if_eligible defaults to True. "
    "Interview-time status establishes no take-up; leaving a projected unit at "
    "that default is a modelling choice this consumer does not make."
)

#: The five separated quantities. ``engine_binding`` names the actual installed
#: policyengine-us variable, and ``is_engine_input`` records whether Microcosm
#: may ever persist it as a dataset input.
HOUSING_QUANTITIES: dict[str, dict[str, Any]] = {
    "observed_receipt_status": {
        "grain": "asec_household_interview",
        "period_kind": SUPPORTED_PERIOD_KIND,
        "units": "categorical",
        "source": "ASEC HPUBLIC/HLORENT with allocation flags",
        "engine_binding": "receives_housing_assistance",
        "is_engine_input": True,
        "defines_receipt": True,
        "additive_to_tenant_rent": False,
    },
    "counterfactual_full_contract_charge": {
        "grain": "spm_unit",
        "period_kind": "annual",
        "units": "usd",
        "source": "not established in this slice",
        "engine_binding": "pre_subsidy_rent",
        "is_engine_input": True,
        "defines_receipt": False,
        "additive_to_tenant_rent": False,
        "available": False,
        "unavailable_reason": (
            "No year-correct unassisted comparable-charge source is bound. A SIPP "
            "December projection with an explicit rent/mortgage, utility, weight and "
            "2022-to-2024 price treatment is required first."
        ),
    },
    "reported_contract_rent": {
        "grain": "household",
        "period_kind": "monthly_reported",
        "units": "usd",
        "source": "ACS RNTP/GRNTP; ASEC carries no tenant rent-dollar question",
        "engine_binding": None,
        "is_engine_input": False,
        "defines_receipt": False,
        "additive_to_tenant_rent": False,
        "note": (
            "Reported contract rent excludes housing-authority subsidies, so for an "
            "assisted household it is a post-assistance amount, not the full charge."
        ),
    },
    "modeled_hap": {
        "grain": "spm_unit",
        "period_kind": "annual",
        "units": "usd",
        "source": "policyengine-us formula output",
        "engine_binding": "hud_hap",
        "is_engine_input": False,
        "defines_receipt": False,
        "additive_to_tenant_rent": False,
        "note": (
            "A model output cannot define its own pre_subsidy_rent input. HAP may "
            "also exceed rent to owner, reimbursing tenant-paid utilities."
        ),
    },
    "capped_spm_valuation": {
        "grain": "spm_unit",
        "period_kind": "annual",
        "units": "usd",
        "source": "ASEC SPM_CAPHOUSESUB / spm_unit_capped_housing_subsidy",
        "engine_binding": "spm_unit_capped_housing_subsidy",
        "is_engine_input": False,
        "defines_receipt": False,
        "additive_to_tenant_rent": False,
        "note": (
            "A censored SPM resource valuation. Zero is compatible with positive "
            "assistance, so it may corroborate but never define receipt."
        ),
    },
}

#: Named refusals for the fusions the reported-rent bridge assessment rejected.
QUANTITY_FUSIONS_REFUSED = {
    ("reported_contract_rent", "capped_spm_valuation"): (
        "CAPPED_VALUATION_NOT_ADDITIVE_TO_RENT"
    ),
    ("reported_contract_rent", "modeled_hap"): (
        "MODEL_OUTPUT_CANNOT_DEFINE_ITS_OWN_INPUT"
    ),
    ("observed_receipt_status", "capped_spm_valuation"): (
        "CAPPED_VALUATION_CANNOT_DEFINE_RECEIPT"
    ),
    ("counterfactual_full_contract_charge", "modeled_hap"): (
        "MODEL_OUTPUT_CANNOT_DEFINE_ITS_OWN_INPUT"
    ),
}


class HousingReceiptRefusalError(ValueError):
    """Value-free refusal reason for unsupported consumer evidence or fusion."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HousingReceiptRefusalError(reason)


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _counts(codes: np.ndarray, names: tuple[str, ...]) -> dict[str, int]:
    tally = np.bincount(codes.astype("int64"), minlength=len(names))
    return {name: int(tally[index]) for index, name in enumerate(names)}


def _bit_counts(values: np.ndarray, codebook: dict[str, int]) -> dict[str, int]:
    return {
        name: int(((values.astype("int64") & bit) > 0).sum())
        for name, bit in sorted(codebook.items(), key=lambda item: item[1])
    }


def receipt_columns_from_status(status: _HousingStatus) -> dict[str, Any]:
    """Project the carried derived columns and provenance out of an artifact.

    The artifact validates itself first, so a tampered buffer cannot reach the
    consumer. Synthetic and authenticated statuses both work; the projection
    carries ``source_authentication`` so a synthetic-derived interpretation stays
    visibly unauthenticated downstream.
    """
    status.validate()
    header = status.header_data
    return {
        "columns": {name: status.array(name) for name in DERIVED_COLUMNS},
        "source_period_kind": header["source_period_kind"],
        "source_authentication": header["source_authentication"],
        "zero_origin_policy": header["zero_origin_policy"],
        "codebook": header["codebook"],
        "status_content_sha256": status.content_sha256,
    }


def receipt_columns_from_attached_household(
    household: pd.DataFrame, *, source_period_kind: str, source_authentication: str
) -> dict[str, Any]:
    """Read the attached ``asec_housing_observed_*`` columns off a household table."""
    _require(type(household) is pd.DataFrame, "HOUSEHOLD_TABLE")
    _require(set(ATTACHED_COLUMNS) <= set(household.columns), "ATTACHED_COLUMNS")
    return {
        "columns": {
            name: household[attached].to_numpy(dtype="uint8", copy=True)
            for name, attached in zip(DERIVED_COLUMNS, ATTACHED_COLUMNS, strict=True)
        },
        "source_period_kind": source_period_kind,
        "source_authentication": source_authentication,
        "zero_origin_policy": None,
        "codebook": None,
        "status_content_sha256": None,
    }


@dataclass(frozen=True)
class HouseholdReceiptInterpretation:
    """Household-grain receipt view with every carried invalidity preserved."""

    receipt_view: np.ndarray
    receipt_valid: np.ndarray
    conflicts: np.ndarray
    unknown_reasons: np.ndarray
    public_quality: np.ndarray
    lower_quality: np.ndarray
    zero_origin: dict[str, np.ndarray]
    source_period_kind: str
    source_authentication: str
    status_content_sha256: str | None

    @property
    def rows(self) -> int:
        return int(self.receipt_view.shape[0])

    def view_counts(self) -> dict[str, int]:
        return _counts(self.receipt_view, RECEIPT_VIEWS)


def interpret_household_receipt(
    projection: dict[str, Any],
) -> HouseholdReceiptInterpretation:
    """Rename the carried household status into an explicit receipt view.

    Every carried invariant is rechecked rather than trusted: the code domains,
    the ``receipt_valid`` mask's agreement with the status, and the exclusivity
    of conflict. No unknown, conflict or not-in-universe household becomes
    observed nonreceipt, and no zero-origin evidence is discarded.
    """
    _require(type(projection) is dict, "PROJECTION_TYPE")
    columns = projection.get("columns")
    _require(type(columns) is dict, "PROJECTION_COLUMNS")
    _require(set(columns) == set(DERIVED_COLUMNS), "PROJECTION_COLUMNS")
    period_kind = projection.get("source_period_kind")
    _require(period_kind == SUPPORTED_PERIOD_KIND, "UNSUPPORTED_PERIOD_KIND")
    authentication = projection.get("source_authentication")
    _require(type(authentication) is str and authentication, "SOURCE_AUTHENTICATION")
    rows = None
    for name in DERIVED_COLUMNS:
        value = columns[name]
        _require(type(value) is np.ndarray, "COLUMN_TYPE")
        _require(value.dtype == np.dtype("uint8"), "COLUMN_DTYPE")
        _require(value.ndim == 1, "COLUMN_SHAPE")
        rows = value.shape[0] if rows is None else rows
        _require(value.shape[0] == rows, "COLUMN_LENGTH")
    _require(rows is not None and rows > 0, "ROW_COUNT")

    status = columns["status"]
    _require(bool((status < len(RECEIPT_VIEWS)).all()), "STATUS_DOMAIN")
    _require(bool((columns["route"] <= 4).all()), "ROUTE_DOMAIN")
    _require(
        bool((columns["public_quality"] <= 3).all())
        and bool((columns["lower_quality"] <= 3).all()),
        "QUALITY_DOMAIN",
    )
    receipt_valid = columns["receipt_valid"]
    _require(bool((receipt_valid <= 1).all()), "VALIDITY_DOMAIN")
    _require(
        bool((receipt_valid == np.isin(status, [1, 2]).astype("uint8")).all()),
        "VALIDITY_DISAGREES_WITH_STATUS",
    )
    conflicts = columns["conflicts"]
    _require(bool(((conflicts > 0) == (status == 4)).all()), "CONFLICT_DISAGREES")
    unknown_reasons = columns["unknown_reasons"]
    _require(
        bool((unknown_reasons[receipt_valid == 1] == 0).all()),
        "UNKNOWN_REASON_ON_VALID_RECEIPT",
    )
    return HouseholdReceiptInterpretation(
        receipt_view=status.copy(),
        receipt_valid=receipt_valid.copy(),
        conflicts=conflicts.copy(),
        unknown_reasons=unknown_reasons.copy(),
        public_quality=columns["public_quality"].copy(),
        lower_quality=columns["lower_quality"].copy(),
        zero_origin={
            name: columns[name].copy()
            for name in DERIVED_COLUMNS
            if name.endswith("_zero_origin")
        },
        source_period_kind=period_kind,
        source_authentication=authentication,
        status_content_sha256=projection.get("status_content_sha256"),
    )


@dataclass(frozen=True)
class SpmReceiptScope:
    """SPM-grain assignment of a household-grain observation, ambiguity kept."""

    spm_unit_id: np.ndarray
    household_row: np.ndarray
    receipt_view: np.ndarray
    spm_units_in_household: np.ndarray
    assignment: np.ndarray
    interpretation: HouseholdReceiptInterpretation

    @property
    def rows(self) -> int:
        return int(self.spm_unit_id.shape[0])

    def assignment_counts(self) -> dict[str, int]:
        return _counts(self.assignment, ASSIGNMENTS)


def _membership(
    *,
    household_id: np.ndarray,
    spm_unit_id: np.ndarray,
    person_household_id: np.ndarray,
    person_spm_unit_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each SPM unit to its single household row and count units per household."""
    for name, value in (
        ("household_id", household_id),
        ("spm_unit_id", spm_unit_id),
        ("person_household_id", person_household_id),
        ("person_spm_unit_id", person_spm_unit_id),
    ):
        _require(type(value) is np.ndarray and value.ndim == 1, "MEMBERSHIP_SHAPE")
        _require(value.dtype.kind == "i", "MEMBERSHIP_DTYPE")
        _require(value.shape[0] > 0, f"MEMBERSHIP_EMPTY:{name}")
    _require(
        len(np.unique(household_id)) == household_id.shape[0], "DUPLICATE_HOUSEHOLD"
    )
    _require(len(np.unique(spm_unit_id)) == spm_unit_id.shape[0], "DUPLICATE_SPM_UNIT")
    _require(
        person_household_id.shape[0] == person_spm_unit_id.shape[0], "PERSON_LENGTH"
    )
    _require(
        bool(np.isin(person_household_id, household_id).all()), "UNKNOWN_HOUSEHOLD"
    )
    _require(bool(np.isin(person_spm_unit_id, spm_unit_id).all()), "UNKNOWN_SPM_UNIT")
    _require(bool(np.isin(spm_unit_id, person_spm_unit_id).all()), "EMPTY_SPM_UNIT")
    _require(bool(np.isin(household_id, person_household_id).all()), "EMPTY_HOUSEHOLD")

    order = np.argsort(household_id, kind="stable")
    household_row_of_person = order[
        np.searchsorted(household_id[order], person_household_id)
    ]
    spm_order = np.argsort(spm_unit_id, kind="stable")
    spm_row_of_person = spm_order[
        np.searchsorted(spm_unit_id[spm_order], person_spm_unit_id)
    ]
    household_row = np.full(spm_unit_id.shape[0], -1, dtype="int64")
    household_row[spm_row_of_person] = household_row_of_person
    spans = np.zeros(spm_unit_id.shape[0], dtype=bool)
    np.logical_or.at(
        spans,
        spm_row_of_person,
        household_row[spm_row_of_person] != household_row_of_person,
    )
    _require(not bool(spans.any()), "SPM_UNIT_SPANS_HOUSEHOLDS")
    units_per_household = np.bincount(household_row, minlength=household_id.shape[0])
    return household_row, units_per_household


def resolve_spm_receipt_scope(
    interpretation: HouseholdReceiptInterpretation,
    *,
    household_id: np.ndarray,
    spm_unit_id: np.ndarray,
    person_household_id: np.ndarray,
    person_spm_unit_id: np.ndarray,
) -> SpmReceiptScope:
    """Carry the household observation to SPM units without inventing a recipient.

    A household-level positive answer names the housing unit, not one of several
    SPM units sharing it, so a multi-SPM-unit household's receipt stays
    ``withheld_scope_ambiguous_receipt``. Replicating it would hand every SPM unit
    its own voucher budget: the installed engine gives each unit the household's
    full payment standard, so the replication is not conservative.

    A household-level negative answer is a statement about the housing unit
    itself and is separated as ``withheld_scope_ambiguous_nonreceipt`` so the
    caller decides explicitly whether to distribute it.
    """
    _require(
        type(interpretation) is HouseholdReceiptInterpretation, "INTERPRETATION_TYPE"
    )
    _require(household_id.shape[0] == interpretation.rows, "HOUSEHOLD_ALIGNMENT")
    household_row, units_per_household = _membership(
        household_id=household_id,
        spm_unit_id=spm_unit_id,
        person_household_id=person_household_id,
        person_spm_unit_id=person_spm_unit_id,
    )
    view = interpretation.receipt_view[household_row]
    units = units_per_household[household_row]
    sole = units == 1
    assignment = np.full(view.shape[0], ASSIGNMENTS.index("withheld_unknown"), "uint8")
    for mask, name in (
        (view == 4, "withheld_conflict"),
        (view == 3, "withheld_not_in_universe"),
        ((view == 1) & sole, "assigned_receipt_sole_spm_unit"),
        ((view == 1) & ~sole, "withheld_scope_ambiguous_receipt"),
        ((view == 2) & sole, "assigned_nonreceipt_sole_spm_unit"),
        ((view == 2) & ~sole, "withheld_scope_ambiguous_nonreceipt"),
    ):
        assignment[mask] = ASSIGNMENTS.index(name)
    return SpmReceiptScope(
        spm_unit_id=spm_unit_id.copy(),
        household_row=household_row,
        receipt_view=view,
        spm_units_in_household=units,
        assignment=assignment,
        interpretation=interpretation,
    )


@dataclass(frozen=True)
class EngineReceiptProjection:
    """Three-state projection: no withheld SPM unit ever becomes a Boolean."""

    spm_unit_id: np.ndarray
    code: np.ndarray
    assignment: np.ndarray
    multi_unit_negative: str
    _receipt: bytes

    @property
    def receipt(self) -> dict:
        return json.loads(self._receipt)

    @property
    def withheld(self) -> np.ndarray:
        return self.code == RECEIPT_INPUT_CODES["withheld"]

    def code_counts(self) -> dict[str, int]:
        names = tuple(
            name for name, _ in sorted(RECEIPT_INPUT_CODES.items(), key=lambda i: i[1])
        )
        return _counts(self.code, names)


def project_engine_receipt_inputs(
    scope: SpmReceiptScope, *, multi_unit_negative: str
) -> EngineReceiptProjection:
    """Emit ``receives_housing_assistance`` only where the assignment is justified.

    ``multi_unit_negative`` has no default: the caller must declare whether a
    multi-SPM-unit household's negative answer distributes to each unit
    (``"distribute"``) or stays withheld (``"refuse"``). Every other withheld
    class stays withheld under both policies.
    """
    _require(type(scope) is SpmReceiptScope, "SCOPE_TYPE")
    _require(
        multi_unit_negative in MULTI_UNIT_NEGATIVE_POLICIES,
        "MULTI_UNIT_NEGATIVE_POLICY",
    )
    assignment = scope.assignment.copy()
    if multi_unit_negative == "distribute":
        assignment[
            assignment == ASSIGNMENTS.index("withheld_scope_ambiguous_nonreceipt")
        ] = ASSIGNMENTS.index("assigned_nonreceipt_household_negative")
    code = np.full(assignment.shape[0], RECEIPT_INPUT_CODES["withheld"], dtype="uint8")
    code[assignment == ASSIGNMENTS.index("assigned_receipt_sole_spm_unit")] = (
        RECEIPT_INPUT_CODES["true"]
    )
    for name in (
        "assigned_nonreceipt_sole_spm_unit",
        "assigned_nonreceipt_household_negative",
    ):
        code[assignment == ASSIGNMENTS.index(name)] = RECEIPT_INPUT_CODES["false"]
    interpretation = scope.interpretation
    receipt = {
        "schema_version": CONSUMER_VERSION,
        "artifact_kind": CONSUMER_KIND,
        "release_eligible": False,
        "source_authentication": interpretation.source_authentication,
        "source_period_kind": interpretation.source_period_kind,
        "status_content_sha256": interpretation.status_content_sha256,
        "annual_income_year_participation": "not_established",
        "multi_unit_negative": multi_unit_negative,
        "engine_input": "receives_housing_assistance",
        "engine_take_up_input": "not_derived",
        "take_up_note": TAKE_UP_NOTE,
        "household_view_counts": interpretation.view_counts(),
        "assignment_counts": _counts(assignment, ASSIGNMENTS),
        "spm_unit_rows": int(assignment.shape[0]),
        "withheld_rows": int((code == RECEIPT_INPUT_CODES["withheld"]).sum()),
    }
    return EngineReceiptProjection(
        spm_unit_id=scope.spm_unit_id.copy(),
        code=code,
        assignment=assignment,
        multi_unit_negative=multi_unit_negative,
        _receipt=_json(receipt),
    )


def engine_receives_housing_assistance(
    projection: EngineReceiptProjection,
) -> np.ndarray:
    """Return the Boolean engine input, or refuse while any SPM unit is withheld.

    This is the blanket-Boolean refusal: an incomplete observation cannot be
    handed to ``receives_housing_assistance``, because the engine's default
    ``False`` would silently assert measured nonreceipt for withheld units.
    """
    _require(type(projection) is EngineReceiptProjection, "PROJECTION_TYPE")
    withheld = int(projection.withheld.sum())
    _require(withheld == 0, f"WITHHELD_SPM_UNITS:{withheld}")
    return projection.code.astype(bool)


def annual_income_year_participation(
    projection: EngineReceiptProjection,
) -> np.ndarray:
    """Always refuse: an interview-time answer is not income-year participation."""
    _require(type(projection) is EngineReceiptProjection, "PROJECTION_TYPE")
    raise HousingReceiptRefusalError("ANNUAL_PARTICIPATION_NOT_OBSERVED_AT_INTERVIEW")


def engine_takes_up_housing_assistance_if_eligible(
    projection: EngineReceiptProjection,
) -> np.ndarray:
    """Always refuse: interview status establishes no take-up behaviour."""
    _require(type(projection) is EngineReceiptProjection, "PROJECTION_TYPE")
    raise HousingReceiptRefusalError("TAKE_UP_NOT_DERIVABLE_FROM_INTERVIEW_STATUS")


def corroborate_with_capped_valuation(
    scope: SpmReceiptScope, capped_valuation: np.ndarray
) -> dict[str, Any]:
    """Cross-tabulate the capped SPM valuation against the view, changing nothing.

    A zero cap is compatible with positive assistance and the valuation is
    censored, so disagreement is diagnostic evidence, never a status correction.
    """
    _require(type(scope) is SpmReceiptScope, "SCOPE_TYPE")
    _require(type(capped_valuation) is np.ndarray, "VALUATION_TYPE")
    _require(capped_valuation.ndim == 1, "VALUATION_SHAPE")
    _require(capped_valuation.shape[0] == scope.rows, "VALUATION_LENGTH")
    _require(capped_valuation.dtype.kind in "fi", "VALUATION_DTYPE")
    _require(bool(np.isfinite(capped_valuation).all()), "VALUATION_FINITE")
    _require(bool((capped_valuation >= 0).all()), "VALUATION_NEGATIVE")
    positive = capped_valuation > 0
    table = {
        view: {
            "positive_valuation": int((positive & (scope.receipt_view == index)).sum()),
            "zero_valuation": int((~positive & (scope.receipt_view == index)).sum()),
        }
        for index, view in enumerate(RECEIPT_VIEWS)
    }
    return {
        "role": "corroboration_only",
        "defines_receipt": False,
        "additive_to_tenant_rent": False,
        "note": HOUSING_QUANTITIES["capped_spm_valuation"]["note"],
        "table": table,
        "receipt_with_zero_valuation": table["observed_receipt"]["zero_valuation"],
        "nonreceipt_with_positive_valuation": table["observed_nonreceipt"][
            "positive_valuation"
        ],
    }


def compose_full_contract_charge(**quantities: np.ndarray) -> np.ndarray:
    """Refuse every fusion the reported-rent bridge assessment rejected.

    No supported combination of the currently bound quantities identifies the
    counterfactual full contract charge, so this always refuses. The refusal
    reason names the specific rejected shortcut where one applies.
    """
    names = set(quantities)
    _require(names <= set(HOUSING_QUANTITIES), "UNKNOWN_QUANTITY")
    for pair, reason in QUANTITY_FUSIONS_REFUSED.items():
        if set(pair) <= names:
            raise HousingReceiptRefusalError(reason)
    raise HousingReceiptRefusalError("FULL_CONTRACT_CHARGE_SOURCE_UNAVAILABLE")


def counterfactual_charge_availability() -> dict[str, Any]:
    """Report the unavailable counterfactual charge and its blocking defects."""
    quantity = HOUSING_QUANTITIES["counterfactual_full_contract_charge"]
    return {
        "available": quantity["available"],
        "reason": quantity["unavailable_reason"],
        "blocking_defects": (
            "installed FMR table falls forward to FY2025 for a 2024 request",
            "utility allowance is encoded for selected PHAs only and is zero elsewhere",
            "bedrooms are not observed for the assisted universe",
            "pha_payment_standard carries an undated LA branch",
        ),
        "refused_shortcut": "FMR_WITHOUT_YEAR_CORRECT_SOURCE_AND_UTILITY_CONTRACT",
        "requirements": (
            "R/us-housing-consumer-sipp-december-projection-requirements.md"
        ),
    }


def housing_receipt_diagnostics(
    interpretation: HouseholdReceiptInterpretation,
    scope: SpmReceiptScope | None = None,
    projection: EngineReceiptProjection | None = None,
) -> dict[str, Any]:
    """Raw status, unknown-reason, conflict, quality and zero-origin counts."""
    _require(
        type(interpretation) is HouseholdReceiptInterpretation, "INTERPRETATION_TYPE"
    )
    quality_names = (
        "not_applicable",
        "no_allocation_code_origin_unresolved",
        "allocated",
        "conflict",
    )
    payload: dict[str, Any] = {
        "schema_version": CONSUMER_VERSION,
        "source_authentication": interpretation.source_authentication,
        "source_period_kind": interpretation.source_period_kind,
        "household_rows": interpretation.rows,
        "household_view_counts": interpretation.view_counts(),
        "receipt_valid_rows": int(interpretation.receipt_valid.sum()),
        "unknown_reason_counts": _bit_counts(
            interpretation.unknown_reasons,
            {"parent_answer": 1, "lower_answer": 2, "scope": 4},
        ),
        "conflict_counts": _bit_counts(
            interpretation.conflicts,
            {
                "excluded_scope": 1,
                "parent_route": 2,
                "public_allocation": 4,
                "lower_allocation": 8,
                "context": 16,
            },
        ),
        "public_quality_counts": _counts(interpretation.public_quality, quality_names),
        "lower_quality_counts": _counts(interpretation.lower_quality, quality_names),
        "zero_origin_counts": {
            name: int(values.sum())
            for name, values in sorted(interpretation.zero_origin.items())
        },
    }
    if scope is not None:
        _require(type(scope) is SpmReceiptScope, "SCOPE_TYPE")
        payload["spm_unit_rows"] = scope.rows
        payload["assignment_counts"] = scope.assignment_counts()
        payload["multi_spm_unit_households"] = int(
            len(np.unique(scope.household_row[scope.spm_units_in_household > 1]))
        )
    if projection is not None:
        _require(type(projection) is EngineReceiptProjection, "PROJECTION_TYPE")
        payload["engine_input_code_counts"] = projection.code_counts()
        payload["multi_unit_negative"] = projection.multi_unit_negative
        payload["engine_boolean_complete"] = bool(projection.withheld.sum() == 0)
    payload["diagnostics_sha256"] = _sha(_json(payload))
    return payload


def describe_integration_seam() -> dict[str, Any]:
    """The proposed attachment point, its inputs, and what it refuses to export."""
    return {
        "schema_version": CONSUMER_VERSION,
        "consumer_module": "microcosm.build.us_runtime.asec_housing_receipt_consumer",
        "reads": {
            "artifact": "microcosm.asec_housing_status_observations.v1",
            "attachment": "microcosm.asec_housing_status_attachment.v1",
            "attached_columns": list(ATTACHED_COLUMNS),
        },
        "attaches_after": (
            "microcosm.build.us_runtime.asec_housing_status_source.attach_housing_status"
        ),
        "membership_source": (
            "the attached Frame's household_id, spm_unit_id and person memberships"
        ),
        "produces": {
            "receives_housing_assistance": "three-state code, Boolean only when complete",
            "takes_up_housing_assistance_if_eligible": "not derived",
            "pre_subsidy_rent": "not derived; needs the counterfactual charge source",
        },
        "refuses_to_export": {
            "rent": "formula-owned",
            "housing_assistance": "formula-owned",
            "hud_hap": "formula-owned",
            "spm_unit_capped_housing_subsidy": "formula-owned",
        },
        "export_gate": "microcosm.frame.adapters.policyengine_us.PolicyEngineUSEngine",
        "quantities": HOUSING_QUANTITIES,
        "unresolved_before_launch": counterfactual_charge_availability(),
    }
