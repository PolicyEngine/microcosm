"""Transfer missing PolicyEngine input leaves onto an ACS recipient frame.

ACS observes a dense demographic and geography spine but not every input the
US model consumes. This module learns missing numeric and boolean leaves from
an ASEC x PUF donor with :mod:`microcosm.fit`. Native values always win: only
null target cells are filled, and model-only predictors are never added to the
returned frame.

Optional ACS predictors keep their source missingness. Recipient rows are
partitioned by the predictors they actually observe, then one weighted QRF is
fit per availability pattern using complete donor rows for that predictor set.
No missing value is ever converted to zero merely to satisfy an estimator.

Imputation provenance is external to the frame. ``imputed_inputs`` records one
immutable entry per filled column, including every availability-pattern fit;
it is not an export column and cannot leak into a PolicyEngine dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import import_module
from typing import Any, Protocol

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.serialization_dtypes import canonicalize_frame_string_dtypes
from microcosm.build.us_runtime.capital_gain_distributions import (
    capital_gain_distribution_shares_asset_identity,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    resolve_formula_owned_outputs,
)
from microcosm.build.us_runtime.support_provenance import (
    has_support_role_metadata,
    support_role_series,
)
from microcosm.fit import DEFAULT_ZERO_ATOL, QRFChainState
from microcosm.fit.qrf import detect_regime
from microcosm.frame import EntitySchema, Frame, Weights

QRF: Any | None = None

__all__ = [
    "ACS_DONOR_CHANNEL_AUTO",
    "ACS_DEFERRED_GEOGRAPHY_INPUTS",
    "ACS_GROUP_TRANSFER_PREDICTORS",
    "DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT",
    "ACS_NATIVE_PERSON_INPUTS",
    "ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS",
    "ACS_PERSON_TRANSFER_PREDICTORS",
    "ASEC_PUF_DONOR_SPINE",
    "AcsImputedInput",
    "AcsTransferBankPatternStep",
    "AcsTransferPattern",
    "AcsTransferResult",
    "AcsTransferTargetBank",
    "AcsTransferTargetCheckpoint",
    "TargetFamilies",
    "acs_transfer_donor_requirements",
    "acs_transfer_execution_contract_identity",
    "acs_adult_care_qualifying_rows",
    "assert_acs_transfer_targets_are_input_leaves",
    "declared_acs_transfer_target_families",
    "default_acs_transfer_target_families",
    "ACS_DERIVED_TRANSFER_INPUTS",
    "derive_acs_schedule_d_capital_gain_distributions",
    "reconcile_acs_adult_care",
    "required_acs_transfer_inputs",
    "resolve_acs_donor_channel",
    "transfer_acs_inputs",
]

ASEC_PUF_DONOR_SPINE = "asec_puf"

# Safe production default. When support-channel metadata is present, AUTO
# resolves to the PUF channel; an unchanneled donor is used as-is. Explicit
# ``None`` remains the deliberate whole-donor escape hatch.
ACS_DONOR_CHANNEL_AUTO = "auto"

# QRF retains one gated forest set per target in a fitted family. Bounding the
# chain length keeps production transfer below the launch worker's memory
# budget while continuing to use microcosm-fit's canonical chained QRF API.
DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT = 8

# Required ACS predictor columns. State is broadcast from the household table
# so every modeled imputation is conditioned on the ACS record's real state
# rather than drawing from a national donor mixture. Cross-spine assembly may
# leave peer rows null; row-level complete-case masks keep those rows out of QRF.
_STATE_FEATURE = "__acs_transfer_state_fips"
ACS_PERSON_TRANSFER_PREDICTORS: tuple[str, ...] = (
    "age",
    "is_female",
    _STATE_FEATURE,
)

# Native leaves are never default transfer targets. Their column-level and
# row-level missingness belongs to the ACS loader.
ACS_NATIVE_PERSON_INPUTS = frozenset(
    {
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
        "real_estate_taxes",
    }
)

# Geography must remain native to the ACS spine. Drawing a donor block, tract,
# county, or district would sever the PUMS state/PUMA geography the second
# spine exists to preserve. ``state_fips`` is already mapped natively, but is
# listed here too so a malformed recipient can never make it a transfer target.
ACS_DEFERRED_GEOGRAPHY_INPUTS = frozenset(
    {
        "block_geoid",
        "congressional_district_geoid",
        "county_fips",
        "tract_geoid",
    }
)

# These inputs are numeric in current dense artifacts, but their domain is a
# finite set of years. QRF quantile interpolation is valid for continuous
# amounts and invalid for a year code, so predictions are snapped to observed
# donor support exactly as the existing PUF support stage does.
_DISCRETE_NUMERIC_TARGETS = frozenset(
    {
        "first_home_mortgage_origination_year",
        "second_home_mortgage_origination_year",
    }
)

_IMMIGRATION_STATUS_TARGETS = (
    "ssn_card_type",
    "immigration_status_str",
)
_IMMIGRATION_STATUS_MODEL_TARGET = "__acs_transfer_immigration_status_pair"

# RNTP/GRNTP are not pre-subsidy rent, so this leaf remains donor-transferable.
_ADDITIONAL_PERSON_TRANSFER_TARGETS = ("pre_subsidy_rent",)
_HOUSING_TRANSFER_TARGETS = frozenset(_ADDITIONAL_PERSON_TRANSFER_TARGETS)

_EMPLOYMENT_FEATURE = "__acs_transfer_employment_income"
_SELF_EMPLOYMENT_FEATURE = "__acs_transfer_self_employment_income"
_SOCIAL_SECURITY_FEATURE = "__acs_transfer_social_security_income"
_RETIREMENT_FEATURE = "__acs_transfer_retirement_income"
_INVESTMENT_FEATURE = "__acs_transfer_interest_dividend_rental_income"
_HEAD_FEATURE = "__acs_transfer_is_household_head"
_TENURE_FEATURE = "__acs_transfer_tenure_code"

ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS: tuple[str, ...] = (
    _EMPLOYMENT_FEATURE,
    _SELF_EMPLOYMENT_FEATURE,
    _SOCIAL_SECURITY_FEATURE,
    _RETIREMENT_FEATURE,
    _INVESTMENT_FEATURE,
    _HEAD_FEATURE,
    _TENURE_FEATURE,
)

_GROUP_PERSON_COUNT = "__acs_transfer_person_count"
_GROUP_AGE_SUM = "__acs_transfer_age_sum"
_GROUP_FEMALE_COUNT = "__acs_transfer_female_count"
ACS_GROUP_TRANSFER_PREDICTORS: tuple[str, ...] = (
    _GROUP_PERSON_COUNT,
    _GROUP_AGE_SUM,
    _GROUP_FEMALE_COUNT,
    _STATE_FEATURE,
)

_GROUP_OPTIONAL_NAMES: Mapping[str, str] = {
    _EMPLOYMENT_FEATURE: "__acs_transfer_group_employment_income_sum",
    _SELF_EMPLOYMENT_FEATURE: "__acs_transfer_group_self_employment_income_sum",
    _SOCIAL_SECURITY_FEATURE: "__acs_transfer_group_social_security_income_sum",
    _RETIREMENT_FEATURE: "__acs_transfer_group_retirement_income_sum",
    _INVESTMENT_FEATURE: "__acs_transfer_group_investment_income_sum",
    _HEAD_FEATURE: "__acs_transfer_group_head_count",
    _TENURE_FEATURE: "__acs_transfer_group_tenure_code",
}

_DONOR_COMBINED_COMPONENTS: Mapping[str, tuple[str, ...]] = {
    _SOCIAL_SECURITY_FEATURE: (
        "social_security_retirement",
        "social_security_disability",
        "social_security_dependents",
        "social_security_survivors",
    ),
    _RETIREMENT_FEATURE: (
        "taxable_private_pension_income",
        "tax_exempt_private_pension_income",
        "taxable_ira_distributions",
    ),
    _INVESTMENT_FEATURE: (
        "taxable_interest_income",
        "tax_exempt_interest_income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
        "rental_income",
        "estate_income",
    ),
}

_RECIPIENT_COMBINED_SOURCES: Mapping[str, str] = {
    _SOCIAL_SECURITY_FEATURE: "acs_social_security_income",
    _RETIREMENT_FEATURE: "acs_retirement_income",
    _INVESTMENT_FEATURE: "acs_interest_dividend_rental_income",
}

_TENURE_CODES: Mapping[str, float] = {
    "NONE": 0.0,
    "OWNED_WITH_MORTGAGE": 1.0,
    "OWNED WITH MORTGAGE": 1.0,
    "OWNER_WITH_MORTGAGE": 1.0,
    "OWNER WITH MORTGAGE": 1.0,
    "OWNED_OUTRIGHT": 2.0,
    "OWNED OUTRIGHT": 2.0,
    "OWNER_WITHOUT_MORTGAGE": 2.0,
    "OWNER WITHOUT MORTGAGE": 2.0,
    "RENTED": 3.0,
    "RENTER": 3.0,
}


def acs_transfer_execution_contract_identity(
    *,
    targets: Sequence[str] | None = None,
    derive_schedule_d: bool = True,
) -> dict[str, object]:
    """Bind the complete runtime predictor and target-codec contract."""

    requested_targets = frozenset(() if targets is None else targets)
    schedule_d_enabled = derive_schedule_d and bool(
        requested_targets & {_SCHEDULE_D_CGD_SOURCE, _SCHEDULE_D_CGD_EXCLUSIVE_WITH}
    )
    adult_care_enabled = _ADULT_CARE_EXPENSE in requested_targets
    payload: dict[str, object] = {
        "schema_version": 2,
        "person_required_predictors": list(ACS_PERSON_TRANSFER_PREDICTORS),
        "person_optional_predictors": list(ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS),
        "group_required_predictors": list(ACS_GROUP_TRANSFER_PREDICTORS),
        "group_optional_names": dict(sorted(_GROUP_OPTIONAL_NAMES.items())),
        "donor_combined_components": {
            feature: list(columns)
            for feature, columns in sorted(_DONOR_COMBINED_COMPONENTS.items())
        },
        "recipient_combined_sources": dict(sorted(_RECIPIENT_COMBINED_SOURCES.items())),
        "housing": {
            "targets": sorted(_HOUSING_TRANSFER_TARGETS),
            "mandatory_features": [_HEAD_FEATURE, _TENURE_FEATURE],
            "head_source_precedence": [
                {"source": "is_household_head", "head_codes": [True]},
                {"source": "A_EXPRRP", "head_codes": [1, 2]},
                {"source": "A_LINENO", "head_codes": [1]},
            ],
            "tenure_source_precedence": [
                "tenure_type",
                "spm_unit_tenure_type",
            ],
        },
        "tenure_codes": dict(sorted(_TENURE_CODES.items())),
        "immigration_status_targets": list(_IMMIGRATION_STATUS_TARGETS),
        "immigration_status_model_target": _IMMIGRATION_STATUS_MODEL_TARGET,
        "discrete_numeric_targets": sorted(_DISCRETE_NUMERIC_TARGETS),
        "post_transfer_structure": {
            "schedule_d_capital_gain_distributions": {
                "enabled": schedule_d_enabled,
                "source": _SCHEDULE_D_CGD_SOURCE,
                "exclusive_with": _SCHEDULE_D_CGD_EXCLUSIVE_WITH,
                "output": _SCHEDULE_D_CGD_COLUMN,
                "preserve_preexisting_nonnull": True,
                "share_asset": (
                    capital_gain_distribution_shares_asset_identity()
                    if schedule_d_enabled
                    else None
                ),
            },
            "adult_care": {
                "enabled": adult_care_enabled,
                "flag": _ADULT_CARE_FLAG,
                "expense": _ADULT_CARE_EXPENSE,
                "tax_unit_role": _ADULT_CARE_ROLE,
                "tax_unit_link": _ADULT_CARE_UNIT,
                "mutable_rows": "newly_imputed_expense_cells_only",
            },
        },
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


type TargetFamilies = Mapping[str, Mapping[str, Sequence[str]]]

# The production QRF surface is an ACS-transfer contract, not the release
# export-coverage contract.  Runtime-owned leaves (take-up draws, immigration,
# hours-before-LSR, and marketplace-plan inputs) are deliberately absent: the
# fiscal-refresh runtime seeds them after the raw donor/base-pool stage.  A raw
# donor therefore cannot provide a meaningful QRF target for those columns.
#
# Keep this declaration entity- and family-scoped so the donor-readiness gate,
# transfer, and post-transfer coverage audit all consume the exact same plan.
_DECLARED_ACS_TRANSFER_TARGET_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "person": {
        # The buildl-era donor excluded both partnership leaves because it
        # carried only combined partnership_income. The 23-stage base now
        # observes partnership_self_employment_net_earnings (restored below
        # via the default outputs), but s_corp_income remains an all-zero
        # schema column on the certified donor — combined income still
        # lives in partnership_income — so its exclusion stands until the
        # base disaggregates it.
        "puf_tax_itemization": tuple(
            target
            for target in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
            if target not in ACS_NATIVE_PERSON_INPUTS | {"s_corp_income"}
        ),
        # The Schedule D capital-gain-distributions memo leg is deliberately
        # NOT a QRF target: the base derives it deterministically as a
        # packaged share of positive long-term gains where the non-Schedule-D
        # route is absent (the reporting routes are mutually exclusive on a
        # real return), and an independent fit would break that accounting
        # identity. The transfer re-derives it from the transferred parents
        # instead — see :func:`derive_acs_schedule_d_capital_gain_distributions`.
        #
        # CDCC adult-care leg (#451 items 1-2): the qualifying-person flag
        # and the pre-subsidy care-expense carrier. Post-fit, the expense
        # surface is reconciled to the statute structure (one qualifying
        # carrier per tax unit) — see :func:`reconcile_acs_adult_care`.
        "adult_care": (
            "is_incapable_of_self_care",
            "pre_subsidy_care_expenses",
        ),
        "housing": _ADDITIONAL_PERSON_TRANSFER_TARGETS,
        "model_required_numeric": (
            "bank_account_assets",
            "bond_assets",
            "health_insurance_premiums_without_medicare_part_b",
            "hours_worked_last_week",
            "other_medical_expenses",
            "over_the_counter_health_expenses",
            "stock_assets",
            "tax_exempt_private_pension_income",
            "unemployment_compensation",
            "veterans_benefits",
        ),
        "model_required_boolean": (
            "has_champva_health_coverage_at_interview",
            "has_esi",
            "has_indian_health_service_coverage_at_interview",
            "has_marketplace_health_coverage_at_interview",
            "has_medicaid_health_coverage_at_interview",
            "has_non_marketplace_direct_purchase_health_coverage_at_interview",
            "has_other_means_tested_health_coverage_at_interview",
            "has_tricare_health_coverage_at_interview",
            "has_va_health_coverage_at_interview",
            "is_blind",
            "is_disabled",
            "is_full_time_college_student",
            "is_pregnant",
            # WICYN's adult-female reporter is only a physical carrier for an
            # SPM-unit receipt fact. Engine consumers are separately guarded
            # to aggregate receives_wic at SPM-unit grain (microcosm#591).
            "receives_wic",
        ),
        "model_required_discrete": ("own_children_in_household",),
    },
    "tax_unit": {
        # The 23-stage base carries the second-home mortgage legs as
        # all-zero schema columns (the mortgage surface consolidated onto
        # the first-home leaves), so they offer no donor signal; the
        # donor-coverage gate hard-fails an all-engine-default target.
        "puf_tax_itemization": tuple(
            target
            for target in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
            if target
            not in {
                "second_home_mortgage_balance",
                "second_home_mortgage_interest",
                "second_home_mortgage_origination_year",
            }
        ),
    },
    "spm_unit": {
        # Housing-assistance receipt is source-observed in the raw pool. Other
        # takes_up_* leaves are runtime-owned draws and are not donor targets.
        "benefit_participation": ("takes_up_housing_assistance_if_eligible",),
        # Reported TANF/SNAP receipt is produced from annual ASEC amounts before
        # cloning and transferred like the person-level required booleans.
        "model_required_boolean": ("is_tanf_enrolled", "receives_snap"),
        "model_required_numeric": ("spm_unit_pre_subsidy_childcare_expenses",),
    },
}


@dataclass(frozen=True)
class AcsTransferPattern:
    """One complete-case QRF used for an ACS availability pattern."""

    name: str
    observed_optional_predictors: tuple[str, ...]
    predictors: tuple[str, ...]
    seed: int
    weight_kind: str
    donor_rows: int
    recipient_rows: int
    #: Ordered ``(model_target, regime)`` pairs explicitly requested by the
    #: transfer owner and detected from this pattern's exact encoded donors.
    #: Ordinary callers opt out, preserving the legacy provenance surface.
    target_regimes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class AcsImputedInput:
    """Provenance for one input column imputed onto the ACS frame."""

    column: str
    entity: str
    family: str
    donor_spine: str
    donor_channel: str | None
    predictors: tuple[str, ...]
    seed: int
    weight_kind: str
    patterns: tuple[AcsTransferPattern, ...] = ()
    #: Number of recipient null cells this operation filled.
    imputed_recipient_rows: int = 0
    unmodeled_recipient_rows: int = 0
    #: Non-fit derivation kind for deterministic post-transfer columns
    #: (e.g. ``"split_component_by_share"``); ``None`` for QRF-fitted ones.
    derivation: str | None = None
    #: Post-fit structural reconciliation counts, when the column's surface
    #: was adjusted to a statute contract after prediction.
    reconciliation: Mapping[str, int] | None = None


@dataclass(frozen=True)
class AcsTransferResult:
    """An ACS frame plus immutable, non-export imputation provenance."""

    frame: Frame
    imputed_inputs: tuple[AcsImputedInput, ...] = ()
    fit_records: tuple[FitWeightRecord, ...] = ()
    deferred_inputs: tuple[str, ...] = ()
    resolved_donor_channel: str | None = None


@dataclass(frozen=True)
class AcsTransferBankPatternStep:
    """One availability pattern's state transition for a banked target."""

    pattern: str
    state_before: QRFChainState
    state_after: QRFChainState


@dataclass(frozen=True)
class AcsTransferTargetCheckpoint:
    """Raw draws and chain states for one ordered ACS model target."""

    target_index: int
    total_targets: int
    entity: str
    family: str
    family_targets: tuple[str, ...]
    model_targets: tuple[str, ...]
    model_target: str
    exported_targets: tuple[str, ...]
    raw_draw: np.ndarray
    pattern_steps: tuple[AcsTransferBankPatternStep, ...]

    def __post_init__(self) -> None:
        raw_draw = np.asarray(self.raw_draw)
        if raw_draw.ndim != 1 or raw_draw.dtype != np.dtype(np.float64):
            raise ValueError(
                "ACS transfer target checkpoints require a one-dimensional "
                "float64 raw draw."
            )
        raw_draw = np.ascontiguousarray(raw_draw)
        raw_draw.setflags(write=False)
        object.__setattr__(self, "raw_draw", raw_draw)


class AcsTransferTargetBank(Protocol):
    """Durable target store consumed by the checkpointed ACS transfer path."""

    def load_target(
        self,
        *,
        target_index: int,
        total_targets: int,
        entity: str,
        family: str,
        family_targets: tuple[str, ...],
        model_targets: tuple[str, ...],
        model_target: str,
        exported_targets: tuple[str, ...],
        recipient_rows: int,
        expected_states: Mapping[str, QRFChainState],
    ) -> AcsTransferTargetCheckpoint | None:
        """Load one valid target or return ``None`` so it is rebuilt."""

        ...

    def write_target(self, checkpoint: AcsTransferTargetCheckpoint) -> None:
        """Atomically persist one completed target."""

        ...


@dataclass(frozen=True)
class _FeatureSurface:
    donor: pd.DataFrame
    recipient: pd.DataFrame
    required: tuple[str, ...]
    optional: tuple[str, ...]


@dataclass(frozen=True)
class _FamilyFit:
    predictions: Mapping[str, np.ndarray]
    patterns: tuple[AcsTransferPattern, ...]
    fit_records: tuple[FitWeightRecord, ...]
    predictors: tuple[str, ...]
    weight_kind: str
    family_seed: int
    target_encodings: Mapping[str, _TargetEncoding]


@dataclass(frozen=True)
class _BankPatternContext:
    pattern: AcsTransferPattern
    recipient_positions: np.ndarray
    donor_mask: np.ndarray


@dataclass(frozen=True)
class _TargetEncoding:
    """A QRF-compatible representation with enough information to decode it."""

    model_values: pd.Series
    kind: str
    support: np.ndarray | None = None
    categories: tuple[Any, ...] = ()
    model_target: str = ""
    category_position: int | None = None


def required_acs_transfer_inputs() -> frozenset[str]:
    """Return every model leaf in the declared production transfer plan."""

    return frozenset(
        target
        for entity_families in _DECLARED_ACS_TRANSFER_TARGET_FAMILIES.values()
        for targets in entity_families.values()
        for target in targets
    )


#: Person columns the default transfer DERIVES deterministically after the
#: QRF fits (never fitted themselves). Coverage checks require them on the
#: recipient exactly like declared plan targets.
ACS_DERIVED_TRANSFER_INPUTS: tuple[str, ...] = (
    "schedule_d_capital_gain_distributions",
)


def acs_derived_transfer_expectations(
    target_families: TargetFamilies,
) -> dict[str, str]:
    """Derived columns a plan implies, mapped to their entity.

    The Schedule D CGD memo leg is derivable exactly when both of its
    parents are person targets of the plan — the same condition
    :func:`_apply_post_transfer_structure` derives under, so coverage
    expectations and production stay in lockstep (a custom test plan that
    never transfers the parents owes no derived column).
    """

    person_targets = {
        target
        for targets in target_families.get("person", {}).values()
        for target in targets
    }
    if {_SCHEDULE_D_CGD_SOURCE, _SCHEDULE_D_CGD_EXCLUSIVE_WITH} <= person_targets:
        return {_SCHEDULE_D_CGD_COLUMN: "person"}
    return {}


_SCHEDULE_D_CGD_COLUMN = "schedule_d_capital_gain_distributions"
_SCHEDULE_D_CGD_SOURCE = "long_term_capital_gains_before_response"
_SCHEDULE_D_CGD_EXCLUSIVE_WITH = "non_sch_d_capital_gains"

_ADULT_CARE_FLAG = "is_incapable_of_self_care"
_ADULT_CARE_EXPENSE = "pre_subsidy_care_expenses"
_ADULT_CARE_ROLE = "tax_unit_role_input"
_ADULT_CARE_UNIT = "person_tax_unit_id"


def derive_acs_schedule_d_capital_gain_distributions(
    person: pd.DataFrame,
    *,
    share: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Re-derive the Schedule D CGD memo leg from transferred parents.

    Mirrors the base's ``split_component_by_share`` operation exactly: the
    packaged SOCA share of positive long-term gains, eligible only where the
    mutually exclusive non-Schedule-D route is non-positive. Deterministic —
    an independent QRF fit of this leaf would break the route-exclusivity
    and share-of-parent identities the base construction guarantees.
    """

    from microcosm.build.us_runtime.capital_gain_distributions import (
        load_capital_gain_distribution_shares,
    )

    ratio = (
        float(
            load_capital_gain_distribution_shares().schedule_d_cgd_share_of_lt_net_gains
        )
        if share is None
        else float(share)
    )
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError(
            "Schedule D capital-gain-distribution share must be finite in (0, 1)."
        )
    source = pd.to_numeric(person[_SCHEDULE_D_CGD_SOURCE], errors="coerce")
    other_route = pd.to_numeric(person[_SCHEDULE_D_CGD_EXCLUSIVE_WITH], errors="coerce")
    if source.isna().any() or other_route.isna().any():
        raise ValueError(
            "Schedule D CGD derivation requires complete transferred "
            f"parents; {_SCHEDULE_D_CGD_SOURCE} has "
            f"{int(source.isna().sum())} and {_SCHEDULE_D_CGD_EXCLUSIVE_WITH} "
            f"has {int(other_route.isna().sum())} missing value(s)."
        )
    eligible = (source > 0.0) & (other_route <= 0.0)
    values = np.where(eligible.to_numpy(), ratio * source.to_numpy(), 0.0)
    provenance = {
        "share": ratio,
        "eligible_rows": int(eligible.sum()),
        "rows": int(len(person)),
        "derived_total": float(values.sum()),
    }
    return values.astype(np.float64), provenance


def acs_adult_care_qualifying_rows(person: pd.DataFrame) -> pd.Series:
    """Return the section 21 qualifying-person mask for ACS adult care.

    An incapable dependent qualifies directly.  An incapable tax-unit head or
    spouse qualifies only when the unit contains a spouse row.  Missing input
    columns fail closed so callers cannot silently broaden the carrier set.
    """

    required = {_ADULT_CARE_FLAG, _ADULT_CARE_ROLE, _ADULT_CARE_UNIT}
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(
            "ACS adult-care qualification is missing required person "
            f"columns: {missing}."
        )

    flag = person[_ADULT_CARE_FLAG].fillna(False).astype(bool)
    role = person[_ADULT_CARE_ROLE].astype(str)
    units = person[_ADULT_CARE_UNIT]
    is_dependent = role.eq("DEPENDENT")
    is_head = role.eq("HEAD")
    is_spouse = role.eq("SPOUSE")
    unit_married = is_spouse.groupby(units).transform("any")
    return flag & (is_dependent | ((is_head | is_spouse) & unit_married))


def reconcile_acs_adult_care(
    person: pd.DataFrame,
    *,
    mutable_rows: np.ndarray | pd.Series | None = None,
) -> tuple[pd.Series, dict[str, int]]:
    """Reconcile transferred adult-care expenses to the statute structure.

    The base's construction guarantees every expense carrier is a section 21
    qualifying person of its tax unit and each unit carries at most one
    carrier; independent person-grain predictions cannot. Deterministically:
    clear expenses on non-qualifying people, then keep only the largest
    carrier per unit (ties broken by row order).

    When ``mutable_rows`` is supplied, only those newly imputed expense cells
    may change. A pre-existing positive carrier blocks every mutable carrier
    in its unit, preserving raw values while preventing the transfer from
    introducing an additional carrier.
    """

    raw_expenses = pd.to_numeric(person[_ADULT_CARE_EXPENSE], errors="coerce")
    expenses = raw_expenses.fillna(0.0)
    if mutable_rows is None:
        mutable = pd.Series(True, index=person.index)
    else:
        mutable_array = np.asarray(mutable_rows, dtype=bool)
        if mutable_array.ndim != 1 or len(mutable_array) != len(person):
            raise ValueError(
                "mutable_rows must be a one-dimensional mask aligned to person."
            )
        mutable = pd.Series(mutable_array, index=person.index)
    units = person[_ADULT_CARE_UNIT]
    qualifying = acs_adult_care_qualifying_rows(person)

    mutable_positive = mutable & (expenses > 0.0)
    cleared_ineligible_mask = mutable_positive & ~qualifying
    cleared_ineligible = int(cleared_ineligible_mask.sum())

    candidates = mutable_positive & qualifying
    immutable_positive = ~mutable & (expenses > 0.0)
    unit_has_immutable = immutable_positive.groupby(units).transform("any")
    ranked = (-expenses.where(candidates, 0.0)).groupby(units).rank(method="first")
    keep = candidates & ~unit_has_immutable & ranked.eq(1.0)
    cleared_multi_carrier = int((candidates & ~keep).sum())

    reconciled = expenses.copy() if mutable_rows is None else raw_expenses.copy()
    reconciled.loc[keep] = expenses.loc[keep]
    reconciled.loc[mutable & ~keep] = 0.0

    counts = {
        "cleared_ineligible_carriers": cleared_ineligible,
        "cleared_multi_carrier_rows": cleared_multi_carrier,
        "remaining_carriers": int((reconciled > 0.0).sum()),
    }
    return reconciled, counts


def declared_acs_transfer_target_families() -> TargetFamilies:
    """Return the donor-independent production QRF target declaration.

    The nested tuples are immutable; fresh dictionaries prevent a caller from
    mutating the module-owned contract. Unlike
    :func:`default_acs_transfer_target_families`, this declaration retains
    absent targets so a preflight gate can fail closed before ACS acquisition.
    """

    return {
        entity: dict(entity_families)
        for entity, entity_families in _DECLARED_ACS_TRANSFER_TARGET_FAMILIES.items()
    }


def default_acs_transfer_target_families(donor: Frame) -> TargetFamilies:
    """Return declared production targets that this donor actually carries.

    Library callers using the implicit plan retain donor-observed behavior.
    The production staging builder instead uses the unfiltered declaration,
    gates it, and passes that exact plan explicitly to the transfer.
    """

    families: dict[str, dict[str, tuple[str, ...]]] = {}
    for entity, entity_families in declared_acs_transfer_target_families().items():
        if entity not in donor.entities:
            continue
        columns = donor.table(entity).columns
        for family, declared_targets in entity_families.items():
            targets = tuple(target for target in declared_targets if target in columns)
            if targets:
                families.setdefault(entity, {})[family] = targets
    return families


def acs_transfer_donor_requirements(
    donor: Frame,
    target_families: TargetFamilies,
) -> dict[str, tuple[str, ...]]:
    """Resolve raw donor columns consumed by one exact QRF transfer plan.

    Required person/state predictors and every entity-scoped target are hard
    requirements. Raw sources for donor-available optional predictors are also
    included because the production ACS recipient exposes their counterparts,
    so QRF availability patterns consume them. Housing transfer additionally
    requires the concrete head and tenure sources that it promotes to mandatory.
    """

    required: dict[str, set[str]] = {
        donor.schema.person_entity: {"age", "is_female"},
    }
    state_owner = _column_owner_or_none(donor, "state_fips")
    if state_owner is None:
        raise ValueError("required donor feature 'state_fips' is absent")
    required.setdefault(state_owner, set()).add("state_fips")

    target_names: set[str] = set()
    for entity, entity_families in target_families.items():
        entity_required = required.setdefault(entity, set())
        for targets in entity_families.values():
            entity_required.update(targets)
            target_names.update(targets)

    person = donor.table(donor.schema.person_entity)
    for source in (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
    ):
        if source in person.columns:
            required[donor.schema.person_entity].add(source)
    for components in _DONOR_COMBINED_COMPONENTS.values():
        if all(component in person.columns for component in components):
            required[donor.schema.person_entity].update(components)

    housing = bool(_HOUSING_TRANSFER_TARGETS.intersection(target_names))
    head_source = next(
        (
            source
            for source in (
                "is_household_head",
                "A_EXPRRP",
                "A_LINENO",
            )
            if source in person.columns
        ),
        None,
    )
    if head_source is not None:
        required[donor.schema.person_entity].add(head_source)
    elif housing:
        raise ValueError(
            "pre_subsidy_rent target has no donor household-head feature source"
        )

    tenure_source = next(
        (
            (owner, source)
            for source in (
                "tenure_type",
                "spm_unit_tenure_type",
            )
            if (owner := _column_owner_or_none(donor, source)) is not None
        ),
        None,
    )
    if tenure_source is not None:
        owner, source = tenure_source
        required.setdefault(owner, set()).add(source)
    elif housing:
        raise ValueError(
            "pre_subsidy_rent target has no donor housing-tenure feature source"
        )

    return {
        entity: tuple(sorted(columns)) for entity, columns in sorted(required.items())
    }


def assert_acs_transfer_targets_are_input_leaves(
    targets: Iterable[str],
    *,
    engine: Any | None = None,
    require_known: bool = False,
) -> None:
    """Reject transfer targets the engine owns or, optionally, does not know.

    The formula-owned branch is the production transfer check. Keeping it in
    one helper lets the committed full-plan sweep exercise the exact same
    classifier and error path before a Modal build starts. Generic library
    callers may use non-engine fixture columns, so strict engine membership is
    opt-in for the production-plan sweep.
    """

    target_names = set(targets)
    formula_owned = sorted(resolve_formula_owned_outputs(target_names, engine=engine))
    if formula_owned:
        raise ValueError(
            "ACS transfer targets must be PolicyEngine input leaves, not "
            f"formula-owned outputs: {formula_owned}."
        )
    if not require_known:
        return

    try:
        if engine is None:
            from microcosm.frame.adapters.policyengine_us import (
                PolicyEngineUSVariableMetadataIndex,
            )

            engine = PolicyEngineUSVariableMetadataIndex()
        engine_leaves = set(engine.variables())
    except ImportError as exc:
        raise RuntimeError(
            "Strict ACS transfer leaf classification requires policyengine-us; "
            "install the microcosm-build US extra."
        ) from exc
    unknown = sorted(target_names - engine_leaves)
    if unknown:
        raise ValueError(
            "ACS transfer targets must be known PolicyEngine input leaves; "
            f"unknown or non-leaf targets: {unknown}."
        )


def transfer_acs_inputs(
    recipient: Frame,
    donor: Frame,
    *,
    target_families: TargetFamilies | None = None,
    donor_spine: str = ASEC_PUF_DONOR_SPINE,
    donor_channel: str | None = ACS_DONOR_CHANNEL_AUTO,
    seed: int = 0,
    n_estimators: int = 100,
    max_targets_per_fit: int = DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    target_bank: AcsTransferTargetBank | None = None,
    derive_schedule_d: bool = True,
    execution_contract: Mapping[str, object] | None = None,
    regime_evidence_targets: Iterable[tuple[str, str]] = (),
) -> AcsTransferResult:
    """Impute requested missing leaves from ``donor`` onto ``recipient``.

    ``target_families`` is an ``entity -> family -> targets`` mapping. Each
    family is split into recipient optional-predictor availability patterns.
    Every pattern gets a separately seeded QRF fit on donor rows complete for
    every family target and finite for exactly the predictors that pattern
    observes. Existing non-null target cells remain unchanged.

    Families wider than ``max_targets_per_fit`` are deterministically split
    into bounded chained-QRF batches so fitted forests cannot grow linearly
    across the full model-input inventory. Joint categorical codecs remain in
    one batch even when their two exported columns cross the nominal limit.

    ``donor_channel="auto"`` is the safe default: it selects the
    ``puf_tax_detail`` clone role when support-role metadata exists and
    otherwise uses an unchanneled donor. Source-spine channel labels never
    determine fit eligibility. Pass ``None`` only to deliberately fit every
    donor row. The resolved role, patterns, seeds, row counts, and weight kind
    are recorded outside the returned frame.

    When ``target_bank`` is provided, every availability-pattern QRF uses the
    target-at-a-time chain API. Raw draws and advanced RNG states are banked
    after each ordered model target, so a retry can continue without changing
    the monolithic chained-QRF result. The ordinary in-memory fit remains the
    default for library callers that do not request durable banking.

    ``regime_evidence_targets`` is an explicit ``(entity, target)`` audit
    selection. Only those targets incur donor-regime detection, fitted-result
    verification, and pattern provenance. The default is empty so an owner
    cannot accidentally broaden every transfer's runtime or receipt contract.
    """

    _validate_frames(recipient, donor)
    _validate_fit_options(
        seed=seed,
        n_estimators=n_estimators,
        max_targets_per_fit=max_targets_per_fit,
    )
    default_plan = target_families is None or (
        target_families == declared_acs_transfer_target_families()
    )
    deferred_inputs = (
        tuple(sorted(ACS_DEFERRED_GEOGRAPHY_INPUTS)) if default_plan else ()
    )
    requested = _normalize_target_families(
        target_families
        if target_families is not None
        else default_acs_transfer_target_families(donor),
        recipient=recipient,
    )
    requested = _split_large_target_families(
        requested,
        max_targets_per_fit=max_targets_per_fit,
    )
    requested_target_keys = frozenset(
        (entity, target) for entity, _family, targets in requested for target in targets
    )
    selected_regime_evidence = _normalize_regime_evidence_targets(
        regime_evidence_targets,
        requested=requested_target_keys,
    )
    if not requested:
        return AcsTransferResult(
            frame=canonicalize_frame_string_dtypes(
                recipient,
                boundary="ACS transfer result",
            ),
            deferred_inputs=deferred_inputs,
        )

    all_targets = [
        target for _entity, _family, targets in requested for target in targets
    ]
    resolved_execution_contract = acs_transfer_execution_contract_identity(
        targets=all_targets,
        derive_schedule_d=derive_schedule_d,
    )
    if execution_contract is not None and dict(execution_contract) != (
        resolved_execution_contract
    ):
        raise ValueError(
            "ACS transfer runtime execution contract differs from its bound input."
        )
    assert_acs_transfer_targets_are_input_leaves(all_targets)

    active = _missing_target_families(requested, recipient=recipient)
    if not active:
        return AcsTransferResult(
            frame=canonicalize_frame_string_dtypes(
                recipient,
                boundary="ACS transfer result",
            ),
            deferred_inputs=deferred_inputs,
        )

    _validate_donor_source(donor_spine=donor_spine, donor_channel=donor_channel)
    fit_donor, resolved_channel = resolve_acs_donor_channel(donor, donor_channel)
    output_tables = {
        entity: recipient.table(entity).copy() for entity in recipient.entities
    }
    provenance: list[AcsImputedInput] = []
    fit_records: list[FitWeightRecord] = []
    imputed_masks: dict[tuple[str, str], np.ndarray] = {}
    ordered_bank_targets = [
        (entity, family, model_target)
        for entity, family, targets in requested
        for model_target in _model_target_names(targets)
    ]
    bank_target_indexes = {key: index for index, key in enumerate(ordered_bank_targets)}

    for entity, family, targets in active:
        family_regime_evidence_targets = tuple(
            target for target in targets if (entity, target) in selected_regime_evidence
        )
        recipient_table = recipient.table(entity)
        target_missing = {
            target: (
                recipient_table[target].isna().to_numpy(dtype=bool)
                if target in recipient_table.columns
                else np.ones(len(recipient_table), dtype=bool)
            )
            for target in targets
        }
        if target_bank is None:
            fitted = _fit_family_patterns(
                fit_donor,
                recipient,
                entity=entity,
                family=family,
                targets=targets,
                target_missing=target_missing,
                seed=seed,
                n_estimators=n_estimators,
                regime_evidence_targets=family_regime_evidence_targets,
            )
        else:
            fitted = _fit_family_patterns_banked(
                fit_donor,
                recipient,
                entity=entity,
                family=family,
                targets=targets,
                target_missing=target_missing,
                seed=seed,
                n_estimators=n_estimators,
                target_bank=target_bank,
                target_indexes={
                    model_target: bank_target_indexes[(entity, family, model_target)]
                    for model_target in _model_target_names(targets)
                },
                total_targets=len(ordered_bank_targets),
                regime_evidence_targets=family_regime_evidence_targets,
            )
        patterns_without_regimes = (
            tuple(replace(pattern, target_regimes=()) for pattern in fitted.patterns)
            if family_regime_evidence_targets
            else fitted.patterns
        )
        for target in targets:
            predicted = _prediction_values(
                fitted.predictions[target],
                encoding=fitted.target_encodings[target],
                entity=entity,
                target=target,
            )
            merged, imputed = _fill_recipient_nulls(
                output_tables[entity],
                target=target,
                predicted=predicted,
            )
            output_tables[entity][target] = merged
            imputed_masks[(entity, target)] = imputed
            missing_rows = target_missing[target]
            if not missing_rows.any():
                continue
            provenance.append(
                AcsImputedInput(
                    column=target,
                    entity=entity,
                    family=family,
                    donor_spine=donor_spine,
                    donor_channel=resolved_channel,
                    predictors=fitted.predictors,
                    seed=fitted.family_seed,
                    weight_kind=fitted.weight_kind,
                    patterns=(
                        fitted.patterns
                        if target in family_regime_evidence_targets
                        else patterns_without_regimes
                    ),
                    imputed_recipient_rows=int(imputed.sum()),
                    unmodeled_recipient_rows=int((missing_rows & ~imputed).sum()),
                )
            )
        fit_records.extend(fitted.fit_records)

    _apply_post_transfer_structure(
        output_tables,
        provenance,
        imputed_masks=imputed_masks,
        donor_spine=donor_spine,
        resolved_channel=resolved_channel,
        execution_contract=resolved_execution_contract,
    )

    tables: dict[str, pd.DataFrame] = dict(output_tables)
    tables.update({name: recipient.link(name) for name in recipient.links})
    frame = Frame(
        tables,
        recipient.schema,
        {
            entity: recipient.weights_for(entity)
            for entity in recipient.weighted_entities
        },
        recipient.strata,
        mass_log=recipient.mass_log,
        metadata=recipient.metadata,
    )
    frame = canonicalize_frame_string_dtypes(
        frame,
        boundary="ACS transfer result",
        in_place=True,
    )
    return AcsTransferResult(
        frame=frame,
        imputed_inputs=tuple(provenance),
        fit_records=tuple(fit_records),
        deferred_inputs=deferred_inputs,
        resolved_donor_channel=resolved_channel,
    )


def _apply_post_transfer_structure(
    output_tables: dict[str, pd.DataFrame],
    provenance: list[AcsImputedInput],
    *,
    imputed_masks: Mapping[tuple[str, str], np.ndarray],
    donor_spine: str,
    resolved_channel: str | None,
    execution_contract: Mapping[str, object],
) -> None:
    """Apply the deterministic post-fit steps the base's construction implies.

    Both steps key off cells THIS transfer filled, so custom test plans that
    never touch these families are unaffected:

    - The Schedule D CGD memo leg is derived from the two transferred
      capital-gain parents at the packaged share with route exclusivity.
    - The adult-care expense surface is reconciled to the statute structure
      (qualifying carriers only, at most one per tax unit).
    """

    person = output_tables.get("person")
    if person is None:
        return

    post_transfer_contract = execution_contract["post_transfer_structure"]
    assert isinstance(post_transfer_contract, Mapping)
    schedule_d_contract = post_transfer_contract[
        "schedule_d_capital_gain_distributions"
    ]
    assert isinstance(schedule_d_contract, Mapping)
    cgd_parents = {_SCHEDULE_D_CGD_SOURCE, _SCHEDULE_D_CGD_EXCLUSIVE_WITH}
    cgd_candidate = np.zeros(len(person), dtype=bool)
    for parent in cgd_parents:
        cgd_candidate |= imputed_masks.get(
            ("person", parent),
            np.zeros(len(person), dtype=bool),
        )
    if (
        schedule_d_contract["enabled"] is True
        and cgd_candidate.any()
        and cgd_parents <= set(person.columns)
    ):
        source = pd.to_numeric(person[_SCHEDULE_D_CGD_SOURCE], errors="coerce")
        other_route = pd.to_numeric(
            person[_SCHEDULE_D_CGD_EXCLUSIVE_WITH],
            errors="coerce",
        )
        derivable = (
            cgd_candidate
            & np.isfinite(source.to_numpy(dtype=np.float64))
            & np.isfinite(other_route.to_numpy(dtype=np.float64))
        )
        if _SCHEDULE_D_CGD_COLUMN in person.columns:
            derived_output = person[_SCHEDULE_D_CGD_COLUMN].copy()
        else:
            derived_output = pd.Series(
                np.nan,
                index=person.index,
                name=_SCHEDULE_D_CGD_COLUMN,
            )
        fill = derivable & derived_output.isna().to_numpy(dtype=bool)
        if not fill.any():
            derivation: dict[str, object] = {}
        else:
            values, derivation = derive_acs_schedule_d_capital_gain_distributions(
                person.loc[fill],
                share=float(
                    schedule_d_contract["share_asset"][
                        "schedule_d_cgd_share_of_lt_net_gains"
                    ]
                ),
            )
            derived_output.loc[fill] = values
            person[_SCHEDULE_D_CGD_COLUMN] = derived_output
        if fill.any():
            provenance.append(
                AcsImputedInput(
                    column=_SCHEDULE_D_CGD_COLUMN,
                    entity="person",
                    family="capital_gain_details",
                    donor_spine=donor_spine,
                    donor_channel=resolved_channel,
                    predictors=(
                        _SCHEDULE_D_CGD_SOURCE,
                        _SCHEDULE_D_CGD_EXCLUSIVE_WITH,
                    ),
                    seed=0,
                    weight_kind="deterministic",
                    imputed_recipient_rows=int(fill.sum()),
                    unmodeled_recipient_rows=int(
                        (
                            cgd_candidate
                            & derived_output.isna().to_numpy(dtype=bool)
                            & ~derivable
                        ).sum()
                    ),
                    derivation="split_component_by_share",
                    reconciliation={
                        key: int(value)
                        for key, value in derivation.items()
                        if isinstance(value, int)
                    },
                )
            )

    adult_care_expense_mask = imputed_masks.get(("person", _ADULT_CARE_EXPENSE))
    if (
        adult_care_expense_mask is not None
        and adult_care_expense_mask.any()
        and _ADULT_CARE_FLAG in person.columns
    ):
        structural = {_ADULT_CARE_ROLE, _ADULT_CARE_UNIT}
        missing = sorted(structural - set(person.columns))
        if missing:
            raise ValueError(
                "Adult-care reconciliation requires the recipient structural "
                f"column(s) {missing}; refusing to ship an unreconciled "
                "expense surface."
            )
        expenses, counts = reconcile_acs_adult_care(
            person,
            mutable_rows=adult_care_expense_mask,
        )
        person[_ADULT_CARE_EXPENSE] = expenses
        for index, item in enumerate(provenance):
            if item.column == _ADULT_CARE_EXPENSE and item.entity == "person":
                provenance[index] = replace(item, reconciliation=counts)
                break


def _model_target_names(targets: Sequence[str]) -> tuple[str, ...]:
    """Return the exact chained-QRF target order for exported leaves."""

    requested = set(targets)
    immigration_pair = set(_IMMIGRATION_STATUS_TARGETS)
    model_targets: list[str] = []
    joint_added = False
    for target in targets:
        if target in immigration_pair and immigration_pair.issubset(requested):
            if not joint_added:
                model_targets.append(_IMMIGRATION_STATUS_MODEL_TARGET)
                joint_added = True
            continue
        model_targets.append(target)
    return tuple(model_targets)


def _selected_model_target_names(
    family_targets: Sequence[str],
    selected_targets: Sequence[str],
) -> tuple[str, ...]:
    """Resolve an exported-target subset against the full family codec."""

    family = tuple(family_targets)
    family_set = set(family)
    selected = set(selected_targets)
    unknown = sorted(selected - family_set)
    if unknown:
        raise ValueError(
            f"Selected ACS regime-evidence targets are outside the family: {unknown}."
        )
    immigration_pair = set(_IMMIGRATION_STATUS_TARGETS)
    selected_model_targets = {
        (
            _IMMIGRATION_STATUS_MODEL_TARGET
            if target in immigration_pair and immigration_pair.issubset(family_set)
            else target
        )
        for target in selected
    }
    return tuple(
        target
        for target in _model_target_names(family)
        if target in selected_model_targets
    )


def _regime_evidence_model_targets(
    *,
    model_targets: Sequence[str],
    target_encodings: Mapping[str, _TargetEncoding],
    regime_evidence_targets: Sequence[str],
) -> tuple[str, ...]:
    """Project selected exported leaves onto the fitted model-target order."""

    selected = {
        target_encodings[target].model_target for target in regime_evidence_targets
    }
    return tuple(target for target in model_targets if target in selected)


def _model_target_regimes(
    model_frame: Frame,
    *,
    entity: str,
    model_targets: Sequence[str],
    zero_atol: float,
) -> tuple[tuple[str, str], ...]:
    """Detect ordered regimes on the exact encoded donor fit surface."""

    table = model_frame.table(entity)
    return tuple(
        (
            target,
            detect_regime(
                table[target].to_numpy(dtype=np.float64),
                zero_atol=zero_atol,
            ),
        )
        for target in model_targets
    )


def _verify_fitted_target_regimes(
    fitted: object,
    *,
    expected: tuple[tuple[str, str], ...],
    entity: str,
    family: str,
    pattern: str,
) -> None:
    """Verify a fitted QRF's reported regimes when its API exposes them."""

    if not expected:
        return
    regimes = getattr(fitted, "regimes", None)
    if not callable(regimes):
        # Lightweight test doubles need only implement the fit/predict surface.
        return
    reported = regimes()
    if not isinstance(reported, Mapping):
        raise TypeError(
            f"ACS transfer {entity!r}/{family!r}/{pattern!r} QRF regimes "
            "must be a mapping."
        )
    expected_targets = tuple(target for target, _regime in expected)
    actual = tuple((target, reported.get(target)) for target in expected_targets)
    if any(target not in reported for target in expected_targets) or actual != expected:
        raise RuntimeError(
            f"ACS transfer {entity!r}/{family!r}/{pattern!r} QRF reported "
            f"regimes {actual!r}, expected exact donor-support regimes "
            f"{expected!r}."
        )


def _verify_chain_target_regime(
    result: object,
    *,
    expected: str,
    entity: str,
    family: str,
    pattern: str,
    model_target: str,
) -> None:
    """Verify one targetwise chain result when it exposes regime evidence."""

    reported = getattr(result, "regime", None)
    if reported is None:
        # Lightweight bank-path test doubles may omit QRF diagnostics.
        return
    if reported != expected:
        raise RuntimeError(
            f"ACS transfer {entity!r}/{family!r}/{pattern!r} target "
            f"{model_target!r} reported regime {reported!r}, expected exact "
            f"donor-support regime {expected!r}."
        )


def _fit_family_patterns(
    donor: Frame,
    recipient: Frame,
    *,
    entity: str,
    family: str,
    targets: tuple[str, ...],
    target_missing: Mapping[str, np.ndarray],
    seed: int,
    n_estimators: int,
    regime_evidence_targets: tuple[str, ...],
) -> _FamilyFit:
    _validate_donor_targets(donor, entity=entity, targets=targets)
    donor_table = donor.table(entity)
    target_complete = _complete_target_mask(donor_table, targets=targets)
    if not target_complete.any():
        raise ValueError(
            f"ACS transfer family {family!r} on entity {entity!r} has no "
            f"donor rows complete for every target {list(targets)}."
        )
    target_encodings = _complete_case_target_encodings(
        donor_table,
        targets=targets,
        complete=target_complete,
    )
    model_targets = _model_target_names(targets)
    evidence_model_targets = _regime_evidence_model_targets(
        model_targets=model_targets,
        target_encodings=target_encodings,
        regime_evidence_targets=regime_evidence_targets,
    )
    surface = _transfer_feature_surface(
        donor,
        recipient,
        entity=entity,
        targets=targets,
    )
    overlap = sorted(set(surface.required).intersection(targets))
    if overlap:
        raise ValueError(
            f"ACS transfer family {family!r} on entity {entity!r} has target(s) "
            f"that are required predictors: {overlap}."
        )

    needs_prediction = np.logical_or.reduce(
        [np.asarray(target_missing[target], dtype=bool) for target in targets]
    )
    eligible = (
        _complete_predictor_mask(
            surface.recipient,
            predictors=surface.required,
        )
        & needs_prediction
    )
    if not eligible.any():
        raise ValueError(
            f"ACS transfer family {family!r} on entity {entity!r} has no "
            f"recipient rows with required predictors {list(surface.required)}."
        )

    n_recipient = len(surface.recipient)
    predictions = {
        target: np.full(n_recipient, np.nan, dtype=np.float64) for target in targets
    }
    pattern_records: list[AcsTransferPattern] = []
    fit_records: list[FitWeightRecord] = []
    family_seed = _family_seed(seed, entity=entity, family=family)

    for position, (observed_optional, recipient_positions) in enumerate(
        _availability_patterns(surface, eligible=eligible)
    ):
        predictors = (*surface.required, *observed_optional)
        donor_mask = (
            _complete_predictor_mask(surface.donor, predictors=predictors)
            & target_complete
        )
        donor_rows = int(donor_mask.sum())
        if donor_rows == 0:
            raise ValueError(
                f"ACS transfer family {family!r} on entity {entity!r} has no "
                f"donor rows finite for availability pattern "
                f"{list(observed_optional)} and predictors {list(predictors)}."
            )

        pattern_name = _pattern_name(position, observed_optional)
        pattern_seed = _pattern_seed(
            seed,
            entity=entity,
            family=family,
            observed_optional=observed_optional,
        )
        model_frame = _model_frame(
            donor,
            entity=entity,
            features=surface.donor,
            predictors=predictors,
            target_encodings=target_encodings,
            mask=donor_mask,
        )
        resolved_kind = model_frame.resolve_weights(entity).kind.value
        model = _qrf()(n_estimators=n_estimators, seed=pattern_seed)
        target_regimes = (
            _model_target_regimes(
                model_frame,
                entity=entity,
                model_targets=evidence_model_targets,
                zero_atol=float(getattr(model, "zero_atol", DEFAULT_ZERO_ATOL)),
            )
            if evidence_model_targets
            else ()
        )
        fitted = model.fit(
            model_frame,
            list(predictors),
            list(model_targets),
            weights=resolved_kind,
        )
        if fitted.weight_kind != resolved_kind:
            raise RuntimeError(
                f"ACS transfer {entity!r}/{family!r}/{pattern_name!r} resolved "
                f"weight kind {fitted.weight_kind!r}, expected the donor "
                f"Frame's {resolved_kind!r}."
            )
        if target_regimes:
            _verify_fitted_target_regimes(
                fitted,
                expected=target_regimes,
                entity=entity,
                family=family,
                pattern=pattern_name,
            )

        recipient_pattern = _encoded_predictor_frame(
            surface.recipient.iloc[recipient_positions],
            predictors=predictors,
        )
        drawn = fitted.predict(recipient_pattern)
        _validate_prediction_values(
            drawn,
            entity=entity,
            family=family,
            pattern=pattern_name,
        )
        for target in targets:
            model_target = target_encodings[target].model_target
            predictions[target][recipient_positions] = drawn[model_target].to_numpy(
                dtype=np.float64
            )

        pattern_record = AcsTransferPattern(
            name=pattern_name,
            observed_optional_predictors=observed_optional,
            predictors=predictors,
            seed=pattern_seed,
            weight_kind=fitted.weight_kind,
            donor_rows=donor_rows,
            recipient_rows=len(recipient_positions),
            target_regimes=target_regimes,
        )
        pattern_records.append(pattern_record)
        fit_records.append(
            FitWeightRecord(
                f"acs_transfer:{entity}:{family}:{pattern_name}",
                fitted.weight_kind,
            )
        )

    patterns = tuple(pattern_records)
    kinds = {pattern.weight_kind for pattern in patterns}
    if len(kinds) != 1:  # pragma: no cover - every subset resolves one donor kind
        raise RuntimeError(
            f"ACS transfer family {family!r} resolved mixed weight kinds: "
            f"{sorted(kinds)}."
        )
    used_predictors = tuple(
        predictor
        for predictor in (*surface.required, *surface.optional)
        if any(predictor in pattern.predictors for pattern in patterns)
    )
    return _FamilyFit(
        predictions=predictions,
        patterns=patterns,
        fit_records=tuple(fit_records),
        predictors=used_predictors,
        weight_kind=next(iter(kinds)),
        family_seed=family_seed,
        target_encodings=target_encodings,
    )


def _fit_family_patterns_banked(
    donor: Frame,
    recipient: Frame,
    *,
    entity: str,
    family: str,
    targets: tuple[str, ...],
    target_missing: Mapping[str, np.ndarray],
    seed: int,
    n_estimators: int,
    target_bank: AcsTransferTargetBank,
    target_indexes: Mapping[str, int],
    total_targets: int,
    regime_evidence_targets: tuple[str, ...],
) -> _FamilyFit:
    """Fit one family targetwise, resuming exact raw chained draws."""

    _validate_donor_targets(donor, entity=entity, targets=targets)
    donor_table = donor.table(entity)
    target_complete = _complete_target_mask(donor_table, targets=targets)
    if not target_complete.any():
        raise ValueError(
            f"ACS transfer family {family!r} on entity {entity!r} has no "
            f"donor rows complete for every target {list(targets)}."
        )
    target_encodings = _complete_case_target_encodings(
        donor_table,
        targets=targets,
        complete=target_complete,
    )
    model_targets = _model_target_names(targets)
    encoded_model_targets = tuple(
        dict.fromkeys(encoding.model_target for encoding in target_encodings.values())
    )
    if model_targets != encoded_model_targets:  # pragma: no cover - codec invariant
        raise AssertionError(
            "ACS transfer model-target ordering changed during encode."
        )
    evidence_model_targets = _regime_evidence_model_targets(
        model_targets=model_targets,
        target_encodings=target_encodings,
        regime_evidence_targets=regime_evidence_targets,
    )

    surface = _transfer_feature_surface(
        donor,
        recipient,
        entity=entity,
        targets=targets,
    )
    overlap = sorted(set(surface.required).intersection(targets))
    if overlap:
        raise ValueError(
            f"ACS transfer family {family!r} on entity {entity!r} has target(s) "
            f"that are required predictors: {overlap}."
        )
    needs_prediction = np.logical_or.reduce(
        [np.asarray(target_missing[target], dtype=bool) for target in targets]
    )
    eligible = (
        _complete_predictor_mask(
            surface.recipient,
            predictors=surface.required,
        )
        & needs_prediction
    )
    if not eligible.any():
        raise ValueError(
            f"ACS transfer family {family!r} on entity {entity!r} has no "
            f"recipient rows with required predictors {list(surface.required)}."
        )

    contexts: list[_BankPatternContext] = []
    states: dict[str, QRFChainState] = {}
    raw_priors: dict[str, pd.DataFrame] = {}
    pattern_records: list[AcsTransferPattern] = []
    fit_records: list[FitWeightRecord] = []
    for position, (observed_optional, recipient_positions) in enumerate(
        _availability_patterns(surface, eligible=eligible)
    ):
        predictors = (*surface.required, *observed_optional)
        donor_mask = (
            _complete_predictor_mask(surface.donor, predictors=predictors)
            & target_complete
        )
        donor_rows = int(donor_mask.sum())
        if donor_rows == 0:
            raise ValueError(
                f"ACS transfer family {family!r} on entity {entity!r} has no "
                f"donor rows finite for availability pattern "
                f"{list(observed_optional)} and predictors {list(predictors)}."
            )
        pattern_name = _pattern_name(position, observed_optional)
        pattern_seed = _pattern_seed(
            seed,
            entity=entity,
            family=family,
            observed_optional=observed_optional,
        )
        model_frame = _model_frame(
            donor,
            entity=entity,
            features=surface.donor,
            predictors=predictors,
            target_encodings=target_encodings,
            mask=donor_mask,
        )
        resolved_kind = model_frame.resolve_weights(entity).kind.value
        model = _qrf()(n_estimators=n_estimators, seed=pattern_seed)
        target_regimes = (
            _model_target_regimes(
                model_frame,
                entity=entity,
                model_targets=evidence_model_targets,
                zero_atol=float(getattr(model, "zero_atol", DEFAULT_ZERO_ATOL)),
            )
            if evidence_model_targets
            else ()
        )
        if not hasattr(model, "start_chain") or not hasattr(model, "fit_draw_next"):
            raise TypeError(
                "Banked ACS transfer requires a QRF with start_chain and "
                "fit_draw_next support."
            )
        state = model.start_chain(
            model_frame,
            list(predictors),
            list(model_targets),
            weights=resolved_kind,
        )
        if state.weight_kind != resolved_kind:
            raise RuntimeError(
                f"ACS transfer {entity!r}/{family!r}/{pattern_name!r} resolved "
                f"weight kind {state.weight_kind!r}, expected the donor "
                f"Frame's {resolved_kind!r}."
            )
        pattern = AcsTransferPattern(
            name=pattern_name,
            observed_optional_predictors=observed_optional,
            predictors=predictors,
            seed=pattern_seed,
            weight_kind=state.weight_kind,
            donor_rows=donor_rows,
            recipient_rows=len(recipient_positions),
            target_regimes=target_regimes,
        )
        contexts.append(
            _BankPatternContext(
                pattern=pattern,
                recipient_positions=recipient_positions,
                donor_mask=donor_mask,
            )
        )
        states[pattern_name] = state
        raw_priors[pattern_name] = pd.DataFrame(
            index=surface.recipient.iloc[recipient_positions].index
        )
        pattern_records.append(pattern)
        fit_records.append(
            FitWeightRecord(
                f"acs_transfer:{entity}:{family}:{pattern_name}",
                state.weight_kind,
            )
        )

    n_recipient = len(surface.recipient)
    raw_by_model_target: dict[str, np.ndarray] = {}
    for model_target in model_targets:
        exported_targets = tuple(
            target
            for target in targets
            if target_encodings[target].model_target == model_target
        )
        expected_states = dict(states)
        checkpoint = target_bank.load_target(
            target_index=target_indexes[model_target],
            total_targets=total_targets,
            entity=entity,
            family=family,
            family_targets=targets,
            model_targets=model_targets,
            model_target=model_target,
            exported_targets=exported_targets,
            recipient_rows=n_recipient,
            expected_states=expected_states,
        )
        if checkpoint is None:
            raw_draw = np.full(n_recipient, np.nan, dtype=np.float64)
            steps: list[AcsTransferBankPatternStep] = []
            for context in contexts:
                pattern = context.pattern
                state_before = states[pattern.name]
                model_frame = _model_frame(
                    donor,
                    entity=entity,
                    features=surface.donor,
                    predictors=pattern.predictors,
                    target_encodings=target_encodings,
                    mask=context.donor_mask,
                )
                recipient_pattern = _encoded_predictor_frame(
                    surface.recipient.iloc[context.recipient_positions],
                    predictors=pattern.predictors,
                )
                result = _qrf()(
                    n_estimators=n_estimators,
                    seed=pattern.seed,
                ).fit_draw_next(
                    model_frame,
                    recipient_pattern,
                    raw_priors[pattern.name],
                    state=state_before,
                    weights=pattern.weight_kind,
                )
                if result.target != model_target:
                    raise AssertionError(
                        f"ACS transfer chain returned {result.target!r}, "
                        f"expected {model_target!r}."
                    )
                if result.weight_kind != pattern.weight_kind:
                    raise RuntimeError(
                        f"ACS transfer {entity!r}/{family!r}/{pattern.name!r} "
                        f"resolved weight kind {result.weight_kind!r}, expected "
                        f"{pattern.weight_kind!r}."
                    )
                expected_regime = dict(pattern.target_regimes).get(model_target)
                if expected_regime is not None:
                    _verify_chain_target_regime(
                        result,
                        expected=expected_regime,
                        entity=entity,
                        family=family,
                        pattern=pattern.name,
                        model_target=model_target,
                    )
                raw_draw[context.recipient_positions] = result.raw_draw
                _validate_prediction_values(
                    pd.DataFrame(
                        {model_target: result.raw_draw},
                        index=recipient_pattern.index,
                    ),
                    entity=entity,
                    family=family,
                    pattern=pattern.name,
                )
                steps.append(
                    AcsTransferBankPatternStep(
                        pattern=pattern.name,
                        state_before=state_before,
                        state_after=result.state,
                    )
                )
            checkpoint = AcsTransferTargetCheckpoint(
                target_index=target_indexes[model_target],
                total_targets=total_targets,
                entity=entity,
                family=family,
                family_targets=targets,
                model_targets=model_targets,
                model_target=model_target,
                exported_targets=exported_targets,
                raw_draw=raw_draw,
                pattern_steps=tuple(steps),
            )
            target_bank.write_target(checkpoint)

        _validate_banked_target_checkpoint(
            checkpoint,
            target_index=target_indexes[model_target],
            total_targets=total_targets,
            entity=entity,
            family=family,
            family_targets=targets,
            model_targets=model_targets,
            model_target=model_target,
            exported_targets=exported_targets,
            recipient_rows=n_recipient,
            expected_states=expected_states,
            expected_patterns=tuple(context.pattern.name for context in contexts),
        )
        steps_by_pattern = {step.pattern: step for step in checkpoint.pattern_steps}
        raw_by_model_target[model_target] = checkpoint.raw_draw
        for context in contexts:
            pattern_name = context.pattern.name
            raw_priors[pattern_name][model_target] = checkpoint.raw_draw[
                context.recipient_positions
            ]
            states[pattern_name] = steps_by_pattern[pattern_name].state_after

    incomplete_states = [
        name for name, state in states.items() if not state.is_complete
    ]
    if incomplete_states:  # pragma: no cover - one step per declared target
        raise AssertionError(
            f"ACS transfer bank left incomplete pattern chains: {incomplete_states}."
        )
    patterns = tuple(pattern_records)
    kinds = {pattern.weight_kind for pattern in patterns}
    if len(kinds) != 1:  # pragma: no cover - every subset resolves one donor kind
        raise RuntimeError(
            f"ACS transfer family {family!r} resolved mixed weight kinds: "
            f"{sorted(kinds)}."
        )
    used_predictors = tuple(
        predictor
        for predictor in (*surface.required, *surface.optional)
        if any(predictor in pattern.predictors for pattern in patterns)
    )
    return _FamilyFit(
        predictions={
            target: raw_by_model_target[target_encodings[target].model_target]
            for target in targets
        },
        patterns=patterns,
        fit_records=tuple(fit_records),
        predictors=used_predictors,
        weight_kind=next(iter(kinds)),
        family_seed=_family_seed(seed, entity=entity, family=family),
        target_encodings=target_encodings,
    )


def _validate_banked_target_checkpoint(
    checkpoint: AcsTransferTargetCheckpoint,
    *,
    target_index: int,
    total_targets: int,
    entity: str,
    family: str,
    family_targets: tuple[str, ...],
    model_targets: tuple[str, ...],
    model_target: str,
    exported_targets: tuple[str, ...],
    recipient_rows: int,
    expected_states: Mapping[str, QRFChainState],
    expected_patterns: tuple[str, ...],
) -> None:
    observed_binding = (
        checkpoint.target_index,
        checkpoint.total_targets,
        checkpoint.entity,
        checkpoint.family,
        checkpoint.family_targets,
        checkpoint.model_targets,
        checkpoint.model_target,
        checkpoint.exported_targets,
    )
    expected_binding = (
        target_index,
        total_targets,
        entity,
        family,
        family_targets,
        model_targets,
        model_target,
        exported_targets,
    )
    if observed_binding != expected_binding:
        raise ValueError(
            f"ACS transfer bank returned the wrong target binding for "
            f"{entity}.{model_target}."
        )
    if len(checkpoint.raw_draw) != recipient_rows:
        raise ValueError(
            f"ACS transfer bank returned {len(checkpoint.raw_draw)} rows for "
            f"{entity}.{model_target}; expected {recipient_rows}."
        )
    if tuple(step.pattern for step in checkpoint.pattern_steps) != expected_patterns:
        raise ValueError(
            f"ACS transfer bank returned the wrong availability patterns for "
            f"{entity}.{model_target}."
        )
    for step in checkpoint.pattern_steps:
        state_before = expected_states[step.pattern]
        if step.state_before != state_before:
            raise ValueError(
                f"ACS transfer bank target {entity}.{model_target} does not "
                f"continue pattern {step.pattern!r}'s prior state."
            )
        if state_before.next_target != model_target:
            raise ValueError(
                f"ACS transfer bank target {entity}.{model_target} is not the "
                f"next target for pattern {step.pattern!r}."
            )
        expected_completed = (*state_before.completed_targets, model_target)
        if step.state_after.completed_targets != expected_completed:
            raise ValueError(
                f"ACS transfer bank target {entity}.{model_target} advanced "
                f"pattern {step.pattern!r} to the wrong target prefix."
            )
        if (
            step.state_after.predictors != state_before.predictors
            or step.state_after.targets != state_before.targets
            or step.state_after.entity != state_before.entity
            or step.state_after.weight_kind != state_before.weight_kind
            or step.state_after.weight_sha256 != state_before.weight_sha256
            or step.state_after.donor_index != state_before.donor_index
            or (
                state_before.recipient_index is not None
                and step.state_after.recipient_index != state_before.recipient_index
            )
        ):
            raise ValueError(
                f"ACS transfer bank target {entity}.{model_target} changed "
                f"pattern {step.pattern!r}'s bound chain identity."
            )


def _availability_patterns(
    surface: _FeatureSurface,
    *,
    eligible: np.ndarray,
) -> list[tuple[tuple[str, ...], np.ndarray]]:
    codes = np.zeros(len(surface.recipient), dtype=np.uint16)
    for bit, predictor in enumerate(surface.optional):
        observed = np.isfinite(_as_float_array(surface.recipient[predictor]))
        codes |= observed.astype(np.uint16) << bit

    patterns: list[tuple[tuple[str, ...], np.ndarray]] = []
    for code in np.unique(codes[eligible]):
        observed_optional = tuple(
            predictor
            for bit, predictor in enumerate(surface.optional)
            if int(code) & (1 << bit)
        )
        positions = np.flatnonzero(eligible & (codes == code))
        patterns.append((observed_optional, positions))
    return patterns


def _transfer_feature_surface(
    donor: Frame,
    recipient: Frame,
    *,
    entity: str,
    targets: Sequence[str],
) -> _FeatureSurface:
    if entity == donor.schema.person_entity:
        surface = _person_feature_surface(donor, recipient)
    else:
        surface = _group_feature_surface(donor, recipient, entity=entity)

    housing = bool(_HOUSING_TRANSFER_TARGETS.intersection(targets))
    if not housing:
        return surface
    if entity != donor.schema.person_entity:
        raise ValueError("pre_subsidy_rent must be transferred on the person entity.")
    mandatory = (_HEAD_FEATURE, _TENURE_FEATURE)
    missing = [name for name in mandatory if name not in surface.optional]
    if missing:
        raise ValueError(
            "pre_subsidy_rent transfer requires head and tenure predictors on "
            f"both donor and recipient; missing temporary feature(s): {missing}."
        )
    return _FeatureSurface(
        donor=surface.donor,
        recipient=surface.recipient,
        required=(*surface.required, *mandatory),
        optional=tuple(name for name in surface.optional if name not in mandatory),
    )


def _person_feature_surface(donor: Frame, recipient: Frame) -> _FeatureSurface:
    donor_required = _required_person_features(donor, role="donor")
    recipient_required = _required_person_features(recipient, role="recipient")
    donor_optional = _person_optional_features(donor, role="donor")
    recipient_optional = _person_optional_features(recipient, role="recipient")
    optional = tuple(
        name
        for name in ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS
        if name in donor_optional and name in recipient_optional
    )
    return _FeatureSurface(
        donor=pd.concat(
            [donor_required, donor_optional.loc[:, list(optional)]], axis=1
        ),
        recipient=pd.concat(
            [recipient_required, recipient_optional.loc[:, list(optional)]], axis=1
        ),
        required=ACS_PERSON_TRANSFER_PREDICTORS,
        optional=optional,
    )


def _required_person_features(frame: Frame, *, role: str) -> pd.DataFrame:
    person = frame.table(frame.schema.person_entity)
    direct = ("age", "is_female")
    _require_columns(
        person,
        direct,
        context=f"ACS transfer {role} person predictors",
    )
    result = person.loc[:, list(direct)].copy()
    state = _column_broadcast_to_person(frame, "state_fips")
    if state is None:
        raise ValueError(
            f"ACS transfer {role} person predictors missing household/person "
            "column 'state_fips'."
        )
    result[_STATE_FEATURE] = state.to_numpy()
    _validate_required_numeric_frame(
        result,
        context=f"ACS transfer {role} person predictors",
    )
    return result


def _person_optional_features(frame: Frame, *, role: str) -> pd.DataFrame:
    person = frame.table(frame.schema.person_entity)
    result = pd.DataFrame(index=person.index)

    direct = {
        _EMPLOYMENT_FEATURE: "employment_income_before_lsr",
        _SELF_EMPLOYMENT_FEATURE: "self_employment_income_before_lsr",
    }
    for feature, source in direct.items():
        values = _person_column(frame, source)
        if values is not None:
            result[feature] = values.to_numpy()

    if role == "donor":
        for feature, components in _DONOR_COMBINED_COMPONENTS.items():
            values = _complete_component_sum(frame, components, feature=feature)
            if values is not None:
                result[feature] = values.to_numpy()
    else:
        for feature, source in _RECIPIENT_COMBINED_SOURCES.items():
            values = _person_column(frame, source)
            if values is not None:
                result[feature] = values.to_numpy()

    head = _person_head_feature(frame)
    if head is not None:
        result[_HEAD_FEATURE] = head.to_numpy()
    tenure = _person_tenure_feature(frame)
    if tenure is not None:
        result[_TENURE_FEATURE] = tenure.to_numpy()

    _validate_optional_numeric_frame(
        result,
        context=f"ACS transfer {role} optional person predictors",
    )
    return result


def _person_column(frame: Frame, column: str) -> pd.Series | None:
    person = frame.table(frame.schema.person_entity)
    if column not in person.columns:
        return None
    series = person[column]
    if not _is_numeric_or_bool(series):
        raise TypeError(
            f"ACS transfer optional person predictor {column!r} must be "
            f"numeric/boolean, got dtype {series.dtype}."
        )
    return pd.Series(_as_float_array(series), index=person.index, name=column)


def _complete_component_sum(
    frame: Frame,
    components: Sequence[str],
    *,
    feature: str,
) -> pd.Series | None:
    person = frame.table(frame.schema.person_entity)
    if any(component not in person.columns for component in components):
        return None
    columns = person.loc[:, list(components)]
    _validate_optional_numeric_frame(
        columns,
        context=f"ACS donor analog {feature!r} components",
    )
    # NumPy propagation is intentional: any missing component makes the
    # combined analog missing, so its row enters a pattern without this feature.
    values = (
        _encoded_predictor_frame(
            columns,
            predictors=components,
        )
        .to_numpy(copy=False)
        .sum(axis=1)
    )
    return pd.Series(values, index=person.index, name=feature)


def _person_head_feature(frame: Frame) -> pd.Series | None:
    person = frame.table(frame.schema.person_entity)
    direct = _person_column(frame, "is_household_head")
    if direct is not None:
        invalid = direct.notna() & ~direct.isin([0.0, 1.0])
        if invalid.any():
            raise ValueError("is_household_head must contain only boolean values.")
        return direct.rename(_HEAD_FEATURE)

    for source, head_codes in (
        ("A_EXPRRP", {1, 2}),
        ("A_LINENO", {1}),
    ):
        if source not in person.columns:
            continue
        raw = _numeric_source(person[source], context=f"head predictor {source}")
        values = np.where(
            np.isnan(raw),
            np.nan,
            np.isin(raw, list(head_codes)).astype(np.float64),
        )
        return pd.Series(values, index=person.index, name=_HEAD_FEATURE)
    return None


def _person_tenure_feature(frame: Frame) -> pd.Series | None:
    for source in ("tenure_type", "spm_unit_tenure_type"):
        values = _column_broadcast_to_person(frame, source)
        if values is None:
            continue
        mapped = np.full(len(values), np.nan, dtype=np.float64)
        invalid: list[object] = []
        for position, value in enumerate(values.to_numpy(dtype=object)):
            if pd.isna(value):
                continue
            if isinstance(value, bytes):
                value = value.decode()
            name = getattr(value, "name", None)
            key = str(name if name is not None else value).upper()
            if key not in _TENURE_CODES:
                invalid.append(value)
                continue
            mapped[position] = _TENURE_CODES[key]
        if invalid:
            raise ValueError(
                f"ACS transfer tenure predictor {source!r} contains unsupported "
                f"value(s): {list(map(str, invalid[:5]))}."
            )
        return pd.Series(mapped, index=frame.person.index, name=_TENURE_FEATURE)

    return None


def _column_broadcast_to_person(frame: Frame, column: str) -> pd.Series | None:
    owner = _column_owner_or_none(frame, column)
    if owner is None:
        return None
    if owner == frame.schema.person_entity:
        return frame.person[column].copy()
    return frame.broadcast(column)


def _numeric_source(series: pd.Series, *, context: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    if invalid.any():
        raise ValueError(f"ACS transfer {context} contains non-numeric values.")
    values = numeric.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError(f"ACS transfer {context} contains infinity.")
    return values


def _group_feature_surface(
    donor: Frame,
    recipient: Frame,
    *,
    entity: str,
) -> _FeatureSurface:
    donor_required = _required_group_features(donor, entity=entity, role="donor")
    recipient_required = _required_group_features(
        recipient, entity=entity, role="recipient"
    )
    donor_person_optional = _person_optional_features(donor, role="donor")
    recipient_person_optional = _person_optional_features(recipient, role="recipient")
    person_optional = tuple(
        name
        for name in ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS
        if name in donor_person_optional and name in recipient_person_optional
    )
    donor_optional = _aggregate_optional_to_group(
        donor,
        entity=entity,
        person_features=donor_person_optional.loc[:, list(person_optional)],
    )
    recipient_optional = _aggregate_optional_to_group(
        recipient,
        entity=entity,
        person_features=recipient_person_optional.loc[:, list(person_optional)],
    )
    optional = tuple(_GROUP_OPTIONAL_NAMES[name] for name in person_optional)
    return _FeatureSurface(
        donor=pd.concat([donor_required, donor_optional], axis=1),
        recipient=pd.concat([recipient_required, recipient_optional], axis=1),
        required=ACS_GROUP_TRANSFER_PREDICTORS,
        optional=optional,
    )


def _required_group_features(
    frame: Frame,
    *,
    entity: str,
    role: str,
) -> pd.DataFrame:
    person = _required_person_features(frame, role=role)
    positions = _group_positions(frame, entity)
    n_groups = frame.n(entity)
    counts = np.bincount(positions, minlength=n_groups).astype(np.float64)
    age_sum = np.zeros(n_groups, dtype=np.float64)
    female_count = np.zeros(n_groups, dtype=np.float64)
    np.add.at(age_sum, positions, _as_float_array(person["age"]))
    np.add.at(female_count, positions, _as_float_array(person["is_female"]))
    person_state = _as_float_array(person[_STATE_FEATURE])
    finite_state = np.isfinite(person_state)
    finite_state_counts = np.bincount(
        positions,
        weights=finite_state.astype(np.int64),
        minlength=n_groups,
    )
    state_low = np.full(n_groups, np.inf, dtype=np.float64)
    state_high = np.full(n_groups, -np.inf, dtype=np.float64)
    np.minimum.at(state_low, positions[finite_state], person_state[finite_state])
    np.maximum.at(state_high, positions[finite_state], person_state[finite_state])
    spans_states = (finite_state_counts > 0) & (state_low != state_high)
    if spans_states.any():
        raise ValueError(
            f"ACS transfer {role} {entity!r} groups span multiple state_fips values."
        )
    state = np.full(n_groups, np.nan, dtype=np.float64)
    complete_state = (counts > 0) & (finite_state_counts == counts)
    state[complete_state] = state_low[complete_state]
    result = pd.DataFrame(
        {
            _GROUP_PERSON_COUNT: counts,
            _GROUP_AGE_SUM: age_sum,
            _GROUP_FEMALE_COUNT: female_count,
            _STATE_FEATURE: state,
        },
        index=frame.table(entity).index,
    )
    _validate_required_numeric_frame(
        result,
        context=f"ACS transfer {role} group predictors for {entity!r}",
    )
    return result


def _aggregate_optional_to_group(
    frame: Frame,
    *,
    entity: str,
    person_features: pd.DataFrame,
) -> pd.DataFrame:
    positions = _group_positions(frame, entity)
    n_groups = frame.n(entity)
    total_counts = np.bincount(positions, minlength=n_groups)
    result = pd.DataFrame(index=frame.table(entity).index)
    for person_name in person_features.columns:
        values = _as_float_array(person_features[person_name])
        finite = np.isfinite(values)
        observed_counts = np.bincount(
            positions,
            weights=finite.astype(np.int64),
            minlength=n_groups,
        )
        complete = observed_counts == total_counts
        group_name = _GROUP_OPTIONAL_NAMES[person_name]
        if person_name == _TENURE_FEATURE:
            low = np.full(n_groups, np.inf, dtype=np.float64)
            high = np.full(n_groups, -np.inf, dtype=np.float64)
            np.minimum.at(low, positions[finite], values[finite])
            np.maximum.at(high, positions[finite], values[finite])
            consistent = complete & (low == high)
            group_values = np.full(n_groups, np.nan, dtype=np.float64)
            group_values[consistent] = low[consistent]
        else:
            sums = np.zeros(n_groups, dtype=np.float64)
            # Missing rows contribute nothing to this internal accumulator, but
            # incomplete groups are reset to NaN below; no partial sum escapes.
            np.add.at(sums, positions[finite], values[finite])
            group_values = sums
            group_values[~complete] = np.nan
        result[group_name] = group_values
    _validate_optional_numeric_frame(
        result,
        context=f"ACS transfer optional group predictors for {entity!r}",
    )
    return result


def _group_positions(frame: Frame, entity: str) -> np.ndarray:
    membership = frame.person[frame.schema.membership_column(entity)].to_numpy()
    ids = frame.table(entity)[frame.schema.id_column(entity)].to_numpy()
    return np.searchsorted(ids, membership)


def _model_frame(
    donor: Frame,
    *,
    entity: str,
    features: pd.DataFrame,
    predictors: Sequence[str],
    target_encodings: Mapping[str, _TargetEncoding],
    mask: np.ndarray,
) -> Frame:
    selected = np.flatnonzero(mask)
    n = len(selected)
    resolved = donor.resolve_weights(entity)
    weights = Weights(resolved.values[selected], resolved.kind)
    model_values = pd.DataFrame(index=np.arange(n))
    encoded_features = _encoded_predictor_frame(features, predictors=predictors)
    for predictor in predictors:
        model_values[predictor] = encoded_features[predictor].iloc[selected].to_numpy()
    seen_model_targets: set[str] = set()
    for encoding in target_encodings.values():
        if encoding.model_target in seen_model_targets:
            continue
        model_values[encoding.model_target] = encoding.model_values.iloc[
            selected
        ].to_numpy()
        seen_model_targets.add(encoding.model_target)

    person_entity = donor.schema.person_entity
    if entity == person_entity:
        model_group = "model_unit"
        schema = EntitySchema(group_entities=(model_group,))
        person = model_values.copy()
        person.insert(0, "person_model_unit_id", np.arange(n, dtype=np.int64))
        person.insert(0, "person_id", np.arange(n, dtype=np.int64))
        unit = pd.DataFrame({"model_unit_id": np.arange(n, dtype=np.int64)})
        return Frame(
            {"person": person, model_group: unit},
            schema,
            {"person": weights},
        )

    schema = EntitySchema(group_entities=(entity,))
    group = model_values.copy()
    group.insert(0, f"{entity}_id", np.arange(n, dtype=np.int64))
    person = pd.DataFrame(
        {
            "person_id": np.arange(n, dtype=np.int64),
            f"person_{entity}_id": np.arange(n, dtype=np.int64),
        }
    )
    return Frame(
        {"person": person, entity: group},
        schema,
        {entity: weights},
    )


def _validate_frames(recipient: Frame, donor: Frame) -> None:
    if not isinstance(recipient, Frame):
        raise TypeError(f"recipient must be a Frame, got {type(recipient).__name__}.")
    if not isinstance(donor, Frame):
        raise TypeError(f"donor must be a Frame, got {type(donor).__name__}.")
    if recipient.schema != donor.schema:
        raise ValueError("ACS recipient and donor must use the same entity schema.")


def _validate_fit_options(
    *,
    seed: int,
    n_estimators: int,
    max_targets_per_fit: int,
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
    if (
        isinstance(n_estimators, bool)
        or not isinstance(n_estimators, int)
        or n_estimators <= 0
    ):
        raise ValueError(
            f"n_estimators must be a positive integer, got {n_estimators!r}."
        )
    if (
        isinstance(max_targets_per_fit, bool)
        or not isinstance(max_targets_per_fit, int)
        or max_targets_per_fit <= 0
    ):
        raise ValueError(
            "max_targets_per_fit must be a positive integer, got "
            f"{max_targets_per_fit!r}."
        )


def _normalize_regime_evidence_targets(
    targets: Iterable[tuple[str, str]],
    *,
    requested: frozenset[tuple[str, str]],
) -> frozenset[tuple[str, str]]:
    """Validate an explicit owner selection against the requested surface."""

    if isinstance(targets, (str, bytes)):
        raise TypeError(
            "regime_evidence_targets must contain (entity, target) pairs, not a string."
        )
    try:
        items = tuple(targets)
    except TypeError as exc:
        raise TypeError(
            "regime_evidence_targets must be an iterable of (entity, target) pairs."
        ) from exc
    malformed = [
        item
        for item in items
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or any(not isinstance(value, str) or not value for value in item)
        )
    ]
    if malformed:
        raise TypeError(
            "regime_evidence_targets contains malformed (entity, target) "
            f"pair(s): {malformed!r}."
        )
    selected = frozenset(items)
    unknown = sorted(selected - requested)
    if unknown:
        raise ValueError(
            "regime_evidence_targets names target(s) outside the requested "
            f"transfer surface: {unknown}."
        )
    return selected


def _normalize_target_families(
    families: TargetFamilies,
    *,
    recipient: Frame,
) -> list[tuple[str, str, tuple[str, ...]]]:
    if not isinstance(families, Mapping):
        raise TypeError("target_families must map entity names to family mappings.")
    unknown_entities = sorted(set(families) - set(recipient.entities))
    if unknown_entities:
        raise ValueError(
            f"target_families names unknown entity/entities: {unknown_entities}; "
            f"schema declares {list(recipient.entities)}."
        )

    normalized: list[tuple[str, str, tuple[str, ...]]] = []
    seen_targets: dict[str, tuple[str, str]] = {}
    for entity in recipient.entities:
        entity_families = families.get(entity, {})
        if not isinstance(entity_families, Mapping):
            raise TypeError(f"target_families[{entity!r}] must be a family mapping.")
        bad_family_names = [
            name
            for name in entity_families
            if not isinstance(name, str) or not name.strip()
        ]
        if bad_family_names:
            raise ValueError(
                f"ACS transfer family names must be non-empty strings on entity "
                f"{entity!r}; got {bad_family_names}."
            )
        for family in sorted(entity_families):
            raw_targets = entity_families[family]
            if isinstance(raw_targets, str) or not isinstance(raw_targets, Sequence):
                raise TypeError(
                    f"ACS transfer family {family!r} on entity {entity!r} must "
                    "provide a sequence of target names, not a string."
                )
            targets = tuple(raw_targets)
            bad = [
                target
                for target in targets
                if not isinstance(target, str) or not target.strip()
            ]
            if bad:
                raise ValueError(
                    f"ACS transfer family {family!r} on entity {entity!r} has "
                    f"invalid target name(s): {bad}."
                )
            duplicates = sorted(
                {target for target in targets if targets.count(target) > 1}
            )
            if duplicates:
                raise ValueError(
                    f"ACS transfer family {family!r} on entity {entity!r} has "
                    f"duplicate target(s): {duplicates}."
                )
            for target in targets:
                previous = seen_targets.get(target)
                if previous is not None:
                    raise ValueError(
                        f"ACS transfer target {target!r} is declared by both "
                        f"{previous} and {(entity, family)}; each column needs "
                        "one transfer family."
                    )
                seen_targets[target] = (entity, family)
            if targets:
                normalized.append((entity, family, targets))
    return normalized


def _split_large_target_families(
    families: Sequence[tuple[str, str, tuple[str, ...]]],
    *,
    max_targets_per_fit: int,
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Bound retained QRF forests without separating joint categorical codecs."""

    bounded: list[tuple[str, str, tuple[str, ...]]] = []
    immigration_pair = set(_IMMIGRATION_STATUS_TARGETS)
    for entity, family, targets in families:
        atoms: list[tuple[str, ...]] = []
        pair_added = False
        for target in targets:
            if target in immigration_pair and immigration_pair.issubset(targets):
                if not pair_added:
                    atoms.append(_IMMIGRATION_STATUS_TARGETS)
                    pair_added = True
                continue
            atoms.append((target,))

        batches: list[tuple[str, ...]] = []
        current: list[str] = []
        for atom in atoms:
            if current and len(current) + len(atom) > max_targets_per_fit:
                batches.append(tuple(current))
                current = []
            current.extend(atom)
        if current:
            batches.append(tuple(current))

        if len(batches) == 1:
            bounded.append((entity, family, batches[0]))
            continue
        width = len(str(len(batches)))
        bounded.extend(
            (
                entity,
                f"{family}__batch_{position:0{width}d}",
                batch,
            )
            for position, batch in enumerate(batches, start=1)
        )
    return bounded


def _missing_target_families(
    families: Sequence[tuple[str, str, tuple[str, ...]]],
    *,
    recipient: Frame,
) -> list[tuple[str, str, tuple[str, ...]]]:
    active: list[tuple[str, str, tuple[str, ...]]] = []
    for entity, family, targets in families:
        incomplete: list[str] = []
        table = recipient.table(entity)
        for target in targets:
            if target in table.columns:
                if table[target].isna().any():
                    incomplete.append(target)
                continue
            owner = _column_owner_or_none(recipient, target)
            if owner is not None:
                raise ValueError(
                    f"ACS transfer target {target!r} was requested on entity "
                    f"{entity!r}, but the recipient already carries it on "
                    f"entity {owner!r}."
                )
            incomplete.append(target)

        # A joint codec needs both columns to learn only observed donor pairs.
        # The complete companion is fitted but its recipient values are not
        # written because target-level null masks remain authoritative.
        immigration_pair = set(_IMMIGRATION_STATUS_TARGETS)
        if immigration_pair.issubset(targets) and immigration_pair.intersection(
            incomplete
        ):
            incomplete = [
                target
                for target in targets
                if target in immigration_pair or target in incomplete
            ]
        if incomplete:
            active.append((entity, family, tuple(incomplete)))
    return active


def _column_owner_or_none(frame: Frame, column: str) -> str | None:
    try:
        return frame.column_entity(column)
    except ValueError:
        return None


def _validate_donor_source(
    *,
    donor_spine: str,
    donor_channel: str | None,
) -> None:
    if not isinstance(donor_spine, str) or not donor_spine.strip():
        raise ValueError("donor_spine must be a non-empty string.")
    if donor_channel is not None and (
        not isinstance(donor_channel, str) or not donor_channel.strip()
    ):
        raise ValueError(
            "donor_channel must be 'auto', None, or a non-empty channel string."
        )


def resolve_acs_donor_channel(
    donor: Frame,
    channel: str | None,
) -> tuple[Frame, str | None]:
    """Resolve the requested transfer role without reading source-spine identity.

    ``donor_channel`` is the legacy public API name. Its non-``None`` values
    identify operator roles derived by :func:`support_role_series`; on an
    assembled pool those roles come from clone indices while receipt-declared
    source channels (for example ``"acs"`` and ``"asec"``) remain untouched.
    """

    if channel is None:
        return donor, None

    # Donor-role selection is person-grain: Frame.select() propagates the
    # person mask to group tables through structural IDs. Other-entity
    # metadata is checked only to distinguish an untagged legacy donor from
    # malformed partial metadata with no person role. Current assembled pools
    # receive their cross-grain provenance validation at the clone boundary.
    metadata_entities = {
        entity
        for entity in donor.entities
        if has_support_role_metadata(donor.table(entity), entity=entity)
    }
    person_entity = donor.schema.person_entity
    if not metadata_entities:
        if channel == ACS_DONOR_CHANNEL_AUTO:
            return donor, None
        raise ValueError(
            f"donor_channel={channel!r} requested a support role, but the donor "
            "carries no support-role metadata."
        )
    if person_entity not in metadata_entities:
        raise ValueError(
            "Donor-role resolution found partial support metadata but the "
            f"{person_entity!r} table has no support-role metadata."
        )

    role = (
        PUF_TAX_DETAIL_SUPPORT_CHANNEL if channel == ACS_DONOR_CHANNEL_AUTO else channel
    )
    roles = support_role_series(donor.table(person_entity), entity=person_entity)
    return _select_donor_role(donor, roles=roles, role=role), role


def _select_donor_role(
    donor: Frame,
    *,
    roles: pd.Series,
    role: str,
) -> Frame:
    mask = roles.eq(role)
    if not mask.any():
        available = sorted(map(str, roles.dropna().unique()))
        raise ValueError(
            f"donor support role {role!r} has no person rows; available roles: "
            f"{available}. Pass None only for a deliberate whole-donor fit."
        )
    if mask.all():
        return donor
    return donor.select(mask)


def _require_columns(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{context} missing column(s): {missing}.")


def _validate_required_numeric_frame(frame: pd.DataFrame, *, context: str) -> None:
    """Validate required predictor columns without rejecting peer-row nulls."""

    _validate_optional_numeric_frame(frame, context=context)


def _validate_optional_numeric_frame(frame: pd.DataFrame, *, context: str) -> None:
    _validate_numeric_kinds(frame, context=context)
    infinite = [
        column
        for column in frame.columns
        if np.isinf(_as_float_array(frame[column])).any()
    ]
    if infinite:
        raise ValueError(f"{context} contains infinite values: {infinite}.")


def _validate_numeric_kinds(frame: pd.DataFrame, *, context: str) -> None:
    non_numeric = [
        column for column in frame.columns if not _is_numeric_or_bool(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"{context} must be numeric/boolean: {non_numeric}.")


def _validate_donor_targets(
    donor: Frame,
    *,
    entity: str,
    targets: Sequence[str],
) -> None:
    table = donor.table(entity)
    missing = [target for target in targets if target not in table.columns]
    if missing:
        wrong_entity = {
            target: owner
            for target in missing
            if (owner := _column_owner_or_none(donor, target)) is not None
        }
        if wrong_entity:
            raise ValueError(
                f"ACS transfer donor target(s) requested on entity {entity!r} "
                f"live on other entities: {wrong_entity}."
            )
        raise ValueError(
            f"ACS transfer donor entity {entity!r} is missing target(s): {missing}."
        )
    metadata_by_target = {
        target: _engine_variable_metadata(target) for target in targets
    }
    unsupported = [
        target
        for target in targets
        if not _is_supported_target(table[target])
        # Known booleans with observed object values reach the encoder, which
        # owns the semantic-boolean check and its value-type diagnostics.
        and not (
            (metadata := metadata_by_target[target]) is not None
            and metadata.dtype == "bool"
            and pd.api.types.is_object_dtype(table[target].dtype)
            and table[target].notna().any()
        )
    ]
    if unsupported:
        raise TypeError(
            f"ACS transfer donor targets on entity {entity!r} must be numeric, "
            f"boolean, or categorical string/enum values: {unsupported}."
        )
    wrong_engine_entity = {
        target: metadata.entity
        for target in targets
        if (metadata := metadata_by_target[target]) is not None
        and metadata.entity != entity
    }
    if wrong_engine_entity:
        raise ValueError(
            f"ACS transfer targets requested on entity {entity!r} disagree with "
            f"PolicyEngine-US ownership: {wrong_engine_entity}."
        )


def _complete_target_mask(
    table: pd.DataFrame,
    *,
    targets: Sequence[str],
) -> np.ndarray:
    """Rows observed and finite for every target in one chained fit."""

    complete = np.ones(len(table), dtype=bool)
    for target in targets:
        values = table[target]
        observed = values.notna().to_numpy(dtype=bool, copy=True)
        if _is_numeric_or_bool(values):
            observed &= np.isfinite(_as_float_array(values))
        complete &= observed
    return complete


def _complete_predictor_mask(
    table: pd.DataFrame,
    *,
    predictors: Sequence[str],
) -> np.ndarray:
    """Rows whose encoded predictor values are all finite.

    ACS transfer owns stricter missing-value semantics than the underlying QRF:
    this mask selects rows complete for required predictors and for the optional
    predictors in each recipient availability pattern. Encoding maps missing
    values to ``NaN``; it never maps a nullable boolean's missing cells to
    ``False``.
    """

    encoded = _encoded_predictor_frame(table, predictors=predictors)
    return np.isfinite(encoded.to_numpy(copy=False)).all(axis=1)


def _is_supported_target(series: pd.Series) -> bool:
    if _is_numeric_or_bool(series):
        return True
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    inferred = pd.api.types.infer_dtype(series, skipna=True)
    return inferred in {
        "bytes",
        "categorical",
        "mixed",
        "string",
        "unicode",
    } and all(_is_hashable(value) for value in series.dropna())


def _is_hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _target_encodings(
    table: pd.DataFrame,
    *,
    targets: Sequence[str],
) -> dict[str, _TargetEncoding]:
    """Encode requested targets, preserving registered joint support."""

    requested = set(targets)
    immigration = requested.intersection(_IMMIGRATION_STATUS_TARGETS)
    if immigration and immigration != set(_IMMIGRATION_STATUS_TARGETS):
        missing = sorted(set(_IMMIGRATION_STATUS_TARGETS) - immigration)
        raise ValueError(
            "ACS transfer must impute ssn_card_type and immigration_status_str "
            f"jointly; missing paired target(s): {missing}."
        )

    result: dict[str, _TargetEncoding] = {}
    if immigration:
        pairs = list(
            zip(
                table[_IMMIGRATION_STATUS_TARGETS[0]].to_numpy(dtype=object),
                table[_IMMIGRATION_STATUS_TARGETS[1]].to_numpy(dtype=object),
                strict=True,
            )
        )
        categories = tuple(
            sorted(
                set(pairs),
                key=lambda pair: tuple(
                    (
                        type(value).__module__,
                        type(value).__qualname__,
                        repr(value),
                    )
                    for value in pair
                ),
            )
        )
        lookup = {pair: position for position, pair in enumerate(categories)}
        codes = np.fromiter(
            (lookup[pair] for pair in pairs),
            dtype=np.float64,
            count=len(pairs),
        )
        model_values = pd.Series(
            codes,
            index=table.index,
            name=_IMMIGRATION_STATUS_MODEL_TARGET,
        )
        for position, target in enumerate(_IMMIGRATION_STATUS_TARGETS):
            result[target] = _TargetEncoding(
                model_values=model_values,
                kind="joint_categorical",
                support=np.arange(len(categories), dtype=np.float64),
                categories=categories,
                model_target=_IMMIGRATION_STATUS_MODEL_TARGET,
                category_position=position,
            )

    for target in targets:
        if target in result:
            continue
        result[target] = _target_encoding(table[target], target=target)
    return result


def _complete_case_target_encodings(
    table: pd.DataFrame,
    *,
    targets: Sequence[str],
    complete: np.ndarray,
) -> dict[str, _TargetEncoding]:
    """Learn target support on complete cases, then restore donor alignment."""

    complete_table = table.iloc[np.flatnonzero(complete)]
    encodings = _target_encodings(complete_table, targets=targets)
    result: dict[str, _TargetEncoding] = {}
    positions = np.flatnonzero(complete)
    for target, encoding in encodings.items():
        aligned = pd.Series(
            np.nan,
            index=table.index,
            name=encoding.model_values.name,
        )
        aligned.iloc[positions] = encoding.model_values.to_numpy(dtype=np.float64)
        result[target] = replace(encoding, model_values=aligned)
    return result


def _target_encoding(series: pd.Series, *, target: str) -> _TargetEncoding:
    """Encode one scalar donor target for QRF without changing its support."""

    metadata = _engine_variable_metadata(target)
    target_dtype = metadata.dtype if metadata is not None else None
    semantic_boolean = _is_semantic_boolean(series)

    if (
        target_dtype == "bool"
        and pd.api.types.is_object_dtype(series.dtype)
        and not semantic_boolean
    ):
        offending_types = sorted(
            {
                f"{type(value).__module__}.{type(value).__qualname__}"
                for value in series.dropna()
                if not isinstance(value, (bool, np.bool_))
            }
        )
        if not offending_types:
            offending_types = ["<no observed values>"]
        raise TypeError(
            f"ACS transfer boolean target {target!r} is object-backed but "
            "contains non-boolean donor values; offending value types: "
            f"{offending_types}."
        )

    if target_dtype == "bool" and not (_is_numeric_or_bool(series) or semantic_boolean):
        raise TypeError(
            f"ACS transfer boolean target {target!r} must be boolean or use "
            "the supported physical numeric 0/1 primary-QRF/HDF representation; got "
            f"dtype {series.dtype!s}."
        )

    if target_dtype == "bool" or semantic_boolean:
        # Retiring primary-PUF and HDF artifacts can carry boolean targets in
        # the audited physical numeric 0/1 representation. Keep that legacy
        # compatibility explicit; the canonical stacked late path now
        # materializes booleans physically and validates every callback output.
        # Object-backed 0/1 values remain rejected rather than being coerced
        # through metadata.
        values = pd.Series(
            _as_float_array(series),
            index=series.index,
            name=target,
        )
        support = np.unique(values.to_numpy(dtype=np.float64))
        if not np.isin(support, [0.0, 1.0]).all():
            raise ValueError(
                f"ACS transfer boolean target {target!r} has donor support "
                f"outside {{0, 1}}: {support.tolist()}."
            )
        return _TargetEncoding(
            model_values=values,
            kind="boolean",
            support=support,
            model_target=target,
        )

    if pd.api.types.is_numeric_dtype(series.dtype):
        values = pd.Series(
            _as_float_array(series),
            index=series.index,
            name=target,
        )
        if target_dtype == "int" or target in _DISCRETE_NUMERIC_TARGETS:
            raw = values.to_numpy(dtype=np.float64)
            if not np.equal(raw, np.rint(raw)).all():
                raise ValueError(
                    f"ACS transfer integer target {target!r} contains "
                    "non-integral donor values."
                )
            return _TargetEncoding(
                model_values=values,
                kind="discrete_numeric",
                support=np.unique(raw),
                model_target=target,
            )
        return _TargetEncoding(
            model_values=values,
            kind="continuous",
            model_target=target,
        )

    raise TypeError(
        f"ACS transfer categorical target {target!r} has no registered joint "
        "support codec."
    )


@lru_cache(maxsize=1)
def _policyengine_us_adapter() -> Any | None:
    try:
        adapter = import_module(
            "microcosm.frame.adapters.policyengine_us"
        ).PolicyEngineUSVariableMetadataIndex()
        # Force the optional source index to load once so an absent extra uses
        # the documented dtype fallback rather than failing mid-family.
        adapter.variables()
    except ImportError:
        return None
    return adapter


def _engine_variable_metadata(target: str) -> Any | None:
    adapter = _policyengine_us_adapter()
    if adapter is None:
        return None
    try:
        return adapter.variable_metadata(target)
    except ValueError:
        return None


def _is_numeric_or_bool(series: pd.Series) -> bool:
    """Recognize physical numeric dtypes and semantic boolean columns."""

    return bool(
        _is_semantic_boolean(series)
        or (
            pd.api.types.is_numeric_dtype(series.dtype)
            and not pd.api.types.is_complex_dtype(series.dtype)
        )
    )


def _is_semantic_boolean(series: pd.Series) -> bool:
    """Recognize nullable object columns whose observed values are booleans."""

    if pd.api.types.is_bool_dtype(series.dtype):
        return True
    return bool(
        pd.api.types.is_object_dtype(series.dtype)
        and pd.api.types.infer_dtype(series, skipna=True) == "boolean"
    )


def _as_float_array(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype=np.float64, na_value=np.nan)


def _encoded_predictor_frame(
    table: pd.DataFrame,
    *,
    predictors: Sequence[str],
) -> pd.DataFrame:
    """Encode validated numeric/semantic-boolean predictors as float64."""

    return pd.DataFrame(
        {predictor: _as_float_array(table[predictor]) for predictor in predictors},
        index=table.index,
    )


def _family_seed(seed: int, *, entity: str, family: str) -> int:
    payload = f"{seed}\0{entity}\0{family}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _pattern_seed(
    seed: int,
    *,
    entity: str,
    family: str,
    observed_optional: Sequence[str],
) -> int:
    pattern = "\0".join(observed_optional)
    payload = f"{seed}\0{entity}\0{family}\0{pattern}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _pattern_name(position: int, observed_optional: Sequence[str]) -> str:
    payload = "\0".join(observed_optional).encode()
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return f"pattern_{position:02d}_{digest}"


def _qrf() -> Any:
    global QRF
    if QRF is None:
        QRF = import_module("microcosm.fit").QRF
    return QRF


def _validate_prediction_values(
    predictions: pd.DataFrame,
    *,
    entity: str,
    family: str,
    pattern: str,
) -> None:
    non_finite = [
        column
        for column in predictions.columns
        if not np.isfinite(_as_float_array(predictions[column])).all()
    ]
    if non_finite:
        raise ValueError(
            f"ACS transfer predictions for {entity!r}/{family!r}/{pattern!r} "
            f"contain non-finite values: {non_finite}."
        )


def _prediction_values(
    prediction: np.ndarray,
    *,
    encoding: _TargetEncoding,
    entity: str,
    target: str,
) -> np.ndarray | pd.arrays.BooleanArray:
    values = np.asarray(prediction, dtype=np.float64)
    if encoding.support is not None:
        finite = np.isfinite(values)
        values = values.copy()
        values[finite] = _snap_to_support(values[finite], encoding.support)

    if encoding.kind == "continuous":
        return values
    if encoding.kind == "discrete_numeric":
        finite = np.isfinite(values)
        if finite.all():
            return values.astype(np.int64)
        result = pd.array([pd.NA] * len(values), dtype="Int64")
        result[finite] = values[finite].astype(np.int64)
        return result

    if encoding.kind == "joint_categorical":
        result = np.full(len(values), np.nan, dtype=object)
        finite = np.isfinite(values)
        codes = values[finite].astype(np.int64)
        if ((codes < 0) | (codes >= len(encoding.categories))).any():
            raise ValueError(
                f"ACS transfer categorical target {target!r} on entity "
                f"{entity!r} produced codes outside donor support."
            )
        if encoding.category_position is None:  # pragma: no cover - codec invariant
            raise RuntimeError("Joint categorical encoding lacks a component index.")
        decoded = np.asarray(
            [encoding.categories[code][encoding.category_position] for code in codes],
            dtype=object,
        )
        result[finite] = decoded
        return result

    finite = np.isfinite(values)
    valid = np.isclose(values[finite], 0.0) | np.isclose(values[finite], 1.0)
    if not valid.all():
        raise ValueError(
            f"ACS transfer boolean target {target!r} on entity {entity!r} "
            "produced values outside boolean support."
        )
    if finite.all():
        return np.isclose(values, 1.0)
    result = pd.array([pd.NA] * len(values), dtype="boolean")
    result[finite] = np.isclose(values[finite], 1.0)
    return result


def _fill_recipient_nulls(
    table: pd.DataFrame,
    *,
    target: str,
    predicted: np.ndarray | pd.api.extensions.ExtensionArray,
) -> tuple[pd.Series, np.ndarray]:
    """Merge predictions into null cells without changing observed values."""

    prediction = pd.Series(predicted, index=table.index, name=target)
    predicted_observed = prediction.notna().to_numpy(dtype=bool)
    if target not in table.columns:
        return prediction, predicted_observed

    result = table[target].copy()
    imputed = result.isna().to_numpy(dtype=bool) & predicted_observed
    if not imputed.any():
        return result, imputed

    if isinstance(result.dtype, pd.CategoricalDtype):
        additions = [
            value
            for value in pd.unique(prediction.iloc[np.flatnonzero(imputed)])
            if value not in result.cat.categories
        ]
        if additions:
            result = result.cat.add_categories(additions)

    positions = np.flatnonzero(imputed)
    result.iloc[positions] = prediction.iloc[positions].to_numpy()
    return result, imputed


def _snap_to_support(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Map values to nearest sorted donor support in O(n log k) memory."""

    observed = np.unique(np.asarray(support, dtype=np.float64))
    if len(observed) == 0:
        raise ValueError("ACS transfer discrete donor support is empty.")
    positions = np.searchsorted(observed, values, side="left")
    upper_positions = np.minimum(positions, len(observed) - 1)
    lower_positions = np.maximum(positions - 1, 0)
    lower = observed[lower_positions]
    upper = observed[upper_positions]
    choose_upper = np.abs(upper - values) < np.abs(values - lower)
    return np.where(choose_upper, upper, lower)
