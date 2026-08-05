"""US support expansion for PUF tax-detail imputations.

The PUF tax-detail donor needs a distinct support channel: the ASEC/CPS
records remain as the baseline channel, and a cloned channel receives
PUF-sourced tax detail without overwriting the baseline rows. The expansion is
mass-conserving: with two support channels, each channel receives half of the
incoming weights so the frame's aggregate population does not double.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import FitWeightRecord
from populace.build.us_runtime.puf_e01000_reconciliation import (
    PUF_SCHEDULE_D_JOINT_COLUMNS,
    puf_capital_gains_joint_metrics,
    puf_processed_capital_gains_stage,
)
from populace.build.us_runtime.puf_interest_components import (
    split_us_puf_e19200_by_agi_band,
)
from populace.build.us_runtime.qbi_inputs import (
    US_QBI_BOOLEAN_OUTPUT_COLUMNS,
    US_QBI_NONNEGATIVE_OUTPUT_COLUMNS,
    US_QBI_OUTPUT_COLUMNS,
)
from populace.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_CLONE_INDEX,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    has_support_role_metadata,
    puf_tax_detail_clone_mask,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_role_series,
    support_source_id_column,
    validate_assembly_provenance,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights, wquantile
from populace.frame.schema import EntitySchema

QRF: Any | None = None

__all__ = [
    "BASE_ASEC_SUPPORT_CHANNEL",
    "PufTaxDetailChainInputs",
    "PUF_ABSENT_CELLS_LEGACY_ZERO_FILL",
    "PUF_ABSENT_CELLS_PRESERVE_NULLS",
    "PUF_CLONE_ATTACHMENT_MANIFEST_KEY",
    "PUF_TAX_DETAIL_CLONE_INDEX",
    "PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS",
    "PUF_TAX_DETAIL_SUPPORT_CHANNEL",
    "PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN",
    "US_PUF_DONOR_MORTGAGE_QUARANTINE_FIELDS",
    "US_PUF_DONOR_MORTGAGE_OUTLIER_CEILING",
    "US_PUF_SUPPORT_FIT_NAME",
    "US_PUF_SUPPORT_STAGE_NAME",
    "assert_formula_owned_blocklist_current",
    "clone_us_frame_for_puf_support",
    "finalize_us_puf_tax_detail_predictions",
    "has_support_role_metadata",
    "impute_us_puf_tax_detail_support",
    "puf_tax_detail_clone_mask",
    "puf_tax_unit_donor_from_arrays",
    "prepare_us_puf_tax_detail_chain_inputs",
    "resolve_formula_owned_outputs",
    "spine_source_id_column",
    "support_channel_column",
    "support_clone_index_column",
    "support_role_series",
    "support_source_id_column",
    "validate_puf_clone_attachment",
]

US_PUF_SUPPORT_STAGE_NAME = "puf_support_channel"

#: Frozen receipt binding a seeded clone attachment (populace#578 revision
#: item 3) to the live rows: fraction, seed, the floor-rule counts, and the
#: digest of the attached households' assembly-unique source IDs.
PUF_CLONE_ATTACHMENT_MANIFEST_KEY = "us_puf_clone_attachment_manifest"
_PUF_CLONE_ATTACHMENT_MANIFEST_VERSION = 1

#: Finalization policies for cells the PUF pass does not own (populace#578
#: revision, audit item 1).  The legacy policy reproduces the historical
#: two-arm behavior byte for byte: every requested output column is coerced
#: with a global ``fillna(0.0)``, so cells that were never imputed read as
#: observed zeros.  The preserve-nulls policy is the stacked-spine doctrine:
#: absence stays null until an authorized stage fills it — finalization only
#: writes the PUF clone arm's cells, creates missing columns as null, and
#: never converts absence into an observed ``0.0``.
PUF_ABSENT_CELLS_LEGACY_ZERO_FILL = "legacy_zero_fill"
PUF_ABSENT_CELLS_PRESERVE_NULLS = "preserve_nulls"
_PUF_ABSENT_CELLS_POLICIES = (
    PUF_ABSENT_CELLS_LEGACY_ZERO_FILL,
    PUF_ABSENT_CELLS_PRESERVE_NULLS,
)

#: The name the PUF tax-detail support fit records in the build weights audit
#: (populace #300). Stable so a release manifest and its allowlist can refer to
#: this fit by name.
US_PUF_SUPPORT_FIT_NAME = "us_puf_tax_detail_support"
PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN = "puf_source_adjusted_gross_income"

# populace#516 donor mortgage quarantine: $10M of annual home-mortgage
# interest implies roughly a $250M mortgage at 4%, not a genuine Schedule A
# return; the pinned artifact's maximum REAL-scale unit values are only low
# single-digit millions. Its grouped-raw >=$10M intersection contains 3,066
# tax units (max $235.97B; weight 3,684 of 161M) and $2.947T of phantom
# mortgage-interest mass, versus $418B retained; 1,823 have synthetic IDs
# >= 1,000,000 and 1,243 have ordinary IDs.
#
# populace#567 measured that whole-row removal also deleted $98.176B of
# positive capital gains that are unrelated to E19200. The quarantine is
# therefore field-local: only the raw E19200 lineage and its conserving
# non-mortgage residual are zeroed. Mortgage balances and origination years
# are independently sourced and remain intact, as do every non-mortgage field.
# This is an outlier screen, NOT aggregate-lineage removal: an ID-range union
# would also delete 2,162 healthy synthetic donors.
US_PUF_DONOR_MORTGAGE_OUTLIER_CEILING = 10_000_000.0

# Reserved internal column used to thread the grouped RAW mortgage value to
# the outlier screen; never a legal requested output (populace#516).
_MORTGAGE_OUTLIER_SCREEN_COLUMN = "_raw_home_mortgage_interest_for_outlier_screen"
_E19200_AGI_BAND_COLUMN = "_adjusted_gross_income_for_e19200_band"

_DEFAULT_SUPPORT_CHANNELS = (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
)

_US_PUF_E19200_LINEAGE_DONOR_COLUMNS = (
    "home_mortgage_interest",
    "first_home_mortgage_interest",
    "second_home_mortgage_interest",
    "interest_deduction",
)
US_PUF_DONOR_MORTGAGE_QUARANTINE_FIELDS = (
    "home_mortgage_interest",
    "investment_interest_expense",
    "first_home_mortgage_interest",
    "second_home_mortgage_interest",
    "interest_deduction",
)


@dataclass(frozen=True)
class PufTaxDetailChainInputs:
    """Normalized donor and recipient surfaces for the raw PUF QRF chain."""

    donor: pd.DataFrame
    donor_frame: Frame
    recipient_features: pd.DataFrame
    recipient_tax_unit_ids: np.ndarray
    predictors: tuple[str, ...]
    person_outputs: tuple[str, ...]
    tax_unit_outputs: tuple[str, ...]

    @property
    def target_order(self) -> tuple[str, ...]:
        """Return the exact person-then-tax-unit chained target order."""

        return (*self.person_outputs, *self.tax_unit_outputs)


PUF_TAX_DETAIL_DEFAULT_PREDICTORS = (
    "puf_predictor_filing_status_code",
    "puf_predictor_tax_unit_person_count",
    "puf_predictor_employment_income",
    "puf_predictor_self_employment_income",
    "puf_predictor_taxable_interest_income",
    "puf_predictor_dividend_income",
    "puf_predictor_short_term_capital_gains",
    "puf_predictor_long_term_capital_gains",
)

PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "taxable_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "tax_exempt_interest_income",
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
    "long_term_capital_gains_on_collectibles",
    "non_sch_d_capital_gains",
    "taxable_private_pension_income",
    "taxable_ira_distributions",
    "social_security_retirement",
    "social_security_disability",
    "social_security_dependents",
    "social_security_survivors",
    "alimony_income",
    "alimony_expense",
    "salt_refund_income",
    "charitable_cash_donations",
    "charitable_non_cash_donations",
    "real_estate_taxes",
    "home_mortgage_interest",
    "investment_interest_expense",
    "investment_income_elected_form_4952",
    "student_loan_interest",
    "educator_expense",
    "qualified_tuition_expenses",
    "casualty_loss",
    "unreimbursed_business_employee_expenses",
    # The engine owns the realized contribution amounts through the
    # IRA-limit scale and self-employment caps; the persistable leaves are
    # the desired contributions, equal to the PUF's observed deductions at
    # baseline (issue #278).
    "traditional_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
    "rental_income",
    "estate_income",
    "farm_income",
    "farm_operations_income",
    "farm_rent_income",
    "miscellaneous_income",
    "partnership_income",
    "s_corp_income",
    "partnership_self_employment_net_earnings",
    *US_QBI_OUTPUT_COLUMNS,
)

PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_dependents",
    "social_security_survivors",
)

PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS: tuple[str, ...] = (
    "domestic_production_ald",
    "unrecaptured_section_1250_gain",
    "first_home_mortgage_balance",
    "second_home_mortgage_balance",
    "first_home_mortgage_interest",
    "second_home_mortgage_interest",
    "first_home_mortgage_origination_year",
    "second_home_mortgage_origination_year",
    "health_savings_account_ald",
)

_PUF_TAX_DETAIL_DISCRETE_TAX_UNIT_OUTPUTS = frozenset(
    {
        "first_home_mortgage_origination_year",
        "second_home_mortgage_origination_year",
    }
)
_PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS = frozenset(US_QBI_BOOLEAN_OUTPUT_COLUMNS)
_PUF_TAX_DETAIL_SPARSE_TAX_UNIT_OUTPUTS = frozenset(
    {
        "domestic_production_ald",
        "unrecaptured_section_1250_gain",
    }
)
_PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS = frozenset(
    {
        "taxable_interest_income",
        "qualified_tuition_expenses",
        "educator_expense",
        "casualty_loss",
        "investment_income_elected_form_4952",
        "long_term_capital_gains_on_collectibles",
        "alimony_income",
        "alimony_expense",
        "salt_refund_income",
    }
)

# Interim weight-reintroduction at the finalize seam: the root fix
# (weight-aware leaf draws) is #481, and the dead manifest support_clip is
# #482. Tail bounds are defined only for passthrough outputs, which removes all
# snapping, sparse-pruning, and signed-calibration ordering interactions by
# construction.
_PUF_TAX_DETAIL_TAIL_BOUND_QUANTILES: dict[str, float] = {
    "non_sch_d_capital_gains": 0.999
}

# ASEC directly measures recipient alimony. The PUF QRF therefore sparsifies
# only the cloned PUF half for this leaf; pruning the ASEC half would discard
# reported source observations. Expense has no ASEC analogue, so its zero ASEC
# half is unaffected by the ordinary all-channel sparsification loop.
_PUF_TAX_DETAIL_PRESERVE_BASE_ASEC_OUTPUTS = frozenset({"alimony_income"})

# Sparse, sign-mixed, heavy-tailed person outputs whose imputed *signed* mass
# must be pinned to the donor instrument. The regime-gated QRF imputes such a
# column as an independent sign gate (a HistGradientBoostingClassifier) times
# per-sign magnitude forests. On a rare loss-mixed column the gate regresses the
# positive/negative/zero shares toward balance (it over-predicts the rarer
# leg) and the magnitude forests inflate each leg, so nothing pins the aggregate
# signed total: the imputed net -- a small difference of two large legs --
# regresses toward a fixed balance point that is nearly independent of the
# donor's true net. A loss-heavy source (SOI Schedule F, net-negative
# nationally) is then dragged toward or past zero, flipping the export sign
# (farm_operations_income) or manufacturing a spurious cancelling leg
# (partnership_self_employment_net_earnings, populace #432). These columns get
# a per-leg mass calibration in finalization -- the signed generalization of the
# donor-positive-rate sparsification -- so the imputed per-unit-weight positive
# and negative leg masses match the donor's and the net sign tracks the source.
_PUF_TAX_DETAIL_SIGNED_MASS_CALIBRATED_PERSON_OUTPUTS = frozenset(
    {
        "farm_operations_income",
        "partnership_self_employment_net_earnings",
    }
)

# Known formula-owned outputs the PUF tax-detail donor must never carry as
# persistable leaves. This is a documented *seed* set, not the whole story:
# :func:`resolve_formula_owned_outputs` unions it with the set derived from the
# installed PolicyEngine-US source metadata, so a new formula-owned aggregate
# added upstream is rejected even before anyone adds it here (populace issue
# #301). Every name here has a stated reason; a build-time consistency check
# (:func:`assert_formula_owned_blocklist_current`) fails if the engine stops
# treating one of these as formula-owned, so a stale entry cannot linger.
PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS = frozenset(
    {
        "interest_deduction",
        "state_withheld_income_tax",
        # The self-employed ALDs are engine aggregates of formula-owned
        # per-person caps; the persistable leaves are the contribution and
        # premium inputs those formulas cap.
        "self_employed_pension_contribution_ald",
        "self_employed_pension_contribution_ald_person",
        "self_employed_health_insurance_ald",
        "self_employed_health_insurance_ald_person",
    }
)

_PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS = frozenset(
    {
        "employment_income",
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
        "taxable_interest_income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
        "tax_exempt_interest_income",
        "long_term_capital_gains_before_response",
        "long_term_capital_gains_on_collectibles",
        "taxable_pension_income",
        "taxable_private_pension_income",
        "taxable_ira_distributions",
        "social_security_retirement",
        "social_security_disability",
        "social_security_dependents",
        "social_security_survivors",
        "alimony_income",
        "alimony_expense",
        "salt_refund_income",
        "charitable_cash_donations",
        "charitable_non_cash_donations",
        "real_estate_taxes",
        "home_mortgage_interest",
        "investment_interest_expense",
        "investment_income_elected_form_4952",
        "student_loan_interest",
        "educator_expense",
        "qualified_tuition_expenses",
        "casualty_loss",
        "unreimbursed_business_employee_expenses",
        "domestic_production_ald",
        "unrecaptured_section_1250_gain",
        "traditional_ira_contributions_desired",
        "self_employed_pension_contributions_desired",
        "health_savings_account_ald",
        "first_home_mortgage_balance",
        "second_home_mortgage_balance",
        "first_home_mortgage_interest",
        "second_home_mortgage_interest",
        "first_home_mortgage_origination_year",
        "second_home_mortgage_origination_year",
        *US_QBI_NONNEGATIVE_OUTPUT_COLUMNS,
    }
)
# Person-grain PE input leaves the processed PUF artifact only observes as
# tax-unit-grain deduction arrays. The donor carries the tax-unit total under
# the leaf's name; imputed totals are distributed over the cloned people by
# the leaf's distribution basis below.
_PERSON_OUTPUT_TAX_UNIT_GRAIN_SOURCES: Mapping[str, str] = {
    "self_employed_pension_contributions_desired": (
        "self_employed_pension_contribution_ald"
    ),
}
# Distribution basis for person outputs the ASEC channel carries no values
# for: the imputed tax-unit total is split by these person columns' shares
# (falling back to the unit's first person when the basis is empty).
# Contributions require compensation, so an earnings basis keeps per-person
# contribution limits binding the way they do on the donor records.
_PERSON_OUTPUT_DISTRIBUTION_BASIS: Mapping[str, tuple[str, ...]] = {
    "casualty_loss": (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
    ),
    "educator_expense": ("employment_income_before_lsr",),
    "traditional_ira_contributions_desired": (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
    ),
    "self_employed_pension_contributions_desired": (
        "self_employment_income_before_lsr",
    ),
    "qualified_tuition_expenses": ("is_full_time_college_student",),
    "estate_income_would_be_qualified": ("estate_income",),
    "farm_operations_income_would_be_qualified": ("farm_operations_income",),
    "farm_rent_income_would_be_qualified": ("farm_rent_income",),
    "partnership_s_corp_income_would_be_qualified": (
        "partnership_income",
        "s_corp_income",
    ),
    "rental_income_would_be_qualified": ("rental_income",),
    "self_employment_income_would_be_qualified": (
        "self_employment_income_before_lsr",
        "sstb_self_employment_income_before_lsr",
    ),
    "sstb_self_employment_income_would_be_qualified": (
        "sstb_self_employment_income_before_lsr",
        "self_employment_income_before_lsr",
    ),
    "business_is_sstb": (
        "sstb_self_employment_income_before_lsr",
        "self_employment_income_before_lsr",
        "partnership_income",
        "s_corp_income",
        "estate_income",
    ),
    "qualified_bdc_income": ("non_qualified_dividend_income",),
    "qualified_reit_and_ptp_income": (
        "non_qualified_dividend_income",
        "partnership_income",
        "s_corp_income",
    ),
    "sstb_self_employment_income_before_lsr": (
        "business_is_sstb",
        "self_employment_income_before_lsr",
    ),
    "sstb_unadjusted_basis_qualified_property": ("business_is_sstb",),
    "sstb_w2_wages_from_qualified_business": ("business_is_sstb",),
    "unadjusted_basis_qualified_property": (
        "self_employment_income_before_lsr",
        "partnership_income",
        "s_corp_income",
        "rental_income",
        "estate_income",
    ),
    "w2_wages_from_qualified_business": (
        "self_employment_income_before_lsr",
        "partnership_income",
        "s_corp_income",
        "rental_income",
        "estate_income",
    ),
}
_PREDICTOR_LEAF_ALIASES: Mapping[str, tuple[str, ...]] = {
    "employment_income": ("employment_income_before_lsr",),
    "self_employment_income": ("self_employment_income_before_lsr",),
    "long_term_capital_gains": ("long_term_capital_gains_before_response",),
    "taxable_pension_income": (
        "taxable_private_pension_income",
        "taxable_public_pension_income",
    ),
    "social_security": (
        "social_security_retirement",
        "social_security_disability",
        "social_security_survivors",
        "social_security_dependents",
    ),
}

_FILING_STATUS_CODES = {
    "SINGLE": 1.0,
    "JOINT": 2.0,
    "SEPARATE": 3.0,
    "HEAD_OF_HOUSEHOLD": 4.0,
    "SURVIVING_SPOUSE": 5.0,
}
_PUF_MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS = {
    "health_insurance_premiums_without_medicare_part_b": 0.453,
    "other_medical_expenses": 0.325,
    "over_the_counter_health_expenses": 0.085,
}
# The omitted 13.7% E17500 share represented reported Medicare Part B premiums.
# Do not redistribute it: the engine computes its distinct singular Part B output.
_PUF_PREDICTOR_PREFIX = "puf_predictor_"


def clone_us_frame_for_puf_support(
    frame: Frame,
    *,
    channels: Sequence[str] = _DEFAULT_SUPPORT_CHANNELS,
    clone_attachment_fraction: float | None = None,
    clone_attachment_seed: int | None = None,
) -> Frame:
    """Clone a US frame into support channels for PUF detail imputation.

    Args:
        frame: A US-schema frame. A frame without support metadata follows the
            legacy ASEC expansion. A frame whose entity tables all carry
            native (clone-index zero) support provenance keeps its source
            channels unchanged while receiving one PUF-detail clone.
        channels: The canonical ``("asec", "puf_tax_detail")`` operator-role
            pair. Custom roles are rejected at this boundary because downstream
            operators accept only these two roles.
        clone_attachment_fraction: Optional seeded-attachment fraction in
            ``(0, 1]`` (populace#578 revision item 3).  When set — assembled
            frames only — the PUF clone arm attaches to a seeded whole-
            household sample of the spine: ``floor(fraction * households)``
            households keep their clone pair at half weight each, every other
            household keeps a single full-weight native lineage, and partial
            attachments carry a manifest binding fraction, seed, and the
            realized selection digest to the live rows.  ``None`` and exact
            ``1.0`` both return the ordinary full two-arm clone unchanged.
        clone_attachment_seed: Non-negative selection seed; required exactly
            when a fraction is given so the attachment identity is always
            explicit.

    Returns:
        A new frame with every entity table cloned once per support role, all
        structural IDs remapped consistently, and typed weights mass-conserved.
        Preassembled source-channel provenance is preserved.

    Raises:
        ValueError: If the frame is not US-schema, channel names are invalid,
            metadata is partial or already operated, an ID remapping would
            collide, or the attachment configuration is invalid.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("PUF support expansion currently requires the US schema.")
    support_channels = _validate_channels(channels)
    has_assembly_provenance = _has_native_assembly_provenance(frame)
    if (clone_attachment_fraction is None) != (clone_attachment_seed is None):
        raise ValueError(
            "clone_attachment_fraction and clone_attachment_seed must be "
            "provided together; a seeded attachment identity is never implicit."
        )
    if clone_attachment_fraction is not None:
        if not has_assembly_provenance:
            raise ValueError(
                "Seeded clone attachment requires an assembled frame with "
                "native support provenance."
            )
        if (
            isinstance(clone_attachment_fraction, bool)
            or not isinstance(clone_attachment_fraction, (int, float))
            or not np.isfinite(clone_attachment_fraction)
            or not 0.0 < float(clone_attachment_fraction) <= 1.0
        ):
            raise ValueError(
                "clone_attachment_fraction must be a finite number in (0, 1]; "
                f"got {clone_attachment_fraction!r}."
            )
        if (
            isinstance(clone_attachment_seed, bool)
            or not isinstance(clone_attachment_seed, int)
            or clone_attachment_seed < 0
        ):
            raise ValueError(
                "clone_attachment_seed must be a non-negative integer; got "
                f"{clone_attachment_seed!r}."
            )
    if has_assembly_provenance:
        validate_assembly_provenance(
            frame,
            boundary="PUF support clone entry",
        )
        if support_channels != _DEFAULT_SUPPORT_CHANNELS:
            raise ValueError(
                "Preassembled spine frames use the canonical native/PUF clone "
                "roles; custom support channels are not accepted."
            )
    else:
        _reject_metadata_collisions(frame, support_channels)
    id_multiplier = _id_multiplier_for_frame(frame)

    tables: dict[str, pd.DataFrame] = {}
    for entity in frame.entities:
        if has_assembly_provenance:
            tables[entity] = _clone_preassembled_entity_table(
                frame.table(entity),
                entity=entity,
                schema=frame.schema,
                id_multiplier=id_multiplier,
            )
        else:
            tables[entity] = _clone_entity_table(
                frame.table(entity),
                entity=entity,
                schema=frame.schema,
                channels=support_channels,
                id_multiplier=id_multiplier,
            )
    for link_name in frame.links:
        tables[link_name] = _clone_link_table(
            frame.link(link_name),
            link_name=link_name,
            schema=frame.schema,
            clone_count=len(support_channels),
            id_multiplier=id_multiplier,
        )

    weights = {
        entity: Weights(
            values=np.tile(
                frame.weights_for(entity).values / len(support_channels),
                len(support_channels),
            ),
            kind=frame.weights_for(entity).kind,
        )
        for entity in frame.weighted_entities
    }
    strata = pd.concat(
        [frame.strata.copy() for _channel in support_channels],
        ignore_index=True,
    )
    result = Frame(
        tables,
        frame.schema,
        weights,
        strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    if has_assembly_provenance:
        validate_assembly_provenance(
            result,
            boundary="PUF support clone output",
        )
    if clone_attachment_fraction is not None:
        assert clone_attachment_seed is not None  # validated at entry
        if float(clone_attachment_fraction) == 1.0:
            validate_puf_clone_attachment(
                result,
                boundary="PUF support full-clone identity output",
                expected_fraction=float(clone_attachment_fraction),
                expected_seed=clone_attachment_seed,
            )
            return result
        result = _attach_clone_arm_to_seeded_sample(
            result,
            fraction=clone_attachment_fraction,
            seed=clone_attachment_seed,
        )
        validate_assembly_provenance(
            result,
            boundary="PUF support clone attachment output",
        )
        validate_puf_clone_attachment(
            result,
            boundary="PUF support clone attachment output",
            expected_fraction=float(clone_attachment_fraction),
            expected_seed=clone_attachment_seed,
        )
    return result


def _attach_clone_arm_to_seeded_sample(
    cloned: Frame,
    *,
    fraction: float,
    seed: int,
) -> Frame:
    """Keep the PUF clone pair on a seeded household sample only.

    Operates on the full two-arm clone so the selected pairs are byte-
    identical to the full expansion: the detail lineages of unselected
    households are dropped whole (via :meth:`Frame.select`) and those
    households' native weights are restored to their pre-clone values.
    Mass is conserved: selected households carry half weight on each arm,
    unselected households carry full weight on their native lineage.
    """

    household = cloned.table("household")
    household_clone = household[support_clone_index_column("household")]
    household_source = household[support_source_id_column("household")]
    native_source_ids = np.sort(
        household_source.loc[household_clone.eq(0)].to_numpy(dtype=np.int64)
    )
    eligible = int(len(native_source_ids))
    requested = int(np.floor(fraction * eligible))
    if requested < 1:
        raise ValueError(
            f"clone_attachment_fraction {fraction!r} floors to zero households "
            f"(floor(fraction * eligible) with eligible={eligible}); the PUF "
            "pass requires at least one attached household."
        )
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(native_source_ids, size=requested, replace=False))
    selected_set = frozenset(int(value) for value in selected)

    person = cloned.table("person")
    person_clone = person[support_clone_index_column("person")]
    person_household_source = _person_household_source_ids(cloned)
    keep_person = person_clone.eq(0).to_numpy() | np.isin(
        person_household_source, selected
    )
    trimmed = cloned.select(keep_person)

    trimmed_household = trimmed.table("household")
    trimmed_clone = trimmed_household[support_clone_index_column("household")]
    trimmed_source = trimmed_household[support_source_id_column("household")]
    restored = np.array(
        trimmed.weights_for("household").values,
        dtype=np.float64,
        copy=True,
    )
    unattached_native = (
        trimmed_clone.eq(0) & ~trimmed_source.isin(list(selected_set))
    ).to_numpy()
    restored[unattached_native] *= 2.0
    weights = {
        entity: trimmed.weights_for(entity) for entity in trimmed.weighted_entities
    }
    weights["household"] = Weights(restored, trimmed.weights_for("household").kind)

    manifest = {
        PUF_CLONE_ATTACHMENT_MANIFEST_KEY: {
            "version": _PUF_CLONE_ATTACHMENT_MANIFEST_VERSION,
            "clone_attachment_fraction": float(fraction),
            "clone_attachment_seed": int(seed),
            "eligible_household_count": eligible,
            "requested_household_count": requested,
            "realized_household_count": requested,
            "exact_count_rule": "floor(fraction * eligible)",
            "selected_household_source_ids_sha256": _source_ids_sha256(selected),
        }
    }
    trimmed_metadata = {**trimmed.metadata, **manifest}
    trimmed_mass_log = trimmed.mass_log
    return Frame(
        {entity: trimmed.table(entity) for entity in trimmed.entities},
        trimmed.schema,
        weights,
        trimmed.strata,
        mass_log=trimmed_mass_log,
        metadata=trimmed_metadata,
    )


def _person_household_source_ids(frame: Frame) -> np.ndarray:
    """Map each person row to its household's assembly-unique source ID."""

    household = frame.table("household")
    lookup = pd.Series(
        household[support_source_id_column("household")].to_numpy(),
        index=household["household_id"].to_numpy(),
    )
    mapped = frame.table("person")["person_household_id"].map(lookup)
    if mapped.isna().any():
        raise ValueError(
            "Clone attachment cannot resolve household source IDs for "
            f"{int(mapped.isna().sum())} person row(s)."
        )
    return mapped.to_numpy(dtype=np.int64)


def validate_puf_clone_attachment(
    frame: Frame,
    *,
    boundary: str,
    expected_fraction: float | None = None,
    expected_seed: int | None = None,
) -> Mapping[str, Any]:
    """Validate the clone-attachment manifest against live clone lineages.

    The realized clone-pair count, the selection digest over live detail-arm
    household source IDs, the floor rule, and per-pair weight symmetry must
    all match the manifest; any mutation of the sample, the counts, or the
    manifest fails closed with a named error.  An explicitly expected full
    attachment uses the metadata-symmetric ordinary clone instead: exact
    native/detail household lineage and pair-weight identity are validated
    and returned as an out-of-frame authority receipt.
    """

    if (expected_fraction is None) != (expected_seed is None):
        raise ValueError(
            f"{boundary}: expected clone attachment fraction and seed must be "
            "provided together."
        )
    if expected_fraction is not None:
        if (
            isinstance(expected_fraction, bool)
            or not isinstance(expected_fraction, (int, float))
            or not np.isfinite(expected_fraction)
            or not 0.0 < float(expected_fraction) <= 1.0
        ):
            raise ValueError(
                f"{boundary}: expected clone attachment fraction must be a "
                f"finite number in (0, 1], got {expected_fraction!r}."
            )
        if (
            isinstance(expected_seed, bool)
            or not isinstance(expected_seed, int)
            or expected_seed < 0
        ):
            raise ValueError(
                f"{boundary}: expected clone attachment seed must be a "
                f"non-negative integer, got {expected_seed!r}."
            )

    manifest = frame.metadata.get(PUF_CLONE_ATTACHMENT_MANIFEST_KEY)
    if expected_fraction is not None and float(expected_fraction) == 1.0:
        if manifest is not None:
            raise ValueError(
                f"{boundary}: full-clone metadata symmetry failed: attachment "
                "manifest must be absent from both full-clone paths."
            )
        return _validate_full_clone_identity(
            frame,
            boundary=boundary,
            expected_seed=expected_seed,
        )
    if manifest is None:
        raise ValueError(
            f"{boundary}: clone attachment manifest "
            f"{PUF_CLONE_ATTACHMENT_MANIFEST_KEY!r} is absent."
        )
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("version") != _PUF_CLONE_ATTACHMENT_MANIFEST_VERSION
    ):
        raise ValueError(f"{boundary}: clone attachment manifest is malformed.")
    fraction = manifest.get("clone_attachment_fraction")
    if not isinstance(fraction, float) or isinstance(fraction, bool):
        raise ValueError(
            f"{boundary}: clone attachment fraction must be a float, got {fraction!r}."
        )
    seed = manifest.get("clone_attachment_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            f"{boundary}: clone attachment seed must be a non-negative "
            f"integer, got {seed!r}."
        )
    if expected_fraction is not None and fraction != float(expected_fraction):
        raise ValueError(
            f"{boundary}: clone attachment fraction {fraction!r} differs from "
            f"the expected fraction {float(expected_fraction)!r}."
        )
    if expected_seed is not None and seed != expected_seed:
        raise ValueError(
            f"{boundary}: clone attachment seed {seed!r} differs from the "
            f"expected seed {expected_seed!r}."
        )

    household = frame.table("household")
    clone_index = household[support_clone_index_column("household")]
    source_ids = household[support_source_id_column("household")]
    native = clone_index.eq(0)
    detail = clone_index.eq(PUF_TAX_DETAIL_CLONE_INDEX)
    eligible = int(native.sum())
    if int(manifest.get("eligible_household_count", -1)) != eligible:
        raise ValueError(
            f"{boundary}: live eligible household count {eligible} differs "
            "from the clone attachment manifest "
            f"{manifest.get('eligible_household_count')!r}."
        )
    requested = int(manifest.get("requested_household_count", -1))
    if requested != int(np.floor(float(fraction) * eligible)):
        raise ValueError(
            f"{boundary}: clone attachment requested count {requested} "
            "violates floor(fraction * eligible) for "
            f"fraction={fraction!r}, eligible={eligible}."
        )
    realized = int(manifest.get("realized_household_count", -1))
    live_detail_ids = np.sort(source_ids.loc[detail].to_numpy(dtype=np.int64))
    if realized != requested or int(detail.sum()) != realized:
        raise ValueError(
            f"{boundary}: live attached clone-pair count {int(detail.sum())} "
            f"differs from the manifest's realized attachment count "
            f"{realized} (requested {requested})."
        )
    live_sha = _source_ids_sha256(live_detail_ids)
    if live_sha != manifest.get("selected_household_source_ids_sha256"):
        raise ValueError(
            f"{boundary}: live attached-household selection digest {live_sha} "
            "differs from the clone attachment manifest digest "
            f"{manifest.get('selected_household_source_ids_sha256')!r}."
        )
    native_ids = set(source_ids.loc[native].to_numpy(dtype=np.int64).tolist())
    orphaned = [int(value) for value in live_detail_ids if int(value) not in native_ids]
    if orphaned:
        raise ValueError(
            f"{boundary}: {len(orphaned)} attached clone lineage(s) have no "
            f"native partner (first: {orphaned[:5]})."
        )

    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    pair_frame = pd.DataFrame(
        {
            "source_id": source_ids.to_numpy(dtype=np.int64),
            "clone_index": clone_index.to_numpy(),
            "weight": weights,
        }
    )
    attached = pair_frame[pair_frame["source_id"].isin(live_detail_ids)]
    by_pair = attached.pivot_table(
        index="source_id",
        columns="clone_index",
        values="weight",
        aggfunc="sum",
    )
    if not np.allclose(
        by_pair[0].to_numpy(dtype=np.float64),
        by_pair[PUF_TAX_DETAIL_CLONE_INDEX].to_numpy(dtype=np.float64),
        rtol=1e-12,
        atol=0.0,
    ):
        raise ValueError(
            f"{boundary}: attached clone pairs must split household mass "
            "evenly between the native and detail arms."
        )
    return manifest


def _validate_full_clone_identity(
    frame: Frame,
    *,
    boundary: str,
    expected_seed: int,
) -> Mapping[str, Any]:
    """Receipt exact full-clone coverage without mutating frame metadata."""

    household = frame.table("household")
    clone_index = household[support_clone_index_column("household")]
    source_ids = household[support_source_id_column("household")]
    native = clone_index.eq(0)
    detail = clone_index.eq(PUF_TAX_DETAIL_CLONE_INDEX)
    weights = frame.weights_for("household").values

    native_ids = source_ids.loc[native].to_numpy(dtype=np.int64)
    detail_ids = source_ids.loc[detail].to_numpy(dtype=np.int64)
    native_order = np.argsort(native_ids, kind="stable")
    detail_order = np.argsort(detail_ids, kind="stable")
    ordered_native_ids = native_ids[native_order]
    ordered_detail_ids = detail_ids[detail_order]
    native_weights = np.ascontiguousarray(weights[native.to_numpy()][native_order])
    detail_weights = np.ascontiguousarray(weights[detail.to_numpy()][detail_order])

    lineages_exact = (
        bool((native | detail).all())
        and len(ordered_native_ids) == len(ordered_detail_ids)
        and len(np.unique(ordered_native_ids)) == len(ordered_native_ids)
        and len(np.unique(ordered_detail_ids)) == len(ordered_detail_ids)
        and np.array_equal(ordered_native_ids, ordered_detail_ids)
    )
    weights_exact = (
        native_weights.dtype == detail_weights.dtype
        and native_weights.shape == detail_weights.shape
        and native_weights.tobytes(order="C") == detail_weights.tobytes(order="C")
    )
    if not lineages_exact or not weights_exact:
        raise ValueError(
            f"{boundary}: full-clone identity failed: native/detail household "
            "lineages or pair weights are not exact."
        )

    return {
        "authority_form": "full_clone_identity_no_manifest",
        "clone_attachment_fraction": 1.0,
        "clone_attachment_seed": expected_seed,
        "eligible_household_count": len(ordered_native_ids),
        "requested_household_count": len(ordered_native_ids),
        "realized_household_count": len(ordered_detail_ids),
        "exact_count_rule": "full native/detail identity",
        "selected_household_source_ids_sha256": _source_ids_sha256(ordered_detail_ids),
    }


def _source_ids_sha256(ids: np.ndarray) -> str:
    payload = json.dumps(
        [int(value) for value in np.asarray(ids).tolist()],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def puf_tax_unit_donor_from_arrays(
    arrays: Mapping[str, Sequence[Any]],
    *,
    adjusted_gross_income: Sequence[Any] | None = None,
    person_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    tax_unit_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    donor_build_summary: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Build a tax-unit donor table from processed PUF array columns.

    The processed PUF array artifact stores person-grain and tax-unit-grain
    arrays in one HDF file. This helper reduces person-grain PUF variables to
    tax-unit aggregates so the shared QRF imputer can fit one tax-unit model
    and later distribute person outputs over the cloned CPS people.

    Args:
        arrays: Mapping from column name to array-like values.
        adjusted_gross_income: Explicit tax-unit-grain AGI used to select the
            published TY2015 SOI Table 2.1 component share for each record.
            Required whenever the processed PUF carries nonzero E19200.
        person_outputs: Person-grain PE input variables to aggregate by tax
            unit.
        tax_unit_outputs: Tax-unit-grain PE input variables to carry or derive.
        donor_build_summary: Optional mutable sink for donor-construction
            provenance. When supplied, receives the field-local mortgage
            quarantine cohort and per-field removed masses.

    Returns:
        A tax-unit donor DataFrame with numeric predictors, requested outputs,
        and a ``weight`` column.
    """

    _require_array_columns(
        arrays,
        ["tax_unit_id", "household_weight", "filing_status", "person_tax_unit_id"],
        label="PUF arrays",
    )
    tax_unit_id = _numeric_array(arrays["tax_unit_id"]).astype("int64")
    person_tax_unit_id = _numeric_array(arrays["person_tax_unit_id"]).astype("int64")

    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": tax_unit_id,
            "weight": _numeric_array(arrays["household_weight"]),
            "filing_status_code": _filing_status_codes(arrays["filing_status"]),
        }
    )
    if adjusted_gross_income is not None:
        # Strict conversion (PR #561 review findings, rounds 1-2):
        # _numeric_array coerces nonnumeric/NaN to 0.0, and errors="raise"
        # alone is parse-strict but not real-number-strict — datetime and
        # timedelta arrays (including NaT) convert to finite epoch/sentinel
        # integers, complex drops its imaginary part, and booleans pass.
        # Any of those would route records to the wrong AGI band with the
        # finiteness check below none the wiser, so non-real dtypes are
        # rejected up front. NaN/inf survive strict conversion and fail the
        # explicit check.
        agi_series = pd.Series(adjusted_gross_income)
        if agi_series.dtype.kind not in "iufO":
            raise TypeError(
                "adjusted_gross_income must be real-valued; got dtype "
                f"{agi_series.dtype}."
            )
        if agi_series.dtype.kind == "O":
            # Element screen (PR #561 review, round 3): object and
            # categorical arrays reach the strict parse below with kind
            # "O", where object-wrapped booleans become 1.0/0.0 and
            # object-wrapped complex keeps its real part with only a
            # warning. Allow only element types whose float conversion is
            # faithful: real numbers, Decimal, strings (strict-parsed),
            # and missing values (which fail the finiteness check).
            inferred = pd.api.types.infer_dtype(agi_series, skipna=True)
            if inferred not in {
                "integer",
                "floating",
                "mixed-integer-float",
                "decimal",
                "string",
                "empty",
            }:
                raise TypeError(
                    f"adjusted_gross_income must be real-valued; got {inferred} values."
                )
        agi = pd.to_numeric(agi_series, errors="raise").to_numpy(dtype=np.float64)
        if len(agi) != len(tax_unit_id):
            raise ValueError(
                "adjusted_gross_income must align one-for-one with tax_unit_id."
            )
        if not np.isfinite(agi).all():
            raise ValueError("adjusted_gross_income must contain only finite values.")
        tax_unit[_E19200_AGI_BAND_COLUMN] = agi
        # Preserve the source-year value solely as donor provenance for the
        # post-QRF capital-gains tail transfer. QRF preparation selects an
        # explicit predictor/output surface, so this never becomes a modeled
        # carrier or a shipped PolicyEngine input.
        tax_unit[PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN] = agi
    person = pd.DataFrame({"tax_unit_id": person_tax_unit_id})
    reserved_outputs = {
        _MORTGAGE_OUTLIER_SCREEN_COLUMN,
        _E19200_AGI_BAND_COLUMN,
    }
    requested_reserved = reserved_outputs.intersection(
        (*person_outputs, *tax_unit_outputs)
    )
    if requested_reserved:
        raise ValueError(
            f"{sorted(requested_reserved)!r} are reserved for donor E19200 "
            "processing and cannot be requested outputs."
        )
    raw_home_mortgage_interest = _person_source_values(
        arrays,
        "home_mortgage_interest",
    )
    if raw_home_mortgage_interest is not None:
        person[_MORTGAGE_OUTLIER_SCREEN_COLUMN] = raw_home_mortgage_interest

    engine = _formula_owned_engine()
    assert_formula_owned_blocklist_current(engine)
    _reject_formula_owned_outputs(person_outputs, tax_unit_outputs, engine=engine)
    person_source_columns = set(person_outputs)
    if "interest_deduction" in tax_unit_outputs:
        person_source_columns.add("home_mortgage_interest")
    for output in sorted(person_source_columns):
        source = _person_source_values(arrays, output)
        if source is not None:
            person[output] = source

    grouped = person.groupby("tax_unit_id", sort=False).sum(numeric_only=True)
    tax_unit = tax_unit.join(grouped, on="tax_unit_id")
    tax_unit["tax_unit_person_count"] = (
        person.groupby("tax_unit_id", sort=False).size().reindex(tax_unit_id).to_numpy()
    )
    for output in person_outputs:
        if output in tax_unit.columns:
            continue
        source = _PERSON_OUTPUT_TAX_UNIT_GRAIN_SOURCES.get(output)
        if source is None or source not in arrays:
            continue
        values = _numeric_array(arrays[source])
        if len(values) == len(tax_unit_id):
            tax_unit[output] = values

    for output in tax_unit_outputs:
        values = _tax_unit_source_values(arrays, tax_unit_id, output, grouped)
        if values is not None:
            tax_unit[output] = values

    required_outputs = [*person_outputs, *tax_unit_outputs]
    missing = [column for column in required_outputs if column not in tax_unit.columns]
    if missing:
        raise ValueError(f"PUF donor cannot derive requested output(s): {missing}.")

    for column in tax_unit.columns:
        if column == "tax_unit_id":
            continue
        tax_unit[column] = pd.to_numeric(tax_unit[column], errors="coerce").fillna(0.0)
    mortgage_quarantine_mask = np.zeros(len(tax_unit), dtype=bool)
    if _MORTGAGE_OUTLIER_SCREEN_COLUMN in tax_unit:
        # Threshold the grouped RAW person value: thresholding the carved
        # mortgage value at the same literal would miss corrupt rows in the
        # $10M-to-$10.75M raw band. Keep the mask while the E19200 split
        # materializes both conserving components, then quarantine only those
        # implicated fields so unrelated donor values remain available.
        mortgage_quarantine_mask = (
            tax_unit[_MORTGAGE_OUTLIER_SCREEN_COLUMN].to_numpy(
                dtype=np.float64,
                copy=False,
            )
            >= US_PUF_DONOR_MORTGAGE_OUTLIER_CEILING
        )
    # Keep the split BEFORE _add_predictor_aliases: no mortgage predictor alias
    # exists today, but if one is ever added it must derive from the decomposed
    # column (aliases skip already-present columns, so a post-alias split would
    # leave a stale total-interest predictor copy).
    _split_us_puf_e19200_components(tax_unit)
    if donor_build_summary is not None and set(PUF_SCHEDULE_D_JOINT_COLUMNS).issubset(
        tax_unit.columns
    ):
        donor_build_summary["capital_gains_before_mortgage_screen"] = (
            puf_processed_capital_gains_stage(tax_unit)
        )
    _quarantine_us_puf_mortgage_fields(
        tax_unit,
        mortgage_quarantine_mask,
        donor_build_summary=donor_build_summary,
    )
    _add_predictor_aliases(tax_unit, PUF_TAX_DETAIL_DEFAULT_PREDICTORS)
    return tax_unit


def _quarantine_us_puf_mortgage_fields(
    donor: pd.DataFrame,
    quarantine_mask: np.ndarray,
    *,
    donor_build_summary: dict[str, object] | None,
) -> None:
    """Zero only E19200-derived fields and record removed donor mass."""

    mask = np.asarray(quarantine_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != len(donor):
        raise ValueError(
            "PUF mortgage quarantine mask must align one-for-one with donor rows."
        )
    weights = donor["weight"].to_numpy(dtype=np.float64, copy=False)
    has_capital_gains = set(PUF_SCHEDULE_D_JOINT_COLUMNS).issubset(donor.columns)
    capital_gains_before = (
        puf_capital_gains_joint_metrics(donor, mask=mask) if has_capital_gains else None
    )
    fields: dict[str, dict[str, float | int]] = {}
    for column in US_PUF_DONOR_MORTGAGE_QUARANTINE_FIELDS:
        if column not in donor:
            continue
        values = donor[column].to_numpy(dtype=np.float64, copy=False)
        screened_values = values[mask]
        screened_weights = weights[mask]
        nonzero = screened_values != 0.0
        positive = screened_values > 0.0
        negative = screened_values < 0.0
        fields[column] = {
            "screened_record_count": int(mask.sum()),
            "screened_nonzero_record_count": int(nonzero.sum()),
            "screened_weight": float(screened_weights.sum()),
            "screened_unweighted_signed_mass": float(screened_values.sum()),
            "screened_weighted_signed_mass": float(
                np.dot(screened_values, screened_weights)
            ),
            "screened_weighted_absolute_mass": float(
                np.dot(np.abs(screened_values), screened_weights)
            ),
            "screened_weighted_positive_mass": float(
                np.dot(screened_values[positive], screened_weights[positive])
            ),
            "screened_weighted_negative_mass": float(
                np.dot(screened_values[negative], screened_weights[negative])
            ),
        }
        donor.loc[mask, column] = 0.0

    if donor_build_summary is not None:
        quarantine: dict[str, object] = {
            "method": "field_local_zero",
            "source_field": "grouped_raw_home_mortgage_interest",
            "comparison": "greater_than_or_equal",
            "ceiling": US_PUF_DONOR_MORTGAGE_OUTLIER_CEILING,
            "screened_record_count": int(mask.sum()),
            "screened_weight": float(weights[mask].sum()),
            "fields": fields,
        }
        if capital_gains_before is not None:
            capital_gains_after = puf_capital_gains_joint_metrics(donor, mask=mask)
            capital_gains_difference = {
                key: capital_gains_after[key] - value
                for key, value in capital_gains_before.items()
            }
            if any(value != 0 for value in capital_gains_difference.values()):
                raise AssertionError(
                    "Field-local PUF mortgage quarantine changed capital gains."
                )
            quarantine["capital_gains_preserved"] = {
                "columns": list(PUF_SCHEDULE_D_JOINT_COLUMNS),
                "before": capital_gains_before,
                "after": capital_gains_after,
                "difference": capital_gains_difference,
            }
        donor_build_summary["mortgage_field_quarantine"] = quarantine


def _split_us_puf_e19200_components(donor: pd.DataFrame) -> None:
    """Split raw E19200 into mortgage and modeled non-mortgage components."""

    if _MORTGAGE_OUTLIER_SCREEN_COLUMN not in donor:
        donor.drop(columns=[_E19200_AGI_BAND_COLUMN], errors="ignore", inplace=True)
        return
    raw_total = donor[_MORTGAGE_OUTLIER_SCREEN_COLUMN].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    has_nonzero_e19200 = bool((raw_total != 0).any())
    if has_nonzero_e19200 and _E19200_AGI_BAND_COLUMN not in donor:
        raise ValueError(
            "adjusted_gross_income is required to split nonzero PUF E19200 "
            "records by the published SOI AGI bands."
        )
    if "investment_interest_expense" in donor:
        existing = donor["investment_interest_expense"].to_numpy(
            dtype=np.float64,
            copy=False,
        )
        if (existing != 0).any():
            raise ValueError(
                "Processed PUF already carries nonzero investment_interest_expense; "
                "refusing to overwrite independently sourced values with the "
                "E19200 residual."
            )
    if not has_nonzero_e19200:
        if "investment_interest_expense" in donor:
            donor["investment_interest_expense"] = np.zeros_like(raw_total)
        donor.drop(
            columns=[
                _MORTGAGE_OUTLIER_SCREEN_COLUMN,
                _E19200_AGI_BAND_COLUMN,
            ],
            errors="ignore",
            inplace=True,
        )
        return

    mortgage, non_mortgage = split_us_puf_e19200_by_agi_band(
        raw_total,
        donor[_E19200_AGI_BAND_COLUMN].to_numpy(dtype=np.float64, copy=False),
    )
    band_share = np.divide(
        mortgage,
        raw_total,
        out=np.ones_like(mortgage),
        where=raw_total != 0,
    )
    for column in _US_PUF_E19200_LINEAGE_DONOR_COLUMNS:
        if column in donor:
            donor[column] = (
                donor[column].to_numpy(dtype=np.float64, copy=False) * band_share
            )
    if "investment_interest_expense" in donor:
        donor["investment_interest_expense"] = non_mortgage
    donor.drop(
        columns=[
            _MORTGAGE_OUTLIER_SCREEN_COLUMN,
            _E19200_AGI_BAND_COLUMN,
        ],
        inplace=True,
    )


def impute_us_puf_tax_detail_support(
    frame: Frame,
    donor_tax_units: pd.DataFrame,
    *,
    predictors: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    person_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    tax_unit_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    seed: int = 0,
    n_estimators: int = 100,
    fit_records: list[FitWeightRecord] | None = None,
    raw_predictions_callback: Callable[[pd.DataFrame], None] | None = None,
    tail_bound_diagnostics: list[dict[str, object]] | None = None,
    require_complete_recipient_predictors: bool = False,
    absent_cells: str = PUF_ABSENT_CELLS_LEGACY_ZERO_FILL,
) -> Frame:
    """Impute PUF-observed inputs onto the PUF support channel.

    Baseline ASEC support rows are left untouched. PUF support rows receive
    tax-unit predictions from the PUF donor; person-grain predicted tax-unit
    totals are distributed over the cloned people using their copied ASEC
    within-tax-unit shares, falling back to the first person in the unit when
    the copied support has no mass for a variable. Boolean QBI targets are
    modeled as tax-unit person counts, snapped to observed integer counts, and
    placed on source-aligned people instead of collapsing every positive count
    onto one person.

    Args:
        fit_records: An optional sink for the build-level weights audit (populace
            #300). When a list is passed, this production fit appends one
            :class:`~populace.build.gates.FitWeightRecord`
            (``US_PUF_SUPPORT_FIT_NAME`` -> the kind the QRF *resolved* to,
            ``"design"`` here) so a release can prove the fit did not silently
            resolve unweighted (:func:`~populace.build.gates.weights_audit_gate`).
            Opt-in: omitting it leaves the imputation byte-for-byte unchanged, so
            existing callers are unaffected. This is the seam a build stage wires
            to run the audit and abort a release on an unweighted fit.
        raw_predictions_callback: Test-only observer called synchronously with
            the complete raw chained draws before any clipping, snapping, or
            finalization. Production callers leave it unset.
        tail_bound_diagnostics: Optional output sink for the per-target tail-bound
            records produced during finalization. Build callers publish these
            records with the QRF-finalization telemetry.
        require_complete_recipient_predictors: Stacked-spine doctrine switch
            (populace#578): build recipient features null-preserving and fail
            closed by name when any recipient row is missing a predictor
            value, instead of the legacy silent zero-fill.
        absent_cells: Finalization policy for cells outside the PUF clone arm
            (see :func:`finalize_us_puf_tax_detail_predictions`).
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("PUF tax-detail support imputation requires the US schema.")
    tax_unit_clone_index = support_clone_index_column("tax_unit")
    person_clone_index = support_clone_index_column("person")
    if tax_unit_clone_index not in frame.table("tax_unit").columns:
        raise ValueError("PUF clone metadata is missing from the tax_unit table.")
    if person_clone_index not in frame.table("person").columns:
        raise ValueError("PUF clone metadata is missing from the person table.")

    engine = _formula_owned_engine()
    assert_formula_owned_blocklist_current(engine)
    _reject_formula_owned_outputs(person_outputs, tax_unit_outputs, engine=engine)
    predictors = tuple(predictors)
    person_outputs = tuple(person_outputs)
    tax_unit_outputs = tuple(tax_unit_outputs)
    outputs = (*person_outputs, *tax_unit_outputs)
    puf_mask = puf_tax_detail_clone_mask(
        frame.table("tax_unit"),
        entity="tax_unit",
    )
    if not puf_mask.any():
        raise ValueError("PUF detail clone has no tax-unit rows.")
    if require_complete_recipient_predictors:
        _require_complete_recipient_predictor_sources(frame, puf_mask, predictors)
    donor_tax_units = donor_tax_units.copy()
    _add_predictor_aliases(donor_tax_units, predictors)
    missing_donor = [
        column
        for column in (*predictors, *outputs, "weight")
        if column not in donor_tax_units
    ]
    if missing_donor:
        raise ValueError(
            f"PUF donor tax-unit table missing column(s): {missing_donor}."
        )

    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("populace.fit").QRF

    donor = donor_tax_units.loc[:, [*predictors, *outputs, "weight"]].copy()
    for column in donor.columns:
        donor[column] = pd.to_numeric(donor[column], errors="coerce").fillna(0.0)
    donor_frame = _tax_unit_model_frame(donor)
    fitted = QRF(n_estimators=n_estimators, seed=seed).fit(
        donor_frame,
        list(predictors),
        list(outputs),
        weights="design",
    )
    if fit_records is not None:
        # Record the kind the fit *resolved* to (not the "design" spec above):
        # the build-level weights audit reads this back to prove the production
        # fit did not silently resolve unweighted (populace #300).
        fit_records.append(FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, fitted.weight_kind))

    features = _tax_unit_feature_frame(
        frame,
        predictors,
        preserve_nulls=require_complete_recipient_predictors,
    )
    if require_complete_recipient_predictors:
        _require_complete_recipient_predictors(features, puf_mask, predictors)
    predictions = fitted.predict(
        features.loc[puf_mask, list(predictors)], release_models=True
    )
    if raw_predictions_callback is not None:
        raw_predictions_callback(predictions)
    return finalize_us_puf_tax_detail_predictions(
        frame,
        donor,
        predictions,
        person_outputs=person_outputs,
        tax_unit_outputs=tax_unit_outputs,
        tail_bound_diagnostics=tail_bound_diagnostics,
        absent_cells=absent_cells,
    )


def prepare_us_puf_tax_detail_chain_inputs(
    frame: Frame,
    donor_tax_units: pd.DataFrame,
    *,
    predictors: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    person_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    tax_unit_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    require_complete_recipient_predictors: bool = False,
) -> PufTaxDetailChainInputs:
    """Prepare lossless donor and recipient inputs for targetwise QRF workers.

    ``require_complete_recipient_predictors`` is the stacked-spine doctrine
    switch (populace#578): recipient features are built null-preserving and a
    missing predictor value on any recipient row is a named terminal failure
    instead of a silent zero-fill.  The legacy default keeps the historical
    zero-filled feature surface byte for byte.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("PUF tax-detail support imputation requires the US schema.")
    tax_unit_clone_index = support_clone_index_column("tax_unit")
    person_clone_index = support_clone_index_column("person")
    if tax_unit_clone_index not in frame.table("tax_unit").columns:
        raise ValueError("PUF clone metadata is missing from the tax_unit table.")
    if person_clone_index not in frame.table("person").columns:
        raise ValueError("PUF clone metadata is missing from the person table.")

    engine = _formula_owned_engine()
    assert_formula_owned_blocklist_current(engine)
    _reject_formula_owned_outputs(person_outputs, tax_unit_outputs, engine=engine)
    predictors = tuple(predictors)
    person_outputs = tuple(person_outputs)
    tax_unit_outputs = tuple(tax_unit_outputs)
    outputs = (*person_outputs, *tax_unit_outputs)
    puf_mask = puf_tax_detail_clone_mask(
        frame.table("tax_unit"),
        entity="tax_unit",
    )
    if not puf_mask.any():
        raise ValueError("PUF detail clone has no tax-unit rows.")
    if require_complete_recipient_predictors:
        _require_complete_recipient_predictor_sources(frame, puf_mask, predictors)
    donor_tax_units = donor_tax_units.copy()
    _add_predictor_aliases(donor_tax_units, predictors)
    missing_donor = [
        column
        for column in (*predictors, *outputs, "weight")
        if column not in donor_tax_units
    ]
    if missing_donor:
        raise ValueError(
            f"PUF donor tax-unit table missing column(s): {missing_donor}."
        )

    donor = donor_tax_units.loc[:, [*predictors, *outputs, "weight"]].copy()
    for column in donor.columns:
        donor[column] = pd.to_numeric(donor[column], errors="coerce").fillna(0.0)
    donor_frame = _tax_unit_model_frame(donor)
    features = _tax_unit_feature_frame(
        frame,
        predictors,
        preserve_nulls=require_complete_recipient_predictors,
    )
    if require_complete_recipient_predictors:
        _require_complete_recipient_predictors(features, puf_mask, predictors)
    recipient_features = features.loc[puf_mask, list(predictors)].copy()
    recipient_tax_unit_ids = (
        frame.table("tax_unit").loc[puf_mask, "tax_unit_id"].to_numpy(copy=True)
    )
    return PufTaxDetailChainInputs(
        donor=donor,
        donor_frame=donor_frame,
        recipient_features=recipient_features,
        recipient_tax_unit_ids=recipient_tax_unit_ids,
        predictors=predictors,
        person_outputs=person_outputs,
        tax_unit_outputs=tax_unit_outputs,
    )


def finalize_us_puf_tax_detail_predictions(
    frame: Frame,
    donor: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    person_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    tax_unit_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    tail_bound_diagnostics: list[dict[str, object]] | None = None,
    tail_bound_quantiles: Mapping[str, float] | None = None,
    absent_cells: str = PUF_ABSENT_CELLS_LEGACY_ZERO_FILL,
) -> Frame:
    """Finalize a complete raw PUF QRF chain onto its support channel.

    Raw draws must arrive in the exact person-then-tax-unit chain order and
    retain the recipient tax-unit index.  Clipping, snapping, reconciliation,
    placement, and sparsification happen only here, after every target has
    drawn; later targets therefore always condition on raw predecessor draws.

    The module tail-bound configuration is validated against the canonical
    production surface and applies whenever a configured target is present.
    Deliberately reduced chains disjoint from every configured target remain
    isolated. Explicit test configurations are instead validated against the
    invocation's exact output surface. Diagnostics report recipient-design-
    weighted mass over the affected raw tax-unit draws before and after
    clipping; build callers must provide the sink and publish every active cap.

    ``absent_cells`` selects the finalization policy for cells outside the PUF
    clone arm (populace#578 audit item 1).  The legacy default reproduces the
    historical global zero-fill byte for byte.  Under
    :data:`PUF_ABSENT_CELLS_PRESERVE_NULLS` — the stacked-spine doctrine —
    absence stays null on every row this pass does not own, and the
    donor-rate sparsification and placement rewrites are scoped to the PUF
    clone arm so no boundary converts absence into an observed zero.
    """

    if absent_cells not in _PUF_ABSENT_CELLS_POLICIES:
        raise ValueError(
            f"absent_cells must be one of {list(_PUF_ABSENT_CELLS_POLICIES)}; "
            f"got {absent_cells!r}."
        )
    preserve_nulls = absent_cells == PUF_ABSENT_CELLS_PRESERVE_NULLS
    person_outputs = tuple(person_outputs)
    tax_unit_outputs = tuple(tax_unit_outputs)
    outputs = (*person_outputs, *tax_unit_outputs)
    if tail_bound_quantiles is None:
        canonical_outputs = (
            *PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
            *PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
        )
        configured_tail_bounds = _validated_tail_bound_quantiles(
            canonical_outputs,
            _PUF_TAX_DETAIL_TAIL_BOUND_QUANTILES,
        )
        requested_outputs = set(outputs)
        active_tail_bounds = {
            output: quantile
            for output, quantile in configured_tail_bounds.items()
            if output in requested_outputs
        }
    else:
        active_tail_bounds = _validated_tail_bound_quantiles(
            outputs,
            tail_bound_quantiles,
        )
    if tuple(predictions.columns) != outputs:
        raise ValueError(
            "PUF raw prediction columns must match the exact target order: "
            f"expected {list(outputs)}, got {list(predictions.columns)}."
        )
    missing_donor = [column for column in (*outputs, "weight") if column not in donor]
    if missing_donor:
        raise ValueError(f"PUF finalization donor missing column(s): {missing_donor}.")

    resolved_tail_bounds: dict[str, tuple[float, float]] = {}
    for output, quantile in active_tail_bounds.items():
        try:
            bound = _weighted_positive_donor_quantile(
                donor[output], donor["weight"], quantile
            )
        except ValueError as exc:
            if "no positive donor support" not in str(exc):
                raise
            raise ValueError(
                f"PUF tax-detail tail-bound output {output!r} has no positive "
                "donor support."
            ) from None
        resolved_tail_bounds[output] = (quantile, bound)
    if resolved_tail_bounds and tail_bound_diagnostics is None:
        raise ValueError(
            "PUF tax-detail tail-bound finalization requires a diagnostics sink; "
            "tail bounds must not be silent."
        )

    tax_unit_clone_index = support_clone_index_column("tax_unit")
    person_clone_index = support_clone_index_column("person")
    puf_mask = puf_tax_detail_clone_mask(
        frame.table("tax_unit"),
        entity="tax_unit",
    )
    expected_index = frame.table("tax_unit").index[puf_mask]
    if not predictions.index.equals(expected_index):
        raise ValueError(
            "PUF raw predictions changed recipient row order or index before "
            "finalization."
        )
    if resolved_tail_bounds:
        recipient_tax_unit_ids = frame.table("tax_unit").loc[puf_mask, "tax_unit_id"]
        recipient_weights = _tax_unit_household_weights(
            frame,
            recipient_tax_unit_ids,
        )

        for output, (quantile, bound) in resolved_tail_bounds.items():
            values = predictions[output].to_numpy(dtype=np.float64, copy=False)
            clipped = values > bound
            clipped_mass_before = float(
                (values[clipped] * recipient_weights[clipped]).sum()
            )
            predictions.loc[clipped, output] = bound
            clipped_mass_after = float(
                (
                    predictions.loc[clipped, output].to_numpy(dtype=np.float64)
                    * recipient_weights[clipped]
                ).sum()
            )
            tail_bound_diagnostics.append(
                {
                    "output": output,
                    "quantile": quantile,
                    "bound_value": bound,
                    "clipped_row_count": int(clipped.sum()),
                    "clipped_mass_before": clipped_mass_before,
                    "clipped_mass_after": clipped_mass_after,
                }
            )

    for column in outputs:
        if column in _PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS:
            predictions[column] = predictions[column].clip(lower=0.0)
        if column in _PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS:
            predictions[column] = _snap_to_observed_values(
                predictions[column],
                donor[column],
            )
        if column in _PUF_TAX_DETAIL_SPARSE_TAX_UNIT_OUTPUTS:
            predictions[column] = _snap_to_observed_values(
                predictions[column],
                donor[column],
            )
        if column in _PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS:
            predictions[column] = _snap_to_observed_values(
                predictions[column],
                donor[column],
            )
        if column in _PUF_TAX_DETAIL_DISCRETE_TAX_UNIT_OUTPUTS:
            predictions[column] = _snap_to_observed_values(
                predictions[column],
                donor[column],
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tax_unit_ids = tables["tax_unit"].loc[puf_mask, "tax_unit_id"].to_numpy()
    _reconcile_puf_social_security_components(
        predictions,
        tables["person"],
        person_clone_index=person_clone_index,
        tax_unit_ids=tax_unit_ids,
        requested_components=person_outputs,
    )
    for column in tax_unit_outputs:
        _ensure_float_output_column(
            tables["tax_unit"],
            column,
            preserve_nulls=preserve_nulls,
        )
        tables["tax_unit"].loc[puf_mask, column] = predictions[column].to_numpy()
    for column in tax_unit_outputs:
        if column in _PUF_TAX_DETAIL_SPARSE_TAX_UNIT_OUTPUTS:
            _sparsify_tax_unit_output_to_donor_positive_rate(
                tables,
                column=column,
                donor_positive_rate=_weighted_positive_rate(
                    donor[column],
                    donor["weight"],
                ),
                household_weights=frame.weights_for("household").values,
                tax_unit_clone_index=tax_unit_clone_index,
                puf_role_only=preserve_nulls,
            )

    person_puf_mask = puf_tax_detail_clone_mask(
        tables["person"],
        entity="person",
    )
    for column in person_outputs:
        _ensure_float_output_column(
            tables["person"],
            column,
            preserve_nulls=preserve_nulls,
        )
        totals = pd.Series(predictions[column].to_numpy(), index=tax_unit_ids)
        if column in _PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS:
            _write_person_tax_unit_boolean_counts(
                tables["person"],
                mask=person_puf_mask,
                column=column,
                totals=totals,
                fallback_basis_columns=_PERSON_OUTPUT_DISTRIBUTION_BASIS.get(
                    column, ()
                ),
            )
        else:
            _write_person_tax_unit_totals(
                tables["person"],
                mask=person_puf_mask,
                column=column,
                totals=totals,
                nonnegative=column in _PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS,
                fallback_basis_columns=_PERSON_OUTPUT_DISTRIBUTION_BASIS.get(
                    column, ()
                ),
            )
    for column in person_outputs:
        if column in _PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS:
            _sparsify_tax_unit_person_output_to_donor_positive_rate(
                tables,
                column=column,
                donor_positive_rate=_weighted_positive_rate(
                    donor[column],
                    donor["weight"],
                ),
                household_weights=frame.weights_for("household").values,
                person_clone_index=person_clone_index,
                tax_unit_clone_index=tax_unit_clone_index,
                puf_role_only=preserve_nulls,
            )
    for column in person_outputs:
        if column in _PUF_TAX_DETAIL_SIGNED_MASS_CALIBRATED_PERSON_OUTPUTS:
            _calibrate_tax_unit_person_output_signed_mass_to_donor(
                tables,
                column=column,
                donor_values=donor[column],
                donor_weights=donor["weight"],
                household_weights=frame.weights_for("household").values,
                person_clone_index=person_clone_index,
                tax_unit_clone_index=tax_unit_clone_index,
            )
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _clone_entity_table(
    table: pd.DataFrame,
    *,
    entity: str,
    schema: EntitySchema,
    channels: tuple[str, ...],
    id_multiplier: int,
) -> pd.DataFrame:
    id_columns = _entity_id_columns(schema, entity)
    source_id = support_source_id_column(entity)
    channel_column = support_channel_column(entity)
    clone_index_column = support_clone_index_column(entity)
    primary_id = schema.entity_id_column(entity)

    clones: list[pd.DataFrame] = []
    for clone_index, channel in enumerate(channels):
        clone = table.copy(deep=True)
        clone[source_id] = table[primary_id].to_numpy()
        clone[channel_column] = channel
        clone[clone_index_column] = clone_index
        for column in id_columns:
            clone[column] = _remap_ids(
                clone[column].to_numpy(),
                clone_index=clone_index,
                id_multiplier=id_multiplier,
            )
        clones.append(clone)
    result = pd.concat(clones, ignore_index=True)
    if result[primary_id].duplicated().any():
        duplicates = result.loc[result[primary_id].duplicated(), primary_id].unique()
        raise ValueError(
            f"remapped {primary_id!r} values are not unique; id multiplier "
            "is too small. Duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )
    return result


def _clone_preassembled_entity_table(
    table: pd.DataFrame,
    *,
    entity: str,
    schema: EntitySchema,
    id_multiplier: int,
) -> pd.DataFrame:
    """Clone one preassembled entity table while preserving source provenance."""

    id_columns = _entity_id_columns(schema, entity)
    clone_index_column = support_clone_index_column(entity)
    primary_id = schema.entity_id_column(entity)

    native = table.copy(deep=True)
    detail = table.copy(deep=True)
    detail[clone_index_column] = PUF_TAX_DETAIL_CLONE_INDEX
    for column in id_columns:
        detail[column] = _remap_ids(
            detail[column].to_numpy(),
            clone_index=PUF_TAX_DETAIL_CLONE_INDEX,
            id_multiplier=id_multiplier,
        )
    result = pd.concat([native, detail], ignore_index=True)
    if result[primary_id].duplicated().any():
        duplicates = result.loc[result[primary_id].duplicated(), primary_id].unique()
        raise ValueError(
            f"remapped {primary_id!r} values are not unique; id multiplier "
            "is too small. Duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )
    return result


def _clone_link_table(
    table: pd.DataFrame,
    *,
    link_name: str,
    schema: EntitySchema,
    clone_count: int,
    id_multiplier: int,
) -> pd.DataFrame:
    links = {link.name: link for link in schema.links}
    link = links[link_name]
    id_columns = (
        schema.entity_id_column(link.left_entity),
        schema.entity_id_column(link.right_entity),
    )
    missing = [column for column in id_columns if column not in table]
    if missing:
        raise ValueError(
            f"link table {link_name!r} is missing ID column(s): {missing}."
        )

    clones: list[pd.DataFrame] = []
    for clone_index in range(clone_count):
        clone = table.copy(deep=True)
        for column in id_columns:
            clone[column] = _remap_ids(
                clone[column].to_numpy(),
                clone_index=clone_index,
                id_multiplier=id_multiplier,
            )
        clones.append(clone)
    return pd.concat(clones, ignore_index=True)


def _entity_id_columns(schema: EntitySchema, entity: str) -> tuple[str, ...]:
    if entity == schema.person_entity:
        return (
            schema.person_id_column,
            *(schema.membership_column(group) for group in schema.group_entities),
        )
    return (schema.entity_id_column(entity),)


def _validate_channels(channels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(channels, str):
        raise ValueError("support channels must be a sequence of names, not a string.")
    values = tuple(channels)
    if len(values) < 2:
        raise ValueError("PUF support expansion requires at least two channels.")
    bad = [value for value in values if not isinstance(value, str) or not value]
    if bad:
        raise ValueError("support channels must be non-empty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"support channels must be unique, got {values!r}.")
    return values


def _validated_integral_ids(
    values: Sequence[Any],
    *,
    label: str,
) -> np.ndarray:
    """Return int64 IDs only when every input value is exactly integral."""

    try:
        numeric = pd.to_numeric(pd.Series(values), errors="raise")
        as_float = numeric.to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be integral.") from exc
    if (
        not np.isfinite(as_float).all()
        or not np.equal(as_float, np.floor(as_float)).all()
    ):
        raise ValueError(f"{label} must be integral.")
    try:
        integral = numeric.astype("int64").to_numpy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be integral.") from exc
    if not np.equal(as_float, integral.astype(np.float64)).all():
        raise ValueError(f"{label} must be integral.")
    return integral


def _has_native_assembly_provenance(frame: Frame) -> bool:
    """Validate support metadata and identify a preassembled native frame."""

    expected_by_entity = {
        entity: {
            support_source_id_column(entity),
            support_channel_column(entity),
            support_clone_index_column(entity),
        }
        for entity in frame.entities
    }
    present_by_entity = {
        entity: expected & set(frame.table(entity).columns)
        for entity, expected in expected_by_entity.items()
    }
    present = {column for columns in present_by_entity.values() for column in columns}
    if not present:
        return False
    missing = sorted(
        column
        for entity, expected in expected_by_entity.items()
        for column in expected - present_by_entity[entity]
    )
    if missing:
        raise ValueError(
            f"PUF support expansion metadata is partial; missing column(s): {missing}."
        )

    for entity in frame.entities:
        table = frame.table(entity)
        clone_index_column = support_clone_index_column(entity)
        clone_indices = pd.to_numeric(
            table[clone_index_column],
            errors="coerce",
        )
        if (
            clone_indices.isna().any()
            or not (clone_indices.to_numpy(dtype=np.float64) == 0.0).all()
        ):
            raise ValueError(
                "PUF support expansion metadata column(s) already exist. "
                "The stage should run exactly once."
            )

    missing_spine_source = sorted(
        spine_source_id_column(entity)
        for entity in frame.entities
        if spine_source_id_column(entity) not in frame.table(entity)
    )
    if missing_spine_source:
        raise ValueError(
            "Preassembled support provenance is incomplete; missing raw spine "
            f"source ID column(s): {missing_spine_source}."
        )

    for entity in frame.entities:
        table = frame.table(entity)
        channel_column = support_channel_column(entity)
        channels = table[channel_column]
        if (
            channels.isna().any()
            or not channels.map(
                lambda value: isinstance(value, str) and bool(value)
            ).all()
        ):
            raise ValueError(
                f"Preassembled support channel {channel_column!r} must contain "
                "non-empty strings."
            )
        for source_id_column in (
            support_source_id_column(entity),
            spine_source_id_column(entity),
        ):
            _validated_integral_ids(
                table[source_id_column],
                label=f"Preassembled source IDs in {source_id_column!r}",
            )
    return True


def _reject_metadata_collisions(
    frame: Frame,
    channels: tuple[str, ...],
) -> None:
    expected = {
        column
        for entity in frame.entities
        for column in (
            support_source_id_column(entity),
            spine_source_id_column(entity),
            support_channel_column(entity),
            support_clone_index_column(entity),
        )
    }
    existing = {
        column for entity in frame.entities for column in frame.table(entity).columns
    }
    collisions = sorted(expected & existing)
    if collisions:
        raise ValueError(
            "PUF support expansion metadata column(s) already exist: "
            f"{collisions}. The stage should run exactly once."
        )
    if channels[0] != BASE_ASEC_SUPPORT_CHANNEL:
        raise ValueError(
            f"support channels must start with {BASE_ASEC_SUPPORT_CHANNEL!r} "
            "so the baseline ASEC channel keeps the original IDs."
        )
    if PUF_TAX_DETAIL_SUPPORT_CHANNEL not in channels:
        raise ValueError(
            f"support channels must include {PUF_TAX_DETAIL_SUPPORT_CHANNEL!r}."
        )
    if channels != _DEFAULT_SUPPORT_CHANNELS:
        raise ValueError(
            "PUF support expansion accepts only the canonical ASEC/PUF roles "
            f"{_DEFAULT_SUPPORT_CHANNELS!r}; got {channels!r}."
        )


def _id_multiplier_for_frame(frame: Frame) -> int:
    values = [
        frame.table(entity)[frame.schema.entity_id_column(entity)]
        for entity in frame.entities
    ]
    return _id_multiplier_for_values(*values)


# The decimal remap (id + clone_index * 10**digits(max_id)) must stay inside
# int64 for every clone index the builder can produce. Capping assembled
# source IDs at 10**15 - 1 bounds the multiplier at 10**16, leaving clone
# indices up to 921 before int64 overflow — orders beyond any configured
# clone count. Assembly enforces this bound; _remap_ids re-checks it so a
# violation is a governed ValueError, never an OverflowError.
PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID = 10**15 - 1

_INT64_MAX = 2**63 - 1


def _id_multiplier_for_values(*values: Sequence[Any]) -> int:
    if not values:
        raise ValueError("at least one ID value sequence is required.")
    max_id = 0
    for sequence in values:
        numeric = _validated_integral_ids(
            sequence,
            label="PUF support structural IDs",
        )
        if len(numeric):
            max_id = max(max_id, int(np.abs(numeric).max()))
    return 10 ** max(1, len(str(max_id)))


def _remap_ids(
    ids: Sequence[Any],
    *,
    clone_index: int,
    id_multiplier: int,
) -> np.ndarray:
    values = _validated_integral_ids(
        ids,
        label="PUF support structural IDs",
    )
    if clone_index == 0:
        return values.copy()
    shift = int(clone_index) * int(id_multiplier)
    if len(values) and shift + int(values.max()) > _INT64_MAX:
        raise ValueError(
            "PUF support ID remap would overflow int64: max source ID "
            f"{int(values.max())} with clone index {clone_index} and "
            f"multiplier {id_multiplier} exceeds {_INT64_MAX}. Assembled "
            "source IDs must not exceed "
            f"{PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID}."
        )
    return values + shift


def _tax_unit_model_frame(donor: pd.DataFrame) -> Frame:
    n = len(donor)
    tax_unit = donor.drop(columns=["weight"]).copy()
    tax_unit.insert(0, "tax_unit_id", np.arange(1, n + 1, dtype="int64"))
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1, dtype="int64"),
            "person_tax_unit_id": tax_unit["tax_unit_id"].to_numpy(),
        }
    )
    schema = EntitySchema(group_entities=("tax_unit",))
    return Frame(
        {"person": person, "tax_unit": tax_unit},
        schema,
        {
            "tax_unit": Weights(
                donor["weight"].to_numpy(dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _validated_tail_bound_quantiles(
    outputs: Sequence[str],
    configured: Mapping[str, float],
) -> dict[str, float]:
    """Validate tail-bound configuration against this finalization surface."""

    transformed_outputs = (
        _PUF_TAX_DETAIL_DISCRETE_TAX_UNIT_OUTPUTS
        | _PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS
        | _PUF_TAX_DETAIL_SPARSE_TAX_UNIT_OUTPUTS
        | _PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS
        | _PUF_TAX_DETAIL_SIGNED_MASS_CALIBRATED_PERSON_OUTPUTS
    )
    requested = set(outputs)
    active: dict[str, float] = {}
    for output, configured_quantile in configured.items():
        if output not in requested:
            raise ValueError(
                f"PUF tax-detail tail-bound configured output {output!r} is "
                "missing from outputs."
            )
        try:
            quantile = float(configured_quantile)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"PUF tax-detail tail-bound quantile for {output!r} must be in "
                f"(0, 1), got {configured_quantile!r}."
            ) from exc
        if not np.isfinite(quantile) or not 0.0 < quantile < 1.0:
            raise ValueError(
                f"PUF tax-detail tail-bound quantile for {output!r} must be in "
                f"(0, 1), got {configured_quantile!r}."
            )
        if output in transformed_outputs:
            raise ValueError(
                f"PUF tax-detail output {output!r}: tail bound is defined for "
                "passthrough outputs only."
            )
        active[output] = quantile
    return active


def _weighted_positive_donor_quantile(
    values: Sequence[Any],
    weights: Sequence[Any],
    quantile: float,
) -> float:
    """Return an inverse-CDF quantile over positive, weighted donor support."""

    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=np.float64
    )
    numeric_weights = pd.to_numeric(pd.Series(weights), errors="coerce").to_numpy(
        dtype=np.float64
    )
    if len(numeric_values) != len(numeric_weights):
        raise ValueError(
            "PUF tail-bound donor values and weights must align, got "
            f"{len(numeric_values)} values and {len(numeric_weights)} weights."
        )
    positive = numeric_values > 0.0
    if not (positive & (numeric_weights > 0.0)).any():
        raise ValueError("PUF tail-bound donor has no positive donor support.")
    positive_donor = pd.DataFrame(
        {
            "tail_bound_value": numeric_values[positive],
            "weight": numeric_weights[positive],
        }
    )
    positive_frame = _tax_unit_model_frame(positive_donor)
    return float(
        wquantile(
            positive_frame,
            "tail_bound_value",
            quantile,
            entity="tax_unit",
        )
    )


def _tax_unit_household_weights(
    frame: Frame,
    tax_unit_ids: Sequence[Any],
) -> np.ndarray:
    """Resolve household design weights for tax units in the requested order."""

    household = frame.table("household")
    household_weights = frame.weights_for("household").values
    if len(household_weights) != len(household):
        raise ValueError(
            "Household weights must align with household rows, got "
            f"{len(household_weights)} weights for {len(household)} households."
        )
    weight_by_household_id = pd.Series(
        np.asarray(household_weights, dtype=np.float64),
        index=household["household_id"].to_numpy(),
    )
    household_id_by_tax_unit_id = (
        frame.table("person")
        .groupby("person_tax_unit_id", sort=False)["person_household_id"]
        .first()
    )
    requested = pd.Series(np.asarray(tax_unit_ids), dtype=object)
    resolved = requested.map(household_id_by_tax_unit_id).map(weight_by_household_id)
    if resolved.isna().any():
        missing = requested.loc[resolved.isna()].tolist()
        raise ValueError(
            "Could not resolve household weights for PUF recipient tax unit(s): "
            f"{missing}."
        )
    return resolved.to_numpy(dtype=np.float64)


def _formula_owned_engine() -> Any | None:
    """An import-free PolicyEngine-US source index, or ``None`` if absent.

    Importing ``policyengine_us`` constructs a full tax-benefit system, and a
    second adapter system registers another complete set of variable modules.
    The ownership guard needs only class metadata, so the source index parses
    the installed variable declarations without importing the country package.
    A missing ``[us]`` extra still degrades to the static seed set.
    """
    try:
        from populace.frame.adapters.policyengine_us import (
            PolicyEngineUSVariableMetadataIndex,
        )
    except ImportError:
        return None
    try:
        return PolicyEngineUSVariableMetadataIndex()
    except ImportError:
        return None


def resolve_formula_owned_outputs(
    requested: Iterable[str],
    *,
    engine: Any | None = None,
) -> set[str]:
    """Return the formula-owned names among ``requested``.

    The rejection set is the union of two sources:

    - the static :data:`PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS` seed set (always
      applied, so the guard works even without ``policyengine_us`` installed),
      and
    - the set PolicyEngine-US variable metadata reports as formula-owned for
      exactly the requested names (a variable with a formula, an
      ``adds``/``subtracts`` aggregation, or a compatibility-blocked aggregate)
      — the part that keeps the guard current as the engine adds variables
      (populace issue #301).

    Deriving the second source from metadata means a newly added formula-owned
    aggregate is rejected the moment it appears in a requested output list, with
    no edit to the static set required.

    Args:
        requested: The output variable names a fit intends to impute/persist.
        engine: A metadata source exposing ``formula_owned_outputs(names) ->
            set[str]`` (a
            :class:`~populace.frame.adapters.policyengine_us.PolicyEngineUSVariableMetadataIndex`
            in production). ``None`` resolves one lazily, falling back to the
            static seed set when ``policyengine_us`` is not installed.

    Returns:
        The subset of ``requested`` that is formula-owned.
    """
    requested_set = set(requested)
    rejected = requested_set & PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS
    if engine is None:
        engine = _formula_owned_engine()
    if engine is not None:
        try:
            rejected |= set(engine.formula_owned_outputs(requested_set))
        except ImportError:
            # An injected metadata source may resolve the optional package
            # lazily; degrade to the static seed exactly as when unavailable.
            pass
    return rejected


def assert_formula_owned_blocklist_current(engine: Any | None = None) -> None:
    """Fail if the static seed set has drifted from engine metadata.

    Every name in :data:`PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS` must still be
    formula-owned according to PolicyEngine-US, so a stale entry (a variable the
    engine has since turned into a plain input, or renamed away) cannot silently
    linger in the blocklist and wrongly reject a legitimate leaf. This is the
    reverse-direction check to :func:`resolve_formula_owned_outputs`: the latter
    catches formula-owned names *missing* from the static set, this catches
    static names the engine no longer *considers* formula-owned.

    A no-op when no metadata index is available or an injected source reports
    the optional package missing, so the check runs only where the ``[us]``
    extra is installed — the build.

    Raises:
        ValueError: If a static-set entry is not reported formula-owned by the
            engine, naming the drifted entries.
    """
    if engine is None:
        engine = _formula_owned_engine()
    if engine is None:
        return
    try:
        still_formula_owned = engine.formula_owned_outputs(
            PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS
        )
    except ImportError:
        # Missing [us] extra at call time: same no-op as having no engine.
        return
    drifted = sorted(PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS - set(still_formula_owned))
    if drifted:
        raise ValueError(
            "PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS has stale entries no longer "
            f"reported as formula-owned by PolicyEngine-US: {drifted}. Remove "
            "them (each blocklist entry must still be a formula-owned output)."
        )


def _reject_formula_owned_outputs(
    person_outputs: Sequence[str],
    tax_unit_outputs: Sequence[str],
    *,
    engine: Any | None = None,
) -> None:
    requested = set(person_outputs) | set(tax_unit_outputs)
    formula_owned = sorted(resolve_formula_owned_outputs(requested, engine=engine))
    if formula_owned:
        raise ValueError(
            "PUF tax-detail support outputs must be PolicyEngine leaf inputs, "
            f"not formula-owned aggregate outputs: {formula_owned}."
        )


def _tax_unit_feature_frame(
    frame: Frame,
    columns: Sequence[str],
    *,
    preserve_nulls: bool = False,
) -> pd.DataFrame:
    """Build the tax-unit predictor surface under the active absence policy.

    The legacy policy zero-fills every missing predictor cell — the exact
    boundary the populace#578 audit identified as collapsing recipient draws
    to zero-conditioned degenerates.  Under ``preserve_nulls`` absence
    propagates as null (a leaf-alias sum is null wherever any component is
    null, and an entirely absent component column is null everywhere) so the
    strict recipient check can fail closed by name instead of a silent fill.
    Structural person counts are not absence and stay zero-filled.
    """

    tax_unit = frame.table("tax_unit")
    person = frame.table("person")
    result = pd.DataFrame(index=tax_unit.index)
    for column in columns:
        source_column = _predictor_source_column(column)
        if source_column == "filing_status_code":
            source = tax_unit.get("filing_status_input")
            if source is None:
                source = tax_unit.get("filing_status")
            if source is None:
                raise ValueError("tax_unit table lacks filing-status input.")
            result[column] = _filing_status_codes(
                source,
                preserve_nulls=preserve_nulls,
            )
        elif source_column == "tax_unit_person_count":
            result[column] = (
                person.groupby("person_tax_unit_id", sort=False)
                .size()
                .reindex(tax_unit["tax_unit_id"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        elif source_column in tax_unit.columns:
            numeric = pd.to_numeric(
                tax_unit[source_column],
                errors="raise" if preserve_nulls else "coerce",
            )
            result[column] = numeric if preserve_nulls else numeric.fillna(0.0)
        else:
            result[column] = _person_tax_unit_sum(
                frame,
                source_column,
                preserve_nulls=preserve_nulls,
            )
    return result


def _person_tax_unit_sum(
    frame: Frame,
    column: str,
    *,
    preserve_nulls: bool = False,
) -> np.ndarray:
    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    if column == "dividend_income" and column not in person.columns:
        values = _optional_person(
            person,
            "non_qualified_dividend_income",
            preserve_nulls=preserve_nulls,
        ) + _optional_person(
            person,
            "qualified_dividend_income",
            preserve_nulls=preserve_nulls,
        )
    elif column not in person.columns and column in _PREDICTOR_LEAF_ALIASES:
        values = np.zeros(len(person), dtype=np.float64)
        for leaf in _PREDICTOR_LEAF_ALIASES[column]:
            values += _optional_person(person, leaf, preserve_nulls=preserve_nulls)
    else:
        if column not in person.columns:
            raise ValueError(
                f"Cannot build tax-unit predictor {column!r}; no matching "
                "tax_unit column or person column exists."
            )
        numeric = pd.to_numeric(
            person[column],
            errors="raise" if preserve_nulls else "coerce",
        )
        if not preserve_nulls:
            numeric = numeric.fillna(0.0)
        values = numeric
    grouped_frame = pd.DataFrame(
        {
            "person_tax_unit_id": person["person_tax_unit_id"],
            column: np.asarray(values, dtype=np.float64),
        }
    ).groupby("person_tax_unit_id", sort=False)[column]
    if preserve_nulls:
        # ``sum`` normally skips partial nulls, which would turn a missing
        # member-level source into an observed unit total.  Poison any such
        # group so strict recipient validation sees the absence by name.
        grouped = grouped_frame.sum(min_count=1)
        grouped[grouped_frame.count() != grouped_frame.size()] = np.nan
        return grouped.reindex(tax_unit["tax_unit_id"]).to_numpy()
    grouped = grouped_frame.sum()
    return grouped.reindex(tax_unit["tax_unit_id"]).fillna(0.0).to_numpy()


def _optional_person(
    person: pd.DataFrame,
    column: str,
    *,
    preserve_nulls: bool = False,
) -> np.ndarray:
    if column not in person.columns:
        if preserve_nulls:
            return np.full(len(person), np.nan, dtype=np.float64)
        return np.zeros(len(person), dtype=np.float64)
    numeric = pd.to_numeric(
        person[column],
        errors="raise" if preserve_nulls else "coerce",
    )
    if not preserve_nulls:
        numeric = numeric.fillna(0.0)
    return numeric.to_numpy(dtype=np.float64)


def _require_complete_recipient_predictor_sources(
    frame: Frame,
    recipient_mask: np.ndarray,
    predictors: Sequence[str],
) -> None:
    """Reject raw recipient absence before feature conversion or coercion."""

    tax_unit = frame.table("tax_unit")
    person = frame.table("person")
    recipient_ids = tax_unit.loc[recipient_mask, "tax_unit_id"]
    offenders: dict[str, int] = {}
    for predictor in predictors:
        source = _predictor_source_column(predictor)
        if source == "filing_status_code":
            values = tax_unit.get("filing_status_input")
            if values is None:
                values = tax_unit.get("filing_status")
            missing = (
                np.ones(int(recipient_mask.sum()), dtype=bool)
                if values is None
                else values.loc[recipient_mask].isna().to_numpy()
            )
        elif source == "tax_unit_person_count":
            missing = np.zeros(int(recipient_mask.sum()), dtype=bool)
        elif source in tax_unit.columns:
            missing = tax_unit.loc[recipient_mask, source].isna().to_numpy()
        else:
            if source == "dividend_income" and source not in person.columns:
                source_columns = (
                    "non_qualified_dividend_income",
                    "qualified_dividend_income",
                )
            elif source not in person.columns and source in _PREDICTOR_LEAF_ALIASES:
                source_columns = _PREDICTOR_LEAF_ALIASES[source]
            else:
                source_columns = (source,)
            absent_columns = [
                column for column in source_columns if column not in person.columns
            ]
            if absent_columns:
                missing = np.ones(int(recipient_mask.sum()), dtype=bool)
            else:
                link = person["person_tax_unit_id"]
                relevant = link.isin(recipient_ids)
                observed_ids = set(link.loc[relevant].tolist())
                null_source = (
                    person.loc[relevant, list(source_columns)].isna().any(axis=1)
                )
                null_ids = set(link.loc[relevant].loc[null_source].tolist())
                missing = np.asarray(
                    [
                        tax_unit_id not in observed_ids or tax_unit_id in null_ids
                        for tax_unit_id in recipient_ids
                    ],
                    dtype=bool,
                )
        count = int(missing.sum())
        if count:
            offenders[str(predictor)] = count
    if offenders:
        raise ValueError(
            "PUF recipient predictor source(s) have missing values before "
            f"coercion: {offenders} (of {int(len(recipient_ids))} recipient rows). "
            "This is a terminal stacked-spine failure; gap-fill the source "
            "before the PUF pass."
        )


def _require_complete_recipient_predictors(
    features: pd.DataFrame,
    recipient_mask: np.ndarray,
    predictors: Sequence[str],
) -> None:
    """Fail closed when any recipient row is missing a predictor value.

    The populace#578 audit found the primary QRF silently zero-filling
    target-like predictors on recipient rows whose source never measured
    them, collapsing those draws to degenerate near-zero values.  Under the
    stacked-spine doctrine that absence is a named terminal failure: the
    PUF pass runs only after gap-fill has made every predictor observable on
    every origin.
    """

    recipient = features.loc[recipient_mask, list(predictors)]
    null_counts = recipient.isna().sum()
    offenders = {
        str(name): int(count) for name, count in null_counts.items() if int(count)
    }
    if offenders:
        raise ValueError(
            "PUF recipient predictor(s) have missing values on recipient "
            f"rows: {offenders} (of {int(len(recipient))} recipient rows). "
            "The primary QRF must not zero-fill absence (populace#578); "
            "gap-fill the stacked spine before the PUF pass."
        )


def _ensure_float_output_column(
    table: pd.DataFrame,
    column: str,
    *,
    preserve_nulls: bool = False,
) -> None:
    """Coerce one requested output column to float64 under the active policy.

    The legacy policy globally converts missing cells to ``0.0`` (the
    historical two-arm behavior).  Under the preserve-nulls doctrine a missing
    column materializes as null and existing nulls survive the coercion:
    absence must stay null until the stage that owns those cells fills them
    (populace#578 audit item 1).  Preserve-nulls coercion is also
    parse-strict — a non-numeric observed value fails closed instead of
    silently becoming absence.
    """

    if column not in table.columns:
        table[column] = np.nan if preserve_nulls else 0.0
        return
    if preserve_nulls:
        table[column] = pd.to_numeric(table[column], errors="raise").astype("float64")
        return
    table[column] = (
        pd.to_numeric(table[column], errors="coerce").fillna(0.0).astype("float64")
    )


def _snap_to_observed_values(
    values: Sequence[Any],
    observed: Sequence[Any],
) -> np.ndarray:
    """Map predictions to nearest donor values without a pairwise matrix.

    ``observed_array`` is sorted by :func:`numpy.unique`, so each finite
    prediction needs only its insertion point and the donor values immediately
    beside it.  Equal-distance ties select the lower value, matching the first
    index returned by ``argmin`` in the former recipient-by-unique-values
    matrix implementation.  The explicit infinity branches preserve that
    implementation's edge-case behavior as well.
    """

    value_array = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0)
    observed_values = pd.to_numeric(pd.Series(observed), errors="coerce").fillna(0.0)
    observed_array = np.unique(
        np.rint(observed_values.to_numpy(dtype=np.float64)).clip(min=0.0)
    )
    if len(observed_array) == 0:
        return value_array.to_numpy(dtype=np.float64)

    numeric_values = value_array.to_numpy(dtype=np.float64)
    snapped = np.empty_like(numeric_values)
    finite_mask = np.isfinite(numeric_values)
    finite_values = numeric_values[finite_mask]
    insertion_points = np.searchsorted(observed_array, finite_values, side="left")
    right_positions = np.minimum(insertion_points, len(observed_array) - 1)
    left_positions = np.maximum(insertion_points - 1, 0)
    left_values = observed_array[left_positions]
    right_values = observed_array[right_positions]
    choose_right = insertion_points == 0
    interior = (insertion_points > 0) & (insertion_points < len(observed_array))
    choose_right[interior] = np.abs(
        right_values[interior] - finite_values[interior]
    ) < np.abs(finite_values[interior] - left_values[interior])
    snapped[finite_mask] = np.where(choose_right, right_values, left_values)

    negative_infinity = np.isneginf(numeric_values)
    snapped[negative_infinity] = observed_array[0]
    positive_infinity = np.isposinf(numeric_values)
    snapped[positive_infinity] = (
        observed_array[-1] if np.isposinf(observed_array[-1]) else observed_array[0]
    )
    return snapped


def _write_person_tax_unit_boolean_counts(
    person: pd.DataFrame,
    *,
    mask: pd.Series,
    column: str,
    totals: pd.Series,
    fallback_basis_columns: tuple[str, ...] = (),
) -> None:
    """Place a predicted number of true people within each tax unit.

    PUF QBI flags are person inputs, but the shared support model trains at
    tax-unit grain. Their donor targets are therefore integer counts. Treating
    a count as an amount would put (say) ``2`` on one person and ``0`` on the
    spouse; a later boolean cast would silently turn two true people into one.
    This placement preserves the snapped count and ranks people by the source
    columns the flag qualifies, with stable first-person fallback for ties.
    """

    row_ids = person.loc[mask, "person_tax_unit_id"]
    if row_ids.empty:
        return
    score = np.zeros(len(row_ids), dtype=np.float64)
    for basis_column in fallback_basis_columns:
        if basis_column not in person.columns:
            continue
        score += (
            pd.to_numeric(person.loc[mask, basis_column], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=np.float64)
        )

    placement = pd.DataFrame(
        {
            "tax_unit_id": row_ids.to_numpy(),
            "score": score,
            "source_position": np.arange(len(row_ids), dtype=np.int64),
        },
        index=row_ids.index,
    ).sort_values(
        ["tax_unit_id", "score", "source_position"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    placement["rank"] = placement.groupby("tax_unit_id", sort=False).cumcount()
    placement["unit_size"] = placement.groupby("tax_unit_id", sort=False)[
        "tax_unit_id"
    ].transform("size")
    desired = (
        placement["tax_unit_id"]
        .map(totals)
        .fillna(0.0)
        .round()
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    desired = np.minimum(
        desired,
        placement["unit_size"].to_numpy(dtype=np.float64),
    )
    placement["selected"] = placement["rank"].to_numpy() < desired
    selected = placement["selected"].reindex(row_ids.index).fillna(False)
    person.loc[mask, column] = selected.to_numpy(dtype=np.float64)


def _write_person_tax_unit_totals(
    person: pd.DataFrame,
    *,
    mask: pd.Series,
    column: str,
    totals: pd.Series,
    nonnegative: bool,
    fallback_basis_columns: tuple[str, ...] = (),
) -> None:
    row_ids = person.loc[mask, "person_tax_unit_id"]
    current = pd.to_numeric(person.loc[mask, column], errors="coerce").fillna(0.0)
    basis = current.clip(lower=0.0) if nonnegative else current
    basis_sum = basis.groupby(row_ids, sort=False).transform("sum")
    target = row_ids.map(totals).fillna(0.0).to_numpy(dtype=np.float64)
    first = row_ids.groupby(row_ids, sort=False).cumcount() == 0

    allocation = np.zeros(len(row_ids), dtype=np.float64)
    has_basis = basis_sum.to_numpy(dtype=np.float64) != 0.0
    allocation[has_basis] = (
        target[has_basis]
        * basis.to_numpy(dtype=np.float64)[has_basis]
        / basis_sum.to_numpy(dtype=np.float64)[has_basis]
    )
    unallocated = ~has_basis
    if fallback_basis_columns:
        fallback = np.zeros(len(row_ids), dtype=np.float64)
        for basis_column in fallback_basis_columns:
            if basis_column not in person.columns:
                continue
            fallback += (
                pd.to_numeric(person.loc[mask, basis_column], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
                .to_numpy(dtype=np.float64)
            )
        fallback_sum = (
            pd.Series(fallback, index=row_ids.index)
            .groupby(row_ids, sort=False)
            .transform("sum")
            .to_numpy(dtype=np.float64)
        )
        use_fallback = unallocated & (fallback_sum > 0.0)
        allocation[use_fallback] = (
            target[use_fallback] * fallback[use_fallback] / fallback_sum[use_fallback]
        )
        unallocated &= ~use_fallback
    allocation[unallocated & first.to_numpy()] = target[unallocated & first.to_numpy()]
    person.loc[mask, column] = allocation


def _weighted_positive_rate(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    numeric_weights = (
        pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    total_weight = float(numeric_weights.sum())
    if total_weight <= 0.0:
        return 0.0
    return float((numeric_weights * (numeric_values > 0.0)).sum() / total_weight)


def _sparsify_tax_unit_output_to_donor_positive_rate(
    tables: Mapping[str, pd.DataFrame],
    *,
    column: str,
    donor_positive_rate: float,
    household_weights: np.ndarray,
    tax_unit_clone_index: str | None = None,
    tax_unit_channel: str | None = None,
    puf_role_only: bool = False,
) -> None:
    """Prune a sparse tax-unit amount to the donor's weighted positive rate.

    ``puf_role_only`` scopes the pruning to the PUF clone arm: under the
    preserve-nulls doctrine other arms' cells are source- or gap-fill-owned
    (observed values or authorized nulls) and must never be rewritten here.
    """

    if (tax_unit_clone_index is None) == (tax_unit_channel is None):
        raise ValueError("Provide exactly one tax-unit support role column.")
    tax_unit_role_column = (
        tax_unit_clone_index if tax_unit_clone_index is not None else tax_unit_channel
    )
    assert tax_unit_role_column is not None
    puf_role: int | str = (
        PUF_TAX_DETAIL_CLONE_INDEX
        if tax_unit_clone_index is not None
        else PUF_TAX_DETAIL_SUPPORT_CHANNEL
    )
    positive_rate = float(np.clip(donor_positive_rate, 0.0, 1.0))
    person = tables["person"]
    household = tables["household"]
    tax_unit = tables["tax_unit"]
    if len(household_weights) != len(household):
        raise ValueError(
            "household_weights must align with household rows, got "
            f"{len(household_weights)} weights for {len(household)} households."
        )
    household_weight = pd.Series(
        np.asarray(household_weights, dtype=np.float64),
        index=household["household_id"],
    )
    tax_unit_household_id = (
        person.groupby("person_tax_unit_id", sort=False)["person_household_id"]
        .first()
        .astype("int64")
    )
    tax_unit_weight = tax_unit_household_id.map(household_weight).fillna(0.0)

    role_values = tax_unit[tax_unit_role_column].dropna().unique()
    if puf_role_only:
        role_values = np.asarray(
            [role_value for role_value in role_values if role_value == puf_role]
        )
    for clone_index in role_values:
        clone_mask = tax_unit[tax_unit_role_column] == clone_index
        channel_rows = tax_unit.loc[clone_mask]
        amounts = pd.Series(
            pd.to_numeric(channel_rows[column], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64),
            index=channel_rows["tax_unit_id"].to_numpy(),
        )
        weights = tax_unit_weight.reindex(amounts.index).fillna(0.0)
        positive = amounts > 0.0
        positive_weight = float(weights[positive].sum())
        desired_positive_weight = positive_rate * float(weights.sum())
        if positive_weight <= desired_positive_weight or positive_weight <= 0.0:
            continue

        ranked = (
            pd.DataFrame({"amount": amounts[positive], "weight": weights[positive]})
            .sort_values("amount", ascending=False)
            .copy()
        )
        cumulative = ranked["weight"].cumsum()
        keep = cumulative <= desired_positive_weight
        if not keep.any() and len(keep) > 0:
            keep.iloc[0] = True
        kept_ids = set(ranked.index[keep])

        sparse_amounts = amounts.copy()
        sparse_amounts.loc[positive & ~amounts.index.isin(kept_ids)] = 0.0
        original_total = float((amounts * weights).sum())
        sparse_total = float((sparse_amounts * weights).sum())
        if original_total != 0.0 and sparse_total != 0.0:
            sparse_amounts *= original_total / sparse_total
        tax_unit.loc[clone_mask, column] = sparse_amounts.to_numpy()


def _sparsify_tax_unit_person_output_to_donor_positive_rate(
    tables: Mapping[str, pd.DataFrame],
    *,
    column: str,
    donor_positive_rate: float,
    household_weights: np.ndarray,
    person_clone_index: str | None = None,
    tax_unit_clone_index: str | None = None,
    person_channel: str | None = None,
    tax_unit_channel: str | None = None,
    puf_role_only: bool = False,
) -> None:
    if (person_clone_index is None) == (person_channel is None) or (
        (tax_unit_clone_index is None) == (tax_unit_channel is None)
    ):
        raise ValueError("Provide exactly one person and tax-unit support role column.")
    person_role_column = (
        person_clone_index if person_clone_index is not None else person_channel
    )
    tax_unit_role_column = (
        tax_unit_clone_index if tax_unit_clone_index is not None else tax_unit_channel
    )
    assert person_role_column is not None
    assert tax_unit_role_column is not None
    puf_role: int | str = (
        PUF_TAX_DETAIL_CLONE_INDEX
        if tax_unit_clone_index is not None
        else PUF_TAX_DETAIL_SUPPORT_CHANNEL
    )
    positive_rate = float(np.clip(donor_positive_rate, 0.0, 1.0))
    person = tables["person"]
    household = tables["household"]
    tax_unit = tables["tax_unit"]
    if len(household_weights) != len(household):
        raise ValueError(
            "household_weights must align with household rows, got "
            f"{len(household_weights)} weights for {len(household)} households."
        )
    household_weight = pd.Series(
        np.asarray(household_weights, dtype=np.float64),
        index=household["household_id"],
    )
    tax_unit_household_id = (
        person.groupby("person_tax_unit_id", sort=False)["person_household_id"]
        .first()
        .astype("int64")
    )
    tax_unit_weight = tax_unit_household_id.map(household_weight).fillna(0.0)
    tax_unit_amount = (
        pd.to_numeric(person[column], errors="coerce")
        .fillna(0.0)
        .groupby(person["person_tax_unit_id"], sort=False)
        .sum()
    )

    clone_indices = tax_unit[tax_unit_role_column].dropna().unique()
    if puf_role_only or column in _PUF_TAX_DETAIL_PRESERVE_BASE_ASEC_OUTPUTS:
        clone_indices = np.asarray(
            [clone_index for clone_index in clone_indices if clone_index == puf_role]
        )
    for clone_index in clone_indices:
        channel_tax_unit_ids = tax_unit.loc[
            tax_unit[tax_unit_role_column] == clone_index,
            "tax_unit_id",
        ]
        amounts = tax_unit_amount.reindex(channel_tax_unit_ids).fillna(0.0)
        weights = tax_unit_weight.reindex(channel_tax_unit_ids).fillna(0.0)
        positive = amounts > 0.0
        positive_weight = float(weights[positive].sum())
        desired_positive_weight = positive_rate * float(weights.sum())
        if positive_weight <= desired_positive_weight or positive_weight <= 0.0:
            continue

        ranked = (
            pd.DataFrame({"amount": amounts[positive], "weight": weights[positive]})
            .sort_values("amount", ascending=False)
            .copy()
        )
        cumulative = ranked["weight"].cumsum()
        keep = cumulative <= desired_positive_weight
        if not keep.any() and len(keep) > 0:
            keep.iloc[0] = True
        kept_ids = set(ranked.index[keep])

        sparse_totals = amounts.copy()
        sparse_totals.loc[positive & ~amounts.index.isin(kept_ids)] = 0.0
        original_total = float((amounts * weights).sum())
        sparse_total = float((sparse_totals * weights).sum())
        if original_total != 0.0 and sparse_total != 0.0:
            sparse_totals *= original_total / sparse_total

        mask = person[person_role_column] == clone_index
        _write_person_tax_unit_totals(
            person,
            mask=mask,
            column=column,
            totals=sparse_totals,
            nonnegative=column in _PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS,
        )


def _weighted_signed_leg_masses(
    values: pd.Series, weights: pd.Series
) -> tuple[float, float, float]:
    """Return donor-scale (positive, negative, total) weighted leg masses.

    The two legs are reported separately so a signed calibration can pin each
    one; the total weight is returned so callers can work in donor-invariant
    per-unit-weight leg masses rather than population-scaled totals.
    """

    numeric_values = (
        pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    )
    numeric_weights = (
        pd.to_numeric(weights, errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )
    positive_mass = float((np.maximum(numeric_values, 0.0) * numeric_weights).sum())
    negative_mass = float((np.minimum(numeric_values, 0.0) * numeric_weights).sum())
    return positive_mass, negative_mass, float(numeric_weights.sum())


def _calibrate_tax_unit_person_output_signed_mass_to_donor(
    tables: Mapping[str, pd.DataFrame],
    *,
    column: str,
    donor_values: pd.Series,
    donor_weights: pd.Series,
    household_weights: np.ndarray,
    person_clone_index: str,
    tax_unit_clone_index: str,
) -> None:
    """Pin a signed person output's per-leg mass to the donor instrument.

    Scales the imputed positive and negative legs on the PUF support channel by
    separate scalars so each leg's per-unit-weight weighted mass equals the
    donor's. The gate's sign assignment (which rows are positive, negative, or
    zero) and each leg's relative shape are untouched; only the two leg totals
    move, restoring the net signed mass that the regime-gated QRF regresses
    toward balance on a sparse, sign-mixed, heavy-tailed column. This is the
    signed generalization of
    :func:`_sparsify_tax_unit_person_output_to_donor_positive_rate`.

    Applied to the PUF support channel only: the ASEC channel carries measured
    source observations (Schedule F operations income is measured on ASEC as
    ``FRSE_VAL``) that must never be rescaled.
    """

    person = tables["person"]
    household = tables["household"]
    tax_unit = tables["tax_unit"]
    if len(household_weights) != len(household):
        raise ValueError(
            "household_weights must align with household rows, got "
            f"{len(household_weights)} weights for {len(household)} households."
        )

    donor_positive, donor_negative, donor_weight_total = _weighted_signed_leg_masses(
        donor_values, donor_weights
    )
    if donor_weight_total <= 0.0:
        return
    donor_positive_per_weight = donor_positive / donor_weight_total
    donor_negative_per_weight = donor_negative / donor_weight_total

    household_weight = pd.Series(
        np.asarray(household_weights, dtype=np.float64),
        index=household["household_id"],
    )
    tax_unit_household_id = (
        person.groupby("person_tax_unit_id", sort=False)["person_household_id"]
        .first()
        .astype("int64")
    )
    tax_unit_weight = tax_unit_household_id.map(household_weight).fillna(0.0)
    tax_unit_amount = (
        pd.to_numeric(person[column], errors="coerce")
        .fillna(0.0)
        .groupby(person["person_tax_unit_id"], sort=False)
        .sum()
    )

    channel_tax_unit_ids = tax_unit.loc[
        tax_unit[tax_unit_clone_index] == PUF_TAX_DETAIL_CLONE_INDEX,
        "tax_unit_id",
    ]
    amounts = tax_unit_amount.reindex(channel_tax_unit_ids).fillna(0.0)
    weights = tax_unit_weight.reindex(channel_tax_unit_ids).fillna(0.0)
    channel_weight_total = float(weights.sum())
    if channel_weight_total <= 0.0:
        return

    values_array = amounts.to_numpy(dtype=np.float64)
    weights_array = weights.to_numpy(dtype=np.float64)
    positive = values_array > 0.0
    negative = values_array < 0.0
    imputed_positive_per_weight = (
        float((values_array[positive] * weights_array[positive]).sum())
        / channel_weight_total
    )
    imputed_negative_per_weight = (
        float((values_array[negative] * weights_array[negative]).sum())
        / channel_weight_total
    )

    calibrated = values_array.copy()
    # A leg with imputed mass can be rescaled to the donor's; a leg the gate
    # produced but the donor lacks is scaled to zero (a one-sided donor pins the
    # sign). A donor leg the gate produced no rows for cannot be injected by
    # scaling, so it is left untouched rather than fabricated.
    if imputed_positive_per_weight > 0.0:
        calibrated[positive] *= donor_positive_per_weight / imputed_positive_per_weight
    if imputed_negative_per_weight < 0.0:
        calibrated[negative] *= donor_negative_per_weight / imputed_negative_per_weight

    calibrated_totals = pd.Series(calibrated, index=amounts.index)
    mask = person[person_clone_index] == PUF_TAX_DETAIL_CLONE_INDEX
    _write_person_tax_unit_totals(
        person,
        mask=mask,
        column=column,
        totals=calibrated_totals,
        nonnegative=False,
    )


def _reconcile_puf_social_security_components(
    predictions: pd.DataFrame,
    person: pd.DataFrame,
    *,
    person_clone_index: str,
    tax_unit_ids: np.ndarray,
    requested_components: Sequence[str],
) -> None:
    components = tuple(
        component
        for component in PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS
        if component in requested_components
    )
    if not components:
        return
    if set(components) != set(PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS):
        raise ValueError(
            "PUF Social Security support must request all Social Security "
            f"component leaves, got {components!r}."
        )
    missing = [component for component in components if component not in predictions]
    if missing:
        raise ValueError(
            "PUF Social Security predictions are missing component column(s): "
            f"{missing}."
        )

    puf_person = person.loc[
        person[person_clone_index] == PUF_TAX_DETAIL_CLONE_INDEX,
        ["person_tax_unit_id"],
    ].copy()
    for component in components:
        if component in person.columns:
            puf_person[component] = (
                pd.to_numeric(
                    person.loc[
                        person[person_clone_index] == PUF_TAX_DETAIL_CLONE_INDEX,
                        component,
                    ],
                    errors="coerce",
                )
                .fillna(0.0)
                .clip(lower=0.0)
                .to_numpy(dtype=np.float64)
            )
        else:
            puf_person[component] = np.zeros(
                len(puf_person),
                dtype=np.float64,
            )
    basis = (
        puf_person.groupby("person_tax_unit_id", sort=False)[list(components)]
        .sum()
        .reindex(tax_unit_ids)
        .fillna(0.0)
    )
    basis_values = basis.to_numpy(dtype=np.float64)
    basis_sums = basis_values.sum(axis=1)
    shares = np.zeros_like(basis_values)
    has_basis = basis_sums > 0
    shares[has_basis] = basis_values[has_basis] / basis_sums[has_basis, np.newaxis]
    shares[~has_basis] = 1.0 / len(components)

    total = (
        predictions.loc[:, list(components)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .sum(axis=1)
        .to_numpy(dtype=np.float64)
    )
    for index, component in enumerate(components):
        predictions[component] = shares[:, index] * total


def _person_source_values(
    arrays: Mapping[str, Sequence[Any]],
    output: str,
) -> np.ndarray | None:
    if output in arrays:
        return _numeric_array(arrays[output])
    source_aliases = {
        "employment_income_before_lsr": ("employment_income",),
        "self_employment_income_before_lsr": ("self_employment_income",),
        "sstb_self_employment_income_before_lsr": ("sstb_self_employment_income",),
        "long_term_capital_gains_before_response": ("long_term_capital_gains",),
        "taxable_private_pension_income": ("taxable_pension_income",),
        # The PUF observes realized IRA deductions; at baseline the engine's
        # desired-contribution leaf equals the realized amount.
        "traditional_ira_contributions_desired": ("traditional_ira_contributions",),
    }
    for source in source_aliases.get(output, ()):
        if source in arrays:
            return _numeric_array(arrays[source])
    if output == "partnership_income" and "partnership_s_corp_income" in arrays:
        return _numeric_array(arrays["partnership_s_corp_income"])
    if output == "s_corp_income" and "partnership_s_corp_income" in arrays:
        return np.zeros_like(_numeric_array(arrays["partnership_s_corp_income"]))
    if output == "partnership_self_employment_net_earnings" and (
        "partnership_se_income" in arrays
    ):
        return _numeric_array(arrays["partnership_se_income"])
    if output == "partnership_income" and {"E25980", "E25960"}.issubset(arrays):
        return _numeric_array(arrays["E25980"]) - _numeric_array(arrays["E25960"])
    if output == "s_corp_income" and {"E26190", "E26180"}.issubset(arrays):
        return _numeric_array(arrays["E26190"]) - _numeric_array(arrays["E26180"])
    if output == "partnership_self_employment_net_earnings" and {
        "E25960",
        "E26180",
    }.issubset(arrays):
        return _numeric_array(arrays["E25960"]) + _numeric_array(arrays["E26180"])
    if output in _PUF_MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS and "E17500" in arrays:
        return (
            _numeric_array(arrays["E17500"])
            * _PUF_MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS[output]
        )
    if output == "unemployment_compensation" and (
        "taxable_unemployment_compensation" in arrays
    ):
        return _numeric_array(arrays["taxable_unemployment_compensation"])
    if output == "qualified_tuition_expenses" and "E03230" in arrays:
        tuition = _numeric_array(arrays["E03230"])
        if "E87530" in arrays:
            tuition = np.maximum(tuition, _numeric_array(arrays["E87530"]))
        return np.maximum(tuition, 0.0)
    if output in PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS:
        if output != "social_security_retirement":
            for source in ("social_security", "total_social_security", "E02400"):
                if source in arrays:
                    return np.zeros_like(_numeric_array(arrays[source]))
            return None
        for source in ("social_security", "total_social_security", "E02400"):
            if source in arrays:
                return _numeric_array(arrays[source])
    return None


def _add_predictor_aliases(
    table: pd.DataFrame,
    predictors: Sequence[str],
) -> None:
    for predictor in predictors:
        if predictor in table.columns:
            continue
        source = _predictor_source_column(predictor)
        if source in table.columns:
            table[predictor] = table[source]
        elif source == "dividend_income" and {
            "qualified_dividend_income",
            "non_qualified_dividend_income",
        }.issubset(table.columns):
            table[predictor] = pd.to_numeric(
                table["qualified_dividend_income"],
                errors="coerce",
            ).fillna(0.0) + pd.to_numeric(
                table["non_qualified_dividend_income"],
                errors="coerce",
            ).fillna(0.0)
        elif source in _PREDICTOR_LEAF_ALIASES:
            pieces = [
                pd.to_numeric(table[leaf], errors="coerce").fillna(0.0)
                for leaf in _PREDICTOR_LEAF_ALIASES[source]
                if leaf in table.columns
            ]
            if pieces:
                table[predictor] = sum(pieces)


def _predictor_source_column(column: str) -> str:
    if column.startswith(_PUF_PREDICTOR_PREFIX):
        return column.removeprefix(_PUF_PREDICTOR_PREFIX)
    return column


def _tax_unit_source_values(
    arrays: Mapping[str, Sequence[Any]],
    tax_unit_id: np.ndarray,
    output: str,
    grouped_person: pd.DataFrame,
) -> np.ndarray | None:
    if output in arrays:
        values = _numeric_array(arrays[output])
        if len(values) == len(tax_unit_id):
            return values
    if (
        output == "state_withheld_income_tax"
        and "state_and_local_sales_or_income_tax" in arrays
    ):
        return _numeric_array(arrays["state_and_local_sales_or_income_tax"])
    if output == "interest_deduction":
        if "home_mortgage_interest" in grouped_person:
            return (
                grouped_person["home_mortgage_interest"]
                .reindex(tax_unit_id)
                .fillna(0.0)
                .to_numpy()
            )
        pieces = [
            _numeric_array(arrays[name])
            for name in (
                "first_home_mortgage_interest",
                "second_home_mortgage_interest",
            )
            if name in arrays
        ]
        if pieces:
            return np.sum(np.column_stack(pieces), axis=1)
    return None


def _filing_status_codes(
    values: Sequence[Any],
    *,
    preserve_nulls: bool = False,
) -> np.ndarray:
    decoded = pd.Series(values).map(_decode_status).str.upper()
    codes = decoded.map(_FILING_STATUS_CODES)
    if not preserve_nulls:
        codes = codes.fillna(0.0)
    return codes.to_numpy(dtype=np.float64)


def _decode_status(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


def _numeric_array(values: Any) -> np.ndarray:
    if values is None:
        raise ValueError("Cannot convert missing array to numeric values.")
    if np.isscalar(values):
        return np.asarray(values, dtype=np.float64)
    return (
        pd.to_numeric(pd.Series(values), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _require_array_columns(
    arrays: Mapping[str, Sequence[Any]],
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    missing = [column for column in columns if column not in arrays]
    if missing:
        raise ValueError(f"{label} missing required array(s): {missing}.")
