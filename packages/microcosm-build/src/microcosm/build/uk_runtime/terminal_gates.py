"""Batched terminal acceptance gates for UK release candidates.

The national builder evaluates this battery once, after every source stage and
immediately before writing its staging H5.  Every evaluator runs even when an
earlier evaluator fails, so one expensive build produces one complete named
failure report.  Evidence that does not exist yet is omitted: gates whose
evidence or reviewed thresholds are absent (the parity trio, the
weighted-integrity pair, the future delivered-take-up gates) are not
represented by placeholder passes.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import (
    GateResult,
    export_surface_gate,
)
from microcosm.build.gates import (
    target_surface_gate as _target_surface_gate,
)
from microcosm.build.uk_runtime.cgt_projection import UKCGTProjection
from microcosm.build.uk_runtime.diagnostics import uk_weight_summary
from microcosm.build.uk_runtime.weighted_integrity import (
    UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE,
    UK_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE,
    UKInputMassParityPolicy,
    UKInputMassReference,
    UKQRFTailConcentrationPolicy,
    UKReviewedExclusion,
    _expired_exclusion_failure,
    _premature_exclusion_failure,
    coerce_reviewed_exclusions,
    exclusion_evaluation_date,
    load_uk_reviewed_exclusion_register,
    uk_input_mass_parity_gate,
    uk_qrf_tail_concentration_gate,
)

__all__ = [
    "UK_ALLOWED_EXTRA_EXPORT_COLUMNS",
    "UK_CANDIDATE_DATASET_NAME",
    "UK_DEFAULT_ZERO_WEIGHT_STRATA",
    "UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS",
    "UK_MAX_TARGET_ABS_RELATIVE_ERROR",
    "UK_REFERENCE_DATASET_NAME",
    "UK_REVIEWED_EXPORT_EXCLUSIONS",
    "UKInputMassParityPolicy",
    "UKInputMassReference",
    "UKQRFTailConcentrationPolicy",
    "UKZeroWeightStratumDeclaration",
    "uk_default_degenerate_reviewed_exclusions",
    "uk_default_target_fit_reviewed_exclusions",
    "uk_degenerate_release_surface_gate",
    "uk_export_candidate_columns",
    "uk_export_surface_gate",
    "uk_input_mass_parity_gate",
    "uk_qrf_tail_concentration_gate",
    "uk_target_fit_gate",
    "uk_target_surface_gate",
    "uk_weight_ess_gate",
    "uk_weight_ratio_gate",
    "uk_zero_weight_strata_gate",
    "uk_cgt_projection_entrants_gate",
]

UK_CANDIDATE_DATASET_NAME = "microcosm_uk_2024_25"
# The label names the pinned reference artifact exactly: the 2024-25 line's
# published enhanced_frs_2024_25.h5 (no separate "recalibrated" variant
# exists at this vintage; the June report strings keep their own label).
# After the swap, "reference" becomes the previous certified microcosm line
# once a second certified cut exists; that later increment flips this value.
UK_REFERENCE_DATASET_NAME = "enhanced_frs_2024_25"
UK_MAX_TARGET_ABS_RELATIVE_ERROR = 0.25


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
    "benunit.frs_benunit_capital",
    "benunit.has_mixed_age_couple_pension_credit_saving",
    "benunit.liable_for_share_of_household_rent",
    "benunit.pension_credit_reported_capital",
    "benunit.uc_deduction_combination",
    "benunit.uc_deduction_random_draw",
    "benunit.uc_deduction_type_random_draw",
    "benunit.uc_latent_deduction_rate",
    "benunit.uc_reported_capital",
    "benunit.would_claim_uc_childcare",
    "household.atomic_area_basis",
    "household.atomic_area_code",
    "household.atomic_area_system",
    "household.bus_fare_spending",
    "household.bus_subsidy_spending",
    "household.cash_isa",
    # #1045: the CGT support split's family copy count.
    "household.cgt_support_copies",
    "household.cgt_support_copy_index",
    # #1063: the residential split's arm flag and index.
    "household.household_is_cgt_residential_clone",
    "household.cgt_residential_clone_index",
    "household.household_clone_index",
    # microcosm#1114: the three area codes leave every artifact under the
    # consumers' names (geography_ladder.UK_EXPORT_AREA_CODE_COLUMNS); the
    # ladder names never reach an artifact and uk_export_candidate_columns
    # translates them before this list is compared.
    "household.constituency_code_oa",
    "household.consumer_debt",
    # microcosm#932: the five nation-native aliases of the atomic assignment
    # (output_area_code, data_zone_code, intermediate_zone_code,
    # super_data_zone_code, district_electoral_area_code) never reach an
    # artifact: graph_terminal._tables drops them at the single-year export
    # boundary. They are listed because the full-build export-surface gate
    # reads the graph frame's in-memory columns (full_gates.py,
    # ``parity_evidence.candidate_columns``) before that boundary, and would
    # otherwise flag them as unreviewed extras (the F8 gap on #932).
    "household.data_zone_code",
    "household.district_electoral_area_code",
    "household.electricity_consumption",
    "household.gas_consumption",
    "household.geography_household_key",
    "household.has_fuel_consumption",
    "household.household_is_capital_gains_clone",
    "household.household_is_cgt_support_copy",
    "household.household_is_spi_income_band_donor",
    "household.household_is_spi_synthetic",
    # #1003: the WAS Lifetime ISA stage's household total and person cells.
    "household.household_lifetime_isa_balance",
    # #930: the NTS bus-travel stage's household journey cell.
    "household.household_local_bus_trips",
    "household.spi_income_band_donor_lower_bound",
    "household.intermediate_zone_code",
    "household.la_code_oa",
    # #953: the engine's household local_authority enum input, written by the
    # rowwise geography ladder from local_authority_code. The incumbent never
    # carried it, so the coverage manifest cannot list it; this allow-list
    # binds at the national release-cut export gate, and the rowwise lane's
    # own guard is the ladder gate's code/member consistency check.
    "household.local_authority",
    "household.lsoa_code",
    "household.mortgage_debt",
    "household.msoa_code",
    "household.num_vehicles",
    "household.oa_code",
    "household.ons_household_type",
    "household.output_area_code",
    "household.private_pension_wealth",
    "household.property_purchased",
    "household.rail_usage",
    "household.region_code_oa",
    "household.stocks_and_shares_isa",
    "household.super_data_zone_code",
    "person.a_and_e_visits",
    "person.aa_category",
    "person.admitted_patient_visits",
    "person.age_started_or_accepted_current_education_or_training",
    "person.attends_private_school_random_draw",
    "person.capital_gains_asset_type",
    "person.capital_gains_badr",
    "person.capital_gains_residential_property",
    "person.cgt_residential_probability",
    "person.bus_in_london_trips",
    "person.bus_pass_eligible",
    "person.care_hours",
    "person.charitable_investment_gifts",
    "person.dla_m_category",
    "person.dla_sc_category",
    "person.employment_sector",
    "person.esa_health_condition_proxy",
    "person.esa_support_group_proxy",
    "person.gift_aid",
    "person.has_lifetime_isa",
    "person.highest_education",
    "person.is_before_universal_credit_qualifying_young_person_terminal_date",
    "person.is_blind",
    "person.is_claimant_or_partner",
    "person.is_hbai_dependent_child",
    "person.is_in_non_advanced_education",
    "person.is_parent",
    "person.is_uc_claimant",
    "person.legacy_jobseeker_proxy",
    "person.lifetime_isa_balance",
    "person.local_bus_single_fare_share",
    "person.local_bus_trips",
    "person.local_bus_use_band",
    "person.ons_family_index",
    "person.ons_family_role",
    "person.other_local_bus_trips",
    "person.outpatient_visits",
    "person.pension_contributions_via_salary_sacrifice",
    "person.pip_dl_category",
    "person.pip_m_category",
    "person.property_finance_costs",
    "person.property_rental_income",
    "person.receives_benefits_in_own_right",
    "person.rent_paid_as_boarder",
    "person.rent_paid_as_lodger",
    "person.relationship_to_head",
    "person.salary_sacrifice_asked",
    "person.salary_sacrifice_reported",
    "person.sic_industry_division",
    "person.student_loan_balance",
    "person.student_loan_plan",
    "person.tax_free_childcare_spend_routed_share",
    "person.uc_is_in_startup_period",
    "person.would_claim_carers_allowance",
    "person.would_claim_marriage_allowance",
    "person.would_claim_scp",
    "person.person_is_spi_income_band_carrier",
    # microcosm#1063 c9 (ruling 2026-10-02): every declared stage output the
    # enhanced-FRS incumbent never carried is allow-listed rather than
    # dropped, so the release candidate ships the spine's own surface. The
    # certifier rehearsal R5 on the 2026-09-30 build named them: the source
    # and support-channel lineage of the three entities, the council tax
    # family (#934), the reported-benefit inputs the take-up stages read
    # (the five internal disability carriers are not among them: they leave
    # at the release boundary, UK_RELEASE_EXPORT_DROPPED_COLUMNS),
    # the SPI channel's hmrc_spi_* leaves (#717), the FRS education and
    # housing fields, the two raw FRS codes the spine fences
    # (ossben_identifiable_subset, srp_regular_code5) and
    # other_investment_income. Only incapacity_benefit_reported is dropped
    # (UK_REVIEWED_EXPORT_EXCLUSIONS).
    "benunit.benunit_source_id",
    "benunit.benunit_support_channel",
    "benunit.benunit_support_clone_index",
    "benunit.dependent_children",
    "household.council_tax_rebate",
    "household.council_tax_reported",
    "household.council_tax_single_adult_raw",
    "household.household_source_id",
    "household.household_support_channel",
    "household.household_support_clone_index",
    "household.num_bedrooms",
    "household.source_household_id",
    "household.source_household_key",
    "household.source_year",
    "household.subrent",
    "person.disabled_students_allowance_eligible_expenses",
    "person.free_school_breakfasts",
    "person.hmrc_spi_assessable_income",
    "person.hmrc_spi_employed_income",
    "person.hmrc_spi_employment_benefits",
    "person.hmrc_spi_employment_expenses",
    "person.hmrc_spi_incapacity_benefit_income",
    "person.hmrc_spi_miscellaneous_employment_income",
    "person.hmrc_spi_other_income",
    "person.hmrc_spi_other_social_security_income",
    "person.hmrc_spi_pay",
    "person.hmrc_spi_state_pension_income",
    "person.hmrc_spi_taxable_termination_pay",
    "person.hmrc_spi_total_earned_income",
    "person.hmrc_spi_total_investment_income",
    "person.hmrc_spi_unemployment_benefit_income",
    "person.is_in_approved_training",
    "person.ossben_identifiable_subset",
    "person.other_investment_income",
    "person.person_source_id",
    "person.person_support_channel",
    "person.person_support_clone_index",
    "person.srp_regular_code5",
    "benunit.benunit_clone_index",
    "person.person_clone_index",
    "household.itl1_code",
    "household.itl2_code",
    "household.itl3_code",
    "household.ward_code",
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
    "household.property_wealth": (
        "policyengine-uk derives property_wealth from main_residence_value, "
        "other_residential_property_value and non_residential_property_value; "
        "a persisted copy overrides that sum and has no uprating index, so the "
        "candidate must drop it (microcosm#1106, the uk-data#543 defect)."
    ),
}

_STRUCTURAL_COLUMNS: Mapping[str, frozenset[str]] = {
    "person": frozenset({"person_id", "person_household_id", "person_benunit_id"}),
    "benunit": frozenset({"benunit_id"}),
    "household": frozenset({"household_id", _WEIGHT_COLUMN}),
}


@functools.cache
def uk_default_degenerate_reviewed_exclusions() -> Mapping[str, UKReviewedExclusion]:
    """The committed degenerate-surface register (#630), loaded lazily."""

    return MappingProxyType(
        load_uk_reviewed_exclusion_register(
            None, resource=UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE
        )
    )


@functools.cache
def uk_default_target_fit_reviewed_exclusions() -> Mapping[str, UKReviewedExclusion]:
    """The committed target-fit deferral register (#796), loaded lazily."""

    return MappingProxyType(
        load_uk_reviewed_exclusion_register(
            None, resource=UK_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE
        )
    )


def uk_export_candidate_columns(frame: Any) -> set[str]:
    """The ``entity.column`` surface a frame exports, as the gate reads it.

    Structural id columns are not exported content (microcosm#1063 c9: the
    certifier rehearsal listed every id as an unreviewed extra), and the
    frame's household weights live beside the tables, so the surface carries
    ``household.household_weight`` explicitly, as the enhanced-FRS reference
    does.
    """

    from microcosm.build.uk_runtime.geography_ladder import (
        UK_EXPORT_AREA_CODE_COLUMNS,
    )

    columns: set[str] = set()
    for entity in frame.entities:
        structural = _STRUCTURAL_COLUMNS.get(str(entity), frozenset())
        for column in frame.table(entity).columns:
            if column in structural:
                continue
            # microcosm#1114: the frame holds the ladder names; the artifact
            # carries the consumers' names, which is what the surface compares.
            if str(entity) == "household":
                column = UK_EXPORT_AREA_CODE_COLUMNS.get(str(column), column)
            columns.add(f"{entity}.{column}")
    columns.add(f"household.{_WEIGHT_COLUMN}")
    return columns


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
    reviewed_exclusions: Mapping[str, UKReviewedExclusion] | None = None,
    now: date | None = None,
    dropped_at_export: Mapping[str, Iterable[str]] | None = None,
) -> GateResult:
    """Reject every all-null, all-zero, or constant nonstructural column.

    ``reviewed_exclusions`` are schema-2 approval records (#610); an entry
    suppresses from its ``approved_on`` through its ``expires_on``, and any
    out-of-force entry fails the gate with correct-or-renew context — even
    when its column is absent or carries signal, so the register cannot rot
    silently at any column state (matching the input-mass and QRF wrappers).
    """

    evaluated_on = exclusion_evaluation_date(now)
    exclusions = coerce_reviewed_exclusions(
        reviewed_exclusions, label="UK degenerate-surface"
    )
    # Columns the release boundary drops before writing: a gate fed the
    # pre-export frame skips them, exactly as the certifier never sees them
    # on the exported H5; the skipped names are recorded in the details.
    dropped = {
        entity: frozenset(str(column) for column in columns)
        for entity, columns in (dropped_at_export or {}).items()
    }
    skipped_at_export: list[str] = []
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
            if column in dropped.get(entity, frozenset()):
                skipped_at_export.append(f"{entity}.{column}")
                continue
            checked += 1
            name = f"{entity}.{column}"
            present.add(name)
            finding = _degenerate_kind(table[column])
            if finding is None:
                continue
            kind, value = finding
            detail = {"kind": kind, "value": value}
            record = exclusions.get(name)
            if (
                record is not None
                and not record.expired(evaluated_on)
                and not record.premature(evaluated_on)
            ):
                excluded[name] = {
                    **detail,
                    "reason": record.reason,
                    "approved_by": record.approved_by,
                    "adjudication": record.adjudication,
                    "expires_on": record.expires_on,
                }
                continue
            live[name] = detail
            degenerate_message = (
                f"{name}: persisted release column is {kind.replace('_', '-')}"
                + (f" at {value!r}" if kind == "constant" else "")
            )
            if record is not None and record.expired(evaluated_on):
                failures.append(
                    f"{degenerate_message}; its reviewed exclusion expired "
                    f"{record.expires_on} (approved_by {record.approved_by}, "
                    f"{record.adjudication}) — renew the adjudication or "
                    "remove the entry."
                )
            elif record is not None:
                failures.append(
                    f"{degenerate_message}; its reviewed exclusion takes force "
                    f"{record.approved_on} (approved_by {record.approved_by}, "
                    f"{record.adjudication}) — correct the receipt's "
                    "approved_on or wait for it."
                )
            else:
                failures.append(
                    f"{degenerate_message}; populate it with signal, drop it, "
                    "or record a reviewed exclusion."
                )

    expired = sorted(
        name for name, record in exclusions.items() if record.expired(evaluated_on)
    )
    premature = sorted(
        name for name, record in exclusions.items() if record.premature(evaluated_on)
    )
    # Stale probing covers in-force entries only: an out-of-force entry gets
    # receipt-context failures below, never "now carry signal; remove them."
    stale = sorted(
        name
        for name in exclusions
        if name in present
        and name not in excluded
        and name not in live
        and name not in expired
        and name not in premature
    )
    dormant = sorted(set(exclusions) - present)
    if stale:
        failures.append(
            "Stale reviewed degenerate-column exclusions now carry signal; remove "
            f"them: {stale}."
        )
    # Out-of-force entries whose column did not fail above (absent, or
    # present without a degenerate finding) must still fail the gate: the
    # register cannot rot just because its column moved. Live columns
    # already carry per-column receipt context, so only the remainder gets
    # the combined message (one failure per condition, never two per entry).
    unreported_expired = [name for name in expired if name not in live]
    if unreported_expired:
        failures.append(
            _expired_exclusion_failure(
                exclusions, unreported_expired, family="degenerate-column"
            )
        )
    unreported_premature = [name for name in premature if name not in live]
    if unreported_premature:
        failures.append(
            _premature_exclusion_failure(
                exclusions, unreported_premature, family="degenerate-column"
            )
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
            "dropped_at_export": sorted(skipped_at_export),
            "findings": dict(sorted(live.items())),
            "all_null_columns": by_kind["all_null"],
            "all_zero_columns": by_kind["all_zero"],
            "constant_columns": by_kind["constant"],
            "reviewed_exclusions": dict(sorted(excluded.items())),
            "stale_exclusions": stale,
            "dormant_exclusions": dormant,
            "expired_exclusions": expired,
            "premature_exclusions": premature,
            "exclusions_evaluated_on": evaluated_on.isoformat(),
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


def _family_evaluation(
    weights: Sequence[float] | np.ndarray,
    family_weights: Sequence[float] | np.ndarray | None,
    family_fold: Mapping[str, object] | None,
) -> tuple[dict[str, int | float | None], dict[str, object]]:
    """The summary a weight gate evaluates and the details it reports.

    The row-level summary is always reported at the top level (the release
    diagnostics restate it). When family-folded weights are supplied, the
    gate evaluates on their summary instead: each support-split root with its
    copies summed is the quantity the June certification measured on a frame
    without copies, so the fence reads weight dispersion, not the number of
    copies (microcosm#1045 review).
    """

    summary = uk_weight_summary(weights)
    if family_weights is None:
        return summary, {**summary, "evaluated_on": "row_weights"}
    family_summary = uk_weight_summary(family_weights)
    return family_summary, {
        **summary,
        "evaluated_on": "family_folded_weights",
        "family_folded": {**family_summary, **dict(family_fold or {})},
    }


def uk_weight_ess_gate(
    weights: Sequence[float] | np.ndarray,
    *,
    minimum_ess_fraction: float,
    family_weights: Sequence[float] | np.ndarray | None = None,
    family_fold: Mapping[str, object] | None = None,
) -> GateResult:
    """Require the shipped household weights to retain effective support.

    With ``family_weights`` the fraction is evaluated on the support-family
    fold (see :func:`_family_evaluation`); the row-level summary is reported
    beside it.
    """

    minimum = float(minimum_ess_fraction)
    if not math.isfinite(minimum) or not 0.0 < minimum <= 1.0:
        raise ValueError("minimum_ess_fraction must be finite and in (0, 1].")
    evaluated, details = _family_evaluation(weights, family_weights, family_fold)
    fraction = float(evaluated["ess_fraction"])
    if fraction < minimum:
        basis = " (family-folded)" if family_weights is not None else ""
        failures = (
            f"ESS fraction{basis} {fraction:.6g} is below the reviewed minimum "
            f"{minimum:.6g}.",
        )
    else:
        failures = ()
    return GateResult(
        name="weight_ess",
        passed=not failures,
        failures=failures,
        details={**details, "minimum_ess_fraction": minimum},
    )


def uk_weight_ratio_gate(
    weights: Sequence[float] | np.ndarray,
    *,
    maximum_max_to_median_ratio: float,
    family_weights: Sequence[float] | np.ndarray | None = None,
    family_fold: Mapping[str, object] | None = None,
) -> GateResult:
    """Backstop a shipped-weight max/positive-median concentration blowout.

    With ``family_weights`` the ratio is evaluated on the support-family fold
    (see :func:`_family_evaluation`); the row-level summary is reported
    beside it.
    """

    maximum = float(maximum_max_to_median_ratio)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(
            "maximum_max_to_median_ratio must be finite and strictly positive."
        )
    evaluated, details = _family_evaluation(weights, family_weights, family_fold)
    raw_ratio = evaluated["max_to_median_positive_weight"]
    failures: tuple[str, ...]
    if raw_ratio is None:
        failures = (
            "Max/positive-median weight ratio is undefined because the release "
            "has no positive median weight.",
        )
    else:
        ratio = float(raw_ratio)
        basis = " (family-folded)" if family_weights is not None else ""
        failures = (
            (
                f"Max/positive-median weight ratio{basis} {ratio!r} exceeds the "
                f"reviewed maximum {maximum!r}.",
            )
            if ratio > maximum
            else ()
        )
    return GateResult(
        name="weight_ratio",
        passed=not failures,
        failures=failures,
        details={**details, "maximum_max_to_median_ratio": maximum},
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
    reviewed_exclusions: Mapping[str, UKReviewedExclusion] | None = None,
    now: date | None = None,
) -> GateResult:
    """Fail a UK artifact with severe shipped-weight target errors.

    ``reviewed_exclusions`` are schema-2 approval records (#610): each defers
    one named target's breach of the release bound through its approval
    window. The exclusion suppresses the release fence only — the target
    stays bound in calibration, so the solver keeps pulling toward it. An
    out-of-force entry fails the gate with correct-or-renew context, and an
    in-force entry whose target is back inside the bound is stale and fails
    (matching :func:`microcosm.build.gates.target_fit_gate`'s rot rule), so
    the register cannot outlive the defects it defers.
    """

    maximum = float(max_abs_relative_error)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("max_abs_relative_error must be finite and non-negative.")
    evaluated_on = exclusion_evaluation_date(now)
    exclusions = coerce_reviewed_exclusions(reviewed_exclusions, label="UK target-fit")
    errors = {str(name): float(error) for name, error in target_relative_errors.items()}
    nonfinite = sorted(
        name for name, error in errors.items() if not math.isfinite(error)
    )
    if nonfinite:
        raise ValueError(f"UK target relative errors must be finite: {nonfinite}.")
    breaching = {name: error for name, error in errors.items() if abs(error) > maximum}
    failing: dict[str, float] = {}
    excluded: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for name in sorted(breaching, key=lambda name: abs(breaching[name]), reverse=True):
        error = breaching[name]
        record = exclusions.get(name)
        breach_message = (
            f"{UK_CANDIDATE_DATASET_NAME}: {name} relative error "
            f"{error:+.1%} exceeds {maximum:.0%}."
        )
        if (
            record is not None
            and not record.expired(evaluated_on)
            and not record.premature(evaluated_on)
        ):
            excluded[name] = {
                "relative_error": error,
                "reason": record.reason,
                "approved_by": record.approved_by,
                "adjudication": record.adjudication,
                "expires_on": record.expires_on,
            }
            continue
        failing[name] = error
        if record is not None and record.expired(evaluated_on):
            failures.append(
                f"{breach_message[:-1]}; its reviewed exclusion expired "
                f"{record.expires_on} (approved_by {record.approved_by}, "
                f"{record.adjudication}) — renew the adjudication or remove "
                "the entry."
            )
        elif record is not None:
            failures.append(
                f"{breach_message[:-1]}; its reviewed exclusion takes force "
                f"{record.approved_on} (approved_by {record.approved_by}, "
                f"{record.adjudication}) — correct the receipt's approved_on "
                "or wait for it."
            )
        elif len(failing) <= 20:
            failures.append(breach_message)

    expired = sorted(
        name for name, record in exclusions.items() if record.expired(evaluated_on)
    )
    premature = sorted(
        name for name, record in exclusions.items() if record.premature(evaluated_on)
    )
    # Stale probing covers in-force entries only: an out-of-force entry gets
    # receipt-context failures instead, never "back inside the bound".
    stale = sorted(
        name
        for name in exclusions
        if name in errors
        and name not in breaching
        and name not in expired
        and name not in premature
    )
    dormant = sorted(name for name in exclusions if name not in errors)
    if stale:
        failures.append(
            "Stale reviewed target-fit exclusions are back inside the bound; "
            f"remove them: {stale}."
        )
    unreported_expired = [name for name in expired if name not in breaching]
    if unreported_expired:
        failures.append(
            _expired_exclusion_failure(
                exclusions, unreported_expired, family="target-fit"
            )
        )
    unreported_premature = [name for name in premature if name not in breaching]
    if unreported_premature:
        failures.append(
            _premature_exclusion_failure(
                exclusions, unreported_premature, family="target-fit"
            )
        )
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
            "reviewed_exclusions": dict(sorted(excluded.items())),
            "failing_targets": failing,
            "stale_exclusions": stale,
            "dormant_exclusions": dormant,
            "expired_exclusions": expired,
            "premature_exclusions": premature,
            "exclusions_evaluated_on": evaluated_on.isoformat(),
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


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, probabilities: Sequence[float]
) -> dict[str, float | None]:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    mass = weights[order]
    total = float(mass.sum())
    if total <= 0.0 or ordered.size == 0:
        return {f"p{int(round(p * 100))}": None for p in probabilities}
    cumulative = np.cumsum(mass) / total
    return {
        f"p{int(round(p * 100))}": float(
            ordered[
                min(int(np.searchsorted(cumulative, p, side="left")), ordered.size - 1)
            ]
        )
        for p in probabilities
    }


def uk_cgt_projection_entrants_gate(
    person: pd.DataFrame,
    person_weights: np.ndarray,
    projection: UKCGTProjection,
    *,
    bound: float,
    bound_source: str,
) -> GateResult:
    """Fence the gainers the engine would tip into liability by uprating.

    The engine freezes the annual exempt amount at its nominal value and
    uprates ``capital_gains`` by per-capita GDP growth, so a gainer at or
    just below the exempt amount in the build period becomes a taxpayer in
    a later projected year without any change in behaviour. For every year
    to the horizon the gate counts the cumulative stock of build-period
    sub-exempt gainers whose uprated gains exceed that year's exempt amount
    (non-decreasing in the year), and the largest count must not exceed
    ``bound``: the published count of the thinnest liable band, people
    already above the exempt amount, so a plausibility ceiling rather than
    an entrant count (microcosm#970). A frame without ``capital_gains``
    cannot be fenced and fails closed.
    """

    if "capital_gains" not in person.columns:
        raise ValueError(
            "cgt_projection_entrants requires person.capital_gains; a frame "
            "without it cannot be fenced."
        )
    gains = pd.to_numeric(person["capital_gains"], errors="raise").to_numpy(dtype=float)
    weights = np.asarray(person_weights, dtype=float)
    if weights.shape != gains.shape:
        raise ValueError("cgt_projection_entrants needs one weight per person.")
    if not np.isfinite(gains).all() or not np.isfinite(weights).all():
        raise ValueError("cgt_projection_entrants requires finite gains and weights.")
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("cgt_projection_entrants requires a positive finite bound.")
    base_exempt = float(projection.exempt_amount_by_year[str(projection.base_year)])
    sub_exempt = (gains > 0.0) & (gains <= base_exempt)
    entrants_by_year: dict[str, float] = {}
    for year in projection.projected_years:
        factor = float(projection.cumulative_gains_factor_by_year[str(year)])
        exempt = float(projection.exempt_amount_by_year[str(year)])
        crossing = sub_exempt & (gains * factor > exempt)
        entrants_by_year[str(year)] = float(weights[crossing].sum())
    worst_year = max(
        entrants_by_year, key=lambda year: (entrants_by_year[year], -int(year))
    )
    max_entrants = entrants_by_year[worst_year]
    sub_exempt_weights = weights[sub_exempt]
    details: dict[str, object] = {
        "base_year": projection.base_year,
        "horizon_year": projection.horizon_year,
        "entrants_by_year": entrants_by_year,
        "worst_year": int(worst_year),
        "max_entrants": max_entrants,
        "bound": float(bound),
        "bound_source": bound_source,
        "cumulative_gains_factor_by_year": dict(
            projection.cumulative_gains_factor_by_year
        ),
        "exempt_amount_by_year": dict(projection.exempt_amount_by_year),
        "gains_growth_parameter": projection.growth_parameter,
        "exempt_amount_parameter": projection.exempt_amount_parameter,
        "projection_engine": projection.engine,
        "sub_exempt": {
            "rows": int(sub_exempt.sum()),
            "weighted_persons": float(sub_exempt_weights.sum()),
            "weighted_at_exempt_amount": float(
                weights[sub_exempt & (gains == base_exempt)].sum()
            ),
            **_weighted_quantiles(
                gains[sub_exempt], sub_exempt_weights, (0.1, 0.5, 0.9)
            ),
        },
    }
    failures: tuple[str, ...] = ()
    if max_entrants > bound:
        failures = (
            f"{UK_CANDIDATE_DATASET_NAME}: {max_entrants:,.0f} weighted sub-exempt "
            f"gainers cross the frozen annual exempt amount by {worst_year} under "
            f"the engine's uprating, above the bound of {bound:,.0f} "
            f"({bound_source}).",
        )
    return GateResult(
        name="cgt_projection_entrants",
        passed=not failures,
        failures=failures,
        details=details,
    )
