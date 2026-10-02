"""Benefit reports and take-up on SPI-channel rows (microcosm#1095, uk-data#514).

The SPI support channel copies whole FRS households, replaces their adults'
incomes with SPI draws and re-imputes each adult's benefit reports with the
FRS-only stage-2 forests of ``hmrc_spi_income_spine``, which condition on age,
gender, region and the new incomes and see nothing of the benefit unit's
entitlement or claim history. On the 30 September national build the SPI rows
reported Income Support, Working and Child Tax Credit and income-based JSA at
2.6 to 9.1 times the FRS rate and contributory JSA at 17 times (two thirds of
its SPI reporters earning above the permitted-work limit), held 64%, 72% and
67% of the reported IIDB, AFCS and bereavement support, kept the FRS twin's
``receives_benefits_in_own_right`` (3.04m SPI persons disagreed with their own
reports) and kept the twin's early Universal Credit take-up anchor (1.51m SPI
benefit units claimed only because the twin reported Universal Credit).

This stage runs on SPI-channel rows only, after ``uc_reporter_redraw`` (the
last writer of Universal Credit reports) and before ``uc_capital_coherence``
and the Pension Credit and Child Benefit take-up stages, which read
qualifying-young-person status and the taxable benefits:

- it zeroes the reports whose award needs an existing legacy claim, an
  out-of-work or incapacity test, or a claim the copied household's new
  incomes cannot support: contributory and income-based JSA, Income Support,
  Working and Child Tax Credit, Severe Disablement Allowance and the Sure
  Start Maternity Grant;
- it restores Industrial Injuries Disablement Benefit, the Armed Forces
  Compensation Scheme and bereavement support from the FRS twin, since they
  follow an injury, service or a death and not income;
- it re-derives the disability flags from the final reports with the FRS
  stage's rule (the categories, which read only AA, DLA and PIP, are
  unchanged and asserted so);
- it rebuilds ``receives_benefits_in_own_right`` from each person's own
  Universal Credit, JSA and ESA reports with ``frs_education``'s rule;
- it redraws ``would_claim_uc`` for SPI benefit units as ``frs_take_up`` drew
  it for the FRS units: a unit with a Universal Credit reporter claims, and the
  other units in the Universal Credit age population claim as the residual of
  the take-up contract's rate, on draws keyed on the SPI unit's own identity.

ESA, the disability benefits, Carer's Allowance, State Pension and Winter Fuel
Payment keep their draws, and FRS base rows are never modified.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    assign_binary_with_anchored_residual,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime import frs_disability
from microcosm.build.uk_runtime.frs_education import (
    BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS,
)
from microcosm.build.uk_runtime.frs_take_up import (
    UK_UC_AGE_ELIGIBLE_AGGREGATE,
    UK_UC_AGE_ELIGIBLE_METHOD,
    UK_UC_AGE_ELIGIBLE_SOURCE,
    UK_UC_TAKE_UP_OUTPUT,
    UKTakeUpPopulationPolicy,
    assert_take_up_stage_population_declaration,
    uc_age_eligible_benunits,
    uk_take_up_population_policy,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    fill_support_channel_from_source,
    support_channel_column,
    support_source_id_column,
)
from microcosm.build.uk_runtime.take_up_contract import (
    UKTakeUpContract,
    load_uk_take_up_contract,
)
from microcosm.build.uk_runtime.uc_capital_coherence import (
    _household_to_benunit_weights,
)
from microcosm.frame import Frame

SPI_BENEFIT_COHERENCE_STAGE_NAME = "spi_benefit_coherence"
SPI_BENEFIT_COHERENCE_ROWS = "spi_synthetic_support_channel"
#: Reports whose award needs an existing legacy claim, an out-of-work or
#: incapacity test, or a claim the copied household's new incomes cannot carry.
SPI_ZEROED_REPORT_COLUMNS = (
    "jsa_contrib_reported",
    "jsa_income_reported",
    "income_support_reported",
    "working_tax_credit_reported",
    "child_tax_credit_reported",
    "sda_reported",
    "ssmg_reported",
)
SPI_ZEROED_REPORT_REASON = (
    "award needs an existing legacy claim, an out-of-work or incapacity test, "
    "or a claim the copied household's new incomes cannot carry"
)
#: Reports that follow an injury, service or a death, not income: the SPI row
#: keeps the FRS twin's value.
SPI_RESTORED_REPORT_COLUMNS = ("iidb_reported", "afcs_reported", "bsp_reported")
SPI_RESTORED_REPORT_REASON = "follows an injury, service or a death, not income"
SPI_TWIN_LINKAGE = "person_source_id -> FRS-channel person_id"
SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT = "receives_benefits_in_own_right"
SPI_BENEFITS_IN_OWN_RIGHT_RULE = (
    "universal_credit_reported + jsa_contrib_reported + jsa_income_reported + "
    "esa_contrib_reported + esa_income_reported > 0 (frs_education), SPI-channel "
    "rows; FRS rows asserted unchanged"
)
SPI_DISABILITY_REFRESH_SCOPE = (
    "disability categories and flags re-derived on SPI-channel rows from the "
    "final reports with the frs_disability rule at its declared year; the "
    "categories, which read only AA, DLA and PIP, asserted unchanged"
)
SPI_UC_REPORTED_ANCHOR = "universal_credit_reported_anchor"
SPI_UC_TAKE_UP_RATE_KEY = "universal_credit"
SPI_UC_TAKE_UP_SEED = 0
#: The disability flags the refresh writes; the categories read only AA, DLA
#: and PIP, which the stage keeps, so they are asserted unchanged instead.
SPI_DISABILITY_FLAG_COLUMNS = tuple(
    column
    for column in frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS
    if not column.endswith("_category")
)
SPI_BENEFIT_COHERENCE_REWRITES = (
    *SPI_ZEROED_REPORT_COLUMNS,
    *SPI_RESTORED_REPORT_COLUMNS,
    *SPI_DISABILITY_FLAG_COLUMNS,
    SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT,
    UK_UC_TAKE_UP_OUTPUT,
)


def spi_benefit_coherence_operations() -> tuple[tuple[str, dict[str, object]], ...]:
    """The stage's operations, in order, as ``(kind, parameters)`` pairs."""

    return (
        (
            "zero_spi_channel_reports",
            {
                "rows": SPI_BENEFIT_COHERENCE_ROWS,
                "columns": list(SPI_ZEROED_REPORT_COLUMNS),
                "reason": SPI_ZEROED_REPORT_REASON,
            },
        ),
        (
            "restore_spi_channel_reports_from_source",
            {
                "rows": SPI_BENEFIT_COHERENCE_ROWS,
                "columns": list(SPI_RESTORED_REPORT_COLUMNS),
                "linkage": SPI_TWIN_LINKAGE,
                "reason": SPI_RESTORED_REPORT_REASON,
            },
        ),
        ("derive", {"scope": SPI_DISABILITY_REFRESH_SCOPE}),
        (
            "derive",
            {
                "output": SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT,
                "formula": SPI_BENEFITS_IN_OWN_RIGHT_RULE,
            },
        ),
        (
            "aggregate_person_to_benunit",
            {
                "method": "any_positive",
                "consumed_only": True,
                "aggregates": {SPI_UC_REPORTED_ANCHOR: "universal_credit_reported"},
            },
        ),
        (
            "aggregate_person_to_benunit",
            {
                "method": UK_UC_AGE_ELIGIBLE_METHOD,
                "consumed_only": True,
                "aggregates": {UK_UC_AGE_ELIGIBLE_AGGREGATE: UK_UC_AGE_ELIGIBLE_SOURCE},
            },
        ),
        (
            "assign_binary_with_anchored_residual",
            {
                "output": UK_UC_TAKE_UP_OUTPUT,
                "draw": UK_UC_TAKE_UP_OUTPUT,
                "rate_key": SPI_UC_TAKE_UP_RATE_KEY,
                "anchor": SPI_UC_REPORTED_ANCHOR,
                "population": UK_UC_AGE_ELIGIBLE_AGGREGATE,
                "seed": SPI_UC_TAKE_UP_SEED,
                "rows": SPI_BENEFIT_COHERENCE_ROWS,
            },
        ),
    )


@dataclass(frozen=True)
class UKSPIBenefitCoherenceResult:
    """Output frame and executed-effect receipt for the SPI benefit pass."""

    frame: Frame
    spi_person_rows: int
    spi_benefit_units: int
    zeroed: Mapping[str, Mapping[str, float]]
    restored: Mapping[str, Mapping[str, float]]
    disability_flags_changed: Mapping[str, int]
    benefits_in_own_right: Mapping[str, float]
    universal_credit_take_up: Mapping[str, float]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": SPI_BENEFIT_COHERENCE_STAGE_NAME,
            "rows": SPI_BENEFIT_COHERENCE_ROWS,
            "spi_person_rows": self.spi_person_rows,
            "spi_benefit_units": self.spi_benefit_units,
            "zeroed": {key: dict(value) for key, value in self.zeroed.items()},
            "restored": {key: dict(value) for key, value in self.restored.items()},
            "disability_flags_changed": dict(self.disability_flags_changed),
            "benefits_in_own_right": dict(self.benefits_in_own_right),
            "universal_credit_take_up": dict(self.universal_credit_take_up),
            "base_rows_unchanged": True,
        }


@dataclass(frozen=True)
class UKSPIBenefitCoherenceStageTransform:
    """Whole-stage callable for the SPI-channel benefit coherence pass."""

    stage: SourceStageSpec
    contract: UKTakeUpContract | None = None
    population_policy: UKTakeUpPopulationPolicy | None = None
    category_rates: object | None = None
    flag_rates: object | None = None
    last_result: UKSPIBenefitCoherenceResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        assert_spi_benefit_coherence_stage_parameters(self.stage)
        result = apply_spi_benefit_coherence(
            frame,
            contract=self.contract or load_uk_take_up_contract(),
            population_policy=(
                self.population_policy
                or uk_take_up_population_policy(uk_time_period(frame))
            ),
            category_rates=self.category_rates,
            flag_rates=self.flag_rates,
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ()

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def apply_spi_benefit_coherence(
    frame: Frame,
    *,
    contract: UKTakeUpContract,
    population_policy: UKTakeUpPopulationPolicy,
    category_rates: object | None = None,
    flag_rates: object | None = None,
) -> UKSPIBenefitCoherenceResult:
    """Zero, restore and re-derive SPI-channel reports; redraw their UC take-up."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    person_channel = support_channel_column("person")
    benunit_channel = support_channel_column("benunit")
    required_person = {
        "person_id",
        "person_benunit_id",
        "person_household_id",
        "age",
        person_channel,
        support_source_id_column("person"),
        *SPI_ZEROED_REPORT_COLUMNS,
        *SPI_RESTORED_REPORT_COLUMNS,
        *BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS,
        *frs_disability.UK_DISABILITY_FLAG_REPORTED_COLUMNS,
        *frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS,
        SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT,
    }
    missing = sorted(required_person - set(person.columns))
    if missing:
        raise ValueError(f"SPI benefit coherence person columns missing: {missing}.")
    missing = sorted(
        {"benunit_id", benunit_channel, UK_UC_TAKE_UP_OUTPUT} - set(benunit.columns)
    )
    if missing:
        raise ValueError(f"SPI benefit coherence benunit columns missing: {missing}.")
    channels = set(person[person_channel].unique())
    unexpected = sorted(
        channels - {BASE_FRS_SUPPORT_CHANNEL, SPI_SYNTHETIC_SUPPORT_CHANNEL}
    )
    if unexpected:
        raise ValueError(
            "SPI benefit coherence runs before any other support channel exists; "
            f"found {unexpected}."
        )
    spi = (person[person_channel] == SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy()
    base = ~spi
    if not spi.any():
        raise ValueError("SPI benefit coherence found no SPI-channel person rows.")
    rewritten = list(
        dict.fromkeys(
            (
                *SPI_ZEROED_REPORT_COLUMNS,
                *SPI_RESTORED_REPORT_COLUMNS,
                *frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS,
                SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT,
            )
        )
    )
    # The categories are in the base-row check too: the refresh must leave
    # them untouched everywhere.
    before = person[rewritten].copy()
    weights = _person_weights(person, household, frame.weights_for("household").values)

    zeroed: dict[str, dict[str, float]] = {}
    for column in SPI_ZEROED_REPORT_COLUMNS:
        values = pd.to_numeric(person[column], errors="coerce").fillna(0.0).to_numpy()
        reporting = spi & (values > 0.0)
        zeroed[column] = {
            "rows_reporting_before": int(reporting.sum()),
            "weighted_reporters_before": float(weights[reporting].sum()),
            "weighted_amount_before": float((values * weights)[spi].sum()),
        }
        person.loc[spi, column] = 0.0
        after = pd.to_numeric(person[column], errors="coerce").fillna(0.0).to_numpy()
        zeroed[column]["rows_reporting_after"] = int((spi & (after > 0.0)).sum())

    twin = person.loc[base, ["person_id", *SPI_RESTORED_REPORT_COLUMNS]]
    restored_frame = fill_support_channel_from_source(
        person,
        twin,
        entity="person",
        columns=SPI_RESTORED_REPORT_COLUMNS,
    )
    restored: dict[str, dict[str, float]] = {}
    for column in SPI_RESTORED_REPORT_COLUMNS:
        old = pd.to_numeric(person[column], errors="coerce").fillna(0.0).to_numpy()
        new = (
            pd.to_numeric(restored_frame[column], errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        restored[column] = {
            "rows_changed": int((spi & ~np.isclose(old, new)).sum()),
            "weighted_amount_before": float((old * weights)[spi].sum()),
            "weighted_amount_after": float((new * weights)[spi].sum()),
            "weighted_reporters_before": float(weights[spi & (old > 0.0)].sum()),
            "weighted_reporters_after": float(weights[spi & (new > 0.0)].sum()),
        }
        person[column] = restored_frame[column].to_numpy()
    restored_check = fill_support_channel_from_source(
        person,
        person.loc[base, ["person_id", *SPI_RESTORED_REPORT_COLUMNS]],
        entity="person",
        columns=SPI_RESTORED_REPORT_COLUMNS,
    )
    for column in SPI_RESTORED_REPORT_COLUMNS:
        restored[column]["rows_differing_from_twin_after"] = int(
            (
                spi
                & ~np.isclose(
                    pd.to_numeric(person[column], errors="coerce").fillna(0.0),
                    pd.to_numeric(restored_check[column], errors="coerce").fillna(0.0),
                )
            ).sum()
        )

    flags_changed = _refresh_disability(
        person, spi=spi, category_rates=category_rates, flag_rates=flag_rates
    )

    own_right_rule = (
        person[list(BENEFITS_IN_OWN_RIGHT_REPORTED_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum(axis=1)
        .gt(0.0)
        .to_numpy()
    )
    stored_own_right = person[SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT].astype(bool).to_numpy()
    drifted = base & (own_right_rule != stored_own_right)
    if drifted.any():
        raise ValueError(
            f"{int(drifted.sum())} FRS-channel person(s) carry "
            f"{SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT} that disagrees with their own "
            "reports; the frs_education rule and an upstream stage have drifted."
        )
    own_right = {
        "spi_rows_changed": int((spi & (own_right_rule != stored_own_right)).sum()),
        "spi_true_before": int((spi & stored_own_right).sum()),
        "spi_true_after": int((spi & own_right_rule).sum()),
        "weighted_spi_changed": float(
            weights[spi & (own_right_rule != stored_own_right)].sum()
        ),
    }
    person.loc[spi, SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT] = own_right_rule[spi]
    person[SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT] = person[
        SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT
    ].astype(bool)
    own_right["mismatches_after"] = int(
        (
            person[SPI_BENEFITS_IN_OWN_RIGHT_OUTPUT].to_numpy(dtype=bool)
            != own_right_rule
        ).sum()
    )

    for column in rewritten:
        unchanged = before.loc[base, column].equals(person.loc[base, column])
        if not unchanged:
            raise ValueError(
                f"SPI benefit coherence changed FRS-channel {column!r}; base rows "
                "are never modified."
            )

    benunit, take_up = _redraw_spi_uc_take_up(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=frame.weights_for("household").values,
        contract=contract,
        population_policy=population_policy,
    )
    result = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return UKSPIBenefitCoherenceResult(
        frame=result,
        spi_person_rows=int(spi.sum()),
        spi_benefit_units=int(take_up["spi_benefit_units"]),
        zeroed=zeroed,
        restored=restored,
        disability_flags_changed=flags_changed,
        benefits_in_own_right=own_right,
        universal_credit_take_up=take_up,
    )


def _redraw_spi_uc_take_up(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    household_weights: np.ndarray,
    contract: UKTakeUpContract,
    population_policy: UKTakeUpPopulationPolicy,
) -> tuple[pd.DataFrame, dict[str, float]]:
    benunit = benunit.copy()
    spi_units = (
        benunit[support_channel_column("benunit")] == SPI_SYNTHETIC_SUPPORT_CHANNEL
    ).to_numpy()
    reported = (
        pd.to_numeric(person["universal_credit_reported"], errors="coerce").fillna(0.0)
        > 0.0
    )
    reporter = (
        benunit["benunit_id"]
        .isin(set(person.loc[reported, "person_benunit_id"]))
        .to_numpy(dtype=bool)
    )
    population = uc_age_eligible_benunits(person, benunit, population_policy)
    rate = contract.rate(SPI_UC_TAKE_UP_RATE_KEY)
    draws = stable_identity_uniforms(
        benunit.loc[spi_units, "benunit_id"].to_numpy(),
        seed=SPI_UC_TAKE_UP_SEED,
        salt=UK_UC_TAKE_UP_OUTPUT,
    )
    redrawn = assign_binary_with_anchored_residual(
        draws,
        rate,
        anchor=reporter[spi_units],
        population=population[spi_units],
    )
    previous = benunit[UK_UC_TAKE_UP_OUTPUT].fillna(False).astype(bool).to_numpy()
    weights = _household_to_benunit_weights(
        benunit,
        person=person,
        household=household,
        household_weights=household_weights,
    )
    after = previous.copy()
    after[spi_units] = redrawn
    benunit[UK_UC_TAKE_UP_OUTPUT] = after
    spi_population = spi_units & population
    receipt = {
        "rate_key": SPI_UC_TAKE_UP_RATE_KEY,
        "rate": float(rate),
        "seed": SPI_UC_TAKE_UP_SEED,
        "population_state_pension_age": int(population_policy.state_pension_age),
        "spi_benefit_units": int(spi_units.sum()),
        "spi_population_units": int(spi_population.sum()),
        "spi_reporter_units": int((spi_units & reporter).sum()),
        "spi_claiming_before": int((spi_units & previous).sum()),
        "spi_claiming_after": int((spi_units & after).sum()),
        "set_true": int((spi_units & ~previous & after).sum()),
        "set_false": int((spi_units & previous & ~after).sum()),
        "weighted_spi_claiming_before": float(weights[spi_units & previous].sum()),
        "weighted_spi_claiming_after": float(weights[spi_units & after].sum()),
        "weighted_spi_population": float(weights[spi_population].sum()),
        "reporters_not_claiming": int((spi_units & reporter & ~after).sum()),
        "outside_population_non_reporters_claiming": int(
            (spi_units & ~population & ~reporter & after).sum()
        ),
    }
    return benunit, receipt


def _refresh_disability(
    person: pd.DataFrame,
    *,
    spi: np.ndarray,
    category_rates: object | None,
    flag_rates: object | None,
) -> dict[str, int]:
    if category_rates is None or flag_rates is None:
        from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule

        year = resolve_uk_year_rule(frs_disability.YEAR_RULE)
        if category_rates is None:
            category_rates = frs_disability.uk_dwp_disability_category_rates(year)
        if flag_rates is None:
            flag_rates = frs_disability.uk_dwp_disability_flag_rates(year)
    rows = person.loc[spi]
    derived = frs_disability.derive_frs_disability(
        rows, category_rates=category_rates, flag_rates=flag_rates
    )
    changed: dict[str, int] = {}
    for column in frs_disability.FRS_DISABILITY_OUTPUT_COLUMNS:
        new = derived[column].to_numpy()
        old = rows[column].to_numpy()
        moved = int((new != old).sum())
        if column.endswith("_category"):
            if moved:
                raise ValueError(
                    f"SPI benefit coherence moved {moved} {column!r} value(s); the "
                    "categories read only AA, DLA and PIP, which this stage keeps."
                )
            continue
        changed[column] = moved
        person.loc[spi, column] = new.astype(bool)
    return changed


def _person_weights(
    person: pd.DataFrame, household: pd.DataFrame, household_weights: np.ndarray
) -> np.ndarray:
    by_household = pd.Series(
        np.asarray(household_weights, dtype=float),
        index=household["household_id"].to_numpy(),
    )
    weights = person["person_household_id"].map(by_household)
    if weights.isna().any():
        raise ValueError("SPI benefit coherence persons must map to a household.")
    return weights.to_numpy(dtype=float)


def assert_spi_benefit_coherence_stage_parameters(stage: SourceStageSpec) -> None:
    """Refuse a manifest whose operations differ from the code's."""

    if stage.stage != SPI_BENEFIT_COHERENCE_STAGE_NAME:
        raise ValueError(
            f"SPI benefit coherence received stage {stage.stage!r}, expected "
            f"{SPI_BENEFIT_COHERENCE_STAGE_NAME!r}."
        )
    expected = [
        (kind, dict(parameters))
        for kind, parameters in spi_benefit_coherence_operations()
    ]
    actual = [
        (operation.kind, dict(operation.parameters)) for operation in stage.operations
    ]
    if actual != expected:
        raise ValueError(
            f"{SPI_BENEFIT_COHERENCE_STAGE_NAME} operations drifted: expected "
            f"{expected}, got {actual}."
        )
    assert_take_up_stage_population_declaration(stage)
    if stage.outputs != () or tuple(stage.rewrites) != SPI_BENEFIT_COHERENCE_REWRITES:
        raise ValueError(
            f"{SPI_BENEFIT_COHERENCE_STAGE_NAME} must declare no new outputs and "
            f"exactly the rewrites {SPI_BENEFIT_COHERENCE_REWRITES}."
        )


__all__ = [
    "SPI_BENEFIT_COHERENCE_REWRITES",
    "SPI_DISABILITY_FLAG_COLUMNS",
    "SPI_BENEFIT_COHERENCE_STAGE_NAME",
    "SPI_RESTORED_REPORT_COLUMNS",
    "SPI_ZEROED_REPORT_COLUMNS",
    "UKSPIBenefitCoherenceResult",
    "UKSPIBenefitCoherenceStageTransform",
    "apply_spi_benefit_coherence",
    "assert_spi_benefit_coherence_stage_parameters",
    "spi_benefit_coherence_operations",
]
