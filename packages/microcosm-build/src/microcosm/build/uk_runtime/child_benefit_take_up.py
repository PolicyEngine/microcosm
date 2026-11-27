"""Child Benefit claims and opt-outs on post-SPI incomes (microcosm#1063).

``frs_take_up`` draws ``would_claim_child_benefit`` early, at one flat rate
over every benefit unit with the FRS reporters added on top, and
``child_benefit_opts_out`` at a flat rate over every benefit unit whatever its
income. This stage redraws both once the SPI income chain has set the incomes
the charge is assessed on, against HMRC's Child Benefit statistics, after
redrawing the SPI copies' own Child Benefit reports.

Copies' reports. The SPI support channel's copies of FRS households take their
benefit reports from the SPI income chain's forests, which see the copy's age,
gender, region and incomes but not its children: a copy with no eligible child
can report Child Benefit, and copy families report too rarely below the charge
start and too often where it takes the whole benefit, against the FRS families
like them. A reporter always claims and is never opted out, so the
stage first redraws whether each copy's benefit unit reports (microcosm#1508).
A copy family reports at the weighted share of FRS families that report in its
cell: its eligible children (1, 2, 3 or more) by the charge band of its highest
adjusted net income (at most the charge start, inside the taper, or fully
charged). Adjacent bands are pooled until a cell rests on enough FRS source
households, and a children group that cannot is merged with its neighbour
first. The draw is coupled to the copy's FRS twin, the benefit unit it was
cloned from, so the copy keeps as much of its twin's report as its own cell's
rate allows:

    copy reports  iff  u * p_twin < p_copy               (twin reports)
                       p_twin + u * (1 - p_twin) < p_copy (twin does not)
                       u < p_copy                         (twin in no cell)

with ``u`` uniform on the copy's identity: a copy in its twin's cell keeps the
twin's report, and where the twins report at their cells' rates the copies
report at their own. A twin with no eligible child is in no cell and has no
rate to couple to, so its copy draws on its own at its cell's rate. A
reporting copy takes its twin's reported amounts, or the engine's annual Child
Benefit entitlement on its benefit unit head when the twin does not report; a
copy without an eligible child takes its twin's reports. FRS rows are never
modified.

Claims. One temporary engine materialization gives each person's eligibility
(a child, or a qualifying young person). HMRC publishes the share of eligible
children for whom Child Benefit is claimed, opted-out families included, by
single year of age. The model claims for all of a family's children or for
none, so the draw is by family: a family that reports the benefit claims, and
the other families of key age ``g`` claim at a rate ``r_g``.

The key is the age of the family's eldest eligible child under 16, or of its
eldest eligible child when none is under 16 (microcosm#1508). Payments for a
young person of 16 to 19 carry on only while the family extends the claim for
approved education or training, so HMRC's claimed shares fall after 16. A
claim is made child by child, and a family whose young person's claim lapses
keeps claiming for the younger children. Keying such a family on its young
person instead would apply the young person's low rate to the younger
children too, and the rates for younger keys could not make up the loss.

The rates are solved so that the claimed share of the children of each single
year of age is the published one, ages 15 down to 0 first and then 19 down to
16:

    r_g = (t_g * C_g - R_g - sum_{k solved} N_{g,k} * r_k) / N_{g,g}

with ``t_g`` the published rate at age ``g``, ``C_g`` the eligible children
aged ``g``, ``R_g`` those in reporting families and ``N_{g,k}`` those in
non-reporting families of key age ``k``, all weighted at the stage's household
weights. A child under 16 is in a family keyed at its own age or older, so its
older keys are solved first; a young person of 16 or over is in a family keyed
under 16, solved in the first pass, or in a family of young people only, keyed
at its own age or older. A rate outside [0, 1] is clipped and the receipt says
so (reporters alone exceed the published rate, or every family claiming still
falls short of it).

Opt-outs. HMRC publishes the families registered and, of them, the families
that opted out of payment because of the High Income Child Benefit Charge.
That share of the claiming families opts out, drawn among claiming families
that do not report receipt. For opt-out-aware models, candidates must have a
positive charge fraction at least as large as the installed
``opt_out_charge_share``. The default share of one restricts the draw to fully
charged families; a smaller eligible pool leaves an explicit shortfall. Older
models retain their fully charged pool followed by a taper fallback. Within
either contract, fully charged candidates come first.

Models exposing ``gov.hmrc.child_benefit.opt_out_charge_share`` (UK #2140)
read claims and opt-outs separately, so ``would_claim_child_benefit`` includes
registered opted-out families. Older models ignore the opt-out input and need
the legacy ``claims and not opted out`` payment flag. The installed parameter
tree selects the encoding and the receipt records it; no dataset-name or
vintage heuristic is used. A benefit unit with no eligible child keeps its
early draw and is never opted out.

The in-payment audit counts ``claims and not opted out``. New-contract opt-out
candidates match the installed baseline payment-suppression rule; the receipt
is still a draw audit, not a fresh model run or evidence of population fit.

The receipt counts the eligible child rows, overall and at each single year of
age, among geographic clone 0's rows (microcosm#931): the rows a one-clone
frame has, so the stage gate's row floor and its sampling band read the same
count at every clone count. Its weighted figures and realised rates are taken
over every clone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_take_up import geographic_clone_zero_rows
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    support_channel_column,
    support_source_id_column,
)
from microcosm.build.uk_runtime.uc_capital_coherence import (
    _boolean_values,
    _household_to_benunit_weights,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

CHILD_BENEFIT_TAKE_UP_STAGE_NAME = "child_benefit_take_up"
CHILD_BENEFIT_CLAIM_OUTPUT = "would_claim_child_benefit"
CHILD_BENEFIT_OPT_OUT_OUTPUT = "child_benefit_opts_out"
CHILD_BENEFIT_TAKE_UP_SEED = 0
CHILD_BENEFIT_ELIGIBILITY_VARIABLE = (
    "is_child_or_qualifying_young_person_for_child_benefit"
)
CHILD_BENEFIT_CHARGE_INCOME_VARIABLE = "adjusted_net_income"
#: The annual amount a copy's new report carries (benefit unit grain).
CHILD_BENEFIT_ENTITLEMENT_VARIABLE = "child_benefit_entitlement"
CHILD_BENEFIT_ENGINE_VARIABLES = (
    CHILD_BENEFIT_ELIGIBILITY_VARIABLE,
    CHILD_BENEFIT_CHARGE_INCOME_VARIABLE,
    CHILD_BENEFIT_ENTITLEMENT_VARIABLE,
)
CHILD_BENEFIT_REPORT_OUTPUT = "child_benefit_reported"
CHILD_BENEFIT_REPORTED_ANCHOR = "child_benefit_reported_anchor"
CHILD_BENEFIT_AGGREGATES = {CHILD_BENEFIT_REPORTED_ANCHOR: CHILD_BENEFIT_REPORT_OUTPUT}
#: The SPI copies' report cells (microcosm#1508): eligible children by the
#: charge band of the benefit unit's highest adjusted net income.
CHILD_BENEFIT_COPY_CHILDREN = ("1", "2", "3+")
CHILD_BENEFIT_COPY_INCOME_BANDS = ("below_charge_start", "taper", "fully_charged")
CHILD_BENEFIT_COPY_MINIMUM_SOURCE_HOUSEHOLDS = 30
CHILD_BENEFIT_COPY_SOURCE_HOUSEHOLD = "source_household_id"
CHILD_BENEFIT_COPY_LANDING = "is_benunit_head"
#: A receipt figure resting on fewer survey households is withheld (CD171's
#: floor of 3).
CHILD_BENEFIT_RECEIPT_FLOOR = 3
CHILD_BENEFIT_COPY_REPORT_RULE = (
    "an SPI copy's benefit unit with an eligible child reports at the weighted "
    "share of FRS benefit units with an eligible child that report Child "
    "Benefit in its cell of eligible children (1, 2, 3+) by the charge band of "
    "its highest adjusted net income (at most the charge start, inside the "
    "taper, fully charged), adjacent bands pooled until a cell rests on "
    f"{CHILD_BENEFIT_COPY_MINIMUM_SOURCE_HOUSEHOLDS} FRS source households and "
    "a children group short of that merged with its neighbour first; coupled "
    "to its FRS twin (benunit_source_id): a copy of a reporting twin reports "
    "when u * p_twin < p_copy, of a non-reporting twin when p_twin + u * (1 - "
    "p_twin) < p_copy; a reporting copy takes its twin's reported amounts "
    "(person_source_id), or child_benefit_entitlement on its benefit unit head "
    "when the twin does not report; a copy without an eligible child takes its "
    "twin's reports; FRS rows are unchanged"
)
CHILD_BENEFIT_STATISTICS_RESOURCE = "hmrc_child_benefit_statistics.json"
#: HMRC Child Benefit statistics, annual release at August 2025: the claim
#: rates are estimated at the May reference month, the caseload at August.
CHILD_BENEFIT_CLAIM_RATE_PERIOD = "2025-05"
CHILD_BENEFIT_CASELOAD_PERIOD = "2025-08"
CHILD_BENEFIT_CLAIM_RATE_MEASURE = "claim_rate"
CHILD_BENEFIT_FAMILIES_REGISTERED_MEASURE = "families_registered"
CHILD_BENEFIT_FAMILIES_IN_PAYMENT_MEASURE = "families_in_payment"
CHILD_BENEFIT_FAMILIES_OPTED_OUT_MEASURE = "families_opted_out_claimant_all"
CHILD_BENEFIT_CHILDREN_IN_PAYMENT_MEASURE = "children_in_payment_united_kingdom"
CHILD_BENEFIT_CHILDREN_OPTED_OUT_MEASURE = "children_opted_out_united_kingdom"
CHILD_BENEFIT_ALL_AGES = "All ages"
CHILD_BENEFIT_CHARGE_PARAMETERS = "gov.hmrc.income_tax.charges.CB_HITC"
#: Child Benefit's children are under 16; from 16 a qualifying young person's
#: payments need the claim extended (SSCBA 1992 s.142). The family key prefers
#: a child under this age (microcosm#1508).
CHILD_BENEFIT_KEY_AGE_LIMIT = 16
CHILD_BENEFIT_FAMILY_KEY = (
    "age of the benefit unit's eldest eligible child under 16, or of its eldest "
    "eligible child when none is under 16"
)
CHILD_BENEFIT_CLAIM_RULE = (
    "a reporter claims; a non-reporting family of key age g claims at r_g = "
    "(t_g * C_g - R_g - sum over the solved keys k of N_gk * r_k) / N_gg, "
    "weighted, solved for ages 15 down to 0 and then 19 down to 16, and "
    "clipped to [0, 1]"
)
CHILD_BENEFIT_OPT_OUT_RULE = (
    "published opted-out share of registered families, of the claiming "
    "families; drawn among claiming non-reporting families, fully charged first "
    "then inside the taper; opt-out-aware models restrict both pools to a "
    "positive charge fraction at least the installed opt_out_charge_share, "
    "recording any pool shortfall; legacy models retain the unfiltered pools"
)
CHILD_BENEFIT_PAYMENT_RULE = (
    "would_claim_child_benefit = claims on families with an eligible child "
    "when the model exposes gov.hmrc.child_benefit.opt_out_charge_share; "
    "otherwise claims and not opted out for legacy models; a benefit unit "
    "without an eligible child keeps its early draw"
)


@dataclass(frozen=True)
class UKChildBenefitStatistics:
    """HMRC's published rows the stage draws against."""

    claim_rates: Mapping[int, float]
    all_ages_claim_rate: float
    families_registered: float
    families_in_payment: float
    families_opted_out: float
    children_in_payment: float
    children_opted_out: float
    claim_rate_period: str = CHILD_BENEFIT_CLAIM_RATE_PERIOD
    caseload_period: str = CHILD_BENEFIT_CASELOAD_PERIOD

    def __post_init__(self) -> None:
        if not self.claim_rates:
            raise ValueError("Child Benefit claim rates by age are empty.")
        for age, rate in self.claim_rates.items():
            if not isinstance(age, int) or not 0.0 <= float(rate) <= 1.0:
                raise ValueError(
                    f"Child Benefit claim rate at age {age!r} must lie in [0, 1], "
                    f"got {rate!r}."
                )
        if not 0.0 < self.families_opted_out < self.families_registered:
            raise ValueError(
                "Child Benefit opted-out families must be a positive part of the "
                "registered families."
            )

    @property
    def opt_out_share(self) -> float:
        """Opted-out families over registered families."""

        return float(self.families_opted_out) / float(self.families_registered)


@dataclass(frozen=True)
class UKChildBenefitChargeThresholds:
    """Adjusted net income at which the charge starts and takes it all."""

    phase_out_start: float
    phase_out_end: float
    source: str = "caller"
    supports_opt_out: bool = False
    opt_out_charge_share: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.phase_out_start < self.phase_out_end:
            raise ValueError(
                "Child Benefit charge thresholds must satisfy 0 < start < end."
            )
        if self.supports_opt_out:
            share = self.opt_out_charge_share
            if (
                isinstance(share, bool)
                or not isinstance(share, Real)
                or not np.isfinite(share)
            ):
                raise ValueError(
                    "Child Benefit opt_out_charge_share must be a finite numeric "
                    "value when the model supports opt-outs."
                )


def load_child_benefit_statistics() -> UKChildBenefitStatistics:
    """Read the vendored HMRC rows; a copy that lags the feed pin refuses."""

    def one(measure: str, period: str, **dimensions: object) -> float:
        rows = vendored_rows(
            CHILD_BENEFIT_STATISTICS_RESOURCE,
            measure_id=measure,
            period_type="month",
            period_value=period,
            **({"dimensions": dimensions} if dimensions else {}),
        )
        if len(rows) != 1:
            raise ValueError(
                f"{CHILD_BENEFIT_STATISTICS_RESOURCE} must carry exactly one "
                f"{measure} row for {period} {dimensions or ''}; found {len(rows)}."
            )
        return float(rows[0]["value"])

    rate_rows = vendored_rows(
        CHILD_BENEFIT_STATISTICS_RESOURCE,
        measure_id=CHILD_BENEFIT_CLAIM_RATE_MEASURE,
        period_type="month",
        period_value=CHILD_BENEFIT_CLAIM_RATE_PERIOD,
    )
    rates: dict[int, float] = {}
    all_ages: float | None = None
    for row in rate_rows:
        if row.get("unit") != "percent":
            raise ValueError(
                f"{CHILD_BENEFIT_STATISTICS_RESOURCE} claim rates must be in "
                f"percent, got {row.get('unit')!r}."
            )
        age = (row.get("dimensions") or {}).get("child_age")
        # The publisher states a percentage; the stage's arithmetic is on the
        # fraction.
        value = float(row["value"]) / 100.0
        if str(age) == CHILD_BENEFIT_ALL_AGES:
            all_ages = value
            continue
        key = int(age)
        if key in rates:
            raise ValueError(
                f"{CHILD_BENEFIT_STATISTICS_RESOURCE} repeats the claim rate at "
                f"age {key}."
            )
        rates[key] = value
    if all_ages is None or sorted(rates) != list(range(min(rates), max(rates) + 1)):
        raise ValueError(
            f"{CHILD_BENEFIT_STATISTICS_RESOURCE} must carry the all-ages claim "
            "rate and one rate for every single year of age without gaps."
        )
    caseload = CHILD_BENEFIT_CASELOAD_PERIOD
    return UKChildBenefitStatistics(
        claim_rates=rates,
        all_ages_claim_rate=all_ages,
        families_registered=one(CHILD_BENEFIT_FAMILIES_REGISTERED_MEASURE, caseload),
        families_in_payment=one(CHILD_BENEFIT_FAMILIES_IN_PAYMENT_MEASURE, caseload),
        families_opted_out=one(CHILD_BENEFIT_FAMILIES_OPTED_OUT_MEASURE, caseload),
        children_in_payment=one(
            CHILD_BENEFIT_CHILDREN_IN_PAYMENT_MEASURE,
            caseload,
            child_age=CHILD_BENEFIT_ALL_AGES,
        ),
        children_opted_out=one(
            CHILD_BENEFIT_CHILDREN_OPTED_OUT_MEASURE,
            caseload,
            child_age=CHILD_BENEFIT_ALL_AGES,
        ),
    )


def uk_child_benefit_charge_thresholds(
    build_period: int | str,
) -> UKChildBenefitChargeThresholds:
    """Read the taper and payment-suppression share from the installed model."""

    try:
        import policyengine_uk
        from policyengine_core.parameters import ParameterNode
    except ImportError as exc:
        raise ImportError(
            "The Child Benefit charge thresholds require "
            "`uv sync --all-packages --extra uk`."
        ) from exc
    from importlib import metadata

    parameters = ParameterNode(
        directory_path=str(Path(policyengine_uk.__file__).parent / "parameters")
    )
    instant = f"{int(build_period)}-01-01"
    charge = parameters.gov.hmrc.income_tax.charges.CB_HITC
    child_benefit = parameters.gov.hmrc.child_benefit
    supports_opt_out = "opt_out_charge_share" in child_benefit.children
    return UKChildBenefitChargeThresholds(
        phase_out_start=float(charge.phase_out_start(instant)),
        phase_out_end=float(charge.phase_out_end(instant)),
        supports_opt_out=supports_opt_out,
        opt_out_charge_share=(
            child_benefit.opt_out_charge_share(instant) if supports_opt_out else None
        ),
        source=(
            f"policyengine-uk {metadata.version('policyengine-uk')} "
            f"{CHILD_BENEFIT_CHARGE_PARAMETERS} at {instant}"
        ),
    )


@dataclass(frozen=True)
class UKChildBenefitTakeUpResult:
    """Output frame and executed-effect receipt for the Child Benefit redraw."""

    frame: Frame
    claim_export: Mapping[str, object]
    copy_reports: Mapping[str, object]
    claims: Mapping[str, object]
    opt_outs: Mapping[str, object]
    in_payment: Mapping[str, object]
    reporters_without_eligible_child: Mapping[str, float]
    changed_units: Mapping[str, int]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": CHILD_BENEFIT_TAKE_UP_STAGE_NAME,
            "seed": CHILD_BENEFIT_TAKE_UP_SEED,
            "family_key": CHILD_BENEFIT_FAMILY_KEY,
            "copy_report_rule": CHILD_BENEFIT_COPY_REPORT_RULE,
            "claim_rule": CHILD_BENEFIT_CLAIM_RULE,
            "opt_out_rule": CHILD_BENEFIT_OPT_OUT_RULE,
            "payment_rule": CHILD_BENEFIT_PAYMENT_RULE,
            "claim_export": dict(self.claim_export),
            "copy_reports": dict(self.copy_reports),
            "claims": dict(self.claims),
            "opt_outs": dict(self.opt_outs),
            "in_payment": dict(self.in_payment),
            "reporters_without_eligible_child": dict(
                self.reporters_without_eligible_child
            ),
            "changed_units": dict(self.changed_units),
        }


@dataclass(frozen=True)
class UKChildBenefitTakeUpStageTransform:
    """Whole-stage callable for the post-SPI Child Benefit redraw."""

    stage: SourceStageSpec
    engine: object
    statistics: UKChildBenefitStatistics | None = None
    thresholds: UKChildBenefitChargeThresholds | None = None
    last_result: UKChildBenefitTakeUpResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = redraw_child_benefit_take_up(
            frame,
            engine=self.engine,
            statistics=self.statistics or load_child_benefit_statistics(),
            thresholds=self.thresholds
            or uk_child_benefit_charge_thresholds(uk_time_period(frame)),
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


def redraw_child_benefit_take_up(
    frame: Frame,
    *,
    engine: object,
    statistics: UKChildBenefitStatistics,
    thresholds: UKChildBenefitChargeThresholds,
    minimum_source_households: int = CHILD_BENEFIT_COPY_MINIMUM_SOURCE_HOUSEHOLDS,
) -> UKChildBenefitTakeUpResult:
    """Redraw the copies' reports, then the claim and opt-out flags."""

    validate_uk_national_frame(frame)
    assert_rules_engine_country(engine, "uk")
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    for table, columns, label in (
        (
            person,
            (
                "person_id",
                "person_benunit_id",
                "person_household_id",
                "age",
                CHILD_BENEFIT_REPORT_OUTPUT,
                CHILD_BENEFIT_COPY_LANDING,
                support_source_id_column("person"),
            ),
            "person",
        ),
        (
            benunit,
            (
                "benunit_id",
                CHILD_BENEFIT_CLAIM_OUTPUT,
                CHILD_BENEFIT_OPT_OUT_OUTPUT,
                support_channel_column("benunit"),
                support_source_id_column("benunit"),
            ),
            "benunit",
        ),
        (household, ("household_id", CHILD_BENEFIT_COPY_SOURCE_HOUSEHOLD), "household"),
    ):
        missing = sorted(set(columns) - set(table.columns))
        if missing:
            raise ValueError(
                f"Child Benefit take-up {label} columns missing: {missing}."
            )
    period = uk_time_period(frame)
    materialized = engine.materialize(
        frame, list(CHILD_BENEFIT_ENGINE_VARIABLES), period
    )
    missing = sorted(set(CHILD_BENEFIT_ENGINE_VARIABLES) - set(materialized))
    if missing:
        raise ValueError(f"Child Benefit take-up engine outputs missing: {missing}.")
    eligible_child = _aligned(
        materialized[CHILD_BENEFIT_ELIGIBILITY_VARIABLE],
        len(person),
        CHILD_BENEFIT_ELIGIBILITY_VARIABLE,
    ).astype(bool)
    income = _aligned(
        materialized[CHILD_BENEFIT_CHARGE_INCOME_VARIABLE],
        len(person),
        CHILD_BENEFIT_CHARGE_INCOME_VARIABLE,
    ).astype(np.float64)
    weights = _household_to_benunit_weights(
        benunit,
        person=person,
        household=household,
        household_weights=frame.weights_for("household").values,
    )
    benunit_ids = benunit["benunit_id"].to_numpy()
    position = pd.Series(np.arange(len(benunit)), index=benunit_ids)
    person_position = person["person_benunit_id"].map(position).to_numpy()
    if np.isnan(person_position.astype(np.float64)).any():
        raise ValueError("Child Benefit take-up found a person outside every unit.")
    person_position = person_position.astype(np.int64)
    ages = np.floor(pd.to_numeric(person["age"], errors="raise").to_numpy(float))
    child_units = person_position[eligible_child]
    child_ages = ages[eligible_child].astype(np.int64)
    eldest = np.full(len(benunit), -1, dtype=np.int64)
    np.maximum.at(eldest, child_units, child_ages)
    children = np.bincount(child_units, minlength=len(benunit)).astype(np.int64)
    # The charge falls on the unit's highest adjusted net income among the
    # people who are not themselves the children the benefit is paid for.
    highest_income = np.full(len(benunit), -np.inf)
    np.maximum.at(
        highest_income, person_position[~eligible_child], income[~eligible_child]
    )
    highest_income[~np.isfinite(highest_income)] = 0.0
    family = children > 0
    entitlement = _aligned(
        materialized[CHILD_BENEFIT_ENTITLEMENT_VARIABLE],
        len(benunit),
        CHILD_BENEFIT_ENTITLEMENT_VARIABLE,
        table="benefit unit",
    ).astype(np.float64)
    reported = pd.to_numeric(
        person[CHILD_BENEFIT_REPORT_OUTPUT], errors="coerce"
    ).to_numpy(np.float64)
    reported = np.where(np.isnan(reported), 0.0, reported)
    if (reported < 0.0).any():
        raise ValueError(f"{CHILD_BENEFIT_REPORT_OUTPUT} must be nonnegative.")
    landed, copy_rows, copy_receipt = redraw_spi_child_benefit_reports(
        person=person,
        benunit=benunit,
        household=household,
        person_position=person_position,
        eligible_child=eligible_child,
        family=family,
        children=children,
        highest_income=highest_income,
        reported=reported,
        entitlement=entitlement,
        weights=weights,
        thresholds=thresholds,
        minimum_source_households=minimum_source_households,
    )
    person.loc[copy_rows, CHILD_BENEFIT_REPORT_OUTPUT] = landed[copy_rows]
    original = frame.table("person")[CHILD_BENEFIT_REPORT_OUTPUT]
    if not np.array_equal(
        pd.to_numeric(
            person.loc[~copy_rows, CHILD_BENEFIT_REPORT_OUTPUT], errors="coerce"
        ).to_numpy(np.float64),
        pd.to_numeric(original.loc[~copy_rows], errors="coerce").to_numpy(np.float64),
        equal_nan=True,
    ):
        raise RuntimeError("Child Benefit take-up touched FRS reported amounts.")
    reporter = _units_with_positive(landed, person_position, len(benunit))
    claim_draws = stable_identity_uniforms(
        benunit_ids, seed=CHILD_BENEFIT_TAKE_UP_SEED, salt=CHILD_BENEFIT_CLAIM_OUTPUT
    )
    opt_out_draws = stable_identity_uniforms(
        benunit_ids, seed=CHILD_BENEFIT_TAKE_UP_SEED, salt=CHILD_BENEFIT_OPT_OUT_OUTPUT
    )

    claims, claim_receipt = assign_child_benefit_claims(
        child_units=child_units,
        child_ages=child_ages,
        eldest=eldest,
        reporter=reporter,
        weights=weights,
        draws=claim_draws,
        statistics=statistics,
        # Each eligible child's person row is geographic clone 0's or not, so
        # the receipt's row counts are one clone's (microcosm#931).
        child_clone_zero=geographic_clone_zero_rows(frame, "person")[eligible_child],
    )
    opt_out, opt_out_receipt = assign_child_benefit_opt_outs(
        claims=claims,
        reporter=reporter,
        highest_income=highest_income,
        children=children,
        weights=weights,
        draws=opt_out_draws,
        statistics=statistics,
        thresholds=thresholds,
    )
    paid = claims & ~opt_out
    previous_claim = benunit[CHILD_BENEFIT_CLAIM_OUTPUT].fillna(False).to_numpy(bool)
    previous_opt_out = (
        benunit[CHILD_BENEFIT_OPT_OUT_OUTPUT].fillna(False).to_numpy(bool)
    )
    exported_claims = claims if thresholds.supports_opt_out else paid
    would_claim = np.where(family, exported_claims, previous_claim)
    benunit[CHILD_BENEFIT_CLAIM_OUTPUT] = would_claim
    benunit[CHILD_BENEFIT_OPT_OUT_OUTPUT] = opt_out
    result = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=period,
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    paid_families = float(weights[family & paid].sum())
    paid_children = float((weights * children)[family & paid].sum())
    charged = family & paid & (highest_income > thresholds.phase_out_start)
    outside = reporter & ~family
    return UKChildBenefitTakeUpResult(
        frame=result,
        claim_export={
            "encoding": (
                "registered_claims" if thresholds.supports_opt_out else "legacy_payment"
            ),
            "supports_opt_out": thresholds.supports_opt_out,
            "engine_parameters_source": thresholds.source,
        },
        copy_reports=copy_receipt,
        claims=claim_receipt,
        opt_outs=opt_out_receipt,
        in_payment={
            "weights": "household weights at the stage, before calibration",
            "basis": "claims and not opted out in the draw; not a model cash rerun",
            "families": paid_families,
            "children": paid_children,
            "published_families": statistics.families_in_payment,
            "published_children": statistics.children_in_payment,
            "published_period": statistics.caseload_period,
            "families_with_income_above_charge_start": float(weights[charged].sum()),
        },
        reporters_without_eligible_child={
            "units": int(outside.sum()),
            "weighted_units": float(weights[outside].sum()),
        },
        changed_units={
            CHILD_BENEFIT_CLAIM_OUTPUT: int((previous_claim != would_claim).sum()),
            CHILD_BENEFIT_OPT_OUT_OUTPUT: int((previous_opt_out != opt_out).sum()),
            # Person rows, all on SPI copies.
            CHILD_BENEFIT_REPORT_OUTPUT: int((landed != reported).sum()),
        },
    )


def redraw_spi_child_benefit_reports(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    person_position: np.ndarray,
    eligible_child: np.ndarray,
    family: np.ndarray,
    children: np.ndarray,
    highest_income: np.ndarray,
    reported: np.ndarray,
    entitlement: np.ndarray,
    weights: np.ndarray,
    thresholds: UKChildBenefitChargeThresholds,
    minimum_source_households: int = CHILD_BENEFIT_COPY_MINIMUM_SOURCE_HOUSEHOLDS,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Redraw the SPI copies' reports against their FRS twins and cells.

    Returns the person-aligned reported amounts, the person rows the stage may
    write (the SPI copies' members) and the receipt. Only the copies' rows
    change; the caller checks the FRS rows against the frame after writing.
    """

    unit_count = len(benunit)
    channel = benunit[support_channel_column("benunit")].astype(str).to_numpy()
    base = channel == BASE_FRS_SUPPORT_CHANNEL
    copy = channel == SPI_SYNTHETIC_SUPPORT_CHANNEL
    if np.any(~(base | copy)):
        raise ValueError(
            "Child Benefit take-up expects only the FRS and SPI support channels."
        )
    reporter = _units_with_positive(reported, person_position, unit_count)
    twin = _benunit_twins(benunit, base=base, copy=copy)
    copy_rows = copy[person_position]
    twin_rows = _person_twins(
        person, copy_rows=copy_rows, person_position=person_position, twin=twin
    )
    source_households = _benunit_source_households(
        person, household, person_position=person_position, unit_count=unit_count
    )
    cell, cells = child_benefit_copy_report_cells(
        family=family,
        children=children,
        highest_income=highest_income,
        base=base,
        source_households=source_households,
        thresholds=thresholds,
        minimum_source_households=minimum_source_households,
    )
    rates = _cell_report_rates(
        cell, len(cells), members=base & family, reporter=reporter, weights=weights
    )
    if np.any(copy & family) and not np.any(base & family):
        raise ValueError(
            "Child Benefit take-up has SPI copy families but no FRS family to "
            "take report rates from."
        )
    draws = stable_identity_uniforms(
        benunit["benunit_id"].to_numpy(),
        seed=CHILD_BENEFIT_TAKE_UP_SEED,
        salt=CHILD_BENEFIT_REPORT_OUTPUT,
    )
    drawn = assign_spi_child_benefit_reports(
        copy=copy,
        family=family,
        twin=twin,
        cell=cell,
        rates=rates,
        reporter=reporter,
        draws=draws,
    )
    twin_reporter = np.zeros(unit_count, dtype=bool)
    twin_reporter[copy] = reporter[twin[copy]]
    takes_twin = copy & (~family | (drawn & twin_reporter))
    new_report = copy & family & drawn & ~twin_reporter
    unpayable = new_report & ~(entitlement > 0.0)
    landing = _landing_rows(
        person,
        person_position=person_position,
        eligible_child=eligible_child,
        unit_count=unit_count,
    )
    landed = reported.copy()
    rows = np.flatnonzero(copy_rows)
    unit = person_position[rows]
    values = np.where(takes_twin[unit], reported[twin_rows[rows]], 0.0)
    lands = (new_report & ~unpayable)[unit] & landing[rows]
    landed[rows] = np.where(lands, entitlement[unit], values)

    # Checked from the landed amounts, against inputs they were not derived
    # from: every copy family reporting exactly when its draw says so (bar an
    # entitlement of nothing), every other copy carrying its twin's amounts
    # row for row. The caller checks the FRS rows against the frame.
    reporter_after = _units_with_positive(landed, person_position, unit_count)
    copy_family = copy & family
    if np.any(reporter_after[copy_family] != (drawn & ~unpayable)[copy_family]):
        raise RuntimeError(
            "Child Benefit take-up landed copy reports that disagree with the draw."
        )
    childless_rows = copy_rows & ~family[person_position]
    if not np.array_equal(landed[childless_rows], reported[twin_rows[childless_rows]]):
        raise RuntimeError(
            "Child Benefit take-up must give a copy without an eligible child its "
            "twin's reports."
        )
    receipt = _copy_report_receipt(
        cells=cells,
        cell=cell,
        rates=rates,
        base=base,
        copy=copy,
        family=family,
        reporter_before=reporter,
        reporter_after=reporter_after,
        twin_reporter=twin_reporter,
        new_report=new_report,
        unpayable=unpayable,
        weights=weights,
        source_households=source_households,
        thresholds=thresholds,
        minimum_source_households=minimum_source_households,
    )
    return landed, copy_rows, receipt


def child_benefit_copy_report_cells(
    *,
    family: np.ndarray,
    children: np.ndarray,
    highest_income: np.ndarray,
    base: np.ndarray,
    source_households: np.ndarray,
    thresholds: UKChildBenefitChargeThresholds,
    minimum_source_households: int = CHILD_BENEFIT_COPY_MINIMUM_SOURCE_HOUSEHOLDS,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Each family's report cell (-1 for a unit without an eligible child).

    A cell is a set of children groups (1, 2, 3+) by a run of adjacent charge
    bands. Children groups whose FRS families rest on fewer than
    ``minimum_source_households`` distinct source households are merged with
    a neighbour, the smallest first; within a group, bands are taken in order
    and closed into a cell once its FRS families reach the minimum, a short
    remainder joining the last cell. Support counts FRS families only; the
    cell covers both channels.
    """

    family = np.asarray(family, dtype=bool)
    base = np.asarray(base, dtype=bool)
    if int(minimum_source_households) < 1:
        raise ValueError("minimum_source_households must be at least 1.")
    group = np.clip(np.asarray(children, dtype=np.int64), 1, 3) - 1
    band = np.select(
        [
            np.asarray(highest_income) <= thresholds.phase_out_start,
            np.asarray(highest_income) < thresholds.phase_out_end,
        ],
        [0, 1],
        2,
    )
    support_rows = base & family
    sources = np.asarray(source_households)

    def support(mask: np.ndarray) -> int:
        return int(np.unique(sources[support_rows & mask]).size)

    groups = [[0], [1], [2]]
    while len(groups) > 1:
        supports = [support(np.isin(group, members)) for members in groups]
        short = [
            i for i, value in enumerate(supports) if value < minimum_source_households
        ]
        if not short:
            break
        index = min(short, key=lambda i: (supports[i], i))
        neighbour = index - 1 if index > 0 else 1
        low, high = sorted((index, neighbour))
        groups[low : high + 1] = [groups[low] + groups[high]]
    cell = np.full(len(family), -1, dtype=np.int64)
    cells: list[dict[str, object]] = []
    for members in groups:
        in_group = family & np.isin(group, members)
        runs: list[list[int]] = []
        current: list[int] = []
        for value in range(len(CHILD_BENEFIT_COPY_INCOME_BANDS)):
            current.append(value)
            if support(in_group & np.isin(band, current)) >= minimum_source_households:
                runs.append(current)
                current = []
        if current:
            if runs:
                runs[-1] = runs[-1] + current
            else:
                runs.append(current)
        for run in runs:
            cell[in_group & np.isin(band, run)] = len(cells)
            cells.append(
                {
                    "children": [CHILD_BENEFIT_COPY_CHILDREN[i] for i in members],
                    "income_bands": [CHILD_BENEFIT_COPY_INCOME_BANDS[i] for i in run],
                    "frs_source_households": support(in_group & np.isin(band, run)),
                }
            )
    return cell, cells


def assign_spi_child_benefit_reports(
    *,
    copy: np.ndarray,
    family: np.ndarray,
    twin: np.ndarray,
    cell: np.ndarray,
    rates: np.ndarray,
    reporter: np.ndarray,
    draws: np.ndarray,
) -> np.ndarray:
    """Whether each copy family reports, coupled to its twin's report.

    A copy reports when ``u * p_twin < p_copy`` if its twin reports and when
    ``p_twin + u * (1 - p_twin) < p_copy`` if it does not: a copy in its
    twin's cell keeps the twin's report, and a twin reporting at its cell's
    rate gives its copies their own cell's rate. A twin in no cell (it has no
    eligible child) has no rate to couple to, so its copy reports when
    ``u < p_copy``, at its own cell's rate whatever the twin reports. False
    outside the copy families.
    """

    copy = np.asarray(copy, dtype=bool)
    family = np.asarray(family, dtype=bool)
    twin = np.asarray(twin, dtype=np.int64)
    cell = np.asarray(cell, dtype=np.int64)
    rates = np.asarray(rates, dtype=np.float64)
    reporter = np.asarray(reporter, dtype=bool)
    draws = np.asarray(draws, dtype=np.float64)
    target = copy & family
    if np.any(twin[target] < 0):
        raise ValueError("Every SPI copy family needs an FRS twin.")
    if np.any(cell[target] < 0):
        raise ValueError("Every SPI copy family needs a report cell.")
    p_copy = rates[cell[target]]
    twin_cell = cell[twin[target]]
    p_twin = rates[np.maximum(twin_cell, 0)]
    u = draws[target]
    coupled = np.where(reporter[twin[target]], u * p_twin, p_twin + u * (1.0 - p_twin))
    # Standing in p_copy for a twin in no cell would make the copy report
    # always (twin reports) or never (it does not): draw such a copy uncoupled.
    drawn = np.zeros(len(copy), dtype=bool)
    drawn[target] = np.where(twin_cell >= 0, coupled, u) < p_copy
    return drawn


def _units_with_positive(
    values: np.ndarray, person_position: np.ndarray, unit_count: int
) -> np.ndarray:
    return (
        np.bincount(
            person_position,
            weights=(np.asarray(values) > 0.0).astype(np.float64),
            minlength=unit_count,
        )
        > 0
    )


def _benunit_twins(
    benunit: pd.DataFrame, *, base: np.ndarray, copy: np.ndarray
) -> np.ndarray:
    """Each SPI copy's FRS twin as a benefit-unit position, -1 elsewhere."""

    ids = benunit["benunit_id"].to_numpy()
    if pd.Index(ids).has_duplicates:
        raise ValueError("Child Benefit take-up needs unique benefit unit ids.")
    position = pd.Series(np.arange(len(benunit)), index=ids)
    source = benunit[support_source_id_column("benunit")].to_numpy()
    mapped = position.reindex(source[copy]).to_numpy(dtype=np.float64)
    if np.isnan(mapped).any():
        raise ValueError(
            "An SPI copy's benunit_source_id names no benefit unit in the frame."
        )
    twin = np.full(len(benunit), -1, dtype=np.int64)
    twin[copy] = mapped.astype(np.int64)
    if not base[twin[copy]].all():
        raise ValueError("An SPI copy's twin must be an FRS benefit unit.")
    return twin


def _person_twins(
    person: pd.DataFrame,
    *,
    copy_rows: np.ndarray,
    person_position: np.ndarray,
    twin: np.ndarray,
) -> np.ndarray:
    """Each copy member's twin person as a row position, -1 elsewhere."""

    ids = person["person_id"].to_numpy()
    if pd.Index(ids).has_duplicates:
        raise ValueError("Child Benefit take-up needs unique person ids.")
    position = pd.Series(np.arange(len(person)), index=ids)
    source = person[support_source_id_column("person")].to_numpy()
    mapped = position.reindex(source[copy_rows]).to_numpy(dtype=np.float64)
    if np.isnan(mapped).any():
        raise ValueError(
            "An SPI copy member's person_source_id names no person in the frame."
        )
    rows = np.full(len(person), -1, dtype=np.int64)
    rows[copy_rows] = mapped.astype(np.int64)
    if np.any(person_position[rows[copy_rows]] != twin[person_position[copy_rows]]):
        raise ValueError(
            "An SPI copy member's twin must belong to the copy's twin benefit unit."
        )
    return rows


def _benunit_source_households(
    person: pd.DataFrame,
    household: pd.DataFrame,
    *,
    person_position: np.ndarray,
    unit_count: int,
) -> np.ndarray:
    """The survey household each benefit unit comes from, through its members."""

    first = np.full(unit_count, len(person), dtype=np.int64)
    np.minimum.at(first, person_position, np.arange(len(person), dtype=np.int64))
    if np.any(first == len(person)):
        raise ValueError("Every benefit unit needs at least one person row.")
    sources = household.set_index("household_id")[CHILD_BENEFIT_COPY_SOURCE_HOUSEHOLD]
    mapped = person["person_household_id"].to_numpy()[first]
    values = sources.reindex(mapped)
    if values.isna().any():
        raise ValueError(
            f"A benefit unit's household carries no {CHILD_BENEFIT_COPY_SOURCE_HOUSEHOLD}."
        )
    return values.to_numpy()


def _cell_report_rates(
    cell: np.ndarray,
    count: int,
    *,
    members: np.ndarray,
    reporter: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """The weighted reporting share of each cell's FRS families."""

    rates = np.zeros(count, dtype=np.float64)
    for index in range(count):
        in_cell = members & (cell == index)
        mass = float(weights[in_cell].sum())
        if mass > 0.0:
            rates[index] = float(weights[in_cell & reporter].sum()) / mass
    return rates


def _landing_rows(
    person: pd.DataFrame,
    *,
    person_position: np.ndarray,
    eligible_child: np.ndarray,
    unit_count: int,
) -> np.ndarray:
    """One row per benefit unit for a new report: its head, else an adult."""

    head = _boolean_values(person[CHILD_BENEFIT_COPY_LANDING])
    preference = np.where(head & ~eligible_child, 0, np.where(eligible_child, 2, 1))
    order = np.lexsort((np.arange(len(person)), preference, person_position))
    leading = np.ones(len(order), dtype=bool)
    leading[1:] = person_position[order][1:] != person_position[order][:-1]
    landing = np.zeros(len(person), dtype=bool)
    landing[order[leading]] = True
    return landing


def _floored(value: int) -> int | None:
    """A count, withheld when it rests on one or two cases."""

    return None if 0 < value < CHILD_BENEFIT_RECEIPT_FLOOR else int(value)


def _copy_report_receipt(
    *,
    cells: list[dict[str, object]],
    cell: np.ndarray,
    rates: np.ndarray,
    base: np.ndarray,
    copy: np.ndarray,
    family: np.ndarray,
    reporter_before: np.ndarray,
    reporter_after: np.ndarray,
    twin_reporter: np.ndarray,
    new_report: np.ndarray,
    unpayable: np.ndarray,
    weights: np.ndarray,
    source_households: np.ndarray,
    thresholds: UKChildBenefitChargeThresholds,
    minimum_source_households: int,
) -> dict[str, object]:
    """The copies' report audit; figures on one or two households are withheld."""

    def share(mask: np.ndarray, numerator: np.ndarray) -> float | None:
        mass = float(weights[mask].sum())
        return float(weights[mask & numerator].sum()) / mass if mass > 0.0 else None

    rows = []
    for index, definition in enumerate(cells):
        frs = base & family & (cell == index)
        copies = copy & family & (cell == index)
        frs_support = int(np.unique(source_households[frs]).size)
        copy_support = int(np.unique(source_households[copies]).size)
        frs_shown = frs_support >= CHILD_BENEFIT_RECEIPT_FLOOR
        copy_shown = copy_support >= CHILD_BENEFIT_RECEIPT_FLOOR
        rows.append(
            {
                **definition,
                "frs_source_households": frs_support if frs_shown else None,
                "frs_rate": float(rates[index]) if frs_shown else None,
                "copy_family_units": _floored(int(copies.sum())),
                "copy_source_households": _floored(copy_support),
                "copy_rate_before": (
                    share(copies, reporter_before) if copy_shown else None
                ),
                "copy_rate_after": share(copies, reporter_after)
                if copy_shown
                else None,
            }
        )
    copy_family = copy & family
    copy_mass = float(weights[copy_family].sum())
    expected = (
        float(
            (weights * np.where(cell >= 0, rates[np.maximum(cell, 0)], 0.0))[
                copy_family
            ].sum()
        )
        / copy_mass
        if copy_mass > 0.0
        else None
    )
    childless = copy & ~family
    return {
        "channel": SPI_SYNTHETIC_SUPPORT_CHANNEL,
        "twin": support_source_id_column("benunit"),
        "source_household": CHILD_BENEFIT_COPY_SOURCE_HOUSEHOLD,
        "minimum_source_households": int(minimum_source_households),
        "phase_out_start": thresholds.phase_out_start,
        "phase_out_end": thresholds.phase_out_end,
        "weights": "household weights at the stage, before calibration",
        "cells": rows,
        "copy_family_units": _floored(int(copy_family.sum())),
        "copy_families": copy_mass,
        "copy_rate_before": share(copy_family, reporter_before),
        "copy_rate_expected": expected,
        "copy_rate_after": share(copy_family, reporter_after),
        "transitions": {
            "promoted": _floored(
                int((copy_family & ~reporter_before & reporter_after).sum())
            ),
            "demoted": _floored(
                int((copy_family & reporter_before & ~reporter_after).sum())
            ),
            "held_reporter": _floored(
                int((copy_family & reporter_before & reporter_after).sum())
            ),
            "held_nonreporter": _floored(
                int((copy_family & ~reporter_before & ~reporter_after).sum())
            ),
        },
        "landing": {
            "twin_amounts": _floored(
                int((copy_family & reporter_after & twin_reporter).sum())
            ),
            "entitlement_amounts": _floored(int((new_report & ~unpayable).sum())),
            "no_entitlement": _floored(int(unpayable.sum())),
        },
        "copies_without_eligible_child": {
            "units": _floored(int(childless.sum())),
            "reporting_before": _floored(int((childless & reporter_before).sum())),
            "reporting_after": _floored(int((childless & reporter_after).sum())),
        },
        "frs_rows_unchanged": True,
    }


def solve_child_benefit_claim_rates(
    *,
    child_ages: np.ndarray,
    child_key_ages: np.ndarray,
    child_reporter: np.ndarray,
    child_weights: np.ndarray,
    claim_rates: Mapping[int, float],
    child_clone_zero: np.ndarray | None = None,
) -> tuple[dict[int, float], list[dict[str, object]]]:
    """Non-reporter claim rate by key age, and one receipt row per age.

    ``child_*`` describe the eligible children: own age, the family's key age
    (see ``child_benefit_family_key_ages``), whether the family reports, and
    the weight. Ages under 16 are solved from 15 down first, then 19 down to
    16, so each age's other keys are solved before it. ``child_clone_zero``
    flags the children whose rows are geographic clone 0's (where it is not
    given, every child's, as on a one-clone frame): each row's
    ``eligible_child_rows`` counts the age's children among them, so the
    count does not grow with the clone count (microcosm#931). The rates and
    the weighted figures are taken over every child.
    """

    child_ages = np.asarray(child_ages, dtype=np.int64)
    child_key_ages = np.asarray(child_key_ages, dtype=np.int64)
    child_reporter = np.asarray(child_reporter, dtype=bool)
    child_weights = np.asarray(child_weights, dtype=np.float64)
    counted = _clone_zero_children(child_clone_zero, len(child_ages))
    unknown = sorted(set(child_ages.tolist()) - set(claim_rates))
    if unknown:
        raise ValueError(
            "HMRC publishes no Child Benefit claim rate for eligible children "
            f"aged {unknown}."
        )
    # A key below a child's own age is the family's younger key child, which
    # only a young person of 16 or over may have.
    young_person_under_child_key = (child_ages >= CHILD_BENEFIT_KEY_AGE_LIMIT) & (
        child_key_ages < CHILD_BENEFIT_KEY_AGE_LIMIT
    )
    if ((child_key_ages < child_ages) & ~young_person_under_child_key).any():
        raise ValueError(
            "A family's key age is below one of its children's ages, and that "
            "child is not a young person in a family keyed under 16."
        )
    residual: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    for age in _claim_rate_solve_order(claim_rates):
        of_age = child_ages == age
        eligible = float(child_weights[of_age].sum())
        reporting = float(child_weights[of_age & child_reporter].sum())
        own = of_age & ~child_reporter & (child_key_ages == age)
        own_mass = float(child_weights[own].sum())
        # Children of this age in non-reporting families keyed at another age
        # claim at that key's rate, which the solve order has already set.
        others = of_age & ~child_reporter & (child_key_ages != age)
        unsolved = others & ~np.isin(child_key_ages, list(residual))
        if unsolved.any():
            raise ValueError(
                f"Children aged {age} sit in families keyed at ages solved "
                "after theirs."
            )
        older_claimed = float(
            sum(
                child_weights[others & (child_key_ages == key)].sum() * rate
                for key, rate in residual.items()
            )
        )
        raw: float | None = None
        rate = 0.0
        if own_mass > 0.0:
            raw = (claim_rates[age] * eligible - reporting - older_claimed) / own_mass
            rate = float(np.clip(raw, 0.0, 1.0))
        residual[age] = rate
        rows.append(
            {
                "age": int(age),
                "published_rate": float(claim_rates[age]),
                "eligible_child_rows": int((of_age & counted).sum()),
                "eligible_children": eligible,
                "reporter_children": reporting,
                "residual_rate_unclipped": raw,
                "residual_rate": rate,
                "clipped": bool(raw is not None and not 0.0 <= raw <= 1.0),
            }
        )
    rows.sort(key=lambda row: row["age"])
    return residual, rows


def _clone_zero_children(child_clone_zero: np.ndarray | None, count: int) -> np.ndarray:
    """One flag per eligible child: ``child_clone_zero``, or every child where
    it is not given. A misaligned array is refused in a fixed sentence that
    names no count."""

    if child_clone_zero is None:
        return np.ones(count, dtype=bool)
    flags = np.asarray(child_clone_zero, dtype=bool)
    if flags.shape != (count,):
        raise ValueError("child_clone_zero must hold one flag per eligible child.")
    return flags


def _claim_rate_solve_order(claim_rates: Mapping[int, float]) -> list[int]:
    """Ages under 16 from the eldest down, then ages 16 and over likewise."""

    ages = sorted(claim_rates, reverse=True)
    return [age for age in ages if age < CHILD_BENEFIT_KEY_AGE_LIMIT] + [
        age for age in ages if age >= CHILD_BENEFIT_KEY_AGE_LIMIT
    ]


def child_benefit_family_key_ages(
    child_units: np.ndarray, child_ages: np.ndarray, unit_count: int
) -> np.ndarray:
    """Each unit's key age: its eldest eligible child under 16, else its eldest.

    ``child_units`` holds each eligible child's benefit-unit position and
    ``child_ages`` its whole age. A unit without an eligible child gets -1.
    """

    child_units = np.asarray(child_units, dtype=np.int64)
    child_ages = np.asarray(child_ages, dtype=np.int64)
    eldest = np.full(unit_count, -1, dtype=np.int64)
    np.maximum.at(eldest, child_units, child_ages)
    under = child_ages < CHILD_BENEFIT_KEY_AGE_LIMIT
    eldest_under = np.full(unit_count, -1, dtype=np.int64)
    np.maximum.at(eldest_under, child_units[under], child_ages[under])
    return np.where(eldest_under >= 0, eldest_under, eldest)


def assign_child_benefit_claims(
    *,
    child_units: np.ndarray,
    child_ages: np.ndarray,
    eldest: np.ndarray,
    reporter: np.ndarray,
    weights: np.ndarray,
    draws: np.ndarray,
    statistics: UKChildBenefitStatistics,
    child_clone_zero: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Family claims: reporters, and the rest at their key age's solved rate.

    ``child_units`` holds each eligible child's benefit-unit position and
    ``child_ages`` its whole age; ``eldest`` is each unit's eldest eligible
    child's age, or -1 for a unit without one, and marks the families. The key
    each family claims at is ``child_benefit_family_key_ages``.
    ``child_clone_zero`` flags the children whose rows are geographic clone
    0's, among whom the receipt's ``eligible_child_rows`` are counted, overall
    and by age (where it is not given, every child, as on a one-clone frame;
    microcosm#931).
    """

    child_units = np.asarray(child_units, dtype=np.int64)
    child_ages = np.asarray(child_ages, dtype=np.int64)
    eldest = np.asarray(eldest, dtype=np.int64)
    reporter = np.asarray(reporter, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    draws = np.asarray(draws, dtype=np.float64)
    counted = _clone_zero_children(child_clone_zero, len(child_units))
    family = eldest >= 0
    key = child_benefit_family_key_ages(child_units, child_ages, len(eldest))
    if not np.array_equal(key >= 0, family):
        raise ValueError(
            "Child Benefit families must be the units with an eligible child."
        )
    child_weights = weights[child_units]
    residual, rows = solve_child_benefit_claim_rates(
        child_ages=child_ages,
        child_key_ages=key[child_units],
        child_reporter=reporter[child_units],
        child_weights=child_weights,
        claim_rates=statistics.claim_rates,
        child_clone_zero=counted,
    )
    unit_rate = np.zeros(len(eldest), dtype=np.float64)
    unit_rate[family] = np.asarray([residual[int(age)] for age in key[family]])
    claims = family & (reporter | (draws < unit_rate))
    child_claims = claims[child_units]
    for row in rows:
        of_age = child_ages == row["age"]
        eligible = float(row["eligible_children"])
        row["realized_rate"] = (
            float(child_weights[of_age & child_claims].sum()) / eligible
            if eligible > 0.0
            else None
        )
    total = float(child_weights.sum())
    target = (
        float(
            sum(
                statistics.claim_rates[int(row["age"])] * row["eligible_children"]
                for row in rows
            )
        )
        / total
        if total > 0.0
        else None
    )
    receipt = {
        "published_period": statistics.claim_rate_period,
        "published_all_ages_rate": statistics.all_ages_claim_rate,
        # The published rates by age at the frame's own age mix: what the
        # solved rates aim at overall.
        "target_rate": target,
        "realized_rate": (
            float(child_weights[child_claims].sum()) / total if total > 0.0 else None
        ),
        "eligible_children": total,
        "eligible_child_rows": int(counted.sum()),
        "eligible_families": float(weights[family].sum()),
        "eligible_family_units": int(family.sum()),
        "reporter_families": float(weights[family & reporter].sum()),
        "claiming_families": float(weights[claims].sum()),
        "clipped_ages": [int(row["age"]) for row in rows if row["clipped"]],
        "ages": rows,
    }
    return claims, receipt


def assign_child_benefit_opt_outs(
    *,
    claims: np.ndarray,
    reporter: np.ndarray,
    highest_income: np.ndarray,
    children: np.ndarray,
    weights: np.ndarray,
    draws: np.ndarray,
    statistics: UKChildBenefitStatistics,
    thresholds: UKChildBenefitChargeThresholds,
) -> tuple[np.ndarray, dict[str, object]]:
    """Draw opt-outs from unpaid candidates, retaining the legacy taper fallback."""

    claims = np.asarray(claims, dtype=bool)
    reporter = np.asarray(reporter, dtype=bool)
    highest_income = np.asarray(highest_income, dtype=np.float64)
    children = np.asarray(children, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    draws = np.asarray(draws, dtype=np.float64)
    candidates = claims & ~reporter
    if thresholds.supports_opt_out:
        # The supported build boundary has positive taper width. Match the
        # installed model's payment-suppression predicate, including share=0:
        # a family at the charge start must still receive payment.
        charge_fraction = np.clip(
            np.maximum(highest_income - thresholds.phase_out_start, 0.0)
            / (thresholds.phase_out_end - thresholds.phase_out_start),
            0.0,
            1.0,
        )
        candidates &= (
            (children > 0)
            & (charge_fraction > 0)
            & (charge_fraction >= thresholds.opt_out_charge_share)
        )
    pools = (
        (
            "fully_charged",
            candidates & (highest_income >= thresholds.phase_out_end),
        ),
        (
            "taper",
            candidates
            & (highest_income > thresholds.phase_out_start)
            & (highest_income < thresholds.phase_out_end),
        ),
    )
    claiming = float(weights[claims].sum())
    target = statistics.opt_out_share * claiming
    remaining = target
    opt_out = np.zeros(len(claims), dtype=bool)
    pool_rows = []
    for name, pool in pools:
        mass = float(weights[pool].sum())
        rate = float(np.clip(remaining / mass, 0.0, 1.0)) if mass > 0.0 else 0.0
        drawn = pool & (draws < rate)
        opt_out |= drawn
        pool_rows.append(
            {
                "pool": name,
                "units": int(pool.sum()),
                "weighted_families": mass,
                "rate": rate,
                "opted_out_units": int(drawn.sum()),
                "opted_out_families": float(weights[drawn].sum()),
            }
        )
        remaining = max(0.0, remaining - mass)
    realized = float(weights[opt_out].sum())
    opted_children = float((weights * children)[opt_out].sum())
    receipt = {
        "published_period": statistics.caseload_period,
        "published_families_registered": statistics.families_registered,
        "published_families_opted_out": statistics.families_opted_out,
        "target_share": statistics.opt_out_share,
        "claiming_families": claiming,
        "target_families": target,
        "opted_out_families": realized,
        "realized_share": realized / claiming if claiming > 0.0 else None,
        # Target mass no charged family was left to carry.
        "pool_shortfall_families": remaining,
        "pool_exhausted": bool(remaining > 0.0),
        "opted_out_children": opted_children,
        "children_per_opted_out_family": (
            opted_children / realized if realized > 0.0 else None
        ),
        "published_children_per_opted_out_family": (
            statistics.children_opted_out / statistics.families_opted_out
        ),
        "charge_income": CHILD_BENEFIT_CHARGE_INCOME_VARIABLE,
        "phase_out_start": thresholds.phase_out_start,
        "phase_out_end": thresholds.phase_out_end,
        "opt_out_charge_share": (
            float(thresholds.opt_out_charge_share)
            if thresholds.supports_opt_out
            else None
        ),
        "thresholds_source": thresholds.source,
        "reporters_above_charge_start": float(
            weights[
                claims & reporter & (highest_income > thresholds.phase_out_start)
            ].sum()
        ),
        "pools": pool_rows,
    }
    return opt_out, receipt


def _aligned(
    values: object, expected: int, label: str, *, table: str = "person"
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (expected,):
        raise ValueError(f"{label} must align to the {table} table.")
    if array.dtype != bool and not np.isfinite(array.astype(np.float64)).all():
        raise ValueError(f"{label} must be finite.")
    return array


def child_benefit_take_up_operation_parameters() -> dict[str, dict[str, object]]:
    """The reviewed operation payload the manifests must carry verbatim."""

    return {
        "materialize_rules_engine_predictors": {
            "predictors": list(CHILD_BENEFIT_ENGINE_VARIABLES),
            "consumed_only": True,
        },
        "redraw_spi_reported_child_benefit": {
            "output": CHILD_BENEFIT_REPORT_OUTPUT,
            "draw": CHILD_BENEFIT_REPORT_OUTPUT,
            "seed": CHILD_BENEFIT_TAKE_UP_SEED,
            "rows": f"{support_channel_column('benunit')} == {SPI_SYNTHETIC_SUPPORT_CHANNEL}",
            "twin": support_source_id_column("benunit"),
            "person_twin": support_source_id_column("person"),
            "eligibility": CHILD_BENEFIT_ELIGIBILITY_VARIABLE,
            "charge_income": CHILD_BENEFIT_CHARGE_INCOME_VARIABLE,
            "charge_parameters": CHILD_BENEFIT_CHARGE_PARAMETERS,
            "children": list(CHILD_BENEFIT_COPY_CHILDREN),
            "income_bands": list(CHILD_BENEFIT_COPY_INCOME_BANDS),
            "source_household": CHILD_BENEFIT_COPY_SOURCE_HOUSEHOLD,
            "minimum_source_households": CHILD_BENEFIT_COPY_MINIMUM_SOURCE_HOUSEHOLDS,
            "amount": CHILD_BENEFIT_ENTITLEMENT_VARIABLE,
            "landing": CHILD_BENEFIT_COPY_LANDING,
            "weight_mapping": "household_to_benunit",
            "rule": CHILD_BENEFIT_COPY_REPORT_RULE,
        },
        "aggregate_person_to_benunit": {
            "method": "any_positive",
            "consumed_only": True,
            "aggregates": dict(CHILD_BENEFIT_AGGREGATES),
        },
        "assign_family_claims_by_child_age": {
            "output": CHILD_BENEFIT_CLAIM_OUTPUT,
            "draw": CHILD_BENEFIT_CLAIM_OUTPUT,
            "seed": CHILD_BENEFIT_TAKE_UP_SEED,
            "anchor": CHILD_BENEFIT_REPORTED_ANCHOR,
            "eligibility": CHILD_BENEFIT_ELIGIBILITY_VARIABLE,
            "family_key": CHILD_BENEFIT_FAMILY_KEY,
            "resource": CHILD_BENEFIT_STATISTICS_RESOURCE,
            "rate_measure": CHILD_BENEFIT_CLAIM_RATE_MEASURE,
            "rate_period": CHILD_BENEFIT_CLAIM_RATE_PERIOD,
            "weight_mapping": "household_to_benunit",
            "rule": CHILD_BENEFIT_CLAIM_RULE,
        },
        "assign_opt_out_by_charge_income": {
            "output": CHILD_BENEFIT_OPT_OUT_OUTPUT,
            "draw": CHILD_BENEFIT_OPT_OUT_OUTPUT,
            "seed": CHILD_BENEFIT_TAKE_UP_SEED,
            "resource": CHILD_BENEFIT_STATISTICS_RESOURCE,
            "numerator_measure": CHILD_BENEFIT_FAMILIES_OPTED_OUT_MEASURE,
            "denominator_measure": CHILD_BENEFIT_FAMILIES_REGISTERED_MEASURE,
            "caseload_period": CHILD_BENEFIT_CASELOAD_PERIOD,
            "charge_income": CHILD_BENEFIT_CHARGE_INCOME_VARIABLE,
            "charge_parameters": CHILD_BENEFIT_CHARGE_PARAMETERS,
            "pools": ["fully_charged", "taper"],
            "weight_mapping": "household_to_benunit",
            "rule": CHILD_BENEFIT_OPT_OUT_RULE,
            "payment_rule": CHILD_BENEFIT_PAYMENT_RULE,
        },
    }


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    if stage.stage != CHILD_BENEFIT_TAKE_UP_STAGE_NAME:
        raise ValueError(
            f"Child Benefit take-up received stage {stage.stage!r}, expected "
            f"{CHILD_BENEFIT_TAKE_UP_STAGE_NAME!r}."
        )
    expected = child_benefit_take_up_operation_parameters()
    kinds = [operation.kind for operation in stage.operations]
    if kinds != list(expected):
        raise ValueError(
            f"{CHILD_BENEFIT_TAKE_UP_STAGE_NAME} operation order drifted: "
            f"expected {list(expected)}, got {kinds}."
        )
    actual = {
        operation.kind: dict(operation.parameters) for operation in stage.operations
    }
    if actual != expected:
        drifted = sorted(kind for kind in expected if actual[kind] != expected[kind])
        raise ValueError(
            f"{CHILD_BENEFIT_TAKE_UP_STAGE_NAME} parameters drifted from the "
            f"reviewed mapping on operation(s) {drifted}."
        )
    rewrites = (
        CHILD_BENEFIT_CLAIM_OUTPUT,
        CHILD_BENEFIT_OPT_OUT_OUTPUT,
        CHILD_BENEFIT_REPORT_OUTPUT,
    )
    if stage.outputs != () or stage.rewrites != rewrites:
        raise ValueError(
            f"{CHILD_BENEFIT_TAKE_UP_STAGE_NAME} must declare no new outputs and "
            f"exactly the rewrites {list(rewrites)}."
        )


__all__ = [
    "CHILD_BENEFIT_ENGINE_VARIABLES",
    "CHILD_BENEFIT_STATISTICS_RESOURCE",
    "CHILD_BENEFIT_TAKE_UP_STAGE_NAME",
    "UKChildBenefitChargeThresholds",
    "UKChildBenefitStatistics",
    "UKChildBenefitTakeUpResult",
    "UKChildBenefitTakeUpStageTransform",
    "assign_child_benefit_claims",
    "assign_child_benefit_opt_outs",
    "assign_spi_child_benefit_reports",
    "child_benefit_copy_report_cells",
    "child_benefit_family_key_ages",
    "child_benefit_take_up_operation_parameters",
    "load_child_benefit_statistics",
    "redraw_child_benefit_take_up",
    "redraw_spi_child_benefit_reports",
    "solve_child_benefit_claim_rates",
    "uk_child_benefit_charge_thresholds",
]
