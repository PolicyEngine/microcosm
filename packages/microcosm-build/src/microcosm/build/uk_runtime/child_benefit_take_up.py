"""Child Benefit claims and opt-outs on post-SPI incomes (microcosm#1063).

``frs_take_up`` draws ``would_claim_child_benefit`` early, at one flat rate
over every benefit unit with the FRS reporters added on top, and
``child_benefit_opts_out`` at a flat rate over every benefit unit whatever its
income; the engine pays on the first and never reads the second. This stage
redraws both once the SPI income chain has set the incomes the charge is
assessed on, against HMRC's Child Benefit statistics.

Claims. One temporary engine materialization gives each person's eligibility
(a child, or a qualifying young person). HMRC publishes the share of eligible
children for whom Child Benefit is claimed, opted-out families included, by
single year of age. A family claims for all its children or for none, so the
draw is by family, keyed on the age of its eldest eligible child: a family
that reports the benefit claims, and the other families of key age ``g``
claim at a rate ``r_g``. The rates are solved from the eldest age down so that
the claimed share of the children of each single year of age is the published
one:

    r_g = (t_g * C_g - R_g - sum_{k > g} N_{g,k} * r_k) / N_{g,g}

with ``t_g`` the published rate at age ``g``, ``C_g`` the eligible children
aged ``g``, ``R_g`` those in reporting families and ``N_{g,k}`` those in
non-reporting families whose eldest eligible child is aged ``k``, all weighted
at the stage's household weights. A rate outside [0, 1] is clipped and the
receipt says so (reporters alone exceed the published rate, or every family
claiming still falls short of it).

Opt-outs. HMRC publishes the families registered and, of them, the families
that opted out of payment because of the High Income Child Benefit Charge.
That share of the claiming families opts out, drawn among claiming families
that do not report receipt and whose highest adjusted net income is at or
above the income at which the charge takes the whole benefit; where that pool
is lighter than the target the rest is drawn from the families inside the
taper. A family in the taper still gains from the payment, so the fully
charged families come first.

The engine pays on ``would_claim_child_benefit`` alone, so the stage stores
``claims and not opted out`` there and the opt-out in
``child_benefit_opts_out``. A benefit unit with no eligible child keeps its
early draw and is never opted out.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.uc_capital_coherence import (
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
CHILD_BENEFIT_ENGINE_VARIABLES = (
    CHILD_BENEFIT_ELIGIBILITY_VARIABLE,
    CHILD_BENEFIT_CHARGE_INCOME_VARIABLE,
)
CHILD_BENEFIT_REPORTED_ANCHOR = "child_benefit_reported_anchor"
CHILD_BENEFIT_AGGREGATES = {CHILD_BENEFIT_REPORTED_ANCHOR: "child_benefit_reported"}
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
CHILD_BENEFIT_FAMILY_KEY = "age of the benefit unit's eldest eligible child"
CHILD_BENEFIT_CLAIM_RULE = (
    "a reporter claims; a non-reporting family whose eldest eligible child is "
    "aged g claims at r_g = (t_g * C_g - R_g - sum over k > g of N_gk * r_k) / "
    "N_gg, weighted, solved from the eldest age down and clipped to [0, 1]"
)
CHILD_BENEFIT_OPT_OUT_RULE = (
    "published opted-out share of registered families, of the claiming "
    "families; drawn among claiming non-reporting families with the highest "
    "adjusted net income at or above the full-charge income, then, for any "
    "remainder, among those inside the taper"
)
CHILD_BENEFIT_PAYMENT_RULE = (
    "would_claim_child_benefit = claims and not opted out on families with an "
    "eligible child; a benefit unit without one keeps its early draw"
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

    def __post_init__(self) -> None:
        if not 0.0 < self.phase_out_start < self.phase_out_end:
            raise ValueError(
                "Child Benefit charge thresholds must satisfy 0 < start < end."
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
    """Read the charge's taper from the engine at ``{year}-01-01``."""

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
    return UKChildBenefitChargeThresholds(
        phase_out_start=float(charge.phase_out_start(instant)),
        phase_out_end=float(charge.phase_out_end(instant)),
        source=(
            f"policyengine-uk {metadata.version('policyengine-uk')} "
            f"{CHILD_BENEFIT_CHARGE_PARAMETERS} at {instant}"
        ),
    )


@dataclass(frozen=True)
class UKChildBenefitTakeUpResult:
    """Output frame and executed-effect receipt for the Child Benefit redraw."""

    frame: Frame
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
            "claim_rule": CHILD_BENEFIT_CLAIM_RULE,
            "opt_out_rule": CHILD_BENEFIT_OPT_OUT_RULE,
            "payment_rule": CHILD_BENEFIT_PAYMENT_RULE,
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
) -> UKChildBenefitTakeUpResult:
    """Redraw the claim and opt-out flags from engine eligibility and income."""

    validate_uk_national_frame(frame)
    assert_rules_engine_country(engine, "uk")
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    for table, columns, label in (
        (
            person,
            (
                "person_benunit_id",
                "person_household_id",
                "age",
                "child_benefit_reported",
            ),
            "person",
        ),
        (
            benunit,
            ("benunit_id", CHILD_BENEFIT_CLAIM_OUTPUT, CHILD_BENEFIT_OPT_OUT_OUTPUT),
            "benunit",
        ),
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
    reporter_ids = person.loc[
        pd.to_numeric(person["child_benefit_reported"], errors="coerce").fillna(0.0)
        > 0,
        "person_benunit_id",
    ]
    reporter = benunit["benunit_id"].isin(reporter_ids).to_numpy(dtype=bool)
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
    family = children > 0
    paid = claims & ~opt_out
    previous_claim = benunit[CHILD_BENEFIT_CLAIM_OUTPUT].fillna(False).to_numpy(bool)
    previous_opt_out = (
        benunit[CHILD_BENEFIT_OPT_OUT_OUTPUT].fillna(False).to_numpy(bool)
    )
    would_claim = np.where(family, paid, previous_claim)
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
        claims=claim_receipt,
        opt_outs=opt_out_receipt,
        in_payment={
            "weights": "household weights at the stage, before calibration",
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
        },
    )


def solve_child_benefit_claim_rates(
    *,
    child_ages: np.ndarray,
    child_key_ages: np.ndarray,
    child_reporter: np.ndarray,
    child_weights: np.ndarray,
    claim_rates: Mapping[int, float],
) -> tuple[dict[int, float], list[dict[str, object]]]:
    """Non-reporter claim rate by key age, and one receipt row per age.

    ``child_*`` describe the eligible children: own age, the family's key age
    (its eldest eligible child's), whether the family reports, and the weight.
    """

    child_ages = np.asarray(child_ages, dtype=np.int64)
    child_key_ages = np.asarray(child_key_ages, dtype=np.int64)
    child_reporter = np.asarray(child_reporter, dtype=bool)
    child_weights = np.asarray(child_weights, dtype=np.float64)
    unknown = sorted(set(child_ages.tolist()) - set(claim_rates))
    if unknown:
        raise ValueError(
            "HMRC publishes no Child Benefit claim rate for eligible children "
            f"aged {unknown}."
        )
    if (child_key_ages < child_ages).any():
        raise ValueError("A family's key age is below one of its children's ages.")
    residual: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    for age in sorted(claim_rates, reverse=True):
        of_age = child_ages == age
        eligible = float(child_weights[of_age].sum())
        reporting = float(child_weights[of_age & child_reporter].sum())
        own = of_age & ~child_reporter & (child_key_ages == age)
        own_mass = float(child_weights[own].sum())
        # Children of this age in non-reporting families keyed on an older
        # sibling claim at that sibling's rate, already solved.
        older = of_age & ~child_reporter & (child_key_ages > age)
        older_claimed = float(
            sum(
                child_weights[older & (child_key_ages == key)].sum() * rate
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
                "eligible_child_rows": int(of_age.sum()),
                "eligible_children": eligible,
                "reporter_children": reporting,
                "residual_rate_unclipped": raw,
                "residual_rate": rate,
                "clipped": bool(raw is not None and not 0.0 <= raw <= 1.0),
            }
        )
    rows.sort(key=lambda row: row["age"])
    return residual, rows


def assign_child_benefit_claims(
    *,
    child_units: np.ndarray,
    child_ages: np.ndarray,
    eldest: np.ndarray,
    reporter: np.ndarray,
    weights: np.ndarray,
    draws: np.ndarray,
    statistics: UKChildBenefitStatistics,
) -> tuple[np.ndarray, dict[str, object]]:
    """Family claims: reporters, and the rest at their key age's solved rate.

    ``child_units`` holds each eligible child's benefit-unit position and
    ``child_ages`` its whole age; ``eldest`` is each unit's eldest eligible
    child's age, or -1 for a unit without one.
    """

    child_units = np.asarray(child_units, dtype=np.int64)
    child_ages = np.asarray(child_ages, dtype=np.int64)
    eldest = np.asarray(eldest, dtype=np.int64)
    reporter = np.asarray(reporter, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    draws = np.asarray(draws, dtype=np.float64)
    family = eldest >= 0
    child_weights = weights[child_units]
    residual, rows = solve_child_benefit_claim_rates(
        child_ages=child_ages,
        child_key_ages=eldest[child_units],
        child_reporter=reporter[child_units],
        child_weights=child_weights,
        claim_rates=statistics.claim_rates,
    )
    unit_rate = np.zeros(len(eldest), dtype=np.float64)
    unit_rate[family] = np.asarray([residual[int(age)] for age in eldest[family]])
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
        "eligible_child_rows": int(len(child_units)),
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
    """Opt out the published share of claiming families, fully charged first."""

    claims = np.asarray(claims, dtype=bool)
    reporter = np.asarray(reporter, dtype=bool)
    highest_income = np.asarray(highest_income, dtype=np.float64)
    children = np.asarray(children, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    draws = np.asarray(draws, dtype=np.float64)
    candidates = claims & ~reporter
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
        "thresholds_source": thresholds.source,
        "reporters_above_charge_start": float(
            weights[
                claims & reporter & (highest_income > thresholds.phase_out_start)
            ].sum()
        ),
        "pools": pool_rows,
    }
    return opt_out, receipt


def _aligned(values: object, expected: int, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (expected,):
        raise ValueError(f"{label} must align to the person table.")
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
    rewrites = (CHILD_BENEFIT_CLAIM_OUTPUT, CHILD_BENEFIT_OPT_OUT_OUTPUT)
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
    "child_benefit_take_up_operation_parameters",
    "load_child_benefit_statistics",
    "redraw_child_benefit_take_up",
    "solve_child_benefit_claim_rates",
    "uk_child_benefit_charge_thresholds",
]
