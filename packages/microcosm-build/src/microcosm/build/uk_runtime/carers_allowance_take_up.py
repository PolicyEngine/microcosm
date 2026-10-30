"""Carer's Allowance claims by the carers the earnings limit keeps out
(microcosm#1319).

``frs_spine`` sets ``would_claim_carers_allowance`` from reported receipt, and
the SPI income chain re-derives it from the refilled receipt on the rows it
redraws, so that ``care_hours`` qualifies a carer for the Universal Credit
carer element without paying the allowance to every carer (#882). A carer
whose earnings are above the limit cannot report the allowance, so no reform
that widens it, such as the April 2025 rise of the limit from £151 to £196 a
week, reached a carer who would claim once entitled.

This stage runs after the SPI income chain and the other take-up redraws. It
opens the flag to the carers the earnings limit alone keeps from the
allowance at the survey year: a person whose ``care_hours`` reach the engine's
``gov.dwp.carers_allowance.min_hours``, who cares for a severely disabled
person, who reports no Carer's Allowance, and whom
``is_gainfully_employed_for_carers_allowance`` holds gainfully employed at the
survey year (the 2024-25 limit of £151 a week; the engine reads a year's
parameters at 30 April). Such a person claims at the take-up contract's
``carers_allowance_newly_entitled`` rate, on an identity-keyed uniform placed
by weight within strata (below):

    would_claim = reported OR (eligible AND position < rate)

The allowance is paid only for caring for a person the engine holds
``is_severely_disabled_for_carers_allowance`` (SSCBA 1992 s. 70(2)), which the
FRS care hours do not record (microcosm#1418). A carer whose largest
in-household band alone reaches the hours (``frs_care_hours_in_household``,
the largest of the four in-household FRS bands) cares for such a person when
another member of the household is one at the survey year: the allowance
needs the hours for one person (SSCBA 1992 s. 70). A carer whose largest
in-household band does not reach the hours cares outside the household, or
spreads the hours over bands; such a carer joins the pool on a second
identity-keyed uniform below the share of the in-household carers who report
no allowance that have such a member (María's decision (a) of 2026-10-28):

    pool = (in_household AND other_member_severely_disabled)
           OR (other_carer AND position_pool < in_household_share)

Both draws are balanced by weight within strata (microcosm#1418). Each
uniform orders a stratum's records, and each record takes the midpoint of its
slice of the stratum's cumulative weight (``weight_balanced_uniforms``, as
policyengine-uk's ``stratified_uniform``), rotated by a start drawn per
stratum from its code and its members' identities, so every stratum claims
the rate of its weight to within one record and every carer claims with
probability equal to the rate, however few share the stratum (the independent
review of microcosm#1420, finding 4, and its re-review, G). The stage
refuses a stratum that misses by more than its heaviest record (Anthony's
reviews of microcosm#1420). Independent draws reach
it only in expectation, and on the few source households over the limit one draw put the
2031 taper's cost a quarter below its expectation. The take-up strata are the
route (in-household or not), whether the benefit unit reports a means-tested
benefit at the survey year, and the survey-year weekly earnings band, one
weekly rate wide above the limit and open above three. The pool's strata are
the last two. The rate, the pool and the caseload do not change.

A reporter stays true. A carer under the limit who reports no allowance
stays false, as the survey records them entitled and not claiming. Scotland
is drawn too, at the same rate (María's ruling of 2026-10-23): Carer Support
Payment replaces the allowance there and reads the same flag, and the take-up
contract holds no rate of its own for it. Since policyengine-uk 2.146.1
(PolicyEngine/policyengine-uk#2473) the engine pays it only under its own
earnings limit, so a drawn Scottish carer is paid only where its rules allow.
The receipt reports Scotland apart. The incoming flag must still follow
reported receipt, since the stage rebuilds it from that anchor.

The earnings test is read where the stage runs, before ``salary_sacrifice``
moves some contributors' pension amounts out of their pay, which lowers the
earnings the engine tests by about half the amount moved. A drawn carer whom
that conversion takes under the survey year's limit keeps the flag, although
the released file then shows them entitled and not reporting. Reading the
converted pay would need the stage after ``salary_sacrifice``, with a draw
keyed on lineage so that the CGT clones keep their source's draw.

The receipt is aggregate-only (UKDS disclosure control): support as bands of
distinct source households, a weighted total only where ten or more carry
it, and a share only where neither of its sides rests on one to nine.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    integer_record_keys,
    stable_identity_uniforms,
    weight_balanced_uniforms,
)
from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    small_cell_share,
    source_household_ids,
    support_band,
)
from microcosm.build.uk_runtime.take_up_contract import (
    UKTakeUpContract,
    load_uk_take_up_contract,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

__all__ = [
    "CARERS_ALLOWANCE_DISABILITY_TEST",
    "CARERS_ALLOWANCE_ENGINE_VARIABLES",
    "CARERS_ALLOWANCE_NEWLY_ENTITLED_RATE_KEY",
    "CARERS_ALLOWANCE_TAKE_UP_OUTPUT",
    "CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME",
    "UKCarersAllowancePolicy",
    "UKCarersAllowancePool",
    "UKCarersAllowanceTakeUpResult",
    "UKCarersAllowanceTakeUpStageTransform",
    "assert_strata_within_one_record",
    "carers_allowance_earnings_band",
    "carers_allowance_eligible",
    "carers_allowance_pool",
    "carers_allowance_take_up_operation_parameters",
    "means_tested_benefit_unit",
    "other_member_severely_disabled",
    "redraw_carers_allowance_take_up",
    "stratum_starts",
    "uk_carers_allowance_policy",
]

CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME = "carers_allowance_take_up"
CARERS_ALLOWANCE_TAKE_UP_OUTPUT = "would_claim_carers_allowance"
CARERS_ALLOWANCE_TAKE_UP_SEED = 0
CARERS_ALLOWANCE_REPORTED = "carers_allowance_reported"
CARERS_ALLOWANCE_CARE_HOURS = "care_hours"
CARERS_ALLOWANCE_EARNINGS_TEST = "is_gainfully_employed_for_carers_allowance"
#: The person a carer benefit is paid for caring for (SSCBA 1992 s. 70(2)).
CARERS_ALLOWANCE_DISABILITY_TEST = "is_severely_disabled_for_carers_allowance"
#: The annual earnings the engine's test spreads over 52 weeks; the draw's
#: earnings bands read them the same way.
CARERS_ALLOWANCE_EARNINGS = "carers_allowance_earnings"
CARERS_ALLOWANCE_ENGINE_VARIABLES = (
    CARERS_ALLOWANCE_EARNINGS_TEST,
    CARERS_ALLOWANCE_DISABILITY_TEST,
    CARERS_ALLOWANCE_EARNINGS,
)
CARERS_ALLOWANCE_WEEKS_IN_YEAR = 52
#: The means-tested benefits a benefit unit reports at the survey year: a
#: carer's allowance counts against them pound for pound, so the draw keeps
#: claimants on them and off them at the rate apart (microcosm#1418).
CARERS_ALLOWANCE_MEANS_TESTED_REPORTED = (
    "universal_credit_reported",
    "housing_benefit_reported",
    "income_support_reported",
    "jsa_income_reported",
    "esa_income_reported",
    "working_tax_credit_reported",
    "child_tax_credit_reported",
    "pension_credit_reported",
)
#: The earnings bands, one weekly rate wide above the survey-year limit; the
#: last is open. A carer at or under the limit is band -1.
CARERS_ALLOWANCE_EARNINGS_BANDS = 4
#: frs_spine.FRS_IN_HOUSEHOLD_CARE_HOURS_COLUMN: the weekly hours of the
#: largest in-household FRS care band.
CARERS_ALLOWANCE_IN_HOUSEHOLD_HOURS = "frs_care_hours_in_household"
#: The second identity-keyed uniform, which draws a carer whose hours are not
#: all given inside the household into the pool (microcosm#1418).
CARERS_ALLOWANCE_POOL_SALT = "carers_allowance_outside_household_pool"
#: The earnings test is read on the frame's own values at the survey year, the
#: limit the survey's carers were held to; the stage refuses any other period.
CARERS_ALLOWANCE_YEAR_RULE = "survey_year"
CARERS_ALLOWANCE_NEWLY_ENTITLED_RATE_KEY = "carers_allowance_newly_entitled"
#: Where Carer Support Payment replaces the allowance and reads the same flag:
#: drawn as everywhere else, under the engine's own earnings limit from
#: policyengine-uk 2.146.1, and reported apart in the receipt.
CARER_SUPPORT_PAYMENT_REGION = "SCOTLAND"
CARERS_ALLOWANCE_PARAMETERS = "gov.dwp.carers_allowance"
CARERS_ALLOWANCE_REPORTED_ANCHOR = "carers_allowance_reported > 0"
CARERS_ALLOWANCE_POOL_RULE = (
    "a carer whose frs_care_hours_in_household reach min_hours cares for a "
    "severely disabled person when another member of the household is "
    "is_severely_disabled_for_carers_allowance at the survey year; a carer "
    "whose largest in-household band does not reach the hours does on a second "
    "identity-keyed uniform (salt carers_allowance_outside_household_pool), "
    "placed by weight within strata of survey-year means-tested receipt and "
    "earnings band, below the household-weighted share of the in-household "
    "carers reporting no allowance who have such a member"
)
CARERS_ALLOWANCE_STRATA = (
    "the route (in-household or not), whether the benefit unit reports a "
    "means-tested benefit at the survey year (universal credit, housing "
    "benefit, income support, income-based JSA, income-related ESA, tax "
    "credits or pension credit), and the survey-year weekly earnings band, one "
    "weekly rate wide above the limit and open above three; the pool's strata "
    "are the last two"
)
CARERS_ALLOWANCE_ELIGIBILITY = (
    "persons whose care_hours reach policyengine-uk's "
    "gov.dwp.carers_allowance.min_hours, who care for a severely disabled "
    "person, who report no Carer's Allowance, and whom "
    "is_gainfully_employed_for_carers_allowance holds gainfully employed at the "
    "survey year: the carers the earnings limit alone keeps from the allowance "
    f"({CARERS_ALLOWANCE_POOL_RULE}). In Scotland the flag is Carer Support "
    "Payment's, which policyengine-uk pays under its own earnings limit from "
    "2.146.1"
)
CARERS_ALLOWANCE_RATE_DERIVATION = (
    "the take-up contract's carers_allowance_newly_entitled rate, with the "
    "identity-keyed uniforms placed by weight within strata "
    f"({CARERS_ALLOWANCE_STRATA}) and rotated by a start drawn per stratum, so "
    "that every stratum claims the rate of its eligible weight to within one "
    "record and every eligible carer claims with probability equal to the rate; "
    "would_claim = reported OR (eligible "
    "AND position < rate), so a reporter stays true and every other person false"
)


@dataclass(frozen=True)
class UKCarersAllowancePolicy:
    """The engine's hours condition, weekly earnings limit and rate at one instant.

    ``min_hours`` forms the eligible set; the earnings test itself is the
    engine's own ``is_gainfully_employed_for_carers_allowance``, so
    ``earnings_limit`` and ``weekly_rate`` only band the survey-year earnings
    for the draw's strata and are recorded for the receipt.
    """

    min_hours: float
    earnings_limit: float
    weekly_rate: float
    instant: str
    source: str


def uk_carers_allowance_policy(year: int | str) -> UKCarersAllowancePolicy:
    """Read the hours condition and the earnings limit as the engine reads ``year``.

    policyengine-uk reads a yearly parameter at 30 April of the year
    (``convert_to_fiscal_year_parameters``), so the survey year 2024 reads the
    2024-25 limit.
    """

    try:
        import policyengine_uk
        from policyengine_core.parameters import ParameterNode
    except ImportError as exc:
        raise ImportError(
            "The Carer's Allowance take-up policy requires "
            "`uv sync --all-packages --extra uk`."
        ) from exc

    parameters = ParameterNode(
        directory_path=str(Path(policyengine_uk.__file__).parent / "parameters")
    )
    instant = f"{int(year)}-04-30"
    carers_allowance = parameters.gov.dwp.carers_allowance
    return UKCarersAllowancePolicy(
        min_hours=float(carers_allowance.min_hours(instant)),
        earnings_limit=float(carers_allowance.earnings_limit(instant)),
        weekly_rate=float(carers_allowance.rate(instant)),
        instant=instant,
        source=(
            f"policyengine-uk {metadata.version('policyengine-uk')} "
            f"{CARERS_ALLOWANCE_PARAMETERS} at {instant}"
        ),
    )


@dataclass(frozen=True)
class UKCarersAllowancePool:
    """The carers of a severely disabled person, and the groups they come from.

    ``in_household`` carers reach the hours in their largest in-household band
    and ``other`` carers do not (they care outside the household, or spread
    the hours over bands); ``pool`` holds the first where another member is
    severely disabled and the second where their pool uniform falls below
    ``share``. With ``pool_strata`` the uniforms of the other carers who report
    no allowance are first placed by weight within those strata and rotated by
    a start drawn per stratum from its members' identities (``pool_identities``),
    so that each stratum draws ``share`` of its weight to within one record and
    each carer joins with probability ``share``.
    """

    pool: np.ndarray
    in_household: np.ndarray
    other: np.ndarray
    share: float | None


def carers_allowance_pool(
    *,
    care_hours: np.ndarray,
    in_household_hours: np.ndarray,
    other_member_severely_disabled: np.ndarray,
    reported: np.ndarray,
    weights: np.ndarray,
    pool_draws: np.ndarray,
    min_hours: float,
    pool_strata: np.ndarray | None = None,
    pool_identities: np.ndarray | None = None,
) -> UKCarersAllowancePool:
    """The carers whom the allowance can pay for the care they give."""

    hours = np.asarray(care_hours, dtype=np.float64) >= float(min_hours)
    in_household = hours & (
        np.asarray(in_household_hours, dtype=np.float64) >= float(min_hours)
    )
    other = hours & ~in_household
    disabled_member = np.asarray(other_member_severely_disabled, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    base = in_household & ~np.asarray(reported, dtype=bool)
    base_weight = float(weights[base].sum())
    share = (
        float(weights[base & disabled_member].sum()) / base_weight
        if base_weight > 0.0
        else None
    )
    draws = np.asarray(pool_draws, dtype=np.float64)
    if pool_strata is not None:
        if pool_identities is None:
            raise ValueError("Balanced pool draws need the carers' identities.")
        # Only the other carers who report no allowance can join the pool's
        # eligible part, so the balance is struck over them.
        drawable = other & ~np.asarray(reported, dtype=bool)
        draws = draws.copy()
        drawable_strata = np.asarray(pool_strata)[drawable]
        draws[drawable] = weight_balanced_uniforms(
            drawable_strata,
            draws[drawable],
            weights[drawable],
            starts=stratum_starts(
                drawable_strata,
                salt=CARERS_ALLOWANCE_POOL_SALT,
                identities=np.asarray(pool_identities)[drawable],
            ),
        )
    drawn = draws < (share or 0.0)
    pool = (in_household & disabled_member) | (other & drawn)
    return UKCarersAllowancePool(
        pool=pool, in_household=in_household, other=other, share=share
    )


def stratum_starts(
    strata: np.ndarray, *, salt: str, identities: np.ndarray
) -> np.ndarray:
    """Each record's stratum start: one identity-keyed uniform per stratum.

    Keyed on the stratum code and a digest of its members' sorted identities,
    under the draw's salt. A rebuild of the same people rotates each stratum
    the same way whatever the row order, as the records' own draws are keyed
    on the person, while a stratum of other people starts elsewhere: a carer
    alone in a stratum claims with probability equal to the rate over
    identities, in any build or survey vintage (the re-review of
    microcosm#1420, G). Keyed on the code alone, a one-record stratum's
    outcome was fixed per code in every build.
    """

    codes = np.asarray(strata)
    members = integer_record_keys(identities, label="Carer's Allowance identities")
    if codes.shape != members.shape:
        raise ValueError("Stratum starts need one identity per record.")
    unique = np.unique(codes)
    keys = []
    for code in unique:
        digest = hashlib.blake2b(digest_size=16)
        for member in np.sort(members[codes == code]):
            digest.update(f"{int(member)},".encode())
        keys.append(f"{code}:{digest.hexdigest()}")
    starts = stable_identity_uniforms(
        keys, seed=CARERS_ALLOWANCE_TAKE_UP_SEED, salt=f"{salt}:stratum_start"
    )
    return starts[np.searchsorted(unique, codes)]


def assert_strata_within_one_record(
    members: np.ndarray,
    strata: np.ndarray,
    taken: np.ndarray,
    weights: np.ndarray,
    share: float,
    *,
    label: str,
) -> None:
    """Refuse a stratum whose taken weight misses ``share`` of its weight by
    more than its heaviest record.

    The rotated placement takes each stratum's share to within half a record
    at each end of the stratum's arc, one record in all, by construction. A
    larger miss means the draw was not balanced.
    """

    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    strata = np.asarray(strata)
    for code in np.unique(strata[members]):
        stratum = members & (strata == code)
        stratum_weights = weights[stratum]
        miss = abs(
            float(weights[stratum & taken].sum())
            - float(share) * float(stratum_weights.sum())
        )
        if miss > float(stratum_weights.max()) * (1.0 + 1e-9) + 1e-9:
            raise ValueError(
                f"Carer's Allowance {label} stratum {int(code)} misses its share by "
                "more than one record: the draw was not balanced by weight."
            )


def carers_allowance_eligible(
    *,
    pool: np.ndarray,
    reported: np.ndarray,
    gainfully_employed: np.ndarray,
) -> np.ndarray:
    """The carers the earnings limit alone keeps from the allowance."""

    return (
        np.asarray(pool, dtype=bool)
        & ~np.asarray(reported, dtype=bool)
        & np.asarray(gainfully_employed, dtype=bool)
    )


def other_member_severely_disabled(
    household_ids: np.ndarray, severely_disabled: np.ndarray
) -> np.ndarray:
    """Whether someone else in each person's household is severely disabled."""

    disabled = np.asarray(severely_disabled, dtype=bool).astype(np.int64)
    in_household = (
        pd.Series(disabled).groupby(np.asarray(household_ids)).transform("sum")
    ).to_numpy()
    return (in_household - disabled) > 0


def means_tested_benefit_unit(
    benunit_ids: np.ndarray, reported_amounts: np.ndarray
) -> np.ndarray:
    """Whether anyone in each person's benefit unit reports a means-tested benefit."""

    reports = (np.asarray(reported_amounts, dtype=np.float64) > 0.0).any(axis=1)
    return (
        pd.Series(reports)
        .groupby(np.asarray(benunit_ids))
        .transform("max")
        .to_numpy(dtype=bool)
    )


def carers_allowance_earnings_band(
    annual_earnings: np.ndarray, *, earnings_limit: float, weekly_rate: float
) -> np.ndarray:
    """The survey-year weekly earnings band against the limit.

    The engine's test spreads the year's earnings over 52 weeks and disregards
    a fraction of a penny; read the same way, a carer at or under the limit is
    band -1, and one over it takes the whole weekly rates by which the earnings
    exceed it, the last band open (``CARERS_ALLOWANCE_EARNINGS_BANDS``).
    """

    if not weekly_rate > 0.0:
        raise ValueError("The Carer's Allowance weekly rate must be positive.")
    weekly = (
        np.floor(
            np.asarray(annual_earnings, dtype=np.float64)
            / CARERS_ALLOWANCE_WEEKS_IN_YEAR
            * 100
            + 1e-6
        )
        / 100
    )
    over = weekly > float(earnings_limit)
    rates_over = np.floor((weekly - float(earnings_limit)) / float(weekly_rate))
    return np.where(
        over, np.minimum(rates_over, CARERS_ALLOWANCE_EARNINGS_BANDS - 1), -1
    ).astype(np.int64)


_BAND_COUNT = CARERS_ALLOWANCE_EARNINGS_BANDS + 1


def _band_label(band: int) -> str:
    if band < 0:
        return "at or under the limit"
    if band == CARERS_ALLOWANCE_EARNINGS_BANDS - 1:
        return f"{band} or more weekly rates over"
    return f"{band} to {band + 1} weekly rates over"


@dataclass(frozen=True)
class UKCarersAllowanceTakeUpResult:
    """Output frame and aggregate receipt for the Carer's Allowance draw."""

    frame: Frame
    receipt: dict[str, object]

    def evidence(self) -> dict[str, object]:
        return {"stage": CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME, **self.receipt}


@dataclass(frozen=True)
class UKCarersAllowanceTakeUpStageTransform:
    """Whole-stage callable for the post-SPI Carer's Allowance draw."""

    stage: SourceStageSpec
    engine: object
    contract: UKTakeUpContract | None = None
    policy: UKCarersAllowancePolicy | None = None
    last_result: UKCarersAllowanceTakeUpResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = redraw_carers_allowance_take_up(
            frame,
            engine=self.engine,
            contract=self.contract or load_uk_take_up_contract(),
            policy=self.policy,
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


def redraw_carers_allowance_take_up(
    frame: Frame,
    *,
    engine: object,
    contract: UKTakeUpContract,
    policy: UKCarersAllowancePolicy | None = None,
) -> UKCarersAllowanceTakeUpResult:
    """Open ``would_claim_carers_allowance`` to carers over the earnings limit."""

    validate_uk_national_frame(frame)
    assert_rules_engine_country(engine, "uk")
    person = frame.table("person").copy()
    household = frame.table("household").copy()
    for table, columns, label in (
        (
            person,
            (
                "person_id",
                "person_household_id",
                CARERS_ALLOWANCE_REPORTED,
                CARERS_ALLOWANCE_CARE_HOURS,
                CARERS_ALLOWANCE_IN_HOUSEHOLD_HOURS,
                CARERS_ALLOWANCE_TAKE_UP_OUTPUT,
                "person_benunit_id",
                *CARERS_ALLOWANCE_MEANS_TESTED_REPORTED,
            ),
            "person",
        ),
        (household, ("household_id", "region"), "household"),
    ):
        missing = sorted(set(columns) - set(table.columns))
        if missing:
            raise ValueError(
                f"Carer's Allowance take-up {label} columns missing: {missing}."
            )
    # Compared as the label the frame carries, so that a "2024-25" or another
    # year is refused by this message, not by a failed int().
    period = uk_time_period(frame)
    survey_year = resolve_uk_year_rule(CARERS_ALLOWANCE_YEAR_RULE)
    if period != str(survey_year):
        raise ValueError(
            "Carer's Allowance take-up reads the earnings test on the frame's "
            f"{CARERS_ALLOWANCE_YEAR_RULE} {survey_year} values, but the frame's "
            f"period is {period!r}."
        )
    # The graph node is an open surface, so the lineage arrives with every
    # other live cell; the receipt counts support by it.
    household_sources = pd.Series(
        source_household_ids(household, stage="The Carer's Allowance take-up stage"),
        index=household["household_id"].to_numpy(),
    )
    reported = (
        pd.to_numeric(person[CARERS_ALLOWANCE_REPORTED], errors="coerce")
        .fillna(0.0)
        .to_numpy()
        > 0.0
    )
    previous = person[CARERS_ALLOWANCE_TAKE_UP_OUTPUT].fillna(False).to_numpy(bool)
    if (previous != reported).any():
        raise ValueError(
            f"{CARERS_ALLOWANCE_TAKE_UP_OUTPUT} must arrive as reported receipt "
            f"({CARERS_ALLOWANCE_REPORTED_ANCHOR}), as frs_spine and the SPI "
            "refresh set it: the stage rebuilds the flag from that anchor and "
            "would overwrite another writer."
        )
    resolved = policy or uk_carers_allowance_policy(survey_year)
    rate = _rate(contract.rate(CARERS_ALLOWANCE_NEWLY_ENTITLED_RATE_KEY))
    materialized = engine.materialize(
        frame, list(CARERS_ALLOWANCE_ENGINE_VARIABLES), period
    )
    missing = sorted(set(CARERS_ALLOWANCE_ENGINE_VARIABLES) - set(materialized))
    if missing:
        raise ValueError(
            f"Carer's Allowance take-up engine outputs missing: {missing}."
        )
    gainfully_employed = _aligned(
        materialized[CARERS_ALLOWANCE_EARNINGS_TEST],
        len(person),
        CARERS_ALLOWANCE_EARNINGS_TEST,
    ).astype(bool)
    severely_disabled = _aligned(
        materialized[CARERS_ALLOWANCE_DISABILITY_TEST],
        len(person),
        CARERS_ALLOWANCE_DISABILITY_TEST,
    ).astype(bool)
    earnings = _aligned(
        materialized[CARERS_ALLOWANCE_EARNINGS], len(person), CARERS_ALLOWANCE_EARNINGS
    ).astype(np.float64)
    means_tested = means_tested_benefit_unit(
        person["person_benunit_id"].to_numpy(),
        person[list(CARERS_ALLOWANCE_MEANS_TESTED_REPORTED)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64),
    )
    bands = carers_allowance_earnings_band(
        earnings,
        earnings_limit=resolved.earnings_limit,
        weekly_rate=resolved.weekly_rate,
    )
    # The pool's strata: means-tested receipt by earnings band.
    pool_strata = means_tested.astype(np.int64) * _BAND_COUNT + (bands + 1)
    person_households = person["person_household_id"]
    regions = person_households.map(
        household.set_index("household_id")["region"].astype(str)
    )
    if regions.isna().any():
        raise ValueError("Every person's household must carry a region.")
    scotland = regions.eq(CARER_SUPPORT_PAYMENT_REGION).to_numpy(dtype=bool)
    care_hours = (
        pd.to_numeric(person[CARERS_ALLOWANCE_CARE_HOURS], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    in_household_hours = (
        pd.to_numeric(person[CARERS_ALLOWANCE_IN_HOUSEHOLD_HOURS], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    pool = carers_allowance_pool(
        care_hours=care_hours,
        in_household_hours=in_household_hours,
        other_member_severely_disabled=other_member_severely_disabled(
            person_households.to_numpy(), severely_disabled
        ),
        reported=reported,
        weights=weights,
        pool_draws=stable_identity_uniforms(
            person["person_id"].to_numpy(),
            seed=CARERS_ALLOWANCE_TAKE_UP_SEED,
            salt=CARERS_ALLOWANCE_POOL_SALT,
        ),
        min_hours=resolved.min_hours,
        pool_strata=pool_strata,
        pool_identities=person["person_id"].to_numpy(),
    )
    eligible = carers_allowance_eligible(
        pool=pool.pool,
        reported=reported,
        gainfully_employed=gainfully_employed,
    )
    draws = stable_identity_uniforms(
        person["person_id"].to_numpy(),
        seed=CARERS_ALLOWANCE_TAKE_UP_SEED,
        salt=CARERS_ALLOWANCE_TAKE_UP_OUTPUT,
    )
    # The take-up strata add the route to the pool's.
    take_up_strata = pool.other.astype(np.int64) * 2 * _BAND_COUNT + pool_strata
    positions = draws.copy()
    positions[eligible] = weight_balanced_uniforms(
        take_up_strata[eligible],
        draws[eligible],
        weights[eligible],
        starts=stratum_starts(
            take_up_strata[eligible],
            salt=CARERS_ALLOWANCE_TAKE_UP_OUTPUT,
            identities=person["person_id"].to_numpy()[eligible],
        ),
    )
    drawn = eligible & (positions < rate)
    drawable_other = pool.other & ~reported
    assert_strata_within_one_record(
        eligible, take_up_strata, drawn, weights, rate, label="take-up"
    )
    assert_strata_within_one_record(
        drawable_other, pool_strata, pool.pool, weights, pool.share or 0.0, label="pool"
    )
    would_claim = reported | drawn
    person[CARERS_ALLOWANCE_TAKE_UP_OUTPUT] = would_claim
    result = uk_national_frame(
        person=person,
        benunit=frame.table("benunit").copy(),
        household=household,
        time_period=period,
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)

    source_ids = person_households.map(household_sources).to_numpy()

    # Weighted aggregates only; a figure one to nine source households carry
    # is "<10" (UKDS disclosure control).
    def sources(mask: np.ndarray) -> int:
        return distinct_count(source_ids, mask, weights)

    def guarded_share(part: np.ndarray, whole: np.ndarray) -> float | str | None:
        # The share times the whole's weight gives back either side, so a
        # share is given only where neither side rests on one to nine sources;
        # the part's weight is never given beside the whole's for the same
        # reason.
        whole_weight = float(weights[whole].sum())
        if not whole_weight:
            return None
        if 0 < min(sources(part & whole), sources(whole & ~part)) < 10:
            return "<10"
        return small_cell_share(
            float(weights[part & whole].sum()) / whole_weight, sources(whole)
        )

    def weighted(mask: np.ndarray) -> float | str:
        return small_cell_share(float(weights[mask].sum()), sources(mask))

    def strata_rows(
        members: np.ndarray, strata: np.ndarray, claimed: np.ndarray, target: object
    ) -> list[dict[str, object]]:
        rows = []
        for code in np.unique(strata[members]):
            stratum = members & (strata == code)
            within = int(code) % (2 * _BAND_COUNT)
            rows.append(
                {
                    "route": "other"
                    if int(code) >= 2 * _BAND_COUNT
                    else "in_household",
                    "means_tested": bool(within // _BAND_COUNT),
                    "earnings_band": _band_label(within % _BAND_COUNT - 1),
                    "support": support_band(sources(stratum)),
                    "target": target,
                    "realised_share": guarded_share(claimed, stratum),
                }
            )
        return rows

    eligible_sources = sources(eligible)
    eligible_weight = float(weights[eligible].sum())
    hours = care_hours >= resolved.min_hours
    over_limit = hours & ~reported & gainfully_employed
    in_household_base = pool.in_household & ~reported
    receipt = {
        "seed": CARERS_ALLOWANCE_TAKE_UP_SEED,
        "salt": CARERS_ALLOWANCE_TAKE_UP_OUTPUT,
        "eligibility": CARERS_ALLOWANCE_ELIGIBILITY,
        "rule": CARERS_ALLOWANCE_RATE_DERIVATION,
        "strata": CARERS_ALLOWANCE_STRATA,
        "rate_key": CARERS_ALLOWANCE_NEWLY_ENTITLED_RATE_KEY,
        "rate": rate,
        "survey_year": survey_year,
        "min_hours": resolved.min_hours,
        "weekly_earnings_limit": resolved.earnings_limit,
        "policy_source": resolved.source,
        "weights": "household weights at the stage, before calibration, per person",
        "pool_rule": CARERS_ALLOWANCE_POOL_RULE,
        "pool_salt": CARERS_ALLOWANCE_POOL_SALT,
        # The share the other carers are drawn at: in-household carers who
        # report no allowance, with a severely disabled member.
        "in_household_disabled_member_share": guarded_share(
            pool.pool, in_household_base
        ),
        # The carers over the limit before and after the pool: the hours alone
        # (microcosm#1319), then those caring for a severely disabled person,
        # each split by where the hours are given.
        "over_limit_carers_support": support_band(sources(over_limit)),
        # Beside the eligible weight, it would give back the weight the pool
        # leaves out, so it is withheld where that rests on one to nine.
        "over_limit_carers_weighted_persons": (
            "<10" if 0 < sources(over_limit & ~eligible) < 10 else weighted(over_limit)
        ),
        "over_limit_in_household_support": support_band(
            sources(over_limit & pool.in_household)
        ),
        "over_limit_other_support": support_band(sources(over_limit & pool.other)),
        "eligible_in_household_support": support_band(
            sources(eligible & pool.in_household)
        ),
        "eligible_other_support": support_band(sources(eligible & pool.other)),
        "eligible_support": support_band(eligible_sources),
        "eligible_weighted_persons": small_cell_share(
            eligible_weight, eligible_sources
        ),
        "drawn_support": support_band(sources(drawn)),
        "drawn_share_of_over_limit_pool": guarded_share(drawn, eligible),
        # Each stratum's realised share against its target, to within one
        # record by construction and refused at more; a share either side of
        # which one to nine source households carry is "<10".
        "weekly_rate": resolved.weekly_rate,
        "strata_within_one_record": True,
        "take_up_strata": strata_rows(eligible, take_up_strata, drawn, rate),
        "pool_strata": strata_rows(
            drawable_other,
            pool_strata + 2 * _BAND_COUNT,
            pool.pool,
            "in_household_disabled_member_share",
        ),
        # Scotland apart: there the flag is Carer Support Payment's (María's
        # ruling of 2026-10-23).
        "scotland_eligible_support": support_band(sources(eligible & scotland)),
        "scotland_eligible_weighted_persons": small_cell_share(
            float(weights[eligible & scotland].sum()),
            sources(eligible & scotland),
        ),
        "scotland_drawn_support": support_band(sources(drawn & scotland)),
    }
    return UKCarersAllowanceTakeUpResult(frame=result, receipt=receipt)


def _rate(value: float) -> float:
    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(
            f"{CARERS_ALLOWANCE_NEWLY_ENTITLED_RATE_KEY} must be a rate in [0, 1], "
            f"got {rate}."
        )
    return rate


def _aligned(values: object, expected: int, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (expected,):
        raise ValueError(f"{label} must align to the person table.")
    if array.dtype != bool and not np.isfinite(array.astype(np.float64)).all():
        raise ValueError(f"{label} must be finite.")
    return array


def carers_allowance_take_up_operation_parameters() -> dict[str, dict[str, object]]:
    """The reviewed operation payload the manifests must carry verbatim."""

    return {
        "materialize_rules_engine_predictors": {
            "predictors": list(CARERS_ALLOWANCE_ENGINE_VARIABLES),
            "consumed_only": True,
            "year_rule": CARERS_ALLOWANCE_YEAR_RULE,
        },
        "assign_binary_from_rate": {
            "output": CARERS_ALLOWANCE_TAKE_UP_OUTPUT,
            "draw": CARERS_ALLOWANCE_TAKE_UP_OUTPUT,
            "assignment_unit": "person",
            "eligibility": CARERS_ALLOWANCE_ELIGIBILITY,
            "reported_true_anchor": CARERS_ALLOWANCE_REPORTED_ANCHOR,
            "rate_key": CARERS_ALLOWANCE_NEWLY_ENTITLED_RATE_KEY,
            "rate_derivation": CARERS_ALLOWANCE_RATE_DERIVATION,
            "seed": CARERS_ALLOWANCE_TAKE_UP_SEED,
        },
    }


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    if stage.stage != CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME:
        raise ValueError(
            f"Carer's Allowance take-up received stage {stage.stage!r}, expected "
            f"{CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME!r}."
        )
    expected = carers_allowance_take_up_operation_parameters()
    kinds = [operation.kind for operation in stage.operations]
    if kinds != list(expected):
        raise ValueError(
            f"{CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME} operation order drifted: "
            f"expected {list(expected)}, got {kinds}."
        )
    actual = {
        operation.kind: dict(operation.parameters) for operation in stage.operations
    }
    if actual != expected:
        drifted = sorted(kind for kind in expected if actual[kind] != expected[kind])
        raise ValueError(
            f"{CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME} parameters drifted from the "
            f"reviewed mapping on operation(s) {drifted}."
        )
    if stage.outputs != () or stage.rewrites != (CARERS_ALLOWANCE_TAKE_UP_OUTPUT,):
        raise ValueError(
            f"{CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME} must declare no new outputs and "
            f"exactly the rewrite {CARERS_ALLOWANCE_TAKE_UP_OUTPUT!r}."
        )
    if stage.grain != "person":
        raise ValueError(
            f"{CARERS_ALLOWANCE_TAKE_UP_STAGE_NAME} is person grain, got "
            f"{stage.grain!r}."
        )
