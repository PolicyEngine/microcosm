"""UK salary-sacrifice QRF and headcount support conversion.

policyengine-uk's pension_contributions_via_salary_sacrifice_pre_2015 stays at
its default of nil: the FRS records no start date for an arrangement, so all
salary sacrifice counts in threshold income for the tapered annual allowance
(FA 2004 s. 228ZA(6)). For arrangements made before 9 July 2015 that
overstates threshold income, and so the taper and the number of people the
annual allowance charge reaches, in any comparison with HMRC.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    integer_record_keys,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.cgt_structure import (
    HOUSEHOLD_IS_CGT_CLONE,
    _assert_closed_world_operations,
)
from microcosm.build.uk_runtime.cgt_support import HOUSEHOLD_IS_CGT_SUPPORT_COPY
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
from microcosm.build.uk_runtime.spi_support import support_channel_column
from microcosm.frame import Frame, MassChangeRecord

QRF: Any | None = None

SALSAC_PREDICTORS = ("age", "employment_income")
SALSAC_OUTPUT = "pension_contributions_via_salary_sacrifice"
#: A converted record's employment income before the conversion, zero for
#: everyone the stage does not convert (microcosm#1063). The conversion
#: rewrites a converted record's pay and employee contribution in place after
#: the capital-gains stages have banded on them, which left the income those
#: stages saw unreconstructible from the artifact (the E8 identity receipt on
#: the 2 October spine). The carrier makes the rewrite reversible; it is an
#: internal column and leaves at the release boundary
#: (``UK_RELEASE_EXPORT_DROPPED_COLUMNS``).
SALSAC_PRE_CONVERSION_PAY_COLUMN = "salary_sacrifice_pre_conversion_pay"
SALSAC_STAGE_TARGET = 5_400_000.0
SALSAC_HMRC_ANCHOR = 7_700_000.0
SALSAC_ABOVE_2000_ANCHOR = 3_300_000.0
SALSAC_BELOW_2000_ANCHOR = 4_300_000.0
SALSAC_STAGING_RATIO = SALSAC_STAGE_TARGET / SALSAC_HMRC_ANCHOR
SALSAC_RATE_CAP = 0.5
#: The share of a converted record's pay that at most moves into salary
#: sacrifice. Half keeps the sacrifice within the pay that remains, and the
#: rest of the contribution stays an employee contribution (uk-data#518).
SALSAC_MOVE_PAY_SHARE_CAP = 0.5
SALSAC_QRF_SEED = 42
SALSAC_QRF_ESTIMATORS = 100
SALSAC_CONVERSION_SEED = 2024
SALSAC_CONVERSION_SALT = "salary_sacrifice_conversion"
#: Whom the QRF predicts for. Salary sacrifice is pay given up for a pension
#: contribution, so a person not asked the question who has no employment
#: income has none: the prediction covers the people not asked who are paid,
#: and everyone else not asked is zero (microcosm#1063).
SALSAC_TARGET_POPULATION = (
    "salary_sacrifice_asked != 1 and employment_income > 0, frame-wide; people "
    "not asked who have no employment income are set to zero"
)
#: Whom the QRF trains on. A respondent asked the question who has no pay
#: cannot show what a paid person sacrifices, and only predictions for paid
#: people are kept (uk-data#518).
SALSAC_TRAINING_POPULATION = (
    "support_channel == frs and not capital-gains clone and not CGT support "
    "copy and salary_sacrifice_asked == 1 and employment_income > 0"
)
#: The limit on every amount. Salary sacrifice is pay given up, so no amount,
#: reported or predicted, exceeds the pay the person still receives, and a
#: person without pay has none. It is an imputation bound, not a legal one.
#: It bites on SPI-channel rows above all: they keep their FRS twin's answer
#: beside pay redrawn from the SPI (uk-data#518).
SALSAC_AMOUNT_LIMIT = (
    "employment_income floored at zero; every row, reported or predicted, "
    "FRS and SPI channel"
)
SALSAC_MASS_CHANGE_REASON = (
    "Salary-sacrifice support stage rewrites pension columns and converted "
    "records' pay; household rows and typed household weights pass through and "
    "total household mass is conserved."
)
#: policyengine-uk 2.149.0's input for employer pension contributions made in
#: place of a bonus (pe-uk#2572 section 1). The GBP 2,000 NICs cap from April
#: 2029 and the annual allowance tests read it summed with salary sacrifice, so
#: the HMRC salary-sacrifice rows, which ASHE builds and which cover no bonus
#: sacrifice, keep binding the salary route alone (microcosm#1340).
BONUS_SACRIFICE_OUTPUT = "pension_contributions_via_bonus_sacrifice"
#: Each adult's gross FRS bonuses over the last 12 months
#: (frs_spine.FRS_BONUS_CARRIER_COLUMN).
BONUS_SACRIFICE_SOURCE = "frs_bonus_amount"
BONUS_SACRIFICE_ANCHOR_RESOURCE = "bonus_sacrifice_anchor.json"
BONUS_SACRIFICE_FACTS_RESOURCE = "obr_salary_sacrifice_costing_facts.json"
#: The OBR costing's tax-base record sets the vendored resource holds; the
#: anchor reads 2027-28's bonus-sacrifice line, the base's first year.
BONUS_SACRIFICE_RECORD_SETS = tuple(
    f"obr.salary_sacrifice_costing_2026_02.tax_base.fy{year}"
    for year in (2027, 2028, 2029, 2030)
)
BONUS_SACRIFICE_CONVERSION_SALT = "bonus_sacrifice_conversion"
#: The spine's year, and the OBR base's first year (2027-28), which the
#: engine's projection of the converted bonuses must meet: the feed's OBR rows
#: start there, after the calibration year, so the stage anchors on them.
BONUS_SACRIFICE_DATA_YEAR = 2024
BONUS_SACRIFICE_BASE_YEAR = 2027
#: The weekly upper earnings limit for Class 1 employee NICs, read where the
#: engine reads a year's parameters (30 April) in the data year.
BONUS_SACRIFICE_UEL_PARAMETER = (
    "gov.hmrc.national_insurance.class_1.thresholds.upper_earnings_limit"
)
BONUS_SACRIFICE_UEL_INSTANT = f"{BONUS_SACRIFICE_DATA_YEAR}-04-30"
#: The annual threshold the OBR's bonus-sacrifice base lies above (OBR,
#: supplementary forecast information on salary sacrifice, 5 February 2026,
#: Table 1.1).
BONUS_SACRIFICE_THRESHOLD = 2_000.0
#: Whom the conversion may convert. The OBR's costing applies employee NICs of
#: 2% to bonus sacrifice, the rate above the upper earnings limit, so a
#: candidate's pay reaches that limit and the sacrificed bonus sits above it.
#: The FRS records no bonus sacrifice (HMRC builds the base from scheme
#: returns), so a candidate is an adult with a bonus: an FRS adult's own, or
#: the share of pay an SPI-channel adult draws (``BONUS_SACRIFICE_SPI_DRAW``).
BONUS_SACRIFICE_CANDIDATES = (
    "a bonus (frs_bonus_amount, or an SPI-channel adult's drawn bonus) > 0 and "
    "employment_income at least the anchor's annual upper earnings limit, after "
    "the salary conversion"
)
#: A bonus sacrificed is never received, so the FRS records it in neither pay
#: nor its bonus amounts, and the stage adds it beside pay (María's ruling of
#: 2026-10-24 on Vahid's review of #1383, item 5). A received bonus marks a
#: bonus-paying job and its size; the salary route differs, since the FRS's
#: gross pay holds the contribution it converts.
BONUS_SACRIFICE_MOVE = (
    "the bonus, at most move_pay_share_cap of employment_income, to "
    "pension_contributions_via_bonus_sacrifice; employment_income unchanged, "
    "since a sacrificed bonus is never received"
)
#: An SPI-channel adult's pay is the tape's, and the FRS bonus it carries is
#: its twin's, so the two do not belong together (Vahid's review of #1383,
#: item 6). It draws a bonus share of pay from a base-FRS adult paid alike
#: instead (María's ruling of 2026-10-24); FRS rows keep their own bonus.
BONUS_SACRIFICE_SPI_DRAW = (
    "an SPI-channel adult with pay draws the bonus share of pay "
    "(frs_bonus_amount over employment_income) of a base-FRS adult with pay in "
    "the same pay cell, by person weight, keyed on person_id; the cells are the "
    "base-FRS paid adults' weighted pay quantiles at spi_cell_quantiles, and a "
    "cell with fewer than spi_min_donor_sources donor source households joins "
    "the one below"
)
BONUS_SACRIFICE_SPI_CELL_QUANTILES = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
)
BONUS_SACRIFICE_SPI_MIN_DONOR_SOURCES = 20
BONUS_SACRIFICE_SPI_DRAW_SALT = "bonus_sacrifice_spi_bonus_share"
BONUS_SACRIFICE_SOLVE = (
    "the candidates in the order of their identity-keyed uniforms, taken while "
    "the projected base is nearer the anchor's OBR base: the anchor's weight "
    "growth times the sum of person weight times "
    "(max(0, g (S + B) - threshold) - max(0, g S - threshold)), with g the "
    "anchor's earnings growth, S the salary sacrifice and B the bonus "
    "sacrificed; every candidate where even all fall short"
)


def load_salary_sacrifice_anchor() -> Mapping[str, Any]:
    """Load the committed HMRC salary-sacrifice anchor."""

    return json.loads(
        files("microcosm.build.uk")
        .joinpath("salary_sacrifice_anchor.json")
        .read_text(encoding="utf-8")
    )


def load_bonus_sacrifice_anchor() -> Mapping[str, Any]:
    """Load the committed bonus-sacrifice anchor (microcosm#1340).

    ``tools/pin_uk_bonus_sacrifice_anchor.py`` writes it from the OBR row the
    build vendors and the locked policyengine-uk's own projection factors.
    """

    return json.loads(
        files("microcosm.build.uk")
        .joinpath(BONUS_SACRIFICE_ANCHOR_RESOURCE)
        .read_text(encoding="utf-8")
    )


@dataclass(frozen=True)
class UKBonusSacrificeConversion:
    """Who sacrifices a bonus, how much, and the base it projects to."""

    sacrificed: np.ndarray
    candidates: np.ndarray
    participants: np.ndarray
    pay_share_capped: np.ndarray
    projected_base: float
    target_base: float
    all_candidates_fall_short: bool


def bonus_sacrifice_excess(
    salary_sacrifice: np.ndarray,
    bonus_sacrificed: np.ndarray,
    *,
    earnings_growth: float,
    threshold: float = BONUS_SACRIFICE_THRESHOLD,
) -> np.ndarray:
    """Each person's bonus sacrifice above the threshold in the anchor's year.

    The threshold applies to the two routes summed, as policyengine-uk reads
    them, so the bonus route's part is the excess of the sum over the excess
    of the salary sacrifice alone.
    """

    salary = np.asarray(salary_sacrifice, dtype=float) * earnings_growth
    bonus = np.asarray(bonus_sacrificed, dtype=float) * earnings_growth
    return np.maximum(0.0, salary + bonus - threshold) - np.maximum(
        0.0, salary - threshold
    )


def convert_bonuses_to_bonus_sacrifice(
    *,
    bonus: np.ndarray,
    employment_income: np.ndarray,
    salary_sacrifice: np.ndarray,
    weights: np.ndarray,
    uniforms: np.ndarray,
    anchor: Mapping[str, Any],
) -> UKBonusSacrificeConversion:
    """Add the bonus sacrifice the OBR's base needs beside pay.

    A candidate sacrifices its bonus, at most half its pay, and its pay stays
    as it is (``BONUS_SACRIFICE_MOVE``). Candidates are taken in the order of
    their identity-keyed uniforms while the projected base comes nearer the
    anchor's (``BONUS_SACRIFICE_SOLVE``): a person's place in the order is
    their own, but where it stops depends on everyone's contributions.
    """

    bonus = np.maximum(np.asarray(bonus, dtype=float), 0.0)
    pay = np.asarray(employment_income, dtype=float)
    salary = np.asarray(salary_sacrifice, dtype=float)
    weights = np.asarray(weights, dtype=float)
    uniforms = np.asarray(uniforms, dtype=float)
    limit = float(anchor["upper_earnings_limit"]["annual"])
    earnings_growth = float(anchor["earnings_growth"]["factor"])
    weight_growth = float(anchor["weight_growth"]["factor"])
    target = float(anchor["obr_base_above_threshold"])
    threshold = float(anchor["threshold"])
    candidates = (bonus > 0.0) & (pay >= limit) & (weights > 0.0)
    amount = np.minimum(bonus, SALSAC_MOVE_PAY_SHARE_CAP * np.maximum(pay, 0.0))
    amount = np.where(candidates, np.maximum(amount, 0.0), 0.0)
    pay_share_capped = candidates & (amount < bonus)
    contribution = (
        weight_growth
        * weights
        * bonus_sacrifice_excess(
            salary, amount, earnings_growth=earnings_growth, threshold=threshold
        )
    )
    order = np.flatnonzero(candidates)
    order = order[np.lexsort((order, uniforms[order]))]
    cumulative = np.cumsum(contribution[order])
    # Take the prefix whose projected base is nearest the target: none, or the
    # first k candidates in uniform order.
    totals = np.concatenate(([0.0], cumulative))
    taken = int(np.argmin(np.abs(totals - target)))
    participants = np.zeros(len(bonus), dtype=bool)
    participants[order[:taken]] = True
    sacrificed = np.where(participants, amount, 0.0)
    return UKBonusSacrificeConversion(
        sacrificed=sacrificed,
        candidates=candidates,
        participants=participants,
        pay_share_capped=pay_share_capped & participants,
        projected_base=float(totals[taken]),
        target_base=target,
        all_candidates_fall_short=bool(totals[-1] < target),
    )


@dataclass(frozen=True)
class UKSpiBonusDraw:
    """The bonuses the conversion reads: FRS rows' own, SPI rows' drawn."""

    bonus: np.ndarray
    recipients: np.ndarray
    cells: int


def draw_spi_bonuses(
    *,
    bonus: np.ndarray,
    employment_income: np.ndarray,
    weights: np.ndarray,
    person_ids: np.ndarray,
    source_ids: np.ndarray,
    donors: np.ndarray,
    spi: np.ndarray,
) -> UKSpiBonusDraw:
    """Each SPI-channel adult's bonus, as a share of pay drawn from FRS adults.

    ``donors`` marks the base-FRS adults (no clone, no support copy) and
    ``spi`` the SPI-channel adults. Each SPI adult with pay takes the bonus
    share of pay of one donor with pay in its pay cell, drawn by person weight
    on a uniform keyed on its own ``person_id`` (``BONUS_SACRIFICE_SPI_DRAW``);
    an SPI adult without pay has no bonus. Every other row keeps its own.
    """

    bonus = np.maximum(np.asarray(bonus, dtype=float), 0.0)
    pay = np.asarray(employment_income, dtype=float)
    weights = np.asarray(weights, dtype=float)
    spi = np.asarray(spi, dtype=bool)
    paid_donors = np.asarray(donors, dtype=bool) & (pay > 0.0) & (weights > 0.0)
    recipients = spi & (pay > 0.0)
    drawn = np.where(spi, 0.0, bonus)
    if not recipients.any():
        return UKSpiBonusDraw(bonus=drawn, recipients=recipients, cells=0)
    if not paid_donors.any():
        raise ValueError("SPI-channel bonuses need base-FRS adults with pay to draw.")
    donor_rows = np.flatnonzero(paid_donors)
    order = donor_rows[np.argsort(pay[donor_rows], kind="stable")]
    cumulative = np.cumsum(weights[order]) / weights[order].sum()
    edges = np.unique(
        [
            pay[order[min(np.searchsorted(cumulative, q), len(order) - 1)]]
            for q in BONUS_SACRIFICE_SPI_CELL_QUANTILES
        ]
    )
    raw_cells = np.searchsorted(edges, pay, side="right")
    # A cell with too few donor source households joins the one below, and the
    # lowest joins the one above, so every cell draws from at least the floor.
    sources = np.asarray(source_ids)
    groups = [[cell] for cell in range(len(edges) + 1)]

    def donor_sources(group: list[int]) -> int:
        return len(pd.unique(sources[paid_donors & np.isin(raw_cells, group)]))

    position = len(groups) - 1
    while len(groups) > 1 and position >= 0:
        if donor_sources(groups[position]) >= BONUS_SACRIFICE_SPI_MIN_DONOR_SOURCES:
            position -= 1
        elif position > 0:
            groups[position - 1] += groups.pop(position)
            position -= 1
        else:
            lowest = groups.pop(0)
            groups[0] = lowest + groups[0]
    lookup = np.empty(len(edges) + 1, dtype=int)
    for index, group in enumerate(groups):
        lookup[group] = index
    cells = lookup[raw_cells]
    share = np.zeros(len(pay))
    share[paid_donors] = np.minimum(bonus[paid_donors] / pay[paid_donors], 1.0)
    uniforms = stable_identity_uniforms(
        np.asarray(person_ids),
        seed=SALSAC_CONVERSION_SEED,
        salt=BONUS_SACRIFICE_SPI_DRAW_SALT,
    )
    ids = np.asarray(person_ids)
    for cell in np.unique(cells[recipients]):
        pool = np.flatnonzero(paid_donors & (cells == cell))
        if not len(pool):
            raise ValueError("An SPI-channel bonus pay cell has no donor to draw.")
        pool = pool[np.argsort(ids[pool], kind="stable")]
        pool_cumulative = np.cumsum(weights[pool]) / weights[pool].sum()
        takers = np.flatnonzero(recipients & (cells == cell))
        picks = np.minimum(
            np.searchsorted(pool_cumulative, uniforms[takers], side="right"),
            len(pool) - 1,
        )
        drawn[takers] = share[pool[picks]] * pay[takers]
    return UKSpiBonusDraw(bonus=drawn, recipients=recipients, cells=len(groups))


@dataclass(frozen=True)
class UKSalarySacrificeResult:
    """Transformed frame and the full headcount executed-effect receipt."""

    frame: Frame
    training_rows: int
    prediction_rows: int
    not_asked_without_pay_rows: int
    not_asked_without_pay_mass: float
    limited_to_pay_rows: int
    limited_to_pay_mass: float
    limited_to_pay_amount: float
    pre_headcount: float
    post_headcount: float
    shortfall: float
    donor_pool_mass: float
    rate: float
    cap_bound: bool
    converted_rows: int
    converted_mass: float
    moved_amount: float
    pay_share_capped_rows: int
    retained_employee_amount: float
    expected_converted_mass: float
    realization_deviation: float
    #: The bonus conversion's receipt (microcosm#1340).
    bonus_receipt: Mapping[str, object] | None = None

    def evidence(self) -> dict[str, object]:
        return {
            "stage": "salary_sacrifice",
            "qrf": {
                "training_rows": self.training_rows,
                "prediction_rows": self.prediction_rows,
                "target_population": SALSAC_TARGET_POPULATION,
                # People not asked who have no employment income: set to
                # zero, never predicted for.
                "not_asked_without_pay_rows": self.not_asked_without_pay_rows,
                "not_asked_without_pay_mass": self.not_asked_without_pay_mass,
                # Amounts above the person's pay, reported or predicted,
                # limited to it: rows, weighted people and unweighted pounds
                # removed.
                "amount_limit": SALSAC_AMOUNT_LIMIT,
                "limited_to_pay_rows": self.limited_to_pay_rows,
                "limited_to_pay_mass": self.limited_to_pay_mass,
                "limited_to_pay_amount": self.limited_to_pay_amount,
                "seed": SALSAC_QRF_SEED,
            },
            "headcount_receipt": {
                "target": SALSAC_STAGE_TARGET,
                "pre_headcount": self.pre_headcount,
                "post_headcount": self.post_headcount,
                "shortfall": self.shortfall,
                "donor_pool_mass": self.donor_pool_mass,
                "rate": self.rate,
                "rate_cap": SALSAC_RATE_CAP,
                "cap_bound": self.cap_bound,
                "converted_rows": self.converted_rows,
                "converted_mass": self.converted_mass,
                "moved_amount": self.moved_amount,
                # Converted records whose contribution was above half their
                # pay, and the unweighted pounds they keep as an employee
                # contribution.
                "move_pay_share_cap": SALSAC_MOVE_PAY_SHARE_CAP,
                "pay_share_capped_rows": self.pay_share_capped_rows,
                "retained_employee_amount": self.retained_employee_amount,
                "expected_converted_mass": self.expected_converted_mass,
                "realization_deviation": self.realization_deviation,
                "pre_conversion_pay_column": SALSAC_PRE_CONVERSION_PAY_COLUMN,
            },
            **(
                {}
                if self.bonus_receipt is None
                else {"bonus_sacrifice_receipt": dict(self.bonus_receipt)}
            ),
        }


@dataclass(frozen=True)
class UKSalarySacrificeStageTransform:
    """Whole-stage transform for salary-sacrifice support."""

    stage: SourceStageSpec
    anchor: Mapping[str, Any] | None = None
    bonus_anchor: Mapping[str, Any] | None = None
    last_result: UKSalarySacrificeResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        resource = self.anchor or load_salary_sacrifice_anchor()
        bonus = self.bonus_anchor or load_bonus_sacrifice_anchor()
        _assert_salary_sacrifice_stage_parameters(
            self.stage, anchor=resource, bonus_anchor=bonus
        )
        result = impute_salary_sacrifice(frame, bonus_anchor=bonus)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (
            SALSAC_OUTPUT,
            BONUS_SACRIFICE_OUTPUT,
            "employee_pension_contributions",
            "employment_income",
            SALSAC_PRE_CONVERSION_PAY_COLUMN,
        )

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def impute_salary_sacrifice(
    frame: Frame, *, bonus_anchor: Mapping[str, Any] | None = None
) -> UKSalarySacrificeResult:
    """Fit on the asked, paid base-FRS subset, limit every amount to pay,
    create additional SS support, then convert the bonuses the OBR's
    bonus-sacrifice base needs (the packaged anchor unless one is given)."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    household = frame.table("household").copy()
    required_person = {
        "person_id",
        "person_household_id",
        "age",
        "employment_income",
        "salary_sacrifice_asked",
        SALSAC_OUTPUT,
        "employee_pension_contributions",
    }
    missing = sorted(required_person - set(person.columns))
    if missing:
        raise ValueError(f"Salary-sacrifice person columns missing: {missing}.")
    households = household.set_index("household_id")
    person_households = person["person_household_id"]
    if not person_households.isin(households.index).all():
        raise ValueError("Salary-sacrifice people must map to a household.")
    channel_column = support_channel_column("household")
    if channel_column not in household.columns:
        raise ValueError(
            f"Salary-sacrifice training requires household {channel_column!r}."
        )
    channels = person_households.map(households[channel_column])
    clones = person_households.map(
        households.get(HOUSEHOLD_IS_CGT_CLONE, pd.Series(False, index=households.index))
    ).fillna(False)
    support_copies = person_households.map(
        households.get(
            HOUSEHOLD_IS_CGT_SUPPORT_COPY,
            pd.Series(False, index=households.index),
        )
    ).fillna(False)
    asked = pd.to_numeric(person["salary_sacrifice_asked"], errors="coerce")
    if asked.isna().any():
        raise ValueError("salary_sacrifice_asked contains non-numeric values.")
    numeric = person.loc[:, [*SALSAC_PREDICTORS, SALSAC_OUTPUT]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Salary-sacrifice QRF columns must be finite numeric values.")
    paid = numeric["employment_income"].gt(0.0)
    # The forest trains on respondents with pay: only predictions for paid
    # people are kept, and a respondent without pay shows nothing about them.
    training_mask = (
        channels.eq("frs")
        & ~clones.astype(bool)
        & ~support_copies.astype(bool)
        & asked.eq(1)
        & paid
    )
    if not training_mask.any():
        raise ValueError(
            "Salary-sacrifice QRF has no eligible asked FRS rows with pay."
        )
    # The prediction population is the people not asked who are paid; a
    # person not asked with no employment income sacrifices none of it.
    not_asked = ~asked.eq(1)
    predict_mask = not_asked & paid
    without_pay = not_asked & ~paid
    person.loc[without_pay, SALSAC_OUTPUT] = 0.0
    household_weights = pd.Series(
        frame.weights_for("household").values,
        index=household["household_id"],
    )
    person_weights = person_households.map(household_weights).to_numpy(dtype=float)
    training = numeric.loc[training_mask, [*SALSAC_PREDICTORS, SALSAC_OUTPUT]].copy()
    training["_fit_weight"] = person_weights[training_mask.to_numpy()]
    model = _qrf_class()(n_estimators=SALSAC_QRF_ESTIMATORS, seed=SALSAC_QRF_SEED)
    fitted = model.fit(
        training,
        list(SALSAC_PREDICTORS),
        [SALSAC_OUTPUT],
        weights="_fit_weight",
    )
    if predict_mask.any():
        # Uniforms keyed on each person's own id, never a positional stream:
        # a person's predicted sacrifice moves only when their own predictors do.
        keys = integer_record_keys(
            person.loc[predict_mask, "person_id"].to_numpy(),
            label="salary sacrifice person_id",
        )
        salt = f"salary_sacrifice:{SALSAC_OUTPUT}"
        predictions = fitted.predict_from_uniforms(
            numeric.loc[predict_mask, list(SALSAC_PREDICTORS)],
            quantiles={
                SALSAC_OUTPUT: stable_identity_uniforms(
                    keys, seed=SALSAC_QRF_SEED, salt=salt
                )
            },
            sign_uniforms={
                SALSAC_OUTPUT: stable_identity_uniforms(
                    keys, seed=SALSAC_QRF_SEED, salt=f"{salt}:sign"
                )
            },
        )
        predicted = pd.to_numeric(predictions[SALSAC_OUTPUT], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(predicted).all():
            raise ValueError("Salary-sacrifice QRF produced non-finite predictions.")
        person.loc[predict_mask, SALSAC_OUTPUT] = np.maximum(0.0, predicted)
    final_ss = pd.to_numeric(person[SALSAC_OUTPUT], errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    employee = pd.to_numeric(
        person["employee_pension_contributions"], errors="coerce"
    ).to_numpy(dtype=float, copy=True)
    employment_income = pd.to_numeric(
        person["employment_income"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(final_ss).all() or (final_ss < 0.0).any():
        raise ValueError("Salary-sacrifice amounts must be finite and non-negative.")
    if not np.isfinite(employee).all() or (employee < 0.0).any():
        raise ValueError("Employee-pension amounts must be finite and non-negative.")
    # A sacrifice comes out of pay, so every amount is limited to the pay the
    # person still receives: reported and predicted alike, and SPI-channel
    # rows too, whose kept answer is their FRS twin's beside redrawn pay.
    pay_limit = np.maximum(employment_income, 0.0)
    limited = final_ss > pay_limit
    limited_to_pay_amount = float((final_ss[limited] - pay_limit[limited]).sum())
    final_ss = np.minimum(final_ss, pay_limit)
    has_ss = final_ss > 0.0
    pre_headcount = float(person_weights[has_ss].sum())
    shortfall = max(0.0, SALSAC_STAGE_TARGET - pre_headcount)
    donor_pool = (employee > 0.0) & ~has_ss & (employment_income > 0.0)
    donor_pool_mass = float(person_weights[donor_pool].sum())
    uncapped_rate = shortfall / donor_pool_mass if donor_pool_mass > 0.0 else 0.0
    rate = min(SALSAC_RATE_CAP, uncapped_rate)
    draws = stable_identity_uniforms(
        person["person_id"].to_numpy(),
        seed=SALSAC_CONVERSION_SEED,
        salt=SALSAC_CONVERSION_SALT,
    )
    converted = donor_pool & (draws < rate)
    # The contribution moves into salary sacrifice, up to half the pay: the
    # FRS pay it was deducted from is before the sacrifice, so pay falls by
    # the amount moved and the sacrifice never exceeds the pay that remains.
    # What does not move stays an employee contribution (uk-data#518).
    moved = np.minimum(employee, SALSAC_MOVE_PAY_SHARE_CAP * employment_income)
    pay_share_capped = converted & (moved < employee)
    moved_amount = float(moved[converted].sum())
    retained_employee_amount = float(
        (employee[pay_share_capped] - moved[pay_share_capped]).sum()
    )
    final_ss[converted] = moved[converted]
    # Sacrificed pay leaves the contract: the engine reads employment_income
    # after sacrifice, so a converted record's pay drops by the amount that
    # becomes salary sacrifice. Without this the converted contribution left
    # taxable pay and NI unchanged while losing its net-pay relief
    # (microcosm#1069 c9).
    # The converted record's pay before the conversion stays on its carrier
    # (zero elsewhere), so the rewrite is reversible for the receipts of the
    # stages that banded on it.
    pre_conversion_pay = np.zeros(len(person), dtype=float)
    pre_conversion_pay[converted] = employment_income[converted]
    employment_income = employment_income.copy()
    employment_income[converted] = employment_income[converted] - moved[converted]
    employee[converted] = employee[converted] - moved[converted]
    # Bonus sacrifice (microcosm#1340): the bonus sacrifice that OBR's 2027-28
    # base above the threshold needs, added beside pay, since a sacrificed
    # bonus is never received (María's ruling of 2026-10-24).
    if BONUS_SACRIFICE_SOURCE not in person.columns:
        raise ValueError(
            f"Salary-sacrifice person column missing: {BONUS_SACRIFICE_SOURCE!r}."
        )
    bonus_source = pd.to_numeric(
        person[BONUS_SACRIFICE_SOURCE], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(bonus_source).all() or (bonus_source < 0.0).any():
        raise ValueError("FRS bonus amounts must be finite and non-negative.")
    bonus_resource = (
        load_bonus_sacrifice_anchor() if bonus_anchor is None else bonus_anchor
    )
    if uk_time_period(frame) != str(bonus_resource["data_year"]):
        raise ValueError(
            "The bonus-sacrifice anchor projects from "
            f"{bonus_resource['data_year']}, but the frame's period is "
            f"{uk_time_period(frame)!r}."
        )
    source_ids = person_households.map(
        pd.Series(
            source_household_ids(household, stage="The salary-sacrifice stage"),
            index=household["household_id"].to_numpy(),
        )
    ).to_numpy()
    spi_draw = draw_spi_bonuses(
        bonus=bonus_source,
        employment_income=employment_income,
        weights=person_weights,
        person_ids=person["person_id"].to_numpy(),
        source_ids=source_ids,
        donors=(
            channels.eq("frs") & ~clones.astype(bool) & ~support_copies.astype(bool)
        ).to_numpy(),
        spi=channels.ne("frs").to_numpy(),
    )
    bonus = convert_bonuses_to_bonus_sacrifice(
        bonus=spi_draw.bonus,
        employment_income=employment_income,
        salary_sacrifice=final_ss,
        weights=person_weights,
        uniforms=stable_identity_uniforms(
            person["person_id"].to_numpy(),
            seed=SALSAC_CONVERSION_SEED,
            salt=BONUS_SACRIFICE_CONVERSION_SALT,
        ),
        anchor=bonus_resource,
    )
    if (final_ss > np.maximum(employment_income, 0.0)).any():
        raise ValueError(
            "Salary-sacrifice amounts must not exceed the pay that remains."
        )
    # Net-pay contributions come out of pay: the spine caps them at pay
    # (uk_runtime.pension_coherence), and the conversion takes the same amount
    # off both, so a contribution above the pay that remains is a defect. The
    # caps hold here only while no stage between them and this one lowers
    # pay: today only hmrc_spi_income_spine and salary_sacrifice write
    # employment_income, so a new pay rewrite must cap contributions again.
    if (employee > np.maximum(employment_income, 0.0) + 1e-6).any():
        raise ValueError(
            "Employee pension contributions must not exceed the pay that remains."
        )
    if (
        bonus.sacrificed
        > SALSAC_MOVE_PAY_SHARE_CAP * np.maximum(employment_income, 0.0) + 1e-6
    ).any():
        raise ValueError("Bonus-sacrifice amounts must not exceed half the pay.")
    person[SALSAC_OUTPUT] = final_ss
    person[BONUS_SACRIFICE_OUTPUT] = bonus.sacrificed
    person["employee_pension_contributions"] = employee
    person["employment_income"] = employment_income
    person[SALSAC_PRE_CONVERSION_PAY_COLUMN] = pre_conversion_pay
    # Weighted aggregates only, banded on distinct source households; a figure
    # one to nine of them carry is "<10" (UKDS disclosure control).

    def bonus_sources(mask: np.ndarray) -> int:
        return distinct_count(source_ids, mask, person_weights)

    bonus_receipt = {
        "anchor": BONUS_SACRIFICE_ANCHOR_RESOURCE,
        "engine": dict(bonus_resource["engine"]),
        "base_year": bonus_resource["base_year"],
        "threshold": bonus_resource["threshold"],
        "target_base": bonus.target_base,
        "earnings_growth": float(bonus_resource["earnings_growth"]["factor"]),
        "weight_growth": float(bonus_resource["weight_growth"]["factor"]),
        "candidate_support": support_band(bonus_sources(bonus.candidates)),
        "participant_support": support_band(bonus_sources(bonus.participants)),
        "projected_base": small_cell_share(
            bonus.projected_base, bonus_sources(bonus.participants)
        ),
        "all_candidates_fall_short": bonus.all_candidates_fall_short,
        "pay_share_capped_support": support_band(bonus_sources(bonus.pay_share_capped)),
        "pay": "unchanged: a sacrificed bonus is never received",
        "spi_drawn_support": support_band(bonus_sources(spi_draw.recipients)),
        "spi_drawn_participant_support": support_band(
            bonus_sources(spi_draw.recipients & bonus.participants)
        ),
        "spi_cells": spi_draw.cells,
        "weights": "household weights at the stage, before calibration, per person",
    }
    post_headcount = float(person_weights[final_ss > 0.0].sum())
    converted_mass = float(person_weights[converted].sum())
    total = frame.weights_for("household").total
    mass_receipt = MassChangeRecord(
        entity="household",
        old_total=total,
        new_total=total,
        declared_factor=1.0,
        reason=SALSAC_MASS_CHANGE_REASON,
    )
    result_frame = uk_national_frame(
        person=person,
        benunit=frame.table("benunit").copy(),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=(*frame.mass_log, mass_receipt),
    )
    validate_uk_national_frame(result_frame)
    return UKSalarySacrificeResult(
        frame=result_frame,
        training_rows=int(training_mask.sum()),
        prediction_rows=int(predict_mask.sum()),
        not_asked_without_pay_rows=int(without_pay.sum()),
        not_asked_without_pay_mass=float(person_weights[without_pay.to_numpy()].sum()),
        limited_to_pay_rows=int(limited.sum()),
        limited_to_pay_mass=float(person_weights[limited].sum()),
        limited_to_pay_amount=limited_to_pay_amount,
        pre_headcount=pre_headcount,
        post_headcount=post_headcount,
        shortfall=shortfall,
        donor_pool_mass=donor_pool_mass,
        rate=rate,
        cap_bound=uncapped_rate > SALSAC_RATE_CAP,
        converted_rows=int(converted.sum()),
        converted_mass=converted_mass,
        moved_amount=moved_amount,
        pay_share_capped_rows=int(pay_share_capped.sum()),
        retained_employee_amount=retained_employee_amount,
        expected_converted_mass=rate * donor_pool_mass,
        realization_deviation=(
            (converted_mass - rate * donor_pool_mass) / (rate * donor_pool_mass)
            if rate * donor_pool_mass > 0.0
            else 0.0
        ),
        bonus_receipt=bonus_receipt,
    )


def _qrf_class():
    if QRF is not None:
        return QRF
    return import_module("microcosm.fit").QRF


def salary_sacrifice_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operations the manifest must carry verbatim."""

    return (
        (
            "verify_vendored_fact_resource",
            {
                "artifact_role": "obr_salary_sacrifice_costing_facts",
                "resource": BONUS_SACRIFICE_FACTS_RESOURCE,
                "feed_pin": "chronicle_feed.json",
                "record_sets": list(BONUS_SACRIFICE_RECORD_SETS),
                "source_vintage": (
                    "OBR supplementary forecast information on salary sacrifice, "
                    "5 February 2026"
                ),
                "mapped_build_period": 2024,
                "period_mapping": "fiscal_year_2027_base_met_by_the_engine_projection",
                "require_before_source_read": True,
                "runtime_sha256_required": True,
                "fail_on_mismatch": True,
            },
        ),
        (
            "fit_weighted_qrf",
            {
                "training_population": SALSAC_TRAINING_POPULATION,
                "target_population": SALSAC_TARGET_POPULATION,
                "predictors": list(SALSAC_PREDICTORS),
                "targets": [SALSAC_OUTPUT],
                "weights": "household_weight",
                "weight_mapping": "household_to_person",
                "seed": SALSAC_QRF_SEED,
                "n_estimators": SALSAC_QRF_ESTIMATORS,
                "clamp_minimum": 0,
                "clamp_maximum": SALSAC_AMOUNT_LIMIT,
                "preserve_asked_rows": True,
                "cache": False,
            },
        ),
        (
            "convert_donors_to_target_stock",
            {
                "resource": "salary_sacrifice_anchor.json",
                "target": int(SALSAC_STAGE_TARGET),
                "donor_pool": (
                    "employee_pension_contributions > 0 and "
                    "pension_contributions_via_salary_sacrifice == 0 and "
                    "employment_income > 0"
                ),
                "rate_cap": SALSAC_RATE_CAP,
                "move_pay_share_cap": SALSAC_MOVE_PAY_SHARE_CAP,
                "move": (
                    "employee_pension_contributions, up to move_pay_share_cap "
                    "of employment_income, to "
                    "pension_contributions_via_salary_sacrifice; the rest "
                    "stays in the source; employment_income lowered by the "
                    "amount sacrificed; the converted record's pay before the "
                    "conversion kept on salary_sacrifice_pre_conversion_pay "
                    "(zero elsewhere)"
                ),
                "seed": SALSAC_CONVERSION_SEED,
                "salt": SALSAC_CONVERSION_SALT,
                "receipt": "weighted_headcount",
                "reason": SALSAC_MASS_CHANGE_REASON,
            },
        ),
        (
            "convert_bonuses_to_bonus_sacrifice",
            {
                "resource": BONUS_SACRIFICE_ANCHOR_RESOURCE,
                "facts": BONUS_SACRIFICE_FACTS_RESOURCE,
                "source": BONUS_SACRIFICE_SOURCE,
                "output": BONUS_SACRIFICE_OUTPUT,
                "candidates": BONUS_SACRIFICE_CANDIDATES,
                "move": BONUS_SACRIFICE_MOVE,
                "move_pay_share_cap": SALSAC_MOVE_PAY_SHARE_CAP,
                "spi_bonus_draw": BONUS_SACRIFICE_SPI_DRAW,
                "spi_cell_quantiles": list(BONUS_SACRIFICE_SPI_CELL_QUANTILES),
                "spi_min_donor_sources": BONUS_SACRIFICE_SPI_MIN_DONOR_SOURCES,
                "spi_salt": BONUS_SACRIFICE_SPI_DRAW_SALT,
                "threshold": int(BONUS_SACRIFICE_THRESHOLD),
                "solve": BONUS_SACRIFICE_SOLVE,
                "seed": SALSAC_CONVERSION_SEED,
                "salt": BONUS_SACRIFICE_CONVERSION_SALT,
                "receipt": "projected_base",
            },
        ),
    )


def _assert_salary_sacrifice_stage_parameters(
    stage: SourceStageSpec,
    *,
    anchor: Mapping[str, Any],
    bonus_anchor: Mapping[str, Any] | None = None,
) -> None:
    """Bind all stage parameters closed-world; result evidence supplies arm 2."""

    _assert_closed_world_operations(stage, salary_sacrifice_operation_parameters())
    _assert_bonus_sacrifice_anchor(
        load_bonus_sacrifice_anchor() if bonus_anchor is None else bonus_anchor
    )
    hmrc = anchor.get("hmrc_anchor", {})
    derived = anchor.get("derived", {})
    checks = {
        "hmrc_anchor.total_users": (hmrc.get("total_users"), SALSAC_HMRC_ANCHOR),
        "hmrc_anchor.above_2000": (
            hmrc.get("above_2000"),
            SALSAC_ABOVE_2000_ANCHOR,
        ),
        "hmrc_anchor.below_2000": (
            hmrc.get("below_2000"),
            SALSAC_BELOW_2000_ANCHOR,
        ),
        "derived.stage_target": (derived.get("stage_target"), SALSAC_STAGE_TARGET),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"Salary-sacrifice resource {label} drifted: expected "
                f"{expected!r}, got {actual!r}."
            )
    ratio = float(derived.get("staging_ratio", np.nan))
    if not np.isclose(ratio, SALSAC_STAGING_RATIO, rtol=0.0, atol=1e-12):
        raise ValueError("Salary-sacrifice resource staging ratio drifted.")


def _assert_bonus_sacrifice_anchor(anchor: Mapping[str, Any]) -> None:
    """The anchor's OBR base is the vendored row's, at the declared threshold.

    The engine factors follow the lock; an engine-tier test holds them to the
    installed policyengine-uk (``tools/pin_uk_bonus_sacrifice_anchor.py``).
    """

    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    source = anchor.get("source", {})
    rows = [
        row
        for row in vendored_rows(BONUS_SACRIFICE_FACTS_RESOURCE)
        if (row.get("layout") or {}).get("record_set_id") == source.get("record_set_id")
        and (row.get("layout") or {}).get("groupby_value_id")
        == source.get("groupby_value_id")
    ]
    if len(rows) != 1:
        raise ValueError(
            f"The bonus-sacrifice anchor names no single vendored OBR row: {source!r}."
        )
    if float(rows[0]["value"]) != float(anchor.get("obr_base_above_threshold", -1)):
        raise ValueError(
            "The bonus-sacrifice anchor's OBR base differs from the vendored row."
        )
    if float(anchor.get("threshold", -1)) != BONUS_SACRIFICE_THRESHOLD:
        raise ValueError("The bonus-sacrifice anchor's threshold drifted.")
