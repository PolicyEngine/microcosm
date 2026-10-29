"""Income tax by HMRC rate category, for salary-sacrifice relief (microcosm#1095).

HMRC's private pension statistics (Table 6.1) report the income tax relief on
salary-sacrificed pension contributions by marginal rate: basic, higher and
additional. HMRC applies income tax rates to employees' pay, so a person's
relief is the rise in tax on their earned income when the sacrifice is paid as
salary instead, under their rest-of-UK or Scottish schedule, and a contribution
that straddles a band boundary is relieved partly at each rate (uk-data#533).

The categories group a schedule's brackets: a bracket taxed below 30% is basic
(HMRC counts the Scottish starter, basic and intermediate rates there), the
schedule's top rate is additional where a higher rate sits below it (the
rest-of-UK additional rate and the Scottish top rate), and every other bracket
is higher (the Scottish higher and advanced rates, which span the rest-of-UK
higher-rate income range). Where the
returned pay tapers the personal allowance, the allowance it withdraws raises
taxable income too, so that tax lands in the brackets it reaches, as
uk-data#533 reads it. From 2.134.3 policyengine-uk starts the Scottish top
rate at the statutory GBP 125,140 of taxable income for 2024-25 to 2026-27
(pe-uk#2437, closing pe-uk#2130). Before that it used GBP 112,570 and filed some
advanced-rate relief as additional.

A Gift Aid donor's bands are extended by the grossed-up gift (ITA 2007
s.414(2)(b)), as policyengine-uk's earned_income_tax extends them from 2.128.0
(pe-uk#2271): the rest-of-UK higher and additional limits move up by the gift,
and so does every Scottish limit above the starter band. Pay returned to a
donor is therefore relieved at the rates of the extended bands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

UK_RATE_CLASSES = ("basic", "higher", "additional")
#: A bracket taxed below this rate is basic rate.
UK_BASIC_RATE_CLASS_CEILING = 0.3
#: The declared allocation a salary-sacrifice relief band binds; the measure
#: refuses a binding that declares anything else.
UK_RATE_CLASS_ALLOCATION: Mapping[str, object] = {
    "kind": "income_tax_rate_class",
    "taxable_income": "earned_taxable_income",
    "schedules": {
        "rest_of_uk": "gov.hmrc.income_tax.rates.uk",
        "scotland": "gov.hmrc.income_tax.rates.scotland.rates",
    },
    "schedule_flag": "pays_scottish_income_tax",
    "basic_below_rate": UK_BASIC_RATE_CLASS_CEILING,
    "additional": "the schedule's top rate",
    "gift_aid_extends_bands": "gift_aid_grossed_up",
}


#: The engine variable the allocation re-implements: the schedule applied to
#: earned taxable income. A person's classes must add up to its rise within
#: the tolerance, or the measure refuses.
UK_RATE_CLASS_ENGINE_VARIABLE = "earned_income_tax"
UK_RATE_CLASS_ENGINE_TOLERANCE_GBP = 1.0
#: The engine computes in float32, whose step grows with the amount (GBP 1 at
#: GBP 10 million, GBP 8 at GBP 100 million). The tolerance adds this many
#: steps at the person's own largest amount to the GBP 1, so rounding alone
#: never refuses a schedule that still matches.
UK_RATE_CLASS_ENGINE_FLOAT32_STEPS = 4


def rate_class_engine_tolerance(*amounts: np.ndarray) -> np.ndarray:
    """Each person's tolerance against the engine variable, in GBP.

    GBP 1 plus ``UK_RATE_CLASS_ENGINE_FLOAT32_STEPS`` float32 steps at the
    largest of the person's ``amounts`` (the incomes and taxes the comparison
    reads). Below about GBP 2 million the steps add less than GBP 1.
    """

    largest = np.maximum.reduce(
        [np.abs(np.asarray(amount, dtype=float)) for amount in amounts]
    )
    step = np.spacing(largest.astype(np.float32)).astype(float)
    return (
        UK_RATE_CLASS_ENGINE_TOLERANCE_GBP + UK_RATE_CLASS_ENGINE_FLOAT32_STEPS * step
    )


def gift_aid_extended_brackets(schedule: str, brackets: int) -> range:
    """The in-force brackets whose lower limit a donor's gross gift moves up.

    policyengine-uk's ``calc_income_tax_with_gift_aid``, as earned_income_tax
    calls it: the rest-of-UK higher and additional limits (not the model's
    extra bracket above them), and every Scottish limit above the starter band
    (above the basic band in the three-rate schedule before 2018).
    """

    if schedule == "rest_of_uk":
        return range(1, min(3, brackets))
    if schedule == "scotland":
        return range(1 if brackets == 3 else 2, brackets)
    raise ValueError(f"unknown income tax schedule {schedule!r}")


def tax_by_rate_class(
    income: np.ndarray,
    thresholds: Sequence[float],
    rates: Sequence[float | None],
    *,
    gross_gift: np.ndarray | None = None,
    schedule: str | None = None,
) -> dict[str, np.ndarray]:
    """Tax on ``income`` within each of HMRC's rate categories.

    The categories sum to the schedule's tax on ``income``. A bracket whose
    rate is ``None`` (not in force at the instant) is skipped. With
    ``gross_gift`` the ``schedule``'s extended limits move up by each person's
    gift, as the engine moves them: the income between a limit and the limit
    plus the gift is taxed at the rate below it instead.
    """

    values = np.asarray(income, dtype=float)
    brackets = [
        (float(threshold), float(rate))
        for threshold, rate in zip(thresholds, rates, strict=True)
        if rate is not None
    ]
    if not brackets:
        raise ValueError("an income tax schedule needs at least one bracket.")
    # The top rate is the additional rate only where a higher rate sits below
    # it; a schedule with one rate above the basic class has no additional rate.
    above_basic = {rate for _, rate in brackets if rate >= UK_BASIC_RATE_CLASS_CEILING}
    top = brackets[-1][1] if len(above_basic) > 1 else None
    uppers = [threshold for threshold, _ in brackets[1:]] + [np.inf]

    def category(rate: float) -> str:
        if rate < UK_BASIC_RATE_CLASS_CEILING:
            return "basic"
        return "additional" if rate == top else "higher"

    classes = {name: np.zeros_like(values) for name in UK_RATE_CLASSES}
    for (lower, rate), upper in zip(brackets, uppers, strict=True):
        name = category(rate)
        classes[name] = classes[name] + rate * np.clip(
            values - lower, 0.0, upper - lower
        )
    if gross_gift is not None:
        if schedule is None:
            raise ValueError("a Gift Aid extension needs the schedule it extends.")
        gift = np.asarray(gross_gift, dtype=float)
        # Each moved limit taxes the slice above it, up to the gift, at the
        # rate below it. Summed over the limits this is exact even when a gift
        # spans several bands, as in the engine.
        for index in gift_aid_extended_brackets(schedule, len(brackets)):
            lower, rate = brackets[index]
            below = brackets[index - 1][1]
            overlap = np.clip(values - lower, 0.0, gift)
            classes[category(rate)] = classes[category(rate)] - rate * overlap
            classes[category(below)] = classes[category(below)] + below * overlap
    return classes


def relief_by_rate_class(
    *,
    baseline_income: np.ndarray,
    counterfactual_income: np.ndarray,
    scottish: np.ndarray,
    schedules: Mapping[str, tuple[Sequence[float], Sequence[float | None]]],
    gross_gift: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Each person's rise in tax on earned income, by rate category.

    ``schedules`` maps ``rest_of_uk`` and ``scotland`` to their (thresholds,
    rates); ``scottish`` picks each person's schedule. ``gross_gift`` is each
    person's grossed-up Gift Aid, which extends their bands in both the
    baseline and the counterfactual.
    """

    scottish = np.asarray(scottish, dtype=bool)
    relief = {name: np.zeros(scottish.shape, dtype=float) for name in UK_RATE_CLASSES}
    for schedule, members in (("rest_of_uk", ~scottish), ("scotland", scottish)):
        thresholds, rates = schedules[schedule]
        after = tax_by_rate_class(
            counterfactual_income,
            thresholds,
            rates,
            gross_gift=gross_gift,
            schedule=schedule,
        )
        before = tax_by_rate_class(
            baseline_income,
            thresholds,
            rates,
            gross_gift=gross_gift,
            schedule=schedule,
        )
        for name in UK_RATE_CLASSES:
            relief[name] = relief[name] + np.where(
                members, after[name] - before[name], 0.0
            )
    return relief
