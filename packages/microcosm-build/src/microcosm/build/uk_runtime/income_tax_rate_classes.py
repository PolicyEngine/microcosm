"""Income tax by HMRC rate category, for salary-sacrifice relief (microcosm#1095).

HMRC's private pension statistics (Table 6.1) report the income tax relief on
salary-sacrificed pension contributions by marginal rate: basic, higher and
additional. HMRC applies income tax rates to employees' pay, so a person's
relief is the rise in tax on their earned income when the sacrifice is paid as
salary instead, under their rest-of-UK or Scottish schedule, and a contribution
that straddles a band boundary is relieved partly at each rate (uk-data#533).

The categories group a schedule's brackets: a bracket taxed below 30% is basic
(HMRC counts the Scottish starter, basic and intermediate rates there), the
schedule's top rate is additional (the rest-of-UK additional rate and the
Scottish top rate), and every other bracket is higher (the Scottish higher and
advanced rates, which span the rest-of-UK higher-rate income range). Where the
returned pay tapers the personal allowance, the allowance it withdraws raises
taxable income too, so that tax lands in the brackets it reaches, as
uk-data#533 reads it. policyengine-uk puts the Scottish top-rate threshold at
GBP 112,570 rather than the statutory GBP 125,140 (pe-uk#2130), so it files
some advanced-rate relief as additional.
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
}


def tax_by_rate_class(
    income: np.ndarray,
    thresholds: Sequence[float],
    rates: Sequence[float | None],
) -> dict[str, np.ndarray]:
    """Tax on ``income`` within each of HMRC's rate categories.

    The categories sum to the schedule's tax on ``income``. A bracket whose
    rate is ``None`` (not in force at the instant) is skipped.
    """

    values = np.asarray(income, dtype=float)
    brackets = [
        (float(threshold), float(rate))
        for threshold, rate in zip(thresholds, rates, strict=True)
        if rate is not None
    ]
    if not brackets:
        raise ValueError("an income tax schedule needs at least one bracket.")
    top = brackets[-1][1]
    uppers = [threshold for threshold, _ in brackets[1:]] + [np.inf]
    classes = {name: np.zeros_like(values) for name in UK_RATE_CLASSES}
    for (lower, rate), upper in zip(brackets, uppers, strict=True):
        if rate < UK_BASIC_RATE_CLASS_CEILING:
            name = "basic"
        elif rate == top:
            name = "additional"
        else:
            name = "higher"
        classes[name] = classes[name] + rate * np.clip(
            values - lower, 0.0, upper - lower
        )
    return classes


def relief_by_rate_class(
    *,
    baseline_income: np.ndarray,
    counterfactual_income: np.ndarray,
    scottish: np.ndarray,
    schedules: Mapping[str, tuple[Sequence[float], Sequence[float | None]]],
) -> dict[str, np.ndarray]:
    """Each person's rise in tax on earned income, by rate category.

    ``schedules`` maps ``rest_of_uk`` and ``scotland`` to their (thresholds,
    rates); ``scottish`` picks each person's schedule.
    """

    scottish = np.asarray(scottish, dtype=bool)
    relief = {name: np.zeros(scottish.shape, dtype=float) for name in UK_RATE_CLASSES}
    for schedule, members in (("rest_of_uk", ~scottish), ("scotland", scottish)):
        thresholds, rates = schedules[schedule]
        after = tax_by_rate_class(counterfactual_income, thresholds, rates)
        before = tax_by_rate_class(baseline_income, thresholds, rates)
        for name in UK_RATE_CLASSES:
            relief[name] = relief[name] + np.where(
                members, after[name] - before[name], 0.0
            )
    return relief
