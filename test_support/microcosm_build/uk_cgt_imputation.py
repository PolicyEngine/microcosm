# ruff: noqa: F401
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import cgt_imputation
from microcosm.build.uk_runtime.advani_summers import (
    ADVANI_SUMMERS_RESOURCE,
    ADVANI_SUMMERS_VINTAGE,
    CGT_PRIOR_PERCENTILE_COLUMNS,
    advani_summers_band_index,
    advani_summers_knots,
    advani_summers_rows,
    exempt_range_quantiles,
    load_advani_summers_distribution,
    quantile_for_amount,
)
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_AGE_GROUP_LOWER_BOUNDS,
    UK_CGT_REGION_GROUP_LABELS,
    UK_CGT_REGION_GROUPS,
    UK_CGT_REMAINDER_POLICY,
    UK_CGT_SPINE_MASS_CONSERVATION_REASON,
    UK_CGT_SPINE_STAGE_NAME,
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
    UKCGTPolicyParameters,
    _band_plans,
    _CellPlan,
    _draw_plan_amounts,
    _joint_plans,
    _map_remainder_amounts,
    _pareto_quantile,
    _pareto_stratum_means,
    _rake_allocation_targets,
    _tilt_group,
    _truncated_exponential_quantile,
    impute_uk_capital_gains,
    impute_uk_capital_gains_with_report,
    summarize_uk_cgt_imputation,
    uk_cgt_component_sum_income,
    uk_cgt_spine_stage_transform,
    uk_cgt_taxable_income_proxy,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
    HMRCCapitalGainsBandTotal,
    HMRCCapitalGainsCell,
    HMRCCapitalGainsIncomeTotal,
    HMRCCapitalGainsJointDistribution,
    HMRCCapitalGainsSourceProvenance,
    load_hmrc_cgt_conditioning_facts,
    load_hmrc_cgt_joint_distribution,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import Frame

PARAMETERS = UKCGTPolicyParameters(
    personal_allowance=12_570.0,
    personal_allowance_taper_threshold=100_000.0,
    personal_allowance_taper_rate=0.5,
    annual_exempt_amount=6_000.0,
    instant="2023-06-01",
    source="test",
)


def _distribution(
    *, cell_people: float = 1_000.0, suppress_top_low_income: bool = False
) -> HMRCCapitalGainsJointDistribution:
    """A synthetic joint distribution with in-band cell means."""
    bounds = HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    uppers = (*bounds[1:], None)
    cells = []
    band_totals = []
    column_people = dict.fromkeys(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS, 0.0)
    column_gains = dict.fromkeys(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS, 0.0)
    for lower, upper in zip(bounds, uppers, strict=True):
        if upper is None:
            mean = lower * 2.0
        else:
            mean = lower + (upper - lower) * 0.4
        band_people = 0.0
        band_gains = 0.0
        for income_lower in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS:
            suppressed = (
                suppress_top_low_income
                and upper is None
                and income_lower == HMRC_CGT_INCOME_BAND_LOWER_BOUNDS[0]
            )
            # Real suppression withholds the count and keeps the amount in
            # all but one published cell, so the fixture matches that shape.
            people = None if suppressed else cell_people
            gains = cell_people * mean
            cells.append(
                HMRCCapitalGainsCell(
                    gain_lower_bound=lower,
                    income_lower_bound=income_lower,
                    individuals=people,
                    gains=gains,
                )
            )
            band_people += cell_people
            band_gains += cell_people * mean
            column_people[income_lower] += cell_people
            column_gains[income_lower] += cell_people * mean
        band_totals.append(
            HMRCCapitalGainsBandTotal(
                gain_lower_bound=lower,
                individuals=band_people,
                gains=band_gains,
            )
        )
    income_totals = tuple(
        HMRCCapitalGainsIncomeTotal(
            income_lower_bound=income_lower,
            individuals=column_people[income_lower],
            gains=column_gains[income_lower],
        )
        for income_lower in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
    )
    return HMRCCapitalGainsJointDistribution(
        cells=tuple(cells),
        band_totals=tuple(band_totals),
        income_totals=income_totals,
        source=HMRCCapitalGainsSourceProvenance(
            resource="synthetic.json",
            resource_sha256="synthetic",
            source_commit="synthetic",
            record_set_prefix="synthetic.",
            source_file="synthetic.ods",
            source_sha256="synthetic",
            source_vintage="2024-25",
            build_period="2024",
        ),
        total_individuals=sum(t.individuals for t in band_totals),
        total_gains=sum(t.gains for t in band_totals),
    )


def _frame(
    person_rows: int,
    *,
    gains,
    incomes,
    ages=None,
    regions=None,
    weights=None,
) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.arange(person_rows, dtype="int64"),
            "person_household_id": np.arange(person_rows, dtype="int64"),
            "person_benunit_id": np.arange(person_rows, dtype="int64"),
            "capital_gains": np.asarray(gains, dtype=float),
            "employment_income": np.asarray(incomes, dtype=float),
            "age": (
                np.full(person_rows, 45, dtype="int64")
                if ages is None
                else np.asarray(ages, dtype="int64")
            ),
        }
    )
    for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS:
        if column not in person.columns:
            person[column] = 0.0
    household = pd.DataFrame(
        {
            "household_id": np.arange(person_rows, dtype="int64"),
            "household_weight": (
                np.full(person_rows, 100.0)
                if weights is None
                else np.asarray(weights, dtype=float)
            ),
            "region": (
                np.full(person_rows, "LONDON", dtype=object)
                if regions is None
                else np.asarray(regions, dtype=object)
            ),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(person_rows, dtype="int64")})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )


def _spine_stage():
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()[UK_CGT_SPINE_STAGE_NAME]


def _real_distribution() -> HMRCCapitalGainsJointDistribution:
    """The published 2024-25 joint, from the committed vendored rows.

    The rows are copied verbatim from the pinned Chronicle feed, so CI
    exercises the real surface without hand-copied values.
    """
    return load_hmrc_cgt_joint_distribution()


__all__ = [name for name in globals() if not name.startswith("__")]
