"""Builders shared by the CGT support-split tests (microcosm#1045).

Everything here runs without a country engine: the policy parameters are
constructed directly, the Table 3 joint is a small synthetic
:class:`HMRCCapitalGainsJointDistribution`, and the frames are hand-built
national frames with the columns the split reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_INVESTABLE_WEALTH_COLUMNS,
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
    UKCGTPolicyParameters,
)
from microcosm.build.uk_runtime.cgt_support import (
    CGT_SUPPORT_COPIES_COLUMN,
    CGT_SUPPORT_SPLIT_STAGE_NAME,
    HOUSEHOLD_IS_CGT_SUPPORT_COPY,
    cgt_support_split_operation_parameters,
)
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_CONDITIONING_RESOURCE,
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
    HMRCCapitalGainsBandTotal,
    HMRCCapitalGainsCell,
    HMRCCapitalGainsIncomeTotal,
    HMRCCapitalGainsJointDistribution,
    HMRCCapitalGainsSourceProvenance,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import Frame

PARAMETERS = UKCGTPolicyParameters(
    personal_allowance=12_570.0,
    personal_allowance_taper_threshold=100_000.0,
    personal_allowance_taper_rate=0.5,
    annual_exempt_amount=3_000.0,
    instant="2024-06-01",
    source="test",
)

#: An employment income whose taxable-income proxy under ``PARAMETERS`` lands
#: in each Table 3 band (proxy = income - 12,570 below the taper; the
#: allowance is gone from 125,140).
BAND_INCOMES: dict[int, float] = {
    0: 20_000.0,
    37_700: 60_000.0,
    50_000: 70_000.0,
    100_000: 115_000.0,
    125_140: 150_000.0,
    200_000: 250_000.0,
}


def support_distribution(
    top_cells: Mapping[int, Mapping[int, float | None]] | None = None,
    *,
    low_band_people: float = 1_000.0,
) -> HMRCCapitalGainsJointDistribution:
    """A small synthetic Table 3 joint in the loader's shape.

    ``top_cells`` maps an income band lower bound to ``{gain band lower
    bound: individuals}``; ``None`` marks a suppressed count, as the loader
    represents it. Every income band also gets one filled cell in the lowest
    band of gains, which the support rule must ignore.
    """

    top_cells = {} if top_cells is None else top_cells
    cells: list[HMRCCapitalGainsCell] = []
    for income_lower in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS:
        cells.append(
            HMRCCapitalGainsCell(
                gain_lower_bound=HMRC_CGT_GAIN_BAND_LOWER_BOUNDS[0],
                income_lower_bound=income_lower,
                individuals=low_band_people,
                gains=low_band_people * 5_000.0,
            )
        )
        for gain_lower, individuals in top_cells.get(income_lower, {}).items():
            cells.append(
                HMRCCapitalGainsCell(
                    gain_lower_bound=gain_lower,
                    income_lower_bound=income_lower,
                    individuals=individuals,
                    gains=(0.0 if individuals is None else individuals) * gain_lower,
                )
            )
    band_totals = tuple(
        HMRCCapitalGainsBandTotal(
            gain_lower_bound=gain_lower,
            individuals=sum(
                cell.individuals or 0.0
                for cell in cells
                if cell.gain_lower_bound == gain_lower
            ),
            gains=sum(
                cell.gains or 0.0
                for cell in cells
                if cell.gain_lower_bound == gain_lower
            ),
        )
        for gain_lower in sorted({cell.gain_lower_bound for cell in cells})
    )
    income_totals = tuple(
        HMRCCapitalGainsIncomeTotal(
            income_lower_bound=income_lower,
            individuals=sum(
                cell.individuals or 0.0
                for cell in cells
                if cell.income_lower_bound == income_lower
            ),
            gains=sum(
                cell.gains or 0.0
                for cell in cells
                if cell.income_lower_bound == income_lower
            ),
        )
        for income_lower in HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
    )
    return HMRCCapitalGainsJointDistribution(
        cells=tuple(cells),
        band_totals=band_totals,
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
        total_individuals=sum(total.individuals or 0.0 for total in band_totals),
        total_gains=sum(total.gains for total in band_totals),
    )


def support_tables(
    *,
    weights: Sequence[float],
    wealth: Sequence[float],
    incomes: Sequence[float | Sequence[float]],
    ages: Sequence[Sequence[int]] | None = None,
    benunits: Sequence[Sequence[int]] | None = None,
    channels: Sequence[str] | None = None,
    person_order: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Person, benunit and household tables plus household weights.

    One household per entry, ids 1..n in order. ``ages`` lists each
    household's member ages (default one 45-year-old); ``benunits`` lists
    each member's benefit-unit slot within its household (default all in
    one); a scalar income goes to the oldest member (lowest person id on a
    tie) and a sequence gives one income per member; ``wealth`` lands in the
    first investable-wealth column with the rest zero; ``channels`` adds the
    ``household_support_channel`` column when given. ``person_order``
    permutes the person rows. Extra non-id columns on every entity let tests
    check that copies carry every value unchanged.
    """

    n = len(weights)
    if not (len(wealth) == len(incomes) == n):
        raise ValueError(
            "support_tables needs one weight, wealth and income per household."
        )
    ages = [(45,)] * n if ages is None else list(ages)
    benunits = (
        [tuple(0 for _ in members) for members in ages]
        if benunits is None
        else list(benunits)
    )
    person_rows: list[dict[str, object]] = []
    benunit_rows: list[dict[str, object]] = []
    next_person = 1
    next_benunit = 1
    for index in range(n):
        household_id = index + 1
        member_ages = tuple(ages[index])
        member_slots = tuple(benunits[index])
        if len(member_slots) != len(member_ages):
            raise ValueError("support_tables benunit slots must match the member ages.")
        slot_ids: dict[int, int] = {}
        for slot in sorted(set(member_slots)):
            slot_ids[slot] = next_benunit
            benunit_rows.append(
                {
                    "benunit_id": next_benunit,
                    "benunit_rent": 100.0 * next_benunit,
                    "benunit_tenure": "rent" if next_benunit % 2 else "own",
                }
            )
            next_benunit += 1
        income = incomes[index]
        if isinstance(income, int | float):
            oldest = int(np.argmax(np.asarray(member_ages)))
            member_incomes = [
                float(income) if position == oldest else 0.0
                for position in range(len(member_ages))
            ]
        else:
            member_incomes = [float(value) for value in income]
            if len(member_incomes) != len(member_ages):
                raise ValueError("support_tables incomes must match the member ages.")
        for age, slot, member_income in zip(
            member_ages, member_slots, member_incomes, strict=True
        ):
            person_rows.append(
                {
                    "person_id": next_person,
                    "person_household_id": household_id,
                    "person_benunit_id": slot_ids[slot],
                    "age": int(age),
                    "employment_income": member_income,
                    "capital_gains": 0.0,
                    "person_note": f"p{next_person}",
                }
            )
            next_person += 1
    person = pd.DataFrame(person_rows)
    for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS:
        if column not in person.columns:
            person[column] = 0.0
    for column in ("person_id", "person_household_id", "person_benunit_id", "age"):
        person[column] = person[column].astype("int64")
    if person_order is not None:
        person = person.iloc[list(person_order)].reset_index(drop=True)
    benunit = pd.DataFrame(benunit_rows)
    benunit["benunit_id"] = benunit["benunit_id"].astype("int64")
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1, dtype="int64"),
            "region": np.asarray(
                ["LONDON" if index % 2 else "WALES" for index in range(n)], dtype=object
            ),
            "council_tax": np.linspace(900.0, 2_400.0, n),
        }
    )
    for column in UK_CGT_INVESTABLE_WEALTH_COLUMNS:
        household[column] = 0.0
    household[UK_CGT_INVESTABLE_WEALTH_COLUMNS[0]] = np.asarray(wealth, dtype=float)
    if channels is not None:
        household["household_support_channel"] = np.asarray(channels, dtype=object)
    return person, benunit, household, np.asarray(weights, dtype=float)


def support_frame(
    *,
    weights: Sequence[float],
    wealth: Sequence[float],
    incomes: Sequence[float | Sequence[float]],
    ages: Sequence[Sequence[int]] | None = None,
    benunits: Sequence[Sequence[int]] | None = None,
    channels: Sequence[str] | None = None,
    person_order: Sequence[int] | None = None,
    time_period: str = "2024",
) -> Frame:
    """A national frame from :func:`support_tables`."""

    person, benunit, household, household_weights = support_tables(
        weights=weights,
        wealth=wealth,
        incomes=incomes,
        ages=ages,
        benunits=benunits,
        channels=channels,
        person_order=person_order,
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=household_weights,
        time_period=time_period,
    )


def support_artifacts() -> tuple[dict[str, object], ...]:
    """The two reference artifacts the stage block declares."""

    return (
        {
            "role": "cgt_conditioning_facts",
            "kind": "public_aggregate_reference",
            "resource": HMRC_CGT_CONDITIONING_RESOURCE,
            "format": "json",
            "runtime_sha256_required": True,
        },
        {
            "role": "policy_parameters",
            "kind": "versioned_parameter_tree",
            "dependency": "policyengine-uk>=2.98.0 via microcosm-build[uk]",
            "parameters": [
                "gov.hmrc.income_tax.allowances.personal_allowance.amount",
                "gov.hmrc.income_tax.allowances.personal_allowance.maximum_ANI",
                "gov.hmrc.income_tax.allowances.personal_allowance.reduction_rate",
            ],
            "instant_rule": (
                "raw dated parameter files evaluated at 1 June of the build "
                "period's tax year"
            ),
            "runtime_sha256_required": False,
        },
    )


def support_stage(
    *,
    stage_name: str = CGT_SUPPORT_SPLIT_STAGE_NAME,
    grain: str = "household",
    outputs: Sequence[str] = (HOUSEHOLD_IS_CGT_SUPPORT_COPY, CGT_SUPPORT_COPIES_COLUMN),
    rewrites: Sequence[str] = (),
    artifacts: Sequence[Mapping[str, object]] | None = None,
) -> SourceStageSpec:
    """A stage spec restating the code's operation dictionary verbatim."""

    operations = tuple(
        SourceOperationSpec(kind=kind, parameters=dict(parameters))
        for kind, parameters in cgt_support_split_operation_parameters()
    )
    return SourceStageSpec(
        stage=stage_name,
        survey="synthetic",
        source="synthetic",
        grain=grain,
        artifacts=tuple(support_artifacts() if artifacts is None else artifacts),
        operations=operations,
        outputs=tuple(outputs),
        rewrites=tuple(rewrites),
        notes="synthetic",
    )


def drift(stage: SourceStageSpec, parameter: str, *, operation_index: int = 0):
    """``stage`` with one operation parameter replaced by a sentinel."""

    operations = list(stage.operations)
    operation = operations[operation_index]
    operations[operation_index] = SourceOperationSpec(
        kind=operation.kind,
        parameters={**operation.parameters, parameter: "__drift__"},
    )
    return replace(stage, operations=tuple(operations))
