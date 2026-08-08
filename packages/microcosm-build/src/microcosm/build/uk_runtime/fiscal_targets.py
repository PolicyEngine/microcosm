"""UK fiscal target facts, declared natively in Microcosm.

The UK national calibration surface is currently ingested from
policyengine-uk-data via :func:`microcosm.calibrate.specs_from_pe_surface`, whose
docstring calls that "the transitional path from harness-extracted surfaces to
declared facts". This module starts the declared-fact side for the UK, following
``us_runtime.fiscal_targets``.

The first facts declared here are capital gains, because their absence is
measurable. The published ``populace-uk`` release (``populace-uk-2023-dd68c73``)
carries 149 targets and none of them constrain capital gains, so the calibrated
weights leave the gains distribution unanchored: reading ``capital_gains``
weighted by ``household_weight`` straight out of ``populace_uk_2023.h5`` gives
1.47m CGT taxpayers holding £96.0bn of gains at a £65,374 mean, against HMRC's
378k, £65.9bn and ~£174,000. The taxpayer count is 3.9x the administrative
figure and the mean gain is 62% below it — the imputation spreads gains across
far too many households in amounts that are individually far too small.

That is distributional error, not a level shift, so it survives any
revenue-side correction. It matters for any CGT analysis: on an uncalibrated
baseline the share of people affected by a rate reform comes out roughly three
times too high.

Two targets are declared:

``hmrc/capital_gains_total``
    Total chargeable gains of CGT taxpayers.

``hmrc/cgt_taxpayers``
    Number of CGT taxpayers.

Both are person-entity, following the UK convention established by
``uk_runtime.hmrc_calibration``: the measures are person-level, so the
constraint rows live on the person table while the calibrated weights stay on
the household table via the frame's household ``Weights``. Both therefore need
prepared columns on the person frame — the registry refuses callables so that
it can serialize, and count-like facts are documented to use prepared
indicator/count columns. See ``UK_CGT_REQUIRED_COLUMNS``, and
``uk_runtime.cgt_calibration`` for the materialization that prepares them.

Open question for reviewers: ``us_runtime.fiscal_targets`` has since moved to
value-free references whose values arrive from an external Ledger artifact at
build time. The values here are inline with citations, which is what the
registry's own ``TargetSpec`` contract describes ("a value, optionally a
standard error, and always a citation"). If the UK should follow the US onto
Ledger-sourced values instead, these specs are the shape to migrate.
"""

from __future__ import annotations

from microcosm.build.gates import TargetCoverageRequirement
from microcosm.calibrate import TargetRegistry, TargetSpec

__all__ = [
    "UK_CGT_REQUIRED_COLUMNS",
    "UK_CGT_TARGET_COVERAGE_REQUIREMENTS",
    "UK_CGT_TARGET_SPECS",
    "UK_FISCAL_TARGET_REGISTRY",
]

#: HMRC Capital Gains Tax statistics, tax year 2023-24 (published August 2025).
_HMRC_CGT_SOURCE = (
    "HMRC Capital Gains Tax statistics, tax year 2023-24, table 1: "
    "https://www.gov.uk/government/statistics/capital-gains-tax-statistics"
)

#: The reference period of the declared facts. HMRC's 2023-24 outturn is the
#: latest published tax year; build-side aging carries it to forecast years.
_HMRC_CGT_PERIOD = 2023

#: Prepared person columns the frame must expose for these facts to compile.
#:
#: ``uk_cgt_measure_gains_amount`` is each person's chargeable gains, zeroed on
#: people who are not CGT taxpayers.
#: ``uk_cgt_measure_taxpayer_count`` is the 0/1 indicator for people whose
#: gains exceed the annual exempt amount **in force for the period** — the AEA moved
#: £12,300 -> £6,000 -> £3,000 across 2022-23 to 2024-25, so a fixed threshold
#: would silently mean different things in different years.
UK_CGT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "uk_cgt_measure_gains_amount",
    "uk_cgt_measure_taxpayer_count",
)

UK_CGT_TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec(
        name="hmrc/capital_gains_total",
        entity="person",
        measure="uk_cgt_measure_gains_amount",
        value=65_900_000_000.0,
        period=_HMRC_CGT_PERIOD,
        source=_HMRC_CGT_SOURCE,
        family="hmrc",
        notes=(
            "Total chargeable gains of CGT taxpayers in 2023-24. Declared "
            "unsigned: the fact is a positive aggregate. Note that a measure "
            "carrying gains net of losses can be negative on individual "
            "person records even though the total is positive, so a build "
            "aggregating losses into this column should confirm the sign "
            "handling it wants."
        ),
    ),
    TargetSpec(
        name="hmrc/cgt_taxpayers",
        entity="person",
        measure="uk_cgt_measure_taxpayer_count",
        value=378_000.0,
        period=_HMRC_CGT_PERIOD,
        source=_HMRC_CGT_SOURCE,
        family="hmrc",
        notes=(
            "Number of CGT taxpayers in 2023-24. The measure column is a "
            "person-level indicator for people above the annual exempt amount in force for the period; "
            "the AEA is policy-dependent and must track the period rather "
            "than being hard-coded."
        ),
    ),
)

UK_CGT_TARGET_COVERAGE_REQUIREMENTS: tuple[TargetCoverageRequirement, ...] = (
    TargetCoverageRequirement(
        requirement_id="uk_capital_gains",
        label="HMRC capital gains totals and taxpayer counts",
        accepted_names=(
            "hmrc/capital_gains_total",
            "hmrc/cgt_taxpayers",
        ),
        min_matches=2,
        notes=(
            "Without both facts the gains distribution is unanchored: the "
            "published populace-uk release has 1.47m CGT taxpayers against "
            "HMRC's 378k. A revenue-side target constrains what CGT raises, "
            "not how gains are spread across households, so it does not "
            "substitute for these."
        ),
    ),
)

UK_FISCAL_TARGET_REGISTRY = TargetRegistry(UK_CGT_TARGET_SPECS, country="uk")
