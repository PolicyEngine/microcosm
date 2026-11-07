"""UK fiscal target coverage declarations.

UK national targets compile at build time from ``uk/target_references.json``,
the Ledger target references generated from the national rows of
``uk/uk_population_targets.json``. This module holds no target values. It
declares the coverage requirement for the national capital gains facts,
following ``us_runtime.fiscal_targets``, and the person columns that
``uk_runtime.cgt_calibration`` prepares for its own capital gains frame.

Capital gains carry a coverage requirement because their absence is
measurable. The published ``populace-uk`` release (``populace-uk-2023-dd68c73``)
carries 149 targets and none of them constrain capital gains, so the calibrated
weights leave the gains distribution unanchored: reading ``capital_gains``
weighted by ``household_weight`` straight out of ``populace_uk_2023.h5`` gives
1.47m CGT taxpayers holding £96.0bn of gains at a £65,374 mean, against HMRC's
2023-24 figures of 378k, £65.9bn and ~£174,000. The taxpayer count is 3.9x the
administrative figure and the mean gain is 62% below it — the imputation
spreads gains across far too many households in amounts that are individually
far too small.

That is distributional error, not a level shift, so it survives any
revenue-side correction. It matters for any CGT analysis: on an uncalibrated
baseline the share of people affected by a rate reform comes out roughly three
times too high.

The capital gains values come from HMRC ``cgt_statistics`` Ledger facts. The
compiled facts measure the prepared person columns ``hmrc/capital_gains_total``,
``hmrc/cgt_taxpayers`` and ``hmrc/cgt_liability``, which
``uk_runtime.ledger_targets.materialize_uk_ledger_targets`` builds from the bindings
in ``uk/uk_population_targets.json``. They follow the UK convention of
``uk_runtime.hmrc_calibration``: the constraint rows live on the person table
while the calibrated weights stay on the household table through the frame's
household ``Weights``. The registry refuses callables so that it can serialize,
so each measure has to be a prepared column. ``UK_CGT_REQUIRED_COLUMNS`` names
other columns: the two that ``uk_runtime.cgt_calibration`` prepares.
"""

from __future__ import annotations

from microcosm.build.gates import TargetCoverageRequirement
from microcosm.calibrate import TargetRegistry

__all__ = [
    "UK_CGT_REQUIRED_COLUMNS",
    "UK_CGT_TARGET_COVERAGE_REQUIREMENTS",
    "UK_CGT_TARGET_SPECS",
    "UK_FISCAL_TARGET_REGISTRY",
]

#: Person columns that ``uk_runtime.cgt_calibration`` prepares for its own
#: capital gains frame. The compiled national facts measure the ``hmrc/``
#: columns that the module docstring names, not these.
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

# Values are Ledger-owned and compile from UK target references.
#: The three national facts are joined by the banded families declared in
#: ``uk/uk_population_targets.json`` (Table 2.1a size of gain, Table 6 age,
#: Table 5 region on the individuals basis; microcosm#467, #725). They
#: compile from the same Ledger feed and need no coverage entry here: the
#: fan-out authoring refuses a missing roster cell at generation time.
UK_CGT_TARGET_SPECS: tuple = ()

UK_CGT_TARGET_COVERAGE_REQUIREMENTS: tuple[TargetCoverageRequirement, ...] = (
    TargetCoverageRequirement(
        requirement_id="uk_capital_gains",
        label="HMRC individual capital gains, taxpayer counts and liability",
        accepted_names=(
            "hmrc.cgt.gains_total",
            "hmrc.cgt.taxpayers_total",
            "hmrc.cgt.liability_total",
        ),
        min_matches=3,
        notes=(
            "All three facts describe individuals in observed FY2024-25. "
            "Liability alone cannot anchor the gains distribution or incidence. "
            "The separately retained OBR cash forecast is an unresolved "
            "diagnostic and cannot substitute for observed liability."
        ),
    ),
)

UK_FISCAL_TARGET_REGISTRY = TargetRegistry(UK_CGT_TARGET_SPECS, country="uk")
