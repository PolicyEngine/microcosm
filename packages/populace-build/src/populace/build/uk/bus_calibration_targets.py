"""DfT-anchored calibration targets for the UK bus-spending variables.

Two household consumption variables imputed from survey microdata must be
anchored to Department for Transport (DfT) Annual Bus Statistics, or they
inherit the survey's transport over/under-estimate:

* ``bus_fare_spending`` — fares households pay (DfT BUS05a fare receipts);
* ``bus_subsidy_spending`` — net government support to bus operators
  (DfT BUS05b net government support).

Without these anchors the survey imputation lands roughly twice the fare
total and well below the subsidy total. The published DfT figures are
England-only; they are uplifted to a UK total by the ONS mid-2023 population
ratio (UK 68.3m / England 57.7m ≈ 1.18), because bus fares and subsidy scale
with population.

These specs feed both the calibration solver (``populace.calibrate``) and the
``aggregate_admin_gate``, which flags a candidate population whose weighted
``bus_fare_spending`` / ``bus_subsidy_spending`` total misses the DfT anchor.
"""

from __future__ import annotations

from populace.calibrate import TargetRegistry, TargetSpec

# ONS mid-2023 population, UK / England (millions). DfT bus statistics are
# published for England only; the England totals are uplifted to UK by this
# ratio.
ENGLAND_TO_UK_POPULATION_UPLIFT = 68.3 / 57.7

# Department for Transport, Annual Bus Statistics: year ending March 2025.
_DFT_BUS_STATISTICS_URL = (
    "https://www.gov.uk/government/statistics/"
    "annual-bus-statistics-year-ending-march-2025/"
    "annual-bus-statistics-year-ending-march-2025"
)

# England totals (DfT, year ending March 2025), in GBP.
_DFT_ENGLAND_FARE_RECEIPTS = 3.4e9  # BUS05a passenger fare receipts
_DFT_ENGLAND_NET_GOVERNMENT_SUPPORT = 3.0e9  # BUS05b net government support

UK_BUS_TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec(
        name="dft/bus_fare_spending",
        entity="household",
        value=_DFT_ENGLAND_FARE_RECEIPTS * ENGLAND_TO_UK_POPULATION_UPLIFT,
        aggregation="sum",
        measure="bus_fare_spending",
        period=2025,
        source=(
            "DfT Annual Bus Statistics year ending March 2025, table BUS05a "
            "(England passenger fare receipts GBP 3.4bn), uplifted to UK by the "
            "ONS mid-2023 population ratio. " + _DFT_BUS_STATISTICS_URL
        ),
        family="dft",
    ),
    TargetSpec(
        name="dft/bus_subsidy_spending",
        entity="household",
        value=_DFT_ENGLAND_NET_GOVERNMENT_SUPPORT * ENGLAND_TO_UK_POPULATION_UPLIFT,
        aggregation="sum",
        measure="bus_subsidy_spending",
        period=2025,
        source=(
            "DfT Annual Bus Statistics year ending March 2025, table BUS05b "
            "(England net government support GBP 3.0bn), uplifted to UK by the "
            "ONS mid-2023 population ratio. " + _DFT_BUS_STATISTICS_URL
        ),
        family="dft",
    ),
)

UK_BUS_TARGET_REGISTRY = TargetRegistry(UK_BUS_TARGET_SPECS, country="uk")

__all__ = [
    "ENGLAND_TO_UK_POPULATION_UPLIFT",
    "UK_BUS_TARGET_REGISTRY",
    "UK_BUS_TARGET_SPECS",
]
