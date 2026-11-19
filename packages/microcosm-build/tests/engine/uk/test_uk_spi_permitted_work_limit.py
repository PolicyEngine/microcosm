"""SPI copies take their twin's contributory JSA and ESA within each benefit's
work limits, read from policyengine-uk's own variable and parameters
(microcosm#1095 B10).
"""

from __future__ import annotations

from microcosm.build.uk_runtime.spi_benefit_coherence import (
    uk_spi_permitted_work_limit,
)


def test_the_esa_limit_is_the_engines_permitted_work_limit_for_the_year() -> None:
    # 16 hours at the national living wage, rounded up to the next 50p (ESA
    # Regs 2008 reg 45(4) and (9A)): GBP 11.44 in 2024-25, GBP 12.21 in 2025-26.
    assert uk_spi_permitted_work_limit(2024).esa_weekly_limit == 183.5
    assert uk_spi_permitted_work_limit(2025).esa_weekly_limit == 195.5
    limit = uk_spi_permitted_work_limit("2024")
    assert limit.year == 2024
    assert limit.esa_hours_limit == 16.0
    assert limit.source.endswith("claimant_hours at 2024")


def test_the_jsa_limits_are_the_engines_rates_for_the_year() -> None:
    # The 2024-25 and 2025-26 age-related amounts, the GBP 5 disregard and
    # 16 hours of remunerative work: the year's own rates, as formulas read
    # them, not the year before's.
    limit = uk_spi_permitted_work_limit(2024)
    assert (limit.jsa_amount_under_25, limit.jsa_amount_from_25) == (71.7, 90.5)
    assert limit.jsa_earnings_disregard == 5.0
    assert limit.jsa_remunerative_hours == 16.0
    # Reg 81's GBP 50 a week, above which pension payments come off JSA.
    assert limit.jsa_pension_disregard == 50.0
    later = uk_spi_permitted_work_limit(2025)
    assert (later.jsa_amount_under_25, later.jsa_amount_from_25) == (72.9, 92.05)
    assert list(limit.jsa_amount([24, 25])) == [71.7, 90.5]
