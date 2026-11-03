"""SPI copies take their twin's contributory JSA and ESA within ESA's
permitted-work limit, read from policyengine-uk's own variable (microcosm#1095
B10).
"""

from __future__ import annotations

from microcosm.build.uk_runtime.spi_benefit_coherence import (
    uk_spi_permitted_work_limit,
)


def test_the_limit_is_the_engines_permitted_work_limit_for_the_year() -> None:
    # 16 hours at the national living wage, rounded up to the next 50p (ESA
    # Regs 2008 reg 45(4) and (9A)): GBP 11.44 in 2024-25, GBP 12.21 in 2025-26.
    assert uk_spi_permitted_work_limit(2024).weekly_limit == 183.5
    assert uk_spi_permitted_work_limit(2025).weekly_limit == 195.5
    limit = uk_spi_permitted_work_limit("2024")
    assert limit.year == 2024
    assert limit.source.endswith("esa_exempt_work_higher_earnings_limit at 2024")
