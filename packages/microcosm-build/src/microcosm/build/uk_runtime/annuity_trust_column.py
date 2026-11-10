"""The name of the UK spine's annuity and trust income descriptor.

FRS rows keep the payments of an annuity not bought with pension funds and of
a trust or covenant in private_pension_income, where policyengine-uk's means
tests, household income and tax read them. SPI Table 3.7 counts both as other
income (note 6), so the spine stores the part they make of
private_pension_income apart, under this name: a calibration-only descriptor
that the Table 3.7 other-income measure adds (microcosm#1428 review, finding
1). frs_spine writes it, the SPI income stage zeros it on the rows it draws,
and measure_simulation's resolver reads it.

The resolver reads zeros where a frame lacks the column, so a renamed constant
would change the measures without re-keying them, and every node whose code
reads the name binds this module. It holds the name alone, nothing that
computes, so the target and problem kernels key on the name without keying on
frs_spine, every edit to which would otherwise re-solve the national line on a
bound spine (microcosm#931 ruling 4).
"""

from __future__ import annotations

#: The descriptor: each person's annual annuity and trust payments, already
#: inside private_pension_income. No engine input reads it.
ANNUITY_AND_TRUST_INCOME_COLUMN = "annuity_and_trust_income"
