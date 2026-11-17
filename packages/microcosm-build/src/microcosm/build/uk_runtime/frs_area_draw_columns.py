"""The names of the two FRS household carriers the spine's area draw reads.

The clone-first spine's area assignment conditions on each household's IMD
decile within its own nation and on whether its council_tax_band is a band the
FRS gives it that its region uses (microcosm#931). ``frs_spine`` writes both
columns at the root; the area draw's inputs (``atomic_area_conditioning``),
the area stages and the council-tax localisation read them, and the area
mapping gates re-derive each household's level from them.

This module holds the names alone, nothing that computes, so a gate or a
national node that reads them keys on the names without keying on frs_spine,
every edit to which would otherwise re-key the local and the national line on
a bound spine (the rule ``annuity_trust_column`` follows; microcosm#931).
"""

from __future__ import annotations

#: Each household's IMD decile within its own nation (an int64, 1 the most
#: deprived; :data:`FRS_IMD_DECILE_UNOBSERVED` where none is observed).
FRS_IMD_DECILE_COLUMN = "imd_decile"
#: Whether the household's council_tax_band is a band the FRS gives it that its
#: region uses (a bool).
FRS_COUNCIL_TAX_BAND_OBSERVED_COLUMN = "council_tax_band_observed"
#: The decile a household with no observed decile carries.
FRS_IMD_DECILE_UNOBSERVED = 0
