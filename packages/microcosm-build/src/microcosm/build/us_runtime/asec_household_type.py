"""Census ASEC HRHTYPE codes, carried from the raw household table.

Codes 1--8 enter housing-unit household estimates; codes 9 and 10 are
group quarters. This housing distinction does not change the CPS SPM universe.
"""

ASEC_HOUSEHOLD_KIND_COLUMN = "HRHTYPE"
ASEC_HOUSING_UNIT_KINDS = tuple(range(1, 9))
ASEC_GROUP_QUARTERS_KINDS = (9, 10)
ASEC_HOUSEHOLD_KINDS = (*ASEC_HOUSING_UNIT_KINDS, *ASEC_GROUP_QUARTERS_KINDS)
