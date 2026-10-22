"""DfE funded early-years registrations on the spring-term basis.

DfE counts the children registered for each funded entitlement in England in
the census of January, by age on the preceding 31 December. policyengine-uk's
entitlements are annual: a child receives one if it is funded in any of the
year's three terms (1 April, 1 September and the next 1 January). A child
counts in DfE's January figures only if it is funded in the spring term, so
these measures classify each child at the spring term of model year Y, which
starts on 1 January of Y + 1:

- **School.** Since policyengine-uk 2.147.2 a child's free hours stop when it
  starts school, from its date of birth.
  ``early_years_term_share_before_school`` is 0, 1/3 or 1, and a child is
  still before school in the spring term only at 1. The share is float32,
  so it is tested against one half, never for equality.
- **Working parent.** Working-parent hours start the term after the child
  reaches the minimum age (SI 2022/1134 reg 27). The year's eligible terms
  always end with the spring term, so a child the engine gives working-parent
  hours in the year (its gross total per child) and that is still before
  school takes part in the spring term. DfE records each child under one
  entitlement: at DfE age 2 a child counted under early learning for
  2-year-olds is not counted again under working parents, and at 3 and 4 a
  working-parent child is one that uses more than the universal hours, as the
  engine nets them.
- **Universal and early learning for 2-year-olds.** These start the term
  after the third or second birthday, so in the spring term they reach every
  child of that DfE age. policyengine-uk still gates them on the whole age at
  6 October, which leaves out the children born from 7 October to 31
  December. The measures keep the engine's family and claim conditions (a
  claim on their own, or allocation from the gross working-parent hours, as
  the engine allocates them) and replace only that age gate, so they count
  the children DfE counts; they do not change what the engine pays.
- **DfE age** is ``Y - birth_year``, the age on 31 December Y.

The take-up flags and family conditions are annual and apply to the spring
term unchanged. ``early_years_school_start_deferred`` is not supplied (the
FRS records no deferred start), so every child starts school on the date its
date of birth implies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

__all__ = [
    "DFE_AGE_AT_31_DECEMBER",
    "DFE_SPRING_EARLY_LEARNING_2",
    "DFE_SPRING_TERM_ENGINE_INPUTS",
    "DFE_SPRING_TERM_MEASURES",
    "DFE_SPRING_UNIVERSAL",
    "DFE_SPRING_WORKING_PARENT",
    "UNIVERSAL_WEEKLY_HOURS",
    "dfe_spring_term_status",
]

DFE_AGE_AT_31_DECEMBER = "dfe_age_at_31_december"
DFE_SPRING_WORKING_PARENT = "dfe_spring_working_parent"
DFE_SPRING_UNIVERSAL = "dfe_spring_universal"
DFE_SPRING_EARLY_LEARNING_2 = "dfe_spring_early_learning_2"
#: Person-level measures, all computed together from one set of engine reads.
DFE_SPRING_TERM_MEASURES = (
    DFE_AGE_AT_31_DECEMBER,
    DFE_SPRING_WORKING_PARENT,
    DFE_SPRING_UNIVERSAL,
    DFE_SPRING_EARLY_LEARNING_2,
)
#: The engine variables the measures read, each at person grain.
DFE_SPRING_TERM_ENGINE_INPUTS = (
    "birth_year",
    "country",
    "early_years_term_share_before_school",
    "extended_childcare_entitlement_total_per_child",
    "is_of_compulsory_school_age",
    "max_free_entitlement_hours_used",
    "maximum_extended_childcare_hours_usage",
    "targeted_childcare_entitlement_eligible",
    "would_claim_targeted_childcare",
    "would_claim_universal_childcare",
)
_UNIVERSAL_DFE_AGES = (3, 4)
_EARLY_LEARNING_DFE_AGE = 2
#: The universal entitlement's weekly hours: 570 a year over 38 weeks
#: (SI 2014/2147 regs 4 and 5). A working-parent child aged 3 or 4 is
#: registered under working parents only if it uses more than these.
UNIVERSAL_WEEKLY_HOURS = 15.0


def _spring_term_start(year: int) -> int:
    from policyengine_uk.utils.dates import term_starts

    spring = int(term_starts(year)[2])
    if spring != (year + 1) * 10_000 + 101:
        raise ValueError(
            f"policyengine-uk's spring term of {year} starts on {spring}, not "
            f"1 January {year + 1}; the DfE spring-term measures assume the "
            "engine's April-to-March year."
        )
    return spring


def _flag(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in "biuf":
        return array.astype(float) > 0.5
    raise ValueError(f"expected a numeric or boolean flag, got dtype {array.dtype}")


def dfe_spring_term_status(
    calculate: Callable[[str], np.ndarray], year: int
) -> Mapping[str, np.ndarray]:
    """Classify each person at the spring term of model year ``year``.

    ``calculate(name)`` returns an engine variable at person grain for
    ``year``, group variables broadcast to their members and enums decoded to
    their names. The result maps each name in ``DFE_SPRING_TERM_MEASURES`` to
    a person array: the DfE age as a float, the others as 0/1 floats.
    """

    _spring_term_start(int(year))
    dfe_age = float(year) - np.asarray(calculate("birth_year"), dtype=float)
    in_nursery = (
        np.asarray(calculate("early_years_term_share_before_school"), dtype=float) > 0.5
    )
    child_hours = np.asarray(calculate("max_free_entitlement_hours_used"), dtype=float)
    family_hours = np.asarray(
        calculate("maximum_extended_childcare_hours_usage"), dtype=float
    )
    uses_hours = child_hours > 0
    # Gross working-parent hours, before the universal or targeted hours are
    # netted out: what the engine allocates those hours from.
    takes_part = (
        np.asarray(
            calculate("extended_childcare_entitlement_total_per_child"), dtype=float
        )
        > 0
    ) & in_nursery
    in_england = np.asarray(calculate("country")).astype(str) == "ENGLAND"
    universal = (
        in_england
        & ~_flag(calculate("is_of_compulsory_school_age"))
        & in_nursery
        & uses_hours
        & np.isin(dfe_age, _UNIVERSAL_DFE_AGES)
        & (_flag(calculate("would_claim_universal_childcare")) | takes_part)
    )
    early_learning_2 = (
        _flag(calculate("targeted_childcare_entitlement_eligible"))
        & in_nursery
        & uses_hours
        & (dfe_age == _EARLY_LEARNING_DFE_AGE)
        & (_flag(calculate("would_claim_targeted_childcare")) | takes_part)
    )
    beyond_universal = np.minimum(child_hours, family_hours) > UNIVERSAL_WEEKLY_HOURS
    working_parent = takes_part & np.select(
        [dfe_age < _EARLY_LEARNING_DFE_AGE, dfe_age == _EARLY_LEARNING_DFE_AGE],
        [True, ~early_learning_2],
        default=beyond_universal,
    )
    return {
        DFE_AGE_AT_31_DECEMBER: dfe_age,
        DFE_SPRING_WORKING_PARENT: working_parent.astype(float),
        DFE_SPRING_UNIVERSAL: universal.astype(float),
        DFE_SPRING_EARLY_LEARNING_2: early_learning_2.astype(float),
    }
