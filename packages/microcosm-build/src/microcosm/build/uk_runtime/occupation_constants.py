"""UK occupation and industry codes: the engine's SOC 2020 groups and the codings the build reads."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

#: The engine's ``soc2020_major_group`` member for a person with no occupation
#: recorded: a child, an adult who has never worked, or an unknown code.
UK_SOC2020_NO_OCCUPATION = "NONE"

#: The engine's ``soc2020_major_group`` members (policyengine-uk
#: ``SOC2020MajorGroup``) by SOC 2020 major group code, in code order (ONS SOC
#: 2020 Volume 1).
UK_SOC2020_MAJOR_GROUPS: Mapping[int, str] = MappingProxyType(
    {
        1: "MANAGERS_DIRECTORS_AND_SENIOR_OFFICIALS",
        2: "PROFESSIONAL",
        3: "ASSOCIATE_PROFESSIONAL",
        4: "ADMINISTRATIVE_AND_SECRETARIAL",
        5: "SKILLED_TRADES",
        6: "CARING_LEISURE_AND_OTHER_SERVICE",
        7: "SALES_AND_CUSTOMER_SERVICE",
        8: "PROCESS_PLANT_AND_MACHINE_OPERATIVES",
        9: "ELEMENTARY",
    }
)

#: The FRS adult-table ``SOC2020`` coding -> the engine's member: 0 is undefined,
#: and each major group is its code times 1000 (UKDS FRS 2024-25 data
#: dictionary, SN 9563).
UK_FRS_SOC2020_MAJOR_GROUP: Mapping[int, str] = MappingProxyType(
    {
        0: UK_SOC2020_NO_OCCUPATION,
        **{code * 1000: name for code, name in UK_SOC2020_MAJOR_GROUPS.items()},
    }
)

#: The employment-status group an occupation draw conditions on, by the
#: engine's ``employment_status`` member. CHILD has none: a child holds no
#: occupation.
UK_OCCUPATION_STATUS_GROUP: Mapping[str, str] = MappingProxyType(
    {
        "FT_EMPLOYED": "EMPLOYEE",
        "PT_EMPLOYED": "EMPLOYEE",
        "FT_SELF_EMPLOYED": "SELF_EMPLOYED",
        "PT_SELF_EMPLOYED": "SELF_EMPLOYED",
        "RETIRED": "RETIRED",
        "UNEMPLOYED": "OTHER",
        "STUDENT": "OTHER",
        "CARER": "OTHER",
        "LONG_TERM_DISABLED": "OTHER",
        "SHORT_TERM_DISABLED": "OTHER",
        "OTHER_INACTIVE": "OTHER",
    }
)

#: The SIC 2007 sections A-U by the first and last of their two-digit
#: divisions (ONS UK SIC 2007). Divisions 4, 34 and the other gaps exist in
#: no section.
UK_SIC2007_SECTION_DIVISIONS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "A": (1, 3),
        "B": (5, 9),
        "C": (10, 33),
        "D": (35, 35),
        "E": (36, 39),
        "F": (41, 43),
        "G": (45, 47),
        "H": (49, 53),
        "I": (55, 56),
        "J": (58, 63),
        "K": (64, 66),
        "L": (68, 68),
        "M": (69, 75),
        "N": (77, 82),
        "O": (84, 84),
        "P": (85, 85),
        "Q": (86, 88),
        "R": (90, 93),
        "S": (94, 96),
        "T": (97, 98),
        "U": (99, 99),
    }
)


def uk_sic2007_sections(divisions) -> np.ndarray:
    """Each two-digit SIC 2007 division's section letter; "" outside every section."""

    values = np.asarray(divisions, dtype=np.int64)
    out = np.full(values.shape, "", dtype=object)
    for section, (first, last) in UK_SIC2007_SECTION_DIVISIONS.items():
        out[(values >= first) & (values <= last)] = section
    return out


__all__ = [
    "UK_FRS_SOC2020_MAJOR_GROUP",
    "UK_OCCUPATION_STATUS_GROUP",
    "UK_SIC2007_SECTION_DIVISIONS",
    "UK_SOC2020_MAJOR_GROUPS",
    "UK_SOC2020_NO_OCCUPATION",
    "uk_sic2007_sections",
]
