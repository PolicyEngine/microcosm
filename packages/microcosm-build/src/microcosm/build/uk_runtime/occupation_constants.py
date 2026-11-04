"""UK occupation codes: the engine's SOC 2020 groups and the codings the build reads."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

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

__all__ = [
    "UK_FRS_SOC2020_MAJOR_GROUP",
    "UK_SOC2020_MAJOR_GROUPS",
    "UK_SOC2020_NO_OCCUPATION",
]
