"""UK tenure categories shared by the stages that group tenure four ways."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

UK_TENURE_OWNED_OUTRIGHT = "owned_outright"
UK_TENURE_OWNED_WITH_MORTGAGE = "owned_with_mortgage"
UK_TENURE_PRIVATE_RENT = "private_rent"
UK_TENURE_SOCIAL_RENT = "social_rent"

#: The four-way grouping in a fixed order: owner-occupiers without and with a
#: mortgage, private renters, and social renters (council and housing
#: association).
UK_TENURE_CATEGORIES: tuple[str, ...] = (
    UK_TENURE_OWNED_OUTRIGHT,
    UK_TENURE_OWNED_WITH_MORTGAGE,
    UK_TENURE_PRIVATE_RENT,
    UK_TENURE_SOCIAL_RENT,
)
UK_OWNER_TENURE_CATEGORIES: tuple[str, ...] = (
    UK_TENURE_OWNED_OUTRIGHT,
    UK_TENURE_OWNED_WITH_MORTGAGE,
)

#: The engine's ``tenure_type`` member names onto the four-way grouping.
UK_TENURE_TYPE_TO_CATEGORY: Mapping[str, str] = MappingProxyType(
    {
        "OWNED_OUTRIGHT": UK_TENURE_OWNED_OUTRIGHT,
        "OWNED_WITH_MORTGAGE": UK_TENURE_OWNED_WITH_MORTGAGE,
        "RENT_PRIVATELY": UK_TENURE_PRIVATE_RENT,
        "RENT_FROM_COUNCIL": UK_TENURE_SOCIAL_RENT,
        "RENT_FROM_HA": UK_TENURE_SOCIAL_RENT,
    }
)

__all__ = [
    "UK_OWNER_TENURE_CATEGORIES",
    "UK_TENURE_CATEGORIES",
    "UK_TENURE_OWNED_OUTRIGHT",
    "UK_TENURE_OWNED_WITH_MORTGAGE",
    "UK_TENURE_PRIVATE_RENT",
    "UK_TENURE_SOCIAL_RENT",
    "UK_TENURE_TYPE_TO_CATEGORY",
]
