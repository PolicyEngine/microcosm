"""SOI filing-status metadata and the PolicyEngine-US enum names it selects."""

from types import MappingProxyType

_SOI_FILING_STATUSES = MappingProxyType(
    {
        "all": ("All", ()),
        "head_of_household": ("Head of Household", ("HEAD_OF_HOUSEHOLD",)),
        "married_filing_jointly_surviving_spouse": (
            "Married Filing Jointly/Surviving Spouse",
            ("JOINT", "SURVIVING_SPOUSE"),
        ),
        "married_filing_separately": ("Married Filing Separately", ("SEPARATE",)),
        "single": ("Single", ("SINGLE",)),
    }
)

SOI_FILING_STATUS_LABELS = MappingProxyType(
    {key: label for key, (label, _) in _SOI_FILING_STATUSES.items()}
)
SOI_FILING_STATUS_ENGINE_NAMES = MappingProxyType(
    {label: names for label, names in _SOI_FILING_STATUSES.values()}
)
#: Release-builder API: ``None`` selects all statuses, a string selects one,
#: and a set selects the joint/surviving-spouse aggregate.
FILING_STATUS_MAP = MappingProxyType(
    {
        label: None if not names else names[0] if len(names) == 1 else frozenset(names)
        for label, names in _SOI_FILING_STATUSES.values()
    }
)
