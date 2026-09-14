"""Canonical columns for the declared ACS/ASEC property-income bridge."""

PROPERTY_COMPONENTS = (
    "property_ordinary_interest",
    "property_retirement_interest",
    "property_dividends",
    "property_broad_receipts",
)
PROPERTY_REPORTED_TOTAL = "property_reported_total"
PROPERTY_DRAW_COLUMNS = tuple("draw_" + name for name in PROPERTY_COMPONENTS)
