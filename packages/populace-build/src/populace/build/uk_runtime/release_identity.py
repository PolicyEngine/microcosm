"""Canonical source-tier and exact-cardinality identity for UK releases.

The UK has one extra release-id axis that the US does not: the licence class
of the household spine.  It is provenance, not a quality label.  Keeping the
closed vocabulary and formatter together prevents a manifest writer from
minting an unratified tier or an id that disagrees with its manifest fields.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "UK_RELEASE_TIERS",
    "UK_RELEASE_TIER_CPS_TRANSFER",
    "UK_RELEASE_TIER_FRS",
    "UKReleaseIdentity",
    "format_uk_release_id",
    "validate_uk_release_tier",
]

UK_RELEASE_TIER_FRS = "frs"
UK_RELEASE_TIER_CPS_TRANSFER = "cps-transfer"
UK_RELEASE_TIERS = frozenset(
    {
        UK_RELEASE_TIER_FRS,
        UK_RELEASE_TIER_CPS_TRANSFER,
    }
)


def validate_uk_release_tier(tier: object) -> str:
    """Return a ratified UK source tier, rejecting every other token."""

    if not isinstance(tier, str) or tier not in UK_RELEASE_TIERS:
        raise ValueError(
            f"UK release tier must be one of {sorted(UK_RELEASE_TIERS)}, got "
            f"{tier!r}. A tier names the spine source licence class; quality "
            "adjectives such as 'true', 'full', and 'public' are invalid."
        )
    return tier


def _validate_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"UK release {field} must be a positive integer, got {value!r}."
        )
    return value


def format_uk_release_id(
    year: object,
    tier: object,
    record_count: object,
) -> str:
    """Format ``populace-uk-<year>-<tier>-k<N>`` after strict validation."""

    valid_year = _validate_positive_int(year, field="year")
    valid_tier = validate_uk_release_tier(tier)
    valid_record_count = _validate_positive_int(record_count, field="record_count")
    return f"populace-uk-{valid_year}-{valid_tier}-k{valid_record_count}"


@dataclass(frozen=True)
class UKReleaseIdentity:
    """The identity slice every future UK release manifest must record.

    :meth:`as_release_manifest_fields` deliberately returns the build id next
    to its component fields.  A manifest assembler consumes this validated
    slice instead of accepting a free-form tier or a separately formatted id.
    """

    year: int
    tier: str
    record_count: int

    def __post_init__(self) -> None:
        _validate_positive_int(self.year, field="year")
        validate_uk_release_tier(self.tier)
        _validate_positive_int(self.record_count, field="record_count")

    @property
    def release_id(self) -> str:
        """Return the canonical exact-cardinality release id."""

        return format_uk_release_id(self.year, self.tier, self.record_count)

    def as_release_manifest_fields(self) -> dict[str, object]:
        """Return the validated fields copied into a UK release manifest."""

        return {
            "build_id": self.release_id,
            "country": "uk",
            "year": self.year,
            "tier": self.tier,
            "record_count": self.record_count,
        }
