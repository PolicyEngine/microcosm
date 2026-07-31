"""Canonical source-tier and exact-cardinality identity for UK releases.

The UK has one extra release-id axis that the US does not: the licence class
of the household spine.  It is provenance, not a quality label.  Keeping the
closed vocabulary and formatter together prevents a manifest writer from
minting an unratified tier or an id that disagrees with its manifest fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "UK_RELEASE_TIERS",
    "UK_RELEASE_TIER_CPS_TRANSFER",
    "UK_RELEASE_TIER_FRS",
    "UKReleaseIdentity",
    "apply_uk_release_identity",
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
            "country": "uk",
            "year": self.year,
            "tier": self.tier,
            "record_count": self.record_count,
            "build": {"build_id": self.release_id},
        }


def apply_uk_release_identity(
    manifest: Mapping[str, object],
    identity: UKReleaseIdentity,
) -> dict[str, object]:
    """Apply a validated UK identity without overwriting conflicting fields.

    Release manifests carry ``tier`` at the top level and ``build_id`` inside
    the existing ``build`` object.  The helper makes that schema seam
    explicit, preserves all other build metadata, and refuses to relabel a
    manifest whose identity was already assembled differently.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("UK release manifest must be a mapping.")
    if not isinstance(identity, UKReleaseIdentity):
        raise TypeError("identity must be a UKReleaseIdentity.")

    expected_fields: dict[str, object] = {
        "country": "uk",
        "year": identity.year,
        "tier": identity.tier,
        "record_count": identity.record_count,
    }
    for field, expected in expected_fields.items():
        if field in manifest and manifest[field] != expected:
            raise ValueError(
                f"UK release manifest {field!r} is {manifest[field]!r}, "
                f"expected {expected!r}."
            )

    raw_build = manifest.get("build")
    if raw_build is None:
        build: dict[str, object] = {}
    elif isinstance(raw_build, Mapping):
        build = dict(raw_build)
    else:
        raise ValueError("UK release manifest 'build' must be an object.")
    if "build_id" in build and build["build_id"] != identity.release_id:
        raise ValueError(
            "UK release manifest 'build.build_id' is "
            f"{build['build_id']!r}, expected {identity.release_id!r}."
        )
    build["build_id"] = identity.release_id

    return {
        **dict(manifest),
        **expected_fields,
        "build": build,
    }
