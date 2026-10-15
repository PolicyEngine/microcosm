"""Canonical UK release ids and the closed source-tier vocabulary.

The UK has one extra release-id axis that the US does not: the licence class
of the household spine.  It is provenance, not a quality label.  The national
and dense lines keep one constant id each across cuts.

The minting helpers for the exact-cardinality ``populace-uk-<year>-<tier>-k<N>``
grammar were retired with no caller left (docs/release-contract.md, retire
list): new releases take their id from C1's declared identity,
``microcosm-<country>-<vintage>-<line>``.
"""

from __future__ import annotations

__all__ = [
    "UK_DENSE_RELEASE_ID",
    "UK_NATIONAL_RELEASE_ID",
    "UK_RELEASE_TIERS",
    "UK_RELEASE_TIER_CPS_TRANSFER",
    "UK_RELEASE_TIER_FRS",
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


#: The certified UK national line's release id (ruling 2026-08-27): one
#: constant name across cuts — run identity and ordering live in the
#: Logbook chain and artifact versioning, not in the id. The "2024-25"
#: vintage segment follows the uk-data survey-year convention; the
#: "-national" segment keeps the id disjoint from the exact-k ladder's
#: "-k<N>" shape and may retire as the migration moves on.
UK_NATIONAL_RELEASE_ID = "microcosm-uk-2024-25-national"

#: The dense joint (national + local) line's release id (microcosm#762 A18,
#: ruling 2026-09-03): the same constant-id approach as the national line,
#: same "2024-25" FRS-release vintage segment, "-dense" because the artifact
#: is the spine cloned K times through the OA ladder (792,690 rows at K=15)
#: — a sparse successor via the L0 penalty is a later increment. Published
#: on the inspect lane only (per-cut tag, never the default slot); consumers
#: reach it by the registry key ("uk", 2025, "dense").
UK_DENSE_RELEASE_ID = "microcosm-uk-2024-25-dense"


def validate_uk_release_tier(tier: object) -> str:
    """Return a ratified UK source tier, rejecting every other token."""

    if not isinstance(tier, str) or tier not in UK_RELEASE_TIERS:
        raise ValueError(
            f"UK release tier must be one of {sorted(UK_RELEASE_TIERS)}, got "
            f"{tier!r}. A tier names the spine source licence class; quality "
            "adjectives such as 'true', 'full', and 'public' are invalid."
        )
    return tier
