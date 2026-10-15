from __future__ import annotations

import pytest

from microcosm.build import uk_runtime
from microcosm.build.uk_runtime import release_identity
from microcosm.build.uk_runtime.release_identity import (
    UK_RELEASE_TIER_CPS_TRANSFER,
    UK_RELEASE_TIER_FRS,
    UK_RELEASE_TIERS,
    validate_uk_release_tier,
)


def test_uk_release_tiers_are_the_two_ratified_provenance_tokens() -> None:
    assert UK_RELEASE_TIER_FRS == "frs"
    assert UK_RELEASE_TIER_CPS_TRANSFER == "cps-transfer"
    assert UK_RELEASE_TIERS == frozenset({"frs", "cps-transfer"})
    assert uk_runtime.validate_uk_release_tier is validate_uk_release_tier


@pytest.mark.parametrize("tier", ["frs", "cps-transfer"])
def test_uk_release_tier_accepts_the_ratified_tokens(tier: str) -> None:
    assert validate_uk_release_tier(tier) == tier


@pytest.mark.parametrize(
    "tier",
    [
        "true",
        "full",
        "public",
        "FRS",
        "frs ",
        "",
        None,
        1,
    ],
)
def test_uk_release_tier_rejects_quality_adjectives_and_unknown_values(
    tier: object,
) -> None:
    with pytest.raises(ValueError, match="spine source licence class"):
        validate_uk_release_tier(tier)


@pytest.mark.parametrize(
    "retired",
    ["UKReleaseIdentity", "apply_uk_release_identity", "format_uk_release_id"],
)
def test_exact_k_id_minting_is_retired(retired: str) -> None:
    """New releases take their id from C1's declared identity
    (docs/release-contract.md), not from the exact-k minting helpers."""

    assert not hasattr(release_identity, retired)
    assert not hasattr(uk_runtime, retired)
