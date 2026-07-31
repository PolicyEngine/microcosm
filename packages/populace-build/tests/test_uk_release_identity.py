from __future__ import annotations

import pytest

from populace.build import uk_runtime
from populace.build.uk_runtime.release_identity import (
    UK_RELEASE_TIER_CPS_TRANSFER,
    UK_RELEASE_TIER_FRS,
    UK_RELEASE_TIERS,
    UKReleaseIdentity,
    apply_uk_release_identity,
    format_uk_release_id,
    validate_uk_release_tier,
)


def test_uk_release_tiers_are_the_two_ratified_provenance_tokens() -> None:
    assert UK_RELEASE_TIER_FRS == "frs"
    assert UK_RELEASE_TIER_CPS_TRANSFER == "cps-transfer"
    assert UK_RELEASE_TIERS == frozenset({"frs", "cps-transfer"})
    assert uk_runtime.apply_uk_release_identity is apply_uk_release_identity


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("frs", "populace-uk-2023-frs-k535080"),
        ("cps-transfer", "populace-uk-2023-cps-transfer-k57240"),
    ],
)
def test_uk_release_id_formatter_uses_tier_and_exact_record_count(
    tier: str,
    expected: str,
) -> None:
    record_count = 535_080 if tier == "frs" else 57_240

    assert format_uk_release_id(2023, tier, record_count) == expected


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
    ("year", "record_count", "match"),
    [
        (0, 1, "year"),
        (-2023, 1, "year"),
        (True, 1, "year"),
        ("2023", 1, "year"),
        (2023, 0, "record_count"),
        (2023, -1, "record_count"),
        (2023, True, "record_count"),
        (2023, 1.5, "record_count"),
    ],
)
def test_uk_release_id_formatter_rejects_non_positive_or_non_integer_axes(
    year: object,
    record_count: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        format_uk_release_id(year, "frs", record_count)


def test_uk_release_identity_is_the_manifest_assembly_surface() -> None:
    identity = UKReleaseIdentity(
        year=2023,
        tier=UK_RELEASE_TIER_FRS,
        record_count=535_080,
    )

    assert identity.release_id == "populace-uk-2023-frs-k535080"
    assert identity.as_release_manifest_fields() == {
        "country": "uk",
        "year": 2023,
        "tier": "frs",
        "record_count": 535_080,
        "n_records": 535_080,
        "build": {"build_id": "populace-uk-2023-frs-k535080"},
    }


def test_uk_release_identity_validates_before_manifest_assembly() -> None:
    with pytest.raises(ValueError, match="spine source licence class"):
        UKReleaseIdentity(year=2023, tier="public", record_count=535_080)


def test_apply_uk_release_identity_preserves_existing_build_metadata() -> None:
    identity = UKReleaseIdentity(year=2023, tier="frs", record_count=535_080)
    manifest = {
        "schema_version": 1,
        "build": {
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.19.0",
            }
        },
    }

    assembled = apply_uk_release_identity(manifest, identity)

    assert assembled["tier"] == "frs"
    assert assembled["build"] == {
        "build_id": "populace-uk-2023-frs-k535080",
        "built_with_core_package": {
            "name": "policyengine-core",
            "version": "3.19.0",
        },
    }
    assert manifest == {
        "schema_version": 1,
        "build": {
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.19.0",
            }
        },
    }


@pytest.mark.parametrize(
    "manifest",
    [
        {"tier": "cps-transfer"},
        {"n_records": 2},
        {"build": {"build_id": "populace-uk-2023-cps-transfer-k535080"}},
    ],
)
def test_apply_uk_release_identity_rejects_conflicting_identity(
    manifest: dict[str, object],
) -> None:
    identity = UKReleaseIdentity(year=2023, tier="frs", record_count=535_080)

    with pytest.raises(ValueError, match="tier|n_records|build.build_id"):
        apply_uk_release_identity(manifest, identity)
