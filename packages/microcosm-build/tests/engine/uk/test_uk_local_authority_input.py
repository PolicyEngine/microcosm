"""Tests split from packages/microcosm-build/tests/test_uk_local_authority_input.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_local_authority_input import *


def test_every_roster_engine_key_is_a_local_authority_member_at_the_pinned_engine() -> (
    None
):
    """Fail closed on the whole roster against the installed engine.

    Red while the engine pin predates the policyengine-uk release that adds
    the six April 2023 unitaries (``APRIL_2023_UNITARIES``); the build refuses
    those households' rows for the same reason, so a partial map never ships.
    """

    missing = local_authority_keys_missing_from_engine()

    assert missing == (), (
        f"{len(missing)} roster authorities are not LocalAuthority members at "
        f"the pinned engine: {missing}"
    )


def test_verify_engine_domain_accepts_members_and_names_missing_keys() -> None:
    verify_local_authority_engine_domain(["CITY_OF_LONDON", "MAIDSTONE"])
    assert local_authority_keys_missing_from_engine(["CITY_OF_LONDON"]) == ()

    with pytest.raises(ValueError, match=r"not in the installed.*NOT_AN_AUTHORITY"):
        verify_local_authority_engine_domain(
            np.asarray(["CITY_OF_LONDON", "NOT_AN_AUTHORITY"], dtype=object)
        )
    assert local_authority_keys_missing_from_engine(["NOT_AN_AUTHORITY"]) == (
        "NOT_AN_AUTHORITY",
    )


def test_engine_domain_gap_is_exactly_the_april_2023_unitaries_or_closed() -> None:
    """Any other miss is a roster or rule defect, not the known enum lag."""

    missing = local_authority_keys_missing_from_engine()

    assert set(missing) <= set(APRIL_2023_UNITARIES)
