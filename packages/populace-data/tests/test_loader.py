"""The loader: resolution, error messages, and (when possible) a live load.

The resolution and error-path tests run everywhere — they never touch the
network or an engine. The live test downloads the published artifact and builds
a PolicyEngine-US simulation, so it skips cleanly when the dataset is not yet
published or the optional engine is absent.
"""

from __future__ import annotations

import pytest

from populace.data import DEFAULT_VARIANT, latest_year, load, resolve


def test_resolve_defaults_to_the_latest_year() -> None:
    spec = resolve("us")
    assert spec.year == latest_year("us")
    assert spec.variant == DEFAULT_VARIANT


def test_resolve_is_case_insensitive_on_country() -> None:
    assert resolve("US", 2024).key == ("us", 2024, DEFAULT_VARIANT)


def test_resolve_can_select_uk_compact_dataset() -> None:
    spec = resolve("uk", 2023)
    assert spec.key == ("uk", 2023, DEFAULT_VARIANT)
    assert spec.filename == "populace_uk_2023.h5"


def test_resolve_unknown_year_names_the_published_years() -> None:
    with pytest.raises(
        ValueError,
        match=r"published years for 'us' variant 'compact': \[2024\]",
    ):
        resolve("us", 1999)


def test_resolve_unknown_variant_names_the_published_variants() -> None:
    with pytest.raises(ValueError, match="variant 'local'"):
        resolve("uk", 2023, variant="local")


def test_resolve_unknown_country_names_the_published_countries() -> None:
    with pytest.raises(ValueError, match="published country variants"):
        resolve("atlantis")


def test_latest_year_unknown_country_raises() -> None:
    with pytest.raises(ValueError, match="No populace dataset for country"):
        latest_year("atlantis")


def _engine_available() -> bool:
    try:
        import policyengine_us  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _engine_available(), reason="policyengine-us not installed")
def test_live_load_builds_a_simulation_with_a_sane_population() -> None:
    """The published dataset loads into PolicyEngine-US and totals a sane count."""
    from policyengine_us import Microsimulation

    try:
        dataset = load("us", 2024)
    except Exception as exc:  # dataset not yet published / no network
        pytest.skip(f"dataset not loadable: {exc}")

    sim = Microsimulation(dataset=dataset)
    # Weighted person count (microdf weights the boolean sum); there is no
    # "people" variable in policyengine-us.
    population = (sim.calculate("age", 2024) >= 0).sum()
    # The US population is ~330-345M; a calibrated dataset must land in range.
    assert 3.0e8 < population < 3.6e8
