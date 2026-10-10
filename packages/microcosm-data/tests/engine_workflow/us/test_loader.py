"""Tests split from packages/microcosm-data/tests/test_loader.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_data.loader import *


@pytest.mark.skipif(not _engine_available(), reason="policyengine-us not installed")
def test_live_load_builds_a_simulation_with_a_sane_population() -> None:
    """The certified dataset loads into PolicyEngine-US with a sane population."""
    from policyengine_us import Microsimulation

    try:
        dataset = load("us", 2024)
    except loader._CertifiedPackageCompatibilityError as exc:
        pytest.skip(f"Published release is certified for another engine lock: {exc}")
    except Exception as exc:  # live network/release availability
        if _is_offline_error(exc):
            pytest.skip(f"Hugging Face dataset unavailable offline: {exc}")
        raise

    sim = Microsimulation(dataset=dataset)
    population = (sim.calculate("age", 2024) >= 0).sum()
    assert 3.0e8 < population < 3.6e8
