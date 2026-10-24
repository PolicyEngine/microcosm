"""Tests split from packages/microcosm-build/tests/test_uk_hmrc_uprating_engine_pins.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_hmrc_uprating_engine_pins import *


def test_the_vendored_pins_match_the_installed_engine() -> None:
    pins = hu.engine_parameter_pins()
    assert pins["engine"]["version"] == hu.installed_engine_version()
    for path, by_instant in pins["values"].items():
        for instant, value in by_instant.items():
            assert hu.live_engine_parameter_value(path, instant) == float(value), (
                path,
                instant,
            )


def test_the_locked_install_passes_the_freeze() -> None:
    # The lock's own install: its version, the files policyengine_uk imports
    # from and their RECORD agree with the pin.
    assert hu.assert_installed_engine_is_pinned() == hu.pinned_engine_version()
