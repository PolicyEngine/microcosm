"""The CHRONICLE_* / POPULACE_LEDGER_* environment dual-read window.

chronicle#143 gives the operational stores a dual-read window so publish
flows and build scripts migrate on their own schedule: ``CHRONICLE_*`` is
preferred, the ledger-era name is still honored, and honoring it warns once
per process. These tests hold both halves — the fallback keeps working, and
the warning stays a single line rather than one per read in a build loop.
"""

from __future__ import annotations

import warnings

import pytest

from microcosm.build.chronicle_env import (
    CHRONICLE_API_KEY_ENV,
    CHRONICLE_ENV_LEGACY_NAMES,
    CHRONICLE_EXPORT_KEY_ENV,
    CHRONICLE_KEY_ENV,
    CHRONICLE_URL_ENV,
    LEGACY_API_KEY_ENV,
    LEGACY_EXPORT_KEY_ENV,
    LEGACY_KEY_ENV,
    LEGACY_URL_ENV,
    chronicle_env,
    chronicle_env_names,
    describe_chronicle_env,
    reset_chronicle_env_deprecation_warnings,
)
from microcosm.build.logbook import _remote_config

PAIRS = (
    (CHRONICLE_URL_ENV, LEGACY_URL_ENV),
    (CHRONICLE_KEY_ENV, LEGACY_KEY_ENV),
    (CHRONICLE_API_KEY_ENV, LEGACY_API_KEY_ENV),
    (CHRONICLE_EXPORT_KEY_ENV, LEGACY_EXPORT_KEY_ENV),
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for preferred, legacy in PAIRS:
        monkeypatch.delenv(preferred, raising=False)
        monkeypatch.delenv(legacy, raising=False)
    reset_chronicle_env_deprecation_warnings()
    yield
    reset_chronicle_env_deprecation_warnings()


@pytest.mark.parametrize(("preferred", "legacy"), PAIRS)
def test_preferred_name_wins_and_warns_about_nothing(
    monkeypatch, preferred: str, legacy: str
) -> None:
    monkeypatch.setenv(preferred, "chronicle-value")
    monkeypatch.setenv(legacy, "ledger-value")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert chronicle_env(preferred) == "chronicle-value"


@pytest.mark.parametrize(("preferred", "legacy"), PAIRS)
def test_legacy_name_is_honored_with_a_deprecation_warning(
    monkeypatch, preferred: str, legacy: str
) -> None:
    monkeypatch.setenv(legacy, "ledger-value")

    with pytest.warns(DeprecationWarning) as record:
        assert chronicle_env(preferred) == "ledger-value"

    message = str(record[0].message)
    assert legacy in message
    assert preferred in message
    assert "chronicle#143" in message


def test_the_deprecation_warning_fires_once_per_process(monkeypatch) -> None:
    """A build loop reads these repeatedly; one warning, not a storm."""
    monkeypatch.setenv(LEGACY_URL_ENV, "https://ledger.example")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            assert chronicle_env(CHRONICLE_URL_ENV) == "https://ledger.example"

    assert [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ].__len__() == 1


def test_each_legacy_name_warns_on_its_own(monkeypatch) -> None:
    monkeypatch.setenv(LEGACY_URL_ENV, "https://ledger.example")
    monkeypatch.setenv(LEGACY_KEY_ENV, "writer-jwt")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chronicle_env(CHRONICLE_URL_ENV)
        chronicle_env(CHRONICLE_KEY_ENV)

    warned = {
        legacy
        for legacy in (LEGACY_URL_ENV, LEGACY_KEY_ENV)
        if any(legacy in str(w.message) for w in caught)
    }
    assert warned == {LEGACY_URL_ENV, LEGACY_KEY_ENV}


def test_unset_returns_the_default_and_empty_counts_as_unset(monkeypatch) -> None:
    assert chronicle_env(CHRONICLE_URL_ENV) is None
    assert chronicle_env(CHRONICLE_URL_ENV, "fallback") == "fallback"

    monkeypatch.setenv(CHRONICLE_URL_ENV, "")
    monkeypatch.setenv(LEGACY_URL_ENV, "https://ledger.example")
    with pytest.warns(DeprecationWarning):
        assert chronicle_env(CHRONICLE_URL_ENV) == "https://ledger.example"


def test_an_explicit_environ_mapping_bypasses_the_process_environment() -> None:
    with pytest.warns(DeprecationWarning):
        assert (
            chronicle_env(CHRONICLE_KEY_ENV, environ={LEGACY_KEY_ENV: "writer-jwt"})
            == "writer-jwt"
        )


def test_a_name_outside_the_window_is_a_programming_error() -> None:
    with pytest.raises(KeyError, match="not a Chronicle environment variable"):
        chronicle_env("CHRONICLE_NOT_A_REAL_VARIABLE")
    with pytest.raises(KeyError, match="not a Chronicle environment variable"):
        chronicle_env_names(LEGACY_URL_ENV)


def test_names_and_descriptions_carry_both_spellings() -> None:
    assert chronicle_env_names(CHRONICLE_URL_ENV) == (CHRONICLE_URL_ENV, LEGACY_URL_ENV)

    described = describe_chronicle_env(CHRONICLE_URL_ENV, CHRONICLE_EXPORT_KEY_ENV)
    # An operator whose environment predates the rename must still be able to
    # match the error text against what they have set.
    for name in (
        CHRONICLE_URL_ENV,
        CHRONICLE_EXPORT_KEY_ENV,
        LEGACY_URL_ENV,
        LEGACY_EXPORT_KEY_ENV,
    ):
        assert name in described


def test_every_windowed_variable_maps_to_exactly_one_legacy_name() -> None:
    assert dict(CHRONICLE_ENV_LEGACY_NAMES) == {
        preferred: (legacy,) for preferred, legacy in PAIRS
    }


def test_logbook_remote_config_reads_both_eras(monkeypatch) -> None:
    assert _remote_config() is None

    monkeypatch.setenv(LEGACY_URL_ENV, "https://ledger.example")
    monkeypatch.setenv(LEGACY_KEY_ENV, "writer-jwt")
    with pytest.warns(DeprecationWarning):
        assert _remote_config() == (
            "https://ledger.example",
            "writer-jwt",
            "writer-jwt",
        )

    monkeypatch.setenv(CHRONICLE_URL_ENV, "https://chronicle.example")
    monkeypatch.setenv(CHRONICLE_KEY_ENV, "chronicle-jwt")
    monkeypatch.setenv(CHRONICLE_API_KEY_ENV, "project-api-key")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert _remote_config() == (
            "https://chronicle.example",
            "chronicle-jwt",
            "project-api-key",
        )
