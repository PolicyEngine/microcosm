"""The LOGBOOK_* / POPULACE_LEDGER_* environment dual-read window.

Logbook (microcosm#632) gives its operational store a dual-read window so
publish flows and build scripts migrate on their own schedule: ``LOGBOOK_*``
is preferred, the ledger-era name is still honored, and honoring it warns
once per process. These tests hold both halves — the fallback keeps working,
and the warning stays a single line rather than one per read in a build loop.
"""

from __future__ import annotations

import warnings

import pytest

from microcosm.build.logbook import _remote_config
from microcosm.build.logbook_env import (
    LEGACY_API_KEY_ENV,
    LEGACY_EXPORT_KEY_ENV,
    LEGACY_KEY_ENV,
    LEGACY_URL_ENV,
    LOGBOOK_API_KEY_ENV,
    LOGBOOK_ENV_LEGACY_NAMES,
    LOGBOOK_EXPORT_KEY_ENV,
    LOGBOOK_KEY_ENV,
    LOGBOOK_URL_ENV,
    describe_logbook_env,
    logbook_env,
    logbook_env_names,
    reset_logbook_env_deprecation_warnings,
)

PAIRS = (
    (LOGBOOK_URL_ENV, LEGACY_URL_ENV),
    (LOGBOOK_KEY_ENV, LEGACY_KEY_ENV),
    (LOGBOOK_API_KEY_ENV, LEGACY_API_KEY_ENV),
    (LOGBOOK_EXPORT_KEY_ENV, LEGACY_EXPORT_KEY_ENV),
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for preferred, legacy in PAIRS:
        monkeypatch.delenv(preferred, raising=False)
        monkeypatch.delenv(legacy, raising=False)
    reset_logbook_env_deprecation_warnings()
    yield
    reset_logbook_env_deprecation_warnings()


@pytest.mark.parametrize(("preferred", "legacy"), PAIRS)
def test_preferred_name_wins_and_warns_about_nothing(
    monkeypatch, preferred: str, legacy: str
) -> None:
    monkeypatch.setenv(preferred, "logbook-value")
    monkeypatch.setenv(legacy, "ledger-value")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert logbook_env(preferred) == "logbook-value"


@pytest.mark.parametrize(("preferred", "legacy"), PAIRS)
def test_legacy_name_is_honored_with_a_deprecation_warning(
    monkeypatch, preferred: str, legacy: str
) -> None:
    monkeypatch.setenv(legacy, "ledger-value")

    with pytest.warns(DeprecationWarning) as record:
        assert logbook_env(preferred) == "ledger-value"

    message = str(record[0].message)
    assert legacy in message
    assert preferred in message
    assert "microcosm#632" in message


def test_the_deprecation_warning_fires_once_per_process(monkeypatch) -> None:
    """A build loop reads these repeatedly; one warning, not a storm."""
    monkeypatch.setenv(LEGACY_URL_ENV, "https://ledger.example")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            assert logbook_env(LOGBOOK_URL_ENV) == "https://ledger.example"

    assert [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ].__len__() == 1


def test_each_legacy_name_warns_on_its_own(monkeypatch) -> None:
    monkeypatch.setenv(LEGACY_URL_ENV, "https://ledger.example")
    monkeypatch.setenv(LEGACY_KEY_ENV, "writer-jwt")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logbook_env(LOGBOOK_URL_ENV)
        logbook_env(LOGBOOK_KEY_ENV)

    warned = {
        legacy
        for legacy in (LEGACY_URL_ENV, LEGACY_KEY_ENV)
        if any(legacy in str(w.message) for w in caught)
    }
    assert warned == {LEGACY_URL_ENV, LEGACY_KEY_ENV}


def test_unset_returns_the_default_and_empty_counts_as_unset(monkeypatch) -> None:
    assert logbook_env(LOGBOOK_URL_ENV) is None
    assert logbook_env(LOGBOOK_URL_ENV, "fallback") == "fallback"

    monkeypatch.setenv(LOGBOOK_URL_ENV, "")
    monkeypatch.setenv(LEGACY_URL_ENV, "https://ledger.example")
    with pytest.warns(DeprecationWarning):
        assert logbook_env(LOGBOOK_URL_ENV) == "https://ledger.example"


def test_an_explicit_environ_mapping_bypasses_the_process_environment() -> None:
    with pytest.warns(DeprecationWarning):
        assert (
            logbook_env(LOGBOOK_KEY_ENV, environ={LEGACY_KEY_ENV: "writer-jwt"})
            == "writer-jwt"
        )


def test_a_name_outside_the_window_is_a_programming_error() -> None:
    with pytest.raises(KeyError, match="not a Logbook environment variable"):
        logbook_env("LOGBOOK_NOT_A_REAL_VARIABLE")
    with pytest.raises(KeyError, match="not a Logbook environment variable"):
        logbook_env_names(LEGACY_URL_ENV)


def test_names_and_descriptions_carry_both_spellings() -> None:
    assert logbook_env_names(LOGBOOK_URL_ENV) == (LOGBOOK_URL_ENV, LEGACY_URL_ENV)

    described = describe_logbook_env(LOGBOOK_URL_ENV, LOGBOOK_EXPORT_KEY_ENV)
    # An operator whose environment predates the rename must still be able to
    # match the error text against what they have set.
    for name in (
        LOGBOOK_URL_ENV,
        LOGBOOK_EXPORT_KEY_ENV,
        LEGACY_URL_ENV,
        LEGACY_EXPORT_KEY_ENV,
    ):
        assert name in described


def test_every_windowed_variable_maps_to_exactly_one_legacy_name() -> None:
    assert dict(LOGBOOK_ENV_LEGACY_NAMES) == {
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

    monkeypatch.setenv(LOGBOOK_URL_ENV, "https://logbook.example")
    monkeypatch.setenv(LOGBOOK_KEY_ENV, "logbook-jwt")
    monkeypatch.setenv(LOGBOOK_API_KEY_ENV, "project-api-key")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert _remote_config() == (
            "https://logbook.example",
            "logbook-jwt",
            "project-api-key",
        )


def test_the_module_import_binds_the_module_not_the_reader() -> None:
    """``import microcosm.build.logbook_env as env`` must give the module.

    The reader function shares its module's name. Re-exporting it from the
    package barrel bound the function to ``microcosm.build.logbook_env``, and
    ``import a.b as c`` returns the *attribute* when one exists — so the
    submodule import silently handed back a callable, and every
    ``env.LOGBOOK_URL_ENV`` after it raised ``AttributeError``. The barrel
    therefore does not re-export the function; callers import it from here.
    """
    import types

    import microcosm.build
    import microcosm.build.logbook_env as env

    assert isinstance(env, types.ModuleType)
    assert env.__name__ == "microcosm.build.logbook_env"
    assert env.LOGBOOK_URL_ENV == LOGBOOK_URL_ENV
    assert callable(env.logbook_env)
    # The package attribute is the module too: `from microcosm.build import
    # logbook_env` and the submodule import must not disagree about what the
    # name means.
    assert microcosm.build.logbook_env is env
    assert "logbook_env" not in microcosm.build.__all__
    # The non-colliding helper is still re-exported, so removing the shadow
    # did not quietly shrink the barrel further than it had to.
    assert microcosm.build.logbook_env_names is logbook_env_names
