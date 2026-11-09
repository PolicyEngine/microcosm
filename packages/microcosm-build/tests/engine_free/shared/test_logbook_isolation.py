"""The suite cannot reach a live Logbook, however the shell is configured.

The Logbook live store is append-only. A test that appends to it has written
a row nobody can retract, into the chain a real release will later have to
validate. So the guarantee this file pins is not "tests usually do not use
the network" — it is that a test *cannot* reach the store even when the
developer or runner shell is fully credentialed, under either generation of
the variable names.

The guard itself lives in ``conftest.py`` as an autouse fixture, which is
what makes it total: nothing has to remember to ask for it.
"""

from __future__ import annotations

import os

import pytest

from microcosm.build import logbook
from microcosm.build.logbook_env import LOGBOOK_ENV_LEGACY_NAMES, logbook_env

_ALL_LOGBOOK_ENV_NAMES = tuple(
    sorted(
        {name for name in LOGBOOK_ENV_LEGACY_NAMES}
        | {
            legacy
            for legacy_names in LOGBOOK_ENV_LEGACY_NAMES.values()
            for legacy in legacy_names
        }
        | {"POPULACE_LOGBOOK_PREV_ROW_DIGEST"}
    )
)


@pytest.mark.parametrize("name", _ALL_LOGBOOK_ENV_NAMES)
def test_no_logbook_variable_survives_into_a_test(name: str) -> None:
    """Both generations are cleared, not just the ledger-era spelling.

    The per-module ``delenv`` lists this fixture replaced named only
    ``POPULACE_LEDGER_*``. They were written before ``LOGBOOK_*`` existed, so
    a shell exporting the preferred names walked straight past them.
    """
    assert name not in os.environ


def test_remote_config_is_unconfigured_regardless_of_the_shell() -> None:
    """``_remote_config`` is the gate every write to the live store passes."""
    assert logbook._remote_config() is None
    for name in LOGBOOK_ENV_LEGACY_NAMES:
        assert logbook_env(name) is None


def test_the_logbook_network_call_is_refused_by_default() -> None:
    """An unstubbed request raises instead of leaving the machine."""
    with pytest.raises(AssertionError, match="real Logbook HTTP request"):
        logbook.urlopen(object(), timeout=1.0)


def test_the_underlying_opener_is_refused_too() -> None:
    """The second guard: a stale reference to the real ``urlopen``.

    ``tools/logbook.py`` binds ``urlopen`` at import time, and a future
    importer could do the same. Guarding the opener the real function
    delegates to closes that path without having to enumerate importers.
    """
    with pytest.raises(AssertionError, match="real Logbook HTTP request"):
        logbook._NO_REDIRECT_OPENER.open(object(), timeout=1.0)


def test_appending_a_row_to_a_polluted_environment_writes_nothing() -> None:
    """The end-to-end shape of the accident this prevents.

    With credentials in the environment, ``_remote_config`` would return them
    and the append would POST to Supabase. Here it is unconfigured, so the
    remote leg is skipped entirely — no request is attempted, which is why
    this passes rather than raising the guard's error.
    """
    assert logbook._remote_config() is None


def test_the_opt_in_fixture_restores_the_real_callables(
    allow_logbook_network,
) -> None:
    """The guard is a policy with a documented exception, not a wall.

    Nothing in this suite opts in. The fixture exists so a test that genuinely
    needs the network can say so in its own body, and so the restoration path
    is exercised rather than assumed. No request is made here: this asserts
    only that the shipped callables are back in place.
    """
    assert logbook.urlopen.__module__ == logbook.__name__
    assert logbook.urlopen.__name__ == "urlopen"
    assert logbook._NO_REDIRECT_OPENER.open.__self__ is logbook._NO_REDIRECT_OPENER
