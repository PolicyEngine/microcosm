"""Dual-read for the Chronicle (formerly Ledger) environment variables.

PolicyEngine/chronicle#143 gives the operational stores — buckets, database
schema, role ids, and env names — a **dual-read window**: ``CHRONICLE_*`` is
preferred, the legacy name is still honored, and honoring it emits a
deprecation warning so publish flows and build scripts migrate on their own
schedule. This module is that window, in one place.

Every variable here is read through :func:`chronicle_env`, which tries the
preferred ``CHRONICLE_*`` name first and falls back to the legacy name,
warning once per process per legacy name. Nothing is renamed on disk: the
legacy names keep working for as long as the window is open, and the legacy
name constants stay exported so error messages and tests can still name them.

Note on scope: the identifiers ``LEDGER_HMRC_BANDS``,
``LEDGER_ONS_TURNOVER_BANDS``, ``LEDGER_ONS_EMPLOYMENT_BANDS``, and
``LEDGER_US_SOURCE_COVERAGE_CONTRACT_COMMIT`` look like environment variables
to a grep but are plain Python module constants (band maps and a git commit
pin). They get chronicle-named *aliases* beside their modules rather than a
dual-read, because there is no environment to read them from.
"""

from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Mapping

__all__ = [
    "CHRONICLE_API_KEY_ENV",
    "CHRONICLE_ENV_LEGACY_NAMES",
    "CHRONICLE_EXPORT_KEY_ENV",
    "CHRONICLE_KEY_ENV",
    "CHRONICLE_URL_ENV",
    "LEGACY_API_KEY_ENV",
    "LEGACY_EXPORT_KEY_ENV",
    "LEGACY_KEY_ENV",
    "LEGACY_URL_ENV",
    "chronicle_env",
    "chronicle_env_names",
    "describe_chronicle_env",
    "reset_chronicle_env_deprecation_warnings",
]

#: Preferred, chronicle-era names.
CHRONICLE_URL_ENV = "CHRONICLE_URL"
CHRONICLE_KEY_ENV = "CHRONICLE_KEY"
CHRONICLE_API_KEY_ENV = "CHRONICLE_API_KEY"
CHRONICLE_EXPORT_KEY_ENV = "CHRONICLE_EXPORT_KEY"

#: Legacy, ledger-era names. Still honored; still named in error messages so
#: an operator running the old environment recognises what is being asked for.
LEGACY_URL_ENV = "POPULACE_LEDGER_URL"
LEGACY_KEY_ENV = "POPULACE_LEDGER_KEY"
LEGACY_API_KEY_ENV = "POPULACE_LEDGER_API_KEY"
LEGACY_EXPORT_KEY_ENV = "POPULACE_LEDGER_EXPORT_KEY"

#: Preferred name -> legacy names, most recent legacy spelling first. Adding a
#: variable to the dual-read window means adding a row here and nothing else.
CHRONICLE_ENV_LEGACY_NAMES: Mapping[str, tuple[str, ...]] = {
    CHRONICLE_URL_ENV: (LEGACY_URL_ENV,),
    CHRONICLE_KEY_ENV: (LEGACY_KEY_ENV,),
    CHRONICLE_API_KEY_ENV: (LEGACY_API_KEY_ENV,),
    CHRONICLE_EXPORT_KEY_ENV: (LEGACY_EXPORT_KEY_ENV,),
}

_WARNED_LEGACY_NAMES: set[str] = set()
_WARNED_LOCK = threading.Lock()


def chronicle_env(
    name: str,
    default: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Read one Chronicle variable, preferring ``name`` over its legacy spelling.

    ``name`` is the preferred ``CHRONICLE_*`` name. When it is unset but a
    legacy name carries a value, that value is returned and a
    :class:`DeprecationWarning` is emitted **once per process per legacy
    name** — repeated reads in a build loop must not turn into a warning
    storm. An empty value is treated as unset, matching how the callers here
    already test these variables.
    """
    source = os.environ if environ is None else environ
    if name not in CHRONICLE_ENV_LEGACY_NAMES:
        raise KeyError(
            f"{name!r} is not a Chronicle environment variable; expected one of "
            f"{sorted(CHRONICLE_ENV_LEGACY_NAMES)}."
        )
    value = source.get(name)
    if value:
        return value
    for legacy_name in CHRONICLE_ENV_LEGACY_NAMES[name]:
        legacy_value = source.get(legacy_name)
        if legacy_value:
            _warn_once(legacy_name, preferred=name)
            return legacy_value
    return default


def chronicle_env_names(name: str) -> tuple[str, ...]:
    """The preferred name followed by every legacy name still honored."""
    if name not in CHRONICLE_ENV_LEGACY_NAMES:
        raise KeyError(
            f"{name!r} is not a Chronicle environment variable; expected one of "
            f"{sorted(CHRONICLE_ENV_LEGACY_NAMES)}."
        )
    return (name, *CHRONICLE_ENV_LEGACY_NAMES[name])


def describe_chronicle_env(*names: str) -> str:
    """Render required variables for an error message, legacy names included.

    Error text names both spellings on purpose: an operator whose environment
    predates the rename must still be able to match the message against what
    they have set.
    """
    preferred = ", ".join(names)
    legacy = ", ".join(
        legacy_name
        for name in names
        for legacy_name in CHRONICLE_ENV_LEGACY_NAMES[name]
    )
    return f"{preferred} (legacy {legacy} still honored)"


def reset_chronicle_env_deprecation_warnings() -> None:
    """Forget which legacy names have warned. For tests only."""
    with _WARNED_LOCK:
        _WARNED_LEGACY_NAMES.clear()


def _warn_once(legacy_name: str, *, preferred: str) -> None:
    with _WARNED_LOCK:
        if legacy_name in _WARNED_LEGACY_NAMES:
            return
        _WARNED_LEGACY_NAMES.add(legacy_name)
    warnings.warn(
        f"{legacy_name} is the ledger-era name for {preferred} and is "
        "deprecated; PolicyEngine Ledger is now Chronicle "
        "(PolicyEngine/chronicle#143). Set "
        f"{preferred} instead — {legacy_name} stays honored only for the "
        "dual-read window.",
        DeprecationWarning,
        stacklevel=3,
    )
