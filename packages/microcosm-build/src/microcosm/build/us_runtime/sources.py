"""Compatibility module for the removed US Python source-stage functions.

US source stages are now manifest-defined. This module stays importable so
older build drivers fail with a clear migration error instead of a bare
``ImportError``.
"""

from __future__ import annotations

from microcosm.build.us_runtime import (
    US_NONNEGATIVE_SOURCE_OUTPUTS,
    US_SOURCE_MANIFEST,
    US_SOURCE_STAGE_SPECS,
)

__all__ = [
    "NONNEGATIVE_SCF_TARGETS",
    "SCF_TARGETS",
    "US_NONNEGATIVE_SOURCE_OUTPUTS",
    "US_SOURCE_MANIFEST",
    "US_SOURCE_STAGE_SPECS",
    "_load_census_cps_person",
    "_support_guard",
    "add_acs_rent",
    "add_meps_esi_premiums",
    "add_mortgage_conversion",
    "add_org_wages",
    "add_prior_year_income",
    "add_scf_wealth",
    "add_sipp_tips",
    "add_vehicle_assets",
]

_REMOVED_NAMES = (
    "add_acs_rent",
    "add_meps_esi_premiums",
    "add_mortgage_conversion",
    "add_org_wages",
    "add_prior_year_income",
    "add_scf_wealth",
    "add_sipp_tips",
    "add_vehicle_assets",
    "_floor_nonnegative",
    "_load_census_cps_person",
    "_support_guard",
    "NONNEGATIVE_SCF_TARGETS",
    "SCF_TARGETS",
)


def _migration_error(name: str) -> RuntimeError:
    return RuntimeError(
        f"{name} was a country-specific Python source-stage bridge and has "
        "been removed. Use US_SOURCE_MANIFEST for source content and inject "
        "a shared runtime implementation through us_plan(...)."
    )


def _removed_callable(name: str):
    def removed(*_args, **_kwargs):
        raise _migration_error(name)

    removed.__name__ = name
    removed.__qualname__ = name
    removed.__doc__ = "Removed US source-stage bridge; call raises a migration error."
    return removed


class _RemovedConstant:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        raise _migration_error(self._name)

    def __bool__(self) -> bool:
        raise _migration_error(self._name)

    def __iter__(self):
        raise _migration_error(self._name)

    def __contains__(self, _item: object) -> bool:
        raise _migration_error(self._name)


add_acs_rent = _removed_callable("add_acs_rent")
add_meps_esi_premiums = _removed_callable("add_meps_esi_premiums")
add_mortgage_conversion = _removed_callable("add_mortgage_conversion")
add_org_wages = _removed_callable("add_org_wages")
add_prior_year_income = _removed_callable("add_prior_year_income")
add_scf_wealth = _removed_callable("add_scf_wealth")
add_sipp_tips = _removed_callable("add_sipp_tips")
add_vehicle_assets = _removed_callable("add_vehicle_assets")
_floor_nonnegative = _removed_callable("_floor_nonnegative")
_load_census_cps_person = _removed_callable("_load_census_cps_person")
_support_guard = _removed_callable("_support_guard")
NONNEGATIVE_SCF_TARGETS = _RemovedConstant("NONNEGATIVE_SCF_TARGETS")
SCF_TARGETS = _RemovedConstant("SCF_TARGETS")


def __getattr__(name: str) -> object:
    if name in _REMOVED_NAMES:
        raise AttributeError(str(_migration_error(name)))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
