"""The declared target construction of the US ACS local release (microcosm#1728).

``us/local_target_construction.json`` declares how the ACS local release
(``tools/build_us_acs_local_release.py``) builds the targets that the compile
does not give it. A shared, country-free module reads each section:

- ``holdouts``: :mod:`microcosm.build.target_holdouts` (the district and the
  county hold-out passes, and how their held rows are scored).
- ``ladder_population``: :mod:`microcosm.build.ladder_population_targets`
  (the PUMA ladder's state and district population rows and specs).

This module names the US file, its passes and its label lookups.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from microcosm.build.declared_registers import read_strict_json
from microcosm.build.ladder_population_targets import (
    LabelLookup,
    LadderPopulation,
    load_ladder_population,
)
from microcosm.build.target_holdouts import (
    HoldoutPass,
    HoldoutScoring,
    TargetHoldouts,
    load_target_holdouts,
)

__all__ = [
    "COUNTY_HOLDOUT",
    "DISTRICT_HOLDOUT",
    "US_DISTRICT_LABEL",
    "US_LOCAL_TARGET_CONSTRUCTION_RESOURCE",
    "US_STATE_LABEL",
    "us_holdout_pass",
    "us_holdout_scoring",
    "us_ladder_population",
    "us_local_target_construction",
    "us_local_target_construction_path",
    "us_population_labels",
    "us_target_holdouts",
]

US_LOCAL_TARGET_CONSTRUCTION_RESOURCE = "local_target_construction.json"
#: The hold-out pass over district SOI rows (``--soi-mode state_cd``).
DISTRICT_HOLDOUT = "district"
#: The hold-out pass over county rows (``--soi-mode state_county``).
COUNTY_HOLDOUT = "county"
#: The label lookups the ladder levels name: a state's Census name, and a
#: district's Census Bureau name (one label for each geography across target
#: sources, microcosm#1335).
US_STATE_LABEL = "us_state_name"
US_DISTRICT_LABEL = "us_congressional_district_name"

_ENVELOPE_KEYS = frozenset(
    {"version", "country", "description", "holdouts", "ladder_population"}
)


def us_local_target_construction_path() -> Path:
    return Path(
        str(files("microcosm.build.us").joinpath(US_LOCAL_TARGET_CONSTRUCTION_RESOURCE))
    )


@lru_cache(maxsize=1)
def us_local_target_construction() -> dict[str, Any]:
    """The packaged US file, with its envelope checked."""

    # Read strictly: a repeated JSON key (a second pass list, a second field
    # of one name) is refused, not resolved to the last one.
    payload = read_strict_json(
        us_local_target_construction_path().read_text(encoding="utf-8"),
        label=US_LOCAL_TARGET_CONSTRUCTION_RESOURCE,
    )
    if not isinstance(payload, dict) or set(payload) != _ENVELOPE_KEYS:
        raise ValueError(
            f"{US_LOCAL_TARGET_CONSTRUCTION_RESOURCE} must hold exactly the keys "
            f"{sorted(_ENVELOPE_KEYS)}."
        )
    if (
        type(payload["version"]) is not int
        or payload["version"] != 1
        or payload["country"] != "us"
    ):
        raise ValueError(
            f"{US_LOCAL_TARGET_CONSTRUCTION_RESOURCE} must be version 1 for country us."
        )
    return payload


@lru_cache(maxsize=1)
def us_target_holdouts() -> TargetHoldouts:
    """The US hold-out passes, read and checked once."""

    return load_target_holdouts(us_local_target_construction()["holdouts"])


def us_holdout_pass(pass_id: str) -> HoldoutPass:
    return us_target_holdouts().holdout_pass(pass_id)


def us_holdout_scoring(pass_id: str) -> HoldoutScoring:
    """The declared scoring of a US hold-out pass; refuse a pass without one."""

    scoring = us_holdout_pass(pass_id).scoring
    if scoring is None:
        raise ValueError(f"The US hold-out pass {pass_id!r} declares no scoring.")
    return scoring


@lru_cache(maxsize=1)
def us_ladder_population() -> LadderPopulation:
    """The US ladder population levels, read and checked once."""

    return load_ladder_population(us_local_target_construction()["ladder_population"])


def _state_label(code: str, geography_id: str) -> str | None:
    from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_NAME

    return US_STATE_FIPS_TO_NAME.get(code)


def _district_label(code: str, geography_id: str) -> str | None:
    from microcosm.build.us_runtime.congressional_district_vintage import (
        congressional_district_label,
    )

    return congressional_district_label(geography_id)


def us_population_labels() -> dict[str, LabelLookup]:
    """The label lookup of each label kind the US ladder levels name."""

    return {US_STATE_LABEL: _state_label, US_DISTRICT_LABEL: _district_label}
