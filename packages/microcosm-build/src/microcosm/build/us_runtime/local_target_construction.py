"""The declared target construction of the US ACS local release (microcosm#1728).

``us/local_target_construction.json`` declares how the ACS local release
(``tools/build_us_acs_local_release.py``) builds the targets that the compile
does not give it. A shared, country-free module reads each section:

- ``holdouts``: :mod:`microcosm.build.target_holdouts` (the district and the
  county hold-out passes, and how their held rows are scored).

This module names the US file and its passes.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from microcosm.build.target_holdouts import (
    HoldoutPass,
    HoldoutScoring,
    TargetHoldouts,
    load_target_holdouts,
)

__all__ = [
    "COUNTY_HOLDOUT",
    "DISTRICT_HOLDOUT",
    "US_LOCAL_TARGET_CONSTRUCTION_RESOURCE",
    "us_holdout_pass",
    "us_holdout_scoring",
    "us_local_target_construction",
    "us_local_target_construction_path",
    "us_target_holdouts",
]

US_LOCAL_TARGET_CONSTRUCTION_RESOURCE = "local_target_construction.json"
#: The hold-out pass over district SOI rows (``--soi-mode state_cd``).
DISTRICT_HOLDOUT = "district"
#: The hold-out pass over county rows (``--soi-mode state_county``).
COUNTY_HOLDOUT = "county"

_ENVELOPE_KEYS = frozenset({"version", "country", "description", "holdouts"})


def us_local_target_construction_path() -> Path:
    return Path(
        str(files("microcosm.build.us").joinpath(US_LOCAL_TARGET_CONSTRUCTION_RESOURCE))
    )


@lru_cache(maxsize=1)
def us_local_target_construction() -> dict[str, Any]:
    """The packaged US file, with its envelope checked."""

    payload = json.loads(
        us_local_target_construction_path().read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict) or set(payload) != _ENVELOPE_KEYS:
        raise ValueError(
            f"{US_LOCAL_TARGET_CONSTRUCTION_RESOURCE} must hold exactly the keys "
            f"{sorted(_ENVELOPE_KEYS)}."
        )
    if payload["version"] != 1 or payload["country"] != "us":
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
