"""The declared target sets of the US releases (microcosm#1728).

``us/target_sets.json`` declares what each US product calibrates to in each
mode: the national release (``tools/build_us_fiscal_refresh_release.py``,
``--target-surface``) and the ACS local release
(``tools/build_us_acs_local_release.py``, ``--soi-mode``). The shared reader
and selector are in :mod:`microcosm.calibrate.target_sets`; this module names
the US file, products, predicates and rules. The file also declares the US
level rule, which gives every US target its geography level; the UK declares
its rule the same way, in ``uk/target_sets.json`` (microcosm#1759).
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from microcosm.calibrate.target_sets import (
    TargetSetDeclaration,
    TargetSetDeclarations,
    load_target_set_declarations,
)

__all__ = [
    "CONGRESSIONAL_DISTRICT_PREDICATE",
    "LOCAL_PRODUCT",
    "NATIONAL_PRODUCT",
    "STATE_CD_RULE",
    "US_TARGET_SETS_RESOURCE",
    "us_target_level",
    "us_target_set_declaration",
    "us_target_set_declarations",
    "us_target_sets_path",
]

US_TARGET_SETS_RESOURCE = "target_sets.json"
NATIONAL_PRODUCT = "national"
LOCAL_PRODUCT = "local"
#: The predicate the national ``national_state`` mode excludes by: the shared
#: congressional-district classifier
#: (``microcosm.data.us_critical_targets.is_congressional_district_target``).
CONGRESSIONAL_DISTRICT_PREDICATE = "congressional_district"
#: The rule of the local ``state_cd`` mode
#: (``tools/us_acs_local_cd_surface.py`` ``state_cd_soi_surface``).
STATE_CD_RULE = "state_cd_district_shares"


def us_target_sets_path() -> Path:
    return Path(str(files("microcosm.build.us").joinpath(US_TARGET_SETS_RESOURCE)))


@lru_cache(maxsize=1)
def us_target_set_declarations() -> TargetSetDeclarations:
    """The packaged US declarations, read and checked once."""

    return load_target_set_declarations(us_target_sets_path())


def us_target_set_declaration(product: str, mode: str) -> TargetSetDeclaration:
    return us_target_set_declarations().declaration(product, mode)


def us_target_level(spec: Any) -> str:
    """A US target's geography level, by the declared level rule."""

    return us_target_set_declarations().level_rule.level_of(spec)
