"""The declared target sets of the US releases (microcosm#1728).

``us/spec/target_sets.yaml`` declares what each US product calibrates to in each
mode: the national release (``tools/build_us_fiscal_refresh_release.py``,
``--target-surface``) and the ACS local release
(``tools/build_us_acs_local_release.py``, ``--soi-mode``). The shared reader
and selector are in :mod:`microcosm.calibrate.target_sets`; this module names
the US file, products, predicates and rules. The file also declares the US
level rule, which gives every US target its geography level; the UK declares
its rule the same way, in ``uk/spec/target_sets.yaml`` (microcosm#1759).

The file is a typed resource of the US spec bundle (kind ``target_sets``):
the spec engine reads it strictly, checks it against
``target_sets.schema.json`` and keeps its descriptions and rationales apart
from its content, which is part of the bundle's ``spec_sha256``. It is
written by hand: the constants-era code never held it, so
``tools/generate_us_bundle_from_constants.py`` does not write it.
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

US_TARGET_SETS_RESOURCE = "spec/target_sets.yaml"
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
    """The packaged US declarations, read and checked once.

    They come through :func:`microcosm.build.country_spec.load_country_target_sets`,
    which reads the one typed resource as the bundle loader does.
    """

    # Imported here: the country spec module is large, and this module is
    # imported by modules that the country spec's own imports reach.
    from microcosm.build.country_spec import load_country_target_sets

    document = load_country_target_sets("us")
    if document is None:  # pragma: no cover - the package declares the resource
        raise RuntimeError("The US package declares no target_sets resource.")
    return load_target_set_declarations(document)


def us_target_set_declaration(product: str, mode: str) -> TargetSetDeclaration:
    return us_target_set_declarations().declaration(product, mode)


def us_target_level(spec: Any) -> str:
    """A US target's geography level, by the declared level rule."""

    return us_target_set_declarations().level_rule.level_of(spec)
