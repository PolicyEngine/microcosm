"""Guarantees for the engine's ``local_authority`` input (microcosm#953).

The rowwise ladder resolves every household's April 2023 ONS local authority
code to a policyengine-uk ``LocalAuthority`` member name through the committed
ONS names resource and the mechanical key rule. These tests pin the resource,
its coherence with the local-area crosswalk roster, the rule, the fail-closed
resolver, and membership in the installed engine's enum.
"""

# ruff: noqa: F401

from __future__ import annotations

import hashlib
import json
import re
from importlib import resources as importlib_resources

import numpy as np
import pandas as pd
import pytest

import microcosm.build.uk_runtime.geography_sources as geography_sources
from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import (
    LAD23_NAMES_ITEM_ID,
    LAD23_NAMES_SHA256,
    LAD23_NAMES_URL,
    LOCAL_AUTHORITY_ENGINE_KEY_ALIASES,
    UK_GEOGRAPHY_LADDER_COLUMNS,
    UK_LOCAL_AUTHORITY_NAMES_KIND,
    UK_LOCAL_AUTHORITY_NAMES_RESOURCE,
    UK_LOCAL_AUTHORITY_VINTAGE,
    load_uk_local_authority_names_resource,
    local_authority_consistency_failures,
    local_authority_engine_key,
    local_authority_engine_key_by_code,
    local_authority_keys_missing_from_engine,
    resolve_local_authority_engine_keys,
    verify_local_authority_engine_domain,
)
from tools.generate_uk_local_authority_names import (
    DEFAULT_SOURCE_CSV,
    build_local_authority_names,
)

#: The reviewed digest of the ONS lookup, held here as a literal so a re-pin
#: of ``LAD23_NAMES_SHA256`` is visible in this test's diff too.
SOURCE_SHA256 = "6c2d811f50756c459c6a1c7c692ae9262fb4d70df3f535958ef7cdbbbd76cc44"

#: The April 2023 unitaries the engine enum did not carry at 2.98.0; the
#: roster-wide membership test below names them until the pin moves past the
#: policyengine-uk release that adds them.
APRIL_2023_UNITARIES = (
    "CUMBERLAND",
    "NORTH_NORTHAMPTONSHIRE",
    "NORTH_YORKSHIRE",
    "SOMERSET",
    "WEST_NORTHAMPTONSHIRE",
    "WESTMORLAND_AND_FURNESS",
)


def _resource() -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(UK_LOCAL_AUTHORITY_NAMES_RESOURCE)
        .read_text(encoding="utf-8")
    )


def _crosswalk_area_ids() -> list[str]:
    crosswalk = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_area_crosswalk.json")
        .read_text(encoding="utf-8")
    )
    return crosswalk["levels"]["local_authority"]["area_ids"]


__all__ = [name for name in globals() if not name.startswith("__")]
