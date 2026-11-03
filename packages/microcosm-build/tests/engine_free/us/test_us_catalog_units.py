"""Properties of the unit the US catalog gives each numeric pool input."""

import json
import re
from importlib import resources
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tools.us_bundle_generation.core import (
    _CATALOG_UNIT_BY_ENGINE_UNIT,
    _REVIEWED_UNIT_BY_INPUT,
    _numeric_catalog_unit,
)

# The declared units of the catalog schema, without the waiver sentinel.
_SCHEMA_UNITS = frozenset(
    json.loads(
        resources.files("microcosm.build.spec_engine")
        .joinpath("schema/catalogs.schema.json")
        .read_text()
    )["properties"]["columns"]["items"]["properties"]["contract"]["properties"]["unit"][
        "oneOf"
    ][0]["enum"]
)
# Names over a two-letter alphabet, so drawn inputs and engine units often
# match the keys of the drawn tables.
_NAMES = st.text(alphabet="ab", max_size=2)
_ENGINE_UNITS = st.none() | _NAMES
_CATALOG_UNITS = st.sampled_from(sorted(_SCHEMA_UNITS))


def test_every_unit_the_catalog_can_assign_is_a_schema_unit() -> None:
    assigned = set(_CATALOG_UNIT_BY_ENGINE_UNIT.values()) | {
        unit for _, unit in _REVIEWED_UNIT_BY_INPUT.values()
    }
    assert assigned <= _SCHEMA_UNITS


# Inputs cost nothing to draw; a busy host can still trip the timing check.
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    mapped=st.dictionaries(_NAMES, _CATALOG_UNITS),
    reviewed=st.dictionaries(_NAMES, st.tuples(_ENGINE_UNITS, _CATALOG_UNITS)),
    variable=_NAMES,
    engine_unit=_ENGINE_UNITS,
)
def test_catalog_unit_maps_declared_or_reviewed_units_and_refuses_the_rest(
    mapped: dict[str, str],
    reviewed: dict[str, tuple[str | None, str]],
    variable: str,
    engine_unit: str | None,
) -> None:
    if variable in reviewed:
        reviewed_engine_unit, unit = reviewed[variable]
        expected = unit if engine_unit == reviewed_engine_unit else None
    else:
        expected = None if engine_unit is None else mapped.get(engine_unit)
    with (
        patch.dict(_CATALOG_UNIT_BY_ENGINE_UNIT, mapped, clear=True),
        patch.dict(_REVIEWED_UNIT_BY_INPUT, reviewed, clear=True),
    ):
        if expected is None:
            with pytest.raises(ValueError, match=re.escape(repr(variable))):
                _numeric_catalog_unit(variable, engine_unit)
        else:
            assert _numeric_catalog_unit(variable, engine_unit) == expected
