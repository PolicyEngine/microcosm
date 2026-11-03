"""Properties of the unit the US catalog gives each numeric pool input."""

import json
import re
from importlib import resources

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tools.us_bundle_generation.core import (
    _CATALOG_UNIT_BY_ENGINE_UNIT,
    _REVIEWED_UNIT_BY_UNITLESS_INPUT,
    _numeric_catalog_unit,
)

# The declared units of the catalog schema, without the waiver sentinel.
_SCHEMA_UNITS = frozenset(
    json.loads(
        resources.files("microcosm.build.spec_engine")
        .joinpath("schema/catalogs.schema.json")
        .read_text()
    )["properties"]["columns"]["items"]["properties"]["contract"]["properties"][
        "unit"
    ]["oneOf"][0]["enum"]
)


def test_every_unit_the_catalog_can_assign_is_a_schema_unit() -> None:
    assigned = set(_CATALOG_UNIT_BY_ENGINE_UNIT.values()) | set(
        _REVIEWED_UNIT_BY_UNITLESS_INPUT.values()
    )
    assert assigned <= _SCHEMA_UNITS


# Inputs cost nothing to draw; a busy host can still trip the timing check.
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    variable=st.one_of(
        st.sampled_from(sorted(_REVIEWED_UNIT_BY_UNITLESS_INPUT)), st.text()
    ),
    engine_unit=st.one_of(
        st.none(), st.sampled_from(sorted(_CATALOG_UNIT_BY_ENGINE_UNIT)), st.text()
    ),
)
def test_catalog_unit_maps_declared_or_reviewed_units_and_refuses_the_rest(
    variable: str, engine_unit: str | None
) -> None:
    reviewed = variable in _REVIEWED_UNIT_BY_UNITLESS_INPUT
    if engine_unit is None and reviewed:
        expected = _REVIEWED_UNIT_BY_UNITLESS_INPUT[variable]
    elif engine_unit in _CATALOG_UNIT_BY_ENGINE_UNIT and not reviewed:
        expected = _CATALOG_UNIT_BY_ENGINE_UNIT[engine_unit]
    else:
        with pytest.raises(ValueError, match=re.escape(repr(variable))):
            _numeric_catalog_unit(variable, engine_unit)
        return
    assert _numeric_catalog_unit(variable, engine_unit) == expected
    assert expected in _SCHEMA_UNITS
