"""The import-free index's variable units against the installed engine's.

The index reads each variable class's ``unit`` from the installed source, and
the US catalog takes its numeric column units from it. So for every ordinary
source variable the index must report the unit the engine itself holds.
"""

import pytest

from microcosm.frame.adapters.policyengine_us import (
    _GENERATED_VARIABLE_GROUPS,
    PolicyEngineUSEngine,
    PolicyEngineUSVariableMetadataIndex,
)


@pytest.fixture(scope="module")
def index() -> PolicyEngineUSVariableMetadataIndex:
    return PolicyEngineUSVariableMetadataIndex()


@pytest.fixture(scope="module")
def engine() -> PolicyEngineUSEngine:
    return PolicyEngineUSEngine(spm={"geography_kind": "national"})


def test_index_units_match_the_engine_for_every_source_variable(
    index: PolicyEngineUSVariableMetadataIndex,
    engine: PolicyEngineUSEngine,
) -> None:
    generated = {name for names, *_ in _GENERATED_VARIABLE_GROUPS for name in names}
    names = sorted(set(index._definitions) - generated)
    assert len(names) == len(index._definitions) - len(generated)
    assert set(names) <= set(engine._tax_benefit_system().variables)

    mismatched = {
        name: (index.variable_unit(name), engine.variable_unit(name))
        for name in names
        if index.variable_unit(name) != engine.variable_unit(name)
    }

    assert mismatched == {}
    assert {index.variable_unit(name) for name in names} >= {
        "currency-USD",
        "/1",
        "hour",
        "week",
        None,
    }
