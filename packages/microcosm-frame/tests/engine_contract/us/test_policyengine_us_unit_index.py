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


# The engine-free job installs no policyengine-us, so this check of what the
# index reads from the installed source runs here, in the engine-us job.
def test_installed_index_reads_unit_declarations(
    index: PolicyEngineUSVariableMetadataIndex,
) -> None:
    assert index.variable_unit("alimony_expense") == "currency-USD"
    assert index.variable_unit("hours_worked_last_week") == "hour"
    assert index.variable_unit("weeks_unemployed") == "week"
    assert index.variable_unit("own_children_in_household") is None
    # The generated-variable snapshot records no unit.
    assert index.variable_unit("e00700") is None
    with pytest.raises(ValueError, match="Unknown PolicyEngine-US"):
        index.variable_unit("not_a_policyengine_variable")


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
