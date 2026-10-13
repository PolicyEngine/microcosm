"""Tests split from packages/microcosm-data/tests/test_stored_inputs.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_data.stored_inputs import *


def test_the_naming_convention_holds_for_the_installed_engine():
    """Every variable is model-named except household Boolean formulas named by
    a two-letter code. Those are formula-owned, so the release writer refuses
    to store them, and uppercase stored columns may pass by rule. An engine
    that breaks this fails here rather than slipping past the check."""

    from importlib import metadata

    from policyengine_us.system import system

    names = set(system.variables)
    outside = {name for name in names if not is_model_named(name)}

    assert not [name for name in names if name[:1] in "_0123456789"]
    assert all(re.fullmatch(r"[A-Z]{2}", name) for name in outside)
    for name in outside:
        variable = system.variables[name]
        assert variable.entity.key == "household"
        assert variable.value_type is bool
        assert variable.formulas
    if metadata.version("policyengine-us") == "2.2.1":
        assert (len(names), outside) == (6167, _US_POSTAL_CODES)


def test_the_register_is_consistent_with_the_installed_engine():
    engine = stored_inputs.installed_us_engine()

    assert (
        register_consistency_failures(
            US_STORED_NON_VARIABLE_COLUMNS, engine_variables=engine.variables
        )
        == []
    )


def test_the_1026_premises_hold_for_the_installed_engine():
    """The two retired inputs, the construction columns the register treats
    and core's role columns, as the installed engine defines them."""

    from policyengine_us.system import system

    variables = system.variables
    assert "would_claim_wic" not in variables
    wic = variables["takes_up_wic_if_eligible"]
    assert (wic.entity.key, wic.value_type, wic.default_value) == ("person", bool, True)
    assert not wic.formulas
    for retired_or_construction in (
        "medicare_part_b_premiums",
        "tax_unit_role_input",
        "filing_status_input",
    ):
        assert retired_or_construction not in variables
    # medicare_part_b_premiums is a Person, YEAR, float input with no formula
    # in every policyengine-us version read from 1.452.0 to 1.670.2; its
    # replacement has that shape.
    part_b = variables["medicare_part_b_premiums_reported"]
    assert (part_b.entity.key, part_b.definition_period, part_b.value_type) == (
        "person",
        "year",
        float,
    )
    assert not part_b.formulas
    # Core reads these to build the group entities; none is a variable.
    assert {entity.key for entity in system.group_entities} == {
        "household",
        "tax_unit",
        "spm_unit",
        "family",
        "marital_unit",
    }
    assert not _CORE_ROLE_COLUMNS & set(variables)
    # puma_geoid is registered as an alias of the puma input.
    assert "puma_geoid" not in variables
    puma = variables["puma"]
    assert (puma.entity.key, bool(puma.formulas)) == ("household", False)
    for derived in (
        "filing_status",
        "is_tax_unit_head",
        "is_tax_unit_spouse",
        "is_tax_unit_dependent",
    ):
        assert variables[derived].formulas


@pytest.mark.parametrize("name", sorted(_EXPECTED_VERDICTS))
def test_each_examined_file_gets_its_expected_verdict(name):
    """The examined files, by their stored-column inventories, against the
    installed engine: each refusal names exactly the stale inputs, one person
    table line each, and every other model-named non-variable column the file
    stores is a register entry."""

    engine = stored_inputs.installed_us_engine()
    tables = _inventories()[name]["tables"]
    refused, registered_count = _EXPECTED_VERDICTS[name]

    failures = stored_input_failures(tables, engine=engine)

    assert [_QUOTED.findall(line) for line in failures] == [
        [column] for column in refused
    ]
    assert all("(person table)" in line for line in failures)
    registered = {
        column
        for columns in tables.values()
        for column in columns
        if is_model_named(column)
        and column not in engine.variables
        and column in US_STORED_NON_VARIABLE_COLUMNS
    }
    assert len(registered) == registered_count
    uppercase = {
        column
        for columns in tables.values()
        for column in columns
        if not is_model_named(column)
    }
    assert not uppercase & engine.variables
