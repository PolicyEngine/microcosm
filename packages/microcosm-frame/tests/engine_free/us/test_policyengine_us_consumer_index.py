from __future__ import annotations

import re
from importlib.util import find_spec
from pathlib import Path
from textwrap import dedent, indent

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.frame.adapters import policyengine_us as module
from microcosm.frame.adapters._policyengine_us_helper_reviews import HelperReview

_HAS_YAML = find_spec("yaml") is not None
_REQUIRES_YAML = pytest.mark.skipif(
    not _HAS_YAML,
    reason="parameter-backed source-index tests require PyYAML",
)


def _clean(source: str) -> str:
    return dedent(source).lstrip()


def _write(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_clean(source), encoding="utf-8")


def _write_parameter(root: Path, dotted_path: str, source: str) -> None:
    relative_path = Path(*dotted_path.split(".")).with_suffix(".yaml")
    _write(root, str(relative_path), source)


def _variable(name: str, body: str = "") -> str:
    source = f"""\
class {name}(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
"""
    if body:
        source += "\n" + indent(dedent(body).strip(), "    ") + "\n"
    return source


def _variables(*names: str) -> str:
    return "\n".join(_variable(name) for name in names)


def _index(
    tmp_path: Path,
    sources: dict[str, str],
    *,
    parameters: dict[str, str] | None = None,
    package: dict[str, str] | None = None,
    helper_reviews: dict[str, HelperReview] | None = None,
) -> module._PolicyEngineUSSourceIndex:
    variables_root = tmp_path / "variables"
    parameters_root = tmp_path / "parameters"
    for relative_path, source in sources.items():
        _write(variables_root, relative_path, source)
    for dotted_path, source in (parameters or {}).items():
        _write_parameter(parameters_root, dotted_path, source)
    # Package modules outside ``variables/`` (``model_api.py``, ``tools/``).
    for relative_path, source in (package or {}).items():
        _write(tmp_path, relative_path, source)
    return module._index_policyengine_us_sources(
        variables_root,
        parameters_root=parameters_root,
        helper_reviews=helper_reviews if helper_reviews is not None else {},
    )


def _receipt_identities(
    index: module._PolicyEngineUSSourceIndex,
    target: str,
) -> set[tuple[str, str, str]]:
    return {
        (receipt.consumer, receipt.path, receipt.kind)
        for receipt in index.consumers.get(target, ())
    }


def _receipt_targets(index: module._PolicyEngineUSSourceIndex) -> set[str]:
    return {name for name, receipts in index.consumers.items() if receipts}


def test_self_exclusion_is_structural_and_same_name_external_helper_counts(
    tmp_path: Path,
) -> None:
    target_source = _variable(
        "target",
        """
        def formula(person, period, parameters):
            return person("target", period)  # own-class reference
        """,
    )
    consumers_source = (
        _variable(
            "sibling",
            """
            def formula(person, period, parameters):
                return person("target", period)  # sibling reference
            """,
        )
        + "\n"
        + _clean(
            """
            def target(person, period):
                return person("target", period)  # same-name external helper
            """
        )
    )

    index = _index(
        tmp_path,
        {
            "target.py": target_source,
            "consumers.py": consumers_source,
        },
    )

    receipts = index.consumers["target"]
    assert _receipt_identities(index, "target") == {
        ("sibling", "variables/consumers.py", "entity_call"),
        ("target", "variables/consumers.py", "entity_call"),
    }
    assert len(receipts) == 2
    assert all(receipt.line > 0 for receipt in receipts)
    assert all(receipt.path != "variables/target.py" for receipt in receipts)
    assert {receipt.receiver_entity for receipt in receipts} == {"person"}


def test_consumer_receipts_capture_the_reference_receiver_entity(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path,
        {
            "consumer.py": _clean(
                """
                class mixed_grain_consumer(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        tax_unit = person.tax_unit
                        return person("direct_leaf", period) + add(
                            tax_unit, period, ["aggregated_leaf"]
                        )

                class grouped_consumer(Variable):
                    value_type = bool
                    entity = SPMUnit
                    definition_period = YEAR

                    def formula(spm_unit, period, parameters):
                        person = spm_unit.members
                        return spm_unit.any(person("grouped_leaf", period))
                """
            ),
            "leaves.py": _variables(
                "direct_leaf",
                "aggregated_leaf",
                "grouped_leaf",
            ),
        },
    )

    (direct,) = index.consumers["direct_leaf"]
    (aggregated,) = index.consumers["aggregated_leaf"]
    (grouped,) = index.consumers["grouped_leaf"]
    assert (direct.kind, direct.receiver_entity, direct.aggregation_entity) == (
        "entity_call",
        "person",
        "",
    )
    assert (
        aggregated.kind,
        aggregated.receiver_entity,
        aggregated.aggregation_entity,
    ) == (
        "add",
        "tax_unit",
        "tax_unit",
    )
    assert (grouped.kind, grouped.receiver_entity, grouped.aggregation_entity) == (
        "entity_call",
        "person",
        "spm_unit",
    )


def test_exact_map_items_and_subscript_do_not_widen_receipts(tmp_path: Path) -> None:
    source = _clean(
        """
        STATE_ITEMS = {
            "CA": {"indiv": "ca_indiv", "joint": "ca_joint"},
            "NY": {"indiv": "ny_indiv", "joint": "ny_joint"},
        }

        class state_itemized(Variable):
            value_type = float
            entity = Person
            definition_period = YEAR

            def formula(person, period, parameters):
                total = 0
                for state, variables in STATE_ITEMS.items():
                    total += person(variables["indiv"], period)
                return total
        """
    )
    index = _index(
        tmp_path,
        {
            "state_itemized.py": source,
            "leaves.py": _variables(
                "ca_indiv",
                "ca_joint",
                "ny_indiv",
                "ny_joint",
            ),
        },
    )

    assert _receipt_targets(index) == {"ca_indiv", "ny_indiv"}
    for target in ("ca_indiv", "ny_indiv"):
        assert _receipt_identities(index, target) == {
            ("state_itemized", "variables/state_itemized.py", "entity_call")
        }
    for invented in ("CA", "NY", "ca_joint", "ny_joint", "indiv", "joint"):
        assert index.consumers.get(invented, ()) == ()


def test_later_import_resolves_without_bare_name_contamination(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        {
            "a_consumer.py": _clean(
                """
                from policyengine_us.variables.z_provider import IMPORTED

                class imported_consumer(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        return add(person, period, IMPORTED)
                """
            ),
            "b_collision.py": 'IMPORTED = ["wrong_name"]\n',
            "z_provider.py": 'IMPORTED = ["right_one", "right_two"]\n',
            "zz_leaves.py": _variables("right_one", "right_two", "wrong_name"),
        },
    )

    assert _receipt_targets(index) == {"right_one", "right_two"}
    for target in ("right_one", "right_two"):
        assert _receipt_identities(index, target) == {
            ("imported_consumer", "variables/a_consumer.py", "add")
        }
    assert index.consumers.get("wrong_name", ()) == ()


def test_package_init_reexport_and_relative_import_resolve_canonically(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path,
        {
            "bundle/__init__.py": "from .source import EXPORTED\n",
            "bundle/source.py": 'EXPORTED = ["reexported_leaf"]\n',
            "consumer.py": _clean(
                """
                from policyengine_us.variables.bundle import EXPORTED

                class reexport_consumer(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        return add(person, period, EXPORTED)
                """
            ),
            "leaf.py": _variable("reexported_leaf"),
        },
    )

    assert _receipt_targets(index) == {"reexported_leaf"}
    assert _receipt_identities(index, "reexported_leaf") == {
        ("reexport_consumer", "variables/consumer.py", "add")
    }


def test_group_member_population_call_is_a_direct_consumer(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        {
            "consumer.py": _variable(
                "member_consumer",
                """
                def formula(spm_unit, period, parameters):
                    return spm_unit.members("member_leaf", period)
                """,
            ),
            "leaf.py": _variable("member_leaf"),
        },
    )

    assert _receipt_identities(index, "member_leaf") == {
        ("member_consumer", "variables/consumer.py", "entity_call")
    }


def test_model_api_reference_helper_indexes_its_variable_list(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        {
            "consumer.py": _variable(
                "helper_consumer",
                """
                def formula(person, period, parameters):
                    return any_(
                        person,
                        period,
                        ["first_helper_leaf", "second_helper_leaf"],
                    )
                """,
            ),
            "leaves.py": _variables("first_helper_leaf", "second_helper_leaf"),
        },
    )

    for target in ("first_helper_leaf", "second_helper_leaf"):
        assert _receipt_identities(index, target) == {
            ("helper_consumer", "variables/consumer.py", "helper_call")
        }


def test_module_qualified_helpers_keep_whole_call_contexts(tmp_path: Path) -> None:
    helper = """
        def read(entity, period, prefix, suffix):
            return entity(prefix + suffix, period)
    """
    index = _index(
        tmp_path,
        {
            "helper_a.py": helper,
            "helper_b.py": helper,
            "consumer.py": _clean(
                """
                from policyengine_us.variables.helper_a import read as read_a
                from policyengine_us.variables.helper_b import read as read_b

                class contextual_consumer(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        left = read_a(person, period, "left", "_a")
                        right = read_a(person, period, "right", "_b")
                        other = read_b(person, period, "other", "_c")
                        return left + right + other
                """
            ),
            "leaves.py": _variables(
                "left_a",
                "right_b",
                "other_c",
                "left_b",
                "right_a",
                "other_a",
            ),
        },
    )

    assert _receipt_targets(index) == {"left_a", "right_b", "other_c"}
    assert _receipt_identities(index, "left_a") == {
        ("read", "variables/helper_a.py", "constructed_entity_call")
    }
    assert _receipt_identities(index, "right_b") == {
        ("read", "variables/helper_a.py", "constructed_entity_call")
    }
    assert _receipt_identities(index, "other_c") == {
        ("read", "variables/helper_b.py", "constructed_entity_call")
    }
    for invented in ("left_b", "right_a", "other_a"):
        assert index.consumers.get(invented, ()) == ()


def test_ast_import_module_qualified_helper_records_receipt(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        {
            "helpers.py": _clean(
                """
                def read(person, period, variable):
                    return person(variable, period)
                """
            ),
            "consumer.py": _clean(
                """
                import policyengine_us.variables.helpers as helpers

                class caller(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        return helpers.read(person, period, "module_leaf")
                """
            ),
        },
    )

    assert _receipt_targets(index) == {"module_leaf"}
    assert _receipt_identities(index, "module_leaf") == {
        ("read", "variables/helpers.py", "entity_call")
    }


def test_ast_import_module_qualified_helper_fails_closed_on_unknown_sink(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="Unresolved dynamic PolicyEngine consumer aggregation",
    ):
        _index(
            tmp_path,
            {
                "helpers.py": _clean(
                    """
                    def read(person, period, variable):
                        return person(variable, period)
                    """
                ),
                "consumer.py": _clean(
                    """
                    import policyengine_us.variables.helpers as helpers

                    class caller(Variable):
                        value_type = float
                        entity = Person
                        definition_period = YEAR

                        def formula(person, period, parameters):
                            return helpers.read(person, period, choose_name())
                    """
                ),
            },
        )


_IMPORT_FROM_MODULE_CASES = (
    pytest.param(
        "from policyengine_us.variables import helpers",
        "helpers",
        id="absolute",
    ),
    pytest.param(
        "from policyengine_us.variables import helpers as imported_helpers",
        "imported_helpers",
        id="aliased-absolute",
    ),
    pytest.param(
        "from . import helpers",
        "helpers",
        id="relative",
    ),
)


@pytest.mark.parametrize(
    ("import_statement", "helper_name"),
    _IMPORT_FROM_MODULE_CASES,
)
def test_import_from_module_qualified_helper_records_receipt(
    tmp_path: Path,
    import_statement: str,
    helper_name: str,
) -> None:
    index = _index(
        tmp_path,
        {
            "helpers.py": _clean(
                """
                def read(person, period, variable):
                    return person(variable, period)
                """
            ),
            "consumer.py": _clean(
                f"""
                {import_statement}

                class caller(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        return {helper_name}.read(
                            person, period, "module_leaf"
                        )
                """
            ),
        },
    )

    assert _receipt_targets(index) == {"module_leaf"}
    assert _receipt_identities(index, "module_leaf") == {
        ("read", "variables/helpers.py", "entity_call")
    }


@pytest.mark.parametrize(
    ("import_statement", "helper_name"),
    _IMPORT_FROM_MODULE_CASES,
)
def test_import_from_module_qualified_helper_fails_closed_on_unknown_sink(
    tmp_path: Path,
    import_statement: str,
    helper_name: str,
) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        _index(
            tmp_path,
            {
                "helpers.py": _clean(
                    """
                    def read(person, period, variable):
                        return person(variable, period)
                    """
                ),
                "consumer.py": _clean(
                    f"""
                    {import_statement}

                    class caller(Variable):
                        value_type = float
                        entity = Person
                        definition_period = YEAR

                        def formula(person, period, parameters):
                            return {helper_name}.read(
                                person, period, choose_name()
                            )
                    """
                ),
            },
        )

    assert str(exc_info.value) == (
        "Unresolved dynamic PolicyEngine consumer aggregation for 'read' at "
        "variables/helpers.py:2: variable."
    )


def test_known_helper_context_cannot_mask_unknown_context(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Unresolved dynamic PolicyEngine consumer"):
        _index(
            tmp_path,
            {
                "known_unknown.py": _clean(
                    """
                    def read(entity, period, variable):
                        return entity(variable, period)

                    class helper_caller(Variable):
                        value_type = float
                        entity = Person
                        definition_period = YEAR

                        def formula(person, period, parameters):
                            known = read(person, period, "known_leaf")
                            unknown = read(person, period, choose_name())
                            return known + unknown
                    """
                ),
                "leaf.py": _variable("known_leaf"),
            },
        )


def test_unknown_survives_every_supported_string_transform(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Unresolved dynamic PolicyEngine consumer"):
        _index(
            tmp_path,
            {
                "unknown_transforms.py": _clean(
                    """
                    class transformed_consumer(Variable):
                        value_type = float
                        entity = Person
                        definition_period = YEAR

                        def formula(person, period, parameters):
                            unknown = choose_name()
                            names = sorted(set(list([unknown])))
                            transformed = [
                                f"{part.lower().replace('x', 'y')}_suffix"
                                for name in names
                                for part in name.split()
                            ]
                            return person(transformed[0], period)
                    """
                ),
            },
        )


@_REQUIRES_YAML
def test_conditional_parameter_loop_replaces_only_matching_values(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path,
        {
            "conditional_loop.py": _clean(
                """
                CIRCULAR_SOURCES = {"old_source": "replacement_source"}

                class conditional_consumer(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        sources = parameters(period).group.sources
                        total = 0
                        for source in sources:
                            if source in CIRCULAR_SOURCES:
                                source = CIRCULAR_SOURCES[source]
                            total += add(person, period, [source])
                        return total
                """
            ),
            "leaves.py": _variables(
                "keep_source",
                "old_source",
                "replacement_source",
            ),
        },
        parameters={
            "group.sources": """
                values:
                  2020-01-01:
                    - keep_source
                    - old_source
            """,
        },
    )

    assert _receipt_targets(index) == {"keep_source", "replacement_source"}
    for target in ("keep_source", "replacement_source"):
        assert _receipt_identities(index, target) == {
            (
                "conditional_consumer",
                "variables/conditional_loop.py",
                "parameter_add",
            )
        }
    assert index.consumers.get("old_source", ()) == ()


@_REQUIRES_YAML
def test_historical_parameters_aliases_and_class_reference_surfaces(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path,
        {
            "aggregates.py": _clean(
                """
                class class_aggregate(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR
                    adds = "group.history"
                    subtracts = ["literal_subtracted"]
                    defined_for = "eligibility_flag"

                class alias_consumer(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        group = parameters(period).group
                        aliased = group.alias
                        return add(person, period, aliased)

                class converted_consumer(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        all_alds = parameters(period).group.all_alds
                        person_alds = ["excluded_ald"]
                        other_alds = list(set(all_alds) - set(person_alds))
                        return add(person, period, other_alds)
                """
            ),
            "leaves.py": _variables(
                "historic_old",
                "historic_new",
                "historic_shared",
                "alias_first",
                "alias_second",
                "literal_subtracted",
                "eligibility_flag",
                "converted_keep",
                "excluded_ald",
            ),
        },
        parameters={
            "group.history": """
                values:
                  2010-01-01:
                    - historic_old
                    - historic_shared
                  2020-01-01:
                    - historic_new
                    - historic_shared
            """,
            "group.alias": """
                values:
                  2018-01-01:
                    - alias_first
                  2024-01-01:
                    - alias_second
            """,
            "group.all_alds": """
                values:
                  2010-01-01:
                    - converted_keep
                    - excluded_ald
                  2020-01-01:
                    - converted_keep
            """,
        },
    )

    for target in ("historic_old", "historic_new", "historic_shared"):
        assert _receipt_identities(index, target) == {
            ("class_aggregate", "variables/aggregates.py", "parameter_adds")
        }
    for target in ("alias_first", "alias_second"):
        assert _receipt_identities(index, target) == {
            ("alias_consumer", "variables/aggregates.py", "parameter_add")
        }
    assert _receipt_identities(index, "literal_subtracted") == {
        ("class_aggregate", "variables/aggregates.py", "subtracts")
    }
    assert _receipt_identities(index, "eligibility_flag") == {
        ("class_aggregate", "variables/aggregates.py", "defined_for")
    }
    assert _receipt_identities(index, "converted_keep") == {
        ("converted_consumer", "variables/aggregates.py", "parameter_add")
    }
    assert index.consumers.get("excluded_ald", ()) == ()


@_REQUIRES_YAML
def test_qbi_parameter_loop_constructs_exact_names_and_kinds(tmp_path: Path) -> None:
    bases = (
        "self_employment_income",
        "partnership_s_corp_income",
        "farm_rent_income",
        "farm_operations_income",
        "rental_income",
        "estate_income",
    )
    constructed = tuple(f"{name}_would_be_qualified" for name in bases)
    index = _index(
        tmp_path,
        {
            "qbi.py": _clean(
                """
                class qbi_consumer(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        p = parameters(period).group
                        total = 0
                        for variable in p.income_definition:
                            total += person(variable, period)
                            total += person(
                                variable + "_would_be_qualified", period
                            )
                        return total
                """
            ),
            "leaves.py": _variables(*bases, *constructed),
        },
        parameters={
            "group.income_definition": """
                values:
                  2018-01-01:
                    - self_employment_income
                    - partnership_s_corp_income
                    - farm_rent_income
                    - farm_operations_income
                    - rental_income
                    - estate_income
            """,
        },
    )

    assert _receipt_targets(index) == set(bases) | set(constructed)
    for target in bases:
        assert _receipt_identities(index, target) == {
            ("qbi_consumer", "variables/qbi.py", "parameter_entity_call")
        }
    for target in constructed:
        assert _receipt_identities(index, target) == {
            (
                "qbi_consumer",
                "variables/qbi.py",
                "constructed_parameter_entity_call",
            )
        }


@_REQUIRES_YAML
def test_list_append_and_extend_preserve_exact_members_and_provenance(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path,
        {
            "mutating_list.py": _clean(
                """
                class mutating_consumer(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        items = []
                        items.append("literal_append")
                        items.extend(["literal_extend"])
                        more = parameters(period).group.more
                        items.extend(more)
                        return add(person, period, items)
                """
            ),
            "leaves.py": _variables(
                "literal_append",
                "literal_extend",
                "parameter_old",
                "parameter_new",
            ),
        },
        parameters={
            "group.more": """
                values:
                  2010-01-01:
                    - parameter_old
                  2020-01-01:
                    - parameter_new
            """,
        },
    )

    for target in ("literal_append", "literal_extend"):
        assert _receipt_identities(index, target) == {
            ("mutating_consumer", "variables/mutating_list.py", "add")
        }
    for target in ("parameter_old", "parameter_new"):
        assert _receipt_identities(index, target) == {
            (
                "mutating_consumer",
                "variables/mutating_list.py",
                "parameter_add",
            )
        }


@_REQUIRES_YAML
def test_override_table_get_resolves_each_parameter_key_exactly(
    tmp_path: Path,
) -> None:
    """``mapping.get(key, key)`` over a parameter loop stays exact.

    This is ``snap_individual_utility_allowance``'s shape in policyengine-us
    2.2.1: the SNAP utility-allowance rework maps each entry of the
    parameter-backed ``utility_types`` list through a module-level override
    table before reading it as an SPM-unit variable. Overridden entries must
    resolve to the override alone and unlisted entries to themselves; neither
    may widen to the whole table, and the unlisted default must not be lost.
    """

    index = _index(
        tmp_path,
        {
            "override_get.py": _clean(
                """
                EXPENSE_VARIABLE_OVERRIDES = {
                    "electricity_expense": "pre_subsidy_electricity_expense",
                    "gas_and_fuel_expense": "has_gas_and_fuel_expense",
                }

                class utility_allowance(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        expense_types = parameters(period).group.utility_types
                        total = 0
                        for expense in expense_types:
                            variable = EXPENSE_VARIABLE_OVERRIDES.get(
                                expense, expense
                            )
                            total += person(variable, period)
                        return total
                """
            ),
            "leaves.py": _variables(
                "electricity_expense",
                "pre_subsidy_electricity_expense",
                "gas_and_fuel_expense",
                "has_gas_and_fuel_expense",
                "water_expense",
                "unlisted_expense",
            ),
        },
        parameters={
            "group.utility_types": """
                values:
                  2020-01-01:
                    - electricity_expense
                    - gas_and_fuel_expense
                    - water_expense
            """,
        },
    )

    assert _receipt_targets(index) == {
        "pre_subsidy_electricity_expense",
        "has_gas_and_fuel_expense",
        "water_expense",
    }
    for target in _receipt_targets(index):
        assert _receipt_identities(index, target) == {
            (
                "utility_allowance",
                "variables/override_get.py",
                "parameter_entity_call",
            )
        }
    # The overridden keys are consumed only through their replacements, and a
    # name that is merely a table value for another key is never widened in.
    for absent in ("electricity_expense", "gas_and_fuel_expense", "unlisted_expense"):
        assert index.consumers.get(absent, ()) == ()


def test_single_argument_get_still_fails_closed(tmp_path: Path) -> None:
    """``mapping.get(key)`` with a key read from the engine still fails closed.

    A one-argument miss is modeled as ``None`` only for exact keys; a key the
    engine computes at run time names no table entry statically.
    """

    with pytest.raises(RuntimeError, match="Unresolved dynamic"):
        _index(
            tmp_path,
            {
                "bare_get.py": _clean(
                    """
                    OVERRIDES = {"a": "leaf_a"}

                    class bare_get_consumer(Variable):
                        value_type = float
                        entity = Person
                        definition_period = YEAR

                        def formula(person, period, parameters):
                            key = person("selector", period)
                            return person(OVERRIDES.get(key), period)
                    """
                ),
                "leaves.py": _variables("leaf_a", "selector"),
            },
        )


@_REQUIRES_YAML
def test_starred_items_splice_module_and_parameter_lists(tmp_path: Path) -> None:
    """``[*CONSTANT]`` and ``[*parameter_list]`` displays resolve exactly.

    policyengine-us 2.60 writes ``add(tax_unit, period, ["adjusted_gross_income",
    *TAX_UNIT_ADDITIONS])`` and loops over ``["ssi_earned_income",
    *ssi_sources.unearned]``.
    """

    index = _index(
        tmp_path,
        {
            "starred.py": _clean(
                """
                ADDITIONS = ["addition_a", "addition_b"]

                class module_starred(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        return add(person, period, ["base_income", *ADDITIONS])

                class parameter_starred(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        sources = parameters(period).group.sources
                        total = 0
                        for source in ["earned", *sources]:
                            total += person(source, period)
                        return total
                """
            ),
            "leaves.py": _variables(
                "base_income",
                "addition_a",
                "addition_b",
                "earned",
                "unearned_a",
                "unearned_b",
            ),
        },
        parameters={
            "group.sources": """
                values:
                  2020-01-01:
                    - unearned_a
                  2024-01-01:
                    - unearned_a
                    - unearned_b
            """,
        },
    )

    assert {
        target
        for target in _receipt_targets(index)
        if any(r[0] == "module_starred" for r in _receipt_identities(index, target))
    } == {"base_income", "addition_a", "addition_b"}
    assert {
        target
        for target in _receipt_targets(index)
        if any(r[0] == "parameter_starred" for r in _receipt_identities(index, target))
    } == {"earned", "unearned_a", "unearned_b"}


@_REQUIRES_YAML
def test_single_argument_get_resolves_an_is_none_skip_exactly(
    tmp_path: Path,
) -> None:
    """A miss of ``mapping.get(key)`` is None, and ``is None`` decides it.

    This is ``de_itemized_deductions_indv``'s shape in policyengine-us 2.60:
    deductions without an ``EXPENSE_SOURCES`` entry are skipped.
    """

    index = _index(
        tmp_path,
        {
            "get_none.py": _clean(
                """
                EXPENSE_SOURCES = {
                    "deduction_a": ["expense_a"],
                    "deduction_b": ["expense_b1", "expense_b2"],
                }

                class allocated(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        total = 0
                        for deduction in parameters(period).group.deductions:
                            expense_sources = EXPENSE_SOURCES.get(deduction)
                            if expense_sources is None:
                                continue
                            total += add(person, period, expense_sources)
                        return total
                """
            ),
            "leaves.py": _variables(
                "expense_a", "expense_b1", "expense_b2", "expense_unused"
            ),
        },
        parameters={
            "group.deductions": """
                values:
                  2020-01-01:
                    - deduction_a
                    - deduction_b
                    - deduction_without_sources
            """,
        },
    )

    assert _receipt_targets(index) == {"expense_a", "expense_b1", "expense_b2"}


def test_unknown_key_on_an_exact_string_map_reads_every_value(
    tmp_path: Path,
) -> None:
    """A run-time key into a fully known name table may read any of its values.

    This is ``capital_gains_behavioral_response``'s shape in policyengine-us
    2.60: the form comes from parameter values, so every form's variable is a
    possible read.
    """

    index = _index(
        tmp_path,
        {
            "forms.py": _clean(
                """
                FORM_VARIABLES = {"a": "form_a_parameter", "b": "form_b_parameter"}

                def selected_form(values):
                    nonzero = {form: value for form, value in values if value}
                    if not nonzero:
                        return None
                    form, value = next(iter(nonzero.items()))
                    return form

                class response(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        form = selected_form(parameters(period).group.forms)
                        if form is None:
                            return 0
                        return person(FORM_VARIABLES[form], period)
                """
            ),
            "leaves.py": _variables("form_a_parameter", "form_b_parameter"),
        },
    )

    assert _receipt_targets(index) == {"form_a_parameter", "form_b_parameter"}


@_REQUIRES_YAML
def test_helper_tuple_return_flows_into_the_callers_sink(tmp_path: Path) -> None:
    """``income, deductions = helper(...)`` binds the helper's returned list.

    This is the policyengine-us 2.60 ``dependent_income_and_deductions``
    shape: the helper maps a parameter list through a module table, and one
    caller filters itself out before summing the rest.
    """

    index = _index(
        tmp_path,
        {
            "helper.py": _clean(
                """
                PERSON_DEDUCTIONS = {
                    "deduction_a": "deduction_a_person",
                    "own_deduction": "own_deduction_person",
                }

                def income_and_deductions(person, period, parameters):
                    listed = parameters(period).group.deductions
                    deductions = [
                        PERSON_DEDUCTIONS[item]
                        for item in listed
                        if item in PERSON_DEDUCTIONS
                    ]
                    return person("gross_income", period), deductions
                """
            ),
            "consumers.py": _clean(
                """
                from policyengine_us.variables.helper import income_and_deductions

                class all_deductions(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        income, deductions = income_and_deductions(
                            person, period, parameters
                        )
                        return income - add(person, period, deductions)

                class own_deduction_person(Variable):
                    value_type = float
                    entity = Person
                    definition_period = YEAR

                    def formula(person, period, parameters):
                        income, deductions = income_and_deductions(
                            person, period, parameters
                        )
                        others = [d for d in deductions if d != "own_deduction_person"]
                        return income - add(person, period, others)
                """
            ),
            "leaves.py": _variables("gross_income", "deduction_a_person"),
        },
        parameters={
            "group.deductions": """
                values:
                  2020-01-01:
                    - deduction_a
                    - own_deduction
                    - deduction_without_person_leaf
            """,
        },
    )

    def consumed_by(consumer: str) -> set[str]:
        return {
            target
            for target in _receipt_targets(index)
            if any(r[0] == consumer for r in _receipt_identities(index, target))
        }

    assert consumed_by("all_deductions") >= {
        "deduction_a_person",
        "own_deduction_person",
    }
    assert "own_deduction_person" not in consumed_by("own_deduction_person")
    assert "deduction_a_person" in consumed_by("own_deduction_person")


def test_helper_return_of_an_engine_value_still_fails_closed(
    tmp_path: Path,
) -> None:
    """A helper that returns a run-time value cannot name a variable."""

    with pytest.raises(RuntimeError, match="Unresolved dynamic"):
        _index(
            tmp_path,
            {
                "passthrough.py": _clean(
                    """
                    def pick(name):
                        return name

                    class picked(Variable):
                        value_type = float
                        entity = Person
                        definition_period = YEAR

                        def formula(person, period, parameters):
                            name = pick(person("selector", period))
                            return person(name, period)
                    """
                ),
                "leaves.py": _variables("selector"),
            },
        )


_NAMES = st.lists(
    st.from_regex(r"leaf_[a-z]{2,6}", fullmatch=True),
    min_size=1,
    max_size=5,
    unique=True,
)


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(first=_NAMES, middle=_NAMES, last=_NAMES)
def test_starred_display_reads_exactly_the_union_of_its_parts(
    tmp_path_factory: pytest.TempPathFactory,
    first: list[str],
    middle: list[str],
    last: list[str],
) -> None:
    """Property: ``[*FIRST, literal..., *LAST]`` reads every item and nothing else."""

    root = tmp_path_factory.mktemp("starred")
    module = _clean(
        f"""
        FIRST = {first!r}
        LAST = {last!r}

        class spliced(Variable):
            value_type = float
            entity = Person
            definition_period = YEAR

            def formula(person, period, parameters):
                return add(person, period, [*FIRST, {", ".join(map(repr, middle))}, *LAST])
        """
    )
    leaves = sorted({*first, *middle, *last})
    index = _index(root, {"spliced.py": module, "leaves.py": _variables(*leaves)})

    assert _receipt_targets(index) == set(leaves)


# Package helpers outside ``variables/`` -------------------------------------

_MODEL_API = "from policyengine_us.tools.general import *\n"
_GENERAL = """
LEAVES = ("first_leaf", "second_leaf")


def tax_unit_non_dep_sum(var, tax_unit, period):
    return tax_unit.sum(
        tax_unit.members(var, period)
        * not_(tax_unit.members("is_dependent", period))
    )


def tax_unit_non_dep_add(tax_unit, period, variables, include_dependents=()):
    \"\"\"Add variables over non-dependents.\"\"\"
    total = 0
    for variable in variables:
        total = total + tax_unit_non_dep_sum(variable, tax_unit, period)
    return total


def eitc_like(tax_unit, period, child_count=None):
    if child_count is None:
        child_count = tax_unit("child_count", period)
    return child_count * tax_unit("takes_up", period)


def any_leaf(tax_unit, period):
    return any(tax_unit(leaf, period) for leaf in LEAVES)


def sum_among_non_dependents(variable):
    def formula(tax_unit, period, parameters):
        return tax_unit_non_dep_sum(variable, tax_unit, period)

    return formula
"""
_HELPER_LEAVES = _variables(
    "wages",
    "interest",
    "is_dependent",
    "child_count",
    "takes_up",
    "first_leaf",
    "second_leaf",
)
_HELPER_REVIEW_SPECS = {
    "policyengine_us.tools.general.tax_unit_non_dep_add": {
        "arguments": ((2, "variables"),),
        "reads": ("is_dependent",),
    },
    "policyengine_us.tools.general.tax_unit_non_dep_sum": {
        "arguments": ((0, "var"),),
        "reads": ("is_dependent",),
    },
    "policyengine_us.tools.general.eitc_like": {
        "reads": ("takes_up",),
        "reads_when_omitted": ((2, "child_count", ("child_count",)),),
    },
    "policyengine_us.tools.general.any_leaf": {
        "reads": ("first_leaf", "second_leaf"),
    },
    "policyengine_us.tools.general.sum_among_non_dependents": {
        "arguments": ((0, "variable"),),
        "reads": ("is_dependent",),
    },
}


def _tax_unit_variable(name: str, body: str) -> str:
    return (
        _clean(
            f"""
        from policyengine_us.model_api import *

        class {name}(Variable):
            value_type = float
            entity = TaxUnit
            definition_period = YEAR

        """
        )
        + indent(dedent(body).strip(), "    ")
        + "\n"
    )


def _helper_digest_error(exc: pytest.ExceptionInfo[RuntimeError]) -> tuple[str, str]:
    match = re.search(
        r"helper (\S+) (?:\(digest|changed \(digest) ([0-9a-f]{64})",
        str(exc.value),
    )
    assert match is not None, str(exc.value)
    return match.group(1), match.group(2)


def _reviewed_index(
    tmp_path: Path,
    sources: dict[str, str],
    *,
    package: dict[str, str],
    specs: dict[str, dict[str, object]] | None = None,
) -> module._PolicyEngineUSSourceIndex:
    """Index ``sources``, reviewing each helper as a reviewer would.

    Every unreviewed helper stops the index with its digest; the review is
    then written from ``specs`` at that digest and the index rerun.
    """

    reviews: dict[str, HelperReview] = {}
    specs = _HELPER_REVIEW_SPECS if specs is None else specs
    for _ in range(len(specs) + 1):
        try:
            return _index(
                tmp_path,
                sources,
                package=package,
                helper_reviews=reviews,
            )
        except RuntimeError as exc:
            match = re.search(
                r"Unreviewed PolicyEngine-US helper (\S+) \(digest ([0-9a-f]{64})\)",
                str(exc),
            )
            if match is None:
                raise
            name, digest = match.groups()
            reviews[name] = HelperReview(digest=digest, **specs[name])  # type: ignore[arg-type]
    raise AssertionError("helper reviews did not converge")


def test_unreviewed_package_helper_fails_closed_with_its_digest(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError) as exc:
        _index(
            tmp_path,
            {
                "filer_total.py": _tax_unit_variable(
                    "filer_total",
                    """
                    def formula(tax_unit, period, parameters):
                        return tax_unit_non_dep_add(tax_unit, period, ["wages"])
                    """,
                ),
                "leaves.py": _HELPER_LEAVES,
            },
            package={"model_api.py": _MODEL_API, "tools/general.py": _GENERAL},
        )
    assert "Unreviewed PolicyEngine-US helper" in str(exc.value)
    assert _helper_digest_error(exc)[0] == (
        "policyengine_us.tools.general.tax_unit_non_dep_add"
    )


def test_reviewed_package_helper_reads_belong_to_the_caller(tmp_path: Path) -> None:
    """Named variables and the helper's own reads are the caller's receipts."""

    index = _reviewed_index(
        tmp_path,
        {
            "filer_total.py": _tax_unit_variable(
                "filer_total",
                """
                def formula(tax_unit, period, parameters):
                    return tax_unit_non_dep_add(
                        tax_unit, period, ["wages", "interest"]
                    ) + any_leaf(tax_unit, period)
                """,
            ),
            "leaves.py": _HELPER_LEAVES,
        },
        package={"model_api.py": _MODEL_API, "tools/general.py": _GENERAL},
    )

    expected = {("filer_total", "variables/filer_total.py", "helper_call")}
    for target in ("wages", "interest", "is_dependent", "first_leaf", "second_leaf"):
        assert _receipt_identities(index, target) == expected
    assert _receipt_targets(index) == {
        "wages",
        "interest",
        "is_dependent",
        "first_leaf",
        "second_leaf",
    }


def test_helper_reads_tied_to_an_omitted_argument(tmp_path: Path) -> None:
    index = _reviewed_index(
        tmp_path,
        {
            "passes_count.py": _tax_unit_variable(
                "passes_count",
                """
                def formula(tax_unit, period, parameters):
                    return eitc_like(tax_unit, period, 2)
                """,
            ),
            "omits_count.py": _tax_unit_variable(
                "omits_count",
                """
                def formula(tax_unit, period, parameters):
                    return eitc_like(tax_unit, period, child_count=None)
                """,
            ),
            "leaves.py": _HELPER_LEAVES,
        },
        package={"model_api.py": _MODEL_API, "tools/general.py": _GENERAL},
    )

    assert {receipt.consumer for receipt in index.consumers["takes_up"]} == {
        "passes_count",
        "omits_count",
    }
    assert {receipt.consumer for receipt in index.consumers["child_count"]} == {
        "omits_count"
    }


def test_class_body_helper_factory_reads_belong_to_the_class(tmp_path: Path) -> None:
    index = _reviewed_index(
        tmp_path,
        {
            "earned.py": _tax_unit_variable(
                "earned",
                """
                formula = sum_among_non_dependents("wages")
                """,
            ),
            "leaves.py": _HELPER_LEAVES,
        },
        package={"model_api.py": _MODEL_API, "tools/general.py": _GENERAL},
    )

    assert _receipt_identities(index, "wages") == {
        ("earned", "variables/earned.py", "helper_call")
    }
    assert _receipt_identities(index, "is_dependent") == {
        ("earned", "variables/earned.py", "helper_call")
    }


def test_class_body_call_to_a_formula_module_function_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="Unresolved dynamic PolicyEngine consumer aggregation",
    ):
        _index(
            tmp_path,
            {
                "factories.py": _clean(
                    """
                    def make(variable):
                        def formula(person, period, parameters):
                            return person(variable, period)

                        return formula
                    """
                ),
                "made.py": _clean(
                    """
                    from policyengine_us.variables.factories import make

                    class made(Variable):
                        value_type = float
                        entity = Person
                        definition_period = YEAR
                        formula = make("wages")
                    """
                ),
                "leaves.py": _HELPER_LEAVES,
            },
        )


def test_local_name_shadowing_a_helper_is_not_the_helper(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        {
            "shadowed.py": _tax_unit_variable(
                "shadowed",
                """
                def formula(tax_unit, period, parameters):
                    tax_unit_non_dep_add = choose_function()
                    return tax_unit_non_dep_add(tax_unit, period, ["wages"])
                """,
            ),
            "leaves.py": _HELPER_LEAVES,
        },
        package={"model_api.py": _MODEL_API, "tools/general.py": _GENERAL},
    )

    assert _receipt_targets(index) == set()


def test_helper_digest_ignores_docstrings_and_follows_code_and_constants(
    tmp_path: Path,
) -> None:
    def digest(root: Path, general: str) -> str:
        with pytest.raises(RuntimeError) as exc:
            _index(
                root,
                {
                    "caller.py": _tax_unit_variable(
                        "caller",
                        """
                        def formula(tax_unit, period, parameters):
                            return any_leaf(tax_unit, period)
                        """,
                    ),
                    "leaves.py": _HELPER_LEAVES,
                },
                package={"model_api.py": _MODEL_API, "tools/general.py": general},
            )
        return _helper_digest_error(exc)[1]

    base = digest(tmp_path / "base", _GENERAL)
    redocumented = _GENERAL.replace(
        "def any_leaf(tax_unit, period):\n",
        'def any_leaf(tax_unit, period):\n    """Any leaf is positive."""\n',
    )
    recoded = _GENERAL.replace(
        "return any(tax_unit(leaf, period) for leaf in LEAVES)",
        "return all(tax_unit(leaf, period) for leaf in LEAVES)",
    )
    reconstant = _GENERAL.replace(
        'LEAVES = ("first_leaf", "second_leaf")',
        'LEAVES = ("first_leaf",)',
    )

    assert digest(tmp_path / "doc", redocumented) == base
    assert digest(tmp_path / "code", recoded) != base
    assert digest(tmp_path / "constant", reconstant) != base


def test_changed_helper_fails_closed_until_reviewed_again(tmp_path: Path) -> None:
    sources = {
        "caller.py": _tax_unit_variable(
            "caller",
            """
            def formula(tax_unit, period, parameters):
                return any_leaf(tax_unit, period)
            """,
        ),
        "leaves.py": _HELPER_LEAVES,
    }
    reviewed = _reviewed_index(
        tmp_path / "reviewed",
        sources,
        package={"model_api.py": _MODEL_API, "tools/general.py": _GENERAL},
    )
    assert _receipt_targets(reviewed) == {"first_leaf", "second_leaf"}
    with pytest.raises(RuntimeError) as exc:
        _index(
            tmp_path / "changed",
            sources,
            package={
                "model_api.py": _MODEL_API,
                "tools/general.py": _GENERAL.replace(
                    'LEAVES = ("first_leaf", "second_leaf")',
                    'LEAVES = ("first_leaf", "second_leaf", "wages")',
                ),
            },
            helper_reviews={
                "policyengine_us.tools.general.any_leaf": HelperReview(
                    digest=_helper_digest_error_free_digest(tmp_path, sources),
                    reads=("first_leaf", "second_leaf"),
                )
            },
        )
    assert "changed (digest" in str(exc.value)


def _helper_digest_error_free_digest(tmp_path: Path, sources: dict[str, str]) -> str:
    with pytest.raises(RuntimeError) as exc:
        _index(
            tmp_path / "original",
            sources,
            package={"model_api.py": _MODEL_API, "tools/general.py": _GENERAL},
        )
    return _helper_digest_error(exc)[1]


def test_review_naming_a_non_variable_read_fails(tmp_path: Path) -> None:
    specs = {
        "policyengine_us.tools.general.any_leaf": {
            "reads": ("first_leaf", "not_a_variable"),
        }
    }
    with pytest.raises(RuntimeError, match="names reads that are not variables"):
        _reviewed_index(
            tmp_path,
            {
                "caller.py": _tax_unit_variable(
                    "caller",
                    """
                    def formula(tax_unit, period, parameters):
                        return any_leaf(tax_unit, period)
                    """,
                ),
                "leaves.py": _HELPER_LEAVES,
            },
            package={"model_api.py": _MODEL_API, "tools/general.py": _GENERAL},
            specs=specs,
        )


# Simulation reads, local functions and lambdas ---------------------------------


def test_simulation_calculate_holder_and_array_reads(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        {
            "branch_reader.py": _tax_unit_variable(
                "branch_reader",
                """
                def formula(tax_unit, period, parameters):
                    simulation = tax_unit.simulation
                    branch = simulation.get_branch("counterfactual")
                    holder = simulation.get_holder("stored_leaf")
                    stored = holder.get_array(period, simulation.branch_name)
                    supplied = simulation.get_array("supplied_leaf", period)
                    return branch.calculate("branch_leaf", period) + stored + supplied
                """,
            ),
            "leaves.py": _variables("stored_leaf", "supplied_leaf", "branch_leaf"),
        },
    )

    for target in ("stored_leaf", "supplied_leaf", "branch_leaf"):
        assert _receipt_identities(index, target) == {
            ("branch_reader", "variables/branch_reader.py", "simulation_read")
        }
    assert _receipt_targets(index) == {"stored_leaf", "supplied_leaf", "branch_leaf"}


def test_simulation_read_of_an_unknown_name_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeError,
        match="Unresolved dynamic PolicyEngine consumer aggregation",
    ):
        _index(
            tmp_path,
            {
                "branch_reader.py": _tax_unit_variable(
                    "branch_reader",
                    """
                    def formula(tax_unit, period, parameters):
                        branch = tax_unit.simulation.get_branch("counterfactual")
                        return branch.calculate(choose_name(), period)
                    """,
                ),
            },
        )


def test_local_function_reads_are_exact_and_belong_to_the_caller(
    tmp_path: Path,
) -> None:
    """A recursive local function reads exactly what its calls reach."""

    index = _index(
        tmp_path,
        {
            "lookback.py": _clean(
                """
                NAMES = ("first_leaf", "third_leaf")

                class lookback(Variable):
                    value_type = float
                    entity = TaxUnit
                    definition_period = YEAR

                    def formula(tax_unit, period, parameters):
                        def provided(variable):
                            if variable == "first_leaf":
                                return provided("second_leaf")
                            return tax_unit(variable, period)

                        total = 0
                        for name in NAMES:
                            total = total + provided(name)
                        return total
                """
            ),
            "leaves.py": _variables("first_leaf", "second_leaf", "third_leaf"),
        },
    )

    assert _receipt_targets(index) == {"second_leaf", "third_leaf"}
    assert _receipt_identities(index, "second_leaf") == {
        ("lookback", "variables/lookback.py", "entity_call")
    }


def test_escaping_local_function_is_read_where_defined(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        {
            "unused.py": _tax_unit_variable(
                "unused",
                """
                def formula(tax_unit, period, parameters):
                    def never_called():
                        return tax_unit("dead_leaf", period)

                    return 0
                """,
            ),
            "leaves.py": _variables("dead_leaf"),
        },
    )
    assert _receipt_identities(index, "dead_leaf") == {
        ("unused", "variables/unused.py", "entity_call")
    }

    with pytest.raises(
        RuntimeError,
        match="Unresolved dynamic PolicyEngine consumer aggregation",
    ):
        _index(
            tmp_path / "escaping",
            {
                "mapped.py": _tax_unit_variable(
                    "mapped",
                    """
                    def formula(tax_unit, period, parameters):
                        def read(name):
                            return tax_unit(name, period)

                        return sum(map(read, ["first_leaf"]))
                    """,
                ),
                "leaves.py": _variables("first_leaf"),
            },
        )


def test_lambda_reads_are_recorded_and_unknown_names_fail_closed(
    tmp_path: Path,
) -> None:
    index = _index(
        tmp_path,
        {
            "deferred.py": _tax_unit_variable(
                "deferred",
                """
                def formula(tax_unit, period, parameters):
                    read = lambda: tax_unit("lambda_leaf", period)
                    return read()
                """,
            ),
            "leaves.py": _variables("lambda_leaf"),
        },
    )
    assert _receipt_identities(index, "lambda_leaf") == {
        ("deferred", "variables/deferred.py", "entity_call")
    }

    with pytest.raises(
        RuntimeError,
        match="Unresolved dynamic PolicyEngine consumer aggregation",
    ):
        _index(
            tmp_path / "dynamic",
            {
                "folded.py": _tax_unit_variable(
                    "folded",
                    """
                    def formula(tax_unit, period, parameters):
                        return reduce(
                            lambda total, name: total + tax_unit(name, period),
                            ["lambda_leaf"],
                            0,
                        )
                    """,
                ),
                "leaves.py": _variables("lambda_leaf"),
            },
        )


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(names=_NAMES)
def test_reviewed_helper_reads_exactly_the_listed_variables_and_its_own(
    tmp_path_factory: pytest.TempPathFactory,
    names: list[str],
) -> None:
    """Property: ``tax_unit_non_dep_add(..., names)`` reads names and nothing else."""

    root = tmp_path_factory.mktemp("helper")
    index = _reviewed_index(
        root,
        {
            "filer_total.py": _tax_unit_variable(
                "filer_total",
                f"""
                def formula(tax_unit, period, parameters):
                    return tax_unit_non_dep_add(tax_unit, period, {names!r})
                """,
            ),
            "leaves.py": _variables(*names, "is_dependent"),
        },
        package={"model_api.py": _MODEL_API, "tools/general.py": _GENERAL},
    )

    assert _receipt_targets(index) == {*names, "is_dependent"}


def test_comparing_an_engine_value_keeps_both_outcomes(tmp_path: Path) -> None:
    """``period.start.year != 2021`` is undecided; exact data still decides."""

    index = _index(
        tmp_path,
        {
            "by_year.py": _tax_unit_variable(
                "by_year",
                """
                def formula(tax_unit, period, parameters):
                    if period.start.year != 2021:
                        return tax_unit("first_leaf", period)
                    if "second_leaf" in ["second_leaf"]:
                        return tax_unit("second_leaf", period)
                    return tax_unit("third_leaf", period)
                """,
            ),
            "leaves.py": _variables("first_leaf", "second_leaf", "third_leaf"),
        },
    )

    assert _receipt_targets(index) == {"first_leaf", "second_leaf"}
