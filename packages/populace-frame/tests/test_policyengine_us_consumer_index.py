from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from textwrap import dedent, indent

import pytest

from populace.frame.adapters import policyengine_us as module

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
) -> module._PolicyEngineUSSourceIndex:
    variables_root = tmp_path / "variables"
    parameters_root = tmp_path / "parameters"
    for relative_path, source in sources.items():
        _write(variables_root, relative_path, source)
    for dotted_path, source in (parameters or {}).items():
        _write_parameter(parameters_root, dotted_path, source)
    return module._index_policyengine_us_sources(
        variables_root,
        parameters_root=parameters_root,
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
