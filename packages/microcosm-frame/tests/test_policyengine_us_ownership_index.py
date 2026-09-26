"""Ownership uses the real declaration parser without consumer parameter I/O."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from microcosm.frame.adapters import _policyengine_us_source_index as source
from microcosm.frame.adapters import policyengine_us as adapter


def _variables(tmp_path):
    root = tmp_path / "variables"
    root.mkdir()
    (root / "example.py").write_text("""
from policyengine_us.model_api import *

class earnings(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR

class deduction(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR

class computed(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    adds = "gov.example.components"

class dated(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    def formula_2024(person, period, parameters):
        return person("earnings", period)
""")
    return root


def test_same_metadata_parser_without_parameter_reads_then_full_consumer_parity(
    tmp_path, monkeypatch
):
    variables = _variables(tmp_path)
    parameters = tmp_path / "parameters"
    path = parameters / "gov/example/components.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("values:\n  2024-01-01:\n    value: [earnings, deduction]\n")
    reads = []
    original = Path.read_text

    def observed(self, *args, **kwargs):
        reads.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", observed)
    definitions = source._index_policyengine_us_variable_sources(variables)
    assert not any(p.suffix == ".yaml" for p in reads)
    assert definitions["computed"].formula_owned
    assert definitions["dated"].computed_at(2024)
    assert not definitions["dated"].computed_at(2023)
    assert not definitions["earnings"].formula_owned
    complete = source._index_policyengine_us_sources(
        variables, parameters_root=parameters
    )
    assert complete.definitions == definitions
    assert path in reads
    assert any(r.consumer == "computed" for r in complete.consumers["earnings"])
    assert adapter._index_policyengine_us_variable_sources(variables) == definitions


def test_unavailable_consumer_parameters_do_not_invalidate_formula_ownership(tmp_path):
    variables = _variables(tmp_path)
    definitions = source._index_policyengine_us_variable_sources(variables)
    assert definitions["computed"].formula_owned
    with pytest.raises(RuntimeError, match="Unresolved dynamic"):
        source._index_policyengine_us_sources(
            variables, parameters_root=tmp_path / "missing"
        )


def test_ownership_parser_retains_missing_metadata_and_duplicate_class_failures(
    tmp_path,
):
    variables = _variables(tmp_path)
    (variables / "duplicate.py").write_text("""
class earnings(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
""")
    with pytest.raises(RuntimeError, match="Duplicate PolicyEngine variable"):
        source._index_policyengine_us_variable_sources(variables)
    (variables / "duplicate.py").write_text(
        "class invalid(Variable):\n    entity = Person\n"
    )
    with pytest.raises(RuntimeError, match="no static assignment"):
        source._index_policyengine_us_variable_sources(variables)


_SOURCE_INPUTS = frozenset(
    {"is_spm_independent_minor_role", "spm_unit_spm_universe_status"}
)
_OUTPUTS = frozenset({"spm_unit_spm_threshold", "spm_unit_net_income"})


@pytest.fixture
def declared_engine(monkeypatch):
    """Country declarations and fallback metadata without a country import."""
    spm = ModuleType("invented_country.spm")
    spm.DATASET_SOURCE_INPUTS = _SOURCE_INPUTS
    spm.REJECTED_DATASET_INPUTS = _OUTPUTS
    monkeypatch.setitem(sys.modules, spm.__name__, spm)
    engine = adapter.PolicyEngineUSEngine()
    variables = {}
    for name, entity, value_type, period, computed in (
        ("is_spm_independent_minor_role", "person", bool, "eternity", True),
        ("spm_unit_spm_universe_status", "spm_unit", int, "year", True),
        ("spm_unit_spm_threshold", "spm_unit", float, "year", True),
        ("spm_unit_net_income", "spm_unit", float, "year", True),
        ("earnings", "person", float, "year", False),
    ):
        variables[name] = SimpleNamespace(
            entity=SimpleNamespace(key=entity),
            value_type=value_type,
            definition_period=period,
            default_value=False if value_type is bool else 0,
            formula=(lambda *_: 0) if computed else None,
            get_formula=lambda _period, computed=computed: (
                object() if computed else None
            ),
        )
    engine._system = SimpleNamespace(variables=variables)
    monkeypatch.setattr(
        engine,
        "_import_policyengine_us",
        lambda: SimpleNamespace(__name__="invented_country"),
    )
    return engine, spm


def test_country_declared_role_and_annual_scope_are_inputs_not_defaults(
    declared_engine,
):
    engine, _ = declared_engine
    assert set(engine.variables()) == _SOURCE_INPUTS | {"earnings"}
    assert engine.formula_owned_outputs([*_SOURCE_INPUTS, *_OUTPUTS]) == _OUTPUTS
    assert engine.default_values([*_SOURCE_INPUTS, "earnings"]) == {"earnings": 0}
    role = engine.variable_metadata("is_spm_independent_minor_role")
    scope = engine.variable_metadata("spm_unit_spm_universe_status")
    assert (role.entity, role.dtype, role.period) == ("person", "bool", "point")
    assert (scope.entity, scope.dtype, scope.period) == ("spm_unit", "int", "year")


def test_declared_source_inputs_do_not_bypass_threshold_or_resource_export_guard(
    declared_engine,
):
    engine, _ = declared_engine
    tables = {
        "person": pd.DataFrame({"is_spm_independent_minor_role": [True, False]}),
        "spm_unit": pd.DataFrame(
            {
                "spm_unit_spm_universe_status": [0, 1],
                "spm_unit_spm_threshold": [100.0, 200.0],
                "spm_unit_net_income": [50.0, 150.0],
            }
        ),
    }
    before = {entity: table.copy(deep=True) for entity, table in tables.items()}
    assert engine._engine_computed_columns(tables, period=2024) == _OUTPUTS
    for entity, table in tables.items():
        pd.testing.assert_frame_equal(table, before[entity])


@pytest.mark.parametrize(
    "declared,rejected",
    [
        (None, _OUTPUTS),
        (set(_SOURCE_INPUTS), _OUTPUTS),
        (frozenset(), _OUTPUTS),
        (frozenset({""}), _OUTPUTS),
        (frozenset({1}), _OUTPUTS),
        (_SOURCE_INPUTS, set(_OUTPUTS)),
        (_SOURCE_INPUTS, frozenset({None})),
        (_SOURCE_INPUTS, _OUTPUTS | _SOURCE_INPUTS),
        (frozenset({"unknown_variable"}), _OUTPUTS),
        (_SOURCE_INPUTS | {"dividend_income"}, _OUTPUTS),
    ],
)
def test_malformed_or_overlapping_country_declarations_refuse(
    declared_engine, declared, rejected
):
    engine, spm = declared_engine
    spm.DATASET_SOURCE_INPUTS = declared
    spm.REJECTED_DATASET_INPUTS = rejected
    with pytest.raises(RuntimeError):
        engine.variables()


def _source_declarations(tmp_path, declaration=None):
    """Static metadata carries an import bomb that must never execute."""
    country = tmp_path / "country"
    calculator = tmp_path / "calculator"
    country.mkdir()
    calculator.mkdir()
    (calculator / "policyengine_adapter.py").write_text(
        f"FORMULA_OWNED_INPUTS = frozenset({set(_OUTPUTS)!r})\n"
        "raise AssertionError('Do not import the calculator')\n"
    )
    (country / "spm.py").write_text(
        "from spm_calculator.policyengine_adapter import FORMULA_OWNED_INPUTS\n"
        "DERIVED = frozenset({'in_poverty'})\n"
        "REJECTED_DATASET_INPUTS = frozenset(FORMULA_OWNED_INPUTS) | DERIVED\n"
        + (
            declaration
            or f"DATASET_SOURCE_INPUTS = frozenset({set(_SOURCE_INPUTS)!r})\n"
        )
        + "raise AssertionError('Do not import the country')\n"
    )
    return country, calculator


def test_source_declaration_reader_matches_country_without_imports(tmp_path):
    country, calculator = _source_declarations(tmp_path)
    assert adapter._source_dataset_source_inputs(country, calculator) == _SOURCE_INPUTS


@pytest.mark.parametrize(
    "declaration",
    [
        "DATASET_SOURCE_INPUTS = choose_inputs()\n",
        "DATASET_SOURCE_INPUTS = frozenset({'spm_unit_net_income'})\n",
        "DATASET_SOURCE_INPUTS = frozenset({'role'})\nDATASET_SOURCE_INPUTS = frozenset({'scope'})\n",
        "DATASET_SOURCE_INPUTS = frozenset({'role'})\nif enabled:\n    DATASET_SOURCE_INPUTS = frozenset({'scope'})\n",
        "from builtins import set as frozenset\nDATASET_SOURCE_INPUTS = frozenset({'role'})\n",
        "def frozenset(values):\n    return values\nDATASET_SOURCE_INPUTS = frozenset({'role'})\n",
        "DATASET_SOURCE_INPUTS = MISSING_DECLARATION\n",
        "DATASET_SOURCE_INPUTS = RECURSIVE\nRECURSIVE = DATASET_SOURCE_INPUTS\n",
    ],
)
def test_source_declaration_reader_refuses_dynamic_overlap_and_rebinding(
    tmp_path, declaration
):
    country, calculator = _source_declarations(tmp_path, declaration)
    with pytest.raises(RuntimeError):
        adapter._source_dataset_source_inputs(country, calculator)


@pytest.mark.parametrize(
    "mutation",
    ["R.append('spm_unit_net_income')", "S = R\nS.append('spm_unit_net_income')"],
)
def test_source_declaration_reader_refuses_mutable_referenced_constants(
    tmp_path, mutation
):
    country, calculator = _source_declarations(tmp_path)
    (country / "spm.py").write_text(
        "R = []\n" + mutation + "\n"
        "REJECTED_DATASET_INPUTS = frozenset(R)\n"
        "DATASET_SOURCE_INPUTS = frozenset({'spm_unit_net_income'})\n"
    )
    with pytest.raises(RuntimeError, match="immutable frozenset"):
        adapter._source_dataset_source_inputs(country, calculator)


@pytest.mark.parametrize("module", ["country", "calculator"])
def test_source_declaration_reader_refuses_wildcard_binding_changes(tmp_path, module):
    country, calculator = _source_declarations(tmp_path)
    path = (
        country / "spm.py"
        if module == "country"
        else calculator / "policyengine_adapter.py"
    )
    path.write_text(path.read_text() + "\nfrom invented_helper import *\n")
    with pytest.raises(RuntimeError, match="Wildcard"):
        adapter._source_dataset_source_inputs(country, calculator)


@pytest.mark.parametrize("module", ["country", "calculator"])
@pytest.mark.parametrize(
    "nested_import",
    [
        "if True:\n    from invented_helper import rejected as {name}\n",
        "try:\n    from invented_helper import rejected as {name}\nexcept ImportError:\n    pass\n",
        "if True:\n    import invented_helper as {name}\n",
        "try:\n    import invented_helper as {name}\nexcept ImportError:\n    pass\n",
    ],
)
def test_source_declaration_reader_refuses_nested_import_rebindings(
    tmp_path, module, nested_import
):
    country, calculator = _source_declarations(tmp_path)
    # The literal declarations alone permit net income. A conditional import
    # can replace the rejected set with overlapping inputs (ImportFrom) or an
    # invalid module value (Import); neither is a statically constant boundary.
    (country / "spm.py").write_text(
        "from spm_calculator.policyengine_adapter import FORMULA_OWNED_INPUTS\n"
        "REJECTED_DATASET_INPUTS = FORMULA_OWNED_INPUTS\n"
        "DATASET_SOURCE_INPUTS = frozenset({'spm_unit_net_income'})\n"
    )
    (calculator / "policyengine_adapter.py").write_text(
        "FORMULA_OWNED_INPUTS = frozenset()\n"
    )
    path, name = (
        (country / "spm.py", "REJECTED_DATASET_INPUTS")
        if module == "country"
        else (calculator / "policyengine_adapter.py", "FORMULA_OWNED_INPUTS")
    )
    path.write_text(path.read_text() + nested_import.format(name=name))
    with pytest.raises(RuntimeError, match="Dynamic dataset ownership declaration"):
        adapter._source_dataset_source_inputs(country, calculator)


_NESTED_DEFINITIONS = (
    "if True:\n    def {name}():\n        pass\n",
    "if True:\n    async def {name}():\n        pass\n",
    "if True:\n    class {name}:\n        pass\n",
)


@pytest.mark.parametrize("module", ["country", "calculator"])
@pytest.mark.parametrize(
    "binding",
    [
        *_NESTED_DEFINITIONS,
        "try:\n    raise ValueError()\nexcept ValueError as {name}:\n    pass\n",
        "match None:\n    case {name}:\n        pass\n",
        "match [None]:\n    case [*{name}]:\n        pass\n",
        "match dict():\n    case {{**{name}}}:\n        pass\n",
    ],
)
def test_source_declaration_reader_refuses_nested_non_name_bindings(
    tmp_path, module, binding
):
    country, calculator = _source_declarations(tmp_path)
    path, name = (
        (country / "spm.py", "REJECTED_DATASET_INPUTS")
        if module == "country"
        else (calculator / "policyengine_adapter.py", "FORMULA_OWNED_INPUTS")
    )
    # These Python binding nodes keep their target as a string attribute,
    # rather than a Name(Store), but still replace or delete the declaration.
    path.write_text(path.read_text() + binding.format(name=name))
    with pytest.raises(RuntimeError, match="Dynamic dataset ownership declaration"):
        adapter._source_dataset_source_inputs(country, calculator)


@pytest.mark.parametrize("binding", _NESTED_DEFINITIONS)
def test_source_declaration_reader_refuses_nested_constructor_definitions(
    tmp_path, binding
):
    country, calculator = _source_declarations(tmp_path)
    path = country / "spm.py"
    path.write_text(binding.format(name="frozenset") + path.read_text())
    with pytest.raises(RuntimeError, match="Rebound frozenset"):
        adapter._source_dataset_source_inputs(country, calculator)


@pytest.mark.parametrize("target", ["REJECTED_DATASET_INPUTS", "frozenset"])
@pytest.mark.parametrize(
    "definition",
    [
        "def helper(value=({name} := frozenset({{'role'}}))):\n    pass\n",
        "async def helper(*, value=({name} := frozenset({{'role'}}))):\n    pass\n",
        "@({name} := decorator)\ndef helper():\n    pass\n",
        "class Helper(({name} := Base)):\n    pass\n",
        "class Helper(metaclass=({name} := Meta)):\n    pass\n",
        "class Helper:\n    global {name}\n    {name} = frozenset({{'role'}})\n",
    ],
)
def test_source_declaration_reader_refuses_import_time_definition_bindings(
    tmp_path, target, definition
):
    country, calculator = _source_declarations(tmp_path)
    path = country / "spm.py"
    path.write_text(definition.format(name=target) + path.read_text())
    with pytest.raises(RuntimeError, match="Dynamic dataset|Rebound frozenset"):
        adapter._source_dataset_source_inputs(country, calculator)


@pytest.mark.parametrize("definition", ["def", "async def"])
def test_source_declaration_reader_excludes_unevaluated_function_bodies(
    tmp_path, definition
):
    country, calculator = _source_declarations(tmp_path)
    path = country / "spm.py"
    path.write_text(
        path.read_text() + f"\n{definition} unused_helper():\n"
        "    global REJECTED_DATASET_INPUTS, frozenset\n"
        "    REJECTED_DATASET_INPUTS = object()\n"
        "    frozenset = object()\n"
    )
    assert adapter._source_dataset_source_inputs(country, calculator) == _SOURCE_INPUTS


@pytest.mark.parametrize("include_consumers", [False, True])
def test_installed_source_metadata_applies_same_declaration_without_parameter_reparse(
    tmp_path, monkeypatch, include_consumers
):
    variables = _variables(tmp_path)
    # Both a generated fallback role and an ordinary annual variable need the
    # same override; the ownership-only path must remain parameter-independent.
    (variables / "scope.py").write_text("""
class spm_unit_spm_universe_status(Variable):
    value_type = int
    entity = SPMUnit
    definition_period = YEAR
    def formula(spm_unit, period, parameters):
        return spm_unit("scope_hint", period)

class scope_hint(Variable):
    value_type = int
    entity = SPMUnit
    definition_period = YEAR

class scope_output(Variable):
    value_type = int
    entity = SPMUnit
    definition_period = YEAR
    def formula(spm_unit, period, parameters):
        return spm_unit("spm_unit_spm_universe_status", period)
""")
    generated = {
        "is_spm_independent_minor_role": source._SourceVariableDefinition(
            metadata=adapter.VariableMetadata(
                name="is_spm_independent_minor_role",
                entity="person",
                dtype="bool",
                period="point",
            ),
            always_computed=True,
            formula_starts=(),
        )
    }
    parameters = tmp_path / "parameters/gov/example/components.yaml"
    parameters.parent.mkdir(parents=True)
    parameters.write_text("values:\n  2024-01-01:\n    value: [earnings, deduction]\n")
    monkeypatch.setattr(
        adapter,
        "_installed_policyengine_us_source_parts",
        lambda: (tmp_path, variables, generated, _SOURCE_INPUTS),
    )
    monkeypatch.setattr(
        adapter, "distribution", lambda _: SimpleNamespace(version="invented")
    )
    adapter._installed_policyengine_us_variable_sources.cache_clear()
    adapter._installed_policyengine_us_variable_definitions.cache_clear()
    try:
        index = adapter.PolicyEngineUSVariableMetadataIndex(
            include_consumers=include_consumers
        )
        assert _SOURCE_INPUTS <= set(index.variables())
        assert index.formula_owned_outputs([*_SOURCE_INPUTS, "computed", "dated"]) == {
            "computed",
            "dated",
        }
        assert (
            index.variable_metadata("is_spm_independent_minor_role").period == "point"
        )
        assert index.variable_metadata("spm_unit_spm_universe_status").period == "year"
        tables = {
            "person": pd.DataFrame(
                {name: [0] for name in [*_SOURCE_INPUTS, "computed", "dated"]}
            )
        }
        assert index._engine_computed_columns(tables, period=2024) == {
            "computed",
            "dated",
        }
        # Lazy consumers must use the same ownership interpretation.
        assert index.consumer_receipts("earnings")
        closure = index.variable_dependency_closure("scope_output")
        assert closure.input_leaves == ("spm_unit_spm_universe_status",)
        assert closure.formula_nodes == ("scope_output",)
        assert closure.edges == (("scope_output", "spm_unit_spm_universe_status"),)
        # Raw formula references are still inspectable even though a supplied
        # source input terminates the dataset dependency closure.
        assert any(
            receipt.consumer == "spm_unit_spm_universe_status"
            for receipt in index.consumer_receipts("scope_hint")
        )
        assert generated["is_spm_independent_minor_role"].formula_owned
    finally:
        adapter._installed_policyengine_us_variable_sources.cache_clear()
        adapter._installed_policyengine_us_variable_definitions.cache_clear()
