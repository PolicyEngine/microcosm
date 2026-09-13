"""Ownership uses the real declaration parser without consumer parameter I/O."""

from pathlib import Path

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
