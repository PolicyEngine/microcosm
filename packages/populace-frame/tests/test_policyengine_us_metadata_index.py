from pathlib import Path

import pytest

from populace.frame.adapters import policyengine_us as module


def _write_variable_source(
    root: Path, source: str, *, name: str = "variables.py"
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(source)


def test_source_index_classifies_leaf_and_formula_mutations(tmp_path: Path) -> None:
    _write_variable_source(
        tmp_path,
        """
class source_leaf(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR

class future_formula(Variable):
    value_type = int
    entity = Person
    definition_period = YEAR

    def formula_2025(person, period, parameters):
        return 0

class aggregate_formula(Variable):
    value_type = bool
    entity = TaxUnit
    definition_period = MONTH
    adds = ["source_leaf"]
""",
    )

    indexed = module._index_policyengine_us_variable_sources(tmp_path)

    assert not indexed["source_leaf"].formula_owned
    assert indexed["future_formula"].formula_owned
    assert not indexed["future_formula"].computed_at(2024)
    assert indexed["future_formula"].computed_at(2025)
    assert indexed["aggregate_formula"].always_computed
    assert indexed["aggregate_formula"].metadata.entity == "tax_unit"
    assert indexed["aggregate_formula"].metadata.dtype == "bool"
    assert indexed["aggregate_formula"].metadata.period == "month"


def test_source_index_rejects_duplicate_variable_classes(tmp_path: Path) -> None:
    source = """
class duplicate(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
"""
    _write_variable_source(tmp_path, source, name="first.py")
    _write_variable_source(tmp_path, source, name="second.py")

    with pytest.raises(RuntimeError, match="Duplicate.*duplicate"):
        module._index_policyengine_us_variable_sources(tmp_path)


def test_source_index_rejects_dynamic_required_metadata(tmp_path: Path) -> None:
    _write_variable_source(
        tmp_path,
        """
class dynamic_metadata(Variable):
    value_type = choose_value_type()
    entity = Person
    definition_period = YEAR
""",
    )

    with pytest.raises(RuntimeError, match="dynamic.*value_type"):
        module._index_policyengine_us_variable_sources(tmp_path)
