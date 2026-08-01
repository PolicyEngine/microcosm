from pathlib import Path

import pytest

from populace.frame.adapters import policyengine_us as module

_EXPECTED_IN_STATE = frozenset(
    "AL AK AZ AR CA CO CT DC DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN "
    "MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA "
    "WV WI WY PR VI".split()
)
_EXPECTED_PUF = frozenset(
    "e02000 e26270 e19200 e18500 e19800 e20400 e20100 e00700 e03270 e24515 "
    "e03300 e07300 e62900 e32800 e87530 e03240 e01100 e01200 e24518 e09900 "
    "e27200 e03290 e58990 e03230 e11200 e07260 e07240 e03220 p08000 e03400 "
    "e09800 e09700 e03500 e87521".split()
)
_EXPECTED_STATE_MFS = frozenset(
    "ar_standard_deduction ar_itemized_deductions ar_taxable_income ar_agi "
    "dc_taxable_income de_standard_deduction de_itemized_deductions "
    "de_taxable_income de_agi ia_standard_deduction ia_itemized_deductions "
    "ia_taxable_income ia_agi ky_standard_deduction ky_itemized_deductions "
    "ky_taxable_income ms_standard_deduction ms_itemized_deductions "
    "ms_taxable_income mt_standard_deduction mt_itemized_deductions "
    "mt_taxable_income".split()
)


def _write_variable_source(
    root: Path, source: str, *, name: str = "variables.py"
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(source)


def _installed_package() -> tuple[object, Path]:
    try:
        package = module.distribution("policyengine-us")
    except module.PackageNotFoundError:
        pytest.skip("requires the policyengine-us [us] extra")
    return package, Path(package.locate_file("policyengine_us"))


def _copy_generated_sources(root: Path) -> tuple[object, Path]:
    package, installed_root = _installed_package()
    for relative_path in module._GENERATED_SOURCE_SHA256:
        source = installed_root / relative_path
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return package, root


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


def test_generated_snapshot_covers_the_pinned_default_system() -> None:
    package, package_root = _installed_package()
    generated = module._index_policyengine_us_generated_variable_sources(
        package_root,
        version=package.version,
    )
    formula_owned = _EXPECTED_IN_STATE | _EXPECTED_STATE_MFS | {"mi_surtax"}

    assert set(generated) == formula_owned | _EXPECTED_PUF
    assert len(generated) == 110
    assert {name for name, item in generated.items() if item.formula_owned} == (
        formula_owned
    )
    assert {name for name, item in generated.items() if not item.formula_owned} == (
        _EXPECTED_PUF
    )

    expected_metadata = {
        "AK": ("household", "bool", "year"),
        "e00700": ("person", "float", "year"),
        "ar_agi": ("tax_unit", "float", "year"),
        "mi_surtax": ("tax_unit", "float", "year"),
    }
    for name, expected in expected_metadata.items():
        metadata = generated[name].metadata
        assert (metadata.entity, metadata.dtype, metadata.period) == expected

    index = module.PolicyEngineUSVariableMetadataIndex()
    assert len(index._definitions) == 5_770
    assert len(index.variables()) == 863
    assert len(index.formula_owned_outputs(index._definitions)) == 4_907
    assert index.formula_owned_outputs(["AK", "e00700", "ar_agi", "mi_surtax"]) == {
        "AK",
        "ar_agi",
        "mi_surtax",
    }


@pytest.mark.parametrize("relative_path", module._GENERATED_SOURCE_SHA256)
def test_generated_snapshot_fails_closed_on_source_mutation(
    tmp_path: Path,
    relative_path: str,
) -> None:
    package, package_root = _copy_generated_sources(tmp_path)
    source = package_root / relative_path
    source.write_bytes(source.read_bytes() + b"\n# metadata audit mutation\n")

    with pytest.raises(RuntimeError, match="source changed without a metadata audit"):
        module._index_policyengine_us_generated_variable_sources(
            package_root,
            version=package.version,
        )


def test_generated_snapshot_fails_closed_on_missing_source(tmp_path: Path) -> None:
    package, package_root = _copy_generated_sources(tmp_path)
    (package_root / "system.py").unlink()

    with pytest.raises(RuntimeError, match="source is unavailable"):
        module._index_policyengine_us_generated_variable_sources(
            package_root,
            version=package.version,
        )


def test_generated_snapshot_fails_closed_on_unreviewed_version(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="has not been audited"):
        module._index_policyengine_us_generated_variable_sources(
            tmp_path,
            version="1.764.7",
        )


def test_installed_index_distinguishes_absent_extra_from_broken_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _BrokenDistribution:
        version = module._GENERATED_SOURCE_VERSION

        @staticmethod
        def locate_file(path: str) -> Path:
            return tmp_path / path

    module._installed_policyengine_us_variable_sources.cache_clear()
    monkeypatch.setattr(module, "distribution", lambda _name: _BrokenDistribution())
    with pytest.raises(RuntimeError, match="source tree is unavailable"):
        module._installed_policyengine_us_variable_sources()
    module._installed_policyengine_us_variable_sources.cache_clear()

    def _missing_distribution(_name: str) -> None:
        raise module.PackageNotFoundError("policyengine-us")

    monkeypatch.setattr(module, "distribution", _missing_distribution)
    with pytest.raises(ImportError, match="requires the 'policyengine-us' package"):
        module._installed_policyengine_us_variable_sources()
    module._installed_policyengine_us_variable_sources.cache_clear()
