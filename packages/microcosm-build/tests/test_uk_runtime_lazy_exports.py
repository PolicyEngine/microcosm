"""Normal-import contracts for the real UK runtime public package.

A fresh no-country-resource guard supplies the cold-import check. In a larger
suite, already imported country modules are allowed; these tests never remove,
stub or replace modules to manufacture an isolated import environment.
"""

import hashlib
import importlib
import json
import sys

import pytest

_PACKAGE = "microcosm.build.uk_runtime"
_UNRELATED_MODULES = (
    f"{_PACKAGE}.calibration_run",
    f"{_PACKAGE}.national_calibration",
    f"{_PACKAGE}.firm_generation",
)
# SHA256 of compact JSON for all 456 ordered entries from ec3b306f3f9742ff...
# This includes the existing duplicate ladder_vs_chronicle_household_dispersion.
_ORDERED_ALL_SHA256 = "282baaa8103b6a68ec916669dc9b96909d6faf0a9beecdbb8359df12f128f66c"


def _unrelated_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _UNRELATED_MODULES
        )
    }


def test_normal_package_and_pure_helper_imports_do_not_load_unrelated_modules():
    before = _unrelated_modules()
    package = importlib.import_module(_PACKAGE)
    helper = importlib.import_module(f"{_PACKAGE}.rowwise_geography")

    assert sys.modules[_PACKAGE] is package
    assert package.rowwise_geography is helper
    assert callable(helper.assign_household_geography)
    assert _unrelated_modules() == before


def test_public_from_import_returns_and_caches_the_real_defining_object():
    from microcosm.build.uk_runtime import assign_household_geography
    from microcosm.build.uk_runtime.rowwise_geography import (
        assign_household_geography as direct,
    )

    package = importlib.import_module(_PACKAGE)
    assert assign_household_geography is direct
    assert package.assign_household_geography is direct
    assert vars(package)["assign_household_geography"] is direct
    assert package.assign_household_geography is assign_household_geography


def test_direct_submodule_import_and_from_import_preserve_module_identity():
    # In the fresh guard this exercises Python's fallback before any direct
    # content_identity import, without removing an already loaded module in CI.
    from microcosm.build.uk_runtime import content_identity

    assert content_identity is sys.modules[f"{_PACKAGE}.content_identity"]

    import microcosm.build.uk_runtime.content_identity as content_direct
    import microcosm.build.uk_runtime.rowwise_geography as direct
    from microcosm.build.uk_runtime import rowwise_geography

    package = importlib.import_module(_PACKAGE)
    assert content_identity is content_direct
    assert content_direct is sys.modules[f"{_PACKAGE}.content_identity"]
    assert package.content_identity is content_direct
    assert "content_identity" not in package.__all__
    assert rowwise_geography is direct
    assert direct is sys.modules[f"{_PACKAGE}.rowwise_geography"]
    assert package.rowwise_geography is direct
    assert "rowwise_geography" not in package.__all__


def test_discovery_preserves_the_frozen_ordered_export_contract_without_resolution():
    package = importlib.import_module(_PACKAGE)
    imports_before = _unrelated_modules()
    aliases_before = {
        name: vars(package)[name] for name in package.__all__ if name in vars(package)
    }

    discovered = dir(package)

    assert len(package.__all__) == 456
    assert len(set(package.__all__)) == 455
    assert package.__all__.count("ladder_vs_chronicle_household_dispersion") == 2
    encoded = json.dumps(package.__all__, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == _ORDERED_ALL_SHA256
    assert set(package.__all__).issubset(discovered)
    assert discovered == sorted(set(discovered))
    aliases_after = {
        name: vars(package)[name] for name in package.__all__ if name in vars(package)
    }
    assert aliases_after.keys() == aliases_before.keys()
    assert all(aliases_after[name] is value for name, value in aliases_before.items())
    assert _unrelated_modules() == imports_before
    with pytest.raises(TypeError):
        package._EXPORTS["invented_missing_export"] = ("invented", "missing")


def test_unknown_attribute_and_unknown_from_import_keep_normal_failures():
    package = importlib.import_module(_PACKAGE)
    with pytest.raises(
        AttributeError, match="has no attribute 'invented_missing_export'"
    ):
        _ = package.invented_missing_export
    with pytest.raises(
        ImportError, match="cannot import name 'invented_missing_export'"
    ):
        from microcosm.build.uk_runtime import invented_missing_export  # noqa: F401

    assert "invented_missing_export" not in vars(package)


def test_reload_clears_public_aliases_and_rebinds_the_same_defining_objects():
    package = importlib.import_module(_PACKAGE)
    helper = importlib.import_module(f"{_PACKAGE}.rowwise_geography")
    real_function = helper.assign_household_geography
    assert package.assign_household_geography is real_function
    assert "assign_household_geography" in vars(package)
    imports_before = _unrelated_modules()

    assert importlib.reload(package) is package

    assert "assign_household_geography" not in vars(package)
    assert package.rowwise_geography is helper
    assert sys.modules[f"{_PACKAGE}.rowwise_geography"] is helper
    assert package.assign_household_geography is real_function
    assert vars(package)["assign_household_geography"] is real_function
    assert _unrelated_modules() == imports_before
