"""Reviewed US implementation scopes bind helpers, resources and live registries."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from microcosm.build.us_runtime import graph_implementation as impl
from microcosm.build.us_runtime import graph_sources


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    roots = impl._package_roots()
    copied = {}
    for package, root in roots.items():
        target = tmp_path / package
        shutil.copytree(root, target, ignore=shutil.ignore_patterns("__pycache__"))
        copied[package] = target
    monkeypatch.setattr(impl, "_package_roots", lambda: copied)
    return copied


def _hashes():
    return {stage: impl.implementation_hash(stage) for stage in impl.STAGE_DEPENDENCIES}


def test_same_content_relocated_roots_preserve_identity(tmp_path, monkeypatch):
    before = _hashes()
    roots = impl._package_roots()
    copied = {}
    for package, root in roots.items():
        copied[package] = tmp_path / package
        shutil.copytree(
            root, copied[package], ignore=shutil.ignore_patterns("__pycache__")
        )
    monkeypatch.setattr(impl, "_package_roots", lambda: copied)
    assert _hashes() == before
    assert str(tmp_path) not in json.dumps(
        impl.implementation_manifest("assembly_prepare")
    )


@pytest.mark.parametrize("filename", ["housing_inputs.py", "hours_worked.py"])
def test_unrelated_operator_body_and_quality_file_do_not_invalidate(
    relocated, filename
):
    before = _hashes()
    root = relocated["microcosm.build"]
    path = root / "us_runtime" / filename
    path.write_text(path.read_text() + "\ndef unrelated_body():\n    return 1729\n")
    (root / "quality_aggregates.py").write_text("def future_quality(): return 42\n")
    assert _hashes() == before


@pytest.mark.parametrize(
    "package,relative,changed,unchanged",
    [
        ("microcosm.build", "frame_checkpoint.py", "asec_codec_v4", "geography"),
        (
            "microcosm.build",
            "us_runtime/asec_checkpoint.py",
            "assembly_prepare",
            "assembly_harmonize",
        ),
        (
            "microcosm.build",
            "us_runtime/reported_coverage_source.py",
            "asec_codec_v4",
            "geography",
        ),
        (
            "microcosm.build",
            "us_runtime/acs_inputs.py",
            "acs_codec_2024",
            "assembly_harmonize",
        ),
        ("microcosm.build", "us_runtime/acs_pums.py", "assembly_prepare", "geography"),
        ("microcosm.build", "frame_sampling.py", "assembly_prepare", "geography"),
        ("microcosm.build", "table_identity.py", "assembly_prepare", "geography"),
        ("microcosm.build", "table_identity.py", "assembly_harmonize", "asec_codec_v4"),
        (
            "microcosm.build",
            "us_runtime/spine_assembly.py",
            "assembly_harmonize",
            "geography",
        ),
        (
            "microcosm.build",
            "us_runtime/puma_ladder.py",
            "geography",
            "assembly_prepare",
        ),
        ("microcosm.frame", "units.py", "assembly_prepare", None),
        ("microcosm.graph", "executor.py", "assembly_harmonize", None),
        ("microcosm.graph", "store.py", "geography", None),
        ("microunit", "rule_helpers.py", "acs_codec_2024", "asec_codec_v4"),
        (
            "microunit",
            "data/dependent_gross_income_limit.yaml",
            "assembly_prepare",
            "geography",
        ),
    ],
)
def test_transitive_module_and_resource_bytes_are_bound(
    relocated, package, relative, changed, unchanged
):
    before = _hashes()
    path = relocated[package] / relative
    path.write_bytes(path.read_bytes() + b"\n# reviewed implementation mutation\n")
    after = _hashes()
    assert after[changed] != before[changed]
    if unchanged:
        assert after[unchanged] == before[unchanged]


def test_live_registry_projection_invalidates_source_not_geography(monkeypatch):
    from microcosm.build.us_runtime import operator_boundary

    before = _hashes()
    families = dict(operator_boundary.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES)
    families["new_operator"] = {"person": frozenset({"new_operator_column"})}
    monkeypatch.setattr(
        operator_boundary, "PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES", families
    )
    after = _hashes()
    for stage in ("asec_codec_v4", "acs_codec_2024", "assembly_prepare"):
        assert after[stage] != before[stage]
    assert after["geography"] == before["geography"]


def test_fresh_process_resolves_provider_constants(tmp_path):
    # A fresh import must project provider values, not a copied registry literal.
    root = impl._package_roots()["microcosm.build"]
    copied = tmp_path / "microcosm" / "build"
    shutil.copytree(root, copied, ignore=shutil.ignore_patterns("__pycache__"))
    provider = copied / "us_runtime/housing_inputs.py"
    provider.write_text(
        provider.read_text()
        + "\nUS_HOUSING_PERSON_OUTPUT_COLUMNS = (*US_HOUSING_PERSON_OUTPUT_COLUMNS, 'fixture_new_housing_column')\n"
    )
    program = """
import json
from microcosm.build.us_runtime.graph_implementation import implementation_manifest
print(json.dumps(implementation_manifest('assembly_prepare')['projections']))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )
    assert "fixture_new_housing_column" in result.stdout


@pytest.mark.parametrize("dependency", ["h5py", "PyYAML", "microunit", "pyarrow"])
def test_declared_dependency_versions_invalidate(monkeypatch, dependency):
    before = _hashes()
    original = impl.importlib_metadata.version
    monkeypatch.setattr(
        impl.importlib_metadata,
        "version",
        lambda name: original(name) + (".changed" if name == dependency else ""),
    )
    after = _hashes()
    assert after["assembly_prepare"] != before["assembly_prepare"]
    assert after["geography"] == before["geography"]


def test_acs_string_storage_option_is_explicit_identity():
    import pandas as pd

    with pd.option_context("mode.string_storage", "python"):
        python = _hashes()
    with pd.option_context("mode.string_storage", "pyarrow"):
        arrow = _hashes()
        assert (
            impl.implementation_manifest("acs_codec_2024")["runtime_options"][
                "acs_reader_resolved_string_storage"
            ]
            == "pyarrow"
        )
    assert python["assembly_prepare"] != arrow["assembly_prepare"]
    assert python["acs_codec_2024"] != arrow["acs_codec_2024"]
    assert python["geography"] == arrow["geography"]


@pytest.mark.parametrize(
    "package,path",
    [
        ("microcosm.frame", "new_helper.py"),
        ("microcosm.graph", "new_helper.py"),
        ("microunit", "data/new_rule.yaml"),
        ("microunit", "new_rule.py"),
    ],
)
def test_unclassified_package_module_or_resource_refuses(relocated, package, path):
    (relocated[package] / path).write_text("# new unclassified dependency\n")
    with pytest.raises(ValueError, match="inventory"):
        impl.implementation_hash("assembly_prepare")


@pytest.mark.parametrize(
    "addition",
    [
        "\nimport newly_added_package\n",
        "\nfrom .new_source_helper import decode\n",
        "\ndef new_resource():\n    return Path('new_rule.yaml').read_bytes()\n",
    ],
)
def test_new_import_or_resource_access_requires_classification(relocated, addition):
    path = relocated["microcosm.build"] / "us_runtime/graph_sources.py"
    path.write_text(path.read_text() + addition)
    with pytest.raises(ValueError, match="dependency|resource"):
        impl.implementation_hash("assembly_prepare")


def test_bound_module_symbol_edits_need_no_generated_inventory_refresh(relocated):
    path = relocated["microcosm.build"] / "frame_sampling.py"
    before = impl.implementation_hash("assembly_prepare")
    path.write_text(
        path.read_text()
        + "\nfrom numpy import add\ndef extra_bound_operation():\n    return add(1, 2)\n"
    )
    assert impl.implementation_hash("assembly_prepare") != before


def test_moving_existing_unbound_helper_into_a_new_route_refuses(relocated):
    path = relocated["microcosm.build"] / "us_runtime/operator_boundary.py"
    path.write_text(
        path.read_text()
        + "\ndef unreviewed_provider_use():\n    return US_HOUSING_PERSON_OUTPUT_COLUMNS\n"
    )
    with pytest.raises(ValueError, match="dependency/resource"):
        impl.implementation_hash("assembly_prepare")


@pytest.mark.parametrize("symbol", ["unreviewed_helper", "*"])
def test_existing_unbound_module_cannot_silently_add_imports(relocated, symbol):
    path = relocated["microcosm.build"] / "us_runtime/operator_boundary.py"
    path.write_text(
        path.read_text()
        + f"\nfrom microcosm.build.us_runtime.housing_inputs import {symbol}\n"
    )
    with pytest.raises(ValueError, match="dependency|wildcard"):
        impl.implementation_hash("assembly_prepare")


def test_missing_declared_module_and_resource_refuse(relocated):
    path = relocated["microunit"] / "data/dependent_gross_income_limit.yaml"
    path.unlink()
    with pytest.raises(ValueError, match="inventory"):
        impl.implementation_hash("acs_codec_2024")


def test_missing_build_module_refuses(relocated):
    (relocated["microcosm.build"] / "us_runtime/asec_checkpoint.py").unlink()
    with pytest.raises(ValueError, match="module inventory missing"):
        impl.implementation_hash("asec_codec_v4")


def test_missing_acs_dependency_does_not_break_geography_or_asec(monkeypatch):
    original = impl.importlib.util.find_spec
    monkeypatch.setattr(
        impl.importlib.util,
        "find_spec",
        lambda package: None if package == "microunit" else original(package),
    )
    assert impl.implementation_hash("geography")
    assert impl.implementation_hash("asec_codec_v4")
    with pytest.raises(ValueError, match="dependency package inventory"):
        impl.implementation_hash("acs_codec_2024")


def test_unavailable_distribution_fails_instead_of_omitting_version(monkeypatch):
    original = impl.importlib_metadata.version

    def version(name):
        if name == "h5py":
            raise impl.importlib_metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(impl.importlib_metadata, "version", version)
    with pytest.raises(impl.importlib_metadata.PackageNotFoundError):
        impl.implementation_hash("assembly_prepare")


def test_manifest_dependency_tuple_cannot_drift_from_capabilities(relocated):
    path = (
        relocated["microcosm.build"] / "us_runtime/graph_implementation_inventory.json"
    )
    document = json.loads(path.read_text())
    document["stages"]["assembly_prepare"]["dependencies"].append("unreviewed")
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="dependency inventory differs"):
        impl.implementation_hash("assembly_prepare")


def test_codec_name_loader_and_capabilities_match_manifest(monkeypatch):
    from microcosm.build.us_runtime.graph_geography import (
        USGeographyAssignKernel,
        USGeographyBoundaryKernel,
        USGeographyGateKernel,
        USGeographyLookupKernel,
    )
    from microcosm.graph.codecs import SourceCodecRegistry

    codecs = graph_sources.us_source_codecs()
    impl.validate_source_codecs(codecs)
    for kernel, stage in (
        (graph_sources.USAssemblyCreateKernel, "assembly_prepare"),
        (graph_sources.USSpineHarmonizeKernel, "assembly_harmonize"),
        *(
            (kernel, "geography")
            for kernel in (
                USGeographyAssignKernel,
                USGeographyBoundaryKernel,
                USGeographyGateKernel,
                USGeographyLookupKernel,
            )
        ),
    ):
        assert kernel.capabilities.dependencies == impl.STAGE_DEPENDENCIES[stage]
        assert kernel().implementation_hash() == impl.implementation_hash(stage)
        assert set(impl.implementation_manifest(stage)["dependencies"]) == set(
            kernel.capabilities.dependencies
        )
    wrong = SourceCodecRegistry()
    wrong.register(graph_sources.ASEC_CODEC, graph_sources.load_graph_acs)
    wrong.register(graph_sources.ACS_CODEC, graph_sources.load_graph_acs)
    with pytest.raises(ValueError, match="loader"):
        impl.validate_source_codecs(wrong)
    monkeypatch.setattr(graph_sources, "load_graph_asec", lambda path: None)
    with pytest.raises(ValueError, match="loader"):
        graph_sources.us_source_codecs()


def test_hashing_does_not_import_optional_frame_adapters():
    program = """
import sys
from microcosm.build.us_runtime.graph_implementation import implementation_hash
before = {name for name in sys.modules if name.startswith('microcosm.frame.adapters.')}
implementation_hash('assembly_prepare')
after = {name for name in sys.modules if name.startswith('microcosm.frame.adapters.')}
assert before == after  # build's ordinary imports already load its US adapter
assert 'microcosm.frame.adapters.axiom' not in sys.modules
assert 'microcosm.frame.adapters.policyengine_uk' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", program], check=True, capture_output=True, text=True
    )
