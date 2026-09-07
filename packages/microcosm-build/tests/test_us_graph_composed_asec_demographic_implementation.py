"""Closed extension identity without upstream invalidation; no survey data."""

import hashlib
import json
import shutil

import pytest

from microcosm.build.us_runtime import (
    graph_composed_asec_demographic_implementation as extension,
)
from microcosm.build.us_runtime import graph_implementation as upstream


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    roots = upstream._package_roots()
    copied = dict(roots)
    target = tmp_path / "build"
    shutil.copytree(
        roots["microcosm.build"],
        target,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    copied["microcosm.build"] = target
    monkeypatch.setattr(upstream, "_package_roots", lambda: copied)
    monkeypatch.setattr(extension, "_package_roots", lambda: copied)
    original_files = extension.demographic.resources.files

    def files(package):
        if package == "microcosm.build.us_runtime":
            return target / "us_runtime"
        if package == "microcosm.build":
            return target
        return original_files(package)

    monkeypatch.setattr(extension.demographic.resources, "files", files)
    return target


def _base_hashes():
    return {
        name: upstream.implementation_hash(name)
        for name in (
            "composed_population_v1",
            "assembly_harmonize",
            "geography",
            "composed_asec_binding_v1",
        )
    }


def test_extension_is_not_registered_in_shared_inventory():
    assert extension.STAGE not in upstream.STAGE_DEPENDENCIES
    manifest = extension.implementation_manifest()
    assert manifest["upstream_implementation"] == upstream.implementation_manifest(
        extension.BASE_STAGE
    )
    assert set(manifest["modules"]) == set(extension.EXTRA_MODULES)
    assert manifest["source_implementation"] == extension.demographic._implementation()
    assert set(manifest["dependencies"]) == set(extension.DEPENDENCIES)


def test_same_bytes_relocated_preserve_extension_identity(relocated):
    manifest = extension.implementation_manifest()
    for name, digest in manifest["modules"].items():
        assert (
            hashlib.sha256((relocated / name.split("/", 1)[1]).read_bytes()).hexdigest()
            == digest
        )
    assert str(relocated) not in json.dumps(manifest)


@pytest.mark.parametrize("name", extension.EXTRA_MODULES)
def test_each_extension_module_changes_only_extension_identity(relocated, name):
    before = extension.implementation_hash()
    base_before = _base_hashes()
    path = relocated / name.split("/", 1)[1]
    path.write_bytes(
        path.read_bytes() + b"\n# reviewed extension-only mapping change\n"
    )
    assert extension.implementation_hash() != before
    assert _base_hashes() == base_before


def test_inventory_classification_bytes_are_bound_only_to_extension(relocated):
    before = extension.implementation_hash()
    base_before = _base_hashes()
    path = relocated / extension.INVENTORY_FILE
    document = json.loads(path.read_text())
    document["import_classifications"]["numpy"] += " Reviewed wording."
    path.write_text(json.dumps(document))
    assert extension.implementation_hash() != before
    assert _base_hashes() == base_before


@pytest.mark.parametrize(
    "addition",
    [
        "import urllib3\n",
        "from pathlib import Path\nPath('unclassified-resource').read_bytes()\n",
        "from microcosm.build.us_runtime import housing_inputs\nhousing_inputs.unclassified_live_value()\n",
    ],
)
def test_new_import_resource_or_unbound_use_refuses(relocated, addition):
    path = relocated / "us_runtime/graph_composed_asec_demographics.py"
    path.write_text(path.read_text() + "\n" + addition)
    with pytest.raises(ValueError, match="unclassified dependency/resource contract"):
        extension.implementation_hash()


@pytest.mark.parametrize(
    "field",
    ["dependencies", "extra_modules", "contracts", "import_classifications"],
)
def test_inventory_roster_cannot_silently_drop_required_entries(relocated, field):
    path = relocated / extension.INVENTORY_FILE
    document = json.loads(path.read_text())
    value = document[field]
    if isinstance(value, list):
        value.pop()
    else:
        value.pop(next(iter(value)))
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="roster"):
        extension.implementation_hash()


def test_existing_inventory_validation_remains_enforced(relocated):
    path = relocated / "us_runtime/graph_implementation_inventory.json"
    document = json.loads(path.read_text())
    document["stages"].pop(extension.BASE_STAGE)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        extension.implementation_hash()


def test_transitive_source_module_and_legacy_identity_bound(relocated):
    before = extension.implementation_manifest()
    path = relocated / "us_runtime/asec_person_income_source.py"
    path.write_bytes(path.read_bytes() + b"\n# transitive source mutation\n")
    after = extension.implementation_manifest()
    assert after["source_implementation"] != before["source_implementation"]
    assert after["upstream_implementation"] != before["upstream_implementation"]
    path = relocated / "us_runtime/asec_household_observations.py"
    path.write_bytes(path.read_bytes() + b"\n# transitive legacy verifier mutation\n")
    final = extension.implementation_manifest()
    assert (
        final["source_implementation"]["legacy_verification"]
        != after["source_implementation"]["legacy_verification"]
    )


def test_live_source_contract_and_pins_are_bound(monkeypatch):
    before = extension.implementation_hash()
    base_before = _base_hashes()
    monkeypatch.setattr(extension.demographic, "_MEMBER_PINS", ())
    assert extension.implementation_hash() != before
    assert _base_hashes() == base_before


def test_dependency_version_changes_are_bound(monkeypatch):
    before = extension.implementation_hash()
    original = upstream.importlib_metadata.version
    monkeypatch.setattr(
        upstream.importlib_metadata,
        "version",
        lambda name: original(name) + (".changed" if name == "h5py" else ""),
    )
    assert extension.implementation_hash() != before


def test_unsupported_stage_refuses():
    with pytest.raises(ValueError, match="unsupported stage"):
        extension.implementation_hash("unclassified_stage")
