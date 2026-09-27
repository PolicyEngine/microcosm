"""Real implementation manifests, without source data or graph execution."""

import copy
import hashlib
import re
from pathlib import Path

import pytest

from microcosm.build.us_runtime import graph_implementation as implementation

_STORE = "microcosm.graph/store.py"
_OLD_RESOURCE_CONTRACT = (
    "99afee26ca660c678dcbd7117eb2e64d70a0343ead9a602fa5819222aca02bc3"
)


def _require_stage_dependencies(stage):
    for dependency in implementation.STAGE_DEPENDENCIES[stage]:
        pytest.importorskip("yaml" if dependency == "PyYAML" else dependency)


@pytest.mark.parametrize("stage", tuple(implementation.STAGE_DEPENDENCIES))
def test_maintained_stage_manifest_accepts_and_binds_reviewed_store(stage):
    _require_stage_dependencies(stage)
    manifest = implementation.implementation_manifest(stage)
    store = implementation._package_roots()["microcosm.graph"] / "store.py"
    assert manifest["modules"][_STORE] == hashlib.sha256(store.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "module,field,old",
    [
        (_STORE, "resource_accesses_sha256", _OLD_RESOURCE_CONTRACT),
        (
            "microcosm.frame/adapters/policyengine_us.py",
            "resource_accesses_sha256",
            "b3a55c9ba22f5a08ec9775278c49b4f6a1f8b2787f182605707ab0978ea1212a",
        ),
        (
            "microcosm.frame/adapters/policyengine_us.py",
            "unbound_uses_sha256",
            "fc1d7672cf7653b8d21a2d19b0e2ae6fb8dbdad78003b8c2ce76d6f3a3741f23",
        ),
    ],
)
def test_stale_reviewed_dependency_classification_still_refuses(
    monkeypatch, module, field, old
):
    stage = "acs_housing_universe_2024"
    _require_stage_dependencies(stage)
    inventory = implementation._inventory

    def stale(roots):
        document, payload = inventory(roots)
        document = copy.deepcopy(document)
        document["contracts"][module][field] = old
        return document, payload

    monkeypatch.setattr(implementation, "_inventory", stale)
    with pytest.raises(
        ValueError,
        match=re.escape("Unclassified US dependency/resource contract: " + module),
    ):
        implementation.implementation_manifest(stage)


def test_additional_store_resource_access_remains_unclassified(monkeypatch):
    stage = "acs_housing_universe_2024"
    _require_stage_dependencies(stage)
    store = implementation._package_roots()["microcosm.graph"] / "store.py"
    read_bytes = Path.read_bytes

    def changed(path):
        payload = read_bytes(path)
        if path == store:
            # Parsed as source only: never execute the invented resource access.
            payload += (
                b"\ndef unreviewed_resource():\n    open('invented-new-resource')\n"
            )
        return payload

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(
        ValueError,
        match=r"Unclassified US dependency/resource contract: microcosm\.graph/store\.py",
    ):
        implementation.implementation_manifest(stage)
