"""Classified implementation identity for the downstream demographic extension.

The existing v1 identity service and inventory remain byte-for-byte unchanged.
This extension attests that service's complete composed-ASEC scope, then adds a
closed, separately reviewed module inventory. Adding this consumer therefore
does not change an upstream key. No recorded hash is overridden or reused for
different code; every manifest is reconstructed from the actual local bytes.
"""

from __future__ import annotations

import hashlib
import json

import microcosm.build.us_runtime.asec_demographic_source as demographic
from microcosm.build.us_runtime.graph_implementation import (
    STAGE_DEPENDENCIES,
    _canonical,
    _covered_imports,
    _dependency_contract,
    _package_roots,
)
from microcosm.build.us_runtime.graph_implementation import (
    implementation_manifest as upstream_implementation_manifest,
)

STAGE = "composed_asec_demographic_binding_v1"
BASE_STAGE = "composed_asec_binding_v1"
DEPENDENCIES = STAGE_DEPENDENCIES[BASE_STAGE]
INVENTORY_FILE = "us_runtime/asec_demographic_graph_inventory.json"
INVENTORY_SCHEMA = "microcosm.us.demographic-extension-inventory.v1"
EXTRA_MODULES = (
    "microcosm.build/us_runtime/asec_demographic_source.py",
    "microcosm.build/us_runtime/graph_composed_asec_demographic_implementation.py",
    "microcosm.build/us_runtime/graph_composed_asec_demographics.py",
)
_INVENTORY_KEYS = frozenset(
    {
        "schema",
        "stage",
        "base_stage",
        "dependencies",
        "extra_modules",
        "contracts",
        "import_classifications",
    }
)


def _require(condition, reason):
    if not condition:
        raise ValueError(f"Demographic extension inventory: {reason}.")


def _read_inventory(roots):
    payload = (roots["microcosm.build"] / INVENTORY_FILE).read_bytes()
    value = json.loads(payload)
    _require(isinstance(value, dict) and set(value) == _INVENTORY_KEYS, "schema keys")
    _require(
        value["schema"] == INVENTORY_SCHEMA
        and value["stage"] == STAGE
        and value["base_stage"] == BASE_STAGE,
        "stage identity",
    )
    _require(value["dependencies"] == list(DEPENDENCIES), "dependency roster")
    _require(value["extra_modules"] == list(EXTRA_MODULES), "module roster")
    _require(set(value["contracts"]) == set(EXTRA_MODULES), "contract roster")
    imports = {
        name for contract in value["contracts"].values() for name in contract["imports"]
    }
    _require(
        set(value["import_classifications"]) == imports
        and all(
            isinstance(label, str) and bool(label.strip())
            for label in value["import_classifications"].values()
        ),
        "import classification roster",
    )
    return value, payload


def implementation_manifest(stage=STAGE):
    """Validate the unchanged upstream scope and all additional live dependencies."""
    _require(stage == STAGE, "unsupported stage")
    # The original validator still checks its complete module/resource/import
    # closure, package rosters, codecs, live registries and runtime options.
    base = upstream_implementation_manifest(BASE_STAGE)
    _require(set(base["dependencies"]) == set(DEPENDENCIES), "upstream dependencies")
    roots = _package_roots()
    inventory, inventory_bytes = _read_inventory(roots)
    # Reuse the existing exact AST import/use/resource tripwires, now under one
    # extension scope. Only imports whose whole module or distribution is
    # actually attested may omit the unbound-symbol tripwire.
    scope = {
        "stages": {
            STAGE: {
                "modules": sorted(set(base["modules"]) | set(EXTRA_MODULES)),
                "dependencies": list(DEPENDENCIES),
            }
        }
    }
    modules = {}
    for name in EXTRA_MODULES:
        package, relative = name.split("/", 1)
        payload = (roots[package] / relative).read_bytes()
        actual = _dependency_contract(payload, name, _covered_imports(name, scope))
        _require(
            actual == inventory["contracts"][name],
            f"unclassified dependency/resource contract: {name}",
        )
        modules[name] = hashlib.sha256(payload).hexdigest()
    # The source classifier attests its own transitive legacy verifier, source
    # pins, three source modules, contract, and NumPy/pandas dependency versions.
    # It reads no survey bytes to produce this implementation document.
    source = demographic._implementation()
    for relative, digest in source["modules"].items():
        name = "microcosm.build/us_runtime/" + relative
        expected = modules.get(name, base["modules"].get(name))
        _require(
            expected == digest, "source implementation module identity: " + relative
        )
    _require(
        all(
            base["dependencies"].get(name) == version
            for name, version in source["dependencies"].items()
        ),
        "source dependency identity",
    )
    return {
        "schema": "microcosm.us.demographic-extension-implementation.v1",
        "stage": STAGE,
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "modules": modules,
        "source_implementation": source,
        "upstream_implementation": base,
        "dependencies": dict(base["dependencies"]),
    }


def implementation_hash(stage=STAGE):
    return hashlib.sha256(
        b"microcosm.us.demographic-extension-implementation.v1\0"
        + _canonical(implementation_manifest(stage))
    ).hexdigest()
