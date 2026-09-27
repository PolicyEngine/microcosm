"""Closed additional identity for native-origin nodes; upstream bytes stay fixed.

The shared v1 service and inventory are unchanged. Their complete composition
and ACS-source scopes are validated before adding this independently classified
inventory. A new origin helper or registry changes only these extension keys.
"""

from __future__ import annotations

import csv
import hashlib
import json

from . import native_household_origin as native
from .graph_implementation import (
    STAGE_DEPENDENCIES,
    _canonical,
    _covered_imports,
    _dependency_contract,
    _package_roots,
)
from .graph_implementation import implementation_manifest as upstream_manifest

STAGE = "native_household_origin_v1"
BASE_STAGES = ("composed_asec_binding_v1", "acs_housing_universe_2024")
DEPENDENCIES = STAGE_DEPENDENCIES[BASE_STAGES[0]]
INVENTORY_FILE = "us_runtime/native_origin_graph_inventory.json"
INVENTORY_SCHEMA = "microcosm.us.native-origin-extension-inventory.v1"
EXTRA_MODULES = (
    "microcosm.build/cd_benchmark/canonical.py",
    "microcosm.build/cd_benchmark/origin.py",
    "microcosm.build/us_runtime/native_household_origin.py",
    "microcosm.build/us_runtime/graph_native_household_origin.py",
    "microcosm.build/us_runtime/graph_native_origin_implementation.py",
)


def implementation_manifest():
    """Verify all live classified dependencies without opening any survey file."""
    bases = {stage: upstream_manifest(stage) for stage in BASE_STAGES}
    roots = _package_roots()
    payload = (roots["microcosm.build"] / INVENTORY_FILE).read_bytes()
    inventory = json.loads(payload)
    native._require(
        set(inventory)
        == {
            "schema",
            "stage",
            "base_stages",
            "dependencies",
            "extra_modules",
            "contracts",
            "import_classifications",
        }
        and inventory["schema"] == INVENTORY_SCHEMA
        and inventory["stage"] == STAGE
        and inventory["base_stages"] == list(BASE_STAGES)
        and inventory["dependencies"] == list(DEPENDENCIES)
        and inventory["extra_modules"] == list(EXTRA_MODULES)
        and set(inventory["contracts"]) == set(EXTRA_MODULES),
        "ORIGIN_IMPLEMENTATION_INVENTORY",
    )
    imports = {
        name
        for contract in inventory["contracts"].values()
        for name in contract["imports"]
    }
    native._require(
        set(inventory["import_classifications"]) == imports
        and all(
            isinstance(label, str) and bool(label.strip())
            for label in inventory["import_classifications"].values()
        ),
        "ORIGIN_IMPORT_CLASSIFICATION",
    )
    modules = set(EXTRA_MODULES)
    for base in bases.values():
        modules.update(base["modules"])
    scope = {
        "stages": {
            STAGE: {"modules": sorted(modules), "dependencies": list(DEPENDENCIES)}
        }
    }
    hashes = {}
    for name in EXTRA_MODULES:
        package, relative = name.split("/", 1)
        code = (roots[package] / relative).read_bytes()
        actual = _dependency_contract(code, name, _covered_imports(name, scope))
        native._require(
            actual == inventory["contracts"][name], "ORIGIN_UNCLASSIFIED_DEPENDENCY"
        )
        hashes[name] = hashlib.sha256(code).hexdigest()
    return {
        "schema": "microcosm.us.native-origin-extension-implementation.v1",
        "stage": STAGE,
        "inventory_sha256": hashlib.sha256(payload).hexdigest(),
        "modules": hashes,
        "upstream_implementations": bases,
        "dependencies": dict(bases[BASE_STAGES[0]]["dependencies"]),
        "acs_source_implementation": native.acs._implementation(),
        "asec_household_member_registry": native.asec_member_registry(),
        "csv_field_size_limit": csv.field_size_limit(),
    }


def implementation_hash():
    """Identity of the additional implementation, not a replacement upstream key."""
    return hashlib.sha256(
        b"microcosm.us.native-origin-extension-implementation.v1\0"
        + _canonical(implementation_manifest())
    ).hexdigest()
