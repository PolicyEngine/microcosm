"""Separately classified CD extension identity; shared source inventory unchanged."""

from __future__ import annotations

import hashlib
import json

from . import enriched_origin_calibration as enriched
from . import graph_native_origin_implementation as native
from .graph_implementation import (
    _canonical,
    _covered_imports,
    _dependency_contract,
    _package_roots,
)
from .graph_implementation import implementation_manifest as upstream_manifest

STAGE = "cd_source_evidence_v1"
DEPENDENCIES = native.DEPENDENCIES
INVENTORY_FILE = "us_runtime/cd_source_graph_inventory.json"
EXTRA_MODULES = (
    "microcosm.build/us_runtime/acs_hu_source_transport.py",
    "microcosm.build/us_runtime/cd_current_source_evidence.py",
    "microcosm.build/us_runtime/graph_cd_source_evidence.py",
    "microcosm.build/us_runtime/graph_cd_source_implementation.py",
    "microcosm.build/cd_benchmark_io/source_adapters.py",
    "microcosm.build/cd_benchmark_io/artifacts.py",
    "microcosm.build/cd_benchmark_io/__init__.py",
    "microcosm.build/cd_benchmark_io/durable.py",
    "microcosm.build/cd_benchmark/__init__.py",
    "microcosm.build/cd_benchmark/approval.py",
    "microcosm.build/cd_benchmark/binding.py",
    "microcosm.build/cd_benchmark/canonical.py",
    "microcosm.build/cd_benchmark/evaluation.py",
    "microcosm.build/cd_benchmark/ledger.py",
    "microcosm.build/cd_benchmark/metrics.py",
    "microcosm.build/cd_benchmark/origin.py",
    "microcosm.build/cd_benchmark/protocol.py",
    "microcosm.build/cd_benchmark/reasons.py",
    "microcosm.build/cd_benchmark/reference.py",
    "microcosm.build/cd_benchmark/support.py",
    "microcosm.build/cd_benchmark/universe.py",
    "microcosm.build/us_runtime/enriched_origin_calibration.py",
)


def implementation_manifest():
    bases = {
        "native": native.implementation_manifest(),
        "geography": upstream_manifest("geography"),
    }
    roots = _package_roots()
    raw = (roots["microcosm.build"] / INVENTORY_FILE).read_bytes()
    inventory = json.loads(raw)
    if (
        set(inventory)
        != {
            "schema",
            "stage",
            "dependencies",
            "extra_modules",
            "contracts",
            "import_classifications",
        }
        or inventory["schema"] != "microcosm.us.cd-source-extension-inventory.v1"
        or inventory["stage"] != STAGE
        or inventory["dependencies"] != list(DEPENDENCIES)
        or inventory["extra_modules"] != list(EXTRA_MODULES)
        or set(inventory["contracts"]) != set(EXTRA_MODULES)
    ):
        raise ValueError("CD_IMPLEMENTATION_INVENTORY")
    imports = {
        name for value in inventory["contracts"].values() for name in value["imports"]
    }
    if set(inventory["import_classifications"]) != imports or not all(
        type(value) is str and value.strip()
        for value in inventory["import_classifications"].values()
    ):
        raise ValueError("CD_IMPORT_CLASSIFICATION")
    modules = set(EXTRA_MODULES)
    # Existing upstream closure identities remain independently authenticated.
    for base in bases["native"]["upstream_implementations"].values():
        modules.update(base["modules"])
    modules.update(bases["native"]["modules"])
    modules.update(bases["geography"]["modules"])
    modules.update(inventory["contracts"])
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
        if actual != inventory["contracts"][name]:
            raise ValueError("CD_UNCLASSIFIED_DEPENDENCY")
        hashes[name] = hashlib.sha256(code).hexdigest()
    protocol = (
        roots["microcosm.build"] / "cd_benchmark/us-cd-published-benchmark-v4.json"
    ).read_bytes()
    return {
        "schema": "microcosm.us.cd-source-extension-implementation.v1",
        "inventory_sha256": hashlib.sha256(raw).hexdigest(),
        "modules": hashes,
        "upstream": bases,
        "enriched_parent_implementation": enriched.implementation_hash(),
        "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
        "dependencies": bases["geography"]["dependencies"],
    }


def implementation_hash():
    return hashlib.sha256(
        b"microcosm.us.cd-source-extension-implementation.v1\0"
        + _canonical(implementation_manifest())
    ).hexdigest()
