"""Re-derive every pin these edits can move, each through its own generator.

Three pin families touch the four edited modules:

1. ``graph_implementation_inventory.json``'s per-module ``contracts`` entry --
   ``imports``, ``unbound_uses_sha256`` and ``resource_accesses_sha256``,
   re-derived through ``graph_graph_implementation._dependency_contract`` against the
   module's own ``_covered_imports``, exactly as ``implementation_manifest``
   checks it. A constant's value is not an import, an unbound use or a resource
   access, so these are expected to hold; the point is to prove it rather than
   assume it.
2. ``acs_native_coverage_acs_native_coverage_binding._ACCEPTED``, which pins four ACS modules by
   whole-file sha256. ``acs_pums.py`` is one of them, so that pin moves.
3. Each declared stage's ``implementation_manifest``, whose ``modules`` map is
   the same whole-file sha256 per inventoried module. These are computed, not
   committed, but every node key and store address is downstream of them, so
   the manifests are rebuilt here to prove they still build.

Nothing is written outside the output path. Not a build, not a certification,
not release eligible.

    python repin.py <out.json>
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(path) for path in sorted((ROOT / "packages").glob("*/src"))]

from microcosm.build.us_runtime import acs_native_coverage_binding, graph_implementation

# The sys.path prelude above must win: a pin re-derived from some other checkout
# would be worthless. Prove the modules resolved inside this tree.
for _module in (acs_native_coverage_binding, graph_implementation):
    if not pathlib.Path(_module.__file__).resolve().is_relative_to(ROOT):
        raise SystemExit(f"{_module.__name__} resolved outside {ROOT}")

US = ROOT / "packages/microcosm-build/src/microcosm/build/us_runtime"
EDITED = (
    "acs_pums.py",
    "acs_person_coverage_columns.py",
    "survey_observed_age.py",
    "survey_origin_budget.py",
)


def main() -> int:
    out = pathlib.Path(sys.argv[1])
    inventory = json.loads((US / "graph_implementation_inventory.json").read_bytes())

    # 1. Inventory contracts, re-derived through the generator for every entry.
    moved, checked = [], 0
    for name, expected in inventory["contracts"].items():
        package, relative = name.split("/", 1)
        roots = graph_implementation._package_roots()
        payload = (roots[package] / relative).read_bytes()
        actual = graph_implementation._dependency_contract(
            payload, name, graph_implementation._covered_imports(name, inventory)
        )
        checked += 1
        if actual != expected:
            moved.append({"contract": name, "old": expected, "new": actual})

    # 2. The ACS whole-file pin that names one of the edited modules.
    accepted = []
    for name, expected in acs_native_coverage_binding._ACCEPTED.items():
        actual = hashlib.sha256((US / name).read_bytes()).hexdigest()
        accepted.append(
            {
                "module": name,
                "old": expected,
                "new": actual,
                "moved": actual != expected,
                "edited_by_this_branch": name in EDITED,
            }
        )

    # 3. Every declared stage manifest, rebuilt through its generator.
    stages = {}
    for stage in sorted(graph_implementation.STAGE_DEPENDENCIES):
        manifest = graph_implementation.implementation_manifest(stage)
        stages[stage] = {
            "inventory_sha256": manifest["inventory_sha256"],
            "module_count": len(manifest["modules"]),
            "edited_modules_in_stage": {
                name: digest
                for name, digest in manifest["modules"].items()
                if pathlib.PurePosixPath(name).name in EDITED
            },
        }

    record = {
        "scope": (
            "Pin re-derivation for the row-ceiling lane. Reads the working tree and "
            "rebuilds each pin through its own generator. Not a build, not a "
            "certification, not release eligible."
        ),
        "release_eligible": False,
        "edited_modules": {
            name: hashlib.sha256((US / name).read_bytes()).hexdigest()
            for name in EDITED
        },
        "inventory_contracts": {
            "generator": (
                "graph_graph_implementation._dependency_contract(payload, name, "
                "graph_graph_implementation._covered_imports(name, inventory))"
            ),
            "checked": checked,
            "moved": moved,
        },
        "acs_native_coverage_binding_accepted": {
            "generator": "hashlib.sha256(<module>.read_bytes()).hexdigest()",
            "entries": accepted,
        },
        "stage_manifests": {
            "generator": "graph_graph_implementation.implementation_manifest(stage)",
            "built": len(stages),
            "stages": stages,
        },
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"inventory contracts checked: {checked}; moved: {len(moved)}")
    for row in moved:
        print("  MOVED", row["contract"])
    print("acs_native_coverage_binding._ACCEPTED:")
    for row in accepted:
        flag = "MOVED" if row["moved"] else "unchanged"
        print(f"  {row['module']}: {flag}")
        if row["moved"]:
            print(f"    old {row['old']}")
            print(f"    new {row['new']}")
    print(f"stage manifests built: {len(stages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
