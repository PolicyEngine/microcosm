"""Regenerate an inventory contract through its own generator. Never hand-edit a pin.

``graph_implementation.implementation_manifest`` recomputes every inventoried
module's dependency contract and refuses the stage if one differs from
``graph_implementation_inventory.json``. This rewrites the differing entries
with exactly what ``_dependency_contract`` returns, preserving the file's key
order and its trailing newline, and prints what moved.

    python regenerate_inventory_contract.py [--write]
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(path) for path in sorted((ROOT / "packages").glob("*/src"))]

from microcosm.build.us_runtime import graph_implementation as implementation  # noqa: E402

INVENTORY = (
    ROOT
    / "packages/microcosm-build/src/microcosm/build/us_runtime"
    / "graph_implementation_inventory.json"
)


def main() -> int:
    write = "--write" in sys.argv[1:]
    raw = INVENTORY.read_text()
    inventory = json.loads(raw)
    roots = implementation._package_roots()
    moved = []
    for name, expected in inventory["contracts"].items():
        package, relative = name.split("/", 1)
        payload = (roots[package] / relative).read_bytes()
        actual = implementation._dependency_contract(
            payload, name, implementation._covered_imports(name, inventory)
        )
        if actual != expected:
            moved.append((name, expected, actual))
            inventory["contracts"][name] = actual
    for name, expected, actual in moved:
        print(f"MOVED {name}")
        for key in ("imports", "unbound_uses_sha256", "resource_accesses_sha256"):
            if expected[key] != actual[key]:
                print(f"  {key}")
                print(f"    old: {expected[key]}")
                print(f"    new: {actual[key]}")
    if not moved:
        print("no contract moved")
        return 0
    if not write:
        print("(dry run; pass --write to apply)")
        return 0
    # The committed file is json.dumps(..., indent=2, sort_keys=True) plus a
    # trailing newline; this was verified byte-identical before any edit, so the
    # rewrite moves the changed values and nothing else.
    INVENTORY.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(f"rewrote {INVENTORY.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
