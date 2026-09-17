"""What this branch moves in the US implementation identity, and what it does not.

``graph_implementation.implementation_manifest`` folds ``sha256`` of the WHOLE
FILE of every module in a stage's roster (``graph_implementation.py:431-435``),
and that manifest reaches ``params["implementation"]`` and therefore
``node_key``. So a receipt of which roster files this branch changed is a
receipt of which node keys and store addresses move.

Run from the worktree root with the workspace interpreter. Reads git objects
and the working tree; writes one JSON file and nothing else.
"""

import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
INVENTORY = (
    "packages/microcosm-build/src/microcosm/build/us_runtime/"
    "graph_implementation_inventory.json"
)
PACKAGE_SOURCE = {
    "microcosm.build": "packages/microcosm-build/src/microcosm/build",
    "microcosm.graph": "packages/microcosm-graph/src/microcosm/graph",
    "microcosm.frame": "packages/microcosm-frame/src/microcosm/frame",
    "microcosm.fit": "packages/microcosm-fit/src/microcosm/fit",
    "microcosm.calibrate": "packages/microcosm-calibrate/src/microcosm/calibrate",
    "microcosm.data": "packages/microcosm-data/src/microcosm/data",
}
# ``microunit`` is an installed dependency, not a path in this repository, so no
# revision of this repository can move it. It is recorded as out of scope rather
# than silently skipped.
OUTSIDE_THE_REPOSITORY = ("microunit",)


def _git(*arguments):
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, check=True
    ).stdout


def _blob(revision, path):
    try:
        return _git("show", f"{revision}:{path}")
    except subprocess.CalledProcessError:
        return None


def _digests(revision, inventory):
    result = {}
    for stage, spec in inventory["stages"].items():
        stage_digests = {}
        for name in spec["modules"]:
            package, relative = name.split("/", 1)
            if package in OUTSIDE_THE_REPOSITORY:
                stage_digests[name] = "outside-the-repository"
                continue
            path = f"{PACKAGE_SOURCE[package]}/{relative}"
            payload = _blob(revision, path)
            stage_digests[name] = (
                None if payload is None else hashlib.sha256(payload).hexdigest()
            )
        result[stage] = stage_digests
    return result


def main():
    revisions = dict(pair.split("=", 1) for pair in sys.argv[1:-1])
    out = pathlib.Path(sys.argv[-1])
    inventory = json.loads((ROOT / INVENTORY).read_bytes())
    per_revision = {label: _digests(rev, inventory) for label, rev in revisions.items()}

    labels = list(revisions)
    base = labels[0]
    moved = {}
    for label in labels[1:]:
        changes = {}
        for stage, digests in per_revision[label].items():
            differing = {
                name: {base: per_revision[base][stage][name], label: digest}
                for name, digest in digests.items()
                if per_revision[base][stage][name] != digest
            }
            if differing:
                changes[stage] = differing
        moved[f"{base}->{label}"] = changes

    # The contract arm: the manifest refuses an unclassified import or resource
    # access, so calling it at the working tree proves this branch added none.
    contracts = {}
    sys.path[:0] = [str(ROOT / p) for p in sorted((ROOT / "packages").glob("*/src"))]
    from microcosm.build.us_runtime.graph_implementation import implementation_manifest

    for stage in inventory["stages"]:
        try:
            manifest = implementation_manifest(stage)
            contracts[stage] = {
                "accepted": True,
                "inventory_sha256": manifest["inventory_sha256"],
                "modules_sha256": hashlib.sha256(
                    json.dumps(manifest["modules"], sort_keys=True).encode()
                ).hexdigest(),
            }
        except Exception as error:  # noqa: BLE001 - the refusal is the result
            contracts[stage] = {"accepted": False, "error": repr(error)}

    out.write_text(
        json.dumps(
            {
                "scope": "diagnostic only; not a build, certification or release "
                "artifact",
                "release_eligible": False,
                "revisions": revisions,
                "stages": sorted(inventory["stages"]),
                "moved_module_digests": moved,
                "working_tree_contract_acceptance": contracts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for comparison, changes in moved.items():
        if not changes:
            print(f"{comparison}: no roster module digest moved")
            continue
        names = sorted({name for stage in changes.values() for name in stage})
        print(f"{comparison}: {len(changes)} stage(s) move; files: {names}")
    refused = [s for s, c in contracts.items() if not c["accepted"]]
    print("contracts accepted at the working tree:", not refused, refused or "")


if __name__ == "__main__":
    main()
