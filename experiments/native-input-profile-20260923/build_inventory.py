"""Merge reviewed classification, mechanical evidence and the historical audit.

Writes ``inventory.json``. Every citation token is resolved to ``path:line`` at
its pinned commit with ``git grep -F``; an unresolved token refuses the build,
so the inventory cannot cite a line that does not exist. Every name in main's
manifest must be classified exactly once, and no extra names are allowed.

Read-only apart from the output file. Run from the repository root after
``collect_evidence.py``::

    python3 experiments/native-input-profile-20260923/build_inventory.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import classification  # noqa: E402

EVIDENCE = HERE / "evidence.json"
AUDIT = HERE / "audit_fef2cb56a_native159.json"
HEADERS = HERE / "source_header_observations.json"
OUT = HERE / "inventory.json"
CLASSES = (
    "native_successor_exists",
    "port_needed",
    "source_absent",
    "declared_scope_exclusion",
)


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True
    ).stdout


def _refs(evidence: dict) -> dict[str, str]:
    refs = {
        classification.INTEGRATION: evidence["commits"]["integration"],
        classification.MAIN: evidence["commits"]["main"],
    }
    for branch, value in evidence["descendants"].items():
        refs[branch] = value["commit"]
    return refs


def resolve(cite: dict, refs: dict[str, str]) -> str:
    commit = refs[cite["ref"]]
    proc = subprocess.run(
        ("git", "grep", "-n", "-F", "-e", cite["token"], commit, "--", cite["path"]),
        capture_output=True,
        text=True,
    )
    lines = proc.stdout.splitlines()
    if proc.returncode != 0 or not lines:
        raise SystemExit(
            f"UNRESOLVED CITATION {cite['ref']}:{cite['path']} token {cite['token']!r}"
        )
    _, path, lineno, _ = lines[0].split(":", 3)
    short = path.removeprefix("packages/microcosm-build/src/microcosm/build/")
    return f"{short}:{lineno}@{commit[:9]}"


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text())
    audit = json.loads(AUDIT.read_text())
    audit_rows = {row["name"]: row for row in audit["rows"]}
    refs = _refs(evidence)
    names = set(evidence["names"])
    seen = collections.Counter(row["name"] for row in classification.ROWS)
    duplicates = sorted(n for n, k in seen.items() if k > 1)
    unknown = sorted(set(seen) - names)
    missing = sorted(names - set(seen))
    if duplicates or unknown or missing:
        raise SystemExit(
            f"CLASSIFICATION ROSTER duplicates={duplicates} unknown={unknown} missing={missing}"
        )
    rows = []
    for spec in sorted(classification.ROWS, key=lambda r: r["name"]):
        if spec["class"] not in CLASSES:
            raise SystemExit(f"CLASS {spec['name']} {spec['class']}")
        name = spec["name"]
        mech = evidence["names"][name]
        historical = audit_rows.get(name)
        descendant_hits = {
            branch: value["new_literal_hits"][name]
            for branch, value in evidence["descendants"].items()
            if name in value["new_literal_hits"]
        }
        rows.append(
            {
                "name": name,
                "main_manifest_status": mech["main_status"],
                "in_native159_profile": historical is not None,
                "historical_fef2cb56a_status": historical["status"]
                if historical
                else None,
                "class": spec["class"],
                "state": spec["state"],
                "native": [resolve(c, refs) for c in spec["native"]],
                "native_source": [resolve(c, refs) for c in spec["native_source"]],
                "port_from": [resolve(c, refs) for c in spec["port_from"]],
                "evidence": [
                    resolve(e, refs) if isinstance(e, dict) else e
                    for e in spec["evidence"]
                ],
                "note": spec["note"],
                "mechanical": {
                    "native_literal_hits_at_integration": len(
                        mech["native_literal_hits_at_integration"]
                    ),
                    "descendant_new_literal_hits": descendant_hits,
                    "main_files_naming_it": list(mech["main_literal_hits_by_file"])[:8],
                },
            }
        )
    by_class = collections.Counter(r["class"] for r in rows)
    by_state = collections.Counter((r["class"], r["state"]) for r in rows)
    historical76 = [
        r
        for r in rows
        if r["historical_fef2cb56a_status"] == "missing_native_successor"
    ]
    doc = {
        "protocol": "microcosm.us.native-input-profile-inventory.v1",
        "scope": (
            "Source-code classification of every input in main's release input "
            "coverage manifest at the native integration commit and its 2026-09-21 "
            "descendants. Not an actual-data missingness count, source-signal "
            "check, or scientific qualification."
        ),
        "commits": evidence["commits"],
        "descendants": {b: v["commit"] for b, v in evidence["descendants"].items()},
        "main_manifest_sha256": evidence["main_manifest_sha256"],
        "main_manifest_counts": evidence["main_manifest_counts"],
        "audit_source": {
            "root_commit": audit["root_commit"],
            "source": audit["source"],
            "source_sha256": audit["source_sha256"],
            "profile": audit["profile"],
        },
        "source_header_observations_sha256": hashlib.sha256(
            HEADERS.read_bytes()
        ).hexdigest(),
        "current_development_request": classification.REQUEST,
        "counts": {
            "names": len(rows),
            "by_class": dict(sorted(by_class.items())),
            "by_class_and_state": {
                f"{c}/{s}": k for (c, s), k in sorted(by_state.items())
            },
            "historical_missing76_now": dict(
                sorted(collections.Counter(r["class"] for r in historical76).items())
            ),
            "historical_missing76_now_by_state": dict(
                sorted(
                    collections.Counter(
                        f"{r['class']}/{r['state']}" for r in historical76
                    ).items()
                )
            ),
        },
        "rows": rows,
    }
    OUT.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    print(json.dumps(doc["counts"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
