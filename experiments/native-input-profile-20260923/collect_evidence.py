"""Collect mechanical source evidence for the native input-profile inventory.

Read-only: runs ``git grep``/``git diff`` against pinned commits and writes one
JSON file. It imports no microcosm module, reads no survey data and runs no
engine. A string-literal hit is evidence that a module *names* an input; it is
not proof that the module produces or attaches it. ``classification.json``
holds the reviewed judgement for every name; ``build_inventory.py`` merges both.

Usage (from the repository root)::

    python3 experiments/native-input-profile-20260923/collect_evidence.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "evidence.json"

INTEGRATION = "47960af430ff7eaec29bfda504a3c814a9a6c264"
# Merge base of the native integration line with main (PR #938 merge).
NATIVE_BASE = "8c44daa52354b4e28af315a486df0b0c006f26ff"
MAIN = "4305a7d34ded67009618990f98c5521731d66e73"  # origin/main, #959 merged
AUDIT = "fef2cb56a4ca24d74e217a0cbe4abab7079da723"
PR991 = "c3456bb10c4722422ffd2c9dab3ae11baec5c3a9"  # us-asec-720-coverage-recodes
# Maximal 2026-09-21 descendant tips of INTEGRATION (branch -> commit).
DESCENDANTS = {
    "native-agi-tail-matching-20260921": "1b35534b5",
    "native-child-pre-geography-integration-20260921": "050c58a60",
    "native-survey-tail-recipients-20260921": "5067f21a4",
    "native-household-domain-projection-20260921": "292c268a0",
    "native-ss-beneficiary-core-20260921": "fa72c60f9",
    "native-other-disability-context-replay-20260921": "aa2eaab11",
    "native-other-disability-completion-20260921": "0e073d127",
}
SRC = "packages/microcosm-build/src/microcosm/build/"
MAIN_PATHS = (
    SRC + "us_runtime",
    SRC + "us",
    "tools/build_us_puf_support_base.py",
    "tools/build_us_fiscal_refresh_release.py",
)


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True
    ).stdout


def resolve(ref: str) -> str:
    return git("rev-parse", "--verify", ref + "^{commit}").strip()


def manifest(ref: str) -> dict:
    return json.loads(
        git("show", f"{ref}:{SRC}us/release_input_coverage_manifest.json")
    )


def grep(ref: str, names: list[str], paths: list[str]) -> dict[str, list[str]]:
    """Return name -> ["path:line", ...] for exact quoted-literal hits."""
    if not paths:
        return {}
    patterns = []
    for name in names:
        patterns += ["-e", f'"{name}"']
    hits: dict[str, list[str]] = defaultdict(list)
    chunk = 200
    for start in range(0, len(paths), chunk):
        proc = subprocess.run(
            (
                "git",
                "grep",
                "-n",
                "-F",
                *patterns,
                ref,
                "--",
                *paths[start : start + chunk],
            ),
            capture_output=True,
            text=True,
        )
        if proc.returncode not in (0, 1):
            raise RuntimeError(proc.stderr)
        for line in proc.stdout.splitlines():
            _, path, lineno, text = line.split(":", 3)
            for name in names:
                if f'"{name}"' in text:
                    hits[name].append(f"{path.removeprefix(SRC)}:{lineno}")
    return hits


def native_added_files(ref: str) -> list[str]:
    rows = git("diff", "--name-status", "--no-renames", NATIVE_BASE, ref, "--", SRC)
    return sorted(
        path
        for status, path in (row.split("\t", 1) for row in rows.splitlines())
        if status == "A" and path.endswith(".py") and "/tests/" not in path
    )


def changed_files(base: str, tip: str) -> list[str]:
    rows = git("diff", "--name-only", "--no-renames", base, tip, "--", SRC)
    return sorted(p for p in rows.splitlines() if p.endswith(".py"))


def main() -> int:
    main_manifest = manifest(MAIN)
    names = sorted(main_manifest["columns"])
    integration = resolve(INTEGRATION)
    native_files = native_added_files(integration)
    native_hits = grep(integration, names, native_files)
    descendants = {}
    for branch, tip in DESCENDANTS.items():
        full = resolve(tip)
        files = changed_files(integration, full)
        at_tip = grep(full, names, files)
        at_base = grep(integration, names, [f for f in files if f in native_files])
        added = {
            name: sorted(set(lines) - set(at_base.get(name, ())))
            for name, lines in at_tip.items()
        }
        descendants[branch] = {
            "commit": full,
            "changed_python_files": [f.removeprefix(SRC) for f in files],
            "new_literal_hits": {k: v for k, v in sorted(added.items()) if v},
        }
    main_files = [
        p
        for p in git(
            "ls-tree", "-r", "--name-only", MAIN, "--", *MAIN_PATHS
        ).splitlines()
        if p.endswith(".py") and "/tests/" not in p
    ]
    main_hits = grep(MAIN, names, main_files)
    main_by_file = {
        name: dict(
            sorted(_count_by_file(lines).items(), key=lambda item: (-item[1], item[0]))
        )
        for name, lines in main_hits.items()
    }
    doc = {
        "protocol": "microcosm.us.native-input-profile-evidence.v1",
        "note": (
            "Mechanical quoted-literal hits only. A hit shows a module names an "
            "input; it does not establish production, attachment, source signal "
            "or applicability. See classification.json for reviewed judgements."
        ),
        "commits": {
            "integration": integration,
            "native_base": resolve(NATIVE_BASE),
            "main": resolve(MAIN),
            "audit": resolve(AUDIT),
            "pr991": resolve(PR991),
        },
        "main_manifest_sha256": hashlib.sha256(
            git("show", f"{MAIN}:{SRC}us/release_input_coverage_manifest.json").encode()
        ).hexdigest(),
        "main_manifest_counts": main_manifest["counts"],
        "native_added_module_count": len(native_files),
        "names": {
            name: {
                "main_status": main_manifest["columns"][name]["status"],
                "native_literal_hits_at_integration": native_hits.get(name, []),
                "main_literal_hits_by_file": main_by_file.get(name, {}),
            }
            for name in names
        },
        "descendants": descendants,
    }
    OUT.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({len(names)} names)")
    return 0


def _count_by_file(lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for line in lines:
        counts[line.rsplit(":", 1)[0]] += 1
    return counts


if __name__ == "__main__":
    sys.exit(main())
