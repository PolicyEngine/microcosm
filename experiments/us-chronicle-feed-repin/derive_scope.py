"""Derive the US Chronicle feed scope file from the pinned feed and the scratch exports.

One-time derivation, kept for the record. The scope rule (decided 2026-09-18) is:
keep exactly the pinned feed's (record set, period) pairs, so the family surface the
release parity gate reviews does not move in the re-pin. For each pair the file names
the Chronicle source package that emits it and the ``--year`` to build that package
with, so ``tools/build_us_chronicle_feed.py`` can rebuild the feed from targeted
single-package runs without any whole-year bundle.

Build-year rule: a package's rows are year-specific for some packages (IRS SOI Table
1.4 emits one tax year per run) and year-independent for others (BEA NIPA emits the
same rows for every year). The build year is the pair's own period year when the
package emits the pair for that year; otherwise it is the earliest export year that
emits it. Every choice below was observed in the scratch exports at Chronicle
``c5e5bf8aa84960c1a200ee47303b19c953092d0f``; the builder verifies each choice again
(a run that does not emit its pair refuses).

Usage (paths default to the 2026-09-18 scratch layout):

    uv run python experiments/us-chronicle-feed-repin/derive_scope.py \
        --pinned ~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl \
        --exports ~/PolicyEngine/_recovered/scratch-backup/893/chronicle-export \
        --aliases ~/PolicyEngine/_recovered/scratch-backup/893/chronicle-export/source_package_aliases.json \
        --out packages/microcosm-build/src/microcosm/build/us/chronicle_feed_scope.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

CHRONICLE_COMMIT = "c5e5bf8aa84960c1a200ee47303b19c953092d0f"


def _pair(row: dict) -> tuple[str, str, str]:
    period = row["period"]
    return (row["layout"]["record_set_id"], period["type"], str(period["value"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    alias_to_path = json.loads(args.aliases.read_text())
    path_to_alias = {f"packages/{v}": k for k, v in alias_to_path.items()}

    pinned_pairs: set[tuple[str, str, str]] = set()
    with args.pinned.open() as stream:
        for line in stream:
            if line.strip():
                pinned_pairs.add(_pair(json.loads(line)))

    producers: dict[tuple[str, str, str], set[tuple[str, str]]] = (
        collections.defaultdict(set)
    )
    listings = glob.glob(str(args.exports / "ledger-us-*" / "source_packages.json"))
    listings += glob.glob(str(args.exports / "targeted" / "*" / "source_packages.json"))
    for listing in sorted(listings):
        payload = json.loads(Path(listing).read_text())
        year = str(payload["year"])
        for entry in payload["source_packages"]:
            facts = entry["outputs"].get("consumer_facts")
            if not facts or not Path(facts).exists() or not entry.get("valid", True):
                continue
            source = entry["source"]
            alias = path_to_alias.get(source, source)
            if alias not in alias_to_path:
                raise SystemExit(f"{listing}: source {source!r} is not a known alias")
            with open(facts) as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    pair = _pair(json.loads(line))
                    if pair in pinned_pairs:
                        producers[pair].add((alias, year))

    missing = sorted(pinned_pairs - set(producers))
    if missing:
        raise SystemExit(
            f"{len(missing)} pinned pairs have no producer: {missing[:10]}"
        )

    entries = []
    for pair in sorted(pinned_pairs):
        aliases = {alias for alias, _ in producers[pair]}
        if len(aliases) != 1:
            raise SystemExit(
                f"{pair}: emitted by more than one package: {sorted(aliases)}"
            )
        alias = aliases.pop()
        years = sorted(year for _, year in producers[pair])
        period_year = pair[2][:4]
        build_year = period_year if period_year in years else years[0]
        entries.append(
            {
                "record_set_id": pair[0],
                "period_type": pair[1],
                "period_value": pair[2],
                "package_id": alias,
                "package": f"packages/{alias_to_path[alias]}",
                "build_year": int(build_year),
            }
        )

    runs = collections.defaultdict(set)
    for entry in entries:
        runs[entry["build_year"]].add(entry["package"])
    scope = {
        "version": 1,
        "country": "us",
        "source_repo": "PolicyEngine/chronicle",
        "source_commit": CHRONICLE_COMMIT,
        "rule": (
            "Exactly the (record_set_id, period) pairs of the pinned feed "
            "consumer_facts_buildn_v9_4.jsonl; each built from the named package "
            "with `chronicle build-bundle --year <build_year> --source <package>`."
        ),
        "pair_count": len(entries),
        "runs": {
            str(year): sorted(packages) for year, packages in sorted(runs.items())
        },
        "pairs": entries,
    }
    args.out.write_text(json.dumps(scope, indent=1, sort_keys=False) + "\n")
    print(
        f"{len(entries)} pairs; runs per year: "
        f"{ {y: len(p) for y, p in sorted(runs.items())} }; "
        f"{sum(len(p) for p in runs.values())} package runs; wrote {args.out}"
    )


if __name__ == "__main__":
    main()
