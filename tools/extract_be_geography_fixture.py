"""Extract test_be_geography.py's fixture from Belgium's pinned Chronicle artifact.

The fixture holds, byte for byte and in artifact order, the ``consumer_facts.jsonl``
lines of two Statbel record sets: the 565 NIS 2025 commune population totals and
the 18 NUTS1 population cells. It is gzipped at level 9 with mtime 0 and no file
name, so the same lines give the same bytes. A provenance file beside it records
each line's artifact line number and sha256, and the artifact it came from.

The tool refuses an artifact whose ``consumer_facts.jsonl`` or ``manifest.json``
sha256 differs from ``be/chronicle_feed.json``'s pin, hashing the facts in the
same pass that selects the lines, and it takes the source commit from that pin.
Maintainers run it after the pin moves; CI has no artifact to run it on, and
test_be_geography.py's fixture test fails until the provenance names the pinned
artifact. ``--check`` compares the committed bytes with a fresh extraction and
never writes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEED = ROOT / "packages/microcosm-build/src/microcosm/build/be/chronicle_feed.json"
FIXTURES = ROOT / "packages/microcosm-build/tests/fixtures"
FIXTURE = "be_statbel_commune_population_2025.jsonl.gz"
PROVENANCE = f"{FIXTURE}.provenance.json"
# record_set_id: (record_set name in the provenance, expected line count).
RECORD_SETS = {
    "statbel.population_structure.cy2025.people.by_commune_nis2025.total": (
        "commune_total",
        565,
    ),
    "statbel.population_structure.cy2025.people.by_nuts1_age_sex": (
        "nuts1_age_sex",
        18,
    ),
}
# A byte prefix every selected line holds, so most lines skip the JSON parse.
MARKER = b"statbel.population_structure.cy2025.people.by_"


def extract(artifact: Path, feed: dict) -> dict[str, bytes]:
    """Return the fixture's and the provenance's bytes, keyed by file name."""

    unpinned = ValueError(
        f"{artifact} is not the artifact be/chronicle_feed.json pins."
    )
    manifest_sha = hashlib.sha256((artifact / "manifest.json").read_bytes()).hexdigest()
    if manifest_sha != feed["manifest_sha256"]:
        raise unpinned
    # One pass hashes every line it selects from, so the selected lines and the
    # pin check come from the same read. A line problem waits for the pin check,
    # so an unpinned artifact is refused as unpinned.
    digest = hashlib.sha256()
    lines, records, problem = [], [], None
    with (artifact / "consumer_facts.jsonl").open("rb") as handle:
        for number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if problem or MARKER not in raw:
                continue
            try:
                fact = json.loads(raw)
                selected = RECORD_SETS.get(fact["layout"]["record_set_id"])
                if selected is not None:
                    geography_id, value = fact["geography"]["id"], fact["value"]
            except (ValueError, KeyError, TypeError) as error:
                problem = ValueError(f"Artifact line {number} is not a fact: {error}")
                continue
            if selected is None:
                continue
            if not raw.endswith(b"\n"):
                problem = ValueError(f"Artifact line {number} has no trailing newline.")
                continue
            lines.append(raw)
            records.append(
                {
                    "artifact_line": number,
                    "record_set": selected[0],
                    "geography_id": geography_id,
                    "value": value,
                    "sha256": hashlib.sha256(raw[:-1]).hexdigest(),
                }
            )
    if digest.hexdigest() != feed["facts_sha256"]:
        raise unpinned
    if problem:
        raise problem
    expected = dict(RECORD_SETS.values())
    counts = {name: sum(r["record_set"] == name for r in records) for name in expected}
    if counts != expected:
        raise ValueError(f"Selected line counts {counts} differ from {expected}.")
    data = b"".join(lines)
    (first, (first_name, first_count)), (second, (second_name, second_count)) = (
        RECORD_SETS.items()
    )
    provenance = {
        "schema_version": 1,
        "kind": "be_statbel_commune_population_2025_fixture",
        "use": (
            "test_be_geography.py compiles Belgium's 565 atomic_geography_support "
            "precal rows from these facts, builds the support from "
            "be/transport_atomic_geography_support.json, runs assignment, "
            "derivation and the gate, and checks region sums against the NUTS1 "
            "facts."
        ),
        "source": {
            "artifact": (
                "Belgium's pinned Chronicle consumer artifact (be/chronicle_feed.json)"
            ),
            "chronicle_source_commit": feed["source_commit"],
            "facts_sha256": feed["facts_sha256"],
            "manifest_sha256": feed["manifest_sha256"],
        },
        "selection": (
            "The artifact's consumer_facts.jsonl lines whose layout.record_set_id "
            f"is {first} ({first_count} lines, record_set {first_name}) or "
            f"{second} ({second_count} lines, record_set {second_name}), copied "
            "byte for byte in artifact order (the two sets interleave) and gzipped "
            "at level 9 with mtime 0 and no file name."
        ),
        "record_sets": list(RECORD_SETS),
        "line_sha256": (
            "Each lines entry's sha256 is of that line's bytes without its "
            "trailing newline."
        ),
        "lines": records,
        "output_sha256": hashlib.sha256(data).hexdigest(),
        "output_bytes": len(data),
        "refresh": (
            "Re-extract with tools/extract_be_geography_fixture.py whenever "
            "be/chronicle_feed.json's pin moves; test_be_geography.py's fixture "
            "test fails until then. The feed test, run with BE_CHRONICLE_ARTIFACT, "
            "checks that a fresh extraction gives these bytes."
        ),
    }
    text = json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
    return {
        FIXTURE: gzip.compress(data, compresslevel=9, mtime=0),
        PROVENANCE: text.encode("utf-8"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="the consumer artifact directory")
    parser.add_argument("--out", type=Path, default=FIXTURES)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    feed = json.loads(FEED.read_text(encoding="utf-8"))
    files = extract(args.artifact, feed)
    if args.check:
        stale = [
            str(args.out / name)
            for name, data in files.items()
            if not (args.out / name).exists() or (args.out / name).read_bytes() != data
        ]
        if stale:
            raise SystemExit(f"Stale fixture file(s): {stale}")
        return
    for name, data in files.items():
        (args.out / name).write_bytes(data)


if __name__ == "__main__":
    main()
