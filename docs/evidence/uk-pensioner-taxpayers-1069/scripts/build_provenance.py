"""Provenance of the baseline and candidate national builds (microcosm#1069).

Reads each build directory's ``build.json`` (the completion record) and
``build_record.json`` and writes, per build, the fields the README cites:
the build id, whether a release was authorised, the H5's sha256, the spine
checkpoint digests, the Chronicle feed pin and the policyengine-uk versions
the build record names.

    python3 build_provenance.py --build baseline=<dir> --build candidate=<dir> \
        --output <json>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ENGINE_VERSION = re.compile(r"policyengine-uk (\d+\.\d+\.\d+)")


def _provenance(directory: Path) -> dict[str, object]:
    completion = json.loads((directory / "build.json").read_text())
    record_text = (directory / "build_record.json").read_text()
    record = json.loads(record_text)
    spine = completion["graph"]["artifacts"]["uk.full.spine_checkpoint"]
    feed = record["run_config"]["chronicle_feed_pin"]
    return {
        "build_id": completion["build_id"],
        "release_authorized": completion["release_authorized"],
        "release_role": completion["release_role"],
        "dataset_sha256": completion["dataset"]["sha256"],
        "spine_build_state": spine["spine_build_state"],
        "spine_provenance": spine["spine_provenance"],
        "chronicle_feed_source_commit": feed["source_commit"],
        "chronicle_feed_facts_sha256": feed["facts_sha256"],
        "input_h5_sha256": record["source_pins"]["input_h5"]["sha256"],
        "policyengine_uk_versions": sorted(set(ENGINE_VERSION.findall(record_text))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    builds = {}
    for item in args.build:
        label, _, directory = item.partition("=")
        builds[label] = _provenance(Path(directory))
    shared = [
        key
        for key in (
            "spine_build_state",
            "spine_provenance",
            "chronicle_feed_source_commit",
            "chronicle_feed_facts_sha256",
            "input_h5_sha256",
        )
        if len({build[key] for build in builds.values()}) == 1
    ]
    output = {"builds": builds, "identical_across_builds": shared}
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
