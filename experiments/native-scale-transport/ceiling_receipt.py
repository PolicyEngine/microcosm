"""Feed the pilot's own preparation receipt, scaled, through both transports.

Reads the recovered 1/1000 artifact, replicates its per-household lists to the
household count a given fraction of the US source implies, and reports what the
single bounded encode does with it against what the segmented transport does.
No graph runs; nothing is written outside the directory given on the command
line. Not a build, not a certification, not release eligible.

    <venv>/bin/python ceiling_receipt.py <artifact.json> <out.json> [spill-dir]
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path[:0] = [
    str(path)
    for path in sorted(
        (pathlib.Path(__file__).resolve().parents[2] / "packages").glob("*/src")
    )
]

from microcosm.build.us_runtime import survey_population_preparation as owner  # noqa: E402

SOURCE_HOUSEHOLDS = 1_587_376  # selection.supplied_households, from the artifact
FRACTIONS = ((1, 1000), (1, 100), (1, 10), (1, 1))


def _scaled(document, factor):
    """Replicate every per-household or per-person list `factor` times."""
    result = json.loads(json.dumps(document))
    selection = result["selection"]
    for name in ("selected", "excluded", "cells"):
        selection[name] = selection[name] * factor
    origins = result["origins"]
    origins["households"] = origins["households"] * factor
    origins["persons"]["rows"] = origins["persons"]["rows"] * factor
    for entity, rows in origins["entities"].items():
        origins["entities"][entity] = rows * factor
    return result


def main():
    artifact = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    spill = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
    raw = artifact.read_bytes()
    document = json.loads(raw)
    selected = len(document["selection"]["selected"])
    supplied = document["selection"]["supplied_households"]
    canonical = owner._encode(document)
    rows = {
        "artifact": str(artifact),
        "artifact_bytes": len(raw),
        "artifact_sha256": owner._sha(raw),
        "canonical_bytes": len(canonical),
        "canonical_equals_file": canonical == raw,
        "selected_households": selected,
        "supplied_households": supplied,
        "source_households_assumed": SOURCE_HOUSEHOLDS,
        "scales": [],
    }
    for numerator, denominator in FRACTIONS:
        target = SOURCE_HOUSEHOLDS * numerator // denominator
        factor = max(1, round(target / selected))
        scaled = _scaled(document, factor)
        row = {
            "fraction": [numerator, denominator],
            "target_households": target,
            "replication_factor": factor,
            "households": selected * factor,
        }
        started = time.process_time()
        try:
            row["single_encode_bytes"] = len(owner._encode(scaled))
            row["single_encode"] = "accepted"
        except owner.SurveyPopulationPreparationError as error:
            row["single_encode"] = "REFUSED: " + type(error).__name__ + " " + str(error)
        row["single_encode_cpu_seconds"] = round(time.process_time() - started, 3)
        started = time.process_time()
        try:
            payload, header = owner._roster_payload(
                scaled,
                spill=spill,
                name="ceiling_receipt" if spill else None,
            )
            row["roster_bytes"] = len(payload)
            row["roster_segments"] = len(header["segments"]) if header else None
            row["roster_digest"] = owner._sha(payload)
            row["roster_digest_equals_stream_digest"] = owner._sha(
                payload
            ) == owner._digest(scaled)
            row["roster"] = "accepted"
        except owner.SurveyPopulationPreparationError as error:
            row["roster"] = "REFUSED: " + type(error).__name__ + " " + str(error)
        row["roster_cpu_seconds"] = round(time.process_time() - started, 3)
        if row.get("single_encode") == "accepted" and row.get("roster") == "accepted":
            row["bytes_identical"] = row["single_encode_bytes"] == row["roster_bytes"]
        rows["scales"].append(row)
        del scaled
    rows["release_eligible"] = False
    rows["scope"] = (
        "Transport comparison over a replicated copy of one recovered "
        "preparation receipt. Not a build, not a certification, not release "
        "eligible."
    )
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
