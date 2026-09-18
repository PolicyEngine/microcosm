"""Rebuild the discrimination battery's receipt from the battery's own rows.

The test file appends one JSON row per driven comparison when
``MICROCOSM_BATTERY_RECEIPT`` names a path -- the test name, which operand was
compared, the object path's verdict and the seal path's. This runs the battery
with that hook on and aggregates those rows, so every figure the report quotes
about the battery (``comparisons``, ``agreements``, ``by_verdict``) is derived
here rather than counted by hand.

    python experiments/native-retention-seal/battery_receipt.py OUT.json

``unreachable_codes`` is the one field that is not a count: it is a claim about
why two of the module's codes cannot be reached, each measured once and carried
forward with its reason. The tests that pin the three codes this change *adds*
drive no comparison -- they are one-sided guards on a malformed seal record,
with no counterpart on the object path -- so they add no row and the comparison
count is unchanged by them.
"""

import collections
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
BATTERY = "packages/microcosm-build/tests/test_us_survey_population_replay.py"
UNREACHABLE = {
    "SURVEY_POPULATION_REPLAY_FRAME_TYPE": (
        "Population validates frame is a Frame (population.py), so no "
        "Population can carry a non-Frame"
    ),
    "SURVEY_POPULATION_REPLAY_STRING_POLICY": (
        "StringDtype.__eq__ compares storage and na_value, so "
        "SERIES_DTYPE_OR_LENGTH fires first"
    ),
}


def main():
    out = pathlib.Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as scratch:
        rows_path = pathlib.Path(scratch) / "rows.jsonl"
        completed = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "-m", "pytest", BATTERY],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={
                **dict(__import__("os").environ),
                "MICROCOSM_BATTERY_RECEIPT": str(rows_path),
            },
        )
        summary = next(
            (
                line.strip()
                for line in reversed(completed.stdout.splitlines())
                if " passed" in line or " failed" in line or " error" in line
            ),
            "<no summary line>",
        )
        if completed.returncode != 0:
            print(completed.stdout[-4000:])
            raise SystemExit(f"battery failed: {summary}")
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def verdict(row):
        return (
            "ACCEPTED (both paths)"
            if row["object_path"] is None and row["seal_path"] is None
            else row["object_path"]
        )

    disagreements = [row for row in rows if row["object_path"] != row["seal_path"]]
    tally = collections.Counter(
        verdict(row) for row in rows if row not in disagreements
    )
    document = {
        "scope": (
            "diagnostic receipt of the discrimination battery; not a build, "
            "certification or release artifact"
        ),
        "release_eligible": False,
        "pytest_summary": summary,
        "comparisons": len(rows),
        "agreements": len(rows) - len(disagreements),
        "disagreements": disagreements,
        "unreachable_codes": UNREACHABLE,
        "by_verdict": dict(tally.most_common()),
        "rows": sorted(rows, key=lambda row: (row["test"], row["operand"])),
    }
    out.write_text(json.dumps(document, indent=2) + "\n")
    print(summary)
    print(
        f"comparisons={document['comparisons']} agreements={document['agreements']} "
        f"disagreements={len(disagreements)} codes={len(tally)}"
    )


if __name__ == "__main__":
    main()
