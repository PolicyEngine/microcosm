"""Measure the ACS coverage authentication body budget, the tightest ceiling censused.

``acs_person_coverage_authentication`` charges every *selected* row
``6 * len(raw) + 1024`` bytes against ``MAX_BODY_BYTES`` before the low-level
reader allocates its DataFrame (:378-382, refusal ``SELECTED_BODY_BUDGET``).
The only variable is ``len(raw)``, the record's own encoded bytes, so the
admitted row count is measurable from the real archive rather than guessed.

Reads a bounded prefix of the pilot's captured public ACS PUMS archive and
reports the record lengths and the household count the 64 MiB budget admits.
Nothing outside the output path is written. Not a build, not a certification,
not release eligible.

    python selected_body_budget.py <snapshot-dir> <out.json>
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(path) for path in sorted((ROOT / "packages").glob("*/src"))]

from microcosm.build.us_runtime import acs_person_coverage_authentication as auth

if not pathlib.Path(auth.__file__).resolve().is_relative_to(ROOT):
    raise SystemExit(f"acs_person_coverage_authentication resolved outside {ROOT}")

# Measured full-source ACS counts; see roster-census.json.
ACS_PERSONS = 3_422_888
ACS_HOUSEHOLDS = 1_531_614
SAMPLE_RECORDS = 200_000


def _sample(archive: pathlib.Path, prefix: str) -> list[int]:
    lengths: list[int] = []
    with zipfile.ZipFile(archive) as zf:
        for name in sorted(n for n in zf.namelist() if prefix in n.lower()):
            with zf.open(name) as member:
                member.readline()  # header
                for raw in member:
                    lengths.append(len(raw))
                    if len(lengths) >= SAMPLE_RECORDS:
                        return lengths
    return lengths


def main() -> int:
    snapshot = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    roles = {
        "person": (snapshot / "csv_pus.zip", "psam_pus", ACS_PERSONS),
        "household": (snapshot / "csv_hus.zip", "psam_hus", ACS_HOUSEHOLDS),
    }
    measured = {}
    for role, (archive, prefix, full_source) in roles.items():
        lengths = _sample(archive, prefix)
        if not lengths:
            raise SystemExit(f"no records sampled for {role} from {archive}")
        mean = statistics.fmean(lengths)
        charge_mean = 6 * mean + 1024
        charge_max = 6 * max(lengths) + 1024
        measured[role] = {
            "archive": str(archive),
            "records_sampled": len(lengths),
            "record_bytes_mean": round(mean, 2),
            "record_bytes_min": min(lengths),
            "record_bytes_max": max(lengths),
            "charge_per_selected_row_mean": round(charge_mean, 2),
            "rows_the_64MiB_budget_admits_mean": int(
                auth.MAX_BODY_BYTES // charge_mean
            ),
            "rows_the_64MiB_budget_admits_worst": int(
                auth.MAX_BODY_BYTES // charge_max
            ),
            "full_source_rows": full_source,
            "fraction_of_source_admitted": (auth.MAX_BODY_BYTES // charge_mean)
            / full_source,
            "full_source_budget_bytes": int(charge_mean * full_source),
            "full_source_over_cap": charge_mean * full_source / auth.MAX_BODY_BYTES,
        }
    record = {
        "scope": (
            "Measurement of acs_person_coverage_authentication's SELECTED_BODY_BUDGET "
            "over a bounded prefix of the pilot's captured public ACS PUMS archive. "
            "Not a build, not a certification, not release eligible."
        ),
        "release_eligible": False,
        "max_body_bytes": auth.MAX_BODY_BYTES,
        "charge_expression": "6 * len(raw) + 1024 per selected row",
        "refusal": "SELECTED_BODY_BUDGET (ACSCoverageAuthenticationError, a ValueError)",
        "roles": measured,
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(measured, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
