"""Measure the origin-budget payload's per-group byte cost, using the module's own encoder.

``survey_origin_budget._budget_payload`` streams one origin record per allocation
group into a single bytearray guarded by ``MAX_PAYLOAD_BYTES``. The record is a
fixed-shape JSON object whose only variable-width field is the raw native key, so
its encoded size is measurable from one faithfully constructed record rather than
estimated. This builds that record through the module's own ``_reference`` and
``_json`` and reports the household count at which the 64 MiB cap is met.

The per-group cost also includes the two ``household_ids`` and two
``group_indices`` entries the header carries for each group's two clone roles.

No graph runs, no source is read, nothing outside the output path is written.
Not a build, not a certification, not release eligible.

    python origin_budget_size.py <out.json>
"""

from __future__ import annotations

import json
import pathlib
import sys
from fractions import Fraction

sys.path[:0] = [
    str(path)
    for path in sorted(
        (pathlib.Path(__file__).resolve().parents[2] / "packages").glob("*/src")
    )
]

from microcosm.build.us_runtime import survey_origin_budget as owner

FULL_SOURCE_HOUSEHOLDS = 1_587_376  # measured; see roster-census.json


def _record(native_id: str, household_id: int, clone_ids: tuple[int, int]) -> dict:
    design = Fraction(137, 1)  # a representative ACS WGTP
    probability = Fraction(1584, 1587376)
    share = Fraction(1, 2)
    actual = float(design * share / probability)
    reference = owner._reference(design, probability, share, actual)
    return {
        "source": "acs",
        "source_year": 2024,
        "survey_year": 2024,
        "raw_native_id": native_id,
        "selected_receiving_household_id": household_id,
        "combined_household_id": household_id,
        "statistical_unit": "occupied_housing_unit",
        "original_design_float64_bytes": "0" * 16,
        **reference,
        "members": [[clone_ids[0], 0], [clone_ids[1], 1]],
        "incoming_clone_float64_bytes": ["0" * 16, "0" * 16],
    }


def main() -> int:
    out = pathlib.Path(sys.argv[1])
    rows = []
    # Measure at household-id magnitudes a full-source build actually reaches, so
    # the integer widths in the encoded record are the real ones.
    for magnitude in (1_000, 1_000_000, FULL_SOURCE_HOUSEHOLDS, 3_174_752):
        record = _record(
            f"2024HU{magnitude % 10_000_000:07d}",
            magnitude,
            (magnitude, magnitude * 2),
        )
        encoded = len(owner._json(record))
        # ",": one separator per record after the first.
        # header ids/groups: two household ids and two group indices per group.
        ids = len(str(magnitude * 2)) * 2 + 2
        groups = len(str(magnitude)) * 2 + 2
        per_group = encoded + 1 + ids + groups
        rows.append(
            {
                "household_id_magnitude": magnitude,
                "record_bytes": encoded,
                "header_ids_and_groups_bytes": ids + groups,
                "per_group_bytes": per_group,
                "households_at_64MiB": owner.MAX_PAYLOAD_BYTES // per_group,
                "full_source_payload_bytes": per_group * FULL_SOURCE_HOUSEHOLDS,
            }
        )
    worst = max(rows, key=lambda r: r["per_group_bytes"])
    record = {
        "scope": (
            "Encoder-measured size law for the origin-budget payload. Builds one "
            "faithful record through the module's own _reference and _json. Not a "
            "build, not a certification, not release eligible."
        ),
        "release_eligible": False,
        "max_payload_bytes": owner.MAX_PAYLOAD_BYTES,
        "max_groups": owner.MAX_GROUPS,
        "full_source_households": FULL_SOURCE_HOUSEHOLDS,
        "measurements": rows,
        "binding": {
            "per_group_bytes_at_full_source_ids": worst["per_group_bytes"],
            "households_the_64MiB_cap_admits": worst["households_at_64MiB"],
            "fraction_of_source_that_admits": worst["households_at_64MiB"]
            / FULL_SOURCE_HOUSEHOLDS,
            "full_source_payload_bytes": worst["full_source_payload_bytes"],
            "full_source_payload_over_cap": worst["full_source_payload_bytes"]
            / owner.MAX_PAYLOAD_BYTES,
        },
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["measurements"], indent=1))
    print(json.dumps(record["binding"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
