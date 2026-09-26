"""Derive the full-source row counts every ceiling in this lane is measured against.

Reads the recovered 1/1000 pilot preparation artifact and reports two things:

1. Its **catalogues**, which are counts of the whole upstream ACS and ASEC files
   and therefore do not scale with the selection fraction. These give the
   full-source selected counts exactly, not by extrapolation: a full-source
   build selects every selectable household in both catalogues.
2. Its **origins** rosters at 1/1000, which give the per-household ratios and
   let the catalogue figures be cross-checked against an observed sample.

The reconciliation that makes (1) authoritative is asserted here rather than
asserted in prose: the ACS catalogue's occupied, institutional-GQ and
noninstitutional-GQ households plus the ASEC catalogue's households equal
``selection.supplied_households`` exactly, so "the whole catalogue" and "what a
full-source selection supplies" are the same set.

No graph runs; nothing outside the output path is written. Not a build, not a
certification, not release eligible.

    python roster_census.py <preparation.json> <out.json>
"""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import sys


def main() -> int:
    artifact = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    raw = artifact.read_bytes()
    document = json.loads(raw)
    selection = document["selection"]
    origins = document["origins"]
    acs = document["catalogues"]["acs"]["counts"]
    asec = document["catalogues"]["asec"]["counts"]

    acs_selectable = (
        acs["occupied_hu"] + acs["institutional_gq"] + acs["noninstitutional_gq"]
    )
    supplied = selection["supplied_households"]
    reconciles = acs_selectable + asec["households"] == supplied
    if not reconciles:
        raise SystemExit(
            "catalogue households do not reconcile to supplied_households; "
            "the full-source counts below would be extrapolations, not measurements"
        )

    columns = origins["persons"]["columns"]
    source_at = columns.index("source")
    sample_persons = collections.Counter(
        row[source_at] for row in origins["persons"]["rows"]
    )
    sample_households = collections.Counter(
        row["source"] for row in origins["households"]
    )
    selected = len(selection["selected"])

    record = {
        "artifact": str(artifact),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": (
            "Census input for the row-count ceiling lane. Reads one recovered "
            "development artifact and writes one JSON summary. Not a build, not "
            "a certification, not release eligible."
        ),
        "release_eligible": False,
        "catalogue_reconciles_to_supplied_households": reconciles,
        "catalogues": {"acs": acs, "asec": asec},
        "full_source": {
            "derivation": (
                "the catalogues themselves; a full-source selection supplies every "
                "selectable household in both, which the reconciliation above proves"
            ),
            "acs_households": acs_selectable,
            "acs_persons": acs["people"],
            "asec_households": asec["households"],
            "asec_persons": asec["persons"],
            "stacked_households": supplied,
            "stacked_persons": acs["people"] + asec["persons"],
            "combined_clone_households": supplied * 2,
            "combined_clone_persons": (acs["people"] + asec["persons"]) * 2,
        },
        "sample_at_one_thousandth": {
            "fraction": selection["fraction"],
            "selected_households": selected,
            "origin_household_rows": len(origins["households"]),
            "origin_person_rows": len(origins["persons"]["rows"]),
            "households_by_source": dict(sample_households),
            "persons_by_source": dict(sample_persons),
            "origin_entities": {k: len(v) for k, v in origins["entities"].items()},
            "entities_per_selected_household": {
                k: len(v) / selected for k, v in origins["entities"].items()
            },
        },
        "one_tenth": {
            "derivation": "the full-source counts above, times 1/10",
            "stacked_households": supplied // 10,
            "acs_households": acs_selectable // 10,
            "acs_persons": acs["people"] // 10,
            "asec_persons": asec["persons"] // 10,
            "stacked_persons": (acs["people"] + asec["persons"]) // 10,
            "combined_clone_persons": (acs["people"] + asec["persons"]) * 2 // 10,
        },
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["full_source"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
