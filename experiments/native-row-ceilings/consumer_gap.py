"""Two ceilings the transport lane's change does not reach, stated from constants.

1. **The preparation receipt's consumer still caps it at 64 MiB.** The transport
   lane raised the producer's total to ``MAX_ROSTER_BYTES`` = 64 x 64 MiB, and
   ``survey_population_preparation`` builds the receipt as one
   ``_roster_payload`` under that total. But the bytes it hands on are checked
   again by ``graph_survey_population._checked_preparation`` against
   ``PREPARATION_MAX_BYTES``, still 64 MiB, with refusal ``PREPARATION_BYTES``.
   The transport lane's own committed ceiling receipt measured a 1/10 roster at
   109,804,304 bytes and recorded it ``accepted`` by the producer; that is 1.64x
   the consumer's cap.

2. **The ACS coverage authentication body budget is tighter than anything else
   censused**, and ``selected_body_budget.py`` measures it.

This reads constants and one committed receipt. It runs nothing, reads no
source archive, and writes only its output path. Not a build, not a
certification, not release eligible.

    python consumer_gap.py <transport-ceiling-receipt.json> <out.json>
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(path) for path in sorted((ROOT / "packages").glob("*/src"))]

from microcosm.build.us_runtime import graph_survey_population as consumer
from microcosm.build.us_runtime import survey_population_preparation as producer

for _module in (consumer, producer):
    if not pathlib.Path(_module.__file__).resolve().is_relative_to(ROOT):
        raise SystemExit(f"{_module.__name__} resolved outside {ROOT}")


def main() -> int:
    receipt = json.loads(pathlib.Path(sys.argv[1]).read_bytes())
    out = pathlib.Path(sys.argv[2])
    scales = {tuple(s["fraction"]): s for s in receipt["scales"]}
    tenth, full = scales[(1, 10)], scales[(1, 1)]
    record = {
        "scope": (
            "Constants and one committed transport-lane receipt. Runs nothing. "
            "Not a build, not a certification, not release eligible."
        ),
        "release_eligible": False,
        "producer": {
            "module": "survey_population_preparation",
            "MAX_SEGMENT_BYTES": producer.MAX_SEGMENT_BYTES,
            "MAX_ROSTER_BYTES": producer.MAX_ROSTER_BYTES,
            "receipt_is_one_joined_payload": "_roster_payload returns b''.join(segments)",
        },
        "consumer": {
            "module": "graph_survey_population",
            "PREPARATION_MAX_BYTES": consumer.PREPARATION_MAX_BYTES,
            "checked_at": (
                "_checked_preparation :305 PREPARATION_BYTES, :309 CONTEXT_BYTES; "
                "SurveyPopulationAllocationKernel.__init__ :577/:582"
            ),
            "reached_from": (
                "graph_survey_population :528, :655, :921, :1190; "
                "graph_atomic_survey_population :172; survey_age_calibration :467"
            ),
        },
        "gap": {
            "producer_total_over_consumer_cap": producer.MAX_ROSTER_BYTES
            / consumer.PREPARATION_MAX_BYTES,
            "measured_tenth_roster_bytes": tenth["roster_bytes"],
            "producer_verdict_at_tenth": tenth["roster"],
            "tenth_over_consumer_cap": tenth["roster_bytes"]
            / consumer.PREPARATION_MAX_BYTES,
            "measured_full_source_roster_bytes": full["roster_bytes"],
            "full_source_over_consumer_cap": full["roster_bytes"]
            / consumer.PREPARATION_MAX_BYTES,
            "households_the_consumer_cap_admits": int(
                consumer.PREPARATION_MAX_BYTES
                / (full["roster_bytes"] / full["households"])
            ),
        },
        "source_receipt": sys.argv[1],
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["gap"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
