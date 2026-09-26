"""Measure why the roles row bound cannot usefully be raised on its own.

``current_survey_household_roles._projection_bytes`` is
``table.reset_index().to_json(orient="table", index=False).encode()`` -- the
whole per-person table materialised as one JSON string -- and the artifact that
carries it is bounded by ``graph_current_survey_household_roles.MAX_ARTIFACT_BYTES``
(64 MiB). So the module's effective ceiling is that byte cap, not its
``MAX_PERSONS`` row bound, and this measures the gap through the module's own
encoder.

The table is invented but faithfully shaped: the real declared columns, the real
dtypes, and role/universe tokens of the lengths the module emits. The marginal
cost per row is taken as a difference between two row counts so the fixed
``orient="table"`` schema preamble cancels.

Nothing outside the output path is written. Not a build, not a certification,
not release eligible.

    python roles_projection_size.py <out.json>
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(path) for path in sorted((ROOT / "packages").glob("*/src"))]

from microcosm.build.us_runtime import current_survey_household_roles as roles
from microcosm.build.us_runtime import graph_current_survey_household_roles as graph

for _module in (roles, graph):
    if not pathlib.Path(_module.__file__).resolve().is_relative_to(ROOT):
        raise SystemExit(f"{_module.__name__} resolved outside {ROOT}")

STACKED_PERSONS = 3_565_013  # measured; see roster-census.json


def _table(rows: int) -> pd.DataFrame:
    pattern = {
        roles.CANONICAL_COLUMN: [True, False, True],
        roles.SURVEY_COLUMN: ["acs", "asec", "acs"],
        roles.NATIVE_ID_COLUMN: [1, 2, 3],
        roles.CODE_COLUMN: [20, 25, 37],
        roles.CODE_KNOWN_COLUMN: [True, True, False],
        roles.ROLE_STATE_COLUMN: ["observed_reference_person"] * 3,
        roles.UNIVERSE_COLUMN: ["housing_unit"] * 3,
    }
    frame = pd.concat([pd.DataFrame(pattern)] * ((rows + 2) // 3)).head(rows)
    frame.index = pd.Index(range(rows), name="person_id")
    return frame


def main() -> int:
    out = pathlib.Path(sys.argv[1])
    small, large = 3, 1_203
    bytes_small = len(roles._projection_bytes(_table(small)))
    bytes_large = len(roles._projection_bytes(_table(large)))
    per_row = (bytes_large - bytes_small) / (large - small)
    admitted = int(graph.MAX_ARTIFACT_BYTES // per_row)
    record = {
        "scope": (
            "Encoder-measured size law for the current-survey household-roles "
            "projection, on a faithfully shaped invented table. Not a build, not a "
            "certification, not release eligible."
        ),
        "release_eligible": False,
        "encoder": 'table.reset_index().to_json(orient="table", index=False).encode()',
        "max_artifact_bytes": graph.MAX_ARTIFACT_BYTES,
        "max_persons_row_bound": roles.MAX_PERSONS,
        "projection_bytes": {str(small): bytes_small, str(large): bytes_large},
        "marginal_bytes_per_person_row": round(per_row, 2),
        "persons_the_byte_cap_admits": admitted,
        "persons_the_row_bound_admits": roles.MAX_PERSONS,
        "byte_cap_is_tighter_by": round(roles.MAX_PERSONS / admitted, 2),
        "full_source_stacked_persons": STACKED_PERSONS,
        "fraction_of_source_the_byte_cap_admits": admitted / STACKED_PERSONS,
        "fraction_of_source_the_row_bound_admits": roles.MAX_PERSONS / STACKED_PERSONS,
        "conclusion": (
            "The byte cap refuses first and by an order of magnitude, so raising "
            "MAX_PERSONS alone would move nothing. This bound takes the "
            "segmented-transport argument, not the row-ceiling rule."
        ),
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
