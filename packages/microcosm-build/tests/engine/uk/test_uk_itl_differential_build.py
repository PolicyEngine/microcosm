"""A calibrated national file against HMRC's Income Tax liabilities projections.

Needs a built file: set ``MICROCOSM_UK_ITL_DIFFERENTIAL_H5`` to a national
H5 (and ``MICROCOSM_UK_ITL_DIFFERENTIAL_ROWS`` to a vendored-format file
that replaces the vendored Table 2.5 rows, to add Tables 2.1 and 2.2 before
they are vendored). Every cell must sit within the
tolerance ``uk_runtime.itl_differential`` fixes; the failure lists the misses
by table, measure and band or area.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from microcosm.build.uk_runtime.hmrc_uprating import UK_HMRC_TAXPAYER_COUNTS_RESOURCE
from microcosm.build.uk_runtime.itl_differential import (
    differential,
    expected_cells,
    measure_resolver,
    resolved_people,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

_H5 = os.environ.get("MICROCOSM_UK_ITL_DIFFERENTIAL_H5")
_ROWS = os.environ.get("MICROCOSM_UK_ITL_DIFFERENTIAL_ROWS")


@pytest.mark.slow
@pytest.mark.skipif(not _H5, reason="set MICROCOSM_UK_ITL_DIFFERENTIAL_H5")
def test_the_national_file_meets_hmrc_projections(tmp_path) -> None:
    if _ROWS:
        rows = json.loads(Path(_ROWS).read_text(encoding="utf-8"))["rows"]
    else:
        rows = vendored_rows(UK_HMRC_TAXPAYER_COUNTS_RESOURCE)
    report = differential(
        expected_cells(rows, year=2025),
        resolved_people(measure_resolver(Path(_H5), year=2025, scratch=tmp_path)),
    )
    assert not report["misses"], "\n".join(report["misses"])
