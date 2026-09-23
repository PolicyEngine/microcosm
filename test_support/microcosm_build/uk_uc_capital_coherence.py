# ruff: noqa: F401
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.graph import uk_spine_graph
from microcosm.build.uk_runtime.graph_kernels import UKStageKernel
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.spi_support import support_channel_column
from microcosm.build.uk_runtime.uc_capital_coherence import (
    UC_CAPITAL_REDRAW_OUTPUT,
    UC_CAPITAL_REDRAW_SALT,
    UC_CAPITAL_REDRAW_SEED,
    UKUCCapitalCoherenceStageTransform,
    _boolean_values,
    _dependent_children_band,
    _redraw_spi_reporter_capital,
    cohere_uc_capital,
)
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine
from microcosm.graph.executor import _project_context
from microcosm.graph.population import Population


def _stage():
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()["uc_capital_coherence"]


def _frame():
    rows = [
        # Base-FRS reporter donors in the 0-child, non-couple cell. Their
        # household weights are 1:9, making the target draw a weighted test.
        (1, 101, 1001, "frs", 100.0, 0, False, False, 10.0, 1.0),
        (2, 201, 2001, "frs", 200.0, 0, False, True, 10.0, 9.0),
        # The only base reporter donor in the 1-child, couple cell.
        (3, 301, 3001, "frs", 3_000.0, 1, True, False, 10.0, 4.0),
        # Base non-reporters exercise both remaining OR truth-table rows.
        (4, 401, 4001, "frs", 999_999.0, 0, False, True, 0.0, 5.0),
        (5, 501, 5001, "frs", 777_777.0, 0, False, False, 0.0, 5.0),
        # SPI post-fill reporters are redrawn; the non-reporter is preserved.
        (6, 1005, 6001, "spi", 999_999.0, 0, False, False, 10.0, 0.5),
        (7, 1006, 7001, "spi", 888_888.0, 0, False, True, 0.0, 0.5),
        (8, 1007, 8001, "spi", 999_999.0, 1, True, False, 10.0, 0.5),
    ]
    person = pd.DataFrame(
        {
            "person_id": [row[2] for row in rows],
            "person_benunit_id": [row[1] for row in rows],
            "person_household_id": [row[0] for row in rows],
            "universal_credit_reported": [row[8] for row in rows],
            "is_benunit_head": True,
            "is_parent": [row[5] > 0 for row in rows],
        }
    )
    other_members = []
    for row in rows:
        if row[6]:
            other_members.append(
                {
                    "person_id": row[2] + 1,
                    "person_benunit_id": row[1],
                    "person_household_id": row[0],
                    "universal_credit_reported": 0.0,
                    "is_benunit_head": False,
                    "is_parent": row[5] > 0,
                }
            )
        for child in range(row[5]):
            other_members.append(
                {
                    "person_id": row[2] + 10 + child,
                    "person_benunit_id": row[1],
                    "person_household_id": row[0],
                    "universal_credit_reported": 0.0,
                    "is_benunit_head": False,
                    "is_parent": False,
                }
            )
    person = pd.concat([person, pd.DataFrame(other_members)], ignore_index=True)
    benunit = pd.DataFrame(
        {
            "benunit_id": [row[1] for row in rows],
            support_channel_column("benunit"): [row[3] for row in rows],
            "frs_benunit_capital": [row[4] for row in rows],
            "dependent_children": [row[5] for row in rows],
            "is_married": [row[6] for row in rows],
            "would_claim_uc": [row[7] for row in rows],
        }
    )
    household = pd.DataFrame({"household_id": [row[0] for row in rows]})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.asarray([row[9] for row in rows]),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


__all__ = [name for name in globals() if not name.startswith("__")]
