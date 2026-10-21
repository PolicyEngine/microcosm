"""policyengine-uk reads the ESA group the spine stores (microcosm#1095).

The root stage writes ``esa_includes_support_component`` from the FRS ESA
group (ESAGROUP, VAR3). A stored value replaces the engine's default, which
assumes the support group for anyone with an ESA award of their own, so the
benefit cap's ESA exemption follows the survey's answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

_READ = ["esa_includes_support_component", "is_benefit_cap_exempt_health_disability"]


def _frame(in_support_group: bool):
    """One adult on income-related ESA, renting privately."""

    person = pd.DataFrame(
        {
            "person_id": [1],
            "person_benunit_id": [1],
            "person_household_id": [1],
            "age": [45.0],
            "esa_income_reported": [5_000.0],
            "esa_includes_support_component": [in_support_group],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [1]})
    household = pd.DataFrame(
        {
            "household_id": [1],
            "region": ["WALES"],
            "council_tax": [0.0],
            "tenure_type": ["RENT_PRIVATELY"],
            "rent": [6_000.0],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.ones(1),
        weight_kind=WeightKind.DESIGN,
        time_period="2025",
    )


def test_the_benefit_cap_exemption_follows_the_stored_esa_group() -> None:
    engine = PolicyEngineUKEngine()
    support = engine.materialize(_frame(True), _READ, 2025)
    work_related = engine.materialize(_frame(False), _READ, 2025)

    assert bool(support["esa_includes_support_component"][0])
    assert not bool(work_related["esa_includes_support_component"][0])
    assert bool(support["is_benefit_cap_exempt_health_disability"][0])
    assert not bool(work_related["is_benefit_cap_exempt_health_disability"][0])
