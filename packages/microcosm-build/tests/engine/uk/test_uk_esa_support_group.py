"""policyengine-uk reads the ESA group the spine stores (microcosm#1095).

The root stage writes ``esa_includes_support_component`` from the FRS ESA
group (ESAGROUP, VAR3). A stored value replaces the engine's default, which
assumes the support group for anyone with an ESA award of their own, so the
benefit cap's ESA exemption follows the survey's answer. Since
policyengine-uk 2.135.0 each scheme applies its own cap: this tenant on
income-related ESA meets Housing Benefit's, whose specified-benefit exemption
reads the stored group. Since 2.149.0 a claim for Universal Credit ends an
income-related ESA award (esa_income_report_closed, pe-uk#2563), and the
engine's default unit claims it, so the tenant, still on legacy ESA, claims
no Universal Credit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

_READ = [
    "esa_includes_support_component",
    "is_housing_benefit_benefit_cap_exempt_specified_benefit",
]


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
    benunit = pd.DataFrame({"benunit_id": [1], "would_claim_uc": [False]})
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
    exemption = "is_housing_benefit_benefit_cap_exempt_specified_benefit"
    assert bool(support[exemption][0])
    assert not bool(work_related[exemption][0])
