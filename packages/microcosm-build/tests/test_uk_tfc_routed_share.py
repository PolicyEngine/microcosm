from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.frs_person_draws import (
    UKFRSPersonDrawsStageTransform,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract

pytestmark = pytest.mark.requires_uk


def _frame():
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1],
                "person_benunit_id": [1],
                "person_household_id": [1],
                "age": [3],
                "childcare_expenses": [1_000.0],
                "tax_free_childcare_uses_qualifying_provider": [True],
                "tax_free_childcare_qualifying_child": [True],
                "is_disabled_for_benefits": [False],
                "is_blind": [False],
            }
        ),
        benunit=pd.DataFrame(
            {
                "benunit_id": [1],
                "tax_free_childcare_eligible": [True],
                "tax_free_childcare_eligible_declaration_periods": [4],
            }
        ),
        household=pd.DataFrame(
            {
                "household_id": [1],
                "region": ["LONDON"],
                "council_tax": [0.0],
                "tenure_type": ["OWNED_OUTRIGHT"],
                "rent": [0.0],
            }
        ),
        household_weights=np.array([1.0]),
        time_period="2024",
    )


def test_engine_and_manifest_define_person_input_contract() -> None:
    import policyengine_uk

    variable = policyengine_uk.CountryTaxBenefitSystem().variables[
        "tax_free_childcare_spend_routed_share"
    ]
    assert variable.entity.key == "person"
    assert variable.is_input_variable()
    assert variable.default_value == 1

    spec = load_country_spec("uk")
    stage = spec.sources.stage_map()["frs_person_draws"]
    assert "tax_free_childcare_spend_routed_share" in stage.outputs
    operation = stage.operations[-1]
    assert operation.kind == "assign_period_constant"
    assert operation.parameters == {
        "contract_key": "tax_free_childcare_spend_routed_share",
        "dtype": "float64",
        "output": "tax_free_childcare_spend_routed_share",
        "period_source": "build_year",
    }


def test_release_h5_round_trip_preserves_routed_share_and_changes_award(
    tmp_path,
) -> None:
    import policyengine_uk
    from policyengine_uk.data import UKSingleYearDataset

    contract = load_uk_take_up_contract()
    routed = UKFRSPersonDrawsStageTransform(contract=contract, stage=None)(_frame())
    routed_path = write_uk_national_frame(routed, tmp_path / "routed.h5")
    routed_simulation = policyengine_uk.Microsimulation(
        dataset=UKSingleYearDataset(file_path=routed_path)
    )
    np.testing.assert_allclose(
        routed_simulation.calculate("tax_free_childcare_spend_routed_share", 2024),
        [0.593],
    )

    default_path = write_uk_national_frame(_frame(), tmp_path / "default.h5")
    default_simulation = policyengine_uk.Microsimulation(
        dataset=UKSingleYearDataset(file_path=default_path)
    )
    routed_award = np.asarray(
        routed_simulation.calculate("tax_free_childcare", 2024), dtype=float
    )
    default_award = np.asarray(
        default_simulation.calculate("tax_free_childcare", 2024), dtype=float
    )
    assert routed_award[0] == pytest.approx(default_award[0] * 0.593)
