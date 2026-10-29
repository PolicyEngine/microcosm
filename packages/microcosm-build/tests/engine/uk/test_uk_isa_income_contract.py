"""The frame's savings interest reaches policyengine-uk as interest outside ISAs.

The frame stores ``savings_interest_income`` gross, ISA interest included,
with the ISA part in ``tax_free_savings_income``. From policyengine-uk 2.162.2
the engine's ``savings_interest_income`` is the interest outside tax-free
accounts: the build declares the ``legacy_gross`` ISA income contract, so the
engine takes the ISA part off when it loads the frame, whatever its default
for a dataset that declares nothing (microcosm#1095).
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import (
    UK_ISA_INCOME_CONTRACT,
    UK_ISA_INCOME_CONTRACT_ATTR,
    PolicyEngineUKEngine,
)


def _frame(time_period: str = "2025"):
    # One saver with 1,000 of gross interest, 300 of it in an ISA, and one
    # whose only interest is 300 in an ISA.
    person = pd.DataFrame(
        {
            "person_id": [11, 21],
            "person_benunit_id": [1, 2],
            "person_household_id": [1, 2],
            "age": [40.0, 40.0],
            "savings_interest_income": [1_000.0, 300.0],
            "tax_free_savings_income": [300.0, 300.0],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [1, 2], "would_claim_uc": [False, False]})
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "region": ["WALES", "WALES"],
            "council_tax": [0.0, 0.0],
            "tenure_type": ["OWNED_OUTRIGHT", "OWNED_OUTRIGHT"],
            "rent": [0.0, 0.0],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.ones(2),
        weight_kind=WeightKind.DESIGN,
        time_period=time_period,
    )


def test_the_adapter_dataset_reads_interest_outside_isas_at_the_frames_year() -> None:
    values = PolicyEngineUKEngine().materialize(
        _frame(), ["savings_interest_income", "tax_free_savings_income"], 2025
    )

    assert np.asarray(values["savings_interest_income"]) == pytest.approx([700.0, 0.0])
    assert np.asarray(values["tax_free_savings_income"]) == pytest.approx(
        [300.0, 300.0]
    )


def test_a_written_frame_declares_the_contract_and_reads_the_same(tmp_path) -> None:
    from policyengine_uk import Microsimulation

    path = write_uk_national_frame(_frame(), tmp_path / "frame.h5")
    with h5py.File(path, mode="r") as file:
        declared = json.loads(file.attrs[UK_ISA_INCOME_CONTRACT_ATTR])
    assert declared == dict(UK_ISA_INCOME_CONTRACT)

    simulation = Microsimulation(dataset=str(path))
    interest = simulation.calculate("savings_interest_income", 2025).values
    assert interest == pytest.approx([700.0, 0.0])


def test_a_2024_frame_read_at_2025_grows_gross_interest_less_a_flat_isa_part(
    tmp_path,
) -> None:
    """The build's frame is FRS 2024-25, read at 2025. Under legacy_gross the
    engine grows the gross interest (by per-capita GDP, 1.0474 to 2025) and takes
    off the stored ISA part unchanged, so the saver reads 1,000 x 1.0474 - 300 and
    the ISA-only saver keeps 300 x 0.0474 as interest outside ISAs. The count
    rows exclude that saver, as they read the frame; the amount rows do not.
    Moving the frame to the additive contract (microcosm#1095) changes both
    values (Vahid's review of #1400, round 3)."""

    from policyengine_uk import Microsimulation

    path = write_uk_national_frame(_frame("2024"), tmp_path / "frame.h5")
    simulation = Microsimulation(dataset=str(path))
    interest = simulation.calculate("savings_interest_income", 2025).values
    assert interest == pytest.approx([747.4, 14.22], abs=0.01)
