"""The frame's ISA income contract reaches policyengine-uk (microcosm#1360).

The producer stores non-ISA interest in ``savings_interest_income`` and ISA
interest in ``individual_savings_account_interest_income`` and declares the
``additive`` contract; ``tax_free_savings_income`` is the engine's sum of the
ISA parts and is never stored. A legacy gross file (the enhanced FRS, Microcosm
before #1360) stores gross interest with the ISA part in
``tax_free_savings_income``; it keeps that reading through every load, and the
build refuses it as an input. Both layouts give the same base-year cash and tax.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    require_uk_producer_isa_income_contract,
    uk_isa_income_contract,
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.frame import WeightKind
from microcosm.frame.adapters.policyengine_uk import (
    UK_ISA_INCOME_CONTRACT,
    UK_ISA_INCOME_CONTRACT_ATTR,
    UK_LEGACY_ISA_INCOME_CONTRACT,
    PolicyEngineUKEngine,
    uk_isa_income_contract_from,
)

#: One saver with 1,000 of interest, 300 of it in an ISA, and one whose only
#: interest is 300 in an ISA, in each layout.
_LAYOUTS = {
    "additive": {
        "savings_interest_income": [700.0, 0.0],
        "individual_savings_account_interest_income": [300.0, 300.0],
    },
    "legacy_gross": {
        "savings_interest_income": [1_000.0, 300.0],
        "tax_free_savings_income": [300.0, 300.0],
    },
}
_CONTRACTS = {
    "additive": UK_ISA_INCOME_CONTRACT,
    "legacy_gross": UK_LEGACY_ISA_INCOME_CONTRACT,
}


def _frame(layout: str = "additive", time_period: str = "2025"):
    person = pd.DataFrame(
        {
            "person_id": [11, 21],
            "person_benunit_id": [1, 2],
            "person_household_id": [1, 2],
            "age": [40.0, 40.0],
            "employment_income": [30_000.0, 60_000.0],
            **_LAYOUTS[layout],
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
        isa_income_contract=_CONTRACTS[layout],
    )


_BASE_YEAR_MEASURES = (
    "savings_interest_income",
    "tax_free_savings_income",
    "capital_income",
    "taxable_savings_interest_income",
    "income_tax",
)


def test_both_layouts_give_the_same_base_year_cash_and_tax() -> None:
    engine = PolicyEngineUKEngine()
    additive = engine.materialize(_frame("additive"), list(_BASE_YEAR_MEASURES), 2025)
    legacy = engine.materialize(_frame("legacy_gross"), list(_BASE_YEAR_MEASURES), 2025)

    assert np.asarray(additive["savings_interest_income"]) == pytest.approx(
        [700.0, 0.0]
    )
    assert np.asarray(additive["tax_free_savings_income"]) == pytest.approx(
        [300.0, 300.0]
    )
    for measure in _BASE_YEAR_MEASURES:
        assert np.asarray(additive[measure]) == pytest.approx(
            np.asarray(legacy[measure]), abs=1e-6
        ), measure


def test_a_written_frame_declares_its_contract_and_loads_with_it(tmp_path) -> None:
    from policyengine_uk import Microsimulation

    path = write_uk_national_frame(_frame(), tmp_path / "frame.h5")
    with h5py.File(path, mode="r") as file:
        declared = json.loads(file.attrs[UK_ISA_INCOME_CONTRACT_ATTR])
    assert declared == dict(UK_ISA_INCOME_CONTRACT)
    loaded, _ = load_uk_national_frame(path)
    assert uk_isa_income_contract(loaded) == dict(UK_ISA_INCOME_CONTRACT)
    require_uk_producer_isa_income_contract(loaded, label="frame.h5")

    simulation = Microsimulation(dataset=str(path))
    interest = simulation.calculate("savings_interest_income", 2025).values
    assert interest == pytest.approx([700.0, 0.0])


def test_a_legacy_file_keeps_its_reading_and_is_refused_as_a_build_input(
    tmp_path,
) -> None:
    path = write_uk_national_frame(_frame("legacy_gross"), tmp_path / "legacy.h5")
    loaded, _ = load_uk_national_frame(path)
    assert uk_isa_income_contract(loaded) == dict(UK_LEGACY_ISA_INCOME_CONTRACT)
    values = PolicyEngineUKEngine().materialize(
        loaded, ["savings_interest_income"], 2025
    )
    assert np.asarray(values["savings_interest_income"]) == pytest.approx([700.0, 0.0])
    with pytest.raises(ValueError, match="rebuild it from source"):
        require_uk_producer_isa_income_contract(loaded, label="legacy.h5")

    # A file declaring nothing reads as legacy gross, as policyengine-uk reads it.
    with h5py.File(path, mode="r+") as file:
        del file.attrs[UK_ISA_INCOME_CONTRACT_ATTR]
    unmarked, _ = load_uk_national_frame(path)
    assert uk_isa_income_contract(unmarked) == dict(UK_LEGACY_ISA_INCOME_CONTRACT)


def test_a_malformed_contract_fails_closed(tmp_path) -> None:
    for declared in (
        {"version": 2, "representation": "additive", "projection": "additive"},
        {"version": 1, "representation": "legacy_gross", "projection": "additive"},
        {"version": True, "representation": "additive", "projection": "additive"},
        "not json",
    ):
        with pytest.raises(ValueError, match="ISA income contract"):
            uk_isa_income_contract_from(declared)
    path = write_uk_national_frame(_frame(), tmp_path / "frame.h5")
    with h5py.File(path, mode="r+") as file:
        file.attrs[UK_ISA_INCOME_CONTRACT_ATTR] = json.dumps({"version": 1})
    with pytest.raises(ValueError, match="ISA income contract"):
        load_uk_national_frame(path)


def test_a_2024_frame_read_at_2025_projects_each_component(tmp_path) -> None:
    """The build's frame is FRS 2024-25, read at 2025.

    Under legacy_gross the engine grows gross interest (by per-capita GDP,
    1.0462 to 2025 on policyengine-uk 2.164.2) and takes off the stored ISA part
    unchanged: 1,000 x 1.0462 - 300 and 300 x 0.0462 outside ISAs. The additive
    contract grows non-ISA and ISA interest each by its own index, so the
    ISA-only saver has no interest outside ISAs and both savers' ISA interest
    grows (microcosm#1360).
    """

    from policyengine_uk import Microsimulation

    readings = {}
    for layout in ("legacy_gross", "additive"):
        path = write_uk_national_frame(
            _frame(layout, "2024"), tmp_path / f"{layout}.h5"
        )
        simulation = Microsimulation(dataset=str(path))
        readings[layout] = {
            name: simulation.calculate(name, 2025).values
            for name in ("savings_interest_income", "tax_free_savings_income")
        }
    legacy, additive = readings["legacy_gross"], readings["additive"]
    assert legacy["savings_interest_income"] == pytest.approx([746.2, 13.86], abs=0.01)
    assert legacy["tax_free_savings_income"] == pytest.approx([300.0, 300.0])
    assert additive["savings_interest_income"] == pytest.approx([732.34, 0.0], abs=0.01)
    assert additive["tax_free_savings_income"] == pytest.approx(
        [313.86, 313.86], abs=0.01
    )
