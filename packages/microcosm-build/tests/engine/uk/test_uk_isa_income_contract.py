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
    UK_DEFAULT_SAVINGS_INCOME_PROJECTION,
    UK_ISA_INCOME_CONTRACT,
    UK_ISA_INCOME_CONTRACT_ATTR,
    UK_LEGACY_ISA_INCOME_CONTRACT,
    UK_SAVINGS_INCOME_PROJECTION,
    UK_SAVINGS_INCOME_PROJECTION_ATTR,
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
            # The second earns little enough for Universal Credit.
            "employment_income": [30_000.0, 8_000.0],
            **_LAYOUTS[layout],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [1, 2], "would_claim_uc": [True, True]})
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
    "universal_credit",
    "household_net_income",
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

    The written file declares the producer's savings income rule, so the
    engine grows savings interest by the OBR's savings income per person
    (1.0032 to 2025 on policyengine-uk 2.174.1). Under legacy_gross it grows
    gross interest and takes off the stored ISA part unchanged: 1,000 x 1.0032
    - 300 and 300 x 0.0032 outside ISAs. The additive contract grows non-ISA
    and ISA interest each by the rule, so the ISA-only saver has no interest
    outside ISAs and both savers' ISA interest grows (microcosm#1360).
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
    assert legacy["savings_interest_income"] == pytest.approx([703.2, 0.96], abs=0.01)
    assert legacy["tax_free_savings_income"] == pytest.approx([300.0, 300.0])
    assert additive["savings_interest_income"] == pytest.approx([702.24, 0.0], abs=0.01)
    assert additive["tax_free_savings_income"] == pytest.approx(
        [300.96, 300.96], abs=0.01
    )


def test_the_engine_grows_savings_income_by_the_rule_the_file_declares(
    tmp_path,
) -> None:
    """policyengine-uk 2.165.2 grows savings income by the rule an H5 declares.

    The producer declares the OBR's savings income per person in 2022-23 to
    2025-26, the path the SPI income stage and the SPI Table 3.7 interest
    targets follow (microcosm#1418). The same file declaring nothing grows by
    GDP per head, as every file did before.
    """

    from policyengine_uk import Microsimulation
    from policyengine_uk.data.savings_income_projection import (
        DEFAULT_SAVINGS_INCOME_PROJECTION,
        OBR_SAVINGS_INCOME_2022_2025,
        SAVINGS_INCOME_PROJECTION_KEY,
        read_savings_income_projection,
    )

    assert UK_SAVINGS_INCOME_PROJECTION_ATTR == SAVINGS_INCOME_PROJECTION_KEY
    assert UK_SAVINGS_INCOME_PROJECTION == OBR_SAVINGS_INCOME_2022_2025
    assert UK_DEFAULT_SAVINGS_INCOME_PROJECTION == DEFAULT_SAVINGS_INCOME_PROJECTION
    declared = write_uk_national_frame(
        _frame("additive", "2024"), tmp_path / "declared.h5"
    )
    undeclared = write_uk_national_frame(
        _frame("additive", "2024"), tmp_path / "undeclared.h5"
    )
    with h5py.File(undeclared, mode="r+") as file:
        del file.attrs[UK_SAVINGS_INCOME_PROJECTION_ATTR]
    assert read_savings_income_projection(declared) == OBR_SAVINGS_INCOME_2022_2025
    assert read_savings_income_projection(undeclared) is None

    simulation = Microsimulation(dataset=str(declared))
    growth = simulation.tax_benefit_system.parameters.gov.economic_assumptions
    per_capita = growth.yoy_growth.obr.per_capita
    obr = float(per_capita.savings_income("2025-01-01"))
    gdp = float(per_capita.gdp("2025-01-01"))
    assert obr != pytest.approx(gdp)
    for dataset, rate in (
        (simulation, obr),
        (Microsimulation(dataset=str(undeclared)), gdp),
    ):
        interest = dataset.calculate("savings_interest_income", 2025).values
        isa = dataset.calculate("tax_free_savings_income", 2025).values
        assert interest == pytest.approx([700.0 * (1 + rate), 0.0], abs=0.01)
        assert isa == pytest.approx([300.0 * (1 + rate)] * 2, abs=0.01)
    # The stages' in-process projection (materialize with dataset_period)
    # declares the producer's rule too, as the released file does.
    projected = PolicyEngineUKEngine().materialize(
        _frame("additive", "2024"),
        ["savings_interest_income"],
        2025,
        dataset_period=2024,
    )
    assert np.asarray(projected["savings_interest_income"]) == pytest.approx(
        [700.0 * (1 + obr), 0.0], abs=0.01
    )
