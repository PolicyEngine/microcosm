"""The Philadelphia wage tax reaches residents built through microcosm.

Each person carries only ``employment_income_before_lsr``, the wage input
every microcosm line writes; nothing supplies the taxable wages by hand.
Households are pinned to fixture blocks in Philadelphia and its neighbour
Delaware County, then built the way the block line builds them
(``with_household_us_geography_ladder``, then ``with_us_local_tax_inputs``)
and the way an existing artifact is repaired
(``rederive_us_local_jurisdiction_flags``), and run through
``PolicyEngineUSEngine.materialize``.

The resident owes wages times the resident rate. The neighbour owes nothing,
because the nonresident (work-location) component has no source. Stripping
the derived taxable wages returns the resident's tax to zero, so the derived
input, not the residence flag alone, moves the tax.
"""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.us_runtime.geography_ladder import (
    with_household_us_geography_ladder,
)
from microcosm.build.us_runtime.local_jurisdictions import (
    rederive_us_local_jurisdiction_flags,
    with_us_local_tax_inputs,
)
from microcosm.frame import Frame
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine
from test_support.microcosm_build.us_local_jurisdictions import (
    block_ladder,
    households_on_blocks,
    us_frame,
)

PERIOD = 2026
TAXABLE_WAGES = "pa_philadelphia_wage_tax_taxable_wages"
DERIVED = ("pa_philadelphia_wage_tax_resident", TAXABLE_WAGES)
WAGES = 50_000.0
# One worker per household; the negative-wage resident owes nothing.
CASES = (
    ("philadelphia", WAGES),
    ("philadelphia", -2_000.0),
    ("delaware_county_pa", WAGES),
)
OUTPUTS = (
    "pa_philadelphia_wage_tax",
    "pa_philadelphia_wage_tax_resident",
    TAXABLE_WAGES,
)


def _frame() -> Frame:
    return us_frame(
        households_on_blocks([name for name, _ in CASES]),
        person_inputs=[
            [{"age": 40, "employment_income_before_lsr": wages}] for _, wages in CASES
        ],
    )


def _without(frame: Frame, columns: tuple[str, ...]) -> Frame:
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = tables["person"].drop(columns=list(columns))
    return Frame(tables, frame.schema, {"household": frame.weights_for("household")})


@pytest.fixture(scope="module")
def block_line() -> Frame:
    placed = with_household_us_geography_ladder(_frame(), block_ladder(), seed=0)
    built, receipt = with_us_local_tax_inputs(placed)
    assert receipt["inputs"][TAXABLE_WAGES]["residents"] == 2
    return built


@pytest.fixture(scope="module")
def engine() -> PolicyEngineUSEngine:
    return PolicyEngineUSEngine()


@pytest.fixture(scope="module")
def resident_rate() -> float:
    from policyengine_us import CountryTaxBenefitSystem

    parameters = CountryTaxBenefitSystem().parameters(PERIOD)
    return float(parameters.gov.local.pa.philadelphia.tax.income.resident_rate)


def test_block_line_resident_owes_the_resident_rate_on_derived_wages(
    engine, block_line, resident_rate
) -> None:
    result = dict(engine.materialize(block_line, OUTPUTS, PERIOD))

    assert result[TAXABLE_WAGES].tolist() == [WAGES, 0.0, 0.0]
    tax = result["pa_philadelphia_wage_tax"]
    assert resident_rate > 0
    assert tax[0] == pytest.approx(WAGES * resident_rate)
    assert tax[0] > 0
    assert tax[1] == 0
    # No source gives where the neighbour works, so no nonresident tax.
    assert tax[2] == 0


def test_the_derived_wages_not_the_flag_alone_move_the_tax(engine, block_line) -> None:
    flag_only = _without(block_line, (TAXABLE_WAGES,))
    result = dict(engine.materialize(flag_only, OUTPUTS, PERIOD))

    assert np.asarray(result["pa_philadelphia_wage_tax_resident"]).tolist() == [
        True,
        True,
        False,
    ]
    assert np.all(np.asarray(result["pa_philadelphia_wage_tax"]) == 0)


def test_rederiving_an_existing_artifact_restores_the_tax(
    engine, block_line, resident_rate
) -> None:
    repaired, receipt = rederive_us_local_jurisdiction_flags(
        _without(block_line, DERIVED)
    )
    result = dict(engine.materialize(repaired, OUTPUTS, PERIOD))

    assert receipt["carried_local_tax_inputs"] == []
    assert result["pa_philadelphia_wage_tax"][0] == pytest.approx(WAGES * resident_rate)
    assert np.asarray(result["pa_philadelphia_wage_tax"])[1:].tolist() == [0.0, 0.0]
