"""The Band D statute and keys against the installed policyengine-uk (microcosm#931).

Microcosm applies its own statute at 2024-25 Band D, so its band proportions
and single-adult discount must be the engine's, and every Great Britain code
it prices must name an authority the engine holds a Band D for. The engine
stores each proportion to nine decimals, hence the 1e-9 tolerance.
"""

from __future__ import annotations

from fractions import Fraction

import policyengine_uk
import pytest

from microcosm.build.uk_runtime.council_tax_band_d import (
    UK_COUNCIL_TAX_BAND_RATIOS,
    UK_COUNCIL_TAX_SINGLE_ADULT_DISCOUNT,
    load_uk_council_tax_band_d,
)
from microcosm.build.uk_runtime.local_authority_input import (
    local_authority_engine_key_by_code,
)

#: The engine's parameter name for each nation that levies council tax.
ENGINE_NATIONS = {"ENGLAND": "england", "WALES": "wales", "SCOTLAND": "scotland"}
#: The build period's parameters, as a 2024 simulation reads them.
INSTANT = "2024-01-01"
TOLERANCE = Fraction(1, 10**9)


@pytest.fixture(scope="module")
def council_tax():
    system = policyengine_uk.CountryTaxBenefitSystem()
    return system, system.parameters(INSTANT).gov.local_authorities.council_tax


def test_the_band_proportions_are_the_engine_s_at_2024(council_tax) -> None:
    _, parameters = council_tax
    assert set(UK_COUNCIL_TAX_BAND_RATIOS) == set(ENGINE_NATIONS)
    for nation, name in ENGINE_NATIONS.items():
        engine = getattr(parameters.band_ratios, name)
        proportions = UK_COUNCIL_TAX_BAND_RATIOS[nation]
        assert set(engine) == set(proportions), nation
        for band, proportion in proportions.items():
            assert abs(Fraction(float(engine[band])) - proportion) <= TOLERANCE, (
                nation,
                band,
            )


def test_the_single_adult_discount_is_the_engine_s_in_every_nation(
    council_tax,
) -> None:
    _, parameters = council_tax
    for name in ENGINE_NATIONS.values():
        engine = getattr(parameters.discounts.appropriate_percentage, name)
        assert Fraction(float(engine)) == UK_COUNCIL_TAX_SINGLE_ADULT_DISCOUNT, name


def test_every_band_d_code_names_an_engine_band_d_and_no_northern_ireland_one(
    council_tax,
) -> None:
    system, _ = council_tax
    # The amounts start in 2025, so the node's keys are read without an instant.
    engine_keys = set(
        system.parameters.gov.local_authorities.council_tax.band_d.amounts.children
    )
    engine_key = local_authority_engine_key_by_code()
    band_d = load_uk_council_tax_band_d().band_d

    assert len(band_d) == 350
    assert sorted(code for code in band_d if engine_key[code] not in engine_keys) == []
    northern_ireland = [code for code in engine_key if code.startswith("N")]
    assert northern_ireland
    assert [code for code in northern_ireland if engine_key[code] in engine_keys] == []
