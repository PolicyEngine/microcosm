"""The released LCFS prepayment-meter flag is the engine's input (microcosm#1110).

policyengine-uk reads the household input ``uses_energy_prepayment_meter``
from 2.123.3 (PolicyEngine/policyengine-uk#2098), for the Energy Price
Guarantee prepayment discount, and the release exports the LCFS stage's flag.
An engine older than that ignores a dataset column it does not know; one that
reads it must read a household-level boolean, the shape the stage writes.
"""

from __future__ import annotations

from microcosm.build.uk_runtime.lcfs_consumption import (
    UK_LCFS_PREPAYMENT_METER_COLUMN,
)
from microcosm.build.uk_runtime.national_frame import (
    UK_RELEASE_EXPORT_DROPPED_COLUMNS,
)


def test_prepayment_meter_is_exported_in_the_engines_shape() -> None:
    from policyengine_uk import CountryTaxBenefitSystem

    assert (
        UK_LCFS_PREPAYMENT_METER_COLUMN
        not in (UK_RELEASE_EXPORT_DROPPED_COLUMNS["household"])
    )
    variables = CountryTaxBenefitSystem().variables
    if UK_LCFS_PREPAYMENT_METER_COLUMN in variables:
        variable = variables[UK_LCFS_PREPAYMENT_METER_COLUMN]
        assert variable.entity.key == "household"
        assert variable.value_type is bool
