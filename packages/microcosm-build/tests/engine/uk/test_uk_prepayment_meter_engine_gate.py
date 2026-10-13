"""The LCFS prepayment-meter flag waits for the engine (microcosm#1110).

policyengine-uk reads the household input ``uses_energy_prepayment_meter``
from 2.123.3 (PolicyEngine/policyengine-uk#2098), for the Energy Price
Guarantee prepayment discount. The LCFS stage draws it, and the release
boundary drops it while the locked engine does not know the variable. Once
the lock reaches a release that reads it, this test fails: export the column
by removing it from ``UK_RELEASE_EXPORT_DROPPED_COLUMNS``.
"""

from __future__ import annotations

from microcosm.build.uk_runtime.lcfs_consumption import (
    UK_LCFS_PREPAYMENT_METER_COLUMN,
)
from microcosm.build.uk_runtime.national_frame import (
    UK_RELEASE_EXPORT_DROPPED_COLUMNS,
)


def test_prepayment_meter_is_dropped_until_the_engine_reads_it() -> None:
    from policyengine_uk import CountryTaxBenefitSystem

    engine_reads_it = (
        UK_LCFS_PREPAYMENT_METER_COLUMN in CountryTaxBenefitSystem().variables
    )
    dropped = (
        UK_LCFS_PREPAYMENT_METER_COLUMN
        in (UK_RELEASE_EXPORT_DROPPED_COLUMNS["household"])
    )
    assert dropped != engine_reads_it, (
        "The locked policyengine-uk now reads uses_energy_prepayment_meter: "
        "remove it from UK_RELEASE_EXPORT_DROPPED_COLUMNS so the release "
        "exports the LCFS stage's flag (microcosm#1110)."
        if engine_reads_it
        else "The release must drop uses_energy_prepayment_meter while the "
        "locked policyengine-uk does not read it."
    )
