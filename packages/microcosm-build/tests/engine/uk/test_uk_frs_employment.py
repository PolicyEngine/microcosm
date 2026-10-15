"""FRS EMPSTATI statuses against the engine's EmploymentStatus enum."""

from policyengine_uk.variables.household.income.employment_status import (
    EmploymentStatus,
)

from microcosm.build.uk_runtime.frs_employment import (
    CHILD_EMPLOYMENT_STATUS,
    FRS_EMPSTATI_EMPLOYMENT_STATUS,
)
from microcosm.build.uk_runtime.frs_legacy_proxies import (
    ESA_HEALTH_EMPLOYMENT_STATUSES,
)


def test_adult_codes_and_child_rows_are_a_bijection_onto_the_engine_enum() -> None:
    statuses = [*FRS_EMPSTATI_EMPLOYMENT_STATUS.values(), CHILD_EMPLOYMENT_STATUS]
    assert len(set(statuses)) == len(statuses)
    assert set(statuses) == set(EmploymentStatus.__members__)


def test_esa_health_statuses_are_engine_statuses() -> None:
    assert set(ESA_HEALTH_EMPLOYMENT_STATUSES) <= set(EmploymentStatus.__members__)
    assert "OTHER_INACTIVE" not in ESA_HEALTH_EMPLOYMENT_STATUSES
