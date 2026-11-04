"""The SOC 2020 occupation constants against the engine's enum and code map."""

from policyengine_uk.variables.household.income.soc2020_major_group import (
    SOC2020_MAJOR_GROUP_CODES,
    SOC2020MajorGroup,
)

from microcosm.build.uk_runtime.occupation_constants import (
    UK_SOC2020_MAJOR_GROUPS,
    UK_SOC2020_NO_OCCUPATION,
)


def test_major_groups_are_the_engine_enum_in_code_order() -> None:
    members = list(SOC2020MajorGroup.__members__)
    assert members[0] == UK_SOC2020_NO_OCCUPATION
    assert members[1:] == list(UK_SOC2020_MAJOR_GROUPS.values())


def test_major_group_codes_are_the_engine_code_map() -> None:
    assert dict(SOC2020_MAJOR_GROUP_CODES) == {
        name: code for code, name in UK_SOC2020_MAJOR_GROUPS.items()
    }
