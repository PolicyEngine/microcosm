"""The SOC 2020 occupation constants and the views derived from them agree."""

from __future__ import annotations

import pytest

from microcosm.build.uk_runtime import frs_employment
from microcosm.build.uk_runtime.occupation_constants import (
    UK_FRS_SOC2020_MAJOR_GROUP,
    UK_SOC2020_MAJOR_GROUPS,
    UK_SOC2020_NO_OCCUPATION,
)


def test_major_groups_are_codes_one_to_nine_in_order() -> None:
    assert list(UK_SOC2020_MAJOR_GROUPS) == list(range(1, 10))
    names = list(UK_SOC2020_MAJOR_GROUPS.values())
    assert len(set(names)) == len(names) == 9
    assert UK_SOC2020_NO_OCCUPATION == "NONE"
    assert UK_SOC2020_NO_OCCUPATION not in names


def test_the_frs_view_is_derived_from_the_major_groups() -> None:
    assert dict(UK_FRS_SOC2020_MAJOR_GROUP) == {
        0: UK_SOC2020_NO_OCCUPATION,
        **{code * 1000: name for code, name in UK_SOC2020_MAJOR_GROUPS.items()},
    }


def test_the_employment_stage_reads_the_shared_frs_view() -> None:
    assert frs_employment.UK_FRS_SOC2020_MAJOR_GROUP is UK_FRS_SOC2020_MAJOR_GROUP
    assert "soc2020_major_group" in frs_employment.FRS_EMPLOYMENT_OUTPUT_COLUMNS


@pytest.mark.parametrize(
    "mapping", [UK_SOC2020_MAJOR_GROUPS, UK_FRS_SOC2020_MAJOR_GROUP]
)
def test_the_code_tables_are_immutable(mapping) -> None:
    with pytest.raises(TypeError):
        mapping[10] = "OTHER"  # type: ignore[index]


def test_every_adult_employment_status_has_an_occupation_status_group() -> None:
    from microcosm.build.uk_runtime.occupation_constants import (
        UK_OCCUPATION_STATUS_GROUP,
    )

    adult_statuses = set(frs_employment.FRS_EMPSTATI_EMPLOYMENT_STATUS.values())
    assert set(UK_OCCUPATION_STATUS_GROUP) == adult_statuses
    assert frs_employment.CHILD_EMPLOYMENT_STATUS not in UK_OCCUPATION_STATUS_GROUP
    assert set(UK_OCCUPATION_STATUS_GROUP.values()) == {
        "EMPLOYEE",
        "SELF_EMPLOYED",
        "RETIRED",
        "OTHER",
    }


def test_sic_sections_are_ordered_and_disjoint() -> None:
    from microcosm.build.uk_runtime.occupation_constants import (
        UK_SIC2007_SECTION_DIVISIONS,
        uk_sic2007_sections,
    )

    ranges = list(UK_SIC2007_SECTION_DIVISIONS.items())
    assert [section for section, _ in ranges] == list("ABCDEFGHIJKLMNOPQRSTU")
    for (_, (_, last)), (_, (first, _)) in zip(ranges, ranges[1:], strict=False):
        assert last < first
    assert list(uk_sic2007_sections([0, 1, 4, 47, 85, 99])) == [
        "",
        "A",
        "",
        "G",
        "P",
        "U",
    ]


def test_each_sub_major_group_nests_in_its_codes_major_group() -> None:
    from microcosm.build.uk_runtime.occupation_constants import (
        UK_SOC2020_MAJOR_GROUP_OF_SUB_MAJOR_GROUP,
        UK_SOC2020_SUB_MAJOR_GROUPS,
    )

    codes = list(UK_SOC2020_SUB_MAJOR_GROUPS)
    assert codes == sorted(codes) and len(codes) == 26
    assert {code // 10 for code in codes} == set(UK_SOC2020_MAJOR_GROUPS)
    names = list(UK_SOC2020_SUB_MAJOR_GROUPS.values())
    assert len(set(names)) == len(names)
    assert UK_SOC2020_NO_OCCUPATION not in names
    assert dict(UK_SOC2020_MAJOR_GROUP_OF_SUB_MAJOR_GROUP) == {
        UK_SOC2020_NO_OCCUPATION: UK_SOC2020_NO_OCCUPATION,
        **{
            name: UK_SOC2020_MAJOR_GROUPS[code // 10]
            for code, name in UK_SOC2020_SUB_MAJOR_GROUPS.items()
        },
    }
