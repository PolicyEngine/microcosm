"""Published allocation meanings over source-pure invented projections."""

import copy

import pytest
from test_us_current_asec_income_routing import _pure_rows, _set_amount

from microcosm.build.us_runtime import current_asec_income_routing_source as owner


@pytest.mark.parametrize(
    "earn_code,farm_code,farm_status,expected",
    [
        (0, 4, "in_printed_range", "allocation_code_meaning_unpublished"),
        (4, 4, "in_printed_range", "publisher_allocated"),
        (4, 0, "in_printed_range", "publisher_allocated"),
        (4, None, "missing", "publisher_allocated"),
        (0, 0, "in_printed_range", "published_flags_all_zero_with_unflagged_fields"),
        (0, None, "missing", "allocation_flag_not_populated"),
        (0, None, "outside_printed_range", "unresolved_allocation_provenance"),
    ],
)
def test_farm_allocation_requires_a_published_code_meaning(
    earn_code, farm_code, farm_status, expected
):
    raw, ages = _pure_rows(1, [40.0], ERN_YN=["1"], FRMOTR=["1"], FRSE_YN=["1"])
    _set_amount(raw, "FRSE_VAL", [100.0])
    raw["allocations"]["I_ERNYN"] = ([earn_code], ["in_printed_range"])
    raw["allocations"]["I_FRMYN"] = ([farm_code], [farm_status])
    before = copy.deepcopy(raw["allocations"])
    output = owner.project_income_routing(raw, ages)
    assert output.farm_allocation_origin.iloc[0] == expected
    assert output.farm_known_amount.iloc[0] == 100.0
    assert raw["allocations"] == before
    assert "meaning of its codes" in owner.AMBIGUOUS_FLAG_COVERAGE["I_FRMYN"]


@pytest.mark.parametrize(
    "name,code", [("I_RNTVAL", 4), ("I_DSTSC", 9), ("I_DSTYNCOMP", 11)]
)
def test_documented_nonzero_flags_still_establish_allocation(name, code):
    assert code in owner.ALLOCATION_ENTRIES[name][5]
    flags = {name: ([code], ["in_printed_range"])}
    assert owner._allocation_origin(flags, unflagged=False)[0] == "publisher_allocated"
