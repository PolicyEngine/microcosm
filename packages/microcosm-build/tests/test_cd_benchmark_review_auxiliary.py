"""Auxiliary review regressions with no home in an existing CD benchmark test file.

Nothing here builds a source frame or calls a pinned adapter: the drop-count
obligation is a caller-supplied predicate, and it is checked before the ACS
housing adapter is imported, so these cases are pure and invented.
"""

from __future__ import annotations

import pytest

from microcosm.build.cd_benchmark.universe import (
    UniverseError,
    acs_housing_universe,
    vacant_drop_count,
)


@pytest.mark.parametrize("value", [None, 0, 1, 12345])
def test_a_known_or_unknown_drop_count_is_read_unchanged(value):
    assert vacant_drop_count(value) == value


@pytest.mark.parametrize("value", [-1, -1000, True, False, 1.0, "0", [], (0,)])
def test_a_drop_count_that_is_not_a_count_is_refused(value):
    """A negative count is not a known nonpositive drop; it is not a count at all."""
    with pytest.raises(UniverseError):
        vacant_drop_count(value)


def test_the_acs_entry_point_refuses_a_negative_drop_count_before_the_adapter():
    with pytest.raises(UniverseError):
        acs_housing_universe(
            object(), tenure_valid=None, vacant_household_rows_dropped=-1
        )
